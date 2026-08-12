"""The asymmetric filter chain.

Every raw technical signal is run past these rules in order. Each returns a
``FilterOutcome`` carrying a multiplier (applied to confidence), a stop
multiplier, and possibly a hard block.

The chain is deliberately **asymmetric**: filters can veto or shrink a trade
freely, but can only boost it modestly (capped around 1.25x). Macro conditions
are far better at telling you when *not* to trade than at telling you when to.

Order matters — cheap hard blocks (event blackout, liquidity halt) run before
expensive scoring rules, and the engine stops evaluating once a signal is
blocked.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from mfie.config import Params, get_params
from mfie.core.types import (
    Direction,
    EconomicEvent,
    FilterAction,
    FilterOutcome,
    Instrument,
    LiquidityRegime,
    MacroContext,
    MacroRegime,
    Quote,
    RawSignal,
)
from mfie.core.universe import is_high_beta
from mfie.core.utils import clamp
from mfie.econ.behavioral import compute_behavioral
from mfie.econ.microstructure import MicrostructureView
from mfie.econ.rates import RateView
from mfie.econ.risk import RiskBudget
from mfie.econ.surprise import esi_differential, esi_multiplier
from mfie.econ.tokenomics import TokenomicsView, tokenomics_multiplier
from mfie.econ.valuation import PPPView, pair_valuation_bias, valuation_multiplier
from mfie.regime.detector import RegimeView

MAX_BOOST = 1.25


@dataclass
class FilterContext:
    """Per-instrument state assembled by the engine before filtering."""

    instrument: Instrument
    macro: MacroContext
    params: Params = field(default_factory=get_params)
    quote: Quote | None = None
    atr: float = 0.0
    regime: RegimeView | None = None
    rate_view: RateView | None = None
    ppp_base: PPPView | None = None
    ppp_quote: PPPView | None = None
    tokenomics: TokenomicsView | None = None
    microstructure: MicrostructureView | None = None
    risk_budget: RiskBudget | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class Filter(ABC):
    name: str = "filter"
    level: int = 1
    applies_to: tuple[str, ...] = ("crypto", "forex")

    def applicable(self, signal: RawSignal, ctx: FilterContext) -> bool:
        return signal.instrument.asset_class.value in self.applies_to

    @abstractmethod
    def evaluate(self, signal: RawSignal, ctx: FilterContext) -> FilterOutcome:
        ...

    def __call__(self, signal: RawSignal, ctx: FilterContext) -> FilterOutcome:
        if not self.applicable(signal, ctx):
            return FilterOutcome(self.name, FilterAction.PASS, reason="not applicable")
        return self.evaluate(signal, ctx)


def _outcome(name: str, multiplier: float, reason: str, detail: dict[str, Any] | None = None,
             stop_multiplier: float = 1.0, block: bool = False) -> FilterOutcome:
    """Build an outcome, classifying the action from the multiplier."""
    if block:
        action = FilterAction.BLOCK
    elif multiplier > 1.001:
        action = FilterAction.BOOST
    elif multiplier < 0.999:
        action = FilterAction.PENALIZE
    else:
        action = FilterAction.PASS
    return FilterOutcome(
        name=name,
        action=action,
        multiplier=clamp(multiplier, 0.0, MAX_BOOST),
        stop_multiplier=stop_multiplier,
        reason=reason,
        detail=detail or {},
    )


# --------------------------------------------------------------------------- #
# Level 1 — monetary conditions
# --------------------------------------------------------------------------- #
class LiquidityFilter(Filter):
    """ΔGLI gate. Contracting liquidity throttles momentum and breakout longs."""

    name = "global_liquidity"
    level = 1

    MOMENTUM_FAMILIES = {"trend_following", "ma_cross", "donchian_breakout", "squeeze_breakout"}

    def evaluate(self, signal: RawSignal, ctx: FilterContext) -> FilterOutcome:
        macro = ctx.macro
        p = ctx.params.liquidity
        detail = {
            "gli_delta": macro.gli_delta,
            "m2_change": macro.m2_change,
            "stablecoin_change": macro.stablecoin_change,
            "regime": macro.gli_regime.value,
        }

        if macro.gli_regime is not LiquidityRegime.CONTRACTION:
            return _outcome(
                self.name, 1.0,
                f"Liquidity {macro.gli_regime.value} (ΔGLI {macro.gli_delta:+.2%})",
                detail,
            )

        # Contraction: momentum longs are the levered bet on liquidity, so they
        # take the penalty. Shorts and mean-reversion are left alone.
        is_momentum = signal.strategy in self.MOMENTUM_FAMILIES
        if signal.direction is Direction.LONG and is_momentum:
            severity = clamp(abs(macro.gli_delta) / max(p.strong_expansion_threshold, 1e-6), 0.0, 1.0)
            multiplier = 1.0 - (1.0 - p.contraction_penalty) * severity
            return _outcome(
                self.name, multiplier,
                f"Liquidity contracting (ΔGLI {macro.gli_delta:+.2%}) — momentum long penalised",
                detail,
            )
        if signal.direction is Direction.SHORT and is_momentum:
            return _outcome(
                self.name, 1.05,
                f"Liquidity contracting (ΔGLI {macro.gli_delta:+.2%}) — supports momentum short",
                detail,
            )
        return _outcome(
            self.name, 1.0,
            f"Liquidity contracting (ΔGLI {macro.gli_delta:+.2%}) — non-momentum setup unaffected",
            detail,
        )


class YieldCurveFilter(Filter):
    """Term-premium gate. An inverted anchor curve throttles high-beta longs."""

    name = "yield_curve"
    level = 1

    def evaluate(self, signal: RawSignal, ctx: FilterContext) -> FilterOutcome:
        p = ctx.params.yields
        view = ctx.rate_view
        spread = view.anchor_spread if view else None
        regime = ctx.macro.macro_regime
        detail = {"anchor_spread": spread, "macro_regime": regime.value}

        if spread is None:
            return _outcome(self.name, 1.0, "Yield curve data unavailable", detail)

        if regime is not MacroRegime.CONTRACTION:
            return _outcome(
                self.name, 1.0,
                f"Curve {regime.value} (10Y-2Y {spread:+.2%})", detail,
            )

        if signal.direction is Direction.LONG and is_high_beta(signal.instrument):
            if p.block_high_beta_on_inversion:
                return _outcome(
                    self.name, p.contraction_penalty,
                    f"Inverted curve (10Y-2Y {spread:+.2%}) — high-beta long blocked",
                    detail, block=True,
                )
            return _outcome(
                self.name, p.contraction_penalty,
                f"Inverted curve (10Y-2Y {spread:+.2%}) — high-beta long penalised", detail,
            )

        if signal.direction is Direction.LONG:
            return _outcome(
                self.name, p.contraction_penalty,
                f"Inverted curve (10Y-2Y {spread:+.2%}) — contraction regime penalty on longs",
                detail,
            )
        return _outcome(
            self.name, 1.0,
            f"Inverted curve (10Y-2Y {spread:+.2%}) — short side unaffected", detail,
        )


# --------------------------------------------------------------------------- #
# Level 2 — event risk
# --------------------------------------------------------------------------- #
def relevant_events(
    events: list[EconomicEvent],
    instrument: Instrument,
    now: datetime,
    minutes_before: int,
    minutes_after: int,
) -> list[EconomicEvent]:
    """Events for either leg of the pair inside the blackout window.

    USD events matter for crypto too — a hot CPI print moves BTC as surely as it
    moves EURUSD, because both are priced in dollars and dollar liquidity.
    """
    currencies = {instrument.base, instrument.quote}
    if instrument.is_crypto:
        currencies.add("USD")

    window_start = now - timedelta(minutes=minutes_after)
    window_end = now + timedelta(minutes=minutes_before)
    return [
        e for e in events
        if e.currency in currencies and window_start <= e.ts <= window_end
    ]


class EventBlockerFilter(Filter):
    """Macro-event blackout: no new signals within ±N minutes of a high-impact release."""

    name = "event_blocker"
    level = 2

    def evaluate(self, signal: RawSignal, ctx: FilterContext) -> FilterOutcome:
        p = ctx.params.events
        now = ctx.macro.ts or datetime.now(timezone.utc)
        hits = relevant_events(
            ctx.macro.events, signal.instrument, now,
            p.blackout_minutes_before, p.blackout_minutes_after,
        )
        if not hits:
            return _outcome(self.name, 1.0, "No high-impact release inside the blackout window")

        high = [e for e in hits if e.impact in p.blocking_impacts]
        if high:
            event = min(high, key=lambda e: abs((e.ts - now).total_seconds()))
            minutes = (event.ts - now).total_seconds() / 60.0
            when = f"in {minutes:.0f} min" if minutes > 0 else f"{abs(minutes):.0f} min ago"
            return _outcome(
                self.name, 0.0,
                f"{event.currency} {event.name} {when} — inside ±"
                f"{p.blackout_minutes_before}/{p.blackout_minutes_after} min blackout",
                {"event": event.name, "currency": event.currency,
                 "minutes_to_event": minutes, "impact": event.impact},
                block=True,
            )

        medium = [e for e in hits if e.impact == "medium"]
        if medium:
            event = medium[0]
            return _outcome(
                self.name, p.medium_impact_penalty,
                f"Medium-impact {event.currency} {event.name} nearby — confidence reduced",
                {"event": event.name, "currency": event.currency},
            )
        return _outcome(self.name, 1.0, "Only low-impact events nearby")


# --------------------------------------------------------------------------- #
# Level 3 — relative economics (forex)
# --------------------------------------------------------------------------- #
class RealYieldFilter(Filter):
    """Real Interest Rate Differential alignment."""

    name = "real_yield_differential"
    level = 3
    applies_to = ("forex",)

    def evaluate(self, signal: RawSignal, ctx: FilterContext) -> FilterOutcome:
        view = ctx.rate_view
        if view is None or view.rird is None:
            return _outcome(self.name, 1.0, "Rate data unavailable for one leg")

        state, multiplier = view.alignment(signal.direction, ctx.params.rird)
        detail = {
            "rird": view.rird,
            "real_rate_base": view.real_rate_base,
            "real_rate_quote": view.real_rate_quote,
            "nominal_carry": view.nominal_carry,
            "state": state,
        }
        label = (
            f"ΔRIRD {view.rird:+.2%} ({view.base} {view.real_rate_base:+.2%} vs "
            f"{view.quote} {view.real_rate_quote:+.2%})"
        )

        if state == "hard_opposed":
            return _outcome(
                self.name, multiplier,
                f"{label} — structurally opposed beyond "
                f"{ctx.params.rird.hard_block_differential:.1%}, signal blocked",
                detail, block=True,
            )
        if state == "aligned":
            return _outcome(self.name, min(multiplier, MAX_BOOST),
                            f"{label} — real yield supports the trade", detail)
        if state == "opposed":
            return _outcome(self.name, multiplier, f"{label} — real yield opposes the trade", detail)
        return _outcome(self.name, 1.0, f"{label} — differential not significant", detail)


class ValuationFilter(Filter):
    """PPP / REER valuation bands — stop chasing multi-sigma extremes."""

    name = "ppp_valuation"
    level = 3
    applies_to = ("forex",)

    def evaluate(self, signal: RawSignal, ctx: FilterContext) -> FilterOutcome:
        if ctx.ppp_base is None and ctx.ppp_quote is None:
            return _outcome(self.name, 1.0, "REER data unavailable")

        pair_z = pair_valuation_bias(signal.instrument, ctx.ppp_base, ctx.ppp_quote)
        state, multiplier, block = valuation_multiplier(pair_z, signal.direction, ctx.params.ppp)
        detail = {
            "pair_z": pair_z,
            "base_z": ctx.ppp_base.z_score if ctx.ppp_base else None,
            "quote_z": ctx.ppp_quote.z_score if ctx.ppp_quote else None,
            "state": state,
            "half_life_days": ctx.ppp_base.half_life_days if ctx.ppp_base else None,
        }
        label = f"Pair REER z-score {pair_z:+.2f}"

        if block:
            return _outcome(
                self.name, multiplier,
                f"{label} — macro-overvalued beyond ±{ctx.params.ppp.block_breakouts_beyond_z}, "
                "trend continuation blocked",
                detail, block=True,
            )
        if state in ("overextended", "stretched"):
            return _outcome(self.name, multiplier,
                            f"{label} — {state.replace('_', ' ')} against the trade", detail)
        if state == "value_tailwind":
            return _outcome(self.name, 1.0, f"{label} — valuation tailwind", detail)
        return _outcome(self.name, 1.0, f"{label} — fair value range", detail)


class SurpriseFilter(Filter):
    """Economic Surprise Index differential."""

    name = "economic_surprise"
    level = 3
    applies_to = ("forex",)

    def evaluate(self, signal: RawSignal, ctx: FilterContext) -> FilterOutcome:
        inst = signal.instrument
        differential = esi_differential(ctx.macro.esi, inst.base, inst.quote)
        state, multiplier = esi_multiplier(differential, signal.direction, ctx.params.esi)
        detail = {
            "esi_base": ctx.macro.esi.get(inst.base, 0.0),
            "esi_quote": ctx.macro.esi.get(inst.quote, 0.0),
            "differential": differential,
            "state": state,
        }
        label = (
            f"ESI {inst.base} {detail['esi_base']:+.2f} vs {inst.quote} "
            f"{detail['esi_quote']:+.2f} (Δ {differential:+.2f})"
        )
        if state == "aligned":
            return _outcome(self.name, min(multiplier, MAX_BOOST),
                            f"{label} — data momentum supports the trade", detail)
        if state == "opposed":
            return _outcome(self.name, multiplier,
                            f"{label} — data momentum opposes the trade", detail)
        return _outcome(self.name, 1.0, f"{label} — no meaningful skew", detail)


# --------------------------------------------------------------------------- #
# Level 3 — network economics (crypto)
# --------------------------------------------------------------------------- #
class TokenomicsFilter(Filter):
    """MV=PQ velocity divergence and NVT valuation."""

    name = "tokenomics"
    level = 3
    applies_to = ("crypto",)

    def evaluate(self, signal: RawSignal, ctx: FilterContext) -> FilterOutcome:
        view = ctx.tokenomics
        if view is None:
            return _outcome(self.name, 1.0, "On-chain data unavailable")

        state, multiplier = tokenomics_multiplier(view, signal.direction, ctx.params.tokenomics)
        detail = {
            "velocity": view.velocity,
            "velocity_z": view.velocity_z,
            "nvt_signal": view.nvt_signal,
            "nvt_z": view.nvt_z,
            "price_change": view.price_change,
            "state": state,
        }
        if state == "n/a":
            return _outcome(self.name, 1.0, "Not applicable to short setups", detail)
        if view.warning:
            return _outcome(
                self.name, multiplier,
                f"{view.warning} (velocity z {view.velocity_z:+.2f}, NVT z {view.nvt_z:+.2f})",
                detail,
            )
        return _outcome(
            self.name, 1.0,
            f"Network healthy (velocity z {view.velocity_z:+.2f}, NVT z {view.nvt_z:+.2f})",
            detail,
        )


# --------------------------------------------------------------------------- #
# Level 4 — microstructure and behaviour
# --------------------------------------------------------------------------- #
class MicrostructureFilter(Filter):
    """Spread/ATR friction: halt on thin books, widen stops when merely thin."""

    name = "microstructure"
    level = 4

    def evaluate(self, signal: RawSignal, ctx: FilterContext) -> FilterOutcome:
        view = ctx.microstructure
        if view is None:
            return _outcome(self.name, 1.0, "No live quote available")

        detail = {
            "spread_bps": view.spread_bps,
            "atr": view.atr,
            "friction": view.friction,
            "state": view.state,
            "imbalance": view.imbalance,
        }
        if view.should_halt:
            return _outcome(
                self.name, 0.0,
                f"Liquidity friction {view.friction:.2f} above halt threshold "
                f"{ctx.params.microstructure.friction_halt:.2f} — execution unsafe",
                detail, block=True,
            )
        if view.state == "thin":
            return _outcome(
                self.name, view.size_multiplier,
                f"Thin book (LF {view.friction:.2f}, spread {view.spread_bps:.1f} bps) — "
                f"stop widened {view.stop_multiplier:.2f}x, size cut to keep risk constant",
                detail, stop_multiplier=view.stop_multiplier,
            )
        return _outcome(
            self.name, 1.0,
            f"Normal depth (LF {view.friction:.2f}, spread {view.spread_bps:.1f} bps)", detail,
        )


class SentimentFilter(Filter):
    """Contrarian penalty on crowded positioning, Fear & Greed, and news polarity."""

    name = "behavioral_sentiment"
    level = 4

    def evaluate(self, signal: RawSignal, ctx: FilterContext) -> FilterOutcome:
        symbol = signal.instrument.symbol
        pct_long = ctx.macro.retail_long_pct.get(symbol, 0.5)
        news = ctx.macro.news_sentiment.get(symbol)
        view = compute_behavioral(
            pct_long, ctx.macro.fear_greed, news, signal.direction, ctx.params.behavioral
        )

        detail = {
            "retail_long_pct": view.pct_long,
            "crowding": view.crowding,
            "contrarian_state": view.contrarian_state,
            "fear_greed": view.fear_greed,
            "fear_greed_state": view.fear_greed_state,
            "news_score": view.news_score,
        }
        multiplier = clamp(view.combined, 0.05, MAX_BOOST)

        parts = [f"Retail {view.pct_long:.0%} net-long ({view.contrarian_state})"]
        if view.fear_greed is not None:
            parts.append(f"Fear & Greed {view.fear_greed:.0f} ({view.fear_greed_state})")
        if view.news_score is not None:
            parts.append(f"news {view.news_score:+.2f} ({view.news_state})")

        return _outcome(self.name, multiplier, "; ".join(parts), detail)


# --------------------------------------------------------------------------- #
# Level 5 — portfolio risk
# --------------------------------------------------------------------------- #
class RiskBudgetFilter(Filter):
    """CVaR budget gate — the portfolio-level circuit breaker."""

    name = "risk_budget"
    level = 5

    def evaluate(self, signal: RawSignal, ctx: FilterContext) -> FilterOutcome:
        budget = ctx.risk_budget
        if budget is None:
            return _outcome(self.name, 1.0, "No portfolio history to evaluate")

        detail = {
            "cvar": budget.cvar,
            "var": budget.var,
            "budget": budget.budget,
            "utilisation": budget.utilisation,
            "sortino": budget.sortino,
            "max_drawdown": budget.max_drawdown,
        }
        if budget.breached:
            return _outcome(
                self.name, budget.scaler,
                f"CVaR{ctx.params.risk.cvar_alpha:.0%} {budget.cvar:.2%} exceeds budget "
                f"{budget.budget:.2%} — exposure scaled {budget.scaler:.2f}x",
                detail,
            )
        return _outcome(
            self.name, 1.0,
            f"CVaR{ctx.params.risk.cvar_alpha:.0%} {budget.cvar:.2%} within budget "
            f"{budget.budget:.2%} ({budget.utilisation:.0%} used)",
            detail,
        )


class RegimeAlignmentFilter(Filter):
    """Down-weight strategies whose premise does not fit the current regime."""

    name = "regime_alignment"
    level = 5

    FAMILY_BY_STRATEGY = {
        "trend_following": "momentum",
        "ma_cross": "momentum",
        "ml_classifier": "momentum",
        "bollinger_reversion": "mean_reversion",
        "rsi_reversion": "mean_reversion",
        "pairs_trading": "mean_reversion",
        "donchian_breakout": "breakout",
        "squeeze_breakout": "breakout",
        "funding_carry": "carry",
        "grid": "market_making",
    }

    def evaluate(self, signal: RawSignal, ctx: FilterContext) -> FilterOutcome:
        regime = ctx.regime
        if regime is None:
            return _outcome(self.name, 1.0, "Regime unavailable")

        family = self.FAMILY_BY_STRATEGY.get(signal.strategy, "momentum")
        weight = regime.strategy_weight(family)
        detail = {
            "regime": regime.regime.value,
            "confidence": regime.confidence,
            "family": family,
            "weight": weight,
            "hurst": regime.hurst,
            "adx": regime.adx,
        }

        # Blend the weight toward 1.0 when the regime call itself is uncertain —
        # a low-confidence classification should not drive a large penalty.
        adjusted = 1.0 + (weight - 1.0) * clamp(regime.confidence, 0.0, 1.0)
        label = (
            f"Regime {regime.regime.value} (confidence {regime.confidence:.0%}, "
            f"Hurst {regime.hurst:.2f}) vs {family} strategy"
        )
        if regime.regime.value == "crash" and signal.direction is Direction.LONG:
            return _outcome(self.name, min(adjusted, 0.5),
                            f"{label} — crash regime, longs heavily de-weighted", detail)
        return _outcome(self.name, clamp(adjusted, 0.2, MAX_BOOST), label, detail)


# --------------------------------------------------------------------------- #
# Chain assembly
# --------------------------------------------------------------------------- #
DEFAULT_FILTERS: tuple[type[Filter], ...] = (
    EventBlockerFilter,     # cheapest hard block first
    MicrostructureFilter,   # second hard block: unexecutable is unexecutable
    LiquidityFilter,
    YieldCurveFilter,
    RealYieldFilter,
    ValuationFilter,
    SurpriseFilter,
    TokenomicsFilter,
    SentimentFilter,
    RegimeAlignmentFilter,
    RiskBudgetFilter,
)


def build_filters(names: list[str] | None = None) -> list[Filter]:
    """Instantiate the chain, optionally restricted to a subset of filter names."""
    filters = [cls() for cls in DEFAULT_FILTERS]
    if names:
        wanted = set(names)
        filters = [f for f in filters if f.name in wanted]
    return sorted(filters, key=lambda f: f.level)
