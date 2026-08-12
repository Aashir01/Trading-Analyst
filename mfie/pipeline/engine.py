"""The orchestrator: data -> indicators -> strategies -> filters -> sizing.

One ``AnalysisEngine.run()`` call produces a complete, explainable answer for a
set of instruments. The macro snapshot is built **once per run** so that every
instrument and every filter sees an identical view of the world — otherwise two
signals in the same run could disagree about whether the yield curve is
inverted, which makes the audit trail worthless.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from mfie.alpha.cycle import get_cycle_engine
from mfie.config import Params, get_params
from mfie.core.types import (
    AssetClass,
    Direction,
    FilterAction,
    FilterOutcome,
    Instrument,
    MacroContext,
    RawSignal,
    ScoredSignal,
)
from mfie.core.universe import TRACKED_CURRENCIES, resolve
from mfie.core.utils import annualization_factor, get_logger, timed
from mfie.data.hub import DataHub, get_hub
from mfie.econ.liquidity import compute_gli
from mfie.econ.microstructure import compute_microstructure
from mfie.econ.rates import compute_rate_view
from mfie.econ.risk import evaluate_risk_budget
from mfie.econ.surprise import compute_esi
from mfie.econ.tokenomics import compute_tokenomics
from mfie.econ.valuation import compute_ppp_deviation
from mfie.pipeline.filters import Filter, FilterContext, build_filters
from mfie.pipeline.sizing import apply_sizing, size_signal
from mfie.regime.detector import detect_regime
from mfie.strategies.base import Strategy, StrategyContext
from mfie.strategies.registry import build_strategies
from mfie.technical.indicators import compute_indicators

log = get_logger(__name__)


@dataclass
class InstrumentAnalysis:
    """Per-instrument output: the data, the diagnostics, and the signals."""

    instrument: Instrument
    df: pd.DataFrame
    indicators: object
    regime: object
    signals: list[ScoredSignal] = field(default_factory=list)
    filter_context: FilterContext | None = None
    error: str | None = None

    @property
    def best(self) -> ScoredSignal | None:
        tradable = [s for s in self.signals if not s.blocked]
        return max(tradable, key=lambda s: s.confidence) if tradable else None

    @property
    def price(self) -> float:
        return float(self.df["close"].iloc[-1]) if not self.df.empty else 0.0


@dataclass
class AnalysisResult:
    ts: datetime
    macro: MacroContext
    analyses: dict[str, InstrumentAnalysis] = field(default_factory=dict)
    timeframe: str = "1h"

    @property
    def all_signals(self) -> list[ScoredSignal]:
        return [s for a in self.analyses.values() for s in a.signals]

    @property
    def tradable(self) -> list[ScoredSignal]:
        signals = [s for s in self.all_signals if not s.blocked and s.risk_fraction > 0]
        return sorted(signals, key=lambda s: s.confidence, reverse=True)

    @property
    def blocked(self) -> list[ScoredSignal]:
        return [s for s in self.all_signals if s.blocked]

    def summary(self) -> dict[str, object]:
        return {
            "ts": self.ts,
            "instruments": len(self.analyses),
            "signals": len(self.all_signals),
            "tradable": len(self.tradable),
            "blocked": len(self.blocked),
            "liquidity_regime": self.macro.gli_regime.value,
            "macro_regime": self.macro.macro_regime.value,
            "gli_delta": self.macro.gli_delta,
            "fear_greed": self.macro.fear_greed,
            "portfolio_cvar": self.macro.portfolio_cvar,
        }


class AnalysisEngine:
    def __init__(
        self,
        hub: DataHub | None = None,
        params: Params | None = None,
        strategies: list[Strategy] | None = None,
        filters: list[Filter] | None = None,
    ) -> None:
        self.hub = hub or get_hub()
        self.params = params or get_params()
        self.strategies = strategies if strategies is not None else build_strategies()
        self.filters = filters if filters is not None else build_filters()
        self.cycle_engine = get_cycle_engine(self.hub, self.params)

    # ------------------------------------------------------------------ macro
    @timed
    def build_macro_context(self, instruments: list[Instrument]) -> MacroContext:
        """Assemble the shared macro snapshot for this run."""
        hub = self.hub
        now = datetime.now(timezone.utc)

        m2 = hub.m2_series()
        stablecoins = hub.stablecoin_supply()
        gli = compute_gli(m2, stablecoins, self.params.liquidity)

        policy_rates = hub.policy_rates()
        inflation = hub.inflation()
        yields_10y, yields_2y = hub.yields()

        events = hub.economic_events()
        esi = compute_esi(events, self.params.esi, as_of=now)

        from mfie.econ.rates import classify_macro_regime

        macro_regime, _ = classify_macro_regime(yields_10y, yields_2y, self.params.yields)

        # REER only for currencies actually in play this run.
        currencies = {c for inst in instruments for c in (inst.base, inst.quote)}
        currencies &= set(TRACKED_CURRENCIES)
        reer_history = {c: hub.reer_series(c) for c in sorted(currencies)}

        # Market Cycle Compass, built once per run for the domains actually in
        # play. It is the slowest block here (daily history for a whole
        # domain), so it is skipped entirely for domains with no instruments.
        cycle_states: dict[str, object] = {}
        domains = {"crypto" if inst.is_crypto else "fx" for inst in instruments}
        for domain in sorted(domains):
            try:
                cycle_states[domain] = self.cycle_engine.evaluate(domain)
            except Exception as exc:
                log.warning("Cycle compass failed for %s: %s", domain, exc)

        fear_greed = hub.fear_greed()
        retail = {inst.symbol: hub.retail_positioning(inst) for inst in instruments}
        news = {inst.symbol: hub.news_sentiment(inst) for inst in instruments}

        token_velocity: dict[str, float] = {}
        token_velocity_z: dict[str, float] = {}
        nvt: dict[str, float] = {}
        nvt_z: dict[str, float] = {}
        self._tokenomics_cache: dict[str, object] = {}
        for inst in instruments:
            if not inst.is_crypto:
                continue
            view = compute_tokenomics(
                inst.symbol,
                hub.onchain_volume(inst),
                hub.market_cap(inst),
                self.params.tokenomics,
            )
            self._tokenomics_cache[inst.symbol] = view
            token_velocity[inst.symbol] = view.velocity
            token_velocity_z[inst.symbol] = view.velocity_z
            nvt[inst.symbol] = view.nvt
            nvt_z[inst.symbol] = view.nvt_z

        return MacroContext(
            ts=now,
            gli_delta=gli.delta,
            gli_regime=gli.regime,
            m2_change=gli.m2_change,
            stablecoin_change=gli.stablecoin_change,
            policy_rates=policy_rates,
            inflation=inflation,
            yield_10y=yields_10y,
            yield_2y=yields_2y,
            macro_regime=macro_regime,
            esi=esi,
            reer_history=reer_history,
            token_velocity=token_velocity,
            token_velocity_z=token_velocity_z,
            nvt=nvt,
            nvt_z=nvt_z,
            fear_greed=fear_greed,
            retail_long_pct=retail,
            news_sentiment=news,
            events=events,
            cycle_states=cycle_states,
            sources=dict(hub.sources),
        )

    # ------------------------------------------------------------------- run
    @timed
    def run(
        self,
        symbols: list[str] | None = None,
        timeframe: str = "1h",
        limit: int = 500,
        asset_class: AssetClass | None = None,
        portfolio_returns: pd.Series | None = None,
    ) -> AnalysisResult:
        instruments = resolve(symbols, asset_class)
        macro = self.build_macro_context(instruments)

        risk_budget = None
        if portfolio_returns is not None and len(portfolio_returns.dropna()) >= 20:
            risk_budget = evaluate_risk_budget(portfolio_returns, self.params.risk)
            macro.portfolio_cvar = risk_budget.cvar
            macro.portfolio_sortino = risk_budget.sortino

        # Price history for every instrument up front — pairs trading needs to
        # see other symbols, so this cannot be done lazily inside the loop.
        frames: dict[str, pd.DataFrame] = {}
        for inst in instruments:
            try:
                frames[inst.symbol] = self.hub.ohlcv(inst, timeframe, limit)
            except Exception as exc:
                log.warning("Failed to load %s: %s", inst.symbol, exc)
                frames[inst.symbol] = pd.DataFrame()

        result = AnalysisResult(ts=macro.ts, macro=macro, timeframe=timeframe)
        open_risk = 0.0

        for inst in instruments:
            analysis = self._analyse_instrument(
                inst, frames, macro, timeframe, risk_budget, open_risk
            )
            result.analyses[inst.symbol] = analysis
            best = analysis.best
            if best is not None:
                open_risk += best.risk_fraction

        macro.open_risk = open_risk
        return result

    # ----------------------------------------------------------- per instrument
    def _analyse_instrument(
        self,
        instrument: Instrument,
        frames: dict[str, pd.DataFrame],
        macro: MacroContext,
        timeframe: str,
        risk_budget,
        open_risk: float,
    ) -> InstrumentAnalysis:
        df = frames.get(instrument.symbol, pd.DataFrame())
        if df.empty or len(df) < 60:
            return InstrumentAnalysis(
                instrument=instrument, df=df, indicators=None, regime=None,
                error="insufficient price history",
            )

        periods_per_year = annualization_factor(timeframe)
        indicators = compute_indicators(df, self.params.technical, periods_per_year)
        regime = detect_regime(indicators, self.params.regime)

        quote = self.hub.quote(instrument)
        atr_value = indicators.last("atr", 0.0)
        micro = compute_microstructure(
            quote, atr_value, self.params.microstructure,
            returns=indicators.series("returns"),
            volume_usd=df["volume"] * df["close"],
        )

        rate_view = compute_rate_view(
            instrument, macro.policy_rates, macro.inflation,
            macro.yield_10y, macro.yield_2y, self.params.yields,
        )

        ppp_base = ppp_quote = None
        if instrument.is_forex:
            if (series := macro.reer_history.get(instrument.base)) is not None:
                ppp_base = compute_ppp_deviation(instrument.base, series, self.params.ppp)
            if (series := macro.reer_history.get(instrument.quote)) is not None:
                ppp_quote = compute_ppp_deviation(instrument.quote, series, self.params.ppp)

        tokenomics = getattr(self, "_tokenomics_cache", {}).get(instrument.symbol)

        filter_ctx = FilterContext(
            instrument=instrument,
            macro=macro,
            params=self.params,
            quote=quote,
            atr=atr_value,
            regime=regime,
            rate_view=rate_view,
            ppp_base=ppp_base,
            ppp_quote=ppp_quote,
            tokenomics=tokenomics,
            microstructure=micro,
            risk_budget=risk_budget,
        )

        strategy_ctx = StrategyContext(
            instrument=instrument,
            df=df,
            indicators=indicators,
            regime=regime,
            timeframe=timeframe,
            params=self.params,
            extras=self._strategy_extras(instrument, frames, quote),
        )

        analysis = InstrumentAnalysis(
            instrument=instrument, df=df, indicators=indicators,
            regime=regime, filter_context=filter_ctx,
        )

        for strategy in self.strategies:
            if not strategy.supports(strategy_ctx):
                continue
            try:
                raw = strategy.generate(strategy_ctx)
            except Exception as exc:
                log.warning("Strategy %s failed on %s: %s", strategy.name, instrument.symbol, exc)
                continue
            if raw is None or raw.direction is Direction.FLAT:
                continue
            analysis.signals.append(
                self._score_signal(raw, filter_ctx, indicators, risk_budget, open_risk, periods_per_year)
            )

        return analysis

    def _strategy_extras(
        self,
        instrument: Instrument,
        frames: dict[str, pd.DataFrame],
        quote,
    ) -> dict[str, object]:
        extras: dict[str, object] = {
            "spread_pct": quote.spread / quote.mid if quote and quote.mid else 0.0005,
        }
        if instrument.is_crypto:
            try:
                extras["funding_rate"] = self.hub.funding_rate(instrument)
            except Exception as exc:
                log.debug("Funding rate unavailable for %s: %s", instrument.symbol, exc)

        partner = self._best_partner(instrument, frames)
        if partner is not None:
            symbol, series = partner
            extras["partner_symbol"] = symbol
            extras["partner_close"] = series
        return extras

    def _best_partner(
        self,
        instrument: Instrument,
        frames: dict[str, pd.DataFrame],
    ) -> tuple[str, pd.Series] | None:
        """Highest-correlation same-asset-class partner, for pairs trading.

        Correlation is a cheap pre-screen; the strategy itself runs the actual
        cointegration test and rejects the pair if it fails.
        """
        own = frames.get(instrument.symbol)
        if own is None or own.empty:
            return None

        best_symbol, best_corr, best_series = None, 0.0, None
        for symbol, frame in frames.items():
            if symbol == instrument.symbol or frame.empty:
                continue
            from mfie.core.universe import get_instrument

            try:
                other = get_instrument(symbol)
            except KeyError:
                continue
            if other.asset_class is not instrument.asset_class:
                continue

            common = own.index.intersection(frame.index)
            if len(common) < 150:
                continue
            corr = float(own.loc[common, "close"].corr(frame.loc[common, "close"]))
            if abs(corr) > abs(best_corr):
                best_symbol, best_corr, best_series = symbol, corr, frame["close"]

        if best_symbol is None or abs(best_corr) < 0.6:
            return None
        return best_symbol, best_series

    def _score_signal(
        self,
        raw: RawSignal,
        filter_ctx: FilterContext,
        indicators,
        risk_budget,
        open_risk: float,
        periods_per_year: float,
    ) -> ScoredSignal:
        scored = ScoredSignal(raw=raw, macro=filter_ctx.macro)

        for filt in self.filters:
            outcome = filt(raw, filter_ctx)
            scored.outcomes.append(outcome)
            if outcome.blocked:
                scored.blocked = True
                scored.block_reasons.append(f"{outcome.name}: {outcome.reason}")
                # Stop on the first hard block: later filters would only add
                # noise to an audit trail whose verdict is already decided.
                break

        if scored.blocked:
            scored.confidence = 0.0
            scored.adjusted_stop = raw.stop
            scored.adjusted_take_profit = raw.take_profit
            return scored

        decision = size_signal(
            raw,
            scored.outcomes,
            params=self.params,
            equity=self.params.risk.account_equity,
            risk_scaler=risk_budget.scaler if risk_budget else 1.0,
            realized_vol=indicators.last("realized_vol", 0.0) or None,
            open_risk=open_risk,
        )
        apply_sizing(scored, decision)

        # A zero-risk outcome is not a "block" (no rule vetoed it) but the user
        # still needs to know why nothing is tradable, so the notes are surfaced.
        if decision.risk_fraction == 0 and decision.notes:
            scored.block_reasons.extend(decision.notes)

        scored.outcomes.append(
            FilterOutcome(
                name="sizing",
                action=FilterAction.PASS,
                multiplier=1.0,
                reason="; ".join(decision.notes) if decision.notes
                else f"Risk {decision.risk_fraction:.2%} of equity, {decision.units:.4f} units",
                detail={
                    "kelly_cap": decision.kelly_cap,
                    "vol_scalar": decision.vol_scalar,
                    "notional": decision.notional,
                    "risk_fraction": decision.risk_fraction,
                },
            )
        )
        return scored


def quick_analyze(
    symbols: list[str] | None = None,
    timeframe: str = "1h",
    limit: int = 500,
) -> AnalysisResult:
    """Convenience entry point used by the CLI, dashboard and bot."""
    return AnalysisEngine().run(symbols=symbols, timeframe=timeframe, limit=limit)
