r"""The Market Cycle Compass.

Fuses the factor panel into a single score in :math:`[-1, +1]`, maps it to a
four-phase cycle state with hysteresis, estimates the probability of leaving
that state, and flags the one configuration worth losing sleep over: price
making new highs while the macro internals deteriorate.

The four phases are the classic cycle, and the order matters — the model tracks
where you are in the loop, not just whether the number is positive:

.. code-block:: text

    EARLY_RECOVERY  ──►  EXPANSION  ──►  LATE_EXPANSION  ──►  CONTRACTION
          ▲                                                        │
          └────────────────────────────────────────────────────────┘

* ``EARLY_RECOVERY`` — score still negative but rising. Historically the highest
  forward returns and the hardest phase to act in, because the news is awful.
* ``EXPANSION`` — score positive and rising. Trend-following works.
* ``LATE_EXPANSION`` — score positive but falling, usually with price/internals
  divergence. The distribution phase: still up, no longer healthy.
* ``CONTRACTION`` — score negative and falling. Capital preservation.

Phase is decided by the *pair* (level, direction), which is why a falling
positive score is a different animal from a rising positive one — a distinction
a single threshold cannot make.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

import numpy as np
import pandas as pd

from mfie.alpha import factors as F
from mfie.alpha.leadlag import (
    LeadLagResult,
    ValidationReport,
    _t_stat,
    align_to_common_horizon,
    forward_return,
    information_coefficient,
    scan_leads,
    shrink_weights,
)
from mfie.config import CycleParams, Params, get_params
from mfie.core.types import AssetClass
from mfie.core.universe import CRYPTO, FOREX, get_instrument
from mfie.core.utils import clamp, get_logger
from mfie.data.hub import DataHub, get_hub

log = get_logger(__name__)

HAVEN_PAIRS = ("USDJPY", "USDCHF")


class CyclePhase(str, Enum):
    EARLY_RECOVERY = "early_recovery"
    EXPANSION = "expansion"
    LATE_EXPANSION = "late_expansion"
    CONTRACTION = "contraction"
    NEUTRAL = "neutral"

    @property
    def is_bullish(self) -> bool:
        return self in (CyclePhase.EARLY_RECOVERY, CyclePhase.EXPANSION)

    @property
    def is_bearish(self) -> bool:
        return self in (CyclePhase.LATE_EXPANSION, CyclePhase.CONTRACTION)

    @property
    def label(self) -> str:
        return self.value.replace("_", " ").title()

    @property
    def stance(self) -> str:
        return {
            CyclePhase.EARLY_RECOVERY: "accumulate — risk is cheap and improving",
            CyclePhase.EXPANSION: "participate — trend strategies have a tailwind",
            CyclePhase.LATE_EXPANSION: "distribute — take profit into strength, tighten stops",
            CyclePhase.CONTRACTION: "preserve — defence, cash, or short",
            CyclePhase.NEUTRAL: "wait — no macro edge either way",
        }[self]


@dataclass
class FactorReading:
    name: str
    label: str
    score: float          # -1..+1
    weight: float
    contribution: float   # weight * score
    lead_days: int
    ic: float
    trusted: bool
    rationale: str

    @property
    def direction(self) -> str:
        if self.score > 0.15:
            return "risk-on"
        if self.score < -0.15:
            return "risk-off"
        return "neutral"


@dataclass
class CycleState:
    domain: str
    ts: datetime
    phase: CyclePhase
    score: float                 # -1..+1 composite
    momentum: float              # change in composite over the impulse window
    readings: list[FactorReading] = field(default_factory=list)
    transition_probability: float = 0.0
    days_in_phase: int = 0
    divergence: float = 0.0
    divergence_flag: bool = False
    confidence: float = 0.0
    previous_phase: CyclePhase | None = None
    history: pd.DataFrame | None = None
    narrative: list[str] = field(default_factory=list)
    data_quality: str = "unknown"

    @property
    def bias(self) -> str:
        """Coarse directional read, derived from the phase.

        Taken from the phase rather than re-thresholded off the score, so the
        two can never contradict each other in the same report.
        """
        if self.phase.is_bullish:
            return "BULL"
        if self.phase.is_bearish:
            return "BEAR"
        return "NEUTRAL"

    def top_drivers(self, n: int = 3) -> list[FactorReading]:
        return sorted(self.readings, key=lambda r: abs(r.contribution), reverse=True)[:n]

    def as_dict(self) -> dict:
        return {
            "domain": self.domain,
            "ts": self.ts,
            "phase": self.phase.value,
            "bias": self.bias,
            "score": self.score,
            "momentum": self.momentum,
            "transition_probability": self.transition_probability,
            "days_in_phase": self.days_in_phase,
            "divergence": self.divergence,
            "divergence_flag": self.divergence_flag,
            "confidence": self.confidence,
            "data_quality": self.data_quality,
            "factors": {
                r.name: {"score": r.score, "weight": r.weight, "lead_days": r.lead_days}
                for r in self.readings
            },
        }


class CycleEngine:
    """Builds the factor panel, fits weights, and reads the current cycle state."""

    def __init__(self, hub: DataHub | None = None, params: Params | None = None) -> None:
        self.hub = hub or get_hub()
        self.params = params or get_params()
        self.cfg: CycleParams = self.params.cycle
        self._panels: dict[str, F.FactorPanel] = {}
        self._leadlag: dict[str, dict[str, LeadLagResult]] = {}

    # ------------------------------------------------------------------ panel
    def build_panel(self, domain: str = "crypto", refresh: bool = False) -> F.FactorPanel:
        """Assemble the daily factor matrix for ``crypto`` or ``fx``."""
        if not refresh and domain in self._panels:
            return self._panels[domain]

        cfg = self.cfg
        days = cfg.history_days
        universe = CRYPTO if domain == "crypto" else FOREX

        # --- price history, risk-oriented -------------------------------
        oriented: dict[str, pd.Series] = {}
        raw_closes: dict[str, pd.Series] = {}
        for symbol, instrument in universe.items():
            frame = self.hub.ohlcv(instrument, "1d", days)
            if frame.empty or len(frame) < 90:
                continue
            raw_closes[symbol] = frame["close"]
            oriented[symbol] = F.risk_oriented(instrument, frame["close"])

        if not oriented:
            log.warning("No price history available for domain %s", domain)
            return F.FactorPanel(domain=domain, frame=pd.DataFrame(), anchor=pd.Series(dtype=float))

        if domain == "crypto":
            anchor = F.to_daily(oriented.get("BTCUSDT", next(iter(oriented.values()))))
        else:
            anchor = F.build_dollar_index(oriented)

        # --- macro inputs -----------------------------------------------
        m2 = self.hub.m2_series(60)
        stablecoins = self.hub.stablecoin_supply(min(days, 900))
        y10 = self.hub.yield_series("USD", "10y", days)
        y2 = self.hub.yield_series("USD", "2y", days)
        policy = self.hub.policy_rate_series("USD", days)
        cpi = self.hub.inflation_series("USD", days)

        columns: dict[str, pd.Series] = {}
        missing: list[str] = []

        def add(name: str, series: pd.Series) -> None:
            if series is None or series.dropna().empty:
                missing.append(name)
                return
            columns[name] = series

        add("liquidity_impulse", F.liquidity_impulse(m2, stablecoins, cfg))
        add("credit_regime", F.credit_regime(y10, y2, cfg))
        add("real_rate_impulse", F.real_rate_impulse(policy, cpi, cfg))
        add("breadth", F.breadth(oriented, cfg))

        # --- domain-specific factors ------------------------------------
        if domain == "crypto":
            btc = get_instrument("BTCUSDT")
            from mfie.econ.tokenomics import nvt_signal

            nvt = nvt_signal(
                self.hub.market_cap(btc, min(days, 365)),
                self.hub.onchain_volume(btc, min(days, 365)),
                cfg.impulse_window,
            )
            add("valuation", F.valuation_stretch(nvt, cfg, invert=True))
            add(
                "risk_appetite",
                F.risk_appetite_crypto(
                    self.hub.fear_greed_history(days),
                    self.hub.funding_rate(btc, 400),
                    cfg,
                ),
            )
        else:
            # A strong dollar is a global tightening — invert so rich USD reads bearish.
            add("valuation", F.valuation_stretch(self.hub.reer_series("USD", days), cfg, invert=True))
            havens = {
                symbol: F.risk_oriented(FOREX[symbol], raw_closes[symbol])
                for symbol in HAVEN_PAIRS
                if symbol in raw_closes
            }
            # risk_oriented already inverts USDxxx, so these series rise when the
            # haven currency strengthens — exactly the input the factor expects.
            add("risk_appetite", F.risk_appetite_fx(havens, cfg))

        add("trend_confirmation", F.trend_confirmation(anchor, cfg))

        frame = pd.concat(columns.values(), axis=1) if columns else pd.DataFrame()
        if not frame.empty:
            frame = frame.ffill()
            if cfg.smoothing_days > 1:
                # Macro factors are noisy day to day and the cycle is a
                # multi-month object; a short mean removes chatter without
                # meaningfully delaying a turn.
                frame = frame.rolling(cfg.smoothing_days, min_periods=1).mean()
            frame = frame.dropna(how="all")

        panel = F.FactorPanel(
            domain=domain,
            frame=frame,
            anchor=anchor,
            sources=dict(self.hub.sources),
            missing=missing,
        )
        self._panels[domain] = panel
        return panel

    # ---------------------------------------------------------------- weights
    def fit_weights(self, panel: F.FactorPanel) -> tuple[dict[str, float], dict[str, LeadLagResult]]:
        """Measure each factor's lead and blend its weight toward the prior."""
        cfg = self.cfg
        results: dict[str, LeadLagResult] = {}

        if len(panel) >= cfg.min_observations:
            for name in panel.frame.columns:
                results[name] = scan_leads(
                    panel.frame[name],
                    panel.anchor,
                    horizon=cfg.forward_horizon,
                    max_lead=cfg.max_lead_days,
                    step=cfg.lead_step_days,
                    min_tstat=cfg.min_tstat,
                    name=name,
                )
        else:
            log.info(
                "Only %d observations for %s — using economic priors without measurement",
                len(panel), panel.domain,
            )

        priors = {
            name: cfg.prior_weights.get(name, F.FACTOR_SPECS[name].prior_weight)
            for name in panel.frame.columns
            if name in F.FACTOR_SPECS
        }
        weights = shrink_weights(
            priors, results,
            max_multiple=cfg.max_weight_multiple,
            min_multiple=cfg.min_weight_multiple,
        )
        self._leadlag[panel.domain] = results
        return weights, results

    # -------------------------------------------------------------- composite
    def composite_series(self, panel: F.FactorPanel, weights: dict[str, float],
                         results: dict[str, LeadLagResult]) -> pd.Series:
        """Weighted composite over the whole history, lead-aligned if configured."""
        if panel.frame.empty:
            return pd.Series(dtype=float, name="cycle")

        frame = panel.frame
        if self.cfg.align_leads and results:
            frame = align_to_common_horizon(frame, results)

        usable = [c for c in frame.columns if c in weights]
        if not usable:
            return pd.Series(dtype=float, name="cycle")

        w = pd.Series({c: weights[c] for c in usable})
        # Renormalise per row over the factors that actually have data there, so
        # a factor that only starts halfway through the history does not silently
        # drag the composite toward zero for the first half.
        values = frame[usable]
        available = values.notna().astype(float) * w
        denominator = available.sum(axis=1).replace(0.0, np.nan)
        composite = (values.fillna(0.0) * w).sum(axis=1) / denominator
        return composite.clip(-1.0, 1.0).rename("cycle")

    # ------------------------------------------------------------------ state
    def _phase_series(self, composite: pd.Series) -> pd.Series:
        """Walk the composite forward through the hysteresis state machine."""
        cfg = self.cfg
        momentum = composite.diff(cfg.impulse_window).fillna(0.0)

        phases: list[str] = []
        current = CyclePhase.NEUTRAL
        streak = 0
        pending: CyclePhase | None = None

        for score, mom in zip(composite.to_numpy(), momentum.to_numpy(), strict=True):
            if not np.isfinite(score):
                phases.append(current.value)
                continue

            # Candidate phase from the (level, direction) pair.
            if score >= cfg.bull_entry:
                candidate = CyclePhase.EXPANSION if mom >= 0 else CyclePhase.LATE_EXPANSION
            elif score <= cfg.bear_entry:
                candidate = CyclePhase.EARLY_RECOVERY if mom > 0 else CyclePhase.CONTRACTION
            elif current.is_bullish and score > cfg.bull_exit:
                candidate = current            # hysteresis: hold the bull
            elif current.is_bearish and score < cfg.bear_exit:
                candidate = current            # hysteresis: hold the bear
            else:
                candidate = CyclePhase.NEUTRAL

            if candidate is current:
                pending, streak = None, 0
            else:
                # A new phase must persist before it is accepted, which is what
                # stops a one-day wobble from flipping the regime.
                if candidate is pending:
                    streak += 1
                else:
                    pending, streak = candidate, 1
                if streak >= cfg.confirm_days:
                    current = candidate
                    pending, streak = None, 0

            phases.append(current.value)

        return pd.Series(phases, index=composite.index, name="phase")

    def _hazard(self, composite: pd.Series, phases: pd.Series, current: CyclePhase) -> float:
        """Empirical probability of leaving the current phase within the horizon.

        Estimated from the realised history of this composite rather than
        assumed: among all past days in this phase with a similar score, how
        often was the phase different ``hazard_horizon`` days later?
        """
        cfg = self.cfg
        if len(phases) < cfg.hazard_horizon * 2:
            return 0.0

        future = phases.shift(-cfg.hazard_horizon)
        in_phase = phases == current.value
        comparable = in_phase & future.notna()
        if comparable.sum() < 30:
            return 0.0

        latest = float(composite.iloc[-1])
        # Condition on a similar composite level; the further from the phase
        # boundary you are, the less likely you are to leave it.
        band = 0.2
        similar = comparable & (composite - latest).abs().le(band)
        if similar.sum() < 20:
            similar = comparable

        changed = (future[similar] != phases[similar]).mean()
        return float(clamp(changed, 0.0, 1.0))

    def _divergence(self, composite: pd.Series, anchor: pd.Series) -> tuple[float, bool]:
        """Macro momentum minus price momentum, in standard deviations.

        Strongly negative is the distribution signature: price is making highs
        that the macro internals no longer support.
        """
        cfg = self.cfg
        window = cfg.divergence_window
        frame = pd.concat(
            [composite.rename("c"), F.to_daily(anchor).rename("p")], axis=1
        ).ffill().dropna()
        if len(frame) < window * 2:
            return 0.0, False

        comp_mom = frame["c"].diff(window)
        price_mom = frame["p"].pct_change(window)

        def _z(s: pd.Series) -> float:
            tail = s.dropna().tail(cfg.zscore_window)
            if len(tail) < 30:
                return 0.0
            sd = float(tail.std(ddof=1))
            return 0.0 if sd <= 0 else float((tail.iloc[-1] - tail.mean()) / sd)

        divergence = _z(comp_mom) - _z(price_mom)
        # Only a real disagreement counts: opposite signs, meaningfully apart.
        opposing = float(comp_mom.iloc[-1]) * float(price_mom.iloc[-1]) < 0
        flag = bool(opposing and abs(divergence) >= cfg.divergence_threshold)
        return float(divergence), flag

    # ----------------------------------------------------------------- public
    def evaluate(self, domain: str = "crypto", refresh: bool = False) -> CycleState:
        """Read the current cycle state for a domain."""
        panel = self.build_panel(domain, refresh=refresh)
        if panel.frame.empty:
            return CycleState(
                domain=domain, ts=datetime.now(timezone.utc), phase=CyclePhase.NEUTRAL,
                score=0.0, momentum=0.0, narrative=["No factor data available."],
                data_quality="unavailable",
            )

        weights, results = self.fit_weights(panel)
        composite = self.composite_series(panel, weights, results).dropna()
        if composite.empty:
            return CycleState(
                domain=domain, ts=datetime.now(timezone.utc), phase=CyclePhase.NEUTRAL,
                score=0.0, momentum=0.0, narrative=["Composite could not be computed."],
                data_quality="unavailable",
            )

        phases = self._phase_series(composite)
        current = CyclePhase(phases.iloc[-1])
        score = float(composite.iloc[-1])
        momentum = float(composite.diff(self.cfg.impulse_window).iloc[-1] or 0.0)

        days_in_phase = int((phases[::-1] != phases.iloc[-1]).cumsum().eq(0).sum())
        previous = None
        changes = phases[phases != phases.shift()]
        if len(changes) >= 2:
            previous = CyclePhase(changes.iloc[-2])

        divergence, divergence_flag = self._divergence(composite, panel.anchor)
        transition = self._hazard(composite, phases, current)

        readings = self._readings(panel, weights, results)

        # Confidence: how emphatic the score is, how much the factors agree, and
        # whether the data behind them is real.
        agreement = self._agreement(readings)
        synthetic_share = self._synthetic_share(panel)
        confidence = clamp(
            0.5 * min(abs(score) / 0.5, 1.0) + 0.3 * agreement + 0.2 * (1.0 - synthetic_share),
            0.0, 1.0,
        )

        state = CycleState(
            domain=domain,
            ts=composite.index[-1].to_pydatetime(),
            phase=current,
            score=score,
            momentum=momentum,
            readings=readings,
            transition_probability=transition,
            days_in_phase=days_in_phase,
            divergence=divergence,
            divergence_flag=divergence_flag,
            confidence=confidence,
            previous_phase=previous,
            history=pd.DataFrame({"cycle": composite, "phase": phases}),
            data_quality="simulated" if synthetic_share > 0.5 else
                         ("mixed" if synthetic_share > 0 else "live"),
        )
        state.narrative = self._narrate(state)
        return state

    def _readings(self, panel: F.FactorPanel, weights: dict[str, float],
                  results: dict[str, LeadLagResult]) -> list[FactorReading]:
        latest = panel.latest()
        readings: list[FactorReading] = []
        for name, score in latest.items():
            spec = F.FACTOR_SPECS.get(name)
            if spec is None:
                continue
            result = results.get(name)
            weight = weights.get(name, 0.0)
            readings.append(
                FactorReading(
                    name=name,
                    label=spec.label,
                    score=float(score),
                    weight=float(weight),
                    contribution=float(weight * score),
                    lead_days=result.best_lead if result and result.significant
                              else spec.typical_lead_days,
                    ic=result.best_ic if result else 0.0,
                    trusted=bool(result.significant) if result else False,
                    rationale=spec.rationale,
                )
            )
        return sorted(readings, key=lambda r: abs(r.contribution), reverse=True)

    @staticmethod
    def _agreement(readings: list[FactorReading]) -> float:
        """Weighted share of factors pointing the same way as the composite."""
        if not readings:
            return 0.0
        total = sum(abs(r.contribution) for r in readings)
        if total <= 0:
            return 0.0
        net = sum(r.contribution for r in readings)
        return float(clamp(abs(net) / total, 0.0, 1.0))

    @staticmethod
    def _synthetic_share(panel: F.FactorPanel) -> float:
        if not panel.sources:
            return 1.0
        synthetic = sum(1 for v in panel.sources.values() if v == "synthetic")
        return synthetic / len(panel.sources)

    def _narrate(self, state: CycleState) -> list[str]:
        """Plain-English explanation. The number is useless without the why."""
        lines = [
            f"{state.phase.label}: {state.phase.stance}.",
            f"Composite {state.score:+.2f} with {state.momentum:+.2f} momentum over "
            f"{self.cfg.impulse_window} days; {state.days_in_phase} days in this phase.",
        ]

        for reading in state.top_drivers(3):
            trust = "" if reading.trusted else " [prior weight — no significant lead measured]"
            lines.append(
                f"{reading.label}: {reading.score:+.2f} ({reading.direction}), "
                f"weight {reading.weight:.0%}, lead ~{reading.lead_days}d{trust}"
            )

        if state.divergence_flag:
            if state.divergence < 0:
                lines.append(
                    f"WARNING — bearish divergence ({state.divergence:+.2f} sd): price is rising "
                    "while macro internals deteriorate. This is the distribution signature."
                )
            else:
                lines.append(
                    f"NOTE — bullish divergence ({state.divergence:+.2f} sd): macro internals are "
                    "improving while price still falls. Historically an accumulation window."
                )

        if state.transition_probability > 0.5:
            lines.append(
                f"Phase is unstable: {state.transition_probability:.0%} of comparable historical "
                f"days had changed phase within {self.cfg.hazard_horizon} days."
            )

        if state.data_quality != "live":
            lines.append(
                f"Data quality: {state.data_quality}. Readings built partly on simulated data "
                "are structurally sound but say nothing about the real market."
            )
        return lines

    # -------------------------------------------------------------- validation
    def validate(self, domain: str = "crypto") -> ValidationReport:
        """Does the composite actually predict forward returns? Measure, don't assume."""
        panel = self.build_panel(domain)
        if panel.frame.empty:
            return ValidationReport(domain=domain, horizon=self.cfg.forward_horizon,
                                    composite_ic=0.0, composite_t=0.0, n=0,
                                    note="No factor data available.")

        weights, results = self.fit_weights(panel)
        composite = self.composite_series(panel, weights, results)

        returns = forward_return(F.to_daily(panel.anchor), 0, self.cfg.forward_horizon)
        ic, n = information_coefficient(composite, returns)

        frame = pd.concat([composite.rename("c"), returns.rename("r")], axis=1).dropna()
        hit_rate = float((np.sign(frame["c"]) == np.sign(frame["r"])).mean()) if len(frame) else 0.0

        note = ""
        if self._synthetic_share(panel) > 0.5:
            note = (
                "Most inputs are simulated, so this measures the fixture generator, not the "
                "market. Add API keys and re-run before drawing any conclusion."
            )

        return ValidationReport(
            domain=domain,
            horizon=self.cfg.forward_horizon,
            composite_ic=ic,
            composite_t=_t_stat(ic, n, self.cfg.forward_horizon),
            per_factor=list(results.values()),
            hit_rate=hit_rate,
            n=n,
            note=note,
        )

    def domain_for(self, asset_class: AssetClass) -> str:
        return "crypto" if asset_class is AssetClass.CRYPTO else "fx"


_ENGINE: CycleEngine | None = None


def get_cycle_engine(hub: DataHub | None = None, params: Params | None = None) -> CycleEngine:
    global _ENGINE
    if _ENGINE is None or hub is not None or params is not None:
        _ENGINE = CycleEngine(hub, params)
    return _ENGINE
