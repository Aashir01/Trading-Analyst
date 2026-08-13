"""Position sizing — from confidence to units.

The chain is:

1. **Confidence** = raw strategy strength x product of all filter multipliers.
2. **Risk fraction** = base risk per trade x confidence, then scaled by the
   portfolio CVaR scaler and a volatility-target scalar, capped by
   ``max_risk_per_trade`` and by a Kelly ceiling.
3. **Units** = (equity x risk fraction) / stop distance.

Because risk is defined by the *stop distance*, widening the stop automatically
reduces the size. That invariant is what keeps the microstructure filter honest:
it can widen a stop for slippage without secretly increasing risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mfie.config import Params, get_params
from mfie.core.types import Direction, FilterOutcome, RawSignal, ScoredSignal
from mfie.core.utils import clamp
from mfie.econ.risk import kelly_fraction, position_size, volatility_target_scalar
from mfie.portfolio.calibration import prior_hit_rate


@dataclass
class SizingDecision:
    confidence: float
    risk_fraction: float
    size_fraction: float
    units: float
    notional: float
    adjusted_stop: float
    adjusted_take_profit: float | None
    kelly_cap: float
    vol_scalar: float
    notes: list[str] = field(default_factory=list)


def combine_multipliers(outcomes: list[FilterOutcome]) -> float:
    """Product of filter multipliers.

    A product, not an average: three independent 0.8x headwinds should compound
    to 0.51x, because each is a separate reason to doubt the trade.
    """
    multiplier = 1.0
    for outcome in outcomes:
        multiplier *= max(outcome.multiplier, 0.0)
    return multiplier


def combined_stop_multiplier(outcomes: list[FilterOutcome]) -> float:
    """Largest stop widening requested by any filter (they do not compound)."""
    return max([o.stop_multiplier for o in outcomes] + [1.0])


def size_signal(
    signal: RawSignal,
    outcomes: list[FilterOutcome],
    params: Params | None = None,
    equity: float | None = None,
    risk_scaler: float = 1.0,
    realized_vol: float | None = None,
    open_risk: float = 0.0,
    win_rate: float | None = None,
) -> SizingDecision:
    p = params or get_params()
    risk_params = p.risk
    equity = equity if equity is not None else risk_params.account_equity
    notes: list[str] = []

    # --- 1. confidence -----------------------------------------------------
    multiplier = combine_multipliers(outcomes)
    confidence = clamp(signal.strength * multiplier, 0.0, 1.0)

    # --- 2. stop adjustment ------------------------------------------------
    stop_multiplier = combined_stop_multiplier(outcomes)
    base_distance = signal.stop_distance
    adjusted_distance = base_distance * stop_multiplier
    if signal.direction is Direction.LONG:
        adjusted_stop = signal.entry - adjusted_distance
    elif signal.direction is Direction.SHORT:
        adjusted_stop = signal.entry + adjusted_distance
    else:
        adjusted_stop = signal.stop

    if stop_multiplier > 1.001:
        notes.append(f"Stop widened {stop_multiplier:.2f}x for execution friction")

    # Keep the original reward:risk by moving the target with the stop.
    if signal.take_profit is not None and base_distance > 0:
        rr = abs(signal.take_profit - signal.entry) / base_distance
        adjusted_take_profit = (
            signal.entry + adjusted_distance * rr
            if signal.direction is Direction.LONG
            else signal.entry - adjusted_distance * rr
        )
    else:
        adjusted_take_profit = signal.take_profit

    # --- 3. risk fraction --------------------------------------------------
    if confidence < risk_params.min_confidence_to_trade:
        notes.append(
            f"Confidence {confidence:.0%} below minimum "
            f"{risk_params.min_confidence_to_trade:.0%} — no position"
        )
        return SizingDecision(
            confidence=confidence, risk_fraction=0.0, size_fraction=0.0, units=0.0,
            notional=0.0, adjusted_stop=adjusted_stop,
            adjusted_take_profit=adjusted_take_profit, kelly_cap=0.0, vol_scalar=1.0,
            notes=notes,
        )

    risk_fraction = risk_params.base_risk_per_trade * confidence
    risk_fraction *= clamp(risk_scaler, 0.0, 1.0)
    if risk_scaler < 0.999:
        notes.append(f"Portfolio risk scaler {risk_scaler:.2f}x applied")

    vol_scalar = volatility_target_scalar(realized_vol or 0.0, risk_params.target_annual_vol)
    risk_fraction *= vol_scalar
    if abs(vol_scalar - 1.0) > 0.01 and realized_vol:
        notes.append(
            f"Volatility scalar {vol_scalar:.2f}x (realised {realized_vol:.1%} vs target "
            f"{risk_params.target_annual_vol:.1%})"
        )

    # Kelly ceiling from the signal's own reward:risk and an assumed hit rate.
    rr = signal.reward_risk or risk_params.reward_risk_target
    # The fallback is the *prior* from mfie.portfolio.calibration — the same
    # curve, defined in one place. When realised trades exist, the portfolio
    # layer re-applies this cap using the calibrated posterior instead, which
    # is almost always the tighter of the two.
    assumed_win_rate = (
        win_rate if win_rate is not None else prior_hit_rate(confidence, p.calibration)
    )
    kelly_cap = kelly_fraction(assumed_win_rate, rr, risk_params.kelly_fraction_cap)
    if kelly_cap <= 0:
        notes.append(
            f"Kelly fraction non-positive at {assumed_win_rate:.0%} win rate and {rr:.1f}R — "
            "no edge, no position"
        )
        return SizingDecision(
            confidence=confidence, risk_fraction=0.0, size_fraction=0.0, units=0.0,
            notional=0.0, adjusted_stop=adjusted_stop,
            adjusted_take_profit=adjusted_take_profit, kelly_cap=0.0, vol_scalar=vol_scalar,
            notes=notes,
        )

    risk_fraction = min(risk_fraction, risk_params.max_risk_per_trade)

    # Portfolio heat cap: never let total open risk exceed the book limit.
    headroom = max(risk_params.max_portfolio_risk - open_risk, 0.0)
    if risk_fraction > headroom:
        notes.append(
            f"Capped by portfolio heat: {open_risk:.1%} already at risk of "
            f"{risk_params.max_portfolio_risk:.1%} limit"
        )
        risk_fraction = headroom

    # --- 4. units ----------------------------------------------------------
    units, notional = position_size(equity, risk_fraction, signal.entry, adjusted_stop)
    size_fraction = notional / equity if equity > 0 else 0.0

    return SizingDecision(
        confidence=confidence,
        risk_fraction=float(risk_fraction),
        size_fraction=float(size_fraction),
        units=float(units),
        notional=float(notional),
        adjusted_stop=float(adjusted_stop),
        adjusted_take_profit=float(adjusted_take_profit) if adjusted_take_profit else None,
        kelly_cap=float(kelly_cap),
        vol_scalar=float(vol_scalar),
        notes=notes,
    )


def apply_sizing(signal: ScoredSignal, decision: SizingDecision) -> ScoredSignal:
    signal.confidence = decision.confidence
    signal.risk_fraction = decision.risk_fraction
    signal.size_fraction = decision.size_fraction
    signal.units = decision.units
    signal.adjusted_stop = decision.adjusted_stop
    signal.adjusted_take_profit = decision.adjusted_take_profit
    return signal
