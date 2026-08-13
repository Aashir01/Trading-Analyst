r"""What the trade costs, measured in the only unit that matters.

Every layer above this one reasons in **R** — multiples of the stop distance.
A reward:risk of 2.0 means "win two stops or lose one". Costs must be quoted
in the same unit or they cannot be compared to the payoff, and a cost quoted
in basis points cannot be:

.. math::

    c_R = \frac{\text{spread} + 2\,\text{slippage} + \text{commission}
                 + \text{carry}}{|\,\text{entry} - \text{stop}\,|}

The denominator is what makes this interesting. A 6 bps round trip against a
3% stop is 0.02R — noise. The same 6 bps against a 0.15% scalping stop is
0.40R, which moves the breakeven hit rate on a 2R target from 33% to 47%.
Nothing else in the engine can see that difference, because nothing else
divides by the stop.

**Carry is signed.** Funding on a perpetual and the interest differential on a
currency pair are charged *or paid* depending on which side you are on. A
short perp in a positive-funding market is collecting; treating that as a cost
throws away the oldest real edge in either market. So ``carry`` may be
negative, and when it is, it widens the trade's edge rather than narrowing it.

**Holding time comes from diffusion, not from a guess.** Under a random walk
with per-bar volatility :math:`\sigma`, the expected time to travel a distance
:math:`D` scales as :math:`(D/\sigma)^2`. A stop two ATRs away is therefore
roughly a four-bar trade, and a stop ten ATRs away a hundred-bar one. That is
the estimate used to accrue funding — crude, but it has the right shape, which
a flat "assume 24 hours" does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mfie.config import CostParams, get_params
from mfie.core.types import AssetClass, Direction, Instrument, Quote, RawSignal
from mfie.core.utils import clamp, safe_div

# Bars per day, used to turn a holding time in bars into one in days for the
# carry accrual. Mirrors mfie.core.utils.annualization_factor / 365.
_BARS_PER_DAY = {
    "1m": 1440.0, "5m": 288.0, "15m": 96.0, "30m": 48.0,
    "1h": 24.0, "2h": 12.0, "4h": 6.0, "12h": 2.0,
    "1d": 1.0, "1w": 1 / 7,
}


def bars_per_day(timeframe: str) -> float:
    return float(_BARS_PER_DAY.get(timeframe, 24.0))


def _scalar(value: object) -> float | None:
    """Coerce a rate to a number, tolerating a series or a missing reading.

    Providers serve funding and yields as histories. A cost model that assumes
    a float and gets a Series should take the latest reading, not take down
    the run.
    """
    if value is None:
        return None
    try:
        import pandas as pd

        if isinstance(value, pd.Series):
            clean = value.dropna()
            return float(clean.iloc[-1]) if not clean.empty else None
        number = float(value)
    except (TypeError, ValueError, IndexError):
        return None
    import math

    return number if math.isfinite(number) else None


@dataclass
class TradeCosts:
    """A round-trip cost estimate, in price units and in R."""

    spread: float = 0.0          # full spread, crossed once in and once out
    slippage: float = 0.0        # both sides
    commission: float = 0.0      # both sides
    carry: float = 0.0           # signed: negative means the trade is paid
    stop_distance: float = 0.0
    holding_bars: float = 0.0
    holding_days: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def friction(self) -> float:
        """Execution cost only — the part that is never a credit."""
        return self.spread + self.slippage + self.commission

    @property
    def total(self) -> float:
        """Net cost in price units. May be negative on a well-paid carry."""
        return self.friction + self.carry

    @property
    def total_r(self) -> float:
        """Net cost as a fraction of the stop distance."""
        return safe_div(self.total, self.stop_distance, 0.0)

    @property
    def friction_r(self) -> float:
        return safe_div(self.friction, self.stop_distance, 0.0)

    @property
    def carry_r(self) -> float:
        return safe_div(self.carry, self.stop_distance, 0.0)

    def breakdown(self) -> dict[str, float]:
        return {
            "spread_r": safe_div(self.spread, self.stop_distance, 0.0),
            "slippage_r": safe_div(self.slippage, self.stop_distance, 0.0),
            "commission_r": safe_div(self.commission, self.stop_distance, 0.0),
            "carry_r": self.carry_r,
            "total_r": self.total_r,
            "holding_bars": self.holding_bars,
        }

    def describe(self) -> str:
        sign = "paid" if self.carry < 0 else "charged"
        return (
            f"round trip {self.total_r:.3f}R "
            f"(friction {self.friction_r:.3f}R, carry {abs(self.carry_r):.3f}R {sign}, "
            f"~{self.holding_bars:.0f} bars)"
        )


def expected_holding_bars(
    stop_distance: float,
    atr: float,
    params: CostParams | None = None,
) -> float:
    r"""Diffusion estimate of how long the trade is open.

    :math:`E[\tau] \approx (D/\sigma)^2` for a driftless walk hitting a barrier
    at distance :math:`D` with per-bar step :math:`\sigma`. ATR stands in for
    :math:`\sigma`. Clamped, because the estimate is quadratic and a bad ATR
    reading would otherwise imply a holding period of years.
    """
    p = params or get_params().costs
    if atr <= 0 or stop_distance <= 0:
        return 1.0
    bars = (stop_distance / atr) ** 2
    return float(clamp(bars, 1.0, p.max_holding_bars))


def estimate_costs(
    signal: RawSignal,
    *,
    quote: Quote | None = None,
    atr: float = 0.0,
    timeframe: str = "1h",
    funding_rate: float | None = None,
    rate_differential: float | None = None,
    params: CostParams | None = None,
    stop_distance: float | None = None,
) -> TradeCosts:
    """Price a round trip for one signal.

    ``funding_rate`` is the per-interval perpetual funding rate (crypto);
    ``rate_differential`` is the annualised base-minus-quote policy rate spread
    (forex). Both are signed from the *market's* point of view — this function
    applies the trade's direction.
    """
    p = params or get_params().costs
    instrument: Instrument = signal.instrument
    entry = float(signal.entry)
    distance = float(stop_distance if stop_distance is not None else signal.stop_distance)
    notes: list[str] = []

    if entry <= 0 or distance <= 0:
        return TradeCosts(stop_distance=distance, notes=["degenerate entry/stop — costs not priced"])

    # --- execution friction ------------------------------------------------
    if quote is not None and quote.mid > 0 and quote.spread > 0:
        spread = float(quote.spread)
    else:
        spread = entry * p.min_spread_bps / 10_000.0
        notes.append("no top-of-book — spread floored at the configured minimum")

    slippage = 2.0 * p.slippage_atr_fraction * float(atr)
    commission = 2.0 * entry * p.commission_bps_per_side / 10_000.0

    # --- holding time ------------------------------------------------------
    holding_bars = expected_holding_bars(distance, float(atr), p)
    holding_days = holding_bars / bars_per_day(timeframe)

    # --- carry -------------------------------------------------------------
    carry = 0.0
    funding = _scalar(funding_rate)
    differential = _scalar(rate_differential)
    if p.apply_carry and signal.direction is not Direction.FLAT:
        sign = signal.direction.sign
        if instrument.asset_class is AssetClass.CRYPTO and funding is not None:
            intervals = holding_days * 24.0 / max(p.funding_interval_hours, 1e-9)
            # Longs pay positive funding; shorts receive it.
            carry = sign * funding * intervals * entry
            if carry < 0:
                notes.append(f"funding credit over ~{intervals:.1f} intervals")
        elif instrument.is_forex and differential is not None:
            # Long the pair = long base, short quote: you earn the differential.
            carry = -sign * differential * (holding_days / 365.0) * entry
            if carry < 0:
                notes.append(f"positive carry over ~{holding_days:.1f} days")

    return TradeCosts(
        spread=float(spread),
        slippage=float(slippage),
        commission=float(commission),
        carry=float(carry),
        stop_distance=distance,
        holding_bars=float(holding_bars),
        holding_days=float(holding_days),
        notes=notes,
    )


def breakeven_hit_rate(reward_risk: float, cost_r: float) -> float:
    r"""The hit rate at which the trade stops losing money.

    Winning pays :math:`b - c`, losing costs :math:`1 + c`. Setting expected
    value to zero:

    .. math::  p^* = \frac{1 + c}{b + 1}

    Note what the cost does *not* do: it does not appear in the denominator.
    Costs raise the bar linearly, and they do it hardest on the low-reward,
    tight-stop trades that look most attractive on a chart.
    """
    b = max(float(reward_risk), 1e-9)
    return float(clamp((1.0 + cost_r) / (b + 1.0), 0.0, 1.0))
