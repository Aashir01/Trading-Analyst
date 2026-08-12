r"""Grid / market-making strategy.

A grid bot is a crude market maker: it lays buy limits below and sell limits
above, harvesting the bid-ask oscillation of a range-bound market. Its P&L is
short volatility and short trend — it prints steadily and then gives it all back
in one directional move.

Two things make the version here different from the retail default:

* It refuses to run outside a ranging regime. Most grid blow-ups are a grid left
  running into a trend.
* Level spacing is volatility-scaled (ATR), and the whole grid carries an
  invalidation level, so the strategy has a defined maximum loss instead of the
  usual "hope it comes back".

Geometric spacing is used for crypto (returns are multiplicative) and arithmetic
for FX (ranges are additive at these scales).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mfie.core.types import Direction, RawSignal
from mfie.core.utils import clamp
from mfie.strategies.base import Strategy, StrategyContext, build_signal, scale_strength


@dataclass
class GridPlan:
    lower: float
    upper: float
    levels: int
    spacing: float
    mode: str                 # geometric | arithmetic
    buy_levels: list[float]
    sell_levels: list[float]
    invalidation_low: float
    invalidation_high: float
    expected_capture: float   # profit per round trip, as a fraction

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "levels": self.levels,
            "spacing": self.spacing,
            "mode": self.mode,
            "invalidation_low": self.invalidation_low,
            "invalidation_high": self.invalidation_high,
            "expected_capture": self.expected_capture,
        }


def build_grid(
    price: float,
    lower: float,
    upper: float,
    levels: int = 10,
    mode: str = "geometric",
) -> GridPlan:
    levels = max(int(levels), 2)
    if mode == "geometric" and lower > 0 and upper > 0:
        grid = np.geomspace(lower, upper, levels)
        spacing = float((upper / lower) ** (1.0 / (levels - 1)) - 1.0)
    else:
        mode = "arithmetic"
        grid = np.linspace(lower, upper, levels)
        spacing = float((upper - lower) / (levels - 1) / max(price, 1e-9))

    buys = [float(x) for x in grid if x < price]
    sells = [float(x) for x in grid if x > price]

    return GridPlan(
        lower=float(lower),
        upper=float(upper),
        levels=levels,
        spacing=spacing,
        mode=mode,
        buy_levels=sorted(buys, reverse=True),
        sell_levels=sorted(sells),
        # Invalidation: half a range beyond each boundary. Beyond that the
        # range thesis is broken and the grid must be shut down.
        invalidation_low=float(lower - (upper - lower) * 0.5),
        invalidation_high=float(upper + (upper - lower) * 0.5),
        expected_capture=float(spacing),
    )


class GridStrategy(Strategy):
    """Propose a grid when the market is genuinely ranging."""

    name = "grid"
    family = "market_making"
    description = "Volatility-scaled grid of limit orders for range-bound markets"
    min_bars = 150

    levels = 10
    lookback = 100

    def generate(self, ctx: StrategyContext) -> RawSignal | None:
        if not ctx.regime.regime.favours_mean_reversion:
            return None

        ind = ctx.indicators
        adx = ind.last("adx", 100.0)
        if adx > ctx.params.regime.adx_trending:
            return None

        window = ctx.df.tail(self.lookback)
        if len(window) < self.lookback // 2:
            return None

        # Trim the extremes so one wick does not define the range.
        lower = float(np.quantile(window["low"], 0.05))
        upper = float(np.quantile(window["high"], 0.95))
        price = ctx.price
        if not (lower < price < upper) or upper <= lower:
            return None

        mode = "geometric" if ctx.instrument.is_crypto else "arithmetic"
        plan = build_grid(price, lower, upper, self.levels, mode)

        # Each captured level must clear the spread with room to spare.
        spread_cost = ctx.extras.get("spread_pct", 0.0005)
        if plan.expected_capture < spread_cost * 3:
            return None

        range_width = (upper - lower) / price
        position_in_range = (price - lower) / (upper - lower)
        # Bias the initial fill toward whichever side has more room to work.
        direction = Direction.LONG if position_in_range < 0.5 else Direction.SHORT

        strength = (
            0.30
            + 0.25 * clamp(1.0 - adx / ctx.params.regime.adx_trending, 0.0, 1.0)
            + 0.20 * clamp(plan.expected_capture / (spread_cost * 10), 0.0, 1.0)
            + 0.15 * clamp(1.0 - abs(position_in_range - 0.5) * 2, 0.0, 1.0)
        )
        strength *= ctx.regime.strategy_weight(self.family)

        stop = plan.invalidation_low if direction is Direction.LONG else plan.invalidation_high

        signal = build_signal(
            ctx,
            direction,
            self.name,
            scale_strength(strength),
            [
                f"Range {lower:.5f} - {upper:.5f} ({range_width:.2%} wide), price at "
                f"{position_in_range:.0%} of range",
                f"{plan.levels} {plan.mode} levels, {plan.spacing:.3%} apart",
                f"Expected capture {plan.expected_capture:.3%} vs spread cost {spread_cost:.3%}",
                f"ADX {adx:.1f} confirms range-bound conditions",
                f"Grid invalidated outside {plan.invalidation_low:.5f} - {plan.invalidation_high:.5f}",
            ],
            features={
                "range_width": range_width,
                "position_in_range": position_in_range,
                "grid_spacing": plan.spacing,
                "adx": adx,
                "levels": float(plan.levels),
            },
            stop=stop,
        )
        signal.take_profit = float(upper if direction is Direction.LONG else lower)
        # The full plan travels with the signal so the UI can render the ladder.
        signal.features.update({f"grid_{k}": v for k, v in plan.as_dict().items()
                                if isinstance(v, (int, float))})
        return signal
