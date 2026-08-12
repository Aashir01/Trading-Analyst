r"""Funding-rate / cash-and-carry basis strategy (crypto only).

Perpetual swaps have no expiry, so exchanges use a funding payment to tether the
perp to spot: when the perp trades above spot, longs pay shorts. That payment is
harvestable with no directional exposure:

    long spot  +  short perpetual  =>  collect funding, delta ≈ 0

Annualised carry from an 8-hourly funding rate :math:`f`:

.. math::  APR = f \times 3 \times 365

What this analysis mode emits is a **carry signal**, not a directional one: the
``direction`` field describes the perp leg, and ``features['hedged']`` marks it
as market neutral so the sizing layer does not apply directional macro filters
to it in the usual way.

Risks the number does not show: funding flips sign without warning, the two legs
can be liquidated independently, and exchange/counterparty risk is real.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mfie.core.types import Direction, RawSignal
from mfie.core.utils import clamp
from mfie.strategies.base import Strategy, StrategyContext, build_signal, scale_strength

FUNDING_INTERVALS_PER_DAY = 3  # most venues settle every 8 hours


def funding_apr(funding_rate: float, intervals_per_day: int = FUNDING_INTERVALS_PER_DAY) -> float:
    """Annualised percentage return from a per-interval funding rate."""
    return float(funding_rate * intervals_per_day * 365.0)


def basis_apr(spot: float, perp: float, days_to_expiry: float | None = None) -> float:
    """Annualised basis. For dated futures pass ``days_to_expiry``."""
    if spot <= 0:
        return 0.0
    raw = (perp - spot) / spot
    if days_to_expiry and days_to_expiry > 0:
        return float(raw * 365.0 / days_to_expiry)
    return float(raw)


class FundingCarryStrategy(Strategy):
    """Harvest persistently positive (or negative) perpetual funding.

    Fires only when funding has been consistently one-sided — a single elevated
    print is noise, and the trade only pays if the regime persists long enough
    to cover two sets of transaction costs.
    """

    name = "funding_carry"
    family = "carry"
    description = "Delta-neutral cash-and-carry harvesting perpetual funding payments"
    asset_classes = ("crypto",)
    min_bars = 60

    min_apr = 0.08          # 8% annualised before it is worth the execution risk
    consistency_window = 21  # ~7 days of 8h funding prints

    def supports(self, ctx: StrategyContext) -> bool:
        if not super().supports(ctx):
            return False
        return isinstance(ctx.extras.get("funding_rate"), pd.Series)

    def generate(self, ctx: StrategyContext) -> RawSignal | None:
        funding: pd.Series = pd.Series(ctx.extras["funding_rate"]).dropna()
        if len(funding) < self.consistency_window:
            return None

        recent = funding.tail(self.consistency_window)
        mean_rate = float(recent.mean())
        apr = funding_apr(mean_rate)

        if abs(apr) < self.min_apr:
            return None

        # Consistency: the share of prints with the same sign as the mean.
        same_sign = float((np.sign(recent) == np.sign(mean_rate)).mean())
        if same_sign < 0.7:
            return None

        # Positive funding: longs pay. Short the perp, hold spot.
        direction = Direction.SHORT if mean_rate > 0 else Direction.LONG

        volatility = float(recent.std(ddof=1))
        stability = clamp(1.0 - (volatility / max(abs(mean_rate), 1e-9)) / 3.0, 0.0, 1.0)

        strength = (
            0.35
            + 0.25 * clamp(abs(apr) / 0.30, 0.0, 1.0)
            + 0.20 * clamp((same_sign - 0.7) / 0.3, 0.0, 1.0)
            + 0.15 * stability
        )
        strength *= ctx.regime.strategy_weight(self.family)

        return build_signal(
            ctx,
            direction,
            self.name,
            scale_strength(strength),
            [
                f"Mean funding {mean_rate * 100:+.4f}% per 8h over {self.consistency_window} prints",
                f"Annualised carry {apr:+.2%}",
                f"Sign consistency {same_sign:.0%}",
                f"Delta-neutral: {'short perp + long spot' if mean_rate > 0 else 'long perp + short spot'}",
                "Directional macro filters are advisory only for this hedged structure",
            ],
            features={
                "funding_mean": mean_rate,
                "funding_apr": apr,
                "consistency": same_sign,
                "hedged": 1.0,
            },
            stop_atr_multiple=4.0,  # wide: the hedge, not the stop, controls risk
            reward_risk=1.0,
        )
