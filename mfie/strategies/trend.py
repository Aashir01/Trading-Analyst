"""Trend-following strategies.

The economic premise: capital reallocates slowly and information diffuses
gradually, so price drifts. That premise holds when liquidity is expanding —
which is precisely why these signals are the ones the GLI filter throttles.
"""

from __future__ import annotations

import numpy as np

from mfie.core.types import Direction, RawSignal
from mfie.core.utils import clamp
from mfie.strategies.base import Strategy, StrategyContext, build_signal, scale_strength


class TrendFollowingStrategy(Strategy):
    """Multi-factor trend confirmation: EMA stack + ADX + Supertrend + MACD.

    Requiring agreement across independent trend measures is what keeps this out
    of chop; any one of them alone whipsaws.
    """

    name = "trend_following"
    family = "momentum"
    description = "EMA stack, ADX strength, Supertrend direction and MACD momentum in agreement"
    min_bars = 220

    def generate(self, ctx: StrategyContext) -> RawSignal | None:
        ind = ctx.indicators
        price = ctx.price
        ema_fast = ind.last("ema_fast")
        ema_slow = ind.last("ema_slow")
        ema_trend = ind.last("ema_trend")
        adx = ind.last("adx", 0.0)
        st_dir = ind.last("supertrend_dir", 0.0)
        macd_hist = ind.last("macd_hist", 0.0)
        macd_hist_prev = ind.prev("macd_hist", 1, 0.0)
        plus_di = ind.last("plus_di", 0.0)
        minus_di = ind.last("minus_di", 0.0)

        if any(np.isnan(x) for x in (ema_fast, ema_slow, ema_trend)):
            return None

        adx_threshold = ctx.params.regime.adx_trending
        if adx < adx_threshold * 0.8:
            return None  # no trend worth following

        bull = (
            price > ema_fast > ema_slow
            and price > ema_trend
            and st_dir > 0
            and plus_di > minus_di
        )
        bear = (
            price < ema_fast < ema_slow
            and price < ema_trend
            and st_dir < 0
            and minus_di > plus_di
        )
        if not bull and not bear:
            return None

        direction = Direction.LONG if bull else Direction.SHORT

        # Conviction from trend strength, DI separation and momentum acceleration.
        adx_score = clamp((adx - adx_threshold) / 30.0, 0.0, 1.0)
        di_spread = abs(plus_di - minus_di) / max(plus_di + minus_di, 1e-9)
        momentum_accelerating = (
            macd_hist > macd_hist_prev if bull else macd_hist < macd_hist_prev
        )
        separation = abs(ema_fast - ema_slow) / max(price, 1e-9)

        strength = (
            0.35
            + 0.25 * adx_score
            + 0.20 * clamp(di_spread, 0.0, 1.0)
            + 0.10 * clamp(separation * 100, 0.0, 1.0)
            + (0.10 if momentum_accelerating else 0.0)
        )
        strength *= ctx.regime.strategy_weight(self.family)

        rationale = [
            f"ADX {adx:.1f} above trend threshold {adx_threshold:.0f}",
            f"EMA stack {'bullish' if bull else 'bearish'} "
            f"({ind.last('ema_fast'):.4f} vs {ind.last('ema_slow'):.4f})",
            f"Supertrend {'up' if st_dir > 0 else 'down'}",
            f"MACD histogram {'expanding' if momentum_accelerating else 'contracting'}",
            f"Regime {ctx.regime.regime.value} (weight {ctx.regime.strategy_weight(self.family):.2f})",
        ]

        return build_signal(
            ctx,
            direction,
            self.name,
            scale_strength(strength),
            rationale,
            features={
                "adx": adx,
                "di_spread": di_spread,
                "macd_hist": macd_hist,
                "ema_separation": separation,
                "hurst": ctx.regime.hurst,
            },
        )


class MACrossStrategy(Strategy):
    """Classic fast/slow moving-average cross, gated by the long-term trend filter.

    Kept because it is the transparent baseline every other momentum strategy
    should be measured against.
    """

    name = "ma_cross"
    family = "momentum"
    description = "Fast/slow EMA crossover filtered by the 200-period trend EMA"
    min_bars = 220

    def generate(self, ctx: StrategyContext) -> RawSignal | None:
        ind = ctx.indicators
        fast_now, fast_prev = ind.last("ema_fast"), ind.prev("ema_fast", 1)
        slow_now, slow_prev = ind.last("ema_slow"), ind.prev("ema_slow", 1)
        trend = ind.last("ema_trend")
        price = ctx.price

        if any(np.isnan(x) for x in (fast_now, fast_prev, slow_now, slow_prev, trend)):
            return None

        crossed_up = fast_prev <= slow_prev and fast_now > slow_now
        crossed_down = fast_prev >= slow_prev and fast_now < slow_now
        if not crossed_up and not crossed_down:
            return None

        # The long-term EMA is the regime filter: no counter-trend crosses.
        if crossed_up and price < trend:
            return None
        if crossed_down and price > trend:
            return None

        direction = Direction.LONG if crossed_up else Direction.SHORT
        volume_z = ind.last("volume_z", 0.0)
        rsi = ind.last("rsi", 50.0)

        strength = 0.45
        if not np.isnan(volume_z) and volume_z > 1.0:
            strength += 0.15  # cross confirmed by participation
        if (direction is Direction.LONG and 45 < rsi < 70) or (
            direction is Direction.SHORT and 30 < rsi < 55
        ):
            strength += 0.10  # not already exhausted
        strength *= ctx.regime.strategy_weight(self.family)

        return build_signal(
            ctx,
            direction,
            self.name,
            scale_strength(strength),
            [
                f"EMA{ctx.params.technical.ema_fast}/EMA{ctx.params.technical.ema_slow} crossed "
                f"{'up' if crossed_up else 'down'}",
                f"Price {'above' if price > trend else 'below'} EMA{ctx.params.technical.ema_trend}",
                f"Volume z-score {volume_z:.2f}",
            ],
            features={"volume_z": volume_z, "rsi": rsi},
        )
