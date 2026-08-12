"""Mean-reversion strategies.

The economic premise is inventory risk: market makers who absorb a one-sided
flow demand compensation, and price overshoots, then reverts once inventory is
cleared. That works in range-bound, low-volatility tape and gets run over in a
trend — hence the regime weighting and the explicit trend veto.
"""

from __future__ import annotations

import numpy as np

from mfie.core.types import Direction, RawSignal
from mfie.core.utils import clamp
from mfie.strategies.base import Strategy, StrategyContext, build_signal, scale_strength
from mfie.technical.indicators import rsi_divergence


class BollingerReversionStrategy(Strategy):
    """Fade a close outside the Bollinger band when the trend is flat.

    Refuses to fire when ADX says a trend is running — a band touch in a strong
    trend is continuation, not exhaustion.
    """

    name = "bollinger_reversion"
    family = "mean_reversion"
    description = "Fade statistical extremes at the Bollinger bands in a non-trending tape"
    min_bars = 120

    def generate(self, ctx: StrategyContext) -> RawSignal | None:
        ind = ctx.indicators
        price = ctx.price
        upper, middle, lower = ind.last("bb_upper"), ind.last("bb_middle"), ind.last("bb_lower")
        percent_b = ind.last("bb_percent_b")
        z = ind.last("zscore", 0.0)
        adx = ind.last("adx", 0.0)
        rsi = ind.last("rsi", 50.0)

        if any(np.isnan(x) for x in (upper, middle, lower, percent_b)):
            return None

        # Hard veto: strong directional trend invalidates the reversion premise.
        if adx > ctx.params.regime.adx_trending * 1.2:
            return None

        if price <= lower and rsi < 40:
            direction = Direction.LONG
        elif price >= upper and rsi > 60:
            direction = Direction.SHORT
        else:
            return None

        distance = abs(z)
        if distance < 1.5:
            return None

        strength = 0.35 + 0.25 * clamp((distance - 1.5) / 1.5, 0.0, 1.0)
        if (direction is Direction.LONG and rsi < 30) or (direction is Direction.SHORT and rsi > 70):
            strength += 0.15
        # Bandwidth expanding while price sits outside the band suggests a
        # genuine breakout rather than an overshoot — reduce conviction.
        bandwidth = ind.last("bb_bandwidth", 0.0)
        bandwidth_prev = ind.prev("bb_bandwidth", 5, bandwidth)
        if bandwidth > bandwidth_prev * 1.3:
            strength *= 0.7
        strength *= ctx.regime.strategy_weight(self.family)

        # Target the mean, not a fixed R multiple: that is the actual thesis.
        stop_distance = ctx.atr * ctx.params.risk.atr_stop_multiple
        stop = price - stop_distance if direction is Direction.LONG else price + stop_distance

        signal = build_signal(
            ctx,
            direction,
            self.name,
            scale_strength(strength),
            [
                f"Close {'below lower' if direction is Direction.LONG else 'above upper'} "
                f"Bollinger band (z={z:.2f})",
                f"RSI {rsi:.1f}",
                f"ADX {adx:.1f} — no dominant trend",
                f"Target: mean reversion to {middle:.5f}",
            ],
            features={"zscore": z, "rsi": rsi, "percent_b": percent_b, "adx": adx},
            stop=stop,
        )
        signal.take_profit = float(middle)
        return signal


class RSIReversionStrategy(Strategy):
    """RSI extreme plus divergence — the higher-quality reversion setup.

    A bare RSI level is close to worthless; requiring price/RSI divergence at the
    extreme is what turns it into an edge.
    """

    name = "rsi_reversion"
    family = "mean_reversion"
    description = "RSI oversold/overbought confirmed by price-momentum divergence"
    min_bars = 120

    def generate(self, ctx: StrategyContext) -> RawSignal | None:
        ind = ctx.indicators
        rsi = ind.last("rsi", 50.0)
        adx = ind.last("adx", 0.0)
        if np.isnan(rsi):
            return None
        if adx > ctx.params.regime.adx_trending * 1.3:
            return None

        bullish_div, bearish_div = rsi_divergence(ind.close, ind.series("rsi"))

        if rsi <= 30 or (rsi < 40 and bullish_div):
            direction = Direction.LONG
            divergence = bullish_div
        elif rsi >= 70 or (rsi > 60 and bearish_div):
            direction = Direction.SHORT
            divergence = bearish_div
        else:
            return None

        extremity = abs(rsi - 50.0) / 50.0
        strength = 0.30 + 0.30 * clamp(extremity, 0.0, 1.0) + (0.20 if divergence else 0.0)

        mfi = ind.last("mfi", 50.0)
        if not np.isnan(mfi):
            # Money-flow agreeing with the RSI extreme adds independent evidence.
            if (direction is Direction.LONG and mfi < 30) or (direction is Direction.SHORT and mfi > 70):
                strength += 0.10
        strength *= ctx.regime.strategy_weight(self.family)

        return build_signal(
            ctx,
            direction,
            self.name,
            scale_strength(strength),
            [
                f"RSI {rsi:.1f}",
                f"{'Bullish' if direction is Direction.LONG else 'Bearish'} divergence: {divergence}",
                f"Money Flow Index {mfi:.1f}",
                f"ADX {adx:.1f}",
            ],
            features={"rsi": rsi, "mfi": mfi, "divergence": float(divergence), "adx": adx},
            reward_risk=1.5,  # reversion targets are closer than trend targets
        )
