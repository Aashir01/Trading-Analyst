"""Breakout strategies.

Premise: liquidity clusters at obvious levels (prior range highs/lows), and when
those levels break, the resting orders behind them cascade. Volume confirmation
is mandatory — an unconfirmed break is usually a liquidity grab that reverses.
"""

from __future__ import annotations

import numpy as np

from mfie.core.types import Direction, RawSignal
from mfie.core.utils import clamp
from mfie.strategies.base import Strategy, StrategyContext, build_signal, scale_strength


class DonchianBreakoutStrategy(Strategy):
    """Turtle-style N-period channel break with volume and volatility confirmation."""

    name = "donchian_breakout"
    family = "breakout"
    description = "Close beyond the prior N-period high/low, confirmed by volume expansion"
    min_bars = 120

    def generate(self, ctx: StrategyContext) -> RawSignal | None:
        ind = ctx.indicators
        price = ctx.price
        upper = ind.last("donchian_upper")
        lower = ind.last("donchian_lower")
        volume_z = ind.last("volume_z", 0.0)
        atr_pct = ind.last("atr_pct", 0.0)

        if any(np.isnan(x) for x in (upper, lower)):
            return None

        if price > upper:
            direction = Direction.LONG
            level = upper
        elif price < lower:
            direction = Direction.SHORT
            level = lower
        else:
            return None

        # Unconfirmed breaks are the classic false-breakout trap.
        if np.isnan(volume_z) or volume_z < 0.5:
            return None

        penetration = abs(price - level) / max(ctx.atr, 1e-9)
        strength = (
            0.35
            + 0.20 * clamp(volume_z / 3.0, 0.0, 1.0)
            + 0.20 * clamp(penetration, 0.0, 1.0)
        )

        adx = ind.last("adx", 0.0)
        if adx > ctx.params.regime.adx_trending:
            strength += 0.10
        strength *= ctx.regime.strategy_weight(self.family)

        # Stop goes just inside the broken level — if price re-enters the range,
        # the breakout thesis is dead and there is no reason to stay.
        buffer = ctx.atr * 0.5
        stop = level - buffer if direction is Direction.LONG else level + buffer

        return build_signal(
            ctx,
            direction,
            self.name,
            scale_strength(strength),
            [
                f"Close broke {ctx.params.technical.donchian_period}-period "
                f"{'high' if direction is Direction.LONG else 'low'} at {level:.5f}",
                f"Volume z-score {volume_z:.2f} confirms participation",
                f"Penetration {penetration:.2f} ATR",
                f"ADX {adx:.1f}",
            ],
            features={
                "volume_z": volume_z,
                "penetration_atr": penetration,
                "atr_pct": atr_pct,
                "adx": adx,
            },
            stop=stop,
        )


class SqueezeBreakoutStrategy(Strategy):
    """TTM-style squeeze release: Bollinger Bands inside Keltner Channels, then expansion.

    Volatility is mean-reverting even when price is not, so a compressed range
    is a genuine forecast of an imminent expansion — the squeeze only tells you
    *when*, so direction comes from momentum at the moment of release.
    """

    name = "squeeze_breakout"
    family = "breakout"
    description = "Volatility compression (BB inside KC) releasing in the direction of momentum"
    min_bars = 120

    def generate(self, ctx: StrategyContext) -> RawSignal | None:
        ind = ctx.indicators
        squeeze = ind.series("squeeze")
        if squeeze.empty or len(squeeze) < 12:
            return None

        was_squeezed = bool(squeeze.iloc[-6:-1].any())
        released_now = not bool(squeeze.iloc[-1])
        if not (was_squeezed and released_now):
            return None

        macd_hist = ind.last("macd_hist", 0.0)
        momentum = ind.last("roc", 0.0)
        if np.isnan(macd_hist) or abs(macd_hist) < 1e-12:
            return None

        direction = Direction.LONG if macd_hist > 0 else Direction.SHORT
        # Momentum and MACD must agree, otherwise the release direction is noise.
        if not np.isnan(momentum) and np.sign(momentum) != np.sign(macd_hist):
            return None

        bandwidth = ind.last("bb_bandwidth", 0.0)
        bandwidth_series = ind.series("bb_bandwidth").dropna().tail(100)
        compression = (
            float((bandwidth_series <= bandwidth).mean()) if not bandwidth_series.empty else 0.5
        )

        volume_z = ind.last("volume_z", 0.0)
        strength = 0.40 + 0.25 * (1.0 - clamp(compression, 0.0, 1.0))
        if not np.isnan(volume_z) and volume_z > 1.0:
            strength += 0.15
        strength *= ctx.regime.strategy_weight(self.family)

        return build_signal(
            ctx,
            direction,
            self.name,
            scale_strength(strength),
            [
                "Volatility squeeze released (Bollinger bands exited Keltner channel)",
                f"MACD histogram {macd_hist:+.6f} sets direction",
                f"Bandwidth percentile {compression:.0%} (lower = tighter coil)",
                f"Volume z-score {volume_z:.2f}",
            ],
            features={
                "macd_hist": macd_hist,
                "bandwidth_percentile": compression,
                "volume_z": volume_z,
            },
            stop_atr_multiple=1.5,
            reward_risk=2.5,
        )
