r"""Global Liquidity Index (GLI).

The single most important macro gate in the engine: momentum strategies are a
levered bet on liquidity expansion, and they fail systematically when the
monetary tide goes out.

.. math::

    \Delta GLI_t = w_1 \frac{M_{2,t} - M_{2,t-n}}{M_{2,t-n}}
                 + w_2 \frac{SC_t - SC_{t-n}}{SC_{t-n}}

``M2`` is broad money (the fiat side), ``SC`` is aggregate stablecoin supply
(the crypto-native side), ``n`` is the lookback window and :math:`w_1 + w_2 = 1`.

Rule: :math:`\Delta GLI_t < 0` applies a penalty multiplier to momentum and
breakout longs; :math:`\Delta GLI_t \ge 0` lets them size normally.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mfie.config import LiquidityParams
from mfie.core.types import LiquidityRegime
from mfie.core.utils import clamp, pct_change


@dataclass
class GlobalLiquidity:
    delta: float                 # ΔGLI
    m2_change: float
    stablecoin_change: float
    regime: LiquidityRegime
    lookback_days: int
    momentum_multiplier: float   # what the filter chain applies to momentum longs
    detail: dict[str, float]

    @property
    def is_contracting(self) -> bool:
        return self.regime is LiquidityRegime.CONTRACTION


def _window_change(series: pd.Series, lookback_days: int) -> float:
    """Relative change over ``lookback_days``, tolerating irregular frequencies.

    M2 is monthly, stablecoin supply is daily; both are handled by selecting the
    last observation at or before the cutoff rather than a fixed row offset.
    """
    s = pd.Series(series).dropna().astype(float).sort_index()
    if len(s) < 2:
        return 0.0
    if not isinstance(s.index, pd.DatetimeIndex):
        past_idx = max(0, len(s) - 1 - lookback_days)
        return pct_change(float(s.iloc[-1]), float(s.iloc[past_idx]))

    cutoff = s.index[-1] - pd.Timedelta(days=lookback_days)
    earlier = s[s.index <= cutoff]
    past = float(earlier.iloc[-1]) if not earlier.empty else float(s.iloc[0])
    return pct_change(float(s.iloc[-1]), past)


def compute_gli(
    m2: pd.Series,
    stablecoin_supply: pd.Series,
    params: LiquidityParams | None = None,
) -> GlobalLiquidity:
    p = params or LiquidityParams()

    w1, w2 = p.m2_weight, p.stablecoin_weight
    total = w1 + w2
    if total > 0:  # normalise so weights always sum to 1
        w1, w2 = w1 / total, w2 / total

    m2_change = _window_change(m2, p.lookback_days)
    sc_change = _window_change(stablecoin_supply, p.lookback_days)
    delta = w1 * m2_change + w2 * sc_change

    if delta < p.contraction_threshold:
        regime = LiquidityRegime.CONTRACTION
        # Scale the penalty with the depth of the contraction rather than
        # applying a cliff: a -0.1% print should not be treated like a -5% one.
        severity = clamp(abs(delta) / max(abs(p.strong_expansion_threshold), 1e-6), 0.0, 1.0)
        multiplier = 1.0 - (1.0 - p.contraction_penalty) * severity
    elif delta >= p.strong_expansion_threshold:
        regime = LiquidityRegime.EXPANSION
        multiplier = 1.0
    else:
        regime = LiquidityRegime.NEUTRAL
        multiplier = 1.0

    return GlobalLiquidity(
        delta=float(delta),
        m2_change=float(m2_change),
        stablecoin_change=float(sc_change),
        regime=regime,
        lookback_days=p.lookback_days,
        momentum_multiplier=float(clamp(multiplier, 0.1, 1.2)),
        detail={
            "w_m2": w1,
            "w_stablecoin": w2,
            "threshold_contraction": p.contraction_threshold,
            "threshold_expansion": p.strong_expansion_threshold,
        },
    )


def liquidity_trend(m2: pd.Series, stablecoin_supply: pd.Series,
                    params: LiquidityParams | None = None, points: int = 60) -> pd.Series:
    """Rolling ΔGLI history, for charting the liquidity cycle on the dashboard."""
    p = params or LiquidityParams()
    sc = pd.Series(stablecoin_supply).dropna().sort_index()
    if sc.empty:
        return pd.Series(dtype=float, name="gli")

    out: dict[pd.Timestamp, float] = {}
    for ts in sc.index[-points:]:
        window_m2 = pd.Series(m2)[pd.Series(m2).index <= ts] if len(m2) else pd.Series(dtype=float)
        window_sc = sc[sc.index <= ts]
        if len(window_sc) < 2:
            continue
        out[ts] = compute_gli(window_m2, window_sc, p).delta
    return pd.Series(out, name="gli")
