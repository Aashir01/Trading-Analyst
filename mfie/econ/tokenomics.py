r"""Crypto network economics — treating a chain as a small sovereign economy.

Quantity theory of money applied to a token:

.. math::

    M \cdot V = P \cdot Q
    \quad\Longrightarrow\quad
    V_t = \frac{\text{on-chain transaction volume (USD)}_t}{\text{market cap (USD)}_t}

Network Value to Transactions, the crypto P/E:

.. math::  NVT_t = \frac{\text{Market Cap}_t}{\text{Daily on-chain volume}_t}

Rules:

* Price rising while velocity collapses (:math:`V_t < \mu_V - 1.5\sigma_V`) is
  speculative hoarding without utility — flag a divergence and penalise longs.
* :math:`NVT_t > \mu + 2\sigma` means the network is expensive relative to the
  economic activity it settles — cut position size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mfie.config import TokenomicsParams
from mfie.core.types import Direction
from mfie.core.utils import clamp, safe_div


@dataclass
class TokenomicsView:
    symbol: str
    velocity: float
    velocity_z: float
    nvt: float
    nvt_z: float
    nvt_signal: float          # NVT smoothed by a 90d moving average of volume
    price_change: float        # over the same window
    velocity_change: float
    divergence: bool           # price up, velocity structurally down
    overvalued: bool           # NVT z above threshold
    detail: dict[str, float]

    @property
    def warning(self) -> str | None:
        if self.divergence and self.overvalued:
            return "Speculative bubble: price advancing on collapsing utility and stretched NVT"
        if self.divergence:
            return "Velocity divergence: price advancing without on-chain utility"
        if self.overvalued:
            return "NVT stretched: network value rich vs. settled volume"
        return None


def token_velocity(onchain_volume: pd.Series, market_cap: pd.Series) -> pd.Series:
    """V = P*Q / M, aligned on the intersection of both series' dates."""
    vol = pd.Series(onchain_volume).dropna().astype(float)
    cap = pd.Series(market_cap).dropna().astype(float)
    if vol.empty or cap.empty:
        return pd.Series(dtype=float, name="velocity")

    frame = pd.concat([vol.rename("vol"), cap.rename("cap")], axis=1, sort=True).dropna()
    if frame.empty:
        # Different sampling grids: reindex volume onto the market-cap dates.
        aligned = vol.reindex(cap.index, method="nearest", tolerance=pd.Timedelta("2D"))
        frame = pd.concat([aligned.rename("vol"), cap.rename("cap")], axis=1, sort=True).dropna()
    if frame.empty:
        return pd.Series(dtype=float, name="velocity")

    return (frame["vol"] / frame["cap"].replace(0.0, np.nan)).rename("velocity").dropna()


def nvt_ratio(market_cap: pd.Series, onchain_volume: pd.Series) -> pd.Series:
    """NVT = market cap / daily on-chain volume — the inverse of velocity."""
    velocity = token_velocity(onchain_volume, market_cap)
    if velocity.empty:
        return pd.Series(dtype=float, name="nvt")
    return (1.0 / velocity.replace(0.0, np.nan)).rename("nvt").dropna()


def nvt_signal(market_cap: pd.Series, onchain_volume: pd.Series, window: int = 90) -> pd.Series:
    """NVT Signal (Kalichkin): cap divided by a moving average of volume.

    Smoothing the denominator removes the daily volume noise that makes raw NVT
    almost unusable as a live indicator.
    """
    vol = pd.Series(onchain_volume).dropna().astype(float)
    cap = pd.Series(market_cap).dropna().astype(float)
    if vol.empty or cap.empty:
        return pd.Series(dtype=float, name="nvt_signal")
    smoothed = vol.rolling(min(window, max(len(vol) // 3, 2)), min_periods=2).mean()
    frame = pd.concat([smoothed.rename("vol"), cap.rename("cap")], axis=1, sort=True).dropna()
    if frame.empty:
        aligned = smoothed.reindex(cap.index, method="nearest", tolerance=pd.Timedelta("2D"))
        frame = pd.concat([aligned.rename("vol"), cap.rename("cap")], axis=1, sort=True).dropna()
    if frame.empty:
        return pd.Series(dtype=float, name="nvt_signal")
    return (frame["cap"] / frame["vol"].replace(0.0, np.nan)).rename("nvt_signal").dropna()


def _tail_z(series: pd.Series, window: int) -> tuple[float, float]:
    s = pd.Series(series).dropna().astype(float).tail(window)
    if len(s) < 10:
        return 0.0, float(s.iloc[-1]) if len(s) else 0.0
    mu, sigma = float(s.mean()), float(s.std(ddof=1))
    latest = float(s.iloc[-1])
    if sigma <= 0 or not np.isfinite(sigma):
        return 0.0, latest
    return (latest - mu) / sigma, latest


def compute_tokenomics(
    symbol: str,
    onchain_volume: pd.Series,
    market_cap: pd.Series,
    params: TokenomicsParams | None = None,
) -> TokenomicsView:
    p = params or TokenomicsParams()

    velocity = token_velocity(onchain_volume, market_cap)
    nvt = nvt_ratio(market_cap, onchain_volume)
    nvt_sig = nvt_signal(market_cap, onchain_volume, p.nvt_lookback)

    v_z, v_latest = _tail_z(velocity, p.velocity_lookback)
    n_z, n_latest = _tail_z(nvt_sig if not nvt_sig.empty else nvt, p.nvt_lookback)
    nvt_signal_latest = float(nvt_sig.iloc[-1]) if not nvt_sig.empty else n_latest

    cap = pd.Series(market_cap).dropna().astype(float)
    window = min(p.velocity_lookback, max(len(cap) - 1, 1))
    price_change = safe_div(
        float(cap.iloc[-1]) - float(cap.iloc[-1 - window]), abs(float(cap.iloc[-1 - window]))
    ) if len(cap) > window else 0.0

    vel = pd.Series(velocity).dropna()
    v_window = min(p.velocity_lookback, max(len(vel) - 1, 1))
    velocity_change = safe_div(
        float(vel.iloc[-1]) - float(vel.iloc[-1 - v_window]), abs(float(vel.iloc[-1 - v_window]))
    ) if len(vel) > v_window else 0.0

    # MV = PQ divergence: M (cap) expanding while V contracts below its band.
    divergence = bool(price_change > 0.0 and v_z <= p.velocity_z_warning)
    overvalued = bool(n_z >= p.nvt_z_overvalued)

    return TokenomicsView(
        symbol=symbol,
        velocity=float(v_latest),
        velocity_z=float(v_z),
        nvt=float(nvt.iloc[-1]) if not nvt.empty else 0.0,
        nvt_z=float(n_z),
        nvt_signal=nvt_signal_latest,
        price_change=float(price_change),
        velocity_change=float(velocity_change),
        divergence=divergence,
        overvalued=overvalued,
        detail={
            "velocity_z_threshold": p.velocity_z_warning,
            "nvt_z_threshold": p.nvt_z_overvalued,
            "observations": float(len(velocity)),
        },
    )


def tokenomics_multiplier(
    view: TokenomicsView,
    direction: Direction,
    params: TokenomicsParams | None = None,
) -> tuple[str, float]:
    """Penalise longs into a hollow network; leave shorts alone."""
    p = params or TokenomicsParams()
    if direction is not Direction.LONG:
        return "n/a", 1.0

    multiplier = 1.0
    states = []
    if view.divergence:
        multiplier *= p.divergence_penalty
        states.append("velocity_divergence")
    if view.overvalued:
        multiplier *= p.nvt_penalty
        states.append("nvt_overvalued")

    if not states:
        return "healthy", 1.0
    return "+".join(states), clamp(multiplier, 0.1, 1.0)
