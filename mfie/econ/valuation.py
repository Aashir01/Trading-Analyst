r"""Purchasing Power Parity / REER valuation bands.

.. math::

    PPP_{dev,t} = \frac{Spot_t - \mu_{REER,n}}{\sigma_{REER,n}}

In practice the spot rate and a REER index are not on the same scale, so the
deviation is computed as the z-score of the REER index itself (the REER already
*is* the trade-weighted, inflation-adjusted price of the currency). When only a
spot series is available the same z-score is taken on spot, which is a weaker
but directionally similar "rubber band" measure.

Rule: a currency more than ``overvalued_z`` standard deviations rich is flagged
Macro-Overvalued, which suppresses trend-following buys — you are chasing an
extreme that mean-reverts on a multi-year horizon.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mfie.config import PPPParams
from mfie.core.types import Direction, Instrument


@dataclass
class PPPView:
    currency: str
    z_score: float
    mean: float
    sigma: float
    latest: float
    state: str          # cheap | fair | stretched | overvalued | undervalued
    half_life_days: float | None = None

    @property
    def is_extreme(self) -> bool:
        return self.state in ("overvalued", "undervalued")


def _z(series: pd.Series, lookback: int) -> tuple[float, float, float, float]:
    s = pd.Series(series).dropna().astype(float).tail(lookback)
    if len(s) < 10:
        return 0.0, float("nan"), float("nan"), float("nan")
    mu = float(s.mean())
    sigma = float(s.std(ddof=1))
    latest = float(s.iloc[-1])
    if sigma <= 0 or not np.isfinite(sigma):
        return 0.0, mu, sigma, latest
    return (latest - mu) / sigma, mu, sigma, latest


def mean_reversion_half_life(series: pd.Series) -> float | None:
    """Ornstein-Uhlenbeck half-life via the AR(1) regression ``dY = a + b*Y``.

    Answers "how long until half the misvaluation unwinds" — the difference
    between an actionable dislocation and a permanent regime shift.
    """
    s = pd.Series(series).dropna().astype(float)
    if len(s) < 30:
        return None
    y_lag = s.shift(1).dropna()
    dy = (s - s.shift(1)).dropna()
    n = min(len(y_lag), len(dy))
    y_lag, dy = y_lag.iloc[-n:], dy.iloc[-n:]
    x = np.column_stack([np.ones(n), y_lag.to_numpy()])
    try:
        beta = np.linalg.lstsq(x, dy.to_numpy(), rcond=None)[0][1]
    except np.linalg.LinAlgError:
        return None
    if beta >= 0:  # not mean reverting
        return None
    return float(-np.log(2.0) / beta)


def compute_ppp_deviation(
    currency: str,
    reer_series: pd.Series,
    params: PPPParams | None = None,
) -> PPPView:
    p = params or PPPParams()
    z, mu, sigma, latest = _z(reer_series, p.lookback_periods)

    if z >= p.overvalued_z:
        state = "overvalued"
    elif z <= -p.overvalued_z:
        state = "undervalued"
    elif abs(z) >= p.stretched_z:
        state = "stretched"
    elif abs(z) <= 0.5:
        state = "fair"
    else:
        state = "cheap" if z < 0 else "rich"

    return PPPView(
        currency=currency,
        z_score=float(z),
        mean=mu,
        sigma=sigma,
        latest=latest,
        state=state,
        half_life_days=mean_reversion_half_life(reer_series),
    )


def pair_valuation_bias(
    instrument: Instrument,
    base_view: PPPView | None,
    quote_view: PPPView | None,
) -> float:
    """Net valuation z for the pair: base richness minus quote richness.

    Positive means the base currency is expensive relative to the quote, i.e. the
    pair itself is macro-overvalued and rallies are suspect.
    """
    base_z = base_view.z_score if base_view else 0.0
    quote_z = quote_view.z_score if quote_view else 0.0
    return float(base_z - quote_z)


def valuation_multiplier(
    pair_z: float,
    direction: Direction,
    params: PPPParams | None = None,
) -> tuple[str, float, bool]:
    """Translate a pair valuation z-score into ``(state, multiplier, block)``.

    Buying an already-overvalued pair is penalised; buying an undervalued one is
    left alone (valuation is a headwind detector, not an entry trigger).
    """
    p = params or PPPParams()
    if direction is Direction.FLAT:
        return "neutral", 1.0, False

    # Positive when the trade pushes further into the stretched direction.
    stretch = pair_z * direction.sign

    if stretch >= p.block_breakouts_beyond_z:
        return "extreme_against", p.penalty, True
    if stretch >= p.overvalued_z:
        return "overextended", p.penalty, False
    if stretch >= p.stretched_z:
        # Partial penalty between the stretched and overvalued bands.
        span = max(p.overvalued_z - p.stretched_z, 1e-6)
        weight = (stretch - p.stretched_z) / span
        return "stretched", float(1.0 - (1.0 - p.penalty) * weight), False
    if stretch <= -p.overvalued_z:
        return "value_tailwind", 1.0, False
    return "fair", 1.0, False
