r"""Behavioural economics: contrarian penalties on crowded positioning.

Retail crowding is the most reliable behavioural anomaly available for free.
The penalty is symmetric and continuous:

.. math::

    C_t = 1.0 - \left| \frac{\%Long_t - 0.50}{0.50} \right|^{k}

At balanced positioning (50% long) :math:`C_t = 1`; as the crowd approaches
100% long, :math:`C_t \to 0`. ``k >= 2`` keeps the penalty mild until
positioning is genuinely extreme.

The penalty applies only to trades that agree with the crowd. Taking the other
side of a 90%-long book is exactly what the rule wants you to do, so those
signals are left alone (and optionally boosted).
"""

from __future__ import annotations

from dataclasses import dataclass

from mfie.config import BehavioralParams
from mfie.core.types import Direction
from mfie.core.utils import clamp


def crowding_score(pct_long: float) -> float:
    """|(%long - 0.5) / 0.5| in [0, 1]. 0 = balanced, 1 = fully one-sided."""
    return clamp(abs((clamp(pct_long, 0.0, 1.0) - 0.5) / 0.5), 0.0, 1.0)


def contrarian_multiplier(
    pct_long: float,
    direction: Direction,
    params: BehavioralParams | None = None,
) -> tuple[str, float]:
    """``(state, multiplier)`` for a trade given retail net-long share.

    ``pct_long`` is a fraction in [0, 1].
    """
    p = params or BehavioralParams()
    if direction is Direction.FLAT:
        return "neutral", 1.0

    long_share = clamp(pct_long, 0.0, 1.0)
    crowd_direction = Direction.LONG if long_share > 0.5 else Direction.SHORT
    score = crowding_score(long_share)

    # C_t from the formula above.
    c_t = clamp(1.0 - score**p.contrarian_k, 0.0, 1.0)

    if score < 0.1:
        return "balanced", 1.0

    if direction is crowd_direction:
        state = "crowded_extreme" if score >= (p.extreme_threshold - 0.5) / 0.5 else "crowded"
        return state, max(c_t, 0.05)

    # Trading against the crowd: no penalty, and a modest edge when extreme.
    if score >= (p.extreme_threshold - 0.5) / 0.5:
        return "contrarian_edge", clamp(1.0 + 0.25 * score, 1.0, 1.25)
    return "against_crowd", 1.0


def fear_greed_multiplier(
    index_value: float | None,
    direction: Direction,
    params: BehavioralParams | None = None,
) -> tuple[str, float]:
    """Crypto Fear & Greed (0-100) as a second crowding gauge.

    Extreme greed penalises longs; extreme fear penalises shorts.
    """
    p = params or BehavioralParams()
    if index_value is None or direction is Direction.FLAT:
        return "unknown", 1.0

    if index_value >= p.fear_greed_extreme_greed and direction is Direction.LONG:
        return "extreme_greed", p.fear_greed_penalty
    if index_value <= p.fear_greed_extreme_fear and direction is Direction.SHORT:
        return "extreme_fear", p.fear_greed_penalty
    if index_value <= p.fear_greed_extreme_fear and direction is Direction.LONG:
        return "fear_opportunity", 1.05
    if index_value >= p.fear_greed_extreme_greed and direction is Direction.SHORT:
        return "greed_opportunity", 1.05
    return "neutral", 1.0


def news_sentiment_multiplier(score: float | None, direction: Direction) -> tuple[str, float]:
    """Headline polarity as a mild confirmation/contradiction signal.

    Kept deliberately weak (±10%): news sentiment is noisy and often already in
    the price by the time an API serves it.
    """
    if score is None or direction is Direction.FLAT or abs(score) < 0.25:
        return "neutral", 1.0
    aligned = (score > 0) == (direction is Direction.LONG)
    magnitude = clamp(abs(score), 0.0, 1.0)
    if aligned:
        return "news_supportive", clamp(1.0 + 0.10 * magnitude, 1.0, 1.10)
    return "news_adverse", clamp(1.0 - 0.10 * magnitude, 0.90, 1.0)


@dataclass
class BehavioralView:
    pct_long: float
    crowding: float
    contrarian_state: str
    contrarian_mult: float
    fear_greed: float | None
    fear_greed_state: str
    fear_greed_mult: float
    news_score: float | None
    news_state: str
    news_mult: float

    @property
    def combined(self) -> float:
        return self.contrarian_mult * self.fear_greed_mult * self.news_mult


def compute_behavioral(
    pct_long: float,
    fear_greed: float | None,
    news_score: float | None,
    direction: Direction,
    params: BehavioralParams | None = None,
) -> BehavioralView:
    c_state, c_mult = contrarian_multiplier(pct_long, direction, params)
    f_state, f_mult = fear_greed_multiplier(fear_greed, direction, params)
    n_state, n_mult = news_sentiment_multiplier(news_score, direction)
    return BehavioralView(
        pct_long=float(pct_long),
        crowding=crowding_score(pct_long),
        contrarian_state=c_state,
        contrarian_mult=c_mult,
        fear_greed=fear_greed,
        fear_greed_state=f_state,
        fear_greed_mult=f_mult,
        news_score=news_score,
        news_state=n_state,
        news_mult=n_mult,
    )
