r"""Economic Surprise Index (ESI).

Markets trade the *delta* between realised data and consensus, not the level.

.. math::

    ESI_t = \sum_{i=1}^{m}
        \left( \frac{Actual_i - Forecast_i}{\sigma_{hist,i}} \right)
        e^{-\lambda (t - t_i)}

Each release is standardised by the historical dispersion of surprises for that
same indicator (so a 10k NFP miss and a 0.1% CPI miss are comparable), then
exponentially decayed so last week's data outweighs last month's.

Rule: if :math:`ESI_A - ESI_B > 0`, skew pair ``A/B`` toward longs.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from math import exp

import numpy as np

from mfie.config import ESIParams
from mfie.core.types import Direction, EconomicEvent
from mfie.core.utils import clamp


def _historical_sigma(events: list[EconomicEvent]) -> dict[str, float]:
    """Dispersion of past surprises, per (currency, indicator).

    This is the standardiser. With too few observations it falls back to the
    absolute forecast level, which keeps the ratio dimensionless.
    """
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for e in events:
        surprise = e.surprise
        if surprise is not None:
            buckets[(e.currency, e.name)].append(surprise)

    sigmas: dict[str, float] = {}
    for (currency, name), values in buckets.items():
        key = f"{currency}|{name}"
        if len(values) >= 4:
            sigma = float(np.std(values, ddof=1))
        else:
            sigma = float(np.mean([abs(v) for v in values])) if values else 0.0
        sigmas[key] = sigma if sigma > 0 else 0.0
    return sigmas


def compute_esi(
    events: list[EconomicEvent],
    params: ESIParams | None = None,
    as_of: datetime | None = None,
) -> dict[str, float]:
    """Time-decayed surprise score per currency."""
    p = params or ESIParams()
    now = as_of or datetime.now(timezone.utc)
    sigmas = _historical_sigma(events)

    scores: dict[str, float] = defaultdict(float)
    for e in events:
        if e.actual is None or e.forecast is None:
            continue
        if e.ts > now:  # not released yet
            continue
        age_days = (now - e.ts).total_seconds() / 86_400.0
        if age_days > p.window_days:
            continue

        sigma = sigmas.get(f"{e.currency}|{e.name}", 0.0)
        if sigma <= 0:
            # No dispersion history: scale by the forecast magnitude so the
            # standardised surprise stays O(1) instead of exploding.
            sigma = max(abs(e.forecast) * 0.05, 1e-6)

        standardised = (e.actual - e.forecast) / sigma
        # Cap individual outliers so one bad print cannot dominate the index.
        standardised = clamp(standardised, -5.0, 5.0)

        weight = exp(-p.decay_lambda * age_days)
        impact_weight = {"high": 1.0, "medium": 0.6, "low": 0.3}.get(e.impact, 0.3)
        scores[e.currency] += standardised * weight * impact_weight

    return dict(scores)


def esi_differential(esi: dict[str, float], base: str, quote: str) -> float:
    """ESI_base - ESI_quote. Positive favours the base currency."""
    return float(esi.get(base, 0.0) - esi.get(quote, 0.0))


def esi_multiplier(
    differential: float,
    direction: Direction,
    params: ESIParams | None = None,
) -> tuple[str, float]:
    """Turn an ESI differential into ``(state, multiplier)`` for a direction."""
    p = params or ESIParams()
    if direction is Direction.FLAT:
        return "neutral", 1.0

    signed = differential * direction.sign
    if abs(signed) < p.significant_delta:
        return "neutral", 1.0
    return ("aligned", p.aligned_bonus) if signed > 0 else ("opposed", p.opposed_penalty)


def esi_history(
    events: list[EconomicEvent],
    currency: str,
    params: ESIParams | None = None,
    points: int = 60,
) -> dict[datetime, float]:
    """ESI recomputed as of each past release date — for charting the trend."""
    p = params or ESIParams()
    released = sorted(
        [e for e in events if e.currency == currency and e.actual is not None],
        key=lambda e: e.ts,
    )
    out: dict[datetime, float] = {}
    for event in released[-points:]:
        out[event.ts] = compute_esi(events, p, as_of=event.ts).get(currency, 0.0)
    return out


def next_release(events: list[EconomicEvent], currency: str,
                 as_of: datetime | None = None) -> EconomicEvent | None:
    now = as_of or datetime.now(timezone.utc)
    upcoming = [e for e in events if e.currency == currency and e.ts > now]
    return min(upcoming, key=lambda e: e.ts) if upcoming else None
