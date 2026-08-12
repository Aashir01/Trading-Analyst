"""Market regime detection — which strategy family should be live right now.

Markets cycle through structural states, and a strategy that prints money in one
loses it in another. Rather than run every strategy all the time, the pipeline
asks this module which family fits, and down-weights the mismatched ones.

Two independent classifiers, deliberately:

* ``detect_regime`` — transparent, rule-based, from ADX / volatility percentile /
  Hurst exponent / drawdown. You can explain it to a risk committee.
* ``gaussian_mixture_states`` — unsupervised, fits a mixture over
  (return, |return|, range) and labels states by their fitted variance. This is
  the practical stand-in for a Hidden Markov Model without adding hmmlearn as a
  dependency; ``transition_matrix`` recovers the Markov dynamics afterwards.

They are cross-checked: when both agree, confidence is high.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mfie.config import RegimeParams
from mfie.core.types import MarketRegime
from mfie.core.utils import clamp, get_logger
from mfie.technical.indicators import IndicatorSet, hurst_exponent

log = get_logger(__name__)


@dataclass
class RegimeView:
    regime: MarketRegime
    confidence: float
    adx: float
    vol_percentile: float
    realized_vol: float
    hurst: float
    drawdown: float
    trend_slope: float
    mixture_state: int | None = None
    detail: dict[str, float] = field(default_factory=dict)

    @property
    def prefers(self) -> str:
        if self.regime is MarketRegime.CRASH:
            return "defensive"
        if self.regime.favours_momentum:
            return "momentum"
        if self.regime.favours_mean_reversion:
            return "mean_reversion"
        return "neutral"

    def strategy_weight(self, family: str) -> float:
        """Multiplier applied to a strategy's raw strength given the regime.

        Families: ``momentum``, ``mean_reversion``, ``breakout``, ``carry``,
        ``market_making``.
        """
        table: dict[MarketRegime, dict[str, float]] = {
            MarketRegime.TRENDING_HIGH_VOL: {
                "momentum": 1.15, "breakout": 1.20, "mean_reversion": 0.50,
                "carry": 0.70, "market_making": 0.40,
            },
            MarketRegime.TRENDING_LOW_VOL: {
                "momentum": 1.10, "breakout": 1.00, "mean_reversion": 0.70,
                "carry": 1.15, "market_making": 0.70,
            },
            MarketRegime.RANGING: {
                "momentum": 0.55, "breakout": 0.60, "mean_reversion": 1.20,
                "carry": 1.05, "market_making": 1.20,
            },
            MarketRegime.CRASH: {
                "momentum": 0.60, "breakout": 0.50, "mean_reversion": 0.35,
                "carry": 0.30, "market_making": 0.20,
            },
            MarketRegime.UNKNOWN: {},
        }
        return float(table.get(self.regime, {}).get(family, 1.0))


def _vol_percentile(realized_vol: pd.Series, lookback: int) -> float:
    s = pd.Series(realized_vol).dropna().astype(float).tail(lookback)
    if len(s) < 10:
        return 0.5
    latest = float(s.iloc[-1])
    return float((s <= latest).mean())


def _trend_slope(close: pd.Series, lookback: int) -> float:
    """Normalised OLS slope of log price — direction and steepness in one number."""
    s = pd.Series(close).dropna().astype(float).tail(lookback)
    if len(s) < 10:
        return 0.0
    y = np.log(s.to_numpy())
    x = np.arange(len(y), dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    return slope * len(y)  # total log-return implied by the fitted trend


def detect_regime(indicators: IndicatorSet, params: RegimeParams | None = None) -> RegimeView:
    p = params or RegimeParams()
    df = indicators.df

    adx_value = indicators.last("adx", 0.0)
    realized = indicators.series("realized_vol")
    vol_pct = _vol_percentile(realized, p.vol_lookback)
    vol_now = indicators.last("realized_vol", 0.0)
    hurst = hurst_exponent(df["close"].tail(max(p.trend_lookback * 4, 120)))
    slope = _trend_slope(df["close"], p.trend_lookback)

    window = df["close"].tail(p.vol_lookback)
    peak = float(window.max()) if not window.empty else 0.0
    drawdown = float((peak - float(window.iloc[-1])) / peak) if peak > 0 else 0.0

    trending = adx_value >= p.adx_trending or hurst > 0.55
    high_vol = vol_pct >= p.high_vol_percentile
    low_vol = vol_pct <= p.low_vol_percentile

    # A sharp drawdown in a high-volatility tape is its own regime: correlations
    # go to 1 and every non-defensive edge stops working.
    if drawdown > 0.15 and high_vol and slope < 0:
        regime = MarketRegime.CRASH
    elif trending and high_vol:
        regime = MarketRegime.TRENDING_HIGH_VOL
    elif trending:
        regime = MarketRegime.TRENDING_LOW_VOL
    elif low_vol or hurst < 0.45:
        regime = MarketRegime.RANGING
    else:
        regime = MarketRegime.UNKNOWN

    # Confidence: how far the evidence is from the decision boundaries.
    adx_conf = clamp(abs(adx_value - p.adx_trending) / p.adx_trending, 0.0, 1.0)
    hurst_conf = clamp(abs(hurst - 0.5) * 4.0, 0.0, 1.0)
    vol_conf = clamp(abs(vol_pct - 0.5) * 2.0, 0.0, 1.0)
    confidence = clamp((adx_conf + hurst_conf + vol_conf) / 3.0, 0.0, 1.0)

    return RegimeView(
        regime=regime,
        confidence=float(confidence),
        adx=float(adx_value),
        vol_percentile=float(vol_pct),
        realized_vol=float(vol_now),
        hurst=float(hurst),
        drawdown=float(drawdown),
        trend_slope=float(slope),
        detail={
            "adx_threshold": p.adx_trending,
            "high_vol_pct": p.high_vol_percentile,
            "low_vol_pct": p.low_vol_percentile,
        },
    )


def gaussian_mixture_states(
    df: pd.DataFrame,
    n_states: int = 3,
    lookback: int = 500,
) -> pd.Series | None:
    """Unsupervised volatility-regime labels, ordered 0 = calmest.

    Features are (return, |return|, high-low range). Returns a Series of integer
    state labels aligned to the frame's index, or ``None`` if the fit fails.
    """
    frame = df.tail(lookback).copy()
    if len(frame) < max(60, n_states * 20):
        return None

    returns = frame["close"].pct_change()
    rng = (frame["high"] - frame["low"]) / frame["close"].replace(0.0, np.nan)
    features = pd.concat(
        [returns.rename("r"), returns.abs().rename("abs_r"), rng.rename("range")], axis=1
    ).dropna()
    if len(features) < max(60, n_states * 20):
        return None

    try:
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler

        x = StandardScaler().fit_transform(features.to_numpy())
        model = GaussianMixture(
            n_components=n_states, covariance_type="full", random_state=7, n_init=3
        )
        labels = model.fit_predict(x)
    except Exception as exc:  # pragma: no cover - numerical edge cases
        log.debug("Gaussian mixture regime fit failed: %s", exc)
        return None

    # Relabel so 0 is the lowest-volatility state; raw component order is arbitrary.
    raw = pd.Series(labels, index=features.index)
    vol_by_state = features["abs_r"].groupby(raw).mean().sort_values()
    remap = {old: new for new, old in enumerate(vol_by_state.index)}
    return raw.map(remap).rename("regime_state")


def transition_matrix(states: pd.Series) -> pd.DataFrame:
    """Empirical Markov transition probabilities between regime states."""
    s = pd.Series(states).dropna().astype(int)
    if len(s) < 3:
        return pd.DataFrame()
    labels = sorted(s.unique())
    counts = pd.DataFrame(0.0, index=labels, columns=labels)
    for a, b in zip(s.iloc[:-1], s.iloc[1:], strict=True):
        counts.loc[a, b] += 1.0
    row_sums = counts.sum(axis=1).replace(0.0, np.nan)
    return counts.div(row_sums, axis=0).fillna(0.0)


def expected_regime_duration(matrix: pd.DataFrame, state: int) -> float:
    """Expected bars remaining in ``state``: 1 / (1 - p_stay)."""
    if matrix.empty or state not in matrix.index:
        return 0.0
    p_stay = float(matrix.loc[state, state])
    if p_stay >= 1.0:
        return float("inf")
    return float(1.0 / max(1.0 - p_stay, 1e-9))
