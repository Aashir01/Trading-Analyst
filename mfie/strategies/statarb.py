r"""Statistical arbitrage — cointegration pairs trading.

This family does not forecast direction. It bets that a *relationship* between
two assets, which has been stable historically, will re-assert itself.

Method:

1. Engle-Granger test that :math:`y_t - \beta x_t` is stationary (ADF on the
   residual of the OLS hedge regression).
2. Trade the spread's z-score: short the rich leg, long the cheap leg, when
   :math:`|z| > entry_z`; exit at :math:`|z| < exit_z`.
3. Size by the half-life of mean reversion — a spread with a 200-bar half-life
   is not tradable on an hourly chart no matter how significant the test.

The obvious failure mode is a structural break: two assets can be cointegrated
for years and then stop, which is why ``cointegration_pvalue`` is re-tested on
every call rather than fitted once.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from mfie.core.types import Direction, RawSignal
from mfie.core.utils import clamp, get_logger
from mfie.strategies.base import Strategy, StrategyContext, build_signal, scale_strength

log = get_logger(__name__)


@dataclass
class SpreadModel:
    hedge_ratio: float
    intercept: float
    spread: pd.Series
    zscore: float
    pvalue: float
    half_life: float | None
    correlation: float

    @property
    def tradable(self) -> bool:
        return self.pvalue < 0.05 and self.half_life is not None and 0 < self.half_life < 200


def hedge_ratio(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    """OLS of ``y`` on ``x`` with an intercept. Returns ``(beta, alpha)``."""
    frame = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(frame) < 30:
        return 1.0, 0.0
    design = np.column_stack([np.ones(len(frame)), frame["x"].to_numpy()])
    try:
        coef = np.linalg.lstsq(design, frame["y"].to_numpy(), rcond=None)[0]
    except np.linalg.LinAlgError:
        return 1.0, 0.0
    return float(coef[1]), float(coef[0])


def cointegration_pvalue(y: pd.Series, x: pd.Series) -> float:
    """Engle-Granger p-value. Lower means a more reliable long-run relationship."""
    frame = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(frame) < 60:
        return 1.0
    try:
        from statsmodels.tsa.stattools import coint

        return float(coint(frame["y"], frame["x"])[1])
    except Exception as exc:  # pragma: no cover - numerical/import edge cases
        log.debug("Cointegration test failed: %s", exc)
        return 1.0


def spread_half_life(spread: pd.Series) -> float | None:
    """Half-life of mean reversion from the AR(1) coefficient."""
    s = pd.Series(spread).dropna().astype(float)
    if len(s) < 40:
        return None
    lag = s.shift(1).dropna()
    delta = (s - s.shift(1)).dropna()
    n = min(len(lag), len(delta))
    design = np.column_stack([np.ones(n), lag.to_numpy()[-n:]])
    try:
        beta = np.linalg.lstsq(design, delta.to_numpy()[-n:], rcond=None)[0][1]
    except np.linalg.LinAlgError:
        return None
    if beta >= 0:
        return None
    return float(-np.log(2.0) / beta)


def build_spread_model(y: pd.Series, x: pd.Series, zscore_window: int = 60) -> SpreadModel:
    beta, alpha = hedge_ratio(y, x)
    frame = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    spread = frame["y"] - (beta * frame["x"] + alpha)

    window = spread.tail(zscore_window)
    sigma = float(window.std(ddof=1))
    z = float((spread.iloc[-1] - window.mean()) / sigma) if sigma > 0 else 0.0

    return SpreadModel(
        hedge_ratio=beta,
        intercept=alpha,
        spread=spread,
        zscore=z,
        pvalue=cointegration_pvalue(frame["y"], frame["x"]),
        half_life=spread_half_life(spread),
        correlation=float(frame["y"].corr(frame["x"])),
    )


def find_cointegrated_pairs(
    prices: pd.DataFrame,
    max_pvalue: float = 0.05,
    min_correlation: float = 0.6,
) -> list[dict[str, float | str]]:
    """Scan a price matrix (columns = symbols) for tradable pairs.

    Note the multiple-comparisons problem: testing N symbols means N(N-1)/2
    tests, so some pairs pass at 5% by chance alone. Treat the output as
    candidates to investigate, not as a portfolio.
    """
    results: list[dict[str, float | str]] = []
    columns = [c for c in prices.columns if prices[c].notna().sum() >= 100]
    for a, b in combinations(columns, 2):
        y, x = prices[a].dropna(), prices[b].dropna()
        common = y.index.intersection(x.index)
        if len(common) < 100:
            continue
        y, x = y.loc[common], x.loc[common]
        corr = float(y.corr(x))
        if abs(corr) < min_correlation:
            continue
        model = build_spread_model(y, x)
        if model.pvalue > max_pvalue:
            continue
        results.append(
            {
                "asset_a": a,
                "asset_b": b,
                "pvalue": model.pvalue,
                "hedge_ratio": model.hedge_ratio,
                "half_life": model.half_life if model.half_life is not None else float("nan"),
                "zscore": model.zscore,
                "correlation": corr,
                "n_tests": len(columns) * (len(columns) - 1) / 2,
            }
        )
    return sorted(results, key=lambda r: r["pvalue"])


class PairsTradingStrategy(Strategy):
    """Trade the ``instrument`` leg of a cointegrated pair.

    Requires ``ctx.extras['partner_close']`` (a price Series) and
    ``ctx.extras['partner_symbol']``. The pipeline supplies these when a
    cointegrated partner is found in the same asset class.
    """

    name = "pairs_trading"
    family = "mean_reversion"
    description = "Cointegration spread z-score reversion against a partner asset"
    min_bars = 150

    entry_z = 2.0
    exit_z = 0.5

    def supports(self, ctx: StrategyContext) -> bool:
        if not super().supports(ctx):
            return False
        return isinstance(ctx.extras.get("partner_close"), pd.Series)

    def generate(self, ctx: StrategyContext) -> RawSignal | None:
        partner: pd.Series = ctx.extras["partner_close"]
        partner_symbol = str(ctx.extras.get("partner_symbol", "partner"))

        y = ctx.df["close"]
        common = y.index.intersection(partner.index)
        if len(common) < self.min_bars:
            return None

        model = build_spread_model(y.loc[common], partner.loc[common])
        if not model.tradable:
            return None
        if abs(model.zscore) < self.entry_z:
            return None

        # Spread rich (z > 0) means this leg is expensive relative to the
        # partner: short it and expect convergence.
        direction = Direction.SHORT if model.zscore > 0 else Direction.LONG

        strength = 0.35 + 0.25 * clamp((abs(model.zscore) - self.entry_z) / 2.0, 0.0, 1.0)
        strength += 0.20 * clamp(1.0 - model.pvalue / 0.05, 0.0, 1.0)
        if model.half_life and model.half_life < 40:
            strength += 0.10  # fast convergence is worth more
        strength *= ctx.regime.strategy_weight(self.family)

        return build_signal(
            ctx,
            direction,
            self.name,
            scale_strength(strength),
            [
                f"Spread vs {partner_symbol} at z={model.zscore:+.2f} "
                f"(entry threshold ±{self.entry_z})",
                f"Engle-Granger p-value {model.pvalue:.4f}",
                f"Hedge ratio {model.hedge_ratio:.4f}, correlation {model.correlation:.2f}",
                f"Half-life {model.half_life:.1f} bars"
                if model.half_life
                else "Half-life unavailable",
            ],
            features={
                "spread_z": model.zscore,
                "pvalue": model.pvalue,
                "hedge_ratio": model.hedge_ratio,
                "half_life": model.half_life or float("nan"),
                "correlation": model.correlation,
            },
            stop_atr_multiple=2.5,
            reward_risk=1.5,
        )
