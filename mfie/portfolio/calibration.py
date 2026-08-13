r"""Turning a confidence score into a hit rate you can bet on.

The sizer needs :math:`p`, the probability this trade wins, because Kelly is
:math:`f^* = p - (1-p)/b` and every position size in the engine descends from
it. Until now that number came from

.. code:: python

    assumed_win_rate = 0.35 + 0.25 * confidence

which is a guess. It is a *reasonable* guess — it is monotonic and lives in a
plausible range — but it has never been checked against a single realised
trade, and it is the input the account balance is most sensitive to.

This module replaces the guess with a measurement, and is careful about the two
ways that measurement goes wrong.

**Small samples lie.** A strategy that won nine of its last ten trades has a
sample hit rate of 90% and has proved nothing. The fix is Beta-Binomial
credibility: treat the prior as :math:`\kappa` imaginary observations and let
the data outvote it only when there is enough of it.

.. math::

    p \sim \mathrm{Beta}(\alpha_0 + w,\; \beta_0 + l), \qquad
    \alpha_0 = \kappa p_0,\; \beta_0 = \kappa (1 - p_0)

    E[p] = \frac{\kappa p_0 + w}{\kappa + n}
         = z\,\hat p + (1 - z)\,p_0, \qquad z = \frac{n}{n + \kappa}

The posterior mean *is* a credibility-weighted blend — the shrinkage is not an
extra step bolted on, it falls out of the arithmetic. At :math:`\kappa = 40`,
ten trades move the estimate 20% of the way from prior to sample.

**Cells are thin.** Splitting by strategy *and* regime gives cells with three
trades in them. So the model is hierarchical: a cell shrinks toward its
strategy's pooled rate, which shrinks toward the confidence prior. A rarely
seen cell inherits its parent's behaviour instead of inventing its own.

.. code::

    p0(confidence)  ──►  strategy pool  ──►  strategy x regime cell
      (prior)              (level 1)              (level 2)

**Sizing uses the lower tail, not the mean.** ``estimate().lower`` is the 25th
percentile of the posterior. Two strategies both measuring 55% — one over 400
trades, one over 12 — size very differently, because the thin one's posterior
is wide and its lower bound is far below its mean. Uncertainty shrinks the bet
without anyone having to write a rule that says so.

The reliability table answers the question the confidence score has never had
to answer: *when this engine says 60%, how often does it win?*
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from mfie.config import CalibrationParams, get_params
from mfie.core.utils import clamp, get_logger, safe_div

log = get_logger(__name__)

GLOBAL_KEY = "__global__"


def prior_hit_rate(confidence: float, params: CalibrationParams | None = None) -> float:
    """The engine's original heuristic, kept as the prior it always was.

    With no trade history this reproduces the previous behaviour exactly, so
    turning calibration on never makes a cold-start engine act unpredictably.
    """
    p = params or get_params().calibration
    raw = p.prior_base + p.prior_slope * float(confidence)
    return float(clamp(raw, p.prior_floor, p.prior_ceiling))


@dataclass(frozen=True)
class TradeRecord:
    """One realised trade, reduced to what the calibrator needs."""

    strategy: str
    won: bool
    confidence: float = 0.5
    regime: str = "unknown"
    asset_class: str = "unknown"
    r_multiple: float = 0.0

    @property
    def cell(self) -> str:
        return f"{self.strategy}|{self.regime}"


@dataclass
class CalibratedHitRate:
    """A posterior over the hit rate, plus how much of it is actually data."""

    mean: float
    lower: float
    prior: float
    observations: int
    credibility: float          # z = n / (n + kappa); 0 = pure prior
    alpha: float
    beta: float
    source: str                 # which level of the hierarchy supplied it

    @property
    def spread(self) -> float:
        """Distance from the mean to the sizing quantile — the humility term."""
        return max(self.mean - self.lower, 0.0)

    def describe(self) -> str:
        if self.observations == 0:
            return f"{self.mean:.0%} (prior only, no realised trades)"
        return (
            f"{self.mean:.0%} [{self.lower:.0%} at the sizing quantile] "
            f"from {self.observations} trades, {self.credibility:.0%} credibility "
            f"({self.source})"
        )


def _beta_quantile(alpha: float, beta: float, q: float) -> float:
    """Inverse Beta CDF, with a normal fallback if SciPy is unavailable."""
    alpha = max(float(alpha), 1e-6)
    beta = max(float(beta), 1e-6)
    try:
        from scipy.stats import beta as beta_dist

        return float(clamp(float(beta_dist.ppf(q, alpha, beta)), 0.0, 1.0))
    except Exception:  # pragma: no cover - SciPy is a hard dependency
        mean = alpha / (alpha + beta)
        var = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1.0))
        # Normal approximation; adequate away from the 0/1 boundaries.
        z = {0.05: -1.645, 0.1: -1.282, 0.25: -0.674, 0.5: 0.0}.get(round(q, 2), -0.674)
        return float(clamp(mean + z * np.sqrt(max(var, 0.0)), 0.0, 1.0))


@dataclass
class _Cell:
    wins: int = 0
    losses: int = 0
    r_sum: float = 0.0

    @property
    def n(self) -> int:
        return self.wins + self.losses

    @property
    def rate(self) -> float:
        return safe_div(self.wins, self.n, 0.0)

    @property
    def avg_r(self) -> float:
        return safe_div(self.r_sum, self.n, 0.0)


class ConfidenceCalibrator:
    """Hierarchical Beta-Binomial calibration of the engine's hit rate.

    Fit it on realised trades — from the backtester, from the trade table, or
    from anywhere else that knows whether a signal worked — then ask it for a
    hit rate at decision time. Unfitted, it returns the prior and says so, so
    it is always safe to construct and always safe to call.
    """

    def __init__(self, params: CalibrationParams | None = None) -> None:
        self.params = params or get_params().calibration
        self._cells: dict[str, _Cell] = {}
        self._strategies: dict[str, _Cell] = {}
        self._global = _Cell()
        self._buckets: dict[int, _Cell] = {}
        self._brier_sum = 0.0
        self._brier_n = 0

    # ------------------------------------------------------------------ fit
    def fit(self, records: Iterable[TradeRecord]) -> ConfidenceCalibrator:
        for record in records:
            self.update(record)
        return self

    def update(self, record: TradeRecord) -> None:
        """Fold one realised trade in. Online, so a live book can keep learning."""
        for store, key in (
            (self._cells, record.cell),
            (self._strategies, record.strategy),
        ):
            cell = store.setdefault(key, _Cell())
            self._accumulate(cell, record)
        self._accumulate(self._global, record)

        bucket = self._bucket_index(record.confidence)
        self._accumulate(self._buckets.setdefault(bucket, _Cell()), record)

        # Brier score of the *prior*, i.e. how good the heuristic was on its own.
        forecast = prior_hit_rate(record.confidence, self.params)
        self._brier_sum += (forecast - (1.0 if record.won else 0.0)) ** 2
        self._brier_n += 1

    @staticmethod
    def _accumulate(cell: _Cell, record: TradeRecord) -> None:
        if record.won:
            cell.wins += 1
        else:
            cell.losses += 1
        cell.r_sum += float(record.r_multiple)

    def _bucket_index(self, confidence: float) -> int:
        n = max(self.params.reliability_buckets, 1)
        return int(clamp(int(float(confidence) * n), 0, n - 1))

    @property
    def observations(self) -> int:
        return self._global.n

    @property
    def fitted(self) -> bool:
        return self._global.n > 0

    # ------------------------------------------------------------- estimate
    def estimate(
        self,
        confidence: float,
        strategy: str | None = None,
        regime: str | None = None,
    ) -> CalibratedHitRate:
        """Posterior hit rate for a signal, walking down the hierarchy.

        Each level blends toward the level above it with weight
        :math:`z = n/(n+\\kappa)`, so a cell only overrides its parent to the
        extent it has the evidence to.
        """
        p = self.params
        kappa = max(p.prior_strength, 1e-6)
        prior = prior_hit_rate(confidence, p)

        estimate, source, observations = prior, "prior", 0

        # Level 0 -> 1: the whole book. Corrects a prior that is globally
        # optimistic or pessimistic before any per-strategy split is trusted.
        if self._global.n > 0:
            z = self._global.n / (self._global.n + kappa)
            estimate = z * self._global.rate + (1 - z) * estimate
            source, observations = "book", self._global.n

        # Level 1 -> 2: this strategy.
        strategy_cell = self._strategies.get(strategy or "", _Cell())
        if strategy_cell.n >= p.min_cell_observations:
            z = strategy_cell.n / (strategy_cell.n + kappa)
            estimate = z * strategy_cell.rate + (1 - z) * estimate
            source, observations = "strategy", strategy_cell.n

        # Level 2 -> 3: this strategy in this regime.
        cell = self._cells.get(f"{strategy}|{regime}", _Cell())
        if cell.n >= p.min_cell_observations:
            z = cell.n / (cell.n + kappa)
            estimate = z * cell.rate + (1 - z) * estimate
            source, observations = "strategy+regime", cell.n

        estimate = float(clamp(estimate, 0.01, 0.99))

        # Rebuild a posterior with the blended mean and the effective sample
        # size, so the width of the interval reflects how much data the final
        # number actually rests on.
        n_eff = observations + kappa
        alpha = estimate * n_eff
        beta = (1.0 - estimate) * n_eff
        lower = _beta_quantile(alpha, beta, p.sizing_quantile)

        return CalibratedHitRate(
            mean=estimate,
            lower=float(min(lower, estimate)),
            prior=prior,
            observations=int(observations),
            credibility=float(observations / (observations + kappa)),
            alpha=float(alpha),
            beta=float(beta),
            source=source,
        )

    # ---------------------------------------------------------- diagnostics
    def reliability_table(self) -> pd.DataFrame:
        """Does a stated confidence mean anything?

        One row per confidence bucket: what the engine claimed, what actually
        happened, and how many trades that rests on. A well-calibrated engine
        produces a monotonically rising ``realised`` column. Most do not, and
        seeing that is the point — an uncalibrated confidence score is a
        number that feels like a probability and is not one.
        """
        if not self._buckets:
            return pd.DataFrame(
                columns=["confidence_range", "forecast", "realised", "trades", "avg_r"]
            )

        n = max(self.params.reliability_buckets, 1)
        rows = []
        for index in sorted(self._buckets):
            cell = self._buckets[index]
            low, high = index / n, (index + 1) / n
            rows.append(
                {
                    "confidence_range": f"{low:.0%}-{high:.0%}",
                    "forecast": prior_hit_rate((low + high) / 2, self.params),
                    "realised": cell.rate,
                    "trades": cell.n,
                    "avg_r": cell.avg_r,
                }
            )
        return pd.DataFrame(rows)

    def brier_score(self) -> float:
        """Mean squared error of the prior's probability forecasts.

        0 is perfect, 0.25 is a coin flip. Compare against the *base rate*
        Brier score below: a forecast that cannot beat "always predict the
        overall hit rate" carries no information, however intuitive it looks.
        """
        return safe_div(self._brier_sum, self._brier_n, 0.0)

    def baseline_brier_score(self) -> float:
        base = self._global.rate
        if self._global.n == 0:
            return 0.0
        return float(base * (1 - base) ** 2 + (1 - base) * base**2)

    def skill_score(self) -> float:
        """Brier skill: 1 - BS/BS_base. Positive means the score adds information."""
        baseline = self.baseline_brier_score()
        if baseline <= 0:
            return 0.0
        return float(1.0 - self.brier_score() / baseline)

    def strategy_table(self) -> pd.DataFrame:
        """Per-strategy posteriors — which parts of the library have earned size."""
        if not self._strategies:
            return pd.DataFrame(columns=["strategy", "trades", "raw", "posterior", "avg_r"])
        rows = []
        for name, cell in sorted(self._strategies.items()):
            posterior = self.estimate(0.5, strategy=name)
            rows.append(
                {
                    "strategy": name,
                    "trades": cell.n,
                    "raw": cell.rate,
                    "posterior": posterior.mean,
                    "avg_r": cell.avg_r,
                }
            )
        return pd.DataFrame(rows).sort_values("posterior", ascending=False)

    def summary(self) -> dict[str, object]:
        return {
            "observations": self.observations,
            "hit_rate": self._global.rate,
            "avg_r": self._global.avg_r,
            "brier": self.brier_score(),
            "brier_baseline": self.baseline_brier_score(),
            "skill": self.skill_score(),
            "cells": len(self._cells),
            "strategies": len(self._strategies),
        }


# --------------------------------------------------------------------------- #
# Adapters — realised trades come from several places
# --------------------------------------------------------------------------- #
def records_from_backtest(result, regime: str = "unknown") -> list[TradeRecord]:
    """Build calibration records from a ``BacktestResult``.

    The R multiple is recovered exactly, without needing to know the account
    equity at entry: the capital risked on a trade is ``units x |entry - stop|``
    by construction of the sizer, so ``pnl / risk_capital`` is the R multiple.
    """
    records: list[TradeRecord] = []
    for trade in getattr(result, "trades", []):
        if getattr(trade, "exit_ts", None) is None:
            continue
        risk_capital = abs(trade.units) * abs(trade.entry_price - trade.stop)
        r_multiple = safe_div(trade.pnl, risk_capital, 0.0)
        records.append(
            TradeRecord(
                strategy=trade.strategy,
                won=bool(trade.pnl > 0),
                confidence=float(getattr(trade, "confidence", 0.5)),
                regime=str(getattr(trade, "regime", regime)),
                asset_class=str(getattr(trade, "asset_class", "unknown")),
                r_multiple=r_multiple,
            )
        )
    return records


def records_from_rows(rows: Sequence[dict]) -> list[TradeRecord]:
    """Build records from persisted trade rows (``TradeRow`` as dicts)."""
    records: list[TradeRecord] = []
    for row in rows:
        pnl = float(row.get("pnl") or 0.0)
        entry = float(row.get("entry_price") or 0.0)
        stop = float(row.get("stop") or 0.0)
        units = abs(float(row.get("units") or 0.0))
        risk_capital = units * abs(entry - stop)
        records.append(
            TradeRecord(
                strategy=str(row.get("strategy") or "unknown"),
                won=pnl > 0,
                confidence=float(row.get("confidence") or 0.5),
                regime=str(row.get("regime") or "unknown"),
                asset_class=str(row.get("asset_class") or "unknown"),
                r_multiple=safe_div(pnl, risk_capital, 0.0),
            )
        )
    return records


def load_calibrator(
    params: CalibrationParams | None = None,
    rows: Sequence[dict] | None = None,
) -> ConfidenceCalibrator:
    """Best-effort calibrator: persisted trades if there are any, else the prior.

    Never raises. A missing database, an empty trade table or a schema from an
    older version all land in the same place — an unfitted calibrator that
    returns the prior and reports ``source="prior"`` so the audit trail is
    honest about it.
    """
    calibrator = ConfidenceCalibrator(params)
    if rows is not None:
        return calibrator.fit(records_from_rows(rows))
    try:
        from mfie.storage.repo import Repository

        loaded = Repository().closed_trades()
    except Exception as exc:
        log.debug("No persisted trades for calibration: %s", exc)
        return calibrator
    return calibrator.fit(records_from_rows(loaded))
