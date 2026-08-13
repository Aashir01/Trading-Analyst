r"""Measuring the risk of a *book*, not a list of trades.

``RiskParams.max_portfolio_risk`` caps the sum of per-trade risk fractions.
Six crypto longs at 1% each sum to 6% and the cap is satisfied. In a
liquidation those six positions do not lose 1% each at different times — they
are one bet on one factor, and the book loses something close to 6% at once.
Additive heat is not a measure of portfolio risk. It is a count.

The measure that is:

.. math::

    \sigma_{book} = \sqrt{\,r^{\mathsf T} C\, r\,}

where :math:`r` is the vector of per-trade risk fractions and :math:`C` the
correlation matrix. For six perfectly correlated trades this returns 6% — the
cap bites, correctly. For six independent ones it returns
:math:`0.01\sqrt 6 = 2.4\%`, and the book has room for more risk that additive
heat would have refused. The measure tightens and loosens in the right
directions, which is why it is worth more than a stricter fixed cap.

**Correlations are signed by direction.** Long BTC and long ETH are one bet;
long BTC and *short* ETH are a spread, and its risk is far lower than either
leg. So the matrix is built as

.. math::  \tilde C_{ij} = d_i d_j\, \rho_{ij}

with :math:`d \in \{+1,-1\}`. Hedges then net off inside
:math:`\sqrt{r^{\mathsf T} \tilde C r}` automatically, and a book that is
genuinely market-neutral is allowed to be much larger than a directional one.
This single sign is the difference between a risk model that understands a
pairs trade and one that double-counts it.

**Unknown correlation is not zero.** Assuming independence when the data is
too short is the single most expensive assumption in portfolio construction.
Missing pairs get ``default_correlation``, a deliberately pessimistic prior.

The clustering is the Hierarchical Risk Parity distance,
:math:`d_{ij} = \sqrt{\tfrac{1}{2}(1 - \rho_{ij})}`, which is a true metric on
correlations, cut at a threshold to give "these are the same bet" groups.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mfie.core.utils import clamp, get_logger, safe_div

log = get_logger(__name__)


@dataclass
class CorrelationView:
    """The signed correlation matrix actually used, and how much of it is real."""

    matrix: pd.DataFrame
    observations: int
    estimated_pairs: int = 0      # pairs filled with the prior, not measured
    total_pairs: int = 0

    @property
    def coverage(self) -> float:
        """Fraction of pairs backed by data rather than by the prior."""
        if self.total_pairs <= 0:
            return 1.0
        return 1.0 - self.estimated_pairs / self.total_pairs


def returns_matrix(frames: dict[str, pd.DataFrame], lookback: int = 250) -> pd.DataFrame:
    """Aligned log-return matrix from the engine's OHLCV frames.

    Log returns, because they aggregate cleanly across time and their
    correlations are the ones that describe co-movement of *compounding*
    positions.
    """
    series: dict[str, pd.Series] = {}
    for symbol, frame in frames.items():
        if frame is None or frame.empty or "close" not in frame:
            continue
        close = pd.Series(frame["close"]).astype(float).tail(lookback + 1)
        if len(close) < 10:
            continue
        series[symbol] = np.log(close).diff().dropna()
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).sort_index()


def signed_correlation(
    returns: pd.DataFrame,
    positions: dict[str, tuple[str, int]],
    default_correlation: float = 0.35,
    min_observations: int = 60,
) -> CorrelationView:
    """Correlation matrix with each entry signed by the two trades' directions.

    ``positions`` maps a unique position key to ``(symbol, direction)``. Keying
    on the position rather than the symbol matters: two strategies can fire on
    the same instrument, and those two positions are perfectly correlated —
    they belong in one cluster and must not be counted as diversification.

    Pairs without ``min_observations`` of overlapping history fall back to
    ``default_correlation`` rather than to zero, and the count of such pairs is
    reported so the caller can say how much of the risk number is measured.

    Off-diagonal entries are clamped just inside +/-1 so the matrix stays
    positive *definite* rather than merely semi-definite. Two positions on the
    same instrument therefore read 0.999 rather than exactly 1.0 — near enough
    to cluster them together, far enough to keep the linear algebra stable.
    """
    keys = list(positions)
    if not keys:
        return CorrelationView(pd.DataFrame(dtype=float), 0)

    matrix = pd.DataFrame(np.eye(len(keys)), index=keys, columns=keys, dtype=float)
    if len(keys) == 1:
        return CorrelationView(matrix, int(len(returns)), 0, 0)

    usable = returns if returns is not None and not returns.empty else pd.DataFrame()
    symbols = sorted({sym for sym, _ in positions.values() if sym in usable.columns})
    if symbols:
        frame = usable[symbols].dropna(how="all")
        raw = frame.corr(min_periods=min_observations)
        observations = int(len(frame))
    else:
        raw, observations = pd.DataFrame(dtype=float), 0

    estimated = 0
    for i, a in enumerate(keys):
        sym_a, dir_a = positions[a]
        for b in keys[i + 1:]:
            sym_b, dir_b = positions[b]
            if sym_a == sym_b:
                rho = 1.0                      # same instrument, same risk
            else:
                value = (
                    raw.loc[sym_a, sym_b]
                    if sym_a in raw.index and sym_b in raw.columns
                    else np.nan
                )
                if value is None or not np.isfinite(value):
                    rho, estimated = default_correlation, estimated + 1
                else:
                    rho = float(value)
            signed = float(clamp(rho, -0.999, 0.999)) * dir_a * dir_b
            matrix.loc[a, b] = matrix.loc[b, a] = signed

    pairs = len(keys) * (len(keys) - 1) // 2
    return CorrelationView(
        matrix=matrix,
        observations=observations,
        estimated_pairs=estimated,
        total_pairs=pairs,
    )


def effective_risk(risks: pd.Series, correlation: pd.DataFrame) -> float:
    r"""The book's risk, :math:`\sqrt{r^{\mathsf T} C r}`.

    Never exceeds the additive sum, equals it only when everything is
    perfectly correlated, and can fall well below it when the book contains
    genuine hedges.
    """
    common = [s for s in risks.index if s in correlation.index]
    if not common:
        return float(max(risks.sum(), 0.0))
    r = risks[common].astype(float).to_numpy()
    c = correlation.loc[common, common].astype(float).to_numpy()
    variance = float(r @ c @ r)
    return float(np.sqrt(max(variance, 0.0)))


def diversification_ratio(risks: pd.Series, correlation: pd.DataFrame) -> float:
    """Additive heat divided by effective risk.

    1.0 means the book is one trade wearing several names. Above 2.0 means the
    positions genuinely offset, and the risk budget can safely carry more of
    them — this is the number that decides whether diversification is real or
    decorative.
    """
    gross = float(risks.astype(float).abs().sum())
    effective = effective_risk(risks, correlation)
    return safe_div(gross, effective, 1.0)


def marginal_risk_contributions(risks: pd.Series, correlation: pd.DataFrame) -> pd.Series:
    r"""Each position's share of total book risk.

    :math:`MRC_i = r_i (C r)_i / \sigma_{book}`, which sums exactly to
    :math:`\sigma_{book}` by Euler's theorem on homogeneous functions. The
    interesting rows are the ones whose contribution far exceeds their size:
    those are the positions the book is quietly concentrated in.
    """
    common = [s for s in risks.index if s in correlation.index]
    if not common:
        return pd.Series(dtype=float)
    r = risks[common].astype(float).to_numpy()
    c = correlation.loc[common, common].astype(float).to_numpy()
    sigma = np.sqrt(max(float(r @ c @ r), 0.0))
    if sigma <= 0:
        return pd.Series(0.0, index=common)
    return pd.Series((r * (c @ r)) / sigma, index=common)


def cluster_by_correlation(
    correlation: pd.DataFrame,
    threshold: float = 0.55,
) -> dict[str, int]:
    r"""Group positions that are, for risk purposes, the same trade.

    Uses the Hierarchical Risk Parity metric
    :math:`d_{ij} = \sqrt{(1 - \rho_{ij})/2}` with single linkage, cut where
    correlation falls below ``threshold``. Single linkage on purpose: risk
    contagion is transitive. If A moves with B and B moves with C, then A and C
    belong in the same risk group even when their direct correlation is
    unremarkable — that chain is exactly how a "diversified" book turns out to
    have been one position.
    """
    symbols = list(correlation.index)
    if len(symbols) <= 1:
        return {s: 0 for s in symbols}

    rho = correlation.to_numpy(dtype=float)
    distance = np.sqrt(np.clip((1.0 - rho) / 2.0, 0.0, 1.0))
    cut = float(np.sqrt(max((1.0 - threshold) / 2.0, 0.0)))

    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        condensed = squareform(distance, checks=False)
        labels = fcluster(linkage(condensed, method="single"), t=cut, criterion="distance")
        return {symbol: int(label) for symbol, label in zip(symbols, labels)}
    except Exception as exc:  # pragma: no cover - SciPy is a hard dependency
        log.debug("SciPy clustering unavailable (%s); using union-find fallback", exc)

    # Union-find over the same cut — identical single-linkage semantics.
    parent = {s: s for s in symbols}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(symbols):
        for j in range(i + 1, len(symbols)):
            if distance[i, j] <= cut:
                ra, rb = find(a), find(symbols[j])
                if ra != rb:
                    parent[ra] = rb

    roots = {}
    labels = {}
    for symbol in symbols:
        root = find(symbol)
        labels[symbol] = roots.setdefault(root, len(roots) + 1)
    return labels


@dataclass
class BookRisk:
    """Diagnostics for a proposed set of positions."""

    gross_risk: float                 # additive heat, the old measure
    effective_risk: float             # sqrt(r' C r), the real one
    diversification_ratio: float
    contributions: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    clusters: dict[str, int] = field(default_factory=dict)
    correlation_coverage: float = 1.0
    observations: int = 0

    @property
    def concentration(self) -> float:
        """Largest single share of book risk. 1.0 is a one-position book."""
        total = float(self.contributions.sum())
        if total <= 0 or self.contributions.empty:
            return 0.0
        return float(self.contributions.max() / total)

    def describe(self) -> str:
        return (
            f"gross {self.gross_risk:.2%} -> effective {self.effective_risk:.2%} "
            f"(diversification {self.diversification_ratio:.2f}x, "
            f"{len(set(self.clusters.values()))} independent clusters, "
            f"largest position {self.concentration:.0%} of book risk)"
        )


def assess_book(
    risks: pd.Series,
    correlation_view: CorrelationView,
    cluster_threshold: float = 0.55,
) -> BookRisk:
    """Full risk picture for a candidate book."""
    matrix = correlation_view.matrix
    return BookRisk(
        gross_risk=float(risks.astype(float).abs().sum()),
        effective_risk=effective_risk(risks, matrix),
        diversification_ratio=diversification_ratio(risks, matrix),
        contributions=marginal_risk_contributions(risks, matrix),
        clusters=cluster_by_correlation(matrix, cluster_threshold),
        correlation_coverage=correlation_view.coverage,
        observations=correlation_view.observations,
    )
