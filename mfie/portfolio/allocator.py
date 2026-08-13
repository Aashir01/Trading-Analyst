r"""Deciding what the book actually holds.

Everything upstream of this module judges trades one at a time. The filter
chain never sees two signals together, and the sizer is handed one signal and
a running total of risk already committed. That running total is the only
thing standing between the engine and a book of eight positions that are all
the same trade — and it is a *sum*, so it cannot tell the difference between
eight independent bets and one bet placed eight times.

This is the layer that sees them together. It runs once per analysis, after
every signal has been scored, and does four things in order:

1. **Rejects trades whose arithmetic does not work.** Expected value net of
   costs, from ``mfie.portfolio.edge``. A signal that survived the whole
   econometric chain and still cannot clear its breakeven hit rate is dropped
   here, with the numbers in the audit trail.

2. **Finds the clones.** Correlation clustering on signed correlations, so
   longs that move together group up and genuine hedges do not.

3. **Discounts repetition inside a cluster.** The best signal in a cluster
   keeps its size; the :math:`k`-th keeps :math:`1/(1 + k\lambda)`. Confirming
   evidence for a view you already hold is worth something — but far less than
   the first piece, and additive heat prices it at full value.

4. **Fits the book to a diversified budget.** Scaling by
   :math:`\sqrt{r^{\mathsf T}Cr}` rather than :math:`\sum r`, which cuts a
   concentrated book harder than the old cap ever did and lets a genuinely
   uncorrelated one carry more.

Step 4 is the one that makes money rather than merely saving it. The additive
cap has to be set low enough to survive the worst case — everything correlated
— so it under-allocates in every other case. Measuring correlation directly
means the budget can be set where it belongs and still tighten automatically
when the book concentrates.

The whole layer is auditable: every position carries the reason its size
changed, and ``PortfolioPlan.explain()`` prints the chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from mfie.config import Params, get_params
from mfie.core.types import FilterAction, FilterOutcome, ScoredSignal
from mfie.core.utils import get_logger, safe_div
from mfie.portfolio.construction import (
    BookRisk,
    CorrelationView,
    assess_book,
    cluster_by_correlation,
    effective_risk,
    signed_correlation,
)
from mfie.portfolio.edge import EdgeEstimate

log = get_logger(__name__)


@dataclass
class Allocation:
    """One position as the portfolio layer left it."""

    signal: ScoredSignal
    edge: EdgeEstimate | None = None
    standalone_risk: float = 0.0     # what per-trade sizing asked for
    risk_fraction: float = 0.0       # what the book can actually afford
    units: float = 0.0
    cluster: int = 0
    cluster_rank: int = 0            # 0 = the best idea in its cluster
    risk_contribution: float = 0.0   # share of total book risk
    dropped: bool = False
    reasons: list[str] = field(default_factory=list)

    @property
    def symbol(self) -> str:
        return self.signal.instrument.symbol

    @property
    def key(self) -> str:
        return f"{self.symbol}:{self.signal.raw.strategy}"

    @property
    def scale(self) -> float:
        """How much of its standalone size the position kept."""
        return safe_div(self.risk_fraction, self.standalone_risk, 0.0)

    def describe(self) -> str:
        head = (
            f"{self.symbol:<10} {self.signal.direction.value:<5} "
            f"{self.signal.raw.strategy:<20} "
            f"{self.standalone_risk:>6.2%} -> {self.risk_fraction:>6.2%}"
        )
        if self.dropped:
            return f"{head}  DROPPED"
        return f"{head}  (cluster {self.cluster}, {self.risk_contribution:.0%} of book risk)"


@dataclass
class PortfolioPlan:
    """The allocation decision, with everything needed to argue with it."""

    allocations: list[Allocation] = field(default_factory=list)
    book: BookRisk | None = None
    correlation: CorrelationView | None = None
    gross_requested: float = 0.0     # additive heat before the portfolio pass
    scale_applied: float = 1.0
    notes: list[str] = field(default_factory=list)
    calibration_source: str = "prior"
    calibration_observations: int = 0

    @property
    def held(self) -> list[Allocation]:
        return [a for a in self.allocations if not a.dropped and a.risk_fraction > 0]

    @property
    def dropped(self) -> list[Allocation]:
        return [a for a in self.allocations if a.dropped or a.risk_fraction <= 0]

    @property
    def gross_risk(self) -> float:
        return float(sum(a.risk_fraction for a in self.held))

    @property
    def effective_risk(self) -> float:
        return self.book.effective_risk if self.book else self.gross_risk

    @property
    def diversification_ratio(self) -> float:
        return self.book.diversification_ratio if self.book else 1.0

    @property
    def expected_r(self) -> float:
        """Book-level expected value, weighted by the risk actually committed."""
        total = self.gross_risk
        if total <= 0:
            return 0.0
        return float(
            sum(a.risk_fraction * (a.edge.expected_r if a.edge else 0.0) for a in self.held)
            / total
        )

    def summary(self) -> dict[str, object]:
        return {
            "positions": len(self.held),
            "dropped": len(self.dropped),
            "gross_requested": self.gross_requested,
            "gross_risk": self.gross_risk,
            "effective_risk": self.effective_risk,
            "diversification_ratio": self.diversification_ratio,
            "clusters": len({a.cluster for a in self.held}),
            "expected_r": self.expected_r,
            "scale_applied": self.scale_applied,
            "calibration": f"{self.calibration_source} ({self.calibration_observations} trades)",
            "correlation_coverage": self.book.correlation_coverage if self.book else 1.0,
        }

    def explain(self) -> list[str]:
        lines: list[str] = []
        if self.book is not None:
            lines.append(self.book.describe())
        lines.extend(self.notes)
        for allocation in self.allocations:
            lines.append(allocation.describe())
            lines.extend(f"    - {reason}" for reason in allocation.reasons)
        return lines


class PortfolioAllocator:
    """Turns a list of independently sized signals into a coherent book."""

    def __init__(self, params: Params | None = None) -> None:
        self.params = params or get_params()

    def allocate(
        self,
        candidates: list[tuple[ScoredSignal, EdgeEstimate | None]],
        returns: pd.DataFrame | None = None,
        equity: float | None = None,
    ) -> PortfolioPlan:
        p = self.params.portfolio
        equity = equity if equity is not None else self.params.risk.account_equity
        plan = PortfolioPlan()

        allocations = [
            Allocation(
                signal=signal,
                edge=edge,
                standalone_risk=float(signal.risk_fraction),
                risk_fraction=float(signal.risk_fraction),
                units=float(signal.units),
            )
            for signal, edge in candidates
        ]
        plan.allocations = allocations
        plan.gross_requested = float(sum(a.standalone_risk for a in allocations))

        # --- 1. expected value ---------------------------------------------
        for allocation in allocations:
            edge = allocation.edge
            if edge is None:
                continue
            if not edge.tradable:
                self._drop(allocation, edge.reason)
            elif edge.kelly < allocation.risk_fraction:
                # Kelly is a ceiling, not a target. Sizing already applies a
                # quarter-Kelly cap on an assumed hit rate; this re-applies it
                # on the calibrated one, which is usually the tighter of the two.
                allocation.reasons.append(
                    f"capped at Kelly {edge.kelly:.2%} on a calibrated "
                    f"{edge.hit_rate.lower:.0%} hit rate ({edge.hit_rate.source})"
                )
                allocation.risk_fraction = float(edge.kelly)

        live = [a for a in allocations if not a.dropped and a.risk_fraction > 0]
        if not live:
            plan.notes.append("No candidate survived the expected-value gate.")
            return plan

        # --- 2. rank, then cluster -----------------------------------------
        # Ranking by expected value per unit of risk, not by confidence: a
        # 0.9-confidence signal with a 0.4R cost drag is a worse use of the
        # budget than a 0.6-confidence one that is cheap to hold.
        live.sort(key=lambda a: a.edge.expected_r_lower if a.edge else 0.0, reverse=True)

        if len(live) > p.max_positions:
            for allocation in live[p.max_positions:]:
                self._drop(
                    allocation,
                    f"beyond the {p.max_positions}-position limit "
                    "(ranked by expected value per unit of risk)",
                )
            live = live[: p.max_positions]

        positions = {
            a.key: (a.symbol, a.signal.direction.sign or 1)
            for a in live
        }
        correlation = signed_correlation(
            returns if returns is not None else pd.DataFrame(),
            positions,
            default_correlation=p.default_correlation,
            min_observations=p.min_correlation_observations,
        )
        plan.correlation = correlation
        clusters = cluster_by_correlation(correlation.matrix, p.cluster_threshold)

        if correlation.estimated_pairs:
            plan.notes.append(
                f"{correlation.estimated_pairs}/{correlation.total_pairs} correlations "
                f"assumed at {p.default_correlation:.2f} for want of overlapping history — "
                "unknown correlation is treated as positive, not zero"
            )

        # --- 3. redundancy decay within clusters ---------------------------
        seen: dict[int, int] = {}
        for allocation in live:            # already in edge order
            label = clusters.get(allocation.key, 0)
            rank = seen.get(label, 0)
            seen[label] = rank + 1
            allocation.cluster = int(label)
            allocation.cluster_rank = rank
            if rank > 0 and p.redundancy_decay > 0:
                decay = 1.0 / (1.0 + rank * p.redundancy_decay)
                allocation.risk_fraction *= decay
                allocation.reasons.append(
                    f"redundancy decay {decay:.2f}x — idea #{rank + 1} in a cluster of "
                    f"{sum(1 for k, v in clusters.items() if v == label)} correlated positions"
                )

        # --- 4. per-cluster cap --------------------------------------------
        for label in set(clusters.values()):
            members = [a for a in live if a.cluster == label]
            total = sum(a.risk_fraction for a in members)
            if total > p.max_cluster_risk > 0:
                scale = p.max_cluster_risk / total
                for allocation in members:
                    allocation.risk_fraction *= scale
                    allocation.reasons.append(
                        f"cluster cap {scale:.2f}x — the group wanted {total:.2%} "
                        f"against a {p.max_cluster_risk:.2%} per-cluster limit"
                    )

        # --- 5. fit the whole book to the diversified budget ---------------
        risks = pd.Series({a.key: a.risk_fraction for a in live}, dtype=float)
        gross_before = float(risks.sum())
        current = effective_risk(risks, correlation.matrix)
        if current > p.max_effective_risk > 0:
            # Effective risk is homogeneous of degree 1 in the weights, so a
            # single uniform scale lands exactly on the budget — no iteration.
            scale = p.max_effective_risk / current
            plan.scale_applied = float(scale)
            for allocation in live:
                allocation.risk_fraction *= scale
            risks *= scale
            plan.notes.append(
                f"Book scaled {scale:.2f}x: correlated risk was {current:.2%} against a "
                f"{p.max_effective_risk:.2%} budget (additive heat read {gross_before:.2%}, "
                "which is the number that would have missed it)"
            )
        else:
            headroom = p.max_effective_risk - current
            if headroom > 0:
                plan.notes.append(
                    f"Correlated risk {current:.2%} of a {p.max_effective_risk:.2%} budget — "
                    f"{headroom:.2%} of headroom remains"
                )

        # --- 6. finalise ----------------------------------------------------
        plan.book = assess_book(risks, correlation, p.cluster_threshold)
        contributions = plan.book.contributions
        total_contribution = float(contributions.sum()) if not contributions.empty else 0.0

        for allocation in live:
            if allocation.risk_fraction <= 0:
                self._drop(allocation, "sized out entirely by the portfolio budget")
                continue
            allocation.risk_contribution = safe_div(
                float(contributions.get(allocation.key, 0.0)), total_contribution, 0.0
            )
            allocation.units = self._units(allocation, equity)

        return plan

    # ------------------------------------------------------------------ util
    @staticmethod
    def _drop(allocation: Allocation, reason: str) -> None:
        allocation.dropped = True
        allocation.risk_fraction = 0.0
        allocation.units = 0.0
        allocation.reasons.append(reason)

    @staticmethod
    def _units(allocation: Allocation, equity: float) -> float:
        """Re-derive units from the final risk, keeping the sizing invariant.

        Risk is defined by the stop distance, so units must be recomputed from
        the allocated risk rather than scaled from the standalone units — the
        two agree only when the stop has not moved, and the microstructure
        filter moves it.
        """
        signal = allocation.signal
        stop = signal.adjusted_stop or signal.raw.stop
        distance = abs(signal.raw.entry - stop)
        if distance <= 0 or equity <= 0:
            return 0.0
        return float(equity * allocation.risk_fraction / distance)


def apply_plan(plan: PortfolioPlan) -> PortfolioPlan:
    """Write the allocation back onto the signals, with an audit entry.

    The portfolio layer is the last word on size, so the ``ScoredSignal``
    objects the rest of the engine renders must carry its numbers rather than
    the standalone ones. A ``portfolio`` outcome is appended to each audit
    trail explaining what changed and why, and a position sized to zero is
    marked blocked so it drops out of ``AnalysisResult.tradable`` the same way
    a vetoed signal does.
    """
    for allocation in plan.allocations:
        signal = allocation.signal
        signal.risk_fraction = allocation.risk_fraction
        signal.units = allocation.units
        equity = get_params().risk.account_equity
        signal.size_fraction = (
            float(allocation.units * signal.raw.entry / equity) if equity > 0 else 0.0
        )

        if allocation.dropped or allocation.risk_fraction <= 0:
            signal.blocked = True
            signal.block_reasons.extend(allocation.reasons or ["dropped by portfolio allocation"])
            action, multiplier = FilterAction.BLOCK, 0.0
        elif allocation.scale < 0.999:
            action, multiplier = FilterAction.PENALIZE, allocation.scale
        else:
            action, multiplier = FilterAction.PASS, 1.0

        detail: dict[str, object] = {
            "standalone_risk": allocation.standalone_risk,
            "allocated_risk": allocation.risk_fraction,
            "scale": allocation.scale,
            "cluster": allocation.cluster,
            "cluster_rank": allocation.cluster_rank,
            "risk_contribution": allocation.risk_contribution,
        }
        if allocation.edge is not None:
            detail.update(allocation.edge.as_dict())

        signal.outcomes.append(
            FilterOutcome(
                name="portfolio",
                action=action,
                multiplier=multiplier,
                reason="; ".join(allocation.reasons) if allocation.reasons
                else f"Full size held: {allocation.risk_fraction:.2%} of equity at risk",
                detail=detail,
            )
        )
    return plan
