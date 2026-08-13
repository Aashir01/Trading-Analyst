r"""Is this trade worth taking at all?

Every filter in ``mfie.pipeline.filters`` answers a *relative* question: does
the macro environment favour this setup more or less than usual. None of them
answers the absolute one — after costs, does this trade have positive expected
value? A signal can clear the entire econometric chain with a confidence of
0.7 and still be a slow way to lose money, because confidence measures
agreement between models, not profit.

The arithmetic is small and unforgiving. Winning pays :math:`b - c` and losing
costs :math:`1 + c`, both in R:

.. math::

    E[R] = p\,(b - c) - (1-p)\,(1 + c)

and the trade is worth taking only when that exceeds a hurdle. The breakeven
probability :math:`p^* = (1+c)/(b+1)` is the number to look at: on a 2R target
with 0.25R of costs it is 42%, not the 33% the chart implies.

**Which p?** Not the posterior mean. Kelly is a ratio of estimated quantities
and overbets systematically when those estimates are noisy — the penalty for
being 10 points too optimistic about your hit rate is much larger than the
reward for being 10 points too pessimistic, because the equity curve compounds
multiplicatively and drawdowns are geometric. So sizing uses the lower tail of
the posterior from ``mfie.portfolio.calibration``. A strategy with 400 trades
behind it and one with 12 can report the same mean and get very different
sizes, which is the correct behaviour and is not obtainable from a point
estimate at all.

The gate is deliberately the *last* thing that speaks. It sits after the macro
chain because it needs the chain's confidence as its input — and because a
trade that macro loves and arithmetic rejects should be rejected, loudly, with
the numbers attached.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mfie.config import Params, get_params
from mfie.core.utils import clamp, safe_div
from mfie.portfolio.calibration import CalibratedHitRate
from mfie.portfolio.costs import TradeCosts, breakeven_hit_rate


@dataclass
class EdgeEstimate:
    """The full expected-value case for one trade, in R."""

    hit_rate: CalibratedHitRate
    reward_risk: float
    cost_r: float
    expected_r: float             # at the posterior mean — the honest central case
    expected_r_lower: float       # at the sizing quantile — what sizing may use
    breakeven_hit_rate: float
    edge_ratio: float             # p_lower / p* — above 1.0 the trade clears its bar
    kelly: float                  # uncertainty-shrunk, capped
    gross_expected_r: float       # before costs, to show what costs took
    tradable: bool = True
    reason: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def cost_drag(self) -> float:
        """How much expected value the round trip consumed, in R."""
        return self.gross_expected_r - self.expected_r

    def describe(self) -> str:
        return (
            f"E[R] {self.expected_r:+.3f} (lower {self.expected_r_lower:+.3f}), "
            f"needs {self.breakeven_hit_rate:.0%} to break even, "
            f"calibrated {self.hit_rate.mean:.0%}, costs {self.cost_r:.3f}R"
        )

    def as_dict(self) -> dict[str, float | str | int]:
        return {
            "hit_rate": self.hit_rate.mean,
            "hit_rate_lower": self.hit_rate.lower,
            "hit_rate_source": self.hit_rate.source,
            "observations": self.hit_rate.observations,
            "reward_risk": self.reward_risk,
            "cost_r": self.cost_r,
            "expected_r": self.expected_r,
            "expected_r_lower": self.expected_r_lower,
            "breakeven_hit_rate": self.breakeven_hit_rate,
            "edge_ratio": self.edge_ratio,
            "kelly": self.kelly,
            "cost_drag": self.cost_drag,
        }


def expected_r(hit_rate: float, reward_risk: float, cost_r: float = 0.0) -> float:
    r"""Expected value of the trade in R: :math:`p(b-c) - (1-p)(1+c)`.

    ``cost_r`` may be negative when carry pays more than friction costs, in
    which case it correctly *raises* expected value on both branches.
    """
    p = clamp(float(hit_rate), 0.0, 1.0)
    b = float(reward_risk)
    c = float(cost_r)
    return float(p * (b - c) - (1.0 - p) * (1.0 + c))


def kelly_with_uncertainty(
    hit_rate: CalibratedHitRate,
    reward_risk: float,
    cost_r: float = 0.0,
    cap: float = 0.25,
) -> float:
    r"""Kelly on the posterior's lower tail, net of costs.

    Two departures from the textbook :math:`f^* = p - (1-p)/b`:

    1. The payoff is the *net* one, :math:`b_{net} = (b - c)/(1 + c)`. Costs
       shrink the win and enlarge the loss, and the ratio of the two is what
       Kelly actually consumes.
    2. :math:`p` is the sizing quantile of the posterior rather than its mean.
       This is the practical form of "bet less when you know less": the
       adjustment is large for a thin cell and vanishes as evidence
       accumulates, with no separate rule governing the transition.

    Returns 0 when the lower bound does not clear breakeven — the correct
    answer to an edge you cannot demonstrate is no position, not a small one.
    """
    b = float(reward_risk)
    c = float(cost_r)
    denominator = 1.0 + c
    if b <= 0 or denominator <= 0:
        return 0.0
    net_b = (b - c) / denominator
    if net_b <= 0:
        return 0.0
    p = clamp(hit_rate.lower, 0.0, 1.0)
    f = p - (1.0 - p) / net_b
    return float(clamp(f, 0.0, cap))


def evaluate_edge(
    hit_rate: CalibratedHitRate,
    reward_risk: float,
    costs: TradeCosts | None = None,
    params: Params | None = None,
) -> EdgeEstimate:
    """Assemble the expected-value case and decide whether it clears the hurdle."""
    p = params or get_params()
    cost_r = float(costs.total_r) if costs is not None else 0.0
    b = float(reward_risk) if reward_risk and np.isfinite(reward_risk) else p.risk.reward_risk_target

    breakeven = breakeven_hit_rate(b, cost_r)
    mean_r = expected_r(hit_rate.mean, b, cost_r)
    lower_r = expected_r(hit_rate.lower, b, cost_r)
    gross_r = expected_r(hit_rate.mean, b, 0.0)
    kelly = kelly_with_uncertainty(hit_rate, b, cost_r, p.risk.kelly_fraction_cap)

    notes: list[str] = []
    if costs is not None and costs.notes:
        notes.extend(costs.notes)
    if cost_r < 0:
        notes.append("carry credit exceeds execution friction — costs are a net credit")
    elif cost_r > 0.20:
        notes.append(
            f"costs are {cost_r:.2f}R: the stop is tight relative to the spread, "
            "which raises the breakeven hit rate sharply"
        )

    hurdle = p.portfolio.min_expected_r
    tradable, reason = True, f"E[R] {mean_r:+.3f} clears the {hurdle:+.2f}R hurdle"

    if mean_r < hurdle:
        tradable = False
        reason = (
            f"E[R] {mean_r:+.3f}R below the {hurdle:+.2f}R hurdle — needs a "
            f"{breakeven:.0%} hit rate, calibration says {hit_rate.mean:.0%} "
            f"({hit_rate.source}, {hit_rate.observations} trades)"
        )
    elif kelly <= 0:
        tradable = False
        reason = (
            f"positive at the mean but not at the {p.calibration.sizing_quantile:.0%} "
            f"quantile ({hit_rate.lower:.0%} vs {breakeven:.0%} breakeven) — "
            "the edge is not distinguishable from noise"
        )

    return EdgeEstimate(
        hit_rate=hit_rate,
        reward_risk=b,
        cost_r=cost_r,
        expected_r=mean_r,
        expected_r_lower=lower_r,
        breakeven_hit_rate=breakeven,
        edge_ratio=safe_div(hit_rate.lower, breakeven, 0.0),
        kelly=kelly,
        gross_expected_r=gross_r,
        tradable=tradable,
        reason=reason,
        notes=notes,
    )
