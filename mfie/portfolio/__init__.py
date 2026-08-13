"""The capital allocation layer: costs, calibration, expected value, the book.

Everything below this package judges one trade at a time. This package is
where the engine stops asking "is this setup good?" and starts asking "is this
setup worth capital, at this size, given everything else I am already holding,
after what it costs to hold it?" — which is the question the account balance
actually answers to.
"""

from mfie.portfolio.allocator import (
    Allocation,
    PortfolioAllocator,
    PortfolioPlan,
    apply_plan,
)
from mfie.portfolio.calibration import (
    CalibratedHitRate,
    ConfidenceCalibrator,
    TradeRecord,
    load_calibrator,
    prior_hit_rate,
    records_from_backtest,
    records_from_rows,
)
from mfie.portfolio.construction import (
    BookRisk,
    CorrelationView,
    assess_book,
    cluster_by_correlation,
    diversification_ratio,
    effective_risk,
    marginal_risk_contributions,
    returns_matrix,
    signed_correlation,
)
from mfie.portfolio.costs import (
    TradeCosts,
    breakeven_hit_rate,
    estimate_costs,
    expected_holding_bars,
)
from mfie.portfolio.edge import (
    EdgeEstimate,
    evaluate_edge,
    expected_r,
    kelly_with_uncertainty,
)

__all__ = [
    "Allocation",
    "BookRisk",
    "CalibratedHitRate",
    "ConfidenceCalibrator",
    "CorrelationView",
    "EdgeEstimate",
    "PortfolioAllocator",
    "PortfolioPlan",
    "TradeCosts",
    "TradeRecord",
    "apply_plan",
    "assess_book",
    "breakeven_hit_rate",
    "cluster_by_correlation",
    "diversification_ratio",
    "effective_risk",
    "estimate_costs",
    "evaluate_edge",
    "expected_holding_bars",
    "expected_r",
    "kelly_with_uncertainty",
    "load_calibrator",
    "marginal_risk_contributions",
    "prior_hit_rate",
    "records_from_backtest",
    "records_from_rows",
    "returns_matrix",
    "signed_correlation",
]
