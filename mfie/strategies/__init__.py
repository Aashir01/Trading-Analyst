"""Strategy library.

Each strategy emits at most one ``RawSignal`` per call and knows nothing about
macro filters or sizing — that separation is what lets the same strategy be
scored, backtested and blocked by the pipeline without modification.
"""

from mfie.strategies.base import Strategy, StrategyContext, build_signal  # noqa: F401
from mfie.strategies.breakout import DonchianBreakoutStrategy, SqueezeBreakoutStrategy  # noqa: F401
from mfie.strategies.funding_basis import FundingCarryStrategy  # noqa: F401
from mfie.strategies.grid import GridStrategy  # noqa: F401
from mfie.strategies.mean_reversion import (  # noqa: F401
    BollingerReversionStrategy,
    RSIReversionStrategy,
)
from mfie.strategies.ml_classifier import MLClassifierStrategy  # noqa: F401
from mfie.strategies.registry import (  # noqa: F401
    DEFAULT_STRATEGIES,
    build_strategies,
    get_strategy,
)
from mfie.strategies.statarb import PairsTradingStrategy, find_cointegrated_pairs  # noqa: F401
from mfie.strategies.trend import MACrossStrategy, TrendFollowingStrategy  # noqa: F401

__all__ = [
    "Strategy",
    "StrategyContext",
    "build_signal",
    "TrendFollowingStrategy",
    "MACrossStrategy",
    "BollingerReversionStrategy",
    "RSIReversionStrategy",
    "DonchianBreakoutStrategy",
    "SqueezeBreakoutStrategy",
    "PairsTradingStrategy",
    "find_cointegrated_pairs",
    "FundingCarryStrategy",
    "GridStrategy",
    "MLClassifierStrategy",
    "DEFAULT_STRATEGIES",
    "build_strategies",
    "get_strategy",
]
