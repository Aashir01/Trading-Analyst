"""Strategy registry — name -> class, and the default active set."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from mfie.strategies.base import Strategy
from mfie.strategies.breakout import DonchianBreakoutStrategy, SqueezeBreakoutStrategy
from mfie.strategies.funding_basis import FundingCarryStrategy
from mfie.strategies.grid import GridStrategy
from mfie.strategies.mean_reversion import BollingerReversionStrategy, RSIReversionStrategy
from mfie.strategies.ml_classifier import MLClassifierStrategy
from mfie.strategies.statarb import PairsTradingStrategy
from mfie.strategies.trend import MACrossStrategy, TrendFollowingStrategy

REGISTRY: dict[str, type[Strategy]] = {
    TrendFollowingStrategy.name: TrendFollowingStrategy,
    MACrossStrategy.name: MACrossStrategy,
    BollingerReversionStrategy.name: BollingerReversionStrategy,
    RSIReversionStrategy.name: RSIReversionStrategy,
    DonchianBreakoutStrategy.name: DonchianBreakoutStrategy,
    SqueezeBreakoutStrategy.name: SqueezeBreakoutStrategy,
    PairsTradingStrategy.name: PairsTradingStrategy,
    FundingCarryStrategy.name: FundingCarryStrategy,
    GridStrategy.name: GridStrategy,
    MLClassifierStrategy.name: MLClassifierStrategy,
}

# The ML strategy trains on first use, so it is opt-in rather than default —
# a dashboard refresh across ten symbols should not kick off ten model fits.
DEFAULT_STRATEGIES: tuple[str, ...] = (
    TrendFollowingStrategy.name,
    MACrossStrategy.name,
    BollingerReversionStrategy.name,
    RSIReversionStrategy.name,
    DonchianBreakoutStrategy.name,
    SqueezeBreakoutStrategy.name,
    PairsTradingStrategy.name,
    FundingCarryStrategy.name,
    GridStrategy.name,
)


def get_strategy(name: str, **kwargs: Any) -> Strategy:
    if name not in REGISTRY:
        raise KeyError(f"Unknown strategy {name!r}. Available: {', '.join(sorted(REGISTRY))}")
    return REGISTRY[name](**kwargs)


def build_strategies(names: Iterable[str] | None = None, **kwargs: Any) -> list[Strategy]:
    selected = list(names) if names else list(DEFAULT_STRATEGIES)
    return [get_strategy(name, **kwargs) for name in selected]


def describe_strategies() -> list[dict[str, str]]:
    return [
        {
            "name": name,
            "family": cls.family,
            "description": cls.description,
            "asset_classes": ", ".join(cls.asset_classes),
            "min_bars": str(cls.min_bars),
            "default": "yes" if name in DEFAULT_STRATEGIES else "no",
        }
        for name, cls in sorted(REGISTRY.items())
    ]
