"""Strategy interface and shared signal construction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from mfie.config import Params, RiskParams, get_params
from mfie.core.types import Direction, Instrument, RawSignal
from mfie.core.utils import clamp
from mfie.regime.detector import RegimeView
from mfie.technical.indicators import IndicatorSet

# Strategy families, used by the regime detector to weight each strategy.
FAMILIES = ("momentum", "mean_reversion", "breakout", "carry", "market_making", "ml")


@dataclass
class StrategyContext:
    """Everything a strategy is allowed to see.

    Note what is *absent*: no macro context, no account equity, no open
    positions. Strategies propose; the pipeline disposes.
    """

    instrument: Instrument
    df: pd.DataFrame
    indicators: IndicatorSet
    regime: RegimeView
    timeframe: str = "1h"
    params: Params = field(default_factory=get_params)
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def price(self) -> float:
        return float(self.df["close"].iloc[-1])

    @property
    def atr(self) -> float:
        value = self.indicators.last("atr", 0.0)
        # Degenerate ATR (flat synthetic data, warm-up) would make stops zero.
        return value if value > 0 else max(self.price * 0.005, 1e-9)


class Strategy(ABC):
    """Base class. Subclasses implement ``generate``."""

    name: str = "base"
    family: str = "momentum"
    description: str = ""
    asset_classes: tuple[str, ...] = ("crypto", "forex")
    min_bars: int = 100

    def __init__(self, **overrides: Any) -> None:
        self.overrides = overrides

    def supports(self, ctx: StrategyContext) -> bool:
        if ctx.instrument.asset_class.value not in self.asset_classes:
            return False
        return len(ctx.df) >= self.min_bars

    @abstractmethod
    def generate(self, ctx: StrategyContext) -> RawSignal | None:
        """Return a signal, or ``None`` when the setup is absent."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name} family={self.family}>"


def build_signal(
    ctx: StrategyContext,
    direction: Direction,
    strategy: str,
    strength: float,
    rationale: list[str],
    features: dict[str, float] | None = None,
    stop_atr_multiple: float | None = None,
    reward_risk: float | None = None,
    entry: float | None = None,
    stop: float | None = None,
) -> RawSignal:
    """Assemble a ``RawSignal`` with ATR-derived stop and target.

    Stops are volatility-scaled rather than fixed-percentage: a 1% stop is
    reckless on BTC and absurdly wide on EURUSD, whereas 2xATR means the same
    thing on both.
    """
    risk: RiskParams = ctx.params.risk
    entry_price = float(entry if entry is not None else ctx.price)
    atr_multiple = stop_atr_multiple if stop_atr_multiple is not None else risk.atr_stop_multiple
    rr = reward_risk if reward_risk is not None else risk.reward_risk_target

    if stop is not None:
        stop_price = float(stop)
    else:
        distance = ctx.atr * atr_multiple
        stop_price = entry_price - distance if direction is Direction.LONG else entry_price + distance

    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0:
        stop_distance = ctx.atr * atr_multiple
        stop_price = (
            entry_price - stop_distance if direction is Direction.LONG else entry_price + stop_distance
        )

    take_profit = (
        entry_price + stop_distance * rr
        if direction is Direction.LONG
        else entry_price - stop_distance * rr
    )

    return RawSignal(
        instrument=ctx.instrument,
        direction=direction,
        strategy=strategy,
        strength=clamp(strength, 0.0, 1.0),
        entry=entry_price,
        stop=stop_price,
        take_profit=take_profit,
        timeframe=ctx.timeframe,
        ts=ctx.df.index[-1].to_pydatetime(),
        rationale=rationale,
        features=features or {},
    )


def scale_strength(raw: float, floor: float = 0.15, ceiling: float = 0.95) -> float:
    """Keep a strategy's self-assessed conviction inside a sane band.

    No single strategy should ever claim near-certainty — the macro layer, not
    the pattern, decides how much conviction survives.
    """
    return float(clamp(raw, floor, ceiling))
