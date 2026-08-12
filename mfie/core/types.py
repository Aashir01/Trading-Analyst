"""Domain types shared by every layer of the engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AssetClass(str, Enum):
    CRYPTO = "crypto"
    FOREX = "forex"


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"

    @property
    def sign(self) -> int:
        return {"long": 1, "short": -1, "flat": 0}[self.value]

    def opposite(self) -> Direction:
        if self is Direction.LONG:
            return Direction.SHORT
        if self is Direction.SHORT:
            return Direction.LONG
        return Direction.FLAT


class LiquidityRegime(str, Enum):
    EXPANSION = "expansion"
    NEUTRAL = "neutral"
    CONTRACTION = "contraction"


class MacroRegime(str, Enum):
    EXPANSION = "expansion"
    NEUTRAL = "neutral"
    CONTRACTION = "contraction"


class MarketRegime(str, Enum):
    TRENDING_HIGH_VOL = "trending_high_vol"
    TRENDING_LOW_VOL = "trending_low_vol"
    RANGING = "ranging"
    CRASH = "crash"
    UNKNOWN = "unknown"

    @property
    def favours_momentum(self) -> bool:
        return self in (MarketRegime.TRENDING_HIGH_VOL, MarketRegime.TRENDING_LOW_VOL)

    @property
    def favours_mean_reversion(self) -> bool:
        return self is MarketRegime.RANGING


class FilterAction(str, Enum):
    PASS = "pass"
    PENALIZE = "penalize"
    BOOST = "boost"
    BLOCK = "block"


@dataclass(frozen=True)
class Instrument:
    """A tradable symbol plus the economic entities behind it.

    For forex, ``base``/``quote`` are ISO currency codes and drive the
    Real-Interest-Rate-Differential and Economic-Surprise modules. For crypto,
    ``base`` is the token and ``quote`` is usually a USD stablecoin.
    """

    symbol: str
    asset_class: AssetClass
    base: str
    quote: str
    coingecko_id: str | None = None
    pip_size: float = 0.0001
    display: str | None = None

    @property
    def name(self) -> str:
        return self.display or self.symbol

    @property
    def is_forex(self) -> bool:
        return self.asset_class is AssetClass.FOREX

    @property
    def is_crypto(self) -> bool:
        return self.asset_class is AssetClass.CRYPTO


@dataclass
class Quote:
    """Top-of-book snapshot used by the microstructure filter."""

    symbol: str
    bid: float
    ask: float
    ts: datetime = field(default_factory=utcnow)

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return max(self.ask - self.bid, 0.0)

    @property
    def spread_bps(self) -> float:
        return (self.spread / self.mid) * 10_000 if self.mid else 0.0


@dataclass
class EconomicEvent:
    """A scheduled macro release (CPI, NFP, rate decision...)."""

    ts: datetime
    country: str          # ISO-3166 alpha-2, e.g. "US"
    currency: str         # ISO-4217, e.g. "USD"
    name: str
    impact: str = "high"  # low | medium | high
    actual: float | None = None
    forecast: float | None = None
    previous: float | None = None
    unit: str = ""

    @property
    def surprise(self) -> float | None:
        if self.actual is None or self.forecast is None:
            return None
        return self.actual - self.forecast


@dataclass
class MacroContext:
    """Everything the filter chain needs to judge a technical signal.

    Built once per analysis run by ``mfie.pipeline.engine`` so every filter sees
    a consistent snapshot of the world.
    """

    ts: datetime = field(default_factory=utcnow)

    # Global monetary conditions
    gli_delta: float = 0.0
    gli_regime: LiquidityRegime = LiquidityRegime.NEUTRAL
    m2_change: float = 0.0
    stablecoin_change: float = 0.0

    # Rates: keyed by currency code
    policy_rates: dict[str, float] = field(default_factory=dict)
    inflation: dict[str, float] = field(default_factory=dict)
    yield_10y: dict[str, float] = field(default_factory=dict)
    yield_2y: dict[str, float] = field(default_factory=dict)
    macro_regime: MacroRegime = MacroRegime.NEUTRAL

    # Relative economics
    esi: dict[str, float] = field(default_factory=dict)          # by currency
    reer_history: dict[str, Any] = field(default_factory=dict)   # currency -> pd.Series

    # Crypto network economics: symbol -> metric
    token_velocity: dict[str, float] = field(default_factory=dict)
    token_velocity_z: dict[str, float] = field(default_factory=dict)
    nvt: dict[str, float] = field(default_factory=dict)
    nvt_z: dict[str, float] = field(default_factory=dict)

    # Sentiment / positioning
    fear_greed: float | None = None
    retail_long_pct: dict[str, float] = field(default_factory=dict)
    news_sentiment: dict[str, float] = field(default_factory=dict)

    # Calendar
    events: list[EconomicEvent] = field(default_factory=list)

    # Portfolio risk state
    portfolio_cvar: float | None = None
    portfolio_sortino: float | None = None
    open_risk: float = 0.0

    # Provenance: which provider served each block ("live" vs "synthetic")
    sources: dict[str, str] = field(default_factory=dict)

    def real_rate(self, currency: str) -> float | None:
        """Nominal policy rate minus trailing CPI, i.e. R = I - pi."""
        i = self.policy_rates.get(currency)
        pi = self.inflation.get(currency)
        if i is None or pi is None:
            return None
        return i - pi

    def yield_spread(self, currency: str) -> float | None:
        """10Y - 2Y sovereign term premium."""
        y10 = self.yield_10y.get(currency)
        y2 = self.yield_2y.get(currency)
        if y10 is None or y2 is None:
            return None
        return y10 - y2


@dataclass
class RawSignal:
    """A pre-filter technical/statistical setup."""

    instrument: Instrument
    direction: Direction
    strategy: str
    strength: float                  # 0..1, the strategy's own conviction
    entry: float
    stop: float
    take_profit: float | None = None
    timeframe: str = "1h"
    ts: datetime = field(default_factory=utcnow)
    rationale: list[str] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict)

    @property
    def stop_distance(self) -> float:
        return abs(self.entry - self.stop)

    @property
    def reward_risk(self) -> float | None:
        if self.take_profit is None or self.stop_distance == 0:
            return None
        return abs(self.take_profit - self.entry) / self.stop_distance


@dataclass
class FilterOutcome:
    """The verdict of one econometric rule, kept for the audit trail."""

    name: str
    action: FilterAction
    multiplier: float = 1.0
    stop_multiplier: float = 1.0
    reason: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return self.action is FilterAction.BLOCK

    def __str__(self) -> str:
        tag = {
            FilterAction.PASS: "PASS",
            FilterAction.BOOST: "BOOST",
            FilterAction.PENALIZE: "PENALTY",
            FilterAction.BLOCK: "BLOCK",
        }[self.action]
        return f"[{tag}] {self.name}: {self.reason} (x{self.multiplier:.2f})"


@dataclass
class ScoredSignal:
    """A raw signal after the full macro filter chain and risk sizing."""

    raw: RawSignal
    outcomes: list[FilterOutcome] = field(default_factory=list)
    confidence: float = 0.0
    size_fraction: float = 0.0     # fraction of equity to allocate
    risk_fraction: float = 0.0     # fraction of equity at risk if stopped out
    units: float = 0.0
    adjusted_stop: float = 0.0
    adjusted_take_profit: float | None = None
    blocked: bool = False
    block_reasons: list[str] = field(default_factory=list)
    macro: MacroContext | None = None

    @property
    def instrument(self) -> Instrument:
        return self.raw.instrument

    @property
    def direction(self) -> Direction:
        return self.raw.direction

    @property
    def verdict(self) -> str:
        if self.blocked:
            return "BLOCKED"
        if self.confidence >= 0.65:
            return "STRONG"
        if self.confidence >= 0.45:
            return "MODERATE"
        return "WEAK"

    def audit(self) -> list[str]:
        return [str(o) for o in self.outcomes]
