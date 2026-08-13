"""SQLAlchemy schema.

Design notes:

* Time-series tables (``ohlcv``, ``quotes``, ``macro_series``, ``onchain_metrics``,
  ``sentiment``) use a composite primary key ending in ``ts``. That is exactly the
  shape TimescaleDB wants for a hypertable partitioned on ``ts``, and it also
  gives SQLite a usable covering index.
* Relational/config tables (``economic_events``, ``signals``, ``trades``) are
  ordinary tables with surrogate keys.

The same models therefore run unchanged on SQLite (local dev) and on
PostgreSQL/TimescaleDB (production); only ``db.py`` differs.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


# --------------------------------------------------------------------------- #
# Time-series (hypertable candidates)
# --------------------------------------------------------------------------- #
class OHLCV(Base):
    __tablename__ = "ohlcv"

    symbol: Mapped[str] = mapped_column(String(24), primary_key=True)
    timeframe: Mapped[str] = mapped_column(String(8), primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, default=0.0)

    __table_args__ = (Index("ix_ohlcv_symbol_ts", "symbol", "ts"),)


class QuoteTick(Base):
    __tablename__ = "quotes"

    symbol: Mapped[str] = mapped_column(String(24), primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    bid: Mapped[float] = mapped_column(Float, nullable=False)
    ask: Mapped[float] = mapped_column(Float, nullable=False)
    spread_bps: Mapped[float] = mapped_column(Float, default=0.0)


class MacroSeries(Base):
    """Any scalar macro time series: M2, DGS10, CPI level, REER, stablecoin supply."""

    __tablename__ = "macro_series"

    series_id: Mapped[str] = mapped_column(String(48), primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str] = mapped_column(String(24), default="unknown")
    currency: Mapped[str | None] = mapped_column(String(8), nullable=True)


class OnChainMetric(Base):
    __tablename__ = "onchain_metrics"

    symbol: Mapped[str] = mapped_column(String(24), primary_key=True)
    metric: Mapped[str] = mapped_column(String(48), primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    value: Mapped[float] = mapped_column(Float, nullable=False)


class SentimentPoint(Base):
    __tablename__ = "sentiment"

    symbol: Mapped[str] = mapped_column(String(24), primary_key=True)
    source: Mapped[str] = mapped_column(String(32), primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    value: Mapped[float] = mapped_column(Float, nullable=False)


# --------------------------------------------------------------------------- #
# Relational
# --------------------------------------------------------------------------- #
class EconomicEventRow(Base):
    __tablename__ = "economic_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    currency: Mapped[str] = mapped_column(String(8), nullable=False, index=True)
    country: Mapped[str] = mapped_column(String(8), default="")
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    impact: Mapped[str] = mapped_column(String(8), default="low")
    actual: Mapped[float | None] = mapped_column(Float, nullable=True)
    forecast: Mapped[float | None] = mapped_column(Float, nullable=True)
    previous: Mapped[float | None] = mapped_column(Float, nullable=True)
    unit: Mapped[str] = mapped_column(String(24), default="")

    __table_args__ = (
        UniqueConstraint("ts", "currency", "name", name="uq_event_identity"),
    )


class SignalRow(Base):
    """Every scored signal is persisted — this is the audit log that lets you
    ask later 'why did the engine block that trade?'."""

    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    asset_class: Mapped[str] = mapped_column(String(8), default="")
    timeframe: Mapped[str] = mapped_column(String(8), default="1h")
    strategy: Mapped[str] = mapped_column(String(48), default="")
    direction: Mapped[str] = mapped_column(String(8), default="flat")
    raw_strength: Mapped[float] = mapped_column(Float, default=0.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    size_fraction: Mapped[float] = mapped_column(Float, default=0.0)
    risk_fraction: Mapped[float] = mapped_column(Float, default=0.0)
    entry: Mapped[float] = mapped_column(Float, default=0.0)
    stop: Mapped[float] = mapped_column(Float, default=0.0)
    take_profit: Mapped[float | None] = mapped_column(Float, nullable=True)
    blocked: Mapped[bool] = mapped_column(Boolean, default=False)
    verdict: Mapped[str] = mapped_column(String(16), default="")
    audit_json: Mapped[str] = mapped_column(Text, default="[]")
    macro_json: Mapped[str] = mapped_column(Text, default="{}")


class TradeRow(Base):
    """Backtest / paper-trade fills."""

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(48), index=True, default="")
    symbol: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    strategy: Mapped[str] = mapped_column(String(48), default="")
    direction: Mapped[str] = mapped_column(String(8), default="flat")
    entry_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    exit_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    entry_price: Mapped[float] = mapped_column(Float, default=0.0)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    units: Mapped[float] = mapped_column(Float, default=0.0)
    pnl: Mapped[float] = mapped_column(Float, default=0.0)
    return_pct: Mapped[float] = mapped_column(Float, default=0.0)
    exit_reason: Mapped[str] = mapped_column(String(24), default="")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    # Needed to recover the R multiple (pnl / (units * |entry - stop|)) and to
    # condition the hit-rate calibration on the regime the trade was born in.
    # Additive columns: a database created by an earlier version lacks them,
    # in which case the calibrator falls back to its prior rather than failing.
    stop: Mapped[float] = mapped_column(Float, default=0.0)
    risk_fraction: Mapped[float] = mapped_column(Float, default=0.0)
    regime: Mapped[str] = mapped_column(String(24), default="unknown")
    asset_class: Mapped[str] = mapped_column(String(12), default="unknown")


class EquityPoint(Base):
    __tablename__ = "equity_curve"

    run_id: Mapped[str] = mapped_column(String(48), primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    equity: Mapped[float] = mapped_column(Float, nullable=False)
    drawdown: Mapped[float] = mapped_column(Float, default=0.0)


# Tables that become TimescaleDB hypertables when running on PostgreSQL.
HYPERTABLES: dict[str, str] = {
    "ohlcv": "ts",
    "quotes": "ts",
    "macro_series": "ts",
    "onchain_metrics": "ts",
    "sentiment": "ts",
}
