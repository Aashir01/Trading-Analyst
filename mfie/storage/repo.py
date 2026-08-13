"""Repository: typed read/write helpers over the schema.

All writes are idempotent upserts keyed on the natural time-series key, so a
re-run of the ingestion loop over overlapping windows is safe.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from mfie.core.types import EconomicEvent, Instrument, Quote, ScoredSignal
from mfie.core.utils import get_logger
from mfie.storage.db import Database, get_database
from mfie.storage.models import (
    OHLCV,
    EconomicEventRow,
    EquityPoint,
    MacroSeries,
    OnChainMetric,
    QuoteTick,
    SentimentPoint,
    SignalRow,
    TradeRow,
)

log = get_logger(__name__)


class Repository:
    def __init__(self, db: Database | None = None) -> None:
        self.db = db or get_database()

    # ------------------------------------------------------------------ upsert
    def _upsert(self, table, rows: Sequence[dict[str, Any]], index_elements: list[str],
                update_columns: list[str] | None = None) -> int:
        if not rows:
            return 0
        insert_fn = pg_insert if self.db.is_postgres else sqlite_insert
        with self.db.session() as sess:
            stmt = insert_fn(table).values(list(rows))
            if update_columns:
                stmt = stmt.on_conflict_do_update(
                    index_elements=index_elements,
                    set_={col: getattr(stmt.excluded, col) for col in update_columns},
                )
            else:
                stmt = stmt.on_conflict_do_nothing(index_elements=index_elements)
            sess.execute(stmt)
        return len(rows)

    # ------------------------------------------------------------------- OHLCV
    def save_ohlcv(self, symbol: str, timeframe: str, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        rows = [
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "ts": ts.to_pydatetime(),
                "open": float(r.open),
                "high": float(r.high),
                "low": float(r.low),
                "close": float(r.close),
                "volume": float(r.volume),
            }
            for ts, r in df.iterrows()
        ]
        return self._upsert(
            OHLCV, rows, ["symbol", "timeframe", "ts"],
            ["open", "high", "low", "close", "volume"],
        )

    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 1000,
        since: datetime | None = None,
    ) -> pd.DataFrame:
        stmt = select(OHLCV).where(OHLCV.symbol == symbol, OHLCV.timeframe == timeframe)
        if since is not None:
            stmt = stmt.where(OHLCV.ts >= since)
        stmt = stmt.order_by(OHLCV.ts.desc()).limit(limit)
        with self.db.session() as sess:
            records = sess.execute(stmt).scalars().all()
        if not records:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        frame = pd.DataFrame(
            [
                {
                    "ts": r.ts, "open": r.open, "high": r.high,
                    "low": r.low, "close": r.close, "volume": r.volume,
                }
                for r in records
            ]
        )
        frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
        return frame.set_index("ts").sort_index()

    # ------------------------------------------------------------------ quotes
    def save_quote(self, quote: Quote) -> int:
        return self._upsert(
            QuoteTick,
            [{
                "symbol": quote.symbol,
                "ts": quote.ts,
                "bid": quote.bid,
                "ask": quote.ask,
                "spread_bps": quote.spread_bps,
            }],
            ["symbol", "ts"],
            ["bid", "ask", "spread_bps"],
        )

    # ------------------------------------------------------------ macro series
    def save_macro_series(
        self,
        series_id: str,
        series: pd.Series,
        source: str = "unknown",
        currency: str | None = None,
    ) -> int:
        if series is None or series.empty:
            return 0
        rows = [
            {
                "series_id": series_id,
                "ts": pd.Timestamp(ts).to_pydatetime(),
                "value": float(value),
                "source": source,
                "currency": currency,
            }
            for ts, value in series.dropna().items()
        ]
        return self._upsert(MacroSeries, rows, ["series_id", "ts"], ["value", "source", "currency"])

    def load_macro_series(self, series_id: str, limit: int = 1000) -> pd.Series:
        stmt = (
            select(MacroSeries)
            .where(MacroSeries.series_id == series_id)
            .order_by(MacroSeries.ts.desc())
            .limit(limit)
        )
        with self.db.session() as sess:
            records = sess.execute(stmt).scalars().all()
        if not records:
            return pd.Series(dtype=float, name=series_id)
        idx = pd.to_datetime([r.ts for r in records], utc=True)
        return pd.Series([r.value for r in records], index=idx, name=series_id).sort_index()

    # --------------------------------------------------------------- on-chain
    def save_onchain(self, symbol: str, metric: str, series: pd.Series) -> int:
        if series is None or series.empty:
            return 0
        rows = [
            {
                "symbol": symbol,
                "metric": metric,
                "ts": pd.Timestamp(ts).to_pydatetime(),
                "value": float(value),
            }
            for ts, value in series.dropna().items()
        ]
        return self._upsert(OnChainMetric, rows, ["symbol", "metric", "ts"], ["value"])

    # -------------------------------------------------------------- sentiment
    def save_sentiment(self, symbol: str, source: str, value: float,
                       ts: datetime | None = None) -> int:
        return self._upsert(
            SentimentPoint,
            [{
                "symbol": symbol,
                "source": source,
                "ts": ts or datetime.now(timezone.utc),
                "value": float(value),
            }],
            ["symbol", "source", "ts"],
            ["value"],
        )

    # ----------------------------------------------------------------- events
    def save_events(self, events: Iterable[EconomicEvent]) -> int:
        rows = [
            {
                "ts": e.ts,
                "currency": e.currency,
                "country": e.country,
                "name": e.name[:160],
                "impact": e.impact,
                "actual": e.actual,
                "forecast": e.forecast,
                "previous": e.previous,
                "unit": e.unit[:24],
            }
            for e in events
        ]
        return self._upsert(
            EconomicEventRow, rows, ["ts", "currency", "name"],
            ["impact", "actual", "forecast", "previous", "unit"],
        )

    def load_events(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
        currencies: Sequence[str] | None = None,
    ) -> list[EconomicEvent]:
        stmt = select(EconomicEventRow)
        if since is not None:
            stmt = stmt.where(EconomicEventRow.ts >= since)
        if until is not None:
            stmt = stmt.where(EconomicEventRow.ts <= until)
        if currencies:
            stmt = stmt.where(EconomicEventRow.currency.in_(list(currencies)))
        with self.db.session() as sess:
            records = sess.execute(stmt.order_by(EconomicEventRow.ts)).scalars().all()
        return [
            EconomicEvent(
                ts=pd.Timestamp(r.ts, tz="UTC").to_pydatetime(),
                country=r.country,
                currency=r.currency,
                name=r.name,
                impact=r.impact,
                actual=r.actual,
                forecast=r.forecast,
                previous=r.previous,
                unit=r.unit,
            )
            for r in records
        ]

    # ---------------------------------------------------------------- signals
    def save_signal(self, signal: ScoredSignal) -> int:
        raw = signal.raw
        macro_summary: dict[str, Any] = {}
        if signal.macro is not None:
            macro_summary = {
                "gli_delta": signal.macro.gli_delta,
                "gli_regime": signal.macro.gli_regime.value,
                "macro_regime": signal.macro.macro_regime.value,
                "fear_greed": signal.macro.fear_greed,
                "portfolio_cvar": signal.macro.portfolio_cvar,
                "sources": signal.macro.sources,
            }
        with self.db.session() as sess:
            sess.add(
                SignalRow(
                    ts=raw.ts,
                    symbol=raw.instrument.symbol,
                    asset_class=raw.instrument.asset_class.value,
                    timeframe=raw.timeframe,
                    strategy=raw.strategy,
                    direction=raw.direction.value,
                    raw_strength=raw.strength,
                    confidence=signal.confidence,
                    size_fraction=signal.size_fraction,
                    risk_fraction=signal.risk_fraction,
                    entry=raw.entry,
                    stop=signal.adjusted_stop or raw.stop,
                    take_profit=signal.adjusted_take_profit or raw.take_profit,
                    blocked=signal.blocked,
                    verdict=signal.verdict,
                    audit_json=json.dumps(signal.audit()),
                    macro_json=json.dumps(macro_summary, default=str),
                )
            )
        return 1

    def recent_signals(self, limit: int = 50, symbol: str | None = None) -> list[dict[str, Any]]:
        stmt = select(SignalRow)
        if symbol:
            stmt = stmt.where(SignalRow.symbol == symbol)
        stmt = stmt.order_by(SignalRow.ts.desc(), SignalRow.id.desc()).limit(limit)
        with self.db.session() as sess:
            records = sess.execute(stmt).scalars().all()
        return [
            {
                "ts": r.ts,
                "symbol": r.symbol,
                "strategy": r.strategy,
                "direction": r.direction,
                "confidence": r.confidence,
                "size_fraction": r.size_fraction,
                "entry": r.entry,
                "stop": r.stop,
                "take_profit": r.take_profit,
                "blocked": r.blocked,
                "verdict": r.verdict,
                "audit": json.loads(r.audit_json or "[]"),
            }
            for r in records
        ]

    # ----------------------------------------------------------------- trades
    def save_trades(self, trades: Sequence[dict[str, Any]], run_id: str | None = None) -> str:
        run_id = run_id or uuid.uuid4().hex[:12]
        if not trades:
            return run_id
        with self.db.session() as sess:
            for t in trades:
                sess.add(TradeRow(run_id=run_id, **t))
        return run_id

    def closed_trades(self, limit: int = 5000, run_id: str | None = None) -> list[dict[str, Any]]:
        """Realised trades, newest first — the input to hit-rate calibration."""
        stmt = select(TradeRow).where(TradeRow.exit_ts.is_not(None))
        if run_id:
            stmt = stmt.where(TradeRow.run_id == run_id)
        stmt = stmt.order_by(TradeRow.exit_ts.desc()).limit(limit)
        with self.db.session() as sess:
            records = sess.execute(stmt).scalars().all()
        return [
            {
                "symbol": r.symbol,
                "strategy": r.strategy,
                "direction": r.direction,
                "entry_price": r.entry_price,
                "exit_price": r.exit_price,
                "stop": r.stop,
                "units": r.units,
                "pnl": r.pnl,
                "return_pct": r.return_pct,
                "confidence": r.confidence,
                "risk_fraction": r.risk_fraction,
                "regime": r.regime,
                "asset_class": r.asset_class,
                "exit_reason": r.exit_reason,
                "exit_ts": r.exit_ts,
            }
            for r in records
        ]

    def save_equity_curve(self, run_id: str, equity: pd.Series, drawdown: pd.Series | None = None) -> int:
        if equity is None or equity.empty:
            return 0
        dd = drawdown if drawdown is not None else pd.Series(0.0, index=equity.index)
        rows = [
            {
                "run_id": run_id,
                "ts": pd.Timestamp(ts).to_pydatetime(),
                "equity": float(value),
                "drawdown": float(dd.get(ts, 0.0)),
            }
            for ts, value in equity.items()
        ]
        return self._upsert(EquityPoint, rows, ["run_id", "ts"], ["equity", "drawdown"])

    def purge_run(self, run_id: str) -> None:
        with self.db.session() as sess:
            sess.execute(delete(TradeRow).where(TradeRow.run_id == run_id))
            sess.execute(delete(EquityPoint).where(EquityPoint.run_id == run_id))

    # ------------------------------------------------------------------ bulk
    def ingest_instrument(self, hub, instrument: Instrument, timeframe: str = "1h",
                          limit: int = 500) -> dict[str, int]:
        """Fetch and persist every block for one instrument. Used by ``mfie ingest``."""
        counts: dict[str, int] = {}
        df = hub.ohlcv(instrument, timeframe, limit)
        counts["ohlcv"] = self.save_ohlcv(instrument.symbol, timeframe, df)
        counts["quote"] = self.save_quote(hub.quote(instrument))
        if instrument.is_crypto:
            counts["onchain_volume"] = self.save_onchain(
                instrument.symbol, "transfer_volume_usd", hub.onchain_volume(instrument)
            )
            counts["market_cap"] = self.save_onchain(
                instrument.symbol, "market_cap", hub.market_cap(instrument)
            )
        counts["retail"] = self.save_sentiment(
            instrument.symbol, "retail_long_pct", hub.retail_positioning(instrument)
        )
        counts["news"] = self.save_sentiment(
            instrument.symbol, "news", hub.news_sentiment(instrument)
        )
        return counts
