"""Engine access for the web layer.

A full ``AnalysisEngine.run()`` builds the macro snapshot, the Cycle Compass for
every domain in play, and then scores every instrument. That is seconds of work
and dozens of upstream requests. A web UI that triggers it once per page view
would be unusable and would exhaust every provider's rate limit within minutes.

So the engine sits behind a TTL cache with three properties that matter:

**Single-flight.** Concurrent requests for the same key wait on one computation
rather than starting their own. Without this, a page that loads four panels at
once runs the engine four times — the classic cache stampede, and the reason
"add a cache" so often fails to fix the load it was added for.

**Stale-while-revalidate.** An expired entry is still served, immediately, while
a refresh runs behind it. A trading UI that blanks its panels for eight seconds
every five minutes is worse than one showing data with a visible age on it, so
every payload carries ``as_of`` and ``stale`` and the interface renders the age.

**Never a partial failure.** The engine already degrades provider by provider;
this layer degrades endpoint by endpoint, returning a typed error envelope so
one dead panel cannot take the page down with it.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from mfie.config import get_params, get_settings
from mfie.core.utils import get_logger

log = get_logger(__name__)

# The engine's own inputs move on macro timescales; a five-minute floor keeps
# the UI responsive without pretending the underlying data is faster than it is.
DEFAULT_TTL_SECONDS = 300
BACKGROUND_REFRESH = True


@dataclass
class CacheEntry:
    value: Any
    created: float
    ttl: float
    refreshing: bool = False

    @property
    def age(self) -> float:
        return time.time() - self.created

    @property
    def stale(self) -> bool:
        return self.age > self.ttl


@dataclass
class Cached:
    """A cached payload plus the provenance the UI needs to be honest about it."""

    value: Any
    as_of: datetime
    age_seconds: float
    stale: bool
    ttl_seconds: float

    def meta(self) -> dict[str, Any]:
        """Provenance for the payload: when it was computed and how old it is.

        Every response carries this. A number on screen with no age attached is
        a number the reader will assume is live, and in a stale-while-revalidate
        cache that assumption is wrong most of the time.
        """
        return {
            "as_of": self.as_of.isoformat(),
            "age_seconds": round(self.age_seconds, 1),
            "stale": self.stale,
            "ttl_seconds": self.ttl_seconds,
        }

    def envelope(self, key: str, payload: Any) -> dict[str, Any]:
        """Wrap an already-serialised payload with its provenance.

        Takes the payload explicitly rather than reading ``self.value``: the
        cache holds live domain objects full of pandas frames, and handing one
        of those to the JSON encoder is a 500 waiting to happen.
        """
        return {key: payload, "meta": self.meta()}


class TTLStore:
    """Single-flight TTL cache with stale-while-revalidate semantics."""

    def __init__(self, ttl: float = DEFAULT_TTL_SECONDS) -> None:
        self.ttl = ttl
        self._entries: dict[str, CacheEntry] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def _lock_for(self, key: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(key, threading.Lock())

    def get(
        self,
        key: str,
        producer: Callable[[], Any],
        ttl: float | None = None,
        force: bool = False,
    ) -> Cached:
        ttl = self.ttl if ttl is None else ttl
        entry = self._entries.get(key)

        if entry is not None and not force and not entry.stale:
            return self._wrap(entry)

        # Stale but present: serve it now, refresh behind it. Only one thread
        # gets to start that refresh.
        if entry is not None and not force and BACKGROUND_REFRESH:
            if not entry.refreshing:
                entry.refreshing = True
                thread = threading.Thread(
                    target=self._refresh, args=(key, producer, ttl), daemon=True,
                    name=f"mfie-refresh-{key[:24]}",
                )
                thread.start()
            return self._wrap(entry)

        # Cold, or a forced refresh: block, but only one caller does the work.
        lock = self._lock_for(key)
        with lock:
            entry = self._entries.get(key)
            if entry is not None and not entry.stale and not force:
                return self._wrap(entry)      # produced while we waited
            value = producer()
            entry = CacheEntry(value=value, created=time.time(), ttl=ttl)
            self._entries[key] = entry
        return self._wrap(entry)

    def _refresh(self, key: str, producer: Callable[[], Any], ttl: float) -> None:
        lock = self._lock_for(key)
        acquired = lock.acquire(blocking=False)
        if not acquired:
            return
        try:
            value = producer()
            self._entries[key] = CacheEntry(value=value, created=time.time(), ttl=ttl)
        except Exception as exc:
            # Keep serving the stale value; a failed refresh must not empty the
            # cache, or a provider outage would turn a degraded page into a blank one.
            log.warning("Background refresh of %s failed: %s", key, exc)
            if (existing := self._entries.get(key)) is not None:
                existing.refreshing = False
        finally:
            if (existing := self._entries.get(key)) is not None:
                existing.refreshing = False
            lock.release()

    @staticmethod
    def _wrap(entry: CacheEntry) -> Cached:
        return Cached(
            value=entry.value,
            as_of=datetime.fromtimestamp(entry.created, tz=timezone.utc),
            age_seconds=entry.age,
            stale=entry.stale,
            ttl_seconds=entry.ttl,
        )

    def invalidate(self, prefix: str | None = None) -> int:
        with self._guard:
            keys = [k for k in self._entries if prefix is None or k.startswith(prefix)]
            for key in keys:
                self._entries.pop(key, None)
        return len(keys)

    def stats(self) -> dict[str, Any]:
        return {
            "entries": len(self._entries),
            "keys": sorted(self._entries),
            "stale": sum(1 for e in self._entries.values() if e.stale),
        }


@dataclass
class EngineService:
    """The single place the web layer touches the engine."""

    store: TTLStore = field(default_factory=TTLStore)
    _engine: Any = None
    _engine_lock: threading.Lock = field(default_factory=threading.Lock)

    # ------------------------------------------------------------------ engine
    @property
    def engine(self):
        """Lazily built and reused.

        Constructing an ``AnalysisEngine`` loads the strategy library, the
        filter chain, the cycle engine and the calibration model. Rebuilding
        that per request would also reload the calibrator mid-session, which
        would let position sizing drift between two panels of the same page.
        """
        if self._engine is None:
            with self._engine_lock:
                if self._engine is None:
                    from mfie.pipeline.engine import AnalysisEngine

                    self._engine = AnalysisEngine()
        return self._engine

    def reset(self) -> None:
        with self._engine_lock:
            self._engine = None
        self.store.invalidate()

    # ------------------------------------------------------------------- runs
    def analysis(
        self,
        symbols: tuple[str, ...] | None,
        timeframe: str,
        limit: int,
        force: bool = False,
    ) -> Cached:
        key = f"analysis:{','.join(symbols) if symbols else 'all'}:{timeframe}:{limit}"

        def produce():
            return self.engine.run(
                symbols=list(symbols) if symbols else None,
                timeframe=timeframe,
                limit=limit,
            )

        return self.store.get(key, produce, force=force)

    def ohlcv(self, symbol: str, timeframe: str, limit: int, force: bool = False) -> Cached:
        key = f"ohlcv:{symbol}:{timeframe}:{limit}"

        def produce():
            from mfie.core.universe import get_instrument
            from mfie.technical.indicators import compute_indicators
            from mfie.core.utils import annualization_factor

            instrument = get_instrument(symbol)
            frame = self.engine.hub.ohlcv(instrument, timeframe, limit)
            if frame.empty:
                return instrument, frame, None
            indicators = compute_indicators(
                frame, get_params().technical, annualization_factor(timeframe)
            )
            return instrument, frame, indicators

        return self.store.get(key, produce, force=force)

    def cycle(self, domain: str, force: bool = False) -> Cached:
        key = f"cycle:{domain}"

        def produce():
            return self.engine.cycle_engine.evaluate(domain)

        # The Compass reads years of daily history for a whole domain; it is the
        # slowest call in the project and the one that changes most slowly.
        return self.store.get(key, produce, ttl=900, force=force)

    def calibration(self, force: bool = False) -> Cached:
        def produce():
            from mfie.portfolio.calibration import load_calibrator

            return load_calibrator()

        return self.store.get("calibration", produce, ttl=1800, force=force)

    # ------------------------------------------------------------------ status
    def status(self) -> dict[str, Any]:
        settings = get_settings()
        providers = {
            "binance": True,                      # keyless
            "alternative.me": True,               # keyless
            "coingecko": bool(settings.coingecko_api_key),
            "alphavantage": bool(settings.alphavantage_api_key),
            "oanda": settings.has("oanda_api_key", "oanda_account_id"),
            "fred": bool(settings.fred_api_key),
            "tradingeconomics": bool(settings.tradingeconomics_api_key),
            "glassnode": bool(settings.glassnode_api_key),
            "newsapi": bool(settings.newsapi_key),
            "lunarcrush": bool(settings.lunarcrush_api_key),
        }
        return {
            "offline": settings.offline,
            "providers": providers,
            "configured": sum(1 for v in providers.values() if v),
            "cache": self.store.stats(),
        }


_service: EngineService | None = None
_service_lock = threading.Lock()


def get_service() -> EngineService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = EngineService()
    return _service
