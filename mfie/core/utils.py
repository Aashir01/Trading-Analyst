"""Small shared helpers: logging, TTL caching, numeric guards."""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import time
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from typing import Any, TypeVar

import numpy as np
import pandas as pd

T = TypeVar("T")

_LOG_CONFIGURED = False


def configure_console() -> None:
    """Make stdout/stderr UTF-8 safe.

    Filter explanations contain economic notation (ΔGLI, ±, σ). The default
    Windows console codepage is cp1252 and raises ``UnicodeEncodeError`` on
    those, which would crash the CLI while merely printing a result. Every
    entry point calls this first.
    """
    import sys

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # detached or already-closed stream
                pass


def get_logger(name: str) -> logging.Logger:
    global _LOG_CONFIGURED
    if not _LOG_CONFIGURED:
        from mfie.config import get_settings

        logging.basicConfig(
            level=getattr(logging, get_settings().log_level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
            datefmt="%H:%M:%S",
        )
        _LOG_CONFIGURED = True
    return logging.getLogger(name)


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #
class TTLCache:
    """Minimal time-to-live cache. Keeps API calls off the rate limiter."""

    def __init__(self, ttl_seconds: int = 300, maxsize: int = 512) -> None:
        self.ttl = ttl_seconds
        self.maxsize = maxsize
        self._store: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        hit = self._store.get(key)
        if hit is None:
            return None
        expires, value = hit
        if time.time() > expires:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        if len(self._store) >= self.maxsize:
            oldest = min(self._store, key=lambda k: self._store[k][0])
            self._store.pop(oldest, None)
        self._store[key] = (time.time() + self.ttl, value)

    def clear(self) -> None:
        self._store.clear()


def cache_key(*parts: Any) -> str:
    blob = json.dumps(parts, default=str, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()


def timed(fn: Callable[..., T]) -> Callable[..., T]:
    """Log wall time of a call at DEBUG level."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        log = get_logger(fn.__module__)
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            log.debug("%s took %.0f ms", fn.__qualname__, (time.perf_counter() - start) * 1000)

    return wrapper


# --------------------------------------------------------------------------- #
# Numeric guards
# --------------------------------------------------------------------------- #
def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that never raises and never returns inf/nan."""
    try:
        if denominator == 0 or not np.isfinite(denominator):
            return default
        out = numerator / denominator
        return float(out) if np.isfinite(out) else default
    except (TypeError, ZeroDivisionError):
        return default


def clamp(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


def pct_change(current: float | None, past: float | None) -> float:
    """Relative change (current - past) / past, guarded."""
    if current is None or past is None:
        return 0.0
    return safe_div(current - past, abs(past), 0.0)


def zscore(series: pd.Series, window: int | None = None) -> float:
    """Z-score of the latest observation, optionally over a rolling window."""
    s = pd.Series(series).dropna().astype(float)
    if window:
        s = s.tail(window)
    if len(s) < 3:
        return 0.0
    sigma = float(s.std(ddof=1))
    if sigma == 0 or not np.isfinite(sigma):
        return 0.0
    return float((s.iloc[-1] - s.mean()) / sigma)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    s = pd.Series(series).astype(float)
    mu = s.rolling(window, min_periods=max(3, window // 3)).mean()
    sd = s.rolling(window, min_periods=max(3, window // 3)).std(ddof=1)
    return ((s - mu) / sd.replace(0.0, np.nan)).fillna(0.0)


def last_valid(series: pd.Series, default: float = 0.0) -> float:
    s = pd.Series(series).dropna()
    if s.empty:
        return default
    val = float(s.iloc[-1])
    return val if np.isfinite(val) else default


def ensure_utc(ts: datetime) -> datetime:
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)


def geometric_mean(values: Iterable[float]) -> float:
    """Product of multipliers, used to combine filter penalties."""
    arr = [float(v) for v in values if v is not None and np.isfinite(v) and v > 0]
    if not arr:
        return 1.0
    return float(np.exp(np.mean(np.log(arr))))


def annualization_factor(timeframe: str) -> float:
    """Periods per year for a timeframe string like '1h', '4h', '1d'."""
    table = {
        "1m": 525_600, "5m": 105_120, "15m": 35_040, "30m": 17_520,
        "1h": 8_760, "2h": 4_380, "4h": 2_190, "12h": 730,
        "1d": 365, "1w": 52,
    }
    return float(table.get(timeframe, 365))
