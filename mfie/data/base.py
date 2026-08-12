"""HTTP plumbing shared by every live provider.

Providers must never crash the pipeline. A failed request returns ``None`` and
the caller falls back to the synthetic provider, tagging the data block as
``synthetic`` in ``MacroContext.sources`` so the UI can show what is real.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import requests

from mfie.config import Settings, get_settings
from mfie.core.utils import TTLCache, cache_key, get_logger

log = get_logger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class ProviderError(RuntimeError):
    """Raised internally; always caught at the provider boundary."""


class HttpClient:
    """requests wrapper with TTL cache, retry and exponential backoff."""

    def __init__(self, settings: Settings | None = None, ttl: int | None = None) -> None:
        self.settings = settings or get_settings()
        self.cache = TTLCache(ttl if ttl is not None else self.settings.cache_ttl_seconds)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "MFIE/0.1 (analysis tool)"})

    def get_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        retries: int = 2,
        use_cache: bool = True,
    ) -> Any | None:
        key = cache_key(url, params, headers)
        if use_cache:
            hit = self.cache.get(key)
            if hit is not None:
                return hit

        delay = 0.5
        for attempt in range(retries + 1):
            try:
                resp = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.settings.http_timeout,
                )
                if resp.status_code == 429:
                    log.warning("Rate limited by %s, backing off %.1fs", url, delay)
                    time.sleep(delay)
                    delay *= 2
                    continue
                resp.raise_for_status()
                data = resp.json()
                if use_cache:
                    self.cache.set(key, data)
                return data
            except (requests.RequestException, ValueError) as exc:
                if attempt == retries:
                    log.warning("GET %s failed after %d attempts: %s", url, attempt + 1, exc)
                    return None
                time.sleep(delay)
                delay *= 2
        return None


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce any provider frame into the canonical OHLCV shape.

    Canonical form: tz-aware UTC ``DatetimeIndex`` named ``ts``, float columns
    ``open, high, low, close, volume``, sorted ascending, no duplicate stamps.
    Every downstream module assumes exactly this.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]

    if not isinstance(out.index, pd.DatetimeIndex):
        for candidate in ("ts", "timestamp", "time", "date", "datetime"):
            if candidate in out.columns:
                out.index = pd.to_datetime(out[candidate], utc=True)
                out = out.drop(columns=[candidate])
                break

    if not isinstance(out.index, pd.DatetimeIndex):
        raise ProviderError("OHLCV frame has no usable timestamp index")

    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    out.index.name = "ts"

    for col in OHLCV_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0 if col == "volume" else out.get("close", pd.Series(dtype=float))
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out[OHLCV_COLUMNS]
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.dropna(subset=["close"])


def timeframe_to_pandas(timeframe: str) -> str:
    """'1h' -> '1h', '1d' -> '1D' — pandas offset aliases."""
    table = {
        "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
        "1h": "1h", "2h": "2h", "4h": "4h", "12h": "12h",
        "1d": "1D", "1w": "1W",
    }
    if timeframe not in table:
        raise ValueError(f"Unsupported timeframe {timeframe!r}. Use one of {sorted(table)}")
    return table[timeframe]


def timeframe_minutes(timeframe: str) -> int:
    table = {
        "1m": 1, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240, "12h": 720,
        "1d": 1440, "1w": 10080,
    }
    if timeframe not in table:
        raise ValueError(f"Unsupported timeframe {timeframe!r}")
    return table[timeframe]
