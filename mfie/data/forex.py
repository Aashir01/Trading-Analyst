"""Forex market data: OANDA (execution-grade + retail positioning) and Alpha Vantage."""

from __future__ import annotations

import pandas as pd

from mfie.core.types import Instrument, Quote
from mfie.core.utils import get_logger, safe_div
from mfie.data.base import HttpClient, normalize_ohlcv

log = get_logger(__name__)

_OANDA_GRANULARITY = {
    "1m": "M1", "5m": "M5", "15m": "M15", "30m": "M30",
    "1h": "H1", "2h": "H2", "4h": "H4", "12h": "H12",
    "1d": "D", "1w": "W",
}

_AV_INTERVAL = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "60min"}


def _oanda_symbol(instrument: Instrument) -> str:
    return f"{instrument.base}_{instrument.quote}"


class OandaProvider:
    """OANDA v20 REST. Requires ``OANDA_API_KEY``.

    Beyond pricing, OANDA publishes aggregate retail positioning, which is the
    single most useful input to the contrarian sentiment override.
    """

    name = "oanda"

    def __init__(self, http: HttpClient) -> None:
        self.http = http
        self.key = http.settings.oanda_api_key
        self.account = http.settings.oanda_account_id
        env = http.settings.oanda_environment
        self.base = (
            "https://api-fxtrade.oanda.com" if env == "live"
            else "https://api-fxpractice.oanda.com"
        )

    @property
    def available(self) -> bool:
        return bool(self.key)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.key}", "Accept-Datetime-Format": "RFC3339"}

    def ohlcv(self, instrument: Instrument, timeframe: str = "1h", limit: int = 500) -> pd.DataFrame | None:
        if not self.available:
            return None
        granularity = _OANDA_GRANULARITY.get(timeframe)
        if granularity is None:
            return None
        data = self.http.get_json(
            f"{self.base}/v3/instruments/{_oanda_symbol(instrument)}/candles",
            params={"granularity": granularity, "count": min(limit, 5000), "price": "M"},
            headers=self._headers(),
        )
        if not isinstance(data, dict) or not data.get("candles"):
            return None
        rows = []
        for candle in data["candles"]:
            if not candle.get("complete", True):
                continue
            mid = candle["mid"]
            rows.append(
                {
                    "ts": candle["time"],
                    "open": float(mid["o"]),
                    "high": float(mid["h"]),
                    "low": float(mid["l"]),
                    "close": float(mid["c"]),
                    "volume": float(candle.get("volume", 0)),
                }
            )
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
        return normalize_ohlcv(df.set_index("ts"))

    def quote(self, instrument: Instrument) -> Quote | None:
        if not self.available or not self.account:
            return None
        data = self.http.get_json(
            f"{self.base}/v3/accounts/{self.account}/pricing",
            params={"instruments": _oanda_symbol(instrument)},
            headers=self._headers(),
            use_cache=False,
        )
        if not isinstance(data, dict) or not data.get("prices"):
            return None
        price = data["prices"][0]
        bids, asks = price.get("bids"), price.get("asks")
        if not bids or not asks:
            return None
        return Quote(symbol=instrument.symbol, bid=float(bids[0]["price"]), ask=float(asks[0]["price"]))

    def retail_positioning(self, instrument: Instrument) -> float | None:
        """Fraction of retail accounts net-long (0..1)."""
        if not self.available:
            return None
        data = self.http.get_json(
            f"{self.base}/v3/instruments/{_oanda_symbol(instrument)}/positionBook",
            headers=self._headers(),
        )
        if not isinstance(data, dict) or "positionBook" not in data:
            return None
        buckets = data["positionBook"].get("buckets", [])
        longs = sum(float(b.get("longCountPercent", 0)) for b in buckets)
        shorts = sum(float(b.get("shortCountPercent", 0)) for b in buckets)
        total = longs + shorts
        if total <= 0:
            return None
        return safe_div(longs, total, 0.5)


class AlphaVantageProvider:
    """Free-tier friendly FX bars and sovereign yields. Requires an API key.

    Note the free tier is heavily rate-limited (a handful of calls per minute),
    so results are cached aggressively upstream in ``HttpClient``.
    """

    name = "alphavantage"
    BASE = "https://www.alphavantage.co/query"

    def __init__(self, http: HttpClient) -> None:
        self.http = http
        self.key = http.settings.alphavantage_api_key

    @property
    def available(self) -> bool:
        return bool(self.key)

    def ohlcv(self, instrument: Instrument, timeframe: str = "1h", limit: int = 500) -> pd.DataFrame | None:
        if not self.available:
            return None
        if timeframe in _AV_INTERVAL:
            params = {
                "function": "FX_INTRADAY",
                "from_symbol": instrument.base,
                "to_symbol": instrument.quote,
                "interval": _AV_INTERVAL[timeframe],
                "outputsize": "full",
                "apikey": self.key,
            }
            series_key = f"Time Series FX ({_AV_INTERVAL[timeframe]})"
        elif timeframe == "1d":
            params = {
                "function": "FX_DAILY",
                "from_symbol": instrument.base,
                "to_symbol": instrument.quote,
                "outputsize": "full",
                "apikey": self.key,
            }
            series_key = "Time Series FX (Daily)"
        else:
            return None

        data = self.http.get_json(self.BASE, params=params)
        if not isinstance(data, dict) or series_key not in data:
            if isinstance(data, dict) and ("Note" in data or "Information" in data):
                log.warning("Alpha Vantage throttled: %s", data.get("Note") or data.get("Information"))
            return None

        rows = data[series_key]
        df = pd.DataFrame.from_dict(rows, orient="index").rename(
            columns={"1. open": "open", "2. high": "high", "3. low": "low", "4. close": "close"}
        )
        df.index = pd.to_datetime(df.index, utc=True)
        df["volume"] = 0.0
        return normalize_ohlcv(df).tail(limit)

    def treasury_yield(self, maturity: str = "10year") -> float | None:
        """Latest US Treasury constant-maturity yield as a decimal."""
        if not self.available:
            return None
        data = self.http.get_json(
            self.BASE,
            params={
                "function": "TREASURY_YIELD",
                "interval": "daily",
                "maturity": maturity,
                "apikey": self.key,
            },
        )
        if not isinstance(data, dict) or not data.get("data"):
            return None
        for row in data["data"]:
            try:
                return float(row["value"]) / 100.0
            except (KeyError, TypeError, ValueError):
                continue
        return None
