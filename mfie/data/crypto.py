"""Crypto market data: Binance (price / order book / funding) and CoinGecko."""

from __future__ import annotations

import pandas as pd

from mfie.core.types import Instrument, Quote
from mfie.core.utils import get_logger
from mfie.data.base import HttpClient, normalize_ohlcv

log = get_logger(__name__)

_BINANCE_INTERVALS = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "2h": "2h", "4h": "4h", "12h": "12h",
    "1d": "1d", "1w": "1w",
}

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
COINGECKO_PRO_BASE = "https://pro-api.coingecko.com/api/v3"
BINANCE_FUTURES_BASE = "https://fapi.binance.com"


class BinanceProvider:
    """Spot klines, top-of-book and perpetual funding rates.

    Public endpoints only — no API key needed for market data, which is why
    this is the default live crypto source.
    """

    name = "binance"

    def __init__(self, http: HttpClient) -> None:
        self.http = http
        self.base = http.settings.binance_base_url.rstrip("/")

    def ohlcv(self, instrument: Instrument, timeframe: str = "1h", limit: int = 500) -> pd.DataFrame | None:
        interval = _BINANCE_INTERVALS.get(timeframe)
        if interval is None:
            return None
        data = self.http.get_json(
            f"{self.base}/api/v3/klines",
            params={"symbol": instrument.symbol, "interval": interval, "limit": min(limit, 1000)},
        )
        if not isinstance(data, list) or not data:
            return None
        df = pd.DataFrame(
            data,
            columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades",
                "taker_base", "taker_quote", "ignore",
            ],
        )
        df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        return normalize_ohlcv(df.set_index("ts")[["open", "high", "low", "close", "volume"]])

    def quote(self, instrument: Instrument) -> Quote | None:
        data = self.http.get_json(
            f"{self.base}/api/v3/ticker/bookTicker",
            params={"symbol": instrument.symbol},
            use_cache=False,
        )
        if not isinstance(data, dict) or "bidPrice" not in data:
            return None
        return Quote(
            symbol=instrument.symbol,
            bid=float(data["bidPrice"]),
            ask=float(data["askPrice"]),
        )

    def order_book_depth(self, instrument: Instrument, limit: int = 100) -> dict | None:
        """Raw depth, used for order-flow imbalance in the microstructure module."""
        return self.http.get_json(
            f"{self.base}/api/v3/depth",
            params={"symbol": instrument.symbol, "limit": limit},
            use_cache=False,
        )

    def funding_rate(self, instrument: Instrument, periods: int = 90) -> pd.Series | None:
        """Historical perpetual funding, the input to the cash-and-carry strategy."""
        data = self.http.get_json(
            f"{BINANCE_FUTURES_BASE}/fapi/v1/fundingRate",
            params={"symbol": instrument.symbol, "limit": min(periods, 1000)},
        )
        if not isinstance(data, list) or not data:
            return None
        df = pd.DataFrame(data)
        idx = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        return pd.Series(df["fundingRate"].astype(float).values, index=idx, name="funding_rate")


class CoinGeckoProvider:
    """Fundamentals: market cap, aggregate volume, stablecoin supply.

    Free tier works without a key; ``COINGECKO_API_KEY`` switches to the Pro
    host and raises the rate limit.
    """

    name = "coingecko"

    def __init__(self, http: HttpClient) -> None:
        self.http = http
        self.key = http.settings.coingecko_api_key
        self.base = COINGECKO_PRO_BASE if self.key else COINGECKO_BASE

    def _headers(self) -> dict[str, str]:
        return {"x-cg-pro-api-key": self.key} if self.key else {}

    def market_chart(self, coingecko_id: str, days: int = 180) -> dict | None:
        return self.http.get_json(
            f"{self.base}/coins/{coingecko_id}/market_chart",
            params={"vs_currency": "usd", "days": days, "interval": "daily"},
            headers=self._headers(),
        )

    def market_cap(self, instrument: Instrument, days: int = 180) -> pd.Series | None:
        if not instrument.coingecko_id:
            return None
        data = self.market_chart(instrument.coingecko_id, days)
        if not isinstance(data, dict) or "market_caps" not in data:
            return None
        rows = data["market_caps"]
        idx = pd.to_datetime([r[0] for r in rows], unit="ms", utc=True)
        return pd.Series([float(r[1]) for r in rows], index=idx, name="market_cap")

    def total_volume(self, instrument: Instrument, days: int = 180) -> pd.Series | None:
        """Exchange-traded volume. A usable stand-in for on-chain volume when
        Glassnode is unavailable — noisier, but the same economic role in NVT."""
        if not instrument.coingecko_id:
            return None
        data = self.market_chart(instrument.coingecko_id, days)
        if not isinstance(data, dict) or "total_volumes" not in data:
            return None
        rows = data["total_volumes"]
        idx = pd.to_datetime([r[0] for r in rows], unit="ms", utc=True)
        return pd.Series([float(r[1]) for r in rows], index=idx, name="volume")

    def stablecoin_supply(self, days: int = 180) -> pd.Series | None:
        """USDT + USDC aggregate market cap — the crypto-native liquidity proxy."""
        total: pd.Series | None = None
        for coin in ("tether", "usd-coin"):
            data = self.market_chart(coin, days)
            if not isinstance(data, dict) or "market_caps" not in data:
                continue
            rows = data["market_caps"]
            idx = pd.to_datetime([r[0] for r in rows], unit="ms", utc=True)
            series = pd.Series([float(r[1]) for r in rows], index=idx)
            total = series if total is None else total.add(series, fill_value=0.0)
        if total is None:
            return None
        return total.rename("stablecoin_supply")
