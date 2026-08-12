"""DataHub — the single entry point every other layer uses to get data.

Responsibilities:

1. Pick the best available provider per data block, given which API keys exist.
2. Fall back to the synthetic provider when a provider is missing or fails, so
   the pipeline never raises on a network problem.
3. Record which source served each block in ``self.sources``, so the dashboard
   can honestly label live data vs. simulated data.
"""

from __future__ import annotations

import functools
from datetime import datetime, timezone

import pandas as pd

from mfie.config import Params, Settings, get_params, get_settings
from mfie.core.types import EconomicEvent, Instrument, Quote
from mfie.core.universe import TRACKED_CURRENCIES
from mfie.core.utils import get_logger
from mfie.data.base import HttpClient
from mfie.data.crypto import BinanceProvider, CoinGeckoProvider
from mfie.data.forex import AlphaVantageProvider, OandaProvider
from mfie.data.macro import FredProvider, TradingEconomicsProvider
from mfie.data.onchain import GlassnodeProvider
from mfie.data.sentiment import FearGreedProvider, LunarCrushProvider, NewsProvider
from mfie.data.synthetic import SyntheticProvider

log = get_logger(__name__)


def _is_empty(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return bool(value.empty)
    if isinstance(value, (list, dict, set, tuple)):
        return len(value) == 0
    return False


class DataHub:
    def __init__(self, settings: Settings | None = None, params: Params | None = None) -> None:
        self.settings = settings or get_settings()
        self.params = params or get_params()
        self.http = HttpClient(self.settings)
        self.synthetic = SyntheticProvider()
        self.sources: dict[str, str] = {}

        self.binance = BinanceProvider(self.http)
        self.coingecko = CoinGeckoProvider(self.http)
        self.oanda = OandaProvider(self.http)
        self.alphavantage = AlphaVantageProvider(self.http)
        self.fred = FredProvider(self.http)
        self.tradingeconomics = TradingEconomicsProvider(self.http)
        self.glassnode = GlassnodeProvider(self.http)
        self.fear_greed_api = FearGreedProvider(self.http)
        self.news = NewsProvider(self.http)
        self.lunarcrush = LunarCrushProvider(self.http)

    # ------------------------------------------------------------------ helper
    @property
    def offline(self) -> bool:
        return self.settings.offline

    def _record(self, block: str, source: str) -> None:
        self.sources[block] = source

    def _try(self, block: str, source: str, fn, *args, **kwargs):
        """Run a live provider call; on failure or empty result return None."""
        if self.offline:
            return None
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:  # providers must never break the pipeline
            log.warning("%s provider %s failed: %s", block, source, exc)
            return None
        if result is None:
            return None
        if isinstance(result, (pd.DataFrame, pd.Series)) and result.empty:
            return None
        if isinstance(result, (list, dict)) and len(result) == 0:
            return None
        # A tuple whose parts are all empty is a failure wearing a success
        # costume — without this, a provider that served nothing still gets
        # credited as a live source in the provenance panel.
        if isinstance(result, tuple) and result and all(_is_empty(part) for part in result):
            return None
        self._record(block, source)
        return result

    # ------------------------------------------------------------------ market
    def ohlcv(self, instrument: Instrument, timeframe: str = "1h", limit: int = 500) -> pd.DataFrame:
        block = f"ohlcv:{instrument.symbol}"
        if instrument.is_crypto:
            result = self._try(block, "binance", self.binance.ohlcv, instrument, timeframe, limit)
        else:
            result = self._try(block, "oanda", self.oanda.ohlcv, instrument, timeframe, limit)
            if result is None:
                result = self._try(block, "alphavantage", self.alphavantage.ohlcv,
                                   instrument, timeframe, limit)
        if result is None or len(result) < 30:
            self._record(block, "synthetic")
            return self.synthetic.ohlcv(instrument, timeframe, limit)
        return result

    def quote(self, instrument: Instrument) -> Quote:
        block = f"quote:{instrument.symbol}"
        if instrument.is_crypto:
            result = self._try(block, "binance", self.binance.quote, instrument)
        else:
            result = self._try(block, "oanda", self.oanda.quote, instrument)
        if result is None:
            self._record(block, "synthetic")
            return self.synthetic.quote(instrument)
        return result

    def funding_rate(self, instrument: Instrument, periods: int = 90) -> pd.Series:
        block = f"funding:{instrument.symbol}"
        if instrument.is_crypto:
            result = self._try(block, "binance", self.binance.funding_rate, instrument, periods)
            if result is not None:
                return result
        self._record(block, "synthetic")
        return self.synthetic.funding_rate(instrument, periods)

    # ------------------------------------------------------------------- macro
    def policy_rates(self) -> dict[str, float]:
        result = self._try("policy_rates", "fred", self.fred.policy_rates, TRACKED_CURRENCIES)
        if result:
            # Fill any currency FRED could not serve, so RIRD always has both legs.
            merged = dict(self.synthetic.policy_rates())
            merged.update(result)
            return merged
        self._record("policy_rates", "synthetic")
        return self.synthetic.policy_rates()

    def inflation(self) -> dict[str, float]:
        result = self._try("inflation", "fred", self.fred.inflation, TRACKED_CURRENCIES)
        if result:
            merged = dict(self.synthetic.inflation())
            merged.update(result)
            return merged
        self._record("inflation", "synthetic")
        return self.synthetic.inflation()

    def yields(self) -> tuple[dict[str, float], dict[str, float]]:
        result = self._try("yields", "fred", self.fred.yields, TRACKED_CURRENCIES)
        if result:
            long_end, short_end = result
            merged_long = dict(self.synthetic.yields_10y())
            merged_long.update(long_end)
            merged_short = dict(self.synthetic.yields_2y())
            merged_short.update(short_end)
            return merged_long, merged_short
        self._record("yields", "synthetic")
        return self.synthetic.yields_10y(), self.synthetic.yields_2y()

    def m2_series(self, observations: int = 60) -> pd.Series:
        result = self._try("m2", "fred", self.fred.m2_series, observations)
        if result is not None:
            return result
        self._record("m2", "synthetic")
        return self.synthetic.m2_series(observations)

    def stablecoin_supply(self, days: int = 180) -> pd.Series:
        result = self._try("stablecoin", "glassnode", self.glassnode.stablecoin_supply, days)
        if result is None:
            result = self._try("stablecoin", "coingecko", self.coingecko.stablecoin_supply, days)
        if result is not None:
            return result
        self._record("stablecoin", "synthetic")
        return self.synthetic.stablecoin_supply(days)

    # ------------------------------------------- historical paths (cycle engine)
    def policy_rate_series(self, currency: str, days: int = 900) -> pd.Series:
        block = f"policy_path:{currency}"
        result = self._try(block, "fred", self.fred.policy_rate_series, currency, days)
        if result is not None:
            return result
        self._record(block, "synthetic")
        return self.synthetic.policy_rate_series(currency, days)

    def inflation_series(self, currency: str, days: int = 900) -> pd.Series:
        block = f"cpi_path:{currency}"
        result = self._try(block, "fred", self.fred.inflation_series, currency)
        if result is not None:
            return result
        self._record(block, "synthetic")
        return self.synthetic.inflation_series(currency, days)

    def yield_series(self, currency: str, tenor: str = "10y", days: int = 900) -> pd.Series:
        block = f"yield_path:{currency}:{tenor}"
        result = self._try(block, "fred", self.fred.yield_series, currency, tenor, days)
        if result is not None:
            return result
        self._record(block, "synthetic")
        return self.synthetic.yield_series(currency, tenor, days)

    def fear_greed_history(self, days: int = 720) -> pd.Series:
        block = "fear_greed_history"
        rows = self._try(block, "alternative.me", self.fear_greed_api.history, min(days, 2000))
        if rows:
            idx = pd.to_datetime([r[0] for r in rows], unit="s", utc=True)
            return pd.Series([r[1] for r in rows], index=idx, name="fear_greed").sort_index()
        self._record(block, "synthetic")
        return self.synthetic.fear_greed_history(days)

    def reer_series(self, currency: str, observations: int = 400) -> pd.Series:
        block = f"reer:{currency}"
        result = self._try(block, "fred", self.fred.reer_series, currency, observations)
        if result is not None:
            return result
        self._record(block, "synthetic")
        return self.synthetic.reer_series(currency, observations)

    def economic_events(self, days_ahead: int = 7, days_back: int = 30) -> list[EconomicEvent]:
        result = self._try("calendar", "tradingeconomics", self.tradingeconomics.economic_events,
                           days_ahead, days_back)
        if result:
            return result
        self._record("calendar", "synthetic")
        return self.synthetic.economic_events(days_ahead, days_back)

    # ---------------------------------------------------------------- on-chain
    def onchain_volume(self, instrument: Instrument, days: int = 180) -> pd.Series:
        block = f"onchain:{instrument.symbol}"
        result = self._try(block, "glassnode", self.glassnode.transfer_volume_usd, instrument, days)
        if result is None:
            result = self._try(block, "coingecko", self.coingecko.total_volume, instrument, days)
        if result is not None:
            return result
        self._record(block, "synthetic")
        return self.synthetic.onchain_volume(instrument, days)

    def market_cap(self, instrument: Instrument, days: int = 180) -> pd.Series:
        block = f"mcap:{instrument.symbol}"
        result = self._try(block, "coingecko", self.coingecko.market_cap, instrument, days)
        if result is None:
            result = self._try(block, "glassnode", self.glassnode.market_cap, instrument, days)
        if result is not None:
            return result
        self._record(block, "synthetic")
        return self.synthetic.market_cap(instrument, days)

    # ---------------------------------------------------------------- sentiment
    def fear_greed(self) -> float:
        result = self._try("fear_greed", "alternative.me", self.fear_greed_api.latest)
        if result is not None:
            return float(result)
        self._record("fear_greed", "synthetic")
        return self.synthetic.fear_greed()

    def retail_positioning(self, instrument: Instrument) -> float:
        block = f"retail:{instrument.symbol}"
        if instrument.is_forex:
            result = self._try(block, "oanda", self.oanda.retail_positioning, instrument)
            if result is not None:
                return float(result)
        self._record(block, "synthetic")
        return self.synthetic.retail_positioning(instrument)

    def news_sentiment(self, instrument: Instrument) -> float:
        block = f"news:{instrument.symbol}"
        result = self._try(block, "newsapi", self.news.sentiment, instrument)
        if result is not None:
            return float(result)
        self._record(block, "synthetic")
        return self.synthetic.news_sentiment(instrument)

    # ------------------------------------------------------------------ status
    def provider_status(self) -> dict[str, bool]:
        """Which live providers are configured. Drives the dashboard status panel."""
        return {
            "binance (crypto price)": not self.offline,
            "coingecko (crypto fundamentals)": not self.offline,
            "oanda (fx price + positioning)": self.oanda.available and not self.offline,
            "alphavantage (fx fallback)": self.alphavantage.available and not self.offline,
            "fred (rates, cpi, m2, reer)": self.fred.available and not self.offline,
            "tradingeconomics (calendar)": self.tradingeconomics.available and not self.offline,
            "glassnode (on-chain)": self.glassnode.available and not self.offline,
            "alternative.me (fear & greed)": not self.offline,
            "newsapi (headlines)": self.news.available and not self.offline,
            "lunarcrush (social)": self.lunarcrush.available and not self.offline,
        }

    def snapshot_ts(self) -> datetime:
        return datetime.now(timezone.utc)


@functools.lru_cache(maxsize=1)
def get_hub() -> DataHub:
    return DataHub()
