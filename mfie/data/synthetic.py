"""Deterministic synthetic market + macro data.

This is the fallback that makes the whole engine runnable with zero API keys,
and it is also what the test-suite runs against. It is *not* a market
simulator for research — it is a fixture generator. Prices come from a
regime-switching GBM with volatility clustering so that trend, range and crash
regimes all appear and every downstream filter gets exercised.

Determinism: every series is seeded from the symbol name, so the same symbol
always produces the same history. Backtests and tests are reproducible.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from mfie.core.types import EconomicEvent, Instrument, Quote
from mfie.core.universe import CURRENCY_TO_COUNTRY, TRACKED_CURRENCIES
from mfie.data.base import normalize_ohlcv, timeframe_minutes

# Rough starting levels so charts look plausible.
_ANCHOR_PRICES = {
    "BTC": 68_000.0, "ETH": 3_400.0, "SOL": 165.0, "BNB": 580.0,
    "XRP": 0.62, "ADA": 0.48, "AVAX": 36.0, "LINK": 17.5,
    "EURUSD": 1.0850, "GBPUSD": 1.2700, "USDJPY": 152.0, "USDCHF": 0.9050,
    "AUDUSD": 0.6600, "NZDUSD": 0.6050, "USDCAD": 1.3650,
    "EURGBP": 0.8540, "EURJPY": 164.5, "GBPJPY": 193.0,
}

# Plausible-but-fabricated macro levels, as decimals (0.0525 == 5.25%).
_POLICY_RATES = {
    "USD": 0.0450, "EUR": 0.0325, "GBP": 0.0475, "JPY": 0.0025,
    "CHF": 0.0125, "AUD": 0.0410, "NZD": 0.0450, "CAD": 0.0400,
}
_INFLATION = {
    "USD": 0.0290, "EUR": 0.0230, "GBP": 0.0310, "JPY": 0.0260,
    "CHF": 0.0110, "AUD": 0.0340, "NZD": 0.0320, "CAD": 0.0250,
}
_YIELD_10Y = {
    "USD": 0.0415, "EUR": 0.0250, "GBP": 0.0405, "JPY": 0.0105,
    "CHF": 0.0060, "AUD": 0.0430, "NZD": 0.0455, "CAD": 0.0340,
}
_YIELD_2Y = {
    "USD": 0.0445, "EUR": 0.0280, "GBP": 0.0430, "JPY": 0.0035,
    "CHF": 0.0085, "AUD": 0.0400, "NZD": 0.0470, "CAD": 0.0355,
}

_ANNUAL_VOL = {"crypto": 0.65, "forex": 0.09}


def _seed(*parts: str) -> np.random.Generator:
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _daily_index(periods: int, freq: str = "D") -> pd.DatetimeIndex:
    """A calendar-aligned index ending today.

    Anchoring on ``datetime.now()`` gives every series a different sub-second
    stamp, so two daily series generated microseconds apart share *no* index
    values and any arithmetic between them (10Y minus 2Y, for instance) silently
    produces an empty result. Normalising to midnight makes them align.
    """
    return pd.date_range(
        end=pd.Timestamp.now(tz="UTC").normalize(), periods=periods, freq=freq, tz="UTC"
    )


class SyntheticProvider:
    """Generates every data block the engine consumes."""

    name = "synthetic"

    # ---------------------------------------------------------------- market
    def ohlcv(self, instrument: Instrument, timeframe: str = "1h", limit: int = 500) -> pd.DataFrame:
        rng = _seed(instrument.symbol, timeframe)
        minutes = timeframe_minutes(timeframe)
        periods_per_year = (365 * 24 * 60) / minutes

        anchor = _ANCHOR_PRICES.get(instrument.base) or _ANCHOR_PRICES.get(instrument.symbol, 100.0)
        base_vol = _ANNUAL_VOL["crypto" if instrument.is_crypto else "forex"]
        sigma = base_vol / np.sqrt(periods_per_year)

        # --- regime-switching drift + vol clustering (GARCH-ish) ---
        n = int(limit)
        # 3 hidden states: trend-up, range, trend-down/crash
        transition = np.array([[0.97, 0.02, 0.01],
                               [0.03, 0.94, 0.03],
                               [0.02, 0.03, 0.95]])
        drifts = np.array([0.9, 0.0, -1.1]) * sigma * 0.35
        vol_mult = np.array([1.0, 0.65, 1.8])

        state = rng.integers(0, 3)
        states = np.empty(n, dtype=int)
        for i in range(n):
            states[i] = state
            state = rng.choice(3, p=transition[state])

        # volatility clustering: sigma_t^2 = w + a*e^2 + b*sigma_{t-1}^2
        eps = rng.standard_normal(n)
        var = np.empty(n)
        var[0] = sigma**2
        a, b = 0.08, 0.90
        w = sigma**2 * (1 - a - b)
        for i in range(1, n):
            var[i] = w + a * (eps[i - 1] * np.sqrt(var[i - 1])) ** 2 + b * var[i - 1]

        vol = np.sqrt(var) * vol_mult[states]
        idiosyncratic = drifts[states] + vol * eps

        # Blend in a shared market factor. Real crypto majors move together
        # (a single beta to "the market"), and FX majors share a dollar factor.
        # Without this, every synthetic symbol is independent and the
        # correlation/cointegration machinery has nothing to find.
        beta = self._market_beta(instrument)
        if beta > 0:
            factor = self._market_factor(instrument, timeframe, n, sigma)
            returns = beta * factor + np.sqrt(max(1.0 - beta**2, 0.0)) * idiosyncratic
        else:
            returns = idiosyncratic

        close = anchor * np.exp(np.cumsum(returns))
        # Re-anchor so the final price sits near the anchor level.
        close = close * (anchor / close[-1])

        open_ = np.concatenate([[close[0] * (1 - returns[0])], close[:-1]])
        wick = np.abs(rng.standard_normal(n)) * vol * close * 0.6
        high = np.maximum(open_, close) + wick
        low = np.minimum(open_, close) - wick
        low = np.maximum(low, 1e-9)

        base_volume = 1_500.0 if instrument.is_crypto else 45_000.0
        volume = base_volume * np.exp(rng.standard_normal(n) * 0.4) * (1 + 2 * np.abs(returns) / sigma)

        end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        index = pd.date_range(end=end, periods=n, freq=f"{minutes}min", tz="UTC")

        return normalize_ohlcv(
            pd.DataFrame(
                {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
                index=index,
            )
        )

    @staticmethod
    def _market_beta(instrument: Instrument) -> float:
        """Loading on the shared factor. Majors carry the market; alts amplify it."""
        if instrument.is_crypto:
            return 0.80 if instrument.base in ("BTC", "ETH") else 0.68
        # FX: only pairs with a USD leg load on the dollar factor.
        if "USD" in (instrument.base, instrument.quote):
            return 0.62
        return 0.35  # crosses inherit the factor indirectly

    @staticmethod
    def _market_factor(instrument: Instrument, timeframe: str, n: int, sigma: float) -> np.ndarray:
        """Common return series for an asset class.

        Signed for FX so that USDJPY and EURUSD respond to a dollar move in
        opposite directions, exactly as they do in the real market.
        """
        family = "crypto" if instrument.is_crypto else "fx"
        rng = _seed("market_factor", family, timeframe)
        factor = rng.standard_normal(n) * sigma
        if not instrument.is_crypto:
            # A stronger dollar lifts USDxxx and depresses xxxUSD.
            sign = 1.0 if instrument.base == "USD" else -1.0
            factor = factor * sign
        return factor

    def quote(self, instrument: Instrument) -> Quote:
        rng = _seed(instrument.symbol, "quote")
        last = float(self.ohlcv(instrument, "1h", 60)["close"].iloc[-1])
        # Crypto majors ~1-3 bps, FX majors ~0.5-2 bps, widened randomly.
        base_bps = 2.0 if instrument.is_crypto else 1.0
        spread = last * (base_bps + rng.exponential(1.5)) / 10_000
        return Quote(symbol=instrument.symbol, bid=last - spread / 2, ask=last + spread / 2)

    # ------------------------------------------------------------------ macro
    def policy_rates(self) -> dict[str, float]:
        return dict(_POLICY_RATES)

    def inflation(self) -> dict[str, float]:
        return dict(_INFLATION)

    def yields_10y(self) -> dict[str, float]:
        return dict(_YIELD_10Y)

    def yields_2y(self) -> dict[str, float]:
        return dict(_YIELD_2Y)

    def m2_series(self, months: int = 36) -> pd.Series:
        """Global M2 proxy in USD trillions, growing with a mid-series squeeze."""
        rng = _seed("global_m2")
        idx = _daily_index(months, freq="ME")
        trend = np.linspace(0, 0.06, months)
        squeeze = -0.035 * np.exp(-((np.arange(months) - months * 0.55) ** 2) / (2 * (months * 0.12) ** 2))
        noise = np.cumsum(rng.standard_normal(months) * 0.002)
        return pd.Series(105.0 * np.exp(trend + squeeze + noise), index=idx, name="m2")

    def stablecoin_supply(self, days: int = 180) -> pd.Series:
        """Aggregate USDT+USDC supply in USD billions."""
        rng = _seed("stablecoin_supply")
        idx = _daily_index(days)
        drift = np.linspace(0, 0.09, days)
        noise = np.cumsum(rng.standard_normal(days) * 0.0025)
        return pd.Series(148.0 * np.exp(drift + noise), index=idx, name="stablecoin_supply")

    # --------------------------------------- historical paths (cycle engine)
    # Policy rates move in persistent hiking/cutting cycles, not random walks —
    # the shape matters because the cycle engine differentiates these series.
    def policy_rate_series(self, currency: str, days: int = 900) -> pd.Series:
        rng = _seed("policy_path", currency)
        idx = _daily_index(days)
        target = _POLICY_RATES.get(currency, 0.03)
        # A hiking cycle that plateaus and begins to ease — a full arc, so the
        # real-rate impulse factor has something to detect.
        t = np.linspace(0, 1, days)
        arc = np.sin(np.pi * np.clip(t * 1.15, 0, 1)) * 0.012
        path = target - 0.010 + arc + np.cumsum(rng.standard_normal(days) * 0.00004)
        return pd.Series(np.maximum(path, 0.0), index=idx, name=f"policy_{currency}")

    def inflation_series(self, currency: str, days: int = 900) -> pd.Series:
        rng = _seed("cpi_path", currency)
        idx = _daily_index(days)
        target = _INFLATION.get(currency, 0.025)
        # Inflation overshoots then decays back toward target, lagging policy.
        t = np.linspace(0, 1, days)
        hump = 0.020 * np.exp(-((t - 0.25) ** 2) / (2 * 0.18**2))
        path = target - 0.004 + hump + np.cumsum(rng.standard_normal(days) * 0.00005)
        return pd.Series(path, index=idx, name=f"cpi_{currency}")

    def yield_series(self, currency: str, tenor: str = "10y", days: int = 900) -> pd.Series:
        rng = _seed("yield_path", currency, tenor)
        idx = _daily_index(days)
        anchor = (_YIELD_10Y if tenor == "10y" else _YIELD_2Y).get(currency, 0.03)
        t = np.linspace(0, 1, days)
        # The short end leads the long end, so the curve inverts mid-sample and
        # then bull-steepens — the exact sequence the credit factor keys on.
        shape = (0.010 * np.sin(np.pi * t) if tenor == "2y" else 0.004 * np.sin(np.pi * t * 0.8))
        path = anchor - shape[-1] + shape + np.cumsum(rng.standard_normal(days) * 0.00012)
        return pd.Series(path, index=idx, name=f"{tenor}_{currency}")

    def fear_greed_history(self, days: int = 720) -> pd.Series:
        rng = _seed("fear_greed_history")
        idx = _daily_index(days)
        # Sentiment is strongly autocorrelated: it stays greedy for months.
        x = np.empty(days)
        x[0] = 50.0
        for i in range(1, days):
            x[i] = x[i - 1] + 0.04 * (52.0 - x[i - 1]) + rng.normal(0, 4.5)
        return pd.Series(np.clip(x, 1, 99), index=idx, name="fear_greed")

    def reer_series(self, currency: str, periods: int = 400) -> pd.Series:
        """Real Effective Exchange Rate index, mean-reverting around 100."""
        rng = _seed("reer", currency)
        idx = _daily_index(periods)
        # Ornstein-Uhlenbeck around 100 -> genuine PPP-style mean reversion.
        theta, mu, sigma = 0.02, 100.0, 0.55
        x = np.empty(periods)
        x[0] = mu + rng.standard_normal() * 3
        for i in range(1, periods):
            x[i] = x[i - 1] + theta * (mu - x[i - 1]) + sigma * rng.standard_normal()
        return pd.Series(x, index=idx, name=f"reer_{currency}")

    # ----------------------------------------------------------- on-chain
    def onchain_volume(self, instrument: Instrument, days: int = 180) -> pd.Series:
        """Daily on-chain transfer volume in USD."""
        rng = _seed("onchain", instrument.symbol)
        idx = _daily_index(days)
        scale = {"BTC": 9e9, "ETH": 5e9}.get(instrument.base, 4e8)
        # Declining utility late in the series creates the velocity divergence
        # that TokenomicsFilter is designed to catch.
        fade = np.linspace(0, -0.45, days)
        noise = rng.standard_normal(days) * 0.25
        return pd.Series(scale * np.exp(fade + noise), index=idx, name="onchain_volume")

    def market_cap(self, instrument: Instrument, days: int = 180) -> pd.Series:
        px = self.ohlcv(instrument, "1d", days)["close"]
        supply = {"BTC": 19.7e6, "ETH": 120e6, "SOL": 470e6, "BNB": 145e6,
                  "XRP": 57e9, "ADA": 35e9, "AVAX": 400e6, "LINK": 620e6}.get(instrument.base, 1e9)
        return pd.Series(px.values * supply, index=px.index, name="market_cap")

    # ---------------------------------------------------------- sentiment
    def fear_greed(self) -> float:
        rng = _seed("fear_greed", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        return float(np.clip(rng.normal(58, 18), 1, 99))

    def retail_positioning(self, instrument: Instrument) -> float:
        """Share of retail accounts net-long, 0..1."""
        rng = _seed("retail", instrument.symbol, datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        return float(np.clip(rng.beta(4.5, 3.0), 0.02, 0.98))

    def news_sentiment(self, instrument: Instrument) -> float:
        """Aggregate headline sentiment, -1..+1."""
        rng = _seed("news", instrument.symbol, datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        return float(np.clip(rng.normal(0.05, 0.35), -1, 1))

    # ----------------------------------------------------------- calendar
    def economic_events(self, days_ahead: int = 7, days_back: int = 30) -> list[EconomicEvent]:
        rng = _seed("calendar", datetime.now(timezone.utc).strftime("%Y-%W"))
        now = datetime.now(timezone.utc)
        templates = [
            ("CPI y/y", "high", 0.031, 0.4),
            ("Core CPI m/m", "high", 0.003, 0.1),
            ("Interest Rate Decision", "high", 0.045, 0.25),
            ("Non-Farm Payrolls", "high", 185.0, 45.0),
            ("Unemployment Rate", "medium", 0.042, 0.2),
            ("Manufacturing PMI", "medium", 50.4, 1.6),
            ("Retail Sales m/m", "medium", 0.004, 0.3),
            ("GDP q/q", "high", 0.006, 0.3),
        ]
        events: list[EconomicEvent] = []
        for currency in TRACKED_CURRENCIES:
            for name, impact, level, sd in templates:
                # Roughly one release of each type per currency per month.
                offset_days = rng.uniform(-days_back, days_ahead)
                ts = now + timedelta(days=float(offset_days),
                                     hours=float(rng.integers(6, 20)))
                forecast = float(level * (1 + rng.normal(0, 0.02)))
                actual = None
                if ts < now:  # already released
                    actual = float(forecast + rng.normal(0, sd * abs(level) * 0.06 + sd * 0.02))
                events.append(
                    EconomicEvent(
                        ts=ts,
                        country=CURRENCY_TO_COUNTRY.get(currency, currency[:2]),
                        currency=currency,
                        name=name,
                        impact=impact,
                        actual=actual,
                        forecast=forecast,
                        previous=float(forecast * (1 + rng.normal(0, 0.03))),
                    )
                )
        # A guaranteed upcoming US high-impact print, placed deliberately just
        # *outside* the default blackout window: close enough to show up in the
        # calendar panel, far enough that the demo is not blocked end to end.
        events.append(
            EconomicEvent(
                ts=now + timedelta(minutes=95),
                country="US",
                currency="USD",
                name="FOMC Rate Decision",
                impact="high",
                forecast=0.045,
                previous=0.045,
            )
        )
        return sorted(events, key=lambda e: e.ts)

    def funding_rate(self, instrument: Instrument, periods: int = 90) -> pd.Series:
        """Perpetual-swap funding rate per 8h interval."""
        rng = _seed("funding", instrument.symbol)
        idx = _daily_index(periods, freq="8h")
        return pd.Series(rng.normal(0.0001, 0.00025, periods), index=idx, name="funding_rate")
