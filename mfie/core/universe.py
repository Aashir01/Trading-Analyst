"""The tradable universe: symbols mapped to the economies behind them.

The mapping from a symbol to ``base``/``quote`` currencies is what lets the
macro layer work at all — without it, ``EURUSD`` is just a line on a chart
rather than a claim on the interest-rate differential between two central banks.
"""

from __future__ import annotations

from mfie.core.types import AssetClass, Instrument

# --------------------------------------------------------------------------- #
# Forex
# --------------------------------------------------------------------------- #
FOREX: dict[str, Instrument] = {
    inst.symbol: inst
    for inst in [
        Instrument("EURUSD", AssetClass.FOREX, "EUR", "USD", display="EUR/USD"),
        Instrument("GBPUSD", AssetClass.FOREX, "GBP", "USD", display="GBP/USD"),
        Instrument("USDJPY", AssetClass.FOREX, "USD", "JPY", pip_size=0.01, display="USD/JPY"),
        Instrument("USDCHF", AssetClass.FOREX, "USD", "CHF", display="USD/CHF"),
        Instrument("AUDUSD", AssetClass.FOREX, "AUD", "USD", display="AUD/USD"),
        Instrument("NZDUSD", AssetClass.FOREX, "NZD", "USD", display="NZD/USD"),
        Instrument("USDCAD", AssetClass.FOREX, "USD", "CAD", display="USD/CAD"),
        Instrument("EURGBP", AssetClass.FOREX, "EUR", "GBP", display="EUR/GBP"),
        Instrument("EURJPY", AssetClass.FOREX, "EUR", "JPY", pip_size=0.01, display="EUR/JPY"),
        Instrument("GBPJPY", AssetClass.FOREX, "GBP", "JPY", pip_size=0.01, display="GBP/JPY"),
    ]
}

# --------------------------------------------------------------------------- #
# Crypto
# --------------------------------------------------------------------------- #
CRYPTO: dict[str, Instrument] = {
    inst.symbol: inst
    for inst in [
        Instrument("BTCUSDT", AssetClass.CRYPTO, "BTC", "USDT", coingecko_id="bitcoin",
                   pip_size=0.01, display="BTC/USDT"),
        Instrument("ETHUSDT", AssetClass.CRYPTO, "ETH", "USDT", coingecko_id="ethereum",
                   pip_size=0.01, display="ETH/USDT"),
        Instrument("SOLUSDT", AssetClass.CRYPTO, "SOL", "USDT", coingecko_id="solana",
                   pip_size=0.01, display="SOL/USDT"),
        Instrument("BNBUSDT", AssetClass.CRYPTO, "BNB", "USDT", coingecko_id="binancecoin",
                   pip_size=0.01, display="BNB/USDT"),
        Instrument("XRPUSDT", AssetClass.CRYPTO, "XRP", "USDT", coingecko_id="ripple",
                   pip_size=0.0001, display="XRP/USDT"),
        Instrument("ADAUSDT", AssetClass.CRYPTO, "ADA", "USDT", coingecko_id="cardano",
                   pip_size=0.0001, display="ADA/USDT"),
        Instrument("AVAXUSDT", AssetClass.CRYPTO, "AVAX", "USDT", coingecko_id="avalanche-2",
                   pip_size=0.01, display="AVAX/USDT"),
        Instrument("LINKUSDT", AssetClass.CRYPTO, "LINK", "USDT", coingecko_id="chainlink",
                   pip_size=0.001, display="LINK/USDT"),
    ]
}

UNIVERSE: dict[str, Instrument] = {**FOREX, **CRYPTO}

# Majors get their own tier: they carry the market, and the "high-beta" rules
# below deliberately exempt them during contraction regimes.
CRYPTO_MAJORS = {"BTCUSDT", "ETHUSDT"}

# Currencies whose macro series the engine tracks.
TRACKED_CURRENCIES = ("USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD")

CURRENCY_TO_COUNTRY = {
    "USD": "US", "EUR": "EA", "GBP": "GB", "JPY": "JP",
    "CHF": "CH", "AUD": "AU", "NZD": "NZ", "CAD": "CA",
}


def get_instrument(symbol: str) -> Instrument:
    """Look up a symbol, tolerating ``EUR/USD``/``eurusd``/``BTC-USDT`` forms."""
    key = symbol.upper().replace("/", "").replace("-", "").replace("_", "")
    if key in UNIVERSE:
        return UNIVERSE[key]
    # Common alias: BTCUSD -> BTCUSDT
    if key.endswith("USD") and f"{key}T" in UNIVERSE:
        return UNIVERSE[f"{key}T"]
    raise KeyError(
        f"Unknown symbol {symbol!r}. Known symbols: {', '.join(sorted(UNIVERSE))}"
    )


def resolve(symbols: list[str] | None, asset_class: AssetClass | None = None) -> list[Instrument]:
    """Resolve a user-supplied symbol list, defaulting to the whole universe."""
    if symbols:
        return [get_instrument(s) for s in symbols]
    if asset_class is AssetClass.FOREX:
        return list(FOREX.values())
    if asset_class is AssetClass.CRYPTO:
        return list(CRYPTO.values())
    return list(UNIVERSE.values())


def is_high_beta(instrument: Instrument) -> bool:
    """Altcoins and commodity/carry currencies contract hardest when liquidity dries up."""
    if instrument.is_crypto:
        return instrument.symbol not in CRYPTO_MAJORS
    return instrument.base in {"AUD", "NZD"} or instrument.quote in {"AUD", "NZD"}
