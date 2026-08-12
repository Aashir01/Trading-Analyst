"""Macroeconomic data: FRED (rates, CPI, M2, REER) and TradingEconomics (calendar).

FRED is the default because it is free, keyed, well documented and covers every
series the econometric layer needs. Series IDs are grouped in one place so they
can be corrected without touching logic — FRED does retire and rename series,
so treat ``_FRED_*`` as configuration, not constants.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from mfie.core.types import EconomicEvent
from mfie.core.utils import get_logger
from mfie.data.base import HttpClient

log = get_logger(__name__)

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# --- central bank policy rates ------------------------------------------------
_FRED_POLICY_RATE = {
    "USD": "DFF",                    # Federal Funds effective rate
    "EUR": "ECBDFR",                 # ECB deposit facility rate
    "GBP": "IUDSOIA",                # SONIA
    "JPY": "IRSTCI01JPM156N",        # Japan immediate rates
    "CHF": "IR3TIB01CHM156N",
    "AUD": "IR3TIB01AUM156N",
    "NZD": "IR3TIB01NZM156N",
    "CAD": "IRSTCI01CAM156N",
}

# --- CPI *levels*; year-over-year inflation is derived from these -------------
_FRED_CPI_LEVEL = {
    "USD": "CPIAUCSL",
    "EUR": "CP0000EZ19M086NEST",
    "GBP": "GBRCPIALLMINMEI",
    "JPY": "JPNCPIALLMINMEI",
    "CHF": "CHECPIALLMINMEI",
    "AUD": "AUSCPIALLQINMEI",
    "NZD": "NZLCPIALLQINMEI",
    "CAD": "CANCPIALLMINMEI",
}

# --- sovereign yields ---------------------------------------------------------
_FRED_YIELD_10Y = {
    "USD": "DGS10",
    "EUR": "IRLTLT01EZM156N",
    "GBP": "IRLTLT01GBM156N",
    "JPY": "IRLTLT01JPM156N",
    "CHF": "IRLTLT01CHM156N",
    "AUD": "IRLTLT01AUM156N",
    "NZD": "IRLTLT01NZM156N",
    "CAD": "IRLTLT01CAM156N",
}
# Only the US publishes a clean daily 2Y on FRED; elsewhere the 3-month
# interbank rate stands in for the short end of the curve.
_FRED_YIELD_SHORT = {
    "USD": "DGS2",
    "EUR": "IR3TIB01EZM156N",
    "GBP": "IR3TIB01GBM156N",
    "JPY": "IR3TIB01JPM156N",
    "CHF": "IR3TIB01CHM156N",
    "AUD": "IR3TIB01AUM156N",
    "NZD": "IR3TIB01NZM156N",
    "CAD": "IR3TIB01CAM156N",
}

# --- BIS real broad effective exchange rates (PPP module) --------------------
_FRED_REER = {
    "USD": "RBUSBIS", "EUR": "RBXMBIS", "GBP": "RBGBBIS", "JPY": "RBJPBIS",
    "CHF": "RBCHBIS", "AUD": "RBAUBIS", "NZD": "RBNZBIS", "CAD": "RBCABIS",
}

_FRED_M2 = "M2SL"  # US M2, the dominant term in any global liquidity proxy


class FredProvider:
    """St. Louis Fed FRED API. Requires ``FRED_API_KEY`` (free)."""

    name = "fred"

    def __init__(self, http: HttpClient) -> None:
        self.http = http
        self.key = http.settings.fred_api_key

    @property
    def available(self) -> bool:
        return bool(self.key)

    def series(self, series_id: str, observations: int = 400) -> pd.Series | None:
        if not self.available:
            return None
        data = self.http.get_json(
            FRED_BASE,
            params={
                "series_id": series_id,
                "api_key": self.key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": observations,
            },
        )
        if not isinstance(data, dict) or not data.get("observations"):
            return None
        rows = [
            (obs["date"], float(obs["value"]))
            for obs in data["observations"]
            if obs.get("value") not in (".", None, "")
        ]
        if not rows:
            return None
        idx = pd.to_datetime([r[0] for r in rows], utc=True)
        return pd.Series([r[1] for r in rows], index=idx, name=series_id).sort_index()

    def latest(self, series_id: str) -> float | None:
        s = self.series(series_id, observations=12)
        return float(s.iloc[-1]) if s is not None and not s.empty else None

    # ------------------------------------------------------------------ blocks
    def policy_rates(self, currencies: tuple[str, ...]) -> dict[str, float]:
        out: dict[str, float] = {}
        for ccy in currencies:
            sid = _FRED_POLICY_RATE.get(ccy)
            if not sid:
                continue
            val = self.latest(sid)
            if val is not None:
                out[ccy] = val / 100.0  # FRED publishes percent; engine uses decimals
        return out

    def inflation(self, currencies: tuple[str, ...]) -> dict[str, float]:
        """Trailing year-over-year CPI inflation, as a decimal."""
        out: dict[str, float] = {}
        for ccy in currencies:
            sid = _FRED_CPI_LEVEL.get(ccy)
            if not sid:
                continue
            s = self.series(sid, observations=40)
            if s is None or len(s) < 13:
                continue
            # Monthly series -> 12 periods back; quarterly (AUD/NZD) -> 4.
            lag = 12 if _infer_monthly(s) else 4
            if len(s) <= lag:
                continue
            yoy = (s.iloc[-1] - s.iloc[-1 - lag]) / s.iloc[-1 - lag]
            out[ccy] = float(yoy)
        return out

    def yields(
        self, currencies: tuple[str, ...]
    ) -> tuple[dict[str, float], dict[str, float]] | None:
        """Long- and short-end yields. ``None`` when nothing could be served.

        Returning a tuple of two empty dicts would look like success to the
        hub's fallback logic and get FRED credited as a live source for data it
        never provided.
        """
        long_end: dict[str, float] = {}
        short_end: dict[str, float] = {}
        for ccy in currencies:
            if sid := _FRED_YIELD_10Y.get(ccy):
                if (val := self.latest(sid)) is not None:
                    long_end[ccy] = val / 100.0
            if sid := _FRED_YIELD_SHORT.get(ccy):
                if (val := self.latest(sid)) is not None:
                    short_end[ccy] = val / 100.0
        if not long_end and not short_end:
            return None
        return long_end, short_end

    # ------------------------------------------------- historical (cycle engine)
    # The filter chain only needs today's level. The cycle engine needs the
    # *path* — an impulse is a derivative, and you cannot differentiate a scalar.
    def policy_rate_series(self, currency: str, observations: int = 900) -> pd.Series | None:
        sid = _FRED_POLICY_RATE.get(currency)
        if not sid:
            return None
        s = self.series(sid, observations)
        return s / 100.0 if s is not None else None

    def inflation_series(self, currency: str, observations: int = 120) -> pd.Series | None:
        """Year-over-year CPI as a time series, derived from the index level."""
        sid = _FRED_CPI_LEVEL.get(currency)
        if not sid:
            return None
        s = self.series(sid, observations)
        if s is None or len(s) < 13:
            return None
        lag = 12 if _infer_monthly(s) else 4
        return (s / s.shift(lag) - 1.0).dropna()

    def yield_series(self, currency: str, tenor: str = "10y",
                     observations: int = 900) -> pd.Series | None:
        table = _FRED_YIELD_10Y if tenor == "10y" else _FRED_YIELD_SHORT
        sid = table.get(currency)
        if not sid:
            return None
        s = self.series(sid, observations)
        return s / 100.0 if s is not None else None

    def m2_series(self, observations: int = 60) -> pd.Series | None:
        return self.series(_FRED_M2, observations)

    def reer_series(self, currency: str, observations: int = 400) -> pd.Series | None:
        sid = _FRED_REER.get(currency)
        return self.series(sid, observations) if sid else None


def _infer_monthly(series: pd.Series) -> bool:
    """True if consecutive observations are roughly a month apart."""
    if len(series) < 3:
        return True
    deltas = series.index.to_series().diff().dropna().dt.days
    return float(deltas.median()) < 70


class TradingEconomicsProvider:
    """Economic calendar. Requires ``TRADINGECONOMICS_API_KEY``.

    The free ``guest:guest`` key exposes a small sample of countries, which is
    enough to smoke-test the event blocker.
    """

    name = "tradingeconomics"
    BASE = "https://api.tradingeconomics.com"

    def __init__(self, http: HttpClient) -> None:
        self.http = http
        self.key = http.settings.tradingeconomics_api_key

    @property
    def available(self) -> bool:
        return bool(self.key)

    def economic_events(self, days_ahead: int = 7, days_back: int = 30) -> list[EconomicEvent] | None:
        if not self.available:
            return None
        now = datetime.now(timezone.utc)
        data = self.http.get_json(
            f"{self.BASE}/calendar",
            params={
                "c": self.key,
                "f": "json",
                "d1": (now - timedelta(days=days_back)).strftime("%Y-%m-%d"),
                "d2": (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d"),
            },
        )
        if not isinstance(data, list) or not data:
            return None

        events: list[EconomicEvent] = []
        for row in data:
            ts = _parse_te_date(row.get("Date"))
            if ts is None:
                continue
            currency = (row.get("Currency") or "").upper()
            if not currency:
                continue
            events.append(
                EconomicEvent(
                    ts=ts,
                    country=(row.get("Country") or "")[:2].upper(),
                    currency=currency,
                    name=row.get("Event") or row.get("Category") or "Unknown",
                    impact=_te_impact(row.get("Importance")),
                    actual=_to_float(row.get("Actual")),
                    forecast=_to_float(row.get("Forecast")) or _to_float(row.get("TEForecast")),
                    previous=_to_float(row.get("Previous")),
                    unit=row.get("Unit") or "",
                )
            )
        return sorted(events, key=lambda e: e.ts)


def _parse_te_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        ts = pd.to_datetime(value, utc=True)
        return ts.to_pydatetime()
    except (ValueError, TypeError):
        return None


def _te_impact(importance: object) -> str:
    try:
        level = int(importance)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "low"
    return {3: "high", 2: "medium"}.get(level, "low")


def _to_float(value: object) -> float | None:
    if value in (None, "", "-"):
        return None
    try:
        text = str(value).replace("%", "").replace(",", "").strip()
        multiplier = 1.0
        if text.endswith("K"):
            multiplier, text = 1e3, text[:-1]
        elif text.endswith("M"):
            multiplier, text = 1e6, text[:-1]
        elif text.endswith("B"):
            multiplier, text = 1e9, text[:-1]
        return float(text) * multiplier
    except (TypeError, ValueError):
        return None
