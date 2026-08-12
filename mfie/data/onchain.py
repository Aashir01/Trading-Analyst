"""On-chain metrics: Glassnode (transfer volume, active addresses, exchange flows).

Glassnode is a paid API. When no key is present the hub substitutes CoinGecko
exchange volume — noisier and not strictly on-chain, but it plays the same role
as the denominator in NVT and the numerator in token velocity.
"""

from __future__ import annotations

import pandas as pd

from mfie.core.types import Instrument
from mfie.core.utils import get_logger
from mfie.data.base import HttpClient

log = get_logger(__name__)

GLASSNODE_BASE = "https://api.glassnode.com/v1/metrics"


class GlassnodeProvider:
    name = "glassnode"

    def __init__(self, http: HttpClient) -> None:
        self.http = http
        self.key = http.settings.glassnode_api_key

    @property
    def available(self) -> bool:
        return bool(self.key)

    def _metric(self, path: str, asset: str, days: int) -> pd.Series | None:
        if not self.available:
            return None
        data = self.http.get_json(
            f"{GLASSNODE_BASE}/{path}",
            params={"a": asset, "api_key": self.key, "i": "24h", "s": _since(days)},
        )
        if not isinstance(data, list) or not data:
            return None
        rows = [(r["t"], r.get("v")) for r in data if r.get("v") is not None]
        if not rows:
            return None
        idx = pd.to_datetime([r[0] for r in rows], unit="s", utc=True)
        return pd.Series([float(r[1]) for r in rows], index=idx).sort_index()

    def transfer_volume_usd(self, instrument: Instrument, days: int = 180) -> pd.Series | None:
        """Daily on-chain transfer volume in USD — the ``P*Q`` term in MV=PQ."""
        return self._metric("transactions/transfers_volume_sum", instrument.base, days)

    def market_cap(self, instrument: Instrument, days: int = 180) -> pd.Series | None:
        return self._metric("market/marketcap_usd", instrument.base, days)

    def exchange_net_flow(self, instrument: Instrument, days: int = 90) -> pd.Series | None:
        """Positive = coins moving *onto* exchanges, classically distribution."""
        return self._metric("transactions/transfers_volume_exchanges_net", instrument.base, days)

    def active_addresses(self, instrument: Instrument, days: int = 180) -> pd.Series | None:
        return self._metric("addresses/active_count", instrument.base, days)

    def stablecoin_supply(self, days: int = 180) -> pd.Series | None:
        usdt = self._metric("market/marketcap_usd", "USDT", days)
        usdc = self._metric("market/marketcap_usd", "USDC", days)
        if usdt is None and usdc is None:
            return None
        if usdt is None:
            return usdc.rename("stablecoin_supply")  # type: ignore[union-attr]
        if usdc is None:
            return usdt.rename("stablecoin_supply")
        return usdt.add(usdc, fill_value=0.0).rename("stablecoin_supply")


def _since(days: int) -> int:
    import time

    return int(time.time() - days * 86_400)


class WhaleAlertProvider:
    """Large-transaction feed. Used as a discrete distribution/accumulation flag."""

    name = "whale_alert"
    BASE = "https://api.whale-alert.io/v1/transactions"

    def __init__(self, http: HttpClient, api_key: str | None = None) -> None:
        self.http = http
        self.key = api_key

    @property
    def available(self) -> bool:
        return bool(self.key)

    def recent_to_exchanges(self, instrument: Instrument, min_value_usd: int = 1_000_000) -> float | None:
        """Net USD value of whale transfers into exchanges over the last hour."""
        if not self.available:
            return None
        import time

        data = self.http.get_json(
            self.BASE,
            params={
                "api_key": self.key,
                "min_value": min_value_usd,
                "start": int(time.time()) - 3600,
                "currency": instrument.base.lower(),
            },
        )
        if not isinstance(data, dict) or not data.get("transactions"):
            return None
        net = 0.0
        for tx in data["transactions"]:
            amount = float(tx.get("amount_usd", 0))
            to_type = (tx.get("to") or {}).get("owner_type")
            from_type = (tx.get("from") or {}).get("owner_type")
            if to_type == "exchange" and from_type != "exchange":
                net += amount
            elif from_type == "exchange" and to_type != "exchange":
                net -= amount
        return net
