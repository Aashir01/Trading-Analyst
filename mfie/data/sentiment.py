"""Sentiment inputs: Fear & Greed, news headlines, social dominance.

Headline scoring uses a finance-tuned lexicon rather than a transformer by
default. That keeps the install light and CPU-only; if ``transformers`` is
installed, ``score_headlines`` transparently upgrades to a FinBERT-style model.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Sequence

from mfie.core.types import Instrument
from mfie.core.utils import clamp, get_logger
from mfie.data.base import HttpClient

log = get_logger(__name__)

FNG_URL = "https://api.alternative.me/fng/"
NEWSAPI_URL = "https://newsapi.org/v2/everything"
LUNARCRUSH_URL = "https://lunarcrush.com/api4/public/coins"

# Finance-specific polarity lexicon. Weights are in [-1, 1]; the scorer averages
# matched terms, so a headline needs several negative words to score near -1.
_LEXICON: dict[str, float] = {
    # bullish
    "surge": 0.8, "surges": 0.8, "soar": 0.9, "soars": 0.9, "rally": 0.7, "rallies": 0.7,
    "jump": 0.6, "jumps": 0.6, "gain": 0.5, "gains": 0.5, "rise": 0.4, "rises": 0.4,
    "bullish": 0.8, "breakout": 0.6, "record high": 0.9, "all-time high": 0.9,
    "upgrade": 0.6, "beats": 0.6, "beat": 0.5, "strong": 0.5, "optimism": 0.6,
    "approval": 0.7, "approved": 0.7, "adoption": 0.6, "inflows": 0.6, "accumulate": 0.5,
    "dovish": 0.6, "stimulus": 0.6, "easing": 0.5, "rate cut": 0.7, "recovery": 0.5,
    # bearish
    "plunge": -0.9, "plunges": -0.9, "crash": -1.0, "crashes": -1.0, "slump": -0.7,
    "tumble": -0.8, "tumbles": -0.8, "fall": -0.4, "falls": -0.4, "drop": -0.5,
    "drops": -0.5, "decline": -0.5, "declines": -0.5, "bearish": -0.8, "selloff": -0.8,
    "sell-off": -0.8, "downgrade": -0.6, "miss": -0.5, "misses": -0.5, "weak": -0.5,
    "fear": -0.6, "panic": -0.9, "liquidation": -0.8, "liquidations": -0.8,
    "hack": -0.9, "exploit": -0.8, "fraud": -0.9, "lawsuit": -0.6, "ban": -0.8,
    "crackdown": -0.7, "probe": -0.5, "outflows": -0.6, "recession": -0.8,
    "hawkish": -0.6, "tightening": -0.5, "rate hike": -0.6, "inflation surge": -0.7,
    "default": -0.8, "bankruptcy": -0.9, "warns": -0.5, "warning": -0.5,
}

_NEGATORS = {"no", "not", "never", "without", "fails", "fail", "halts", "denied", "rejects"}
_INTENSIFIERS = {"very": 1.4, "sharply": 1.4, "massive": 1.5, "record": 1.4, "slightly": 0.6}

_TOKEN_RE = re.compile(r"[a-z][a-z'\-]+")


def score_text(text: str) -> float:
    """Polarity of one headline in [-1, 1]. 0.0 means no signal found."""
    if not text:
        return 0.0
    lowered = text.lower()

    hits: list[float] = []
    # Multi-word phrases first so "rate cut" is not scored as "cut".
    for phrase, weight in _LEXICON.items():
        if " " in phrase and phrase in lowered:
            hits.append(weight)

    tokens = _TOKEN_RE.findall(lowered)
    for i, token in enumerate(tokens):
        weight = _LEXICON.get(token)
        if weight is None:
            continue
        window = tokens[max(0, i - 2): i]
        if any(w in _NEGATORS for w in window):
            weight = -weight * 0.8
        for w in window:
            if w in _INTENSIFIERS:
                weight *= _INTENSIFIERS[w]
        hits.append(weight)

    if not hits:
        return 0.0
    return clamp(sum(hits) / (len(hits) ** 0.5) / 2.0, -1.0, 1.0)


@functools.lru_cache(maxsize=1)
def _transformer_pipeline():
    """Load FinBERT once, if transformers happens to be installed."""
    try:
        from transformers import pipeline  # type: ignore

        return pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except Exception as exc:  # pragma: no cover - optional dependency
        log.debug("Transformer sentiment unavailable (%s); using lexicon", exc)
        return None


def score_headlines(headlines: Sequence[str], use_transformer: bool = False) -> float:
    """Mean polarity across headlines, in [-1, 1]."""
    if not headlines:
        return 0.0

    if use_transformer:
        pipe = _transformer_pipeline()
        if pipe is not None:  # pragma: no cover - optional dependency
            scores = []
            for result in pipe(list(headlines)[:64]):
                label = result["label"].lower()
                sign = {"positive": 1.0, "negative": -1.0}.get(label, 0.0)
                scores.append(sign * float(result["score"]))
            return clamp(sum(scores) / len(scores), -1.0, 1.0) if scores else 0.0

    scores = [score_text(h) for h in headlines]
    return clamp(sum(scores) / len(scores), -1.0, 1.0)


class FearGreedProvider:
    """alternative.me crypto Fear & Greed index (0-100). No key required."""

    name = "alternative.me"

    def __init__(self, http: HttpClient) -> None:
        self.http = http

    available = True

    def latest(self) -> float | None:
        data = self.http.get_json(FNG_URL, params={"limit": 1, "format": "json"})
        if not isinstance(data, dict) or not data.get("data"):
            return None
        try:
            return float(data["data"][0]["value"])
        except (KeyError, IndexError, TypeError, ValueError):
            return None

    def history(self, limit: int = 90) -> list[tuple[int, float]] | None:
        data = self.http.get_json(FNG_URL, params={"limit": limit, "format": "json"})
        if not isinstance(data, dict) or not data.get("data"):
            return None
        out = []
        for row in data["data"]:
            try:
                out.append((int(row["timestamp"]), float(row["value"])))
            except (KeyError, TypeError, ValueError):
                continue
        return sorted(out)


class NewsProvider:
    """NewsAPI.org headline search. Requires ``NEWSAPI_KEY``."""

    name = "newsapi"

    def __init__(self, http: HttpClient) -> None:
        self.http = http
        self.key = http.settings.newsapi_key

    @property
    def available(self) -> bool:
        return bool(self.key)

    def headlines(self, instrument: Instrument, limit: int = 40) -> list[str] | None:
        if not self.available:
            return None
        query = _news_query(instrument)
        data = self.http.get_json(
            NEWSAPI_URL,
            params={
                "q": query,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": min(limit, 100),
                "apiKey": self.key,
            },
        )
        if not isinstance(data, dict) or not data.get("articles"):
            return None
        return [a["title"] for a in data["articles"] if a.get("title")]

    def sentiment(self, instrument: Instrument) -> float | None:
        heads = self.headlines(instrument)
        if not heads:
            return None
        return score_headlines(heads)


def _news_query(instrument: Instrument) -> str:
    if instrument.is_crypto:
        names = {
            "BTC": "Bitcoin", "ETH": "Ethereum", "SOL": "Solana", "BNB": "Binance Coin",
            "XRP": "XRP Ripple", "ADA": "Cardano", "AVAX": "Avalanche", "LINK": "Chainlink",
        }
        return names.get(instrument.base, instrument.base)
    return f"{instrument.base} {instrument.quote} forex OR central bank OR inflation"


class LunarCrushProvider:
    """Social dominance and galaxy score for crypto. Requires an API key."""

    name = "lunarcrush"

    def __init__(self, http: HttpClient) -> None:
        self.http = http
        self.key = http.settings.lunarcrush_api_key

    @property
    def available(self) -> bool:
        return bool(self.key)

    def social_metrics(self, instrument: Instrument) -> dict[str, float] | None:
        if not self.available or not instrument.is_crypto:
            return None
        data = self.http.get_json(
            f"{LUNARCRUSH_URL}/{instrument.base}/v1",
            headers={"Authorization": f"Bearer {self.key}"},
        )
        if not isinstance(data, dict) or "data" not in data:
            return None
        row = data["data"]
        out = {}
        for key in ("galaxy_score", "alt_rank", "social_dominance", "sentiment", "interactions_24h"):
            if row.get(key) is not None:
                try:
                    out[key] = float(row[key])
                except (TypeError, ValueError):
                    continue
        return out or None
