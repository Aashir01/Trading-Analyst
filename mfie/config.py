"""Configuration: API credentials (env) and econometric parameters (YAML).

Two separate concerns on purpose:

* ``Settings``  -> secrets and infrastructure. Read from environment / ``.env``.
                   Never commit these.
* ``Params``    -> the economic thresholds that define the *behaviour* of the
                   engine (GLI weights, PPP z-score bands, CVaR limits...).
                   These belong in version control and should be tuned and
                   backtested, so they live in ``config/params.yaml``.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
PARAMS_FILE = CONFIG_DIR / "params.yaml"

load_dotenv(PROJECT_ROOT / ".env")


# --------------------------------------------------------------------------- #
# Secrets / infrastructure
# --------------------------------------------------------------------------- #
@dataclass
class Settings:
    """Runtime settings sourced from environment variables.

    Every API key is optional. A provider whose key is missing degrades to the
    synthetic provider, so the whole stack runs offline out of the box.
    """

    # --- market data ---
    binance_base_url: str = "https://api.binance.com"
    coingecko_api_key: str | None = None
    alphavantage_api_key: str | None = None
    oanda_api_key: str | None = None
    oanda_account_id: str | None = None
    oanda_environment: str = "practice"  # practice | live

    # --- macro / fundamental ---
    fred_api_key: str | None = None
    tradingeconomics_api_key: str | None = None

    # --- on-chain / sentiment ---
    glassnode_api_key: str | None = None
    newsapi_key: str | None = None
    lunarcrush_api_key: str | None = None

    # --- interfaces ---
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None

    # --- storage ---
    # Default is a local SQLite file so the project runs with zero setup.
    # For production point this at TimescaleDB, e.g.
    #   postgresql+psycopg2://user:pw@host:5432/mfie
    database_url: str = f"sqlite:///{(DATA_DIR / 'mfie.db').as_posix()}"

    # --- behaviour ---
    offline: bool = False  # force synthetic data everywhere
    http_timeout: float = 15.0
    cache_ttl_seconds: int = 300
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> Settings:
        def _s(key: str, default: Any = None) -> Any:
            val = os.getenv(key)
            return val if val not in (None, "") else default

        def _b(key: str, default: bool) -> bool:
            val = os.getenv(key)
            if val is None or val == "":
                return default
            return val.strip().lower() in {"1", "true", "yes", "on"}

        return cls(
            binance_base_url=_s("BINANCE_BASE_URL", "https://api.binance.com"),
            coingecko_api_key=_s("COINGECKO_API_KEY"),
            alphavantage_api_key=_s("ALPHAVANTAGE_API_KEY"),
            oanda_api_key=_s("OANDA_API_KEY"),
            oanda_account_id=_s("OANDA_ACCOUNT_ID"),
            oanda_environment=_s("OANDA_ENVIRONMENT", "practice"),
            fred_api_key=_s("FRED_API_KEY"),
            tradingeconomics_api_key=_s("TRADINGECONOMICS_API_KEY"),
            glassnode_api_key=_s("GLASSNODE_API_KEY"),
            newsapi_key=_s("NEWSAPI_KEY"),
            lunarcrush_api_key=_s("LUNARCRUSH_API_KEY"),
            telegram_bot_token=_s("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=_s("TELEGRAM_CHAT_ID"),
            database_url=_s("DATABASE_URL", f"sqlite:///{(DATA_DIR / 'mfie.db').as_posix()}"),
            offline=_b("MFIE_OFFLINE", False),
            http_timeout=float(_s("HTTP_TIMEOUT", 15.0)),
            cache_ttl_seconds=int(_s("CACHE_TTL_SECONDS", 300)),
            log_level=_s("LOG_LEVEL", "INFO"),
        )

    def has(self, *names: str) -> bool:
        """True only if every named credential is present."""
        return all(getattr(self, n, None) for n in names)


# --------------------------------------------------------------------------- #
# Econometric parameters
# --------------------------------------------------------------------------- #
@dataclass
class LiquidityParams:
    m2_weight: float = 0.6           # w1 in the GLI formula
    stablecoin_weight: float = 0.4   # w2
    lookback_days: int = 90          # n
    contraction_threshold: float = 0.0
    contraction_penalty: float = 0.5     # multiplier applied to momentum longs
    strong_expansion_threshold: float = 0.02


@dataclass
class YieldParams:
    inversion_threshold: float = 0.0     # 10Y-2Y <= 0 => CONTRACTION
    steepening_threshold: float = 0.005
    contraction_penalty: float = 0.6
    block_high_beta_on_inversion: bool = True


@dataclass
class RIRDParams:
    """Real Interest Rate Differential."""
    significant_differential: float = 0.01   # 100 bps
    aligned_bonus: float = 1.15
    opposed_penalty: float = 0.65
    hard_block_differential: float = 0.03    # 300 bps against the trade


@dataclass
class PPPParams:
    lookback_periods: int = 250
    overvalued_z: float = 2.0
    stretched_z: float = 1.5
    penalty: float = 0.5
    block_breakouts_beyond_z: float = 2.5


@dataclass
class ESIParams:
    decay_lambda: float = 0.05    # e^{-lambda * days}
    window_days: int = 90
    significant_delta: float = 1.0
    aligned_bonus: float = 1.10
    opposed_penalty: float = 0.80


@dataclass
class TokenomicsParams:
    velocity_lookback: int = 90
    velocity_z_warning: float = -1.5   # V below mean - 1.5 sigma
    nvt_lookback: int = 90
    nvt_z_overvalued: float = 2.0
    divergence_penalty: float = 0.55
    nvt_penalty: float = 0.7


@dataclass
class MicrostructureParams:
    atr_period: int = 14
    friction_warn: float = 0.10     # (ask-bid)/ATR
    friction_halt: float = 0.30
    widen_stops: bool = True
    max_stop_multiplier: float = 2.5


@dataclass
class BehavioralParams:
    contrarian_k: float = 2.0        # exponent in C_t
    extreme_threshold: float = 0.85  # retail net-long share considered extreme
    fear_greed_extreme_greed: float = 80.0
    fear_greed_extreme_fear: float = 20.0
    fear_greed_penalty: float = 0.75


@dataclass
class EventParams:
    blackout_minutes_before: int = 30
    blackout_minutes_after: int = 30
    blocking_impacts: tuple[str, ...] = ("high",)
    medium_impact_penalty: float = 0.8


@dataclass
class RiskParams:
    account_equity: float = 10_000.0
    base_risk_per_trade: float = 0.01     # 1% of equity at stop
    max_risk_per_trade: float = 0.02
    max_portfolio_risk: float = 0.06
    cvar_alpha: float = 0.95
    cvar_lookback: int = 250
    cvar_budget: float = 0.035            # daily CVaR95 budget as fraction of equity
    cvar_breach_scaler: float = 0.5
    target_annual_vol: float = 0.15
    kelly_fraction_cap: float = 0.25
    min_confidence_to_trade: float = 0.35
    atr_stop_multiple: float = 2.0
    reward_risk_target: float = 2.0


@dataclass
class TechnicalParams:
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    adx_period: int = 14
    ema_fast: int = 20
    ema_slow: int = 50
    ema_trend: int = 200
    donchian_period: int = 20
    zscore_period: int = 20


@dataclass
class RegimeParams:
    vol_lookback: int = 60
    trend_lookback: int = 60
    high_vol_percentile: float = 0.75
    low_vol_percentile: float = 0.35
    adx_trending: float = 25.0
    hmm_states: int = 3


@dataclass
class Params:
    liquidity: LiquidityParams = field(default_factory=LiquidityParams)
    yields: YieldParams = field(default_factory=YieldParams)
    rird: RIRDParams = field(default_factory=RIRDParams)
    ppp: PPPParams = field(default_factory=PPPParams)
    esi: ESIParams = field(default_factory=ESIParams)
    tokenomics: TokenomicsParams = field(default_factory=TokenomicsParams)
    microstructure: MicrostructureParams = field(default_factory=MicrostructureParams)
    behavioral: BehavioralParams = field(default_factory=BehavioralParams)
    events: EventParams = field(default_factory=EventParams)
    risk: RiskParams = field(default_factory=RiskParams)
    technical: TechnicalParams = field(default_factory=TechnicalParams)
    regime: RegimeParams = field(default_factory=RegimeParams)


def _apply_overrides(obj: Any, overrides: dict[str, Any]) -> None:
    """Recursively apply a nested dict onto a dataclass instance."""
    if not is_dataclass(obj) or not isinstance(overrides, dict):
        return
    known = {f.name: f for f in fields(obj)}
    for key, value in overrides.items():
        if key not in known:
            continue
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply_overrides(current, value)
        else:
            setattr(obj, key, value)


def load_params(path: Path | None = None) -> Params:
    """Build ``Params`` from defaults, overlaying ``config/params.yaml``."""
    params = Params()
    path = path or PARAMS_FILE
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        _apply_overrides(params, raw)
    return params


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()


@functools.lru_cache(maxsize=1)
def get_params() -> Params:
    return load_params()


def reset_caches() -> None:
    """Drop cached settings/params (used by tests and the dashboard reload)."""
    get_settings.cache_clear()
    get_params.cache_clear()


DATA_DIR.mkdir(parents=True, exist_ok=True)
