r"""The six leading factors, plus one lagging confirmer.

Every factor is a daily series normalised to :math:`[-1, +1]` where **positive
means risk-on**. That single sign convention is what makes the composite
interpretable: a reading of +0.6 means six-tenths of the way to a maximally
bullish macro configuration, regardless of which factor drove it.

Normalisation is :math:`\tanh(z/2)`, which is bounded (one crazy print cannot
dominate the composite), monotonic, and roughly linear near zero — so ordinary
readings behave like a z-score and extreme ones saturate instead of exploding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mfie.config import CycleParams
from mfie.core.types import Instrument
from mfie.core.utils import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class FactorSpec:
    """Static description of a factor: what it is and why it should lead."""

    name: str
    label: str
    prior_weight: float
    typical_lead_days: int
    rationale: str

    @property
    def lead_months(self) -> float:
        return self.typical_lead_days / 21.0


# The economic priors. `typical_lead_days` encodes the published/observed lead
# of each factor over risk assets and is used as a sanity check against the
# lead the data actually measures.
FACTOR_SPECS: dict[str, FactorSpec] = {
    "liquidity_impulse": FactorSpec(
        "liquidity_impulse", "Liquidity impulse", 0.26, 90,
        "Acceleration in broad money and stablecoin supply. Risk assets are a "
        "levered claim on the marginal unit of liquidity, so the second "
        "derivative of money turns before price does.",
    ),
    "credit_regime": FactorSpec(
        "credit_regime", "Credit / curve regime", 0.20, 120,
        "Shape AND direction of the yield curve. Inversion is an early warning; "
        "the bull steepener out of inversion is the actual trigger.",
    ),
    "real_rate_impulse": FactorSpec(
        "real_rate_impulse", "Real rate impulse", 0.16, 60,
        "Direction of the real policy rate. Falling real rates ease the "
        "discount rate on every future cash flow; rising ones tighten it.",
    ),
    "breadth": FactorSpec(
        "breadth", "Participation breadth", 0.14, 21,
        "Share of the universe in an uptrend. Narrowing leadership is "
        "distribution: the index holds up while its constituents roll over.",
    ),
    "valuation": FactorSpec(
        "valuation", "Valuation stretch", 0.10, 120,
        "NVT for crypto, REER for FX. Slow-moving and near-useless for timing, "
        "but it sets how much downside is available when the cycle turns.",
    ),
    "risk_appetite": FactorSpec(
        "risk_appetite", "Positioning / risk appetite", 0.08, 10,
        "Contrarian. Crowded positioning and euphoric sentiment mean the "
        "marginal buyer has already bought.",
    ),
    "trend_confirmation": FactorSpec(
        "trend_confirmation", "Trend confirmation", 0.06, 0,
        "Deliberately lagging. Carries little weight, but a macro call the "
        "tape flatly contradicts should not be stated with confidence.",
    ),
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def to_daily(series: pd.Series) -> pd.Series:
    """Resample any frequency onto a forward-filled daily grid.

    Monthly M2 and daily yields have to live on one index before they can be
    combined. Forward-fill is the honest choice: on 12 March you knew February's
    M2 print, not March's.
    """
    s = pd.Series(series).dropna().astype(float).sort_index()
    if s.empty:
        return s
    if not isinstance(s.index, pd.DatetimeIndex):
        return s
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    else:
        s.index = s.index.tz_convert("UTC")
    s = s[~s.index.duplicated(keep="last")]
    return s.resample("1D").last().ffill()


def normalize(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score squashed through tanh into [-1, +1]."""
    s = pd.Series(series).astype(float)
    if s.dropna().empty:
        return s
    min_periods = max(30, window // 4)
    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std(ddof=1)
    z = (s - mu) / sd.replace(0.0, np.nan)
    return np.tanh(z.fillna(0.0) / 2.0)


def _change(series: pd.Series, window: int) -> pd.Series:
    return pd.Series(series).astype(float).diff(window)


def risk_oriented(instrument: Instrument, close: pd.Series) -> pd.Series:
    """Re-orient a price series so that **up always means risk-on**.

    ``EURUSD`` up is risk-on; ``USDJPY`` up is risk-*off* (the yen is a haven and
    the dollar is the funding currency). Without this every FX breadth measure
    is the average of a signal and its own negation, which is zero.
    """
    s = pd.Series(close).astype(float)
    if instrument.is_crypto:
        return s
    return 1.0 / s.replace(0.0, np.nan) if instrument.base == "USD" else s


# --------------------------------------------------------------------------- #
# Factor 1 — liquidity impulse
# --------------------------------------------------------------------------- #
def liquidity_impulse(
    m2: pd.Series,
    stablecoins: pd.Series,
    params: CycleParams,
    m2_weight: float = 0.6,
) -> pd.Series:
    """Acceleration of global liquidity.

    Growth rate first (year-over-year for M2, quarterly for stablecoins, since
    they move on different clocks), then the change *in that growth rate*. A
    liquidity impulse that is positive but decelerating is a warning; the level
    alone would still look expansionary.
    """
    m2_daily = to_daily(m2)
    sc_daily = to_daily(stablecoins)

    parts: list[pd.Series] = []
    weights: list[float] = []

    if len(m2_daily) > 365:
        m2_growth = m2_daily.pct_change(365)
        parts.append(m2_growth.diff(params.impulse_window))
        weights.append(m2_weight)
    if len(sc_daily) > params.impulse_window * 2:
        sc_growth = sc_daily.pct_change(params.impulse_window)
        parts.append(sc_growth.diff(params.impulse_window))
        weights.append(1.0 - m2_weight)

    if not parts:
        return pd.Series(dtype=float, name="liquidity_impulse")

    frame = pd.concat(parts, axis=1).ffill()
    total = sum(weights)
    normalised = [
        normalize(frame.iloc[:, i], params.zscore_window) * (weights[i] / total)
        for i in range(frame.shape[1])
    ]
    return sum(normalised).rename("liquidity_impulse")


# --------------------------------------------------------------------------- #
# Factor 2 — credit / curve regime  (the non-monotonic one)
# --------------------------------------------------------------------------- #
def credit_regime(y10: pd.Series, y2: pd.Series, params: CycleParams) -> pd.Series:
    """Yield-curve regime, non-monotonic in the spread.

    The four states, and why the mapping is *not* "steep good, inverted bad":

    ==========================  ===========================  ======
    Curve state                 Cycle phase                  Score
    ==========================  ===========================  ======
    Positive and steepening     early recovery, easing        +1.0
    Positive and flattening     mid expansion, tightening     +0.3
    Inverted and flattening     late cycle — markets melt up  -0.2
    Inverted and steepening     **the recession trigger**     -1.0
    ==========================  ===========================  ======

    The last row is the point. A curve un-inverting from below means the front
    end is collapsing because the market has started pricing cuts, and cuts get
    priced when something is breaking. Recessions and bear markets have
    historically begun *after* the un-inversion, not at the inversion.

    A curve that has recently crossed back above zero from inversion gets the
    same maximally negative score even while positive, for one year.
    """
    spread = (to_daily(y10) - to_daily(y2)).dropna()
    if len(spread) < params.impulse_window * 2:
        return pd.Series(dtype=float, name="credit_regime")

    slope = spread.diff(params.impulse_window)

    positive = spread > 0
    steepening = slope > 0

    base = pd.Series(np.select(
        [positive & steepening, positive & ~steepening, ~positive & ~steepening],
        [1.0, 0.3, -0.2],
        default=-1.0,          # inverted AND steepening — the trigger
    ), index=spread.index)

    # Post-inversion re-steepening: positive now, but inverted within the past
    # year. This is the most dangerous configuration in the whole model and it
    # would otherwise be scored +1.0 for being positive and steepening.
    was_inverted = spread.rolling(250, min_periods=60).min() < 0
    base = base.where(~(positive & was_inverted), -1.0)

    # Conviction scales with how fast the curve is moving, floored so a quiet
    # curve still expresses its regime.
    slope_sd = slope.rolling(params.zscore_window, min_periods=60).std(ddof=1)
    intensity = (0.45 + 0.55 * np.tanh(slope.abs() / slope_sd.replace(0.0, np.nan))).fillna(0.6)

    return (base * intensity).clip(-1.0, 1.0).rename("credit_regime")


# --------------------------------------------------------------------------- #
# Factor 3 — real rate impulse
# --------------------------------------------------------------------------- #
def real_rate_impulse(policy: pd.Series, inflation: pd.Series, params: CycleParams) -> pd.Series:
    """Direction of the real policy rate, sign-flipped.

    Falling real rates are risk-on, so the factor is the *negative* of the
    change. The level is deliberately not used: an economy can run at a high
    real rate indefinitely, but the transition is what repricing keys on.
    """
    real = (to_daily(policy) - to_daily(inflation)).dropna()
    if len(real) < params.real_rate_window * 2:
        return pd.Series(dtype=float, name="real_rate_impulse")
    impulse = -real.diff(params.real_rate_window)
    return normalize(impulse, params.zscore_window).rename("real_rate_impulse")


# --------------------------------------------------------------------------- #
# Factor 4 — participation breadth
# --------------------------------------------------------------------------- #
def breadth(oriented_closes: dict[str, pd.Series], params: CycleParams) -> pd.Series:
    """Share of the universe trading above its own moving average, mapped to [-1, 1].

    Inputs must already be risk-oriented (see ``risk_oriented``). Breadth is the
    cheapest available proxy for whether a rally has participants or is being
    carried by two names.
    """
    flags: list[pd.Series] = []
    for symbol, close in oriented_closes.items():
        s = to_daily(close)
        if len(s) < params.breadth_ma * 2:
            continue
        ma = s.rolling(params.breadth_ma, min_periods=params.breadth_ma // 2).mean()
        flags.append((s > ma).astype(float).rename(symbol))

    if not flags:
        return pd.Series(dtype=float, name="breadth")

    frame = pd.concat(flags, axis=1).ffill().dropna(how="all")
    share = frame.mean(axis=1)
    # 0% above MA -> -1, 50% -> 0, 100% -> +1.
    return (2.0 * share - 1.0).rename("breadth")


# --------------------------------------------------------------------------- #
# Factor 5 — valuation stretch
# --------------------------------------------------------------------------- #
def valuation_stretch(metric: pd.Series, params: CycleParams, invert: bool = True) -> pd.Series:
    """Slow-moving valuation anchor.

    ``invert=True`` for metrics where *high means expensive* (NVT, a strong
    dollar REER), so the factor reads negative when the asset is rich.
    """
    s = to_daily(metric)
    if len(s) < 90:
        return pd.Series(dtype=float, name="valuation")
    normalised = normalize(s, params.zscore_window)
    return (-normalised if invert else normalised).rename("valuation")


# --------------------------------------------------------------------------- #
# Factor 6 — positioning / risk appetite
# --------------------------------------------------------------------------- #
def risk_appetite_crypto(
    fear_greed: pd.Series,
    funding: pd.Series | None,
    params: CycleParams,
) -> pd.Series:
    """Contrarian sentiment and crowding for crypto.

    Funding rate is the best positioning gauge in any market: it is the actual
    price of being long, paid in cash, every eight hours.
    """
    parts = []
    fg = to_daily(fear_greed)
    if len(fg) > 60:
        parts.append(-normalize(fg, params.zscore_window))
    if funding is not None:
        fr = to_daily(funding)
        if len(fr) > 60:
            parts.append(-normalize(fr.rolling(7, min_periods=3).mean(), params.zscore_window))

    if not parts:
        return pd.Series(dtype=float, name="risk_appetite")
    return pd.concat(parts, axis=1).ffill().mean(axis=1).rename("risk_appetite")


def risk_appetite_fx(haven_closes: dict[str, pd.Series], params: CycleParams) -> pd.Series:
    """Risk appetite from haven-currency demand.

    Strength in JPY and CHF is the cleanest risk-off tell in FX, and unlike a
    sentiment survey it is a price — someone actually paid it.
    """
    strengths = []
    for close in haven_closes.values():
        s = to_daily(close)
        if len(s) < 90:
            continue
        # Haven strength = momentum of the haven currency itself.
        strengths.append(normalize(s.pct_change(params.impulse_window), params.zscore_window))

    if not strengths:
        return pd.Series(dtype=float, name="risk_appetite")
    haven_demand = pd.concat(strengths, axis=1).ffill().mean(axis=1)
    return (-haven_demand).rename("risk_appetite")


# --------------------------------------------------------------------------- #
# Factor 7 — trend confirmation (lagging)
# --------------------------------------------------------------------------- #
def trend_confirmation(anchor: pd.Series, params: CycleParams) -> pd.Series:
    """Distance of the anchor from its own long moving average.

    This confirms; it never leads. Low weight by construction.
    """
    s = to_daily(anchor)
    if len(s) < 220:
        return pd.Series(dtype=float, name="trend_confirmation")
    ma = s.rolling(200, min_periods=100).mean()
    return normalize(s / ma - 1.0, params.zscore_window).rename("trend_confirmation")


# --------------------------------------------------------------------------- #
# Panel
# --------------------------------------------------------------------------- #
@dataclass
class FactorPanel:
    """The assembled daily factor matrix for one domain."""

    domain: str
    frame: pd.DataFrame           # daily index, one column per factor, all in [-1, 1]
    anchor: pd.Series             # daily risk-proxy price for this domain
    sources: dict[str, str] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)

    @property
    def factors(self) -> list[str]:
        return list(self.frame.columns)

    def latest(self) -> dict[str, float]:
        if self.frame.empty:
            return {}
        row = self.frame.ffill().iloc[-1]
        return {k: float(v) for k, v in row.items() if np.isfinite(v)}

    def __len__(self) -> int:
        return len(self.frame)


def build_dollar_index(oriented: dict[str, pd.Series]) -> pd.Series:
    """Equal-weighted geometric basket of risk-oriented FX pairs.

    Serves as the FX domain's anchor: rising means risk currencies are gaining
    on the dollar, i.e. risk-on — the same orientation as a crypto price, so one
    composite scale covers both domains.
    """
    logs = []
    for symbol, close in oriented.items():
        s = to_daily(close).dropna()
        if len(s) < 90:
            continue
        logs.append(np.log(s / s.iloc[0]).rename(symbol))
    if not logs:
        return pd.Series(dtype=float, name="risk_basket")
    basket = pd.concat(logs, axis=1).ffill().dropna(how="all").mean(axis=1)
    return (100.0 * np.exp(basket)).rename("risk_basket")
