r"""Market microstructure: execution friction, depth and order-flow imbalance.

Liquidity Friction Factor — the spread measured in units of the asset's own
volatility, so it is comparable across BTC and EURUSD:

.. math::  LF_t = \frac{Ask_t - Bid_t}{ATR_t(n)}

Rules:

* :math:`LF_t >` ``friction_halt``: depth is dangerously thin (typically the
  minutes around a CPI or FOMC print). Halt execution.
* ``friction_warn`` :math:`< LF_t \le` ``friction_halt``: widen the stop by
  :math:`(1 + LF_t)` so ordinary slippage does not take you out of an otherwise
  valid trade, and shrink size to keep risk constant.

Also includes Amihud illiquidity and a Kyle-lambda style price-impact estimate,
which are the standard academic measures of "how much does my order move this
market".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mfie.config import MicrostructureParams
from mfie.core.types import Quote
from mfie.core.utils import clamp, safe_div


def liquidity_friction(quote: Quote, atr_value: float) -> float:
    """LF_t = spread / ATR. Returns 0 when ATR is unusable."""
    return safe_div(quote.spread, atr_value, 0.0)


def amihud_illiquidity(returns: pd.Series, volume_usd: pd.Series, window: int = 30) -> float:
    r"""Amihud (2002): mean of :math:`|r_t| / \text{volume}_t`.

    Higher means each dollar of volume moves price more, i.e. thinner market.
    """
    r = pd.Series(returns).abs()
    v = pd.Series(volume_usd).replace(0.0, np.nan)
    frame = pd.concat([r.rename("r"), v.rename("v")], axis=1).dropna().tail(window)
    if frame.empty:
        return 0.0
    ratio = frame["r"] / frame["v"]
    return float(ratio.mean() * 1e6)  # scaled for readability


def kyle_lambda(price: pd.Series, signed_volume: pd.Series, window: int = 60) -> float:
    r"""Price impact coefficient from :math:`\Delta P = \lambda \cdot \text{signed volume}`.

    Estimated by OLS through the origin over the last ``window`` bars.
    """
    dp = pd.Series(price).diff()
    frame = pd.concat([dp.rename("dp"), pd.Series(signed_volume).rename("sv")], axis=1)
    frame = frame.dropna().tail(window)
    if len(frame) < 10:
        return 0.0
    x = frame["sv"].to_numpy()
    y = frame["dp"].to_numpy()
    denom = float(np.dot(x, x))
    if denom == 0:
        return 0.0
    return float(np.dot(x, y) / denom)


def order_book_imbalance(depth: dict | None, levels: int = 20) -> float:
    """(bid size - ask size) / total, over the top ``levels`` of the book.

    +1 means all resting size is on the bid; -1 all on the ask.
    """
    if not depth or "bids" not in depth or "asks" not in depth:
        return 0.0
    try:
        bid_size = sum(float(row[1]) for row in depth["bids"][:levels])
        ask_size = sum(float(row[1]) for row in depth["asks"][:levels])
    except (TypeError, ValueError, IndexError):
        return 0.0
    total = bid_size + ask_size
    return safe_div(bid_size - ask_size, total, 0.0)


def roll_effective_spread(close: pd.Series, window: int = 60) -> float:
    r"""Roll (1984) implied spread: :math:`2\sqrt{-\text{Cov}(\Delta p_t, \Delta p_{t-1})}`.

    Estimates the true round-trip cost from trade prices alone — useful when a
    venue gives you candles but no book.
    """
    dp = pd.Series(close).diff().dropna().tail(window).to_numpy()
    if len(dp) < 10:
        return 0.0
    cov = float(np.cov(dp[1:], dp[:-1])[0, 1])
    # A non-negative autocovariance means the Roll model does not apply here
    # (trending prices rather than bid-ask bounce); report no estimate.
    if not np.isfinite(cov) or cov >= 0:
        return 0.0
    return float(2.0 * np.sqrt(-cov))


@dataclass
class MicrostructureView:
    symbol: str
    spread: float
    spread_bps: float
    atr: float
    friction: float           # LF_t
    state: str                # normal | thin | halt
    stop_multiplier: float
    size_multiplier: float
    imbalance: float = 0.0
    amihud: float = 0.0
    detail: dict[str, float] | None = None

    @property
    def should_halt(self) -> bool:
        return self.state == "halt"


def compute_microstructure(
    quote: Quote,
    atr_value: float,
    params: MicrostructureParams | None = None,
    returns: pd.Series | None = None,
    volume_usd: pd.Series | None = None,
    depth: dict | None = None,
) -> MicrostructureView:
    p = params or MicrostructureParams()
    friction = liquidity_friction(quote, atr_value)

    if friction >= p.friction_halt:
        state = "halt"
        stop_mult = clamp(1.0 + friction, 1.0, p.max_stop_multiplier)
        size_mult = 0.0
    elif friction > p.friction_warn:
        state = "thin"
        stop_mult = clamp(1.0 + friction, 1.0, p.max_stop_multiplier) if p.widen_stops else 1.0
        # Widening the stop without shrinking size would silently increase risk;
        # keep risk constant by scaling size down by the same factor.
        size_mult = clamp(1.0 / stop_mult, 0.2, 1.0)
    else:
        state = "normal"
        stop_mult = 1.0
        size_mult = 1.0

    return MicrostructureView(
        symbol=quote.symbol,
        spread=quote.spread,
        spread_bps=quote.spread_bps,
        atr=float(atr_value) if np.isfinite(atr_value) else 0.0,
        friction=float(friction),
        state=state,
        stop_multiplier=float(stop_mult),
        size_multiplier=float(size_mult),
        imbalance=order_book_imbalance(depth),
        amihud=amihud_illiquidity(returns, volume_usd) if returns is not None and volume_usd is not None else 0.0,
        detail={"warn_threshold": p.friction_warn, "halt_threshold": p.friction_halt},
    )
