"""Technical indicators implemented directly on pandas/numpy.

Why not TA-Lib: TA-Lib is a C library that needs a compiler or a matching
prebuilt wheel, which is the single most common reason a Python trading project
fails to install on Windows. Everything here is vectorised pandas, so the
project installs anywhere pandas does. The formulas follow the standard
definitions (Wilder smoothing for RSI/ATR/ADX), so values line up with TA-Lib to
floating-point noise once the warm-up period has passed.

Every function takes and returns pandas objects indexed by time and never
mutates its input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mfie.config import TechnicalParams


# --------------------------------------------------------------------------- #
# Moving averages
# --------------------------------------------------------------------------- #
def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(period).apply(
        lambda w: float(np.dot(w, weights) / weights.sum()), raw=True
    )


def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing: an EMA with alpha = 1/period.

    This is what separates a 'real' RSI/ATR/ADX from a naive rolling mean.
    """
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def hma(series: pd.Series, period: int) -> pd.Series:
    """Hull moving average — low lag, used by the trend strategy."""
    half = max(int(period / 2), 1)
    sqrt_p = max(int(np.sqrt(period)), 1)
    return wma(2 * wma(series, half) - wma(series, period), sqrt_p)


# --------------------------------------------------------------------------- #
# Momentum / oscillators
# --------------------------------------------------------------------------- #
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index, Wilder smoothing. RSI = 100 - 100/(1+RS)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = wilder_smooth(gain, period)
    avg_loss = wilder_smooth(loss, period)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss == 0 means an unbroken up-run: RSI is 100 by definition.
    return out.where(avg_loss != 0.0, 100.0)


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (macd_line, signal_line, histogram)."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd_line, signal_line, macd_line - signal_line


def stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3
) -> tuple[pd.Series, pd.Series]:
    lowest = low.rolling(k_period, min_periods=k_period).min()
    highest = high.rolling(k_period, min_periods=k_period).max()
    k = 100.0 * (close - lowest) / (highest - lowest).replace(0.0, np.nan)
    return k, k.rolling(d_period, min_periods=d_period).mean()


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    typical = (high + low + close) / 3.0
    ma = typical.rolling(period, min_periods=period).mean()
    mad = typical.rolling(period, min_periods=period).apply(
        lambda x: float(np.mean(np.abs(x - x.mean()))), raw=True
    )
    return (typical - ma) / (0.015 * mad.replace(0.0, np.nan))


def roc(close: pd.Series, period: int = 10) -> pd.Series:
    return close.pct_change(period) * 100.0


def momentum(close: pd.Series, period: int = 10) -> pd.Series:
    return close.diff(period)


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    highest = high.rolling(period, min_periods=period).max()
    lowest = low.rolling(period, min_periods=period).min()
    return -100.0 * (highest - close) / (highest - lowest).replace(0.0, np.nan)


# --------------------------------------------------------------------------- #
# Volatility
# --------------------------------------------------------------------------- #
def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return wilder_smooth(true_range(high, low, close), period)


def bollinger(
    close: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (upper, middle, lower)."""
    middle = sma(close, period)
    sd = close.rolling(period, min_periods=period).std(ddof=0)
    return middle + num_std * sd, middle, middle - num_std * sd


def bollinger_bandwidth(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """(upper-lower)/middle. Low bandwidth = squeeze = breakout precursor."""
    upper, middle, lower = bollinger(close, period, num_std)
    return (upper - lower) / middle.replace(0.0, np.nan)


def keltner(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20, mult: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    middle = ema(close, period)
    band = atr(high, low, close, period) * mult
    return middle + band, middle, middle - band


def donchian(high: pd.Series, low: pd.Series, period: int = 20) -> tuple[pd.Series, pd.Series]:
    """Rolling channel *excluding* the current bar, so a touch is a real break."""
    return (
        high.shift(1).rolling(period, min_periods=period).max(),
        low.shift(1).rolling(period, min_periods=period).min(),
    )


def realized_volatility(close: pd.Series, period: int = 20, periods_per_year: float = 8760) -> pd.Series:
    returns = np.log(close / close.shift(1))
    return returns.rolling(period, min_periods=period).std(ddof=1) * np.sqrt(periods_per_year)


def parkinson_volatility(high: pd.Series, low: pd.Series, period: int = 20,
                         periods_per_year: float = 8760) -> pd.Series:
    """Range-based volatility — more efficient than close-to-close."""
    hl = np.log(high / low) ** 2
    var = hl.rolling(period, min_periods=period).mean() / (4.0 * np.log(2.0))
    return np.sqrt(var * periods_per_year)


# --------------------------------------------------------------------------- #
# Trend strength
# --------------------------------------------------------------------------- #
def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
        ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Average Directional Index. Returns (adx, plus_di, minus_di)."""
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
                        index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
                         index=high.index)

    atr_ = wilder_smooth(true_range(high, low, close), period).replace(0.0, np.nan)
    plus_di = 100.0 * wilder_smooth(plus_dm, period) / atr_
    minus_di = 100.0 * wilder_smooth(minus_dm, period) / atr_

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return wilder_smooth(dx, period), plus_di, minus_di


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = 10, multiplier: float = 3.0) -> tuple[pd.Series, pd.Series]:
    """Returns (supertrend_line, direction) where direction is +1 up / -1 down."""
    hl2 = (high + low) / 2.0
    band = multiplier * atr(high, low, close, period)
    upper_basic = (hl2 + band).to_numpy()
    lower_basic = (hl2 - band).to_numpy()
    close_arr = close.to_numpy()

    n = len(close)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    trend = np.ones(n)

    for i in range(1, n):
        if np.isnan(upper_basic[i]):
            continue
        upper[i] = (
            upper_basic[i]
            if np.isnan(upper[i - 1]) or upper_basic[i] < upper[i - 1] or close_arr[i - 1] > upper[i - 1]
            else upper[i - 1]
        )
        lower[i] = (
            lower_basic[i]
            if np.isnan(lower[i - 1]) or lower_basic[i] > lower[i - 1] or close_arr[i - 1] < lower[i - 1]
            else lower[i - 1]
        )
        if not np.isnan(upper[i - 1]) and close_arr[i] > upper[i - 1]:
            trend[i] = 1.0
        elif not np.isnan(lower[i - 1]) and close_arr[i] < lower[i - 1]:
            trend[i] = -1.0
        else:
            trend[i] = trend[i - 1]

    line = np.where(trend > 0, lower, upper)
    return pd.Series(line, index=close.index), pd.Series(trend, index=close.index)


def hurst_exponent(series: pd.Series, max_lag: int = 40) -> float:
    """Hurst estimate from the growth of the variance of lagged differences.

    H < 0.5 => mean reverting, H ~ 0.5 => random walk, H > 0.5 => trending.
    Used by the regime detector to choose between momentum and mean reversion.

    Input is a *price* series and the process is assumed stochastic. A purely
    deterministic ramp is outside the estimator's domain — its lagged
    differences are constant, so the fitted slope collapses toward 0 rather
    than rising above 0.5. Real price series always carry stochastic
    increments, so this matters only for synthetic inputs.
    """
    s = pd.Series(series).dropna().astype(float).to_numpy()
    if len(s) < max_lag * 2:
        return 0.5
    lags = range(2, min(max_lag, len(s) // 2))
    tau = []
    valid_lags = []
    for lag in lags:
        diff = s[lag:] - s[:-lag]
        sd = float(np.std(diff))
        if sd > 0:
            tau.append(sd)
            valid_lags.append(lag)
    if len(tau) < 4:
        return 0.5
    slope = np.polyfit(np.log(valid_lags), np.log(tau), 1)[0]
    return float(np.clip(slope, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Volume
# --------------------------------------------------------------------------- #
def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume).cumsum()


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
         period: int | None = None) -> pd.Series:
    typical = (high + low + close) / 3.0
    if period:
        pv = (typical * volume).rolling(period, min_periods=period).sum()
        vol = volume.rolling(period, min_periods=period).sum()
    else:
        pv = (typical * volume).cumsum()
        vol = volume.cumsum()
    return pv / vol.replace(0.0, np.nan)


def money_flow_index(high: pd.Series, low: pd.Series, close: pd.Series,
                     volume: pd.Series, period: int = 14) -> pd.Series:
    typical = (high + low + close) / 3.0
    raw_flow = typical * volume
    direction = typical.diff()
    positive = raw_flow.where(direction > 0, 0.0).rolling(period, min_periods=period).sum()
    negative = raw_flow.where(direction < 0, 0.0).rolling(period, min_periods=period).sum()
    ratio = positive / negative.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + ratio))


def volume_zscore(volume: pd.Series, period: int = 20) -> pd.Series:
    mu = volume.rolling(period, min_periods=period).mean()
    sd = volume.rolling(period, min_periods=period).std(ddof=1)
    return (volume - mu) / sd.replace(0.0, np.nan)


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #
def swing_points(high: pd.Series, low: pd.Series, left: int = 3, right: int = 3
                 ) -> tuple[pd.Series, pd.Series]:
    """Boolean series marking confirmed swing highs and swing lows (fractals)."""
    window = left + right + 1
    is_high = (
        high.rolling(window, center=True).max().eq(high)
        & high.notna()
    ).fillna(False)
    is_low = (
        low.rolling(window, center=True).min().eq(low)
        & low.notna()
    ).fillna(False)
    return is_high, is_low


def rsi_divergence(close: pd.Series, rsi_series: pd.Series, lookback: int = 40
                   ) -> tuple[bool, bool]:
    """Detect regular divergence over the last ``lookback`` bars.

    Returns (bullish_divergence, bearish_divergence): price makes a lower low
    while RSI makes a higher low (bullish), or the mirror image (bearish).
    """
    px = close.tail(lookback).dropna()
    rs = rsi_series.tail(lookback).dropna()
    if len(px) < 10 or len(rs) < 10:
        return False, False

    half = len(px) // 2
    first_px, second_px = px.iloc[:half], px.iloc[half:]
    first_rs, second_rs = rs.iloc[:half], rs.iloc[half:]

    bullish = bool(second_px.min() < first_px.min() and second_rs.min() > first_rs.min())
    bearish = bool(second_px.max() > first_px.max() and second_rs.max() < first_rs.max())
    return bullish, bearish


def zscore_series(series: pd.Series, period: int = 20) -> pd.Series:
    mu = series.rolling(period, min_periods=period).mean()
    sd = series.rolling(period, min_periods=period).std(ddof=1)
    z = (series - mu) / sd
    # A zero-variance window is not undefined — every value equals the mean,
    # so the z-score is exactly 0. Leaving it NaN would silently drop the bar
    # from the ML feature matrix.
    return z.mask(sd == 0.0, 0.0)


# --------------------------------------------------------------------------- #
# Bundle
# --------------------------------------------------------------------------- #
@dataclass
class IndicatorSet:
    """All indicators for one OHLCV frame, plus scalar 'latest' accessors.

    Strategies read from here rather than recomputing, so one pass over the
    frame serves the whole strategy library.
    """

    df: pd.DataFrame

    def last(self, column: str, default: float = float("nan")) -> float:
        if column not in self.df.columns:
            return default
        series = self.df[column].dropna()
        return float(series.iloc[-1]) if not series.empty else default

    def prev(self, column: str, offset: int = 1, default: float = float("nan")) -> float:
        if column not in self.df.columns:
            return default
        series = self.df[column].dropna()
        return float(series.iloc[-1 - offset]) if len(series) > offset else default

    def series(self, column: str) -> pd.Series:
        return self.df[column] if column in self.df.columns else pd.Series(dtype=float)

    @property
    def close(self) -> pd.Series:
        return self.df["close"]

    @property
    def price(self) -> float:
        return self.last("close")


def compute_indicators(df: pd.DataFrame, params: TechnicalParams | None = None,
                       periods_per_year: float = 8760) -> IndicatorSet:
    """Attach the full indicator suite to a copy of an OHLCV frame."""
    p = params or TechnicalParams()
    out = df.copy()
    high, low, close, volume = out["high"], out["low"], out["close"], out["volume"]

    out["returns"] = close.pct_change()
    out["log_returns"] = np.log(close / close.shift(1))

    out["ema_fast"] = ema(close, p.ema_fast)
    out["ema_slow"] = ema(close, p.ema_slow)
    out["ema_trend"] = ema(close, p.ema_trend)
    out["sma_50"] = sma(close, 50)
    out["hma"] = hma(close, p.ema_fast)

    out["rsi"] = rsi(close, p.rsi_period)
    macd_line, macd_signal, macd_hist = macd(close, p.macd_fast, p.macd_slow, p.macd_signal)
    out["macd"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_hist

    upper, middle, lower = bollinger(close, p.bb_period, p.bb_std)
    out["bb_upper"], out["bb_middle"], out["bb_lower"] = upper, middle, lower
    out["bb_bandwidth"] = bollinger_bandwidth(close, p.bb_period, p.bb_std)
    out["bb_percent_b"] = (close - lower) / (upper - lower).replace(0.0, np.nan)

    k_upper, k_middle, k_lower = keltner(high, low, close, p.bb_period)
    out["kc_upper"], out["kc_lower"] = k_upper, k_lower
    # Classic TTM squeeze: Bollinger Bands inside Keltner Channels.
    out["squeeze"] = (upper < k_upper) & (lower > k_lower)

    out["atr"] = atr(high, low, close, p.atr_period)
    out["atr_pct"] = out["atr"] / close
    adx_, plus_di, minus_di = adx(high, low, close, p.adx_period)
    out["adx"], out["plus_di"], out["minus_di"] = adx_, plus_di, minus_di

    st_line, st_dir = supertrend(high, low, close)
    out["supertrend"], out["supertrend_dir"] = st_line, st_dir

    dc_upper, dc_lower = donchian(high, low, p.donchian_period)
    out["donchian_upper"], out["donchian_lower"] = dc_upper, dc_lower

    stoch_k, stoch_d = stochastic(high, low, close)
    out["stoch_k"], out["stoch_d"] = stoch_k, stoch_d
    out["cci"] = cci(high, low, close)
    out["williams_r"] = williams_r(high, low, close)
    out["roc"] = roc(close)

    out["obv"] = obv(close, volume)
    out["mfi"] = money_flow_index(high, low, close, volume)
    out["vwap"] = vwap(high, low, close, volume, period=p.bb_period)
    out["volume_z"] = volume_zscore(volume, p.bb_period)

    out["realized_vol"] = realized_volatility(close, p.bb_period, periods_per_year)
    out["parkinson_vol"] = parkinson_volatility(high, low, p.bb_period, periods_per_year)
    out["zscore"] = zscore_series(close, p.zscore_period)

    swing_high, swing_low = swing_points(high, low)
    out["swing_high"], out["swing_low"] = swing_high, swing_low

    return IndicatorSet(out)
