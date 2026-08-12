"""Feature engineering for the supervised direction model.

Two rules govern everything here:

1. **No lookahead.** Every feature at bar ``t`` uses only information available
   at the close of bar ``t``. Labels look forward; features never do.
2. **Stationarity.** Raw price is non-stationary and a tree will happily
   memorise price levels that never recur. Features are ratios, z-scores,
   normalised distances and returns — all scale-free and comparable across
   assets and epochs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mfie.technical.indicators import IndicatorSet

# The model's input contract. Kept explicit so a model trained today cannot be
# silently fed a different feature set tomorrow.
FEATURE_COLUMNS = [
    "ret_1", "ret_5", "ret_20",
    "rsi_norm", "macd_hist_norm", "percent_b", "zscore",
    "adx_norm", "di_diff", "atr_pct", "vol_ratio",
    "dist_ema_fast", "dist_ema_slow", "dist_ema_trend", "dist_vwap",
    "volume_z", "mfi_norm", "stoch_norm", "williams_norm", "cci_norm",
    "supertrend_dir", "squeeze", "bandwidth_pct",
    "hl_range", "body_ratio", "upper_wick", "lower_wick",
    "ret_skew", "ret_kurt", "autocorr_5",
]


def build_feature_matrix(indicators: IndicatorSet) -> pd.DataFrame:
    """Turn an ``IndicatorSet`` into the model's stationary feature frame."""
    df = indicators.df
    close = df["close"]
    high, low, open_ = df["high"], df["low"], df["open"]

    out = pd.DataFrame(index=df.index)

    # Momentum over several horizons.
    out["ret_1"] = close.pct_change(1)
    out["ret_5"] = close.pct_change(5)
    out["ret_20"] = close.pct_change(20)

    # Oscillators, centred and scaled to roughly [-1, 1].
    out["rsi_norm"] = (df.get("rsi", pd.Series(index=df.index, dtype=float)) - 50.0) / 50.0
    out["macd_hist_norm"] = df.get("macd_hist", 0.0) / close
    out["percent_b"] = df.get("bb_percent_b", 0.5) - 0.5
    out["zscore"] = df.get("zscore", 0.0)

    # Trend strength.
    out["adx_norm"] = df.get("adx", 0.0) / 100.0
    out["di_diff"] = (df.get("plus_di", 0.0) - df.get("minus_di", 0.0)) / 100.0

    # Volatility.
    out["atr_pct"] = df.get("atr_pct", 0.0)
    realized = df.get("realized_vol", pd.Series(index=df.index, dtype=float))
    out["vol_ratio"] = realized / realized.rolling(60, min_periods=20).mean()

    # Distance from reference levels, in ATR units so it is scale-free.
    atr = df.get("atr", pd.Series(index=df.index, dtype=float)).replace(0.0, np.nan)
    for name, column in (
        ("dist_ema_fast", "ema_fast"),
        ("dist_ema_slow", "ema_slow"),
        ("dist_ema_trend", "ema_trend"),
        ("dist_vwap", "vwap"),
    ):
        if column in df.columns:
            out[name] = (close - df[column]) / atr
        else:
            out[name] = 0.0

    # Participation.
    out["volume_z"] = df.get("volume_z", 0.0)
    out["mfi_norm"] = (df.get("mfi", 50.0) - 50.0) / 50.0
    out["stoch_norm"] = (df.get("stoch_k", 50.0) - 50.0) / 50.0
    out["williams_norm"] = (df.get("williams_r", -50.0) + 50.0) / 50.0
    out["cci_norm"] = df.get("cci", 0.0) / 200.0

    out["supertrend_dir"] = df.get("supertrend_dir", 0.0)
    out["squeeze"] = df.get("squeeze", False).astype(float) if "squeeze" in df.columns else 0.0
    bandwidth = df.get("bb_bandwidth", pd.Series(index=df.index, dtype=float))
    out["bandwidth_pct"] = bandwidth.rolling(100, min_periods=20).rank(pct=True)

    # Candle anatomy — cheap proxies for intrabar order flow.
    rng = (high - low).replace(0.0, np.nan)
    out["hl_range"] = rng / close
    out["body_ratio"] = (close - open_) / rng
    out["upper_wick"] = (high - np.maximum(close, open_)) / rng
    out["lower_wick"] = (np.minimum(close, open_) - low) / rng

    # Higher moments and persistence of the return distribution.
    returns = close.pct_change()
    out["ret_skew"] = returns.rolling(60, min_periods=30).skew()
    out["ret_kurt"] = returns.rolling(60, min_periods=30).kurt()
    out["autocorr_5"] = returns.rolling(60, min_periods=30).apply(
        lambda x: pd.Series(x).autocorr(lag=5), raw=False
    )

    out = out.replace([np.inf, -np.inf], np.nan)
    return out[FEATURE_COLUMNS]


def make_labels(
    close: pd.Series,
    horizon: int = 12,
    threshold: float = 0.0,
) -> pd.Series:
    """Binary label: 1 if forward return over ``horizon`` exceeds ``threshold``.

    A non-zero ``threshold`` (e.g. 0.5 * ATR%) turns this into a "meaningful
    move" classifier instead of a coin-flip on noise, which is usually what you
    actually want to trade.
    """
    forward = close.shift(-horizon) / close - 1.0
    labels = (forward > threshold).astype(float)
    # The final ``horizon`` bars have no future — they must not become training rows.
    labels.iloc[-horizon:] = np.nan
    return labels.rename("label")


def align_xy(features: pd.DataFrame, labels: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Drop rows where either side is incomplete, preserving order."""
    frame = features.join(labels.rename("label"), how="inner").dropna()
    if frame.empty:
        return pd.DataFrame(columns=features.columns), pd.Series(dtype=float)
    return frame[features.columns], frame["label"]


def purge_embargo(
    x: pd.DataFrame,
    y: pd.Series,
    split_index: int,
    horizon: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Train/test split with a purge gap of ``horizon`` bars.

    Because labels look ``horizon`` bars forward, the last training rows overlap
    the first test rows in time. Without the gap the model sees the answer —
    this is the single most common way backtested ML results end up fictional.
    """
    train_end = max(split_index - horizon, 0)
    return (
        x.iloc[:train_end],
        y.iloc[:train_end],
        x.iloc[split_index:],
        y.iloc[split_index:],
    )
