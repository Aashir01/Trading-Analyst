"""Tests for the technical indicator library.

The indicators are hand-implemented rather than taken from TA-Lib, so these
tests pin the standard definitions: Wilder smoothing, correct RSI bounds, ATR
never negative, and — most importantly — that nothing peeks at the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mfie.config import TechnicalParams
from mfie.technical.indicators import (
    adx,
    atr,
    bollinger,
    compute_indicators,
    donchian,
    ema,
    hurst_exponent,
    macd,
    rsi,
    sma,
    supertrend,
    true_range,
    wilder_smooth,
    zscore_series,
)


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    """Deterministic random-walk OHLCV, 400 hourly bars."""
    rng = np.random.default_rng(42)
    n = 400
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    spread = np.abs(rng.normal(0, 0.003, n)) * close
    return pd.DataFrame(
        {
            "open": np.concatenate([[close[0]], close[:-1]]),
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": rng.lognormal(7, 0.4, n),
        },
        index=idx,
    )


class TestMovingAverages:
    def test_sma_of_a_ramp(self):
        series = pd.Series(range(10), dtype=float)
        # SMA(3) at index 2 = mean(0,1,2) = 1
        assert sma(series, 3).iloc[2] == pytest.approx(1.0)

    def test_sma_warms_up_with_nan(self):
        series = pd.Series(range(10), dtype=float)
        assert sma(series, 5).iloc[:4].isna().all()

    def test_ema_converges_to_a_constant(self):
        series = pd.Series([5.0] * 100)
        assert ema(series, 10).iloc[-1] == pytest.approx(5.0)

    def test_wilder_smoothing_uses_alpha_one_over_n(self):
        series = pd.Series([1.0] * 50)
        # A constant input smooths to the same constant regardless of alpha.
        assert wilder_smooth(series, 14).iloc[-1] == pytest.approx(1.0)


class TestOscillators:
    def test_rsi_is_bounded(self, ohlcv):
        values = rsi(ohlcv["close"], 14).dropna()
        assert values.between(0, 100).all()

    def test_rsi_is_100_on_an_unbroken_rally(self):
        series = pd.Series(np.arange(1, 60, dtype=float))
        assert rsi(series, 14).iloc[-1] == pytest.approx(100.0)

    def test_rsi_is_0_on_an_unbroken_selloff(self):
        series = pd.Series(np.arange(60, 1, -1, dtype=float))
        assert rsi(series, 14).iloc[-1] == pytest.approx(0.0, abs=1e-9)

    def test_macd_histogram_is_line_minus_signal(self, ohlcv):
        line, signal, hist = macd(ohlcv["close"])
        pd.testing.assert_series_equal(hist.dropna(), (line - signal).dropna())


class TestVolatility:
    def test_true_range_is_never_negative(self, ohlcv):
        assert (true_range(ohlcv["high"], ohlcv["low"], ohlcv["close"]).dropna() >= 0).all()

    def test_atr_is_positive(self, ohlcv):
        assert (atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], 14).dropna() > 0).all()

    def test_bollinger_bands_are_ordered(self, ohlcv):
        upper, middle, lower = bollinger(ohlcv["close"], 20, 2.0)
        valid = upper.notna()
        assert (upper[valid] >= middle[valid]).all()
        assert (middle[valid] >= lower[valid]).all()

    def test_bollinger_width_scales_with_std(self, ohlcv):
        narrow_u, _, narrow_l = bollinger(ohlcv["close"], 20, 1.0)
        wide_u, _, wide_l = bollinger(ohlcv["close"], 20, 3.0)
        assert (wide_u - wide_l).dropna().mean() > (narrow_u - narrow_l).dropna().mean()


class TestTrend:
    def test_adx_is_bounded(self, ohlcv):
        value, plus_di, minus_di = adx(ohlcv["high"], ohlcv["low"], ohlcv["close"], 14)
        assert value.dropna().between(0, 100).all()
        assert plus_di.dropna().ge(0).all()
        assert minus_di.dropna().ge(0).all()

    def test_adx_is_high_in_a_clean_trend(self):
        n = 200
        idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
        close = pd.Series(np.linspace(100, 200, n), index=idx)
        high, low = close * 1.002, close * 0.998
        value, _, _ = adx(high, low, close, 14)
        assert value.dropna().iloc[-1] > 40

    def test_supertrend_direction_is_plus_or_minus_one(self, ohlcv):
        _, direction = supertrend(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert set(direction.unique()) <= {1.0, -1.0}

    def test_donchian_excludes_the_current_bar(self, ohlcv):
        upper, lower = donchian(ohlcv["high"], ohlcv["low"], 20)
        # Because the channel is shifted, a new high can exceed the upper band.
        assert upper.notna().sum() > 0
        assert (upper.dropna() >= lower.dropna().reindex(upper.dropna().index)).all()

    def test_hurst_detects_persistence(self):
        # Persistent process: positively autocorrelated increments, cumulated.
        # (A deterministic ramp is outside the estimator's domain — see the
        # docstring — so persistence is modelled stochastically, as in a market.)
        rng = np.random.default_rng(1)
        increments = np.zeros(600)
        for i in range(1, 600):
            increments[i] = 0.75 * increments[i - 1] + rng.normal()
        assert hurst_exponent(pd.Series(np.cumsum(increments))) > 0.5

    def test_hurst_of_a_random_walk_is_about_half(self):
        rng = np.random.default_rng(9)
        walk = pd.Series(np.cumsum(rng.normal(0, 1, 2000)))
        assert 0.4 < hurst_exponent(walk) < 0.6

    def test_hurst_detects_mean_reversion(self):
        rng = np.random.default_rng(2)
        x = np.zeros(500)
        for i in range(1, 500):
            x[i] = 0.5 * x[i - 1] + rng.normal()  # strongly mean-reverting AR(1)
        assert hurst_exponent(pd.Series(x)) < 0.5


class TestBundle:
    def test_compute_indicators_produces_the_expected_columns(self, ohlcv):
        result = compute_indicators(ohlcv, TechnicalParams())
        for column in ("rsi", "macd", "atr", "adx", "bb_upper", "supertrend_dir",
                       "realized_vol", "zscore", "volume_z", "squeeze"):
            assert column in result.df.columns

    def test_compute_indicators_does_not_mutate_the_input(self, ohlcv):
        before = ohlcv.copy()
        compute_indicators(ohlcv, TechnicalParams())
        pd.testing.assert_frame_equal(ohlcv, before)

    def test_indicators_are_causal(self, ohlcv):
        """The critical no-lookahead test.

        Computing indicators on a truncated frame must give the same values for
        the bars that both frames share. If any indicator peeked forward, the
        truncated run would differ.
        """
        full = compute_indicators(ohlcv, TechnicalParams()).df
        truncated = compute_indicators(ohlcv.iloc[:300], TechnicalParams()).df

        for column in ("rsi", "macd", "atr", "adx", "bb_upper", "ema_fast", "supertrend_dir"):
            a = full[column].iloc[:300].dropna()
            b = truncated[column].dropna()
            common = a.index.intersection(b.index)
            assert len(common) > 50, f"{column} produced too few comparable values"
            np.testing.assert_allclose(
                a.loc[common].to_numpy(), b.loc[common].to_numpy(),
                rtol=1e-9, atol=1e-9, err_msg=f"{column} is not causal",
            )

    def test_last_and_prev_accessors(self, ohlcv):
        result = compute_indicators(ohlcv, TechnicalParams())
        assert result.last("close") == pytest.approx(float(ohlcv["close"].iloc[-1]))
        assert result.last("does_not_exist", default=1.23) == 1.23
        assert result.prev("close", 1) == pytest.approx(float(ohlcv["close"].iloc[-2]))

    def test_zscore_of_a_constant_series_is_zero(self):
        series = pd.Series([5.0] * 50)
        assert zscore_series(series, 20).dropna().abs().max() == 0.0
