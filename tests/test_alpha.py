"""Tests for the Market Cycle Compass.

The important ones are the behavioural tests on the credit factor (the
non-monotonic curve logic) and on the statistics (overlap correction, no sign
flipping). Those are the pieces where a subtle bug produces a confident,
wrong answer rather than a crash.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mfie.alpha import factors as F
from mfie.alpha.cycle import CycleEngine, CyclePhase
from mfie.alpha.leadlag import (
    LeadLagResult,
    _t_stat,
    align_to_common_horizon,
    effective_sample_size,
    forward_return,
    information_coefficient,
    shrink_weights,
)
from mfie.config import CycleParams
from mfie.core.universe import get_instrument


def _days(n: int) -> pd.DatetimeIndex:
    return pd.date_range(end="2026-08-01", periods=n, freq="D", tz="UTC")


@pytest.fixture(scope="module")
def engine() -> CycleEngine:
    return CycleEngine()


@pytest.fixture
def cfg() -> CycleParams:
    return CycleParams()


# --------------------------------------------------------------------------- #
class TestNormalisation:
    def test_output_is_bounded(self):
        rough = pd.Series(np.concatenate([np.random.default_rng(1).normal(0, 1, 400), [500.0]]),
                          index=_days(401))
        out = F.normalize(rough, 200).dropna()
        assert out.between(-1.0, 1.0).all()

    def test_an_outlier_cannot_dominate(self):
        """tanh saturation: a 500-sigma print must not swamp the composite."""
        base = pd.Series(np.random.default_rng(2).normal(0, 1, 400), index=_days(400))
        spiked = base.copy()
        spiked.iloc[-1] = 500.0
        assert F.normalize(spiked, 200).iloc[-1] < 1.001

    def test_daily_resampling_forward_fills(self):
        monthly = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-01-31", "2026-02-28"], utc=True))
        daily = F.to_daily(monthly)
        # Mid-February you knew January's print, not February's.
        assert daily.loc["2026-02-10"] == 1.0
        assert daily.loc["2026-02-28"] == 2.0


class TestRiskOrientation:
    def test_usd_base_pairs_are_inverted(self):
        usdjpy = get_instrument("USDJPY")
        close = pd.Series([150.0, 155.0], index=_days(2))
        oriented = F.risk_oriented(usdjpy, close)
        # USDJPY rising is risk-OFF, so the oriented series must fall.
        assert oriented.iloc[-1] < oriented.iloc[0]

    def test_usd_quote_pairs_are_untouched(self):
        eurusd = get_instrument("EURUSD")
        close = pd.Series([1.08, 1.10], index=_days(2))
        assert F.risk_oriented(eurusd, close).iloc[-1] > F.risk_oriented(eurusd, close).iloc[0]

    def test_crypto_is_untouched(self):
        btc = get_instrument("BTCUSDT")
        close = pd.Series([60000.0, 65000.0], index=_days(2))
        pd.testing.assert_series_equal(F.risk_oriented(btc, close), close)


class TestCreditRegime:
    """The non-monotonic curve factor — the differentiated piece of logic."""

    def _curve(self, spread_path: np.ndarray, cfg: CycleParams):
        idx = _days(len(spread_path))
        y2 = pd.Series(np.full(len(spread_path), 0.04), index=idx)
        y10 = pd.Series(0.04 + spread_path, index=idx)
        return F.credit_regime(y10, y2, cfg).dropna()

    def test_positive_and_steepening_is_bullish(self, cfg):
        path = np.linspace(0.001, 0.02, 400)  # positive, steepening throughout
        assert self._curve(path, cfg).iloc[-1] > 0.3

    def test_positive_and_flattening_is_mildly_bullish(self, cfg):
        path = np.linspace(0.02, 0.004, 400)  # positive but flattening
        score = self._curve(path, cfg).iloc[-1]
        assert 0 < score < 0.35

    def test_inverted_and_deepening_is_only_mildly_negative(self, cfg):
        """Markets historically melt up through an inversion — this must not
        be scored as the maximum bear signal."""
        path = np.linspace(-0.001, -0.02, 400)
        score = self._curve(path, cfg).iloc[-1]
        assert -0.35 < score < 0

    def test_bull_steepener_out_of_inversion_is_maximally_bearish(self, cfg):
        """The actual recession trigger: inverted and re-steepening."""
        path = np.concatenate([np.linspace(0.005, -0.02, 250), np.linspace(-0.02, -0.002, 150)])
        assert self._curve(path, cfg).iloc[-1] < -0.4

    def test_recent_un_inversion_stays_bearish_despite_positive_spread(self, cfg):
        """A curve that just crossed back above zero is the most dangerous
        configuration in the model, and would otherwise score +1.0."""
        path = np.concatenate([np.linspace(0.005, -0.015, 200), np.linspace(-0.015, 0.004, 200)])
        assert self._curve(path, cfg).iloc[-1] < -0.4

    def test_ordering_of_the_four_states(self, cfg):
        """Steep+steepening must beat every other configuration."""
        best = self._curve(np.linspace(0.001, 0.02, 400), cfg).iloc[-1]
        inverted_steepening = self._curve(
            np.concatenate([np.linspace(0.005, -0.02, 250), np.linspace(-0.02, -0.002, 150)]), cfg
        ).iloc[-1]
        assert best > inverted_steepening


class TestBreadth:
    def test_all_above_ma_reads_plus_one(self, cfg):
        rising = {f"A{i}": pd.Series(np.linspace(1, 2, 300), index=_days(300)) for i in range(4)}
        assert F.breadth(rising, cfg).iloc[-1] == pytest.approx(1.0)

    def test_all_below_ma_reads_minus_one(self, cfg):
        falling = {f"A{i}": pd.Series(np.linspace(2, 1, 300), index=_days(300)) for i in range(4)}
        assert F.breadth(falling, cfg).iloc[-1] == pytest.approx(-1.0)

    def test_half_and_half_reads_zero(self, cfg):
        mixed = {
            "up1": pd.Series(np.linspace(1, 2, 300), index=_days(300)),
            "up2": pd.Series(np.linspace(1, 2, 300), index=_days(300)),
            "dn1": pd.Series(np.linspace(2, 1, 300), index=_days(300)),
            "dn2": pd.Series(np.linspace(2, 1, 300), index=_days(300)),
        }
        assert F.breadth(mixed, cfg).iloc[-1] == pytest.approx(0.0)


class TestRealRateImpulse:
    def test_falling_real_rates_score_positive(self, cfg):
        idx = _days(500)
        policy = pd.Series(np.linspace(0.05, 0.02, 500), index=idx)
        cpi = pd.Series(np.full(500, 0.02), index=idx)
        assert F.real_rate_impulse(policy, cpi, cfg).iloc[-1] > 0

    def test_rising_real_rates_score_negative(self, cfg):
        idx = _days(500)
        policy = pd.Series(np.linspace(0.01, 0.06, 500), index=idx)
        cpi = pd.Series(np.full(500, 0.02), index=idx)
        assert F.real_rate_impulse(policy, cpi, cfg).iloc[-1] < 0


# --------------------------------------------------------------------------- #
class TestStatistics:
    def test_overlap_correction_shrinks_the_sample(self):
        """700 daily readings of a 63-day return are ~11 independent observations."""
        assert effective_sample_size(700, 63) == pytest.approx(700 / 63)
        assert effective_sample_size(700, 1) == pytest.approx(700)

    def test_overlap_correction_deflates_the_t_stat(self):
        """The bug this guards against: an IC of 0.27 on 700 overlapping daily
        observations reads t=7.3 uncorrected and t<1 corrected."""
        naive = _t_stat(0.27, 700, horizon=1)
        corrected = _t_stat(0.27, 700, horizon=63)
        assert naive > 7.0
        assert abs(corrected) < 1.5
        assert abs(corrected) < abs(naive) / 5

    def test_perfect_predictor_scores_a_high_ic(self):
        rng = np.random.default_rng(4)
        anchor = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 600))), index=_days(600))
        fwd = forward_return(anchor, 0, 21)
        ic, n = information_coefficient(fwd, fwd)   # a factor equal to the answer
        assert ic > 0.99 and n > 400

    def test_noise_scores_near_zero(self):
        rng = np.random.default_rng(5)
        anchor = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 600))), index=_days(600))
        noise = pd.Series(rng.normal(0, 1, 600), index=_days(600))
        ic, _ = information_coefficient(noise, forward_return(anchor, 0, 21))
        assert abs(ic) < 0.2

    def test_forward_return_has_no_lookahead_at_the_tail(self):
        anchor = pd.Series(np.arange(1.0, 101.0), index=_days(100))
        fwd = forward_return(anchor, 0, 10)
        # The final 10 bars have no future, so they must be NaN, not fabricated.
        assert fwd.tail(10).isna().all()


class TestWeightShrinkage:
    def _result(self, name, ic, confidence, sign_agrees=True):
        return LeadLagResult(
            factor=name, best_lead=30, best_ic=ic, ic_curve={}, observations=500,
            t_stat=3.0, confidence=confidence, sign_agrees=sign_agrees,
        )

    def test_no_measurement_keeps_the_priors(self):
        priors = {"a": 0.6, "b": 0.4}
        assert shrink_weights(priors, {}) == pytest.approx(priors)

    def test_weights_always_sum_to_one(self):
        priors = {"a": 0.5, "b": 0.3, "c": 0.2}
        results = {"a": self._result("a", 0.3, 1.0), "b": self._result("b", 0.1, 0.5)}
        assert sum(shrink_weights(priors, results).values()) == pytest.approx(1.0)

    def test_wrong_sign_falls_back_to_the_prior(self):
        """A factor measuring backwards is evidence the factor is broken, not
        evidence the economics are. It must never be sign-flipped."""
        priors = {"a": 0.5, "b": 0.5}
        results = {"a": self._result("a", -0.5, 0.0, sign_agrees=False)}
        weights = shrink_weights(priors, results)
        assert weights["a"] == pytest.approx(0.5)

    def test_a_strong_factor_cannot_run_away_from_its_prior(self):
        priors = {"a": 0.1, "b": 0.9}
        results = {
            "a": self._result("a", 0.9, 1.0),   # hugely significant
            "b": self._result("b", 0.01, 0.0),
        }
        weights = shrink_weights(priors, results, max_multiple=2.0)
        # 0.1 prior, capped at 2x, then renormalised — must stay well under half.
        assert weights["a"] < 0.35


class TestLeadAlignment:
    def test_slower_factors_are_delayed_to_a_common_horizon(self):
        idx = _days(300)
        frame = pd.DataFrame(
            {"fast": np.arange(300.0), "slow": np.arange(300.0)}, index=idx
        )
        results = {
            "fast": LeadLagResult("fast", 10, 0.3, {}, 300, 3.0, 1.0, True),
            "slow": LeadLagResult("slow", 90, 0.3, {}, 300, 3.0, 1.0, True),
        }
        aligned = align_to_common_horizon(frame, results)
        # slow is read as it stood 80 days ago; fast is untouched.
        assert aligned["fast"].iloc[-1] == pytest.approx(299.0)
        assert aligned["slow"].iloc[-1] == pytest.approx(219.0)

    def test_insignificant_factors_are_left_alone(self):
        idx = _days(300)
        frame = pd.DataFrame({"a": np.arange(300.0), "b": np.arange(300.0)}, index=idx)
        results = {
            "a": LeadLagResult("a", 10, 0.3, {}, 300, 3.0, 1.0, True),
            "b": LeadLagResult("b", 90, -0.3, {}, 300, -3.0, 0.0, False),
        }
        aligned = align_to_common_horizon(frame, results)
        assert aligned["b"].iloc[-1] == pytest.approx(299.0)


# --------------------------------------------------------------------------- #
class TestCycleEngine:
    def test_panel_builds_for_both_domains(self, engine):
        for domain in ("crypto", "fx"):
            panel = engine.build_panel(domain)
            assert not panel.frame.empty
            assert len(panel) > 200
            assert not panel.anchor.empty

    def test_all_factor_values_are_bounded(self, engine):
        panel = engine.build_panel("crypto")
        values = panel.frame.dropna()
        assert values.to_numpy().min() >= -1.0001
        assert values.to_numpy().max() <= 1.0001

    def test_evaluate_returns_a_complete_state(self, engine):
        state = engine.evaluate("crypto")
        assert isinstance(state.phase, CyclePhase)
        assert -1.0 <= state.score <= 1.0
        assert 0.0 <= state.transition_probability <= 1.0
        assert 0.0 <= state.confidence <= 1.0
        assert state.narrative
        assert state.readings

    def test_weights_sum_to_one(self, engine):
        state = engine.evaluate("crypto")
        assert sum(r.weight for r in state.readings) == pytest.approx(1.0, abs=1e-6)

    def test_bias_never_contradicts_the_phase(self, engine):
        for domain in ("crypto", "fx"):
            state = engine.evaluate(domain)
            if state.phase.is_bullish:
                assert state.bias == "BULL"
            elif state.phase.is_bearish:
                assert state.bias == "BEAR"
            else:
                assert state.bias == "NEUTRAL"

    def test_validation_is_honest_on_simulated_data(self, engine):
        """Synthetic macro and synthetic price are independent, so the engine
        must report no edge rather than inventing one."""
        report = engine.validate("crypto")
        assert report.verdict in ("NO MEASURABLE EDGE", "INSUFFICIENT DATA")
        assert "simulated" in report.note.lower()

    def test_composite_stays_in_range(self, engine):
        state = engine.evaluate("fx")
        assert state.history is not None
        assert state.history["cycle"].dropna().between(-1.0, 1.0).all()


class TestStateMachine:
    def test_hysteresis_holds_a_phase_through_a_wobble(self, engine, cfg):
        """A single day back below the threshold must not flip the regime."""
        idx = _days(400)
        score = np.full(400, 0.40)
        score[300] = 0.10          # one-day dip, still above bull_exit
        phases = engine._phase_series(pd.Series(score, index=idx))
        assert phases.iloc[299] == phases.iloc[301]

    def test_a_sustained_move_does_change_the_phase(self, engine):
        idx = _days(400)
        score = np.concatenate([np.full(200, 0.5), np.full(200, -0.5)])
        phases = engine._phase_series(pd.Series(score, index=idx))
        assert phases.iloc[150] != phases.iloc[-1]
        assert phases.iloc[-1] in (CyclePhase.CONTRACTION.value, CyclePhase.EARLY_RECOVERY.value)

    def test_rising_from_deep_negative_is_early_recovery(self, engine):
        idx = _days(400)
        score = np.concatenate([np.full(150, -0.8), np.linspace(-0.8, -0.3, 250)])
        phases = engine._phase_series(pd.Series(score, index=idx))
        assert phases.iloc[-1] == CyclePhase.EARLY_RECOVERY.value

    def test_falling_from_high_is_late_expansion(self, engine):
        idx = _days(400)
        score = np.concatenate([np.full(150, 0.9), np.linspace(0.9, 0.3, 250)])
        phases = engine._phase_series(pd.Series(score, index=idx))
        assert phases.iloc[-1] == CyclePhase.LATE_EXPANSION.value
