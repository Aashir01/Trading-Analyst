"""Tests for the econometric modules.

These check the *formulas*, not the market. Each test states the expected value
from the mathematical definition so a refactor that changes behaviour fails
loudly rather than silently shifting every signal in the system.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mfie.config import (
    BehavioralParams,
    ESIParams,
    LiquidityParams,
    PPPParams,
    RIRDParams,
    RiskParams,
    TokenomicsParams,
)
from mfie.core.types import Direction, EconomicEvent, LiquidityRegime, Quote
from mfie.econ.behavioral import contrarian_multiplier, crowding_score, fear_greed_multiplier
from mfie.econ.liquidity import compute_gli
from mfie.econ.microstructure import compute_microstructure, liquidity_friction
from mfie.econ.rates import classify_macro_regime, real_rate, rird, yield_spread
from mfie.econ.risk import (
    conditional_value_at_risk,
    evaluate_risk_budget,
    kelly_fraction,
    max_drawdown,
    position_size,
    sortino_ratio,
    value_at_risk,
    volatility_target_scalar,
)
from mfie.econ.surprise import compute_esi, esi_differential, esi_multiplier
from mfie.econ.tokenomics import compute_tokenomics, nvt_ratio, token_velocity
from mfie.econ.valuation import compute_ppp_deviation, valuation_multiplier


def _dates(n: int, freq: str = "D") -> pd.DatetimeIndex:
    return pd.date_range(end="2026-08-01", periods=n, freq=freq, tz="UTC")


# --------------------------------------------------------------------------- #
# Global liquidity
# --------------------------------------------------------------------------- #
class TestGlobalLiquidity:
    def test_delta_matches_weighted_formula(self):
        # M2 grows 10% over the window, stablecoins 20%.
        m2 = pd.Series([100.0, 110.0], index=_dates(2, "100D"))
        sc = pd.Series([100.0, 120.0], index=_dates(2, "100D"))
        params = LiquidityParams(m2_weight=0.6, stablecoin_weight=0.4, lookback_days=90)

        gli = compute_gli(m2, sc, params)

        assert gli.m2_change == pytest.approx(0.10)
        assert gli.stablecoin_change == pytest.approx(0.20)
        assert gli.delta == pytest.approx(0.6 * 0.10 + 0.4 * 0.20)

    def test_weights_are_normalised(self):
        m2 = pd.Series([100.0, 110.0], index=_dates(2, "100D"))
        sc = pd.Series([100.0, 110.0], index=_dates(2, "100D"))
        # Weights that do not sum to 1 must not inflate the index.
        gli = compute_gli(m2, sc, LiquidityParams(m2_weight=3.0, stablecoin_weight=1.0))
        assert gli.delta == pytest.approx(0.10)

    def test_contraction_penalises_momentum(self):
        m2 = pd.Series([100.0, 95.0], index=_dates(2, "100D"))
        sc = pd.Series([100.0, 92.0], index=_dates(2, "100D"))
        gli = compute_gli(m2, sc, LiquidityParams())

        assert gli.regime is LiquidityRegime.CONTRACTION
        assert gli.is_contracting
        assert gli.momentum_multiplier < 1.0

    def test_expansion_leaves_momentum_untouched(self):
        m2 = pd.Series([100.0, 105.0], index=_dates(2, "100D"))
        sc = pd.Series([100.0, 106.0], index=_dates(2, "100D"))
        gli = compute_gli(m2, sc, LiquidityParams())

        assert gli.regime is LiquidityRegime.EXPANSION
        assert gli.momentum_multiplier == pytest.approx(1.0)

    def test_empty_series_does_not_raise(self):
        gli = compute_gli(pd.Series(dtype=float), pd.Series(dtype=float))
        assert gli.delta == 0.0


# --------------------------------------------------------------------------- #
# Rates
# --------------------------------------------------------------------------- #
class TestRates:
    def test_real_rate_is_nominal_minus_inflation(self):
        assert real_rate(0.0450, 0.0290) == pytest.approx(0.0160)

    def test_real_rate_returns_none_when_incomplete(self):
        assert real_rate(None, 0.02) is None
        assert real_rate(0.02, None) is None

    def test_rird_decomposition(self):
        rates = {"EUR": 0.0325, "USD": 0.0450}
        cpi = {"EUR": 0.0230, "USD": 0.0290}
        # (I_A - I_B) - (pi_A - pi_B)
        expected = (0.0325 - 0.0450) - (0.0230 - 0.0290)
        assert rird(rates, cpi, "EUR", "USD") == pytest.approx(expected)

    def test_rird_is_antisymmetric(self):
        rates = {"EUR": 0.0325, "USD": 0.0450}
        cpi = {"EUR": 0.0230, "USD": 0.0290}
        assert rird(rates, cpi, "EUR", "USD") == pytest.approx(
            -rird(rates, cpi, "USD", "EUR")
        )

    def test_inverted_curve_flags_contraction(self):
        regime, spread = classify_macro_regime({"USD": 0.0415}, {"USD": 0.0445})
        assert regime.value == "contraction"
        assert spread == pytest.approx(-0.0030)

    def test_steep_curve_flags_expansion(self):
        regime, spread = classify_macro_regime({"USD": 0.045}, {"USD": 0.030})
        assert regime.value == "expansion"
        assert spread == pytest.approx(0.015)

    def test_yield_spread_missing_leg(self):
        assert yield_spread({"USD": 0.04}, {}, "USD") is None


class TestRIRDAlignment:
    def _view(self, differential: float):
        from mfie.core.types import AssetClass, Instrument
        from mfie.econ.rates import compute_rate_view

        inst = Instrument("EURUSD", AssetClass.FOREX, "EUR", "USD")
        # Construct rates so that RIRD == differential exactly.
        return compute_rate_view(
            inst,
            policy_rates={"EUR": differential, "USD": 0.0},
            inflation={"EUR": 0.0, "USD": 0.0},
            yield_10y={"USD": 0.04},
            yield_2y={"USD": 0.03},
        )

    def test_aligned_long_gets_a_bonus(self):
        state, multiplier = self._view(0.02).alignment(Direction.LONG, RIRDParams())
        assert state == "aligned"
        assert multiplier > 1.0

    def test_opposed_long_is_penalised(self):
        state, multiplier = self._view(-0.02).alignment(Direction.LONG, RIRDParams())
        assert state == "opposed"
        assert multiplier < 1.0

    def test_hard_opposition_is_flagged(self):
        state, _ = self._view(-0.05).alignment(Direction.LONG, RIRDParams())
        assert state == "hard_opposed"

    def test_small_differential_is_neutral(self):
        state, multiplier = self._view(0.002).alignment(Direction.LONG, RIRDParams())
        assert state == "neutral"
        assert multiplier == 1.0


# --------------------------------------------------------------------------- #
# PPP / valuation
# --------------------------------------------------------------------------- #
class TestValuation:
    def test_z_score_of_a_known_series(self):
        values = list(np.arange(100.0, 130.0))  # last point is the maximum
        series = pd.Series(values, index=_dates(len(values)))
        view = compute_ppp_deviation("USD", series, PPPParams(lookback_periods=len(values)))
        assert view.z_score > 1.0

    def test_flat_series_is_fair_value(self):
        series = pd.Series([100.0] * 40, index=_dates(40))
        view = compute_ppp_deviation("USD", series)
        assert view.z_score == 0.0
        assert view.state == "fair"

    def test_overvalued_long_is_penalised(self):
        state, multiplier, block = valuation_multiplier(2.2, Direction.LONG, PPPParams())
        assert state == "overextended"
        assert multiplier < 1.0
        assert not block

    def test_extreme_valuation_blocks(self):
        _, _, block = valuation_multiplier(3.0, Direction.LONG, PPPParams())
        assert block

    def test_short_into_overvaluation_is_not_penalised(self):
        state, multiplier, block = valuation_multiplier(2.2, Direction.SHORT, PPPParams())
        assert multiplier == pytest.approx(1.0)
        assert not block


# --------------------------------------------------------------------------- #
# Economic surprise
# --------------------------------------------------------------------------- #
class TestSurprise:
    def _event(self, currency: str, days_ago: float, actual: float, forecast: float):
        from datetime import datetime, timedelta, timezone

        return EconomicEvent(
            ts=datetime.now(timezone.utc) - timedelta(days=days_ago),
            country=currency[:2],
            currency=currency,
            name="CPI y/y",
            impact="high",
            actual=actual,
            forecast=forecast,
        )

    def test_beats_produce_positive_score(self):
        events = [self._event("USD", d, 0.033, 0.030) for d in (1, 5, 10, 20)]
        esi = compute_esi(events, ESIParams())
        assert esi["USD"] > 0

    def test_misses_produce_negative_score(self):
        events = [self._event("EUR", d, 0.027, 0.030) for d in (1, 5, 10, 20)]
        esi = compute_esi(events, ESIParams())
        assert esi["EUR"] < 0

    def test_recent_data_outweighs_stale_data(self):
        recent = compute_esi([self._event("USD", 1, 0.033, 0.030)] * 4, ESIParams())["USD"]
        stale = compute_esi([self._event("USD", 80, 0.033, 0.030)] * 4, ESIParams())["USD"]
        assert abs(recent) > abs(stale)

    def test_unreleased_events_are_ignored(self):
        from datetime import datetime, timedelta, timezone

        future = EconomicEvent(
            ts=datetime.now(timezone.utc) + timedelta(days=2),
            country="US", currency="USD", name="CPI y/y",
            impact="high", actual=None, forecast=0.03,
        )
        assert compute_esi([future], ESIParams()) == {}

    def test_differential_and_multiplier(self):
        esi = {"EUR": 2.0, "USD": 0.0}
        differential = esi_differential(esi, "EUR", "USD")
        assert differential == pytest.approx(2.0)
        state, multiplier = esi_multiplier(differential, Direction.LONG, ESIParams())
        assert state == "aligned"
        assert multiplier > 1.0


# --------------------------------------------------------------------------- #
# Tokenomics
# --------------------------------------------------------------------------- #
class TestTokenomics:
    def test_velocity_is_volume_over_cap(self):
        idx = _dates(30)
        volume = pd.Series([50.0] * 30, index=idx)
        cap = pd.Series([500.0] * 30, index=idx)
        velocity = token_velocity(volume, cap)
        assert velocity.iloc[-1] == pytest.approx(0.1)

    def test_nvt_is_the_reciprocal_of_velocity(self):
        idx = _dates(30)
        volume = pd.Series([50.0] * 30, index=idx)
        cap = pd.Series([500.0] * 30, index=idx)
        assert nvt_ratio(cap, volume).iloc[-1] == pytest.approx(10.0)

    def test_rising_price_with_collapsing_velocity_flags_divergence(self):
        idx = _dates(120)
        # Market cap doubles while on-chain volume decays: classic hoarding.
        cap = pd.Series(np.linspace(100.0, 200.0, 120), index=idx)
        volume = pd.Series(np.concatenate([np.full(100, 10.0), np.full(20, 1.0)]), index=idx)

        view = compute_tokenomics("BTCUSDT", volume, cap, TokenomicsParams())
        assert view.price_change > 0
        assert view.divergence
        assert view.warning is not None

    def test_healthy_network_has_no_warning(self):
        idx = _dates(120)
        rng = np.random.default_rng(3)
        cap = pd.Series(np.linspace(100.0, 120.0, 120), index=idx)
        volume = pd.Series(np.linspace(10.0, 12.0, 120) + rng.normal(0, 0.1, 120), index=idx)
        view = compute_tokenomics("BTCUSDT", volume, cap, TokenomicsParams())
        assert view.warning is None


# --------------------------------------------------------------------------- #
# Microstructure
# --------------------------------------------------------------------------- #
class TestMicrostructure:
    def test_friction_is_spread_over_atr(self):
        quote = Quote("BTCUSDT", bid=99.0, ask=101.0)
        assert liquidity_friction(quote, 20.0) == pytest.approx(0.1)

    def test_zero_atr_is_safe(self):
        quote = Quote("BTCUSDT", bid=99.0, ask=101.0)
        assert liquidity_friction(quote, 0.0) == 0.0

    def test_wide_spread_halts_execution(self):
        quote = Quote("BTCUSDT", bid=90.0, ask=110.0)  # spread 20 vs ATR 20 => LF 1.0
        view = compute_microstructure(quote, 20.0)
        assert view.should_halt
        assert view.size_multiplier == 0.0

    def test_thin_book_widens_stop_and_shrinks_size(self):
        quote = Quote("BTCUSDT", bid=99.0, ask=101.0)  # LF 0.2, between warn and halt
        view = compute_microstructure(quote, 10.0)
        assert view.state == "thin"
        assert view.stop_multiplier > 1.0
        # Risk must stay constant: size shrinks by the same factor the stop grows.
        assert view.size_multiplier == pytest.approx(1.0 / view.stop_multiplier, rel=1e-6)

    def test_normal_book_is_untouched(self):
        quote = Quote("BTCUSDT", bid=99.99, ask=100.01)
        view = compute_microstructure(quote, 10.0)
        assert view.state == "normal"
        assert view.stop_multiplier == 1.0


# --------------------------------------------------------------------------- #
# Behavioural
# --------------------------------------------------------------------------- #
class TestBehavioral:
    def test_balanced_positioning_is_unpenalised(self):
        assert crowding_score(0.5) == 0.0
        _, multiplier = contrarian_multiplier(0.5, Direction.LONG, BehavioralParams())
        assert multiplier == pytest.approx(1.0)

    def test_crowded_long_is_penalised(self):
        _, multiplier = contrarian_multiplier(0.90, Direction.LONG, BehavioralParams())
        # C_t = 1 - |(0.9-0.5)/0.5|^2 = 1 - 0.64 = 0.36
        assert multiplier == pytest.approx(0.36, abs=1e-6)

    def test_trading_against_the_crowd_is_not_penalised(self):
        _, multiplier = contrarian_multiplier(0.90, Direction.SHORT, BehavioralParams())
        assert multiplier >= 1.0

    def test_penalty_is_symmetric(self):
        _, long_side = contrarian_multiplier(0.90, Direction.LONG, BehavioralParams())
        _, short_side = contrarian_multiplier(0.10, Direction.SHORT, BehavioralParams())
        assert long_side == pytest.approx(short_side)

    def test_extreme_greed_penalises_longs_only(self):
        _, long_mult = fear_greed_multiplier(90.0, Direction.LONG, BehavioralParams())
        _, short_mult = fear_greed_multiplier(90.0, Direction.SHORT, BehavioralParams())
        assert long_mult < 1.0
        assert short_mult >= 1.0


# --------------------------------------------------------------------------- #
# Risk
# --------------------------------------------------------------------------- #
class TestRisk:
    def test_cvar_is_at_least_var(self):
        rng = np.random.default_rng(11)
        returns = pd.Series(rng.normal(0.0005, 0.02, 1000))
        var = value_at_risk(returns, 0.95)
        cvar = conditional_value_at_risk(returns, 0.95)
        # Expected shortfall averages the tail beyond VaR, so it is never smaller.
        assert cvar >= var > 0

    def test_cvar_on_a_known_tail(self):
        # 100 observations: 95 at -1%, 5 at -10%. CVaR95 must be the tail mean.
        returns = pd.Series([-0.01] * 95 + [-0.10] * 5)
        assert conditional_value_at_risk(returns, 0.95) == pytest.approx(0.10, abs=0.005)

    def test_parametric_and_historical_agree_on_normal_data(self):
        rng = np.random.default_rng(5)
        returns = pd.Series(rng.normal(0.0, 0.01, 5000))
        historical = conditional_value_at_risk(returns, 0.95, "historical")
        parametric = conditional_value_at_risk(returns, 0.95, "parametric")
        assert historical == pytest.approx(parametric, rel=0.15)

    def test_sortino_ignores_upside_volatility(self):
        # Same mean, but one series has all its variance on the upside.
        steady = pd.Series([0.001] * 200)
        spiky = pd.Series([0.0] * 190 + [0.02] * 10)
        assert sortino_ratio(spiky) >= sortino_ratio(steady) or np.isinf(sortino_ratio(steady))

    def test_max_drawdown_of_a_known_curve(self):
        equity = pd.Series([100.0, 120.0, 90.0, 110.0])
        assert max_drawdown(equity) == pytest.approx(0.25)

    def test_kelly_formula(self):
        # f* = p - (1-p)/b, capped
        assert kelly_fraction(0.6, 2.0, cap=1.0) == pytest.approx(0.6 - 0.4 / 2.0)

    def test_kelly_is_zero_without_edge(self):
        assert kelly_fraction(0.3, 1.0) == 0.0

    def test_kelly_respects_the_cap(self):
        assert kelly_fraction(0.9, 5.0, cap=0.25) == pytest.approx(0.25)

    def test_position_size_risks_exactly_the_requested_fraction(self):
        units, notional = position_size(equity=10_000, risk_fraction=0.01,
                                        entry=100.0, stop=95.0)
        # 1% of 10k = 100 risk; stop distance 5 => 20 units.
        assert units == pytest.approx(20.0)
        assert notional == pytest.approx(2000.0)
        assert units * abs(100.0 - 95.0) == pytest.approx(100.0)

    def test_position_size_is_zero_without_a_stop(self):
        units, notional = position_size(10_000, 0.01, 100.0, 100.0)
        assert units == 0.0 and notional == 0.0

    def test_volatility_scalar_reduces_size_in_high_vol(self):
        assert volatility_target_scalar(0.60, 0.15) < 1.0
        assert volatility_target_scalar(0.05, 0.15) > 1.0

    def test_cvar_breach_scales_exposure_down(self):
        losses = pd.Series([-0.05] * 30 + [0.001] * 220)
        budget = evaluate_risk_budget(losses, RiskParams(cvar_budget=0.01))
        assert budget.breached
        assert budget.scaler < 1.0
