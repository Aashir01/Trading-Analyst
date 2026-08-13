"""Tests for the capital allocation layer.

These pin the arithmetic that decides position sizes, so they are written as
statements about behaviour rather than about implementation: costs must rise
when the stop tightens, a hedge must measure as less risk than either leg, an
unproven strategy must size smaller than a proven one with the same sample
mean, and a book of clones must not be allowed to pass as a diversified one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mfie.backtest.significance import (
    assess_significance,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    sharpe_standard_error,
)
from mfie.config import Params
from mfie.core.types import Direction, Quote, RawSignal, ScoredSignal
from mfie.core.universe import get_instrument
from mfie.portfolio.allocator import PortfolioAllocator, apply_plan
from mfie.portfolio.calibration import (
    ConfidenceCalibrator,
    TradeRecord,
    prior_hit_rate,
)
from mfie.portfolio.construction import (
    cluster_by_correlation,
    diversification_ratio,
    effective_risk,
    marginal_risk_contributions,
    returns_matrix,
    signed_correlation,
)
from mfie.portfolio.costs import breakeven_hit_rate, estimate_costs, expected_holding_bars
from mfie.portfolio.edge import evaluate_edge, expected_r, kelly_with_uncertainty

BTC = get_instrument("BTCUSDT")
ETH = get_instrument("ETHUSDT")
EURUSD = get_instrument("EURUSD")


def make_signal(
    instrument=BTC,
    direction: Direction = Direction.LONG,
    strategy: str = "trend_following",
    entry: float = 60_000.0,
    stop_distance: float = 1_200.0,
    reward_risk: float = 2.0,
) -> RawSignal:
    sign = direction.sign or 1
    stop = entry - sign * stop_distance
    return RawSignal(
        instrument=instrument,
        direction=direction,
        strategy=strategy,
        strength=0.7,
        entry=entry,
        stop=stop,
        take_profit=entry + sign * stop_distance * reward_risk,
    )


def scored(signal: RawSignal, risk: float = 0.01, confidence: float = 0.7) -> ScoredSignal:
    return ScoredSignal(
        raw=signal,
        confidence=confidence,
        risk_fraction=risk,
        units=1.0,
        adjusted_stop=signal.stop,
    )


# --------------------------------------------------------------------------- #
# Costs
# --------------------------------------------------------------------------- #
class TestCosts:
    def test_costs_are_scaled_by_the_stop_not_the_price(self):
        """The same absolute cost is a bigger problem behind a tighter stop.

        This is the whole reason costs are expressed in R. Halving the stop
        must roughly double the cost in R, because the denominator halved
        while the spread and commission did not move.
        """
        quote = Quote(symbol="BTCUSDT", bid=59_994.0, ask=60_006.0)
        wide = estimate_costs(make_signal(stop_distance=1_200.0), quote=quote, atr=600.0)
        tight = estimate_costs(make_signal(stop_distance=600.0), quote=quote, atr=600.0)

        assert tight.total_r > wide.total_r
        # Friction is identical in price terms; only the divisor changed.
        assert wide.friction == pytest.approx(tight.friction, rel=1e-9)
        assert tight.friction_r == pytest.approx(2.0 * wide.friction_r, rel=1e-6)

    def test_short_perp_receives_positive_funding(self):
        """Carry is signed. A short in a positive-funding market is being paid."""
        long_costs = estimate_costs(
            make_signal(direction=Direction.LONG), atr=600.0, funding_rate=0.0005
        )
        short_costs = estimate_costs(
            make_signal(direction=Direction.SHORT), atr=600.0, funding_rate=0.0005
        )
        assert long_costs.carry > 0          # the long pays
        assert short_costs.carry < 0         # the short collects
        assert short_costs.total_r < long_costs.total_r

    def test_long_the_higher_yielder_earns_the_differential(self):
        long_costs = estimate_costs(
            make_signal(instrument=EURUSD, direction=Direction.LONG, entry=1.08,
                        stop_distance=0.004),
            atr=0.002, rate_differential=0.02, timeframe="1d",
        )
        short_costs = estimate_costs(
            make_signal(instrument=EURUSD, direction=Direction.SHORT, entry=1.08,
                        stop_distance=0.004),
            atr=0.002, rate_differential=0.02, timeframe="1d",
        )
        assert long_costs.carry < 0          # long the high-yielder is paid
        assert short_costs.carry > 0

    def test_holding_time_is_quadratic_in_the_stop_distance(self):
        """Diffusion: doubling the distance quadruples the expected time."""
        near = expected_holding_bars(stop_distance=100.0, atr=100.0)
        far = expected_holding_bars(stop_distance=200.0, atr=100.0)
        assert near == pytest.approx(1.0)
        assert far == pytest.approx(4.0)

    def test_holding_time_is_capped(self):
        params = Params().costs
        absurd = expected_holding_bars(stop_distance=10_000.0, atr=1.0, params=params)
        assert absurd == params.max_holding_bars

    def test_breakeven_rises_with_cost_and_falls_with_reward(self):
        free = breakeven_hit_rate(2.0, 0.0)
        costly = breakeven_hit_rate(2.0, 0.25)
        richer = breakeven_hit_rate(4.0, 0.0)
        assert free == pytest.approx(1 / 3)
        assert costly > free
        assert richer < free

    def test_a_series_of_funding_rates_does_not_crash_the_model(self):
        """Providers serve funding as history; the model must cope."""
        series = pd.Series([0.0001, 0.0002, 0.0005])
        costs = estimate_costs(make_signal(), atr=600.0, funding_rate=series)
        assert np.isfinite(costs.total_r)
        assert costs.carry > 0


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
class TestCalibration:
    def test_unfitted_calibrator_returns_the_prior(self):
        """Cold start must reproduce the engine's original heuristic exactly."""
        calibrator = ConfidenceCalibrator()
        estimate = calibrator.estimate(0.6, "trend_following", "ranging")
        assert estimate.mean == pytest.approx(prior_hit_rate(0.6))
        assert estimate.observations == 0
        assert estimate.credibility == 0.0
        assert estimate.source == "prior"

    def test_shrinkage_follows_the_credibility_formula(self):
        """z = n/(n+kappa) is not an approximation here — it must hold exactly."""
        params = Params().calibration
        calibrator = ConfidenceCalibrator(params)
        n, wins = 60, 45
        calibrator.fit(
            [
                TradeRecord("trend_following", won=i < wins, confidence=0.5,
                            regime="trending_low_vol")
                for i in range(n)
            ]
        )
        estimate = calibrator.estimate(0.5, "trend_following", "trending_low_vol")
        assert estimate.credibility == pytest.approx(n / (n + params.prior_strength))
        # The blended estimate must sit strictly between the sample and the prior.
        assert estimate.prior < estimate.mean < wins / n

    def test_a_thin_sample_sizes_smaller_than_a_thick_one(self):
        """Same sample mean, different evidence: the thin cell must bet less.

        This is the property a point estimate cannot express, and the reason
        sizing reads the posterior's lower tail.
        """
        def build(n: int) -> ConfidenceCalibrator:
            calibrator = ConfidenceCalibrator()
            return calibrator.fit(
                [
                    TradeRecord("trend_following", won=i % 2 == 0, confidence=0.6,
                                regime="ranging")
                    for i in range(n)
                ]
            )

        thin = build(12).estimate(0.6, "trend_following", "ranging")
        thick = build(600).estimate(0.6, "trend_following", "ranging")
        assert thin.mean == pytest.approx(thick.mean, abs=0.05)
        assert thin.lower < thick.lower
        assert thin.spread > thick.spread

    def test_thin_cells_inherit_their_parent(self):
        """A cell below the observation floor must not invent its own rate."""
        calibrator = ConfidenceCalibrator()
        calibrator.fit(
            [TradeRecord("trend_following", won=True, confidence=0.6, regime="trending_low_vol")
             for _ in range(50)]
            + [TradeRecord("trend_following", won=False, confidence=0.6, regime="crash")
               for _ in range(3)]
        )
        crash = calibrator.estimate(0.6, "trend_following", "crash")
        assert crash.source == "strategy"       # not "strategy+regime"
        assert crash.mean > 0.5                 # inherited the strategy's optimism

    def test_the_posterior_lower_bound_never_exceeds_the_mean(self):
        calibrator = ConfidenceCalibrator()
        calibrator.fit(
            [TradeRecord("grid", won=i % 3 == 0, confidence=0.4, regime="ranging")
             for i in range(90)]
        )
        for confidence in (0.1, 0.4, 0.7, 0.95):
            estimate = calibrator.estimate(confidence, "grid", "ranging")
            assert 0.0 <= estimate.lower <= estimate.mean <= 1.0

    def test_reliability_table_reports_what_actually_happened(self):
        calibrator = ConfidenceCalibrator()
        # High confidence loses, low confidence wins — a perfectly inverted score.
        calibrator.fit(
            [TradeRecord("trend_following", won=False, confidence=0.9) for _ in range(40)]
            + [TradeRecord("trend_following", won=True, confidence=0.1) for _ in range(40)]
        )
        table = calibrator.reliability_table().set_index("confidence_range")
        assert table.loc["80%-100%", "realised"] == 0.0
        assert table.loc["0%-20%", "realised"] == 1.0
        # An inverted score must not be credited with skill.
        assert calibrator.skill_score() < 0

    def test_online_update_matches_batch_fit(self):
        records = [
            TradeRecord("trend_following", won=i % 3 != 0, confidence=0.5, regime="ranging")
            for i in range(45)
        ]
        batch = ConfidenceCalibrator().fit(records)
        online = ConfidenceCalibrator()
        for record in records:
            online.update(record)
        assert batch.estimate(0.5, "trend_following", "ranging").mean == pytest.approx(
            online.estimate(0.5, "trend_following", "ranging").mean
        )


# --------------------------------------------------------------------------- #
# Expected value
# --------------------------------------------------------------------------- #
class TestEdge:
    def test_expected_value_matches_the_closed_form(self):
        assert expected_r(0.5, 2.0, 0.0) == pytest.approx(0.5 * 2.0 - 0.5 * 1.0)
        assert expected_r(0.5, 2.0, 0.1) == pytest.approx(0.5 * 1.9 - 0.5 * 1.1)

    def test_expected_value_is_zero_at_the_breakeven_hit_rate(self):
        """The two formulas must agree, or the gate rejects the wrong trades."""
        for reward_risk in (1.0, 2.0, 3.5):
            for cost in (0.0, 0.05, 0.3):
                p_star = breakeven_hit_rate(reward_risk, cost)
                assert expected_r(p_star, reward_risk, cost) == pytest.approx(0.0, abs=1e-12)

    def test_a_cheap_trade_beats_an_identical_expensive_one(self):
        calibrator = ConfidenceCalibrator()
        hit_rate = calibrator.estimate(0.7)
        quote = Quote(symbol="BTCUSDT", bid=59_994.0, ask=60_006.0)

        wide = evaluate_edge(hit_rate, 2.0,
                             estimate_costs(make_signal(stop_distance=1_200.0),
                                            quote=quote, atr=600.0))
        tight = evaluate_edge(hit_rate, 2.0,
                              estimate_costs(make_signal(stop_distance=90.0),
                                             quote=quote, atr=600.0))
        assert wide.tradable
        assert not tight.tradable
        assert tight.breakeven_hit_rate > wide.breakeven_hit_rate
        assert tight.cost_drag > wide.cost_drag

    def test_kelly_uses_the_lower_bound_so_uncertainty_shrinks_the_bet(self):
        # A 1R payoff keeps the result away from the quarter-Kelly cap, so the
        # comparison measures the uncertainty adjustment rather than the cap.
        def kelly_for(n: int) -> float:
            calibrator = ConfidenceCalibrator().fit(
                [TradeRecord("trend_following", won=i % 5 < 3, confidence=0.6,
                             regime="ranging")
                 for i in range(n)]
            )
            estimate = calibrator.estimate(0.6, "trend_following", "ranging")
            return kelly_with_uncertainty(estimate, 1.0, 0.02)

        thick, thin = kelly_for(600), kelly_for(15)
        assert 0.0 < thin < thick < 0.25

    def test_kelly_is_zero_when_the_edge_is_not_demonstrable(self):
        calibrator = ConfidenceCalibrator()
        hopeless = calibrator.estimate(0.0)          # prior floor
        assert kelly_with_uncertainty(hopeless, 0.5, 0.3) == 0.0

    def test_kelly_respects_the_configured_cap(self):
        calibrator = ConfidenceCalibrator().fit(
            [TradeRecord("trend_following", won=True, confidence=0.9) for _ in range(2_000)]
        )
        estimate = calibrator.estimate(0.9, "trend_following")
        assert kelly_with_uncertainty(estimate, 5.0, 0.0, cap=0.25) <= 0.25

    def test_a_paid_carry_widens_the_edge(self):
        calibrator = ConfidenceCalibrator()
        hit_rate = calibrator.estimate(0.6)
        paying = estimate_costs(make_signal(direction=Direction.LONG), atr=600.0,
                                funding_rate=0.001)
        collecting = estimate_costs(make_signal(direction=Direction.SHORT), atr=600.0,
                                    funding_rate=0.001)
        assert (
            evaluate_edge(hit_rate, 2.0, collecting).expected_r
            > evaluate_edge(hit_rate, 2.0, paying).expected_r
        )


# --------------------------------------------------------------------------- #
# Portfolio construction
# --------------------------------------------------------------------------- #
@pytest.fixture
def correlated_returns() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    index = pd.date_range("2024-01-01", periods=400, freq="h")
    driver = pd.Series(rng.normal(0, 0.01, 400), index=index)
    return pd.DataFrame(
        {
            "BTCUSDT": driver,
            "ETHUSDT": driver * 0.98 + pd.Series(rng.normal(0, 0.001, 400), index=index),
            "EURUSD": pd.Series(rng.normal(0, 0.002, 400), index=index),
        }
    )


class TestConstruction:
    def test_two_longs_in_correlated_assets_are_one_bet(self, correlated_returns):
        positions = {"BTCUSDT:t": ("BTCUSDT", 1), "ETHUSDT:t": ("ETHUSDT", 1)}
        view = signed_correlation(correlated_returns, positions)
        risks = pd.Series({"BTCUSDT:t": 0.01, "ETHUSDT:t": 0.01})

        # Additive heat says 2%. The real number is close to it, because these
        # are the same trade — which is the point.
        assert effective_risk(risks, view.matrix) == pytest.approx(0.02, rel=0.05)
        assert diversification_ratio(risks, view.matrix) == pytest.approx(1.0, rel=0.05)
        assert len(set(cluster_by_correlation(view.matrix).values())) == 1

    def test_a_hedge_measures_as_far_less_risk_than_either_leg(self, correlated_returns):
        positions = {"BTCUSDT:t": ("BTCUSDT", 1), "ETHUSDT:t": ("ETHUSDT", -1)}
        view = signed_correlation(correlated_returns, positions)
        risks = pd.Series({"BTCUSDT:t": 0.01, "ETHUSDT:t": 0.01})
        assert effective_risk(risks, view.matrix) < 0.005

    def test_uncorrelated_positions_diversify(self, correlated_returns):
        positions = {"BTCUSDT:t": ("BTCUSDT", 1), "EURUSD:t": ("EURUSD", 1)}
        view = signed_correlation(correlated_returns, positions)
        risks = pd.Series({"BTCUSDT:t": 0.01, "EURUSD:t": 0.01})
        # Two independent 1% risks are sqrt(2) x 1%, not 2%.
        assert effective_risk(risks, view.matrix) == pytest.approx(0.01 * np.sqrt(2), rel=0.1)
        assert diversification_ratio(risks, view.matrix) > 1.3

    def test_effective_risk_never_exceeds_additive_heat(self, correlated_returns):
        """A correlation matrix is positive semi-definite with unit diagonal, so
        this must hold for every book the engine can construct."""
        rng = np.random.default_rng(5)
        for _ in range(25):
            directions = {
                f"{symbol}:t": (symbol, int(rng.choice([-1, 1])))
                for symbol in correlated_returns.columns
            }
            view = signed_correlation(correlated_returns, directions)
            risks = pd.Series(
                {key: float(rng.uniform(0.001, 0.02)) for key in directions}
            )
            assert effective_risk(risks, view.matrix) <= risks.sum() + 1e-12

    def test_two_strategies_on_one_instrument_are_perfectly_correlated(self):
        positions = {"BTCUSDT:trend": ("BTCUSDT", 1), "BTCUSDT:breakout": ("BTCUSDT", 1)}
        view = signed_correlation(pd.DataFrame(), positions)
        # Correlations are clamped just inside +/-1 to keep the matrix positive
        # definite rather than merely semi-definite, so this is 0.999, not 1.0.
        assert view.matrix.loc["BTCUSDT:trend", "BTCUSDT:breakout"] == pytest.approx(1.0, abs=2e-3)
        assert len(set(cluster_by_correlation(view.matrix).values())) == 1

    def test_missing_history_falls_back_to_a_positive_prior(self):
        """Unknown correlation must never be treated as zero."""
        positions = {"BTCUSDT:t": ("BTCUSDT", 1), "EURUSD:t": ("EURUSD", 1)}
        view = signed_correlation(pd.DataFrame(), positions, default_correlation=0.35)
        assert view.matrix.loc["BTCUSDT:t", "EURUSD:t"] == pytest.approx(0.35)
        assert view.estimated_pairs == 1
        assert view.coverage == 0.0

    def test_risk_contributions_sum_to_total_book_risk(self, correlated_returns):
        """Euler's theorem — the decomposition is exact, not approximate."""
        positions = {
            "BTCUSDT:t": ("BTCUSDT", 1),
            "ETHUSDT:t": ("ETHUSDT", 1),
            "EURUSD:t": ("EURUSD", -1),
        }
        view = signed_correlation(correlated_returns, positions)
        risks = pd.Series({"BTCUSDT:t": 0.01, "ETHUSDT:t": 0.008, "EURUSD:t": 0.012})
        contributions = marginal_risk_contributions(risks, view.matrix)
        assert contributions.sum() == pytest.approx(effective_risk(risks, view.matrix))

    def test_clustering_is_transitive(self):
        """A correlates with B, B with C, A weakly with C — still one risk group."""
        keys = ["a", "b", "c"]
        matrix = pd.DataFrame(
            [[1.0, 0.8, 0.4], [0.8, 1.0, 0.8], [0.4, 0.8, 1.0]],
            index=keys, columns=keys,
        )
        assert len(set(cluster_by_correlation(matrix, threshold=0.6).values())) == 1

    def test_returns_matrix_ignores_unusable_frames(self):
        frames = {
            "BTCUSDT": pd.DataFrame(
                {"close": np.linspace(100, 120, 50)},
                index=pd.date_range("2024-01-01", periods=50, freq="h"),
            ),
            "ETHUSDT": pd.DataFrame(),
            "EURUSD": pd.DataFrame({"close": [1.0, 1.1]}),
        }
        matrix = returns_matrix(frames)
        assert list(matrix.columns) == ["BTCUSDT"]


# --------------------------------------------------------------------------- #
# Allocation
# --------------------------------------------------------------------------- #
class TestAllocator:
    def _candidates(self, specs):
        calibrator = ConfidenceCalibrator()
        out = []
        for instrument, strategy, direction, risk in specs:
            entry = 60_000.0 if instrument.is_crypto else 1.08
            distance = entry * 0.02
            raw = make_signal(instrument, direction, strategy, entry, distance)
            signal = scored(raw, risk=risk)
            costs = estimate_costs(raw, atr=distance / 2)
            out.append((signal, evaluate_edge(calibrator.estimate(0.7), 2.0, costs)))
        return out

    def test_redundant_ideas_in_a_cluster_are_discounted(self, correlated_returns):
        candidates = self._candidates(
            [
                (BTC, "trend_following", Direction.LONG, 0.01),
                (ETH, "trend_following", Direction.LONG, 0.01),
                (EURUSD, "trend_following", Direction.LONG, 0.01),
            ]
        )
        plan = PortfolioAllocator().allocate(candidates, correlated_returns)
        by_symbol = {a.symbol: a for a in plan.allocations}

        # BTC and ETH are one cluster; the second one is cut. EURUSD is
        # independent and keeps its full size.
        assert by_symbol["BTCUSDT"].cluster == by_symbol["ETHUSDT"].cluster
        assert by_symbol["EURUSD"].cluster != by_symbol["BTCUSDT"].cluster
        assert by_symbol["ETHUSDT"].risk_fraction < by_symbol["BTCUSDT"].risk_fraction
        assert by_symbol["EURUSD"].risk_fraction == pytest.approx(0.01)

    def test_a_book_of_clones_is_cut_to_the_diversified_budget(self, correlated_returns):
        params = Params()
        params.portfolio.max_effective_risk = 0.015
        params.portfolio.redundancy_decay = 0.0     # isolate the budget step
        params.portfolio.max_cluster_risk = 1.0

        candidates = self._candidates(
            [(BTC, "trend_following", Direction.LONG, 0.01),
             (ETH, "trend_following", Direction.LONG, 0.01)]
        )
        plan = PortfolioAllocator(params).allocate(candidates, correlated_returns)
        assert plan.scale_applied < 1.0
        assert plan.effective_risk == pytest.approx(0.015, rel=1e-6)

    def test_a_diversified_book_is_left_alone(self, correlated_returns):
        params = Params()
        params.portfolio.max_effective_risk = 0.05
        candidates = self._candidates(
            [(BTC, "trend_following", Direction.LONG, 0.01),
             (EURUSD, "trend_following", Direction.LONG, 0.01)]
        )
        plan = PortfolioAllocator(params).allocate(candidates, correlated_returns)
        assert plan.scale_applied == 1.0
        assert plan.diversification_ratio > 1.2

    def test_negative_expected_value_is_dropped_with_a_reason(self):
        calibrator = ConfidenceCalibrator()
        raw = make_signal(stop_distance=60.0)        # costs swamp the payoff
        costs = estimate_costs(raw, quote=Quote("BTCUSDT", 59_994.0, 60_006.0), atr=600.0)
        edge = evaluate_edge(calibrator.estimate(0.7), 2.0, costs)
        assert not edge.tradable

        plan = PortfolioAllocator().allocate([(scored(raw), edge)], pd.DataFrame())
        assert plan.held == []
        assert plan.allocations[0].dropped
        assert "hurdle" in plan.allocations[0].reasons[0] or "quantile" in plan.allocations[0].reasons[0]

    def test_the_position_limit_keeps_the_best_ideas(self, correlated_returns):
        params = Params()
        params.portfolio.max_positions = 2
        candidates = self._candidates(
            [(BTC, "trend_following", Direction.LONG, 0.01),
             (ETH, "donchian_breakout", Direction.LONG, 0.01),
             (EURUSD, "trend_following", Direction.LONG, 0.01)]
        )
        plan = PortfolioAllocator(params).allocate(candidates, correlated_returns)
        assert len(plan.held) <= 2
        assert any("position limit" in r for a in plan.dropped for r in a.reasons)

    def test_units_are_rederived_from_the_allocated_risk(self, correlated_returns):
        """The sizing invariant must survive the portfolio pass.

        Risk is defined by the stop distance, so units must be recomputed from
        the final risk rather than scaled from the standalone units — otherwise
        a widened stop and a resized position silently disagree.
        """
        params = Params()
        equity = params.risk.account_equity
        candidates = self._candidates([(BTC, "trend_following", Direction.LONG, 0.01)])
        plan = PortfolioAllocator(params).allocate(candidates, correlated_returns, equity)

        allocation = plan.allocations[0]
        distance = abs(allocation.signal.raw.entry - allocation.signal.adjusted_stop)
        assert allocation.units * distance == pytest.approx(equity * allocation.risk_fraction)

    def test_apply_plan_writes_back_and_leaves_an_audit_trail(self, correlated_returns):
        candidates = self._candidates(
            [(BTC, "trend_following", Direction.LONG, 0.01),
             (ETH, "trend_following", Direction.LONG, 0.01)]
        )
        plan = apply_plan(PortfolioAllocator().allocate(candidates, correlated_returns))
        for allocation in plan.allocations:
            signal = allocation.signal
            assert signal.risk_fraction == pytest.approx(allocation.risk_fraction)
            assert signal.units == pytest.approx(allocation.units)
            assert any(o.name == "portfolio" for o in signal.outcomes)

    def test_a_position_sized_to_nothing_is_marked_blocked(self):
        """Otherwise it would still appear in ``AnalysisResult.tradable``."""
        calibrator = ConfidenceCalibrator()
        raw = make_signal(stop_distance=60.0)
        edge = evaluate_edge(
            calibrator.estimate(0.7), 2.0,
            estimate_costs(raw, quote=Quote("BTCUSDT", 59_994.0, 60_006.0), atr=600.0),
        )
        signal = scored(raw)
        apply_plan(PortfolioAllocator().allocate([(signal, edge)], pd.DataFrame()))
        assert signal.blocked
        assert signal.risk_fraction == 0.0
        assert signal.block_reasons

    def test_empty_candidate_set_is_handled(self):
        plan = PortfolioAllocator().allocate([], pd.DataFrame())
        assert plan.held == []
        assert plan.gross_risk == 0.0


# --------------------------------------------------------------------------- #
# Significance
# --------------------------------------------------------------------------- #
class TestSignificance:
    def test_more_trials_mean_a_higher_bar(self):
        assert expected_max_sharpe(1) == 0.0
        assert expected_max_sharpe(10) < expected_max_sharpe(100)

    def test_the_same_sharpe_deflates_as_the_search_widens(self):
        single = deflated_sharpe_ratio(0.08, 500, trials=1)
        many = deflated_sharpe_ratio(0.08, 500, trials=100)
        assert single > many

    def test_negative_skew_and_fat_tails_are_penalised(self):
        """The return shape of a trend strategy makes its Sharpe worth less."""
        clean = probabilistic_sharpe_ratio(0.1, 500, skew=0.0, excess_kurtosis=0.0)
        ugly = probabilistic_sharpe_ratio(0.1, 500, skew=-1.5, excess_kurtosis=6.0)
        assert ugly < clean
        assert sharpe_standard_error(0.1, 500, -1.5, 6.0) > sharpe_standard_error(0.1, 500)

    def test_more_observations_raise_confidence(self):
        assert probabilistic_sharpe_ratio(0.1, 1_000) > probabilistic_sharpe_ratio(0.1, 100)

    def test_a_negative_edge_never_becomes_significant(self):
        assert min_track_record_length(-0.05) == float("inf")

    def test_noise_is_reported_as_noise(self):
        """A zero-mean series searched over many trials must not pass."""
        rng = np.random.default_rng(42)
        noise = pd.Series(rng.normal(0.0, 0.01, 750))
        report = assess_significance(noise, trials=50, periods_per_year=252)
        assert report.dsr < 0.95
        assert report.verdict != "SIGNIFICANT"

    def test_a_strong_persistent_edge_survives_the_correction(self):
        rng = np.random.default_rng(9)
        strong = pd.Series(rng.normal(0.0015, 0.006, 1_500))
        report = assess_significance(strong, trials=50, periods_per_year=252)
        assert report.dsr > 0.95
        assert report.verdict == "SIGNIFICANT"

    def test_short_samples_are_reported_as_insufficient(self):
        report = assess_significance(pd.Series([0.01, -0.01, 0.02]), trials=5)
        assert report.verdict == "INSUFFICIENT DATA"
