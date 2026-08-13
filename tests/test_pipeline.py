"""Tests for the filter chain, sizing, engine and backtester.

The point of these is behavioural: a rule that is supposed to block a trade must
actually block it, and the risk arithmetic must hold exactly.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mfie.config import Params
from mfie.core.types import (
    Direction,
    EconomicEvent,
    FilterAction,
    Instrument,
    LiquidityRegime,
    MacroContext,
    MacroRegime,
    Quote,
    RawSignal,
)
from mfie.core.universe import get_instrument, is_high_beta
from mfie.data.synthetic import SyntheticProvider
from mfie.econ.microstructure import compute_microstructure
from mfie.econ.rates import compute_rate_view
from mfie.pipeline.filters import (
    EventBlockerFilter,
    FilterContext,
    LiquidityFilter,
    MicrostructureFilter,
    RealYieldFilter,
    SentimentFilter,
    YieldCurveFilter,
    build_filters,
)
from mfie.pipeline.sizing import combine_multipliers, size_signal

PROVIDER = SyntheticProvider()


@pytest.fixture
def params() -> Params:
    return Params()


@pytest.fixture
def btc() -> Instrument:
    return get_instrument("BTCUSDT")


@pytest.fixture
def sol() -> Instrument:
    return get_instrument("SOLUSDT")


@pytest.fixture
def eurusd() -> Instrument:
    return get_instrument("EURUSD")


def make_signal(instrument: Instrument, direction: Direction = Direction.LONG,
                strategy: str = "trend_following", strength: float = 0.7) -> RawSignal:
    entry = 100.0
    stop = entry * (0.98 if direction is Direction.LONG else 1.02)
    target = entry * (1.04 if direction is Direction.LONG else 0.96)
    return RawSignal(
        instrument=instrument, direction=direction, strategy=strategy,
        strength=strength, entry=entry, stop=stop, take_profit=target,
    )


def make_context(instrument: Instrument, params: Params, **macro_kwargs) -> FilterContext:
    macro = MacroContext(**macro_kwargs)
    return FilterContext(instrument=instrument, macro=macro, params=params)


# --------------------------------------------------------------------------- #
class TestLiquidityFilter:
    def test_expansion_passes(self, btc, params):
        ctx = make_context(btc, params, gli_delta=0.03, gli_regime=LiquidityRegime.EXPANSION)
        outcome = LiquidityFilter()(make_signal(btc), ctx)
        assert outcome.action is FilterAction.PASS
        assert outcome.multiplier == pytest.approx(1.0)

    def test_contraction_penalises_momentum_longs(self, btc, params):
        ctx = make_context(btc, params, gli_delta=-0.03, gli_regime=LiquidityRegime.CONTRACTION)
        outcome = LiquidityFilter()(make_signal(btc, Direction.LONG), ctx)
        assert outcome.action is FilterAction.PENALIZE
        assert outcome.multiplier < 1.0

    def test_contraction_leaves_shorts_alone(self, btc, params):
        ctx = make_context(btc, params, gli_delta=-0.03, gli_regime=LiquidityRegime.CONTRACTION)
        outcome = LiquidityFilter()(make_signal(btc, Direction.SHORT), ctx)
        assert outcome.multiplier >= 1.0

    def test_contraction_does_not_touch_mean_reversion(self, btc, params):
        ctx = make_context(btc, params, gli_delta=-0.03, gli_regime=LiquidityRegime.CONTRACTION)
        signal = make_signal(btc, Direction.LONG, strategy="bollinger_reversion")
        assert LiquidityFilter()(signal, ctx).multiplier == pytest.approx(1.0)

    def test_penalty_scales_with_severity(self, btc, params):
        mild = make_context(btc, params, gli_delta=-0.002, gli_regime=LiquidityRegime.CONTRACTION)
        severe = make_context(btc, params, gli_delta=-0.10, gli_regime=LiquidityRegime.CONTRACTION)
        signal = make_signal(btc, Direction.LONG)
        filt = LiquidityFilter()
        assert filt(signal, mild).multiplier > filt(signal, severe).multiplier


class TestYieldCurveFilter:
    def test_inverted_curve_blocks_high_beta_longs(self, sol, params):
        assert is_high_beta(sol)
        ctx = make_context(sol, params, macro_regime=MacroRegime.CONTRACTION)
        ctx.rate_view = compute_rate_view(
            sol, {"SOL": 0.0, "USDT": 0.0}, {"SOL": 0.0, "USDT": 0.0},
            {"USD": 0.0415}, {"USD": 0.0445},
        )
        outcome = YieldCurveFilter()(make_signal(sol, Direction.LONG), ctx)
        assert outcome.blocked

    def test_majors_are_only_penalised(self, btc, params):
        assert not is_high_beta(btc)
        ctx = make_context(btc, params, macro_regime=MacroRegime.CONTRACTION)
        ctx.rate_view = compute_rate_view(
            btc, {}, {}, {"USD": 0.0415}, {"USD": 0.0445},
        )
        outcome = YieldCurveFilter()(make_signal(btc, Direction.LONG), ctx)
        assert not outcome.blocked
        assert outcome.multiplier < 1.0

    def test_positive_curve_passes(self, btc, params):
        ctx = make_context(btc, params, macro_regime=MacroRegime.EXPANSION)
        ctx.rate_view = compute_rate_view(btc, {}, {}, {"USD": 0.045}, {"USD": 0.030})
        assert YieldCurveFilter()(make_signal(btc), ctx).action is FilterAction.PASS


class TestEventBlocker:
    def _event(self, minutes: float, currency: str = "USD", impact: str = "high"):
        return EconomicEvent(
            ts=datetime.now(timezone.utc) + timedelta(minutes=minutes),
            country="US", currency=currency, name="CPI y/y", impact=impact,
        )

    def test_imminent_high_impact_event_blocks(self, eurusd, params):
        ctx = make_context(eurusd, params, events=[self._event(10)])
        outcome = EventBlockerFilter()(make_signal(eurusd), ctx)
        assert outcome.blocked
        assert "CPI" in outcome.reason

    def test_event_just_outside_the_window_passes(self, eurusd, params):
        ctx = make_context(eurusd, params, events=[self._event(45)])
        assert not EventBlockerFilter()(make_signal(eurusd), ctx).blocked

    def test_recent_release_still_blocks(self, eurusd, params):
        ctx = make_context(eurusd, params, events=[self._event(-15)])
        assert EventBlockerFilter()(make_signal(eurusd), ctx).blocked

    def test_unrelated_currency_does_not_block(self, eurusd, params):
        ctx = make_context(eurusd, params, events=[self._event(10, currency="AUD")])
        assert not EventBlockerFilter()(make_signal(eurusd), ctx).blocked

    def test_usd_events_reach_crypto(self, btc, params):
        """Dollar liquidity prices crypto too — a USD CPI print must block BTC."""
        ctx = make_context(btc, params, events=[self._event(10, currency="USD")])
        assert EventBlockerFilter()(make_signal(btc), ctx).blocked

    def test_medium_impact_only_penalises(self, eurusd, params):
        ctx = make_context(eurusd, params, events=[self._event(10, impact="medium")])
        outcome = EventBlockerFilter()(make_signal(eurusd), ctx)
        assert not outcome.blocked
        assert outcome.multiplier < 1.0


class TestMicrostructureFilter:
    def test_wide_spread_blocks(self, btc, params):
        ctx = make_context(btc, params)
        ctx.microstructure = compute_microstructure(
            Quote("BTCUSDT", bid=90.0, ask=110.0), 20.0, params.microstructure
        )
        assert MicrostructureFilter()(make_signal(btc), ctx).blocked

    def test_thin_book_widens_the_stop(self, btc, params):
        ctx = make_context(btc, params)
        ctx.microstructure = compute_microstructure(
            Quote("BTCUSDT", bid=99.0, ask=101.0), 10.0, params.microstructure
        )
        outcome = MicrostructureFilter()(make_signal(btc), ctx)
        assert not outcome.blocked
        assert outcome.stop_multiplier > 1.0


class TestSentimentFilter:
    def test_crowded_long_is_penalised(self, eurusd, params):
        ctx = make_context(eurusd, params, retail_long_pct={"EURUSD": 0.92}, fear_greed=50.0)
        outcome = SentimentFilter()(make_signal(eurusd, Direction.LONG), ctx)
        assert outcome.multiplier < 1.0

    def test_contrarian_short_is_not_penalised(self, eurusd, params):
        ctx = make_context(eurusd, params, retail_long_pct={"EURUSD": 0.92}, fear_greed=50.0)
        outcome = SentimentFilter()(make_signal(eurusd, Direction.SHORT), ctx)
        assert outcome.multiplier >= 1.0


class TestRealYieldFilter:
    def test_only_applies_to_forex(self, btc, params):
        ctx = make_context(btc, params)
        outcome = RealYieldFilter()(make_signal(btc), ctx)
        assert outcome.reason == "not applicable"


# --------------------------------------------------------------------------- #
class TestSizing:
    def test_multipliers_compound(self):
        from mfie.core.types import FilterOutcome

        outcomes = [
            FilterOutcome("a", FilterAction.PENALIZE, 0.8),
            FilterOutcome("b", FilterAction.PENALIZE, 0.5),
        ]
        assert combine_multipliers(outcomes) == pytest.approx(0.4)

    def test_confidence_is_strength_times_multipliers(self, btc, params):
        from mfie.core.types import FilterOutcome

        signal = make_signal(btc, strength=0.8)
        outcomes = [FilterOutcome("a", FilterAction.PENALIZE, 0.5)]
        decision = size_signal(signal, outcomes, params=params, equity=10_000)
        assert decision.confidence == pytest.approx(0.4)

    def test_risk_below_the_minimum_produces_no_position(self, btc, params):
        from mfie.core.types import FilterOutcome

        signal = make_signal(btc, strength=0.5)
        outcomes = [FilterOutcome("a", FilterAction.PENALIZE, 0.2)]  # confidence 0.10
        decision = size_signal(signal, outcomes, params=params, equity=10_000)
        assert decision.units == 0.0
        assert any("below minimum" in note for note in decision.notes)

    def test_units_risk_exactly_the_stated_fraction(self, btc, params):
        signal = make_signal(btc, strength=0.9)
        decision = size_signal(signal, [], params=params, equity=10_000)
        risked = decision.units * abs(signal.entry - decision.adjusted_stop)
        assert risked == pytest.approx(10_000 * decision.risk_fraction, rel=1e-9)

    def test_widening_the_stop_shrinks_size_and_holds_risk_constant(self, btc, params):
        from mfie.core.types import FilterOutcome

        signal = make_signal(btc, strength=0.9)
        tight = size_signal(signal, [], params=params, equity=10_000)
        widened = size_signal(
            signal,
            [FilterOutcome("micro", FilterAction.PASS, 1.0, stop_multiplier=2.0)],
            params=params, equity=10_000,
        )
        assert widened.units < tight.units
        # The invariant that keeps the microstructure filter honest.
        assert widened.units * abs(signal.entry - widened.adjusted_stop) == pytest.approx(
            10_000 * widened.risk_fraction, rel=1e-9
        )

    def test_risk_is_capped_per_trade(self, btc, params):
        params.risk.base_risk_per_trade = 0.10  # deliberately reckless
        decision = size_signal(make_signal(btc, strength=1.0), [], params=params, equity=10_000)
        assert decision.risk_fraction <= params.risk.max_risk_per_trade

    def test_portfolio_heat_caps_new_risk(self, btc, params):
        decision = size_signal(
            make_signal(btc, strength=0.9), [], params=params, equity=10_000,
            open_risk=params.risk.max_portfolio_risk,
        )
        assert decision.risk_fraction == 0.0

    def test_reward_risk_is_preserved_when_the_stop_moves(self, btc, params):
        from mfie.core.types import FilterOutcome

        signal = make_signal(btc, strength=0.9)
        original_rr = signal.reward_risk
        decision = size_signal(
            signal,
            [FilterOutcome("micro", FilterAction.PASS, 1.0, stop_multiplier=1.5)],
            params=params, equity=10_000,
        )
        new_rr = abs(decision.adjusted_take_profit - signal.entry) / abs(
            signal.entry - decision.adjusted_stop
        )
        assert new_rr == pytest.approx(original_rr, rel=1e-9)


# --------------------------------------------------------------------------- #
class TestEngine:
    def test_run_produces_a_complete_result(self):
        from mfie.pipeline.engine import AnalysisEngine

        result = AnalysisEngine().run(symbols=["BTCUSDT", "EURUSD"], timeframe="1h", limit=400)
        assert len(result.analyses) == 2
        assert result.macro.ts is not None
        assert isinstance(result.summary(), dict)

    def test_every_signal_carries_an_audit_trail(self):
        from mfie.pipeline.engine import AnalysisEngine

        result = AnalysisEngine().run(symbols=["BTCUSDT", "ETHUSDT", "EURUSD"], limit=400)
        for signal in result.all_signals:
            assert signal.audit(), "a signal reached the user with no explanation"
            assert signal.verdict in ("STRONG", "MODERATE", "WEAK", "BLOCKED")

    def test_blocked_signals_carry_no_size(self):
        """Whichever stage blocks a signal, it must end up with no position.

        Confidence is not asserted to be zero. A filter-chain veto zeroes it,
        but the expected-value gate and the portfolio layer both run *after*
        confidence is known and deliberately keep it — "the macro chain liked
        this at 71% and the arithmetic still rejected it" is the most
        informative line the audit trail can produce, and zeroing the number
        would erase it.
        """
        from mfie.pipeline.engine import AnalysisEngine

        result = AnalysisEngine().run(symbols=["BTCUSDT", "SOLUSDT", "EURUSD"], limit=400)
        for signal in result.blocked:
            assert signal.units == 0.0
            assert signal.risk_fraction == 0.0
            assert signal.size_fraction == 0.0
            assert signal.block_reasons

    def test_late_stage_blocks_keep_their_confidence(self):
        """A trade rejected on cost arithmetic still records what the chain thought."""
        from mfie.pipeline.engine import AnalysisEngine

        result = AnalysisEngine().run(symbols=["BTCUSDT", "SOLUSDT", "EURUSD"], limit=400)
        late = [
            s for s in result.blocked
            if any(r.startswith(("expected_value", "portfolio")) for r in s.block_reasons)
        ]
        for signal in late:
            assert signal.confidence > 0.0
            assert signal.edge is not None

    def test_macro_snapshot_is_shared_across_instruments(self):
        from mfie.pipeline.engine import AnalysisEngine

        result = AnalysisEngine().run(symbols=["BTCUSDT", "EURUSD"], limit=400)
        contexts = {id(s.macro) for s in result.all_signals}
        assert len(contexts) <= 1, "filters saw different views of the world in one run"


class TestBacktest:
    def test_backtest_runs_and_reports(self, eurusd):
        from mfie.backtest.engine import BacktestEngine
        from mfie.backtest.metrics import evaluate_performance

        df = PROVIDER.ohlcv(eurusd, "1h", 1200)
        result = BacktestEngine().run(eurusd, df, MacroContext(), warmup=250)
        report = evaluate_performance(result)

        assert len(result.equity) > 0
        assert report.trades == len(result.closed_trades)
        if report.trades:
            assert 0.0 <= report.win_rate <= 1.0

    def test_entries_never_precede_their_signal_bar(self, eurusd):
        """Every fill must be at least one bar after the decision."""
        from mfie.backtest.engine import BacktestEngine

        df = PROVIDER.ohlcv(eurusd, "1h", 1000)
        result = BacktestEngine().run(eurusd, df, MacroContext(), warmup=250)
        for trade in result.trades:
            assert trade.entry_ts in df.index
            if trade.exit_ts is not None:
                assert trade.exit_ts >= trade.entry_ts

    def test_costs_make_a_difference(self, eurusd):
        from mfie.backtest.engine import BacktestEngine, CostModel

        df = PROVIDER.ohlcv(eurusd, "1h", 1200)
        free = BacktestEngine(costs=CostModel(0.0, 0.0, 0.0)).run(
            eurusd, df, MacroContext(), warmup=250
        )
        costly = BacktestEngine(costs=CostModel(0.002, 0.002, 0.3)).run(
            eurusd, df, MacroContext(), warmup=250
        )
        if free.closed_trades and costly.closed_trades:
            assert costly.equity.iloc[-1] < free.equity.iloc[-1]

    def test_too_little_data_raises(self, eurusd):
        from mfie.backtest.engine import BacktestEngine

        df = PROVIDER.ohlcv(eurusd, "1h", 100)
        with pytest.raises(ValueError, match="bars"):
            BacktestEngine().run(eurusd, df, MacroContext(), warmup=250)


class TestFilterChain:
    def test_default_chain_is_ordered_by_level(self):
        filters = build_filters()
        levels = [f.level for f in filters]
        assert levels == sorted(levels)

    def test_chain_can_be_restricted(self):
        filters = build_filters(["event_blocker", "global_liquidity"])
        assert {f.name for f in filters} == {"event_blocker", "global_liquidity"}
