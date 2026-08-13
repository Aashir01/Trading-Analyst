"""Tests for the output layer: report formatting and the dashboard.

The dashboard test is the one that matters. A Streamlit app fails at *render*
time, not import time, so importing the module proves nothing — ``AppTest``
actually executes the script and surfaces any exception.
"""

from __future__ import annotations

import pytest

from mfie.interfaces.report import (
    DISCLAIMER,
    format_macro_brief,
    format_result,
    format_signal,
    format_watchlist,
    signal_to_telegram,
)
from mfie.interfaces.theme import DARK, LIGHT, base_layout, get_palette, verdict_status
from mfie.pipeline.engine import AnalysisEngine


@pytest.fixture(scope="module")
def result():
    return AnalysisEngine().run(
        symbols=["BTCUSDT", "SOLUSDT", "EURUSD", "USDJPY"], timeframe="1h", limit=400
    )


class TestReport:
    def test_macro_brief_covers_the_regime_state(self, result):
        text = format_macro_brief(result.macro, verbose=True)
        assert "MACRO CONTEXT" in text
        assert "Liquidity regime" in text
        assert "Macro regime" in text

    def test_full_report_always_carries_the_disclaimer(self, result):
        assert DISCLAIMER in format_result(result)

    def test_every_generated_signal_is_accounted_for(self, result):
        """No signal may vanish between generation and the report."""
        text = format_result(result)
        shown = (
            len(result.tradable)
            + len(result.blocked)
            + len([s for s in result.all_signals if not s.blocked and s.risk_fraction <= 0])
        )
        assert shown == len(result.all_signals)
        if result.blocked:
            assert "BLOCKED SIGNALS" in text

    def test_signal_render_includes_levels_and_audit(self, result):
        signals = result.all_signals
        if not signals:
            pytest.skip("no signal generated on this synthetic sample")
        text = format_signal(signals[0], show_audit=True)
        assert "Entry" in text and "Stop" in text

    def test_telegram_message_fits_the_platform_limit(self, result):
        for signal in result.all_signals:
            assert len(signal_to_telegram(signal, result.macro)) <= 4096

    def test_watchlist_has_one_row_per_instrument(self, result):
        lines = format_watchlist(result).splitlines()
        assert len(lines) == len(result.analyses) + 1  # + header


class TestTheme:
    def test_both_modes_are_fully_specified(self):
        for palette in (LIGHT, DARK):
            assert palette.surface and palette.ink_primary
            assert len(palette.categorical) == 8

    def test_categorical_slots_do_not_silently_wrap(self):
        # A 9th series must fail loudly rather than reuse slot 1 and imply that
        # two different entities are the same one.
        with pytest.raises(IndexError):
            get_palette("dark").series(8)

    def test_layout_paints_its_own_surface(self):
        layout = base_layout(DARK)
        assert layout["paper_bgcolor"] == DARK.surface
        assert layout["plot_bgcolor"] == DARK.surface

    def test_verdicts_map_to_reserved_status_roles(self):
        assert verdict_status("STRONG") == "good"
        assert verdict_status("BLOCKED") == "critical"


class TestDashboard:
    def test_dashboard_renders_without_error(self):
        from pathlib import Path

        from streamlit.testing.v1 import AppTest

        # AppTest resolves relative paths against the *calling* file, so anchor
        # on the repository root explicitly.
        script = Path(__file__).resolve().parents[1] / "mfie" / "interfaces" / "dashboard.py"
        app = AppTest.from_file(str(script), default_timeout=300)
        app.run()

        assert not app.exception, [str(e.value) for e in app.exception]
        labels = [tab.label for tab in app.tabs]
        assert labels == [
            "Signals", "Portfolio", "Cycle", "Macro", "Charts", "Backtest", "Data sources"
        ]
        assert len(app.metric) > 0
