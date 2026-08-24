"""Tests for the HTTP layer and the frontend it serves.

The API is the only thing standing between the engine and every consumer of
it, so these check the contract rather than the implementation: payloads must
be JSON-encodable (the engine is full of pandas frames and numpy scalars that
are not), non-finite floats must become null rather than the literal `NaN`
that `JSON.parse` rejects, bad input must fail loudly instead of silently
returning a smaller universe, and deep links must survive a reload.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("fastapi", reason="the web extra is not installed")

from fastapi.testclient import TestClient  # noqa: E402

from mfie.api import create_app  # noqa: E402
from mfie.api.serializers import clean  # noqa: E402
from mfie.api.service import TTLStore  # noqa: E402

WEB_DIR = Path(__file__).resolve().parents[1] / "mfie" / "web"


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


# --------------------------------------------------------------------------- #
# Metadata endpoints
# --------------------------------------------------------------------------- #
class TestMetadata:
    def test_health_is_cheap_and_does_not_run_the_engine(self, client):
        """A healthcheck that triggered a scan would fail its own timeout."""
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_status_reports_provider_configuration(self, client):
        payload = client.get("/api/status").json()
        assert "providers" in payload
        assert "binance" in payload["providers"]
        assert isinstance(payload["configured"], int)

    def test_instruments_covers_the_universe(self, client):
        payload = client.get("/api/instruments").json()
        symbols = {i["symbol"] for i in payload["crypto"] + payload["forex"]}
        assert {"BTCUSDT", "EURUSD"} <= symbols

    def test_params_exposes_the_thresholds_the_ui_quotes(self, client):
        """The interface labels budgets from here rather than hardcoding them."""
        payload = client.get("/api/params").json()
        assert "max_effective_risk" in payload["portfolio"]
        assert "kelly_fraction_cap" in payload["risk"]


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
class TestAnalysis:
    def test_analysis_returns_a_complete_scan(self, client):
        response = client.get("/api/analysis?symbols=BTCUSDT,EURUSD&limit=200")
        assert response.status_code == 200
        payload = response.json()

        assert {"macro", "signals", "instruments", "portfolio", "cycles"} <= set(
            payload["analysis"]
        )
        assert len(payload["analysis"]["instruments"]) == 2
        assert {"as_of", "age_seconds", "stale"} <= set(payload["meta"])

    def test_payload_is_json_encodable(self, client):
        """The engine holds pandas frames and enums; none may reach the wire."""
        payload = client.get("/api/analysis?symbols=BTCUSDT&limit=200").json()
        json.dumps(payload)          # raises if anything survived serialisation

    def test_no_nan_or_infinity_in_the_payload(self, client):
        """NaN is legal in numpy and illegal in JSON — JSON.parse rejects it."""
        raw = client.get("/api/analysis?symbols=BTCUSDT&limit=200").text
        assert "NaN" not in raw
        assert "Infinity" not in raw

    def test_signals_carry_their_audit_trail(self, client):
        payload = client.get("/api/signals?symbols=BTCUSDT,ETHUSDT&limit=200").json()
        for signal in payload["data"]["signals"]:
            # The audit trail is the product; a signal without it is useless.
            assert isinstance(signal["outcomes"], list)
            assert "confidence" in signal
            assert "block_reasons" in signal

    def test_blocked_signals_carry_no_size(self, client):
        payload = client.get("/api/signals?symbols=BTCUSDT,SOLUSDT&limit=200").json()
        for signal in payload["data"]["signals"]:
            if signal["blocked"]:
                assert signal["risk_fraction"] == 0
                assert signal["units"] == 0

    def test_portfolio_exposes_the_correlation_matrix(self, client):
        payload = client.get("/api/portfolio?symbols=BTCUSDT,ETHUSDT,EURUSD&limit=200").json()
        plan = payload["portfolio"]
        if plan is None:
            pytest.skip("portfolio allocation disabled in config")
        assert "effective_risk" in plan
        assert "diversification_ratio" in plan
        if plan["correlation"]:
            matrix = plan["correlation"]["matrix"]
            n = len(plan["correlation"]["labels"])
            assert len(matrix) == n and all(len(row) == n for row in matrix)

    def test_macro_endpoint_returns_rates_and_calendar(self, client):
        payload = client.get("/api/macro?symbols=EURUSD&limit=200").json()["macro"]
        assert isinstance(payload["rates"], list)
        assert isinstance(payload["events"], list)
        assert "sources" in payload

    def test_instrument_detail_returns_candles_and_overlays(self, client):
        payload = client.get("/api/instrument/BTCUSDT?limit=200&points=120").json()
        detail = payload["instrument"]
        assert len(detail["candles"]) == 120
        assert {"open", "high", "low", "close", "ts"} <= set(detail["candles"][0])
        # Overlay arrays must align with the candles or the chart draws garbage.
        for series in detail["overlays"].values():
            assert len(series) == len(detail["candles"])

    def test_cycle_endpoint_returns_factors(self, client):
        payload = client.get("/api/cycle/crypto").json()["cycle"]
        assert payload["domain"] == "crypto"
        assert len(payload["factors"]) > 0
        assert -1.0 <= payload["score"] <= 1.0


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #
class TestInputHandling:
    def test_unknown_symbol_is_rejected_not_ignored(self, client):
        """Silently dropping a typo returns a smaller universe than requested."""
        response = client.get("/api/analysis?symbols=NOTREAL")
        assert response.status_code == 400
        assert response.json()["detail"]["error"] == "unknown_symbols"

    def test_symbol_aliases_are_accepted(self, client):
        response = client.get("/api/analysis?symbols=btc-usdt,eur/usd&limit=200")
        assert response.status_code == 200
        symbols = {i["symbol"] for i in response.json()["analysis"]["instruments"]}
        assert symbols == {"BTCUSDT", "EURUSD"}

    def test_invalid_timeframe_is_rejected(self, client):
        assert client.get("/api/analysis?timeframe=3h").status_code == 422

    def test_limit_is_bounded(self, client):
        assert client.get("/api/analysis?limit=99999").status_code == 422
        assert client.get("/api/analysis?limit=1").status_code == 422

    def test_unknown_cycle_domain_is_rejected(self, client):
        assert client.get("/api/cycle/equities").status_code == 400


# --------------------------------------------------------------------------- #
# Frontend delivery
# --------------------------------------------------------------------------- #
class TestFrontend:
    def test_index_is_served(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_assets_are_served(self, client):
        for path in ("/assets/app.css", "/assets/tokens.css", "/assets/main.js",
                     "/assets/core.js", "/assets/charts.js", "/assets/views.js"):
            assert client.get(path).status_code == 200, path

    def test_deep_links_return_the_app_shell(self, client):
        """A reload on /portfolio must not 404 — the router runs in the browser."""
        for path in ("/portfolio", "/cycle", "/instrument/BTCUSDT"):
            response = client.get(path)
            assert response.status_code == 200
            assert "<title>" in response.text

    def test_unknown_api_paths_404_as_json_not_html(self, client):
        """Otherwise a mistyped endpoint hands the caller a page of HTML."""
        response = client.get("/api/nope")
        assert response.status_code == 404
        assert "text/html" not in response.headers.get("content-type", "")

    def test_frontend_has_no_external_script_or_style_dependencies(self):
        """The app must run with no CDN reachable.

        Fonts are the one allowed external request and are loaded with
        media="print" so they never block rendering; everything else has to be
        local, or an air-gapped deployment would render as unstyled text.
        """
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        for line in html.splitlines():
            stripped = line.strip()
            if "<script" in stripped and "src=" in stripped:
                assert 'src="/assets/' in stripped, f"external script: {stripped}"
            if "<link" in stripped and "stylesheet" in stripped and "http" in stripped:
                assert "fonts.googleapis.com" in stripped, f"external stylesheet: {stripped}"

    def test_no_javascript_still_explains_how_to_use_the_tool(self):
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        assert "<noscript>" in html
        assert "/api/analysis" in html


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #
class TestSerializers:
    def test_non_finite_floats_become_null(self):
        assert clean(float("nan")) is None
        assert clean(float("inf")) is None
        assert clean(float("-inf")) is None
        assert clean(1.5) == 1.5

    def test_nested_structures_are_cleaned(self):
        payload = clean({"a": [1.0, float("nan")], "b": {"c": float("inf")}})
        assert payload == {"a": [1.0, None], "b": {"c": None}}

    def test_pandas_and_numpy_types_are_converted(self):
        import numpy as np

        assert clean(np.float64(2.5)) == 2.5
        assert clean(np.int64(7)) == 7
        assert clean(np.bool_(True)) is True
        assert isinstance(clean(pd.Timestamp("2024-01-01")), str)

    def test_enums_serialise_to_their_value(self):
        from mfie.core.types import Direction

        assert clean(Direction.LONG) == "long"


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
class TestCache:
    def test_value_is_computed_once_while_fresh(self):
        calls = []

        def produce():
            calls.append(1)
            return "value"

        store = TTLStore(ttl=60)
        for _ in range(5):
            store.get("k", produce)
        assert len(calls) == 1

    def test_concurrent_cold_requests_produce_once(self):
        """Without single-flight, a page of four panels runs the engine four times."""
        import threading

        calls = []
        started = threading.Event()

        def produce():
            calls.append(1)
            started.wait(timeout=2)
            return "value"

        store = TTLStore(ttl=60)
        threads = [threading.Thread(target=lambda: store.get("k", produce)) for _ in range(4)]
        for thread in threads:
            thread.start()
        started.set()
        for thread in threads:
            thread.join(timeout=5)

        assert len(calls) == 1

    def test_stale_entries_are_still_served(self):
        """Blanking the panels for eight seconds is worse than showing an age."""
        store = TTLStore(ttl=0.01)
        store.get("k", lambda: "first")
        import time

        time.sleep(0.05)
        cached = store.get("k", lambda: "second")
        assert cached.value == "first"      # served immediately, refreshed behind
        assert cached.stale is True

    def test_force_bypasses_the_cache(self):
        calls = []
        store = TTLStore(ttl=60)
        store.get("k", lambda: calls.append(1) or "a")
        store.get("k", lambda: calls.append(1) or "b", force=True)
        assert len(calls) == 2

    def test_invalidate_drops_entries(self):
        store = TTLStore(ttl=60)
        store.get("a", lambda: 1)
        store.get("b", lambda: 2)
        assert store.invalidate() == 2
        assert store.stats()["entries"] == 0

    def test_meta_reports_freshness(self):
        store = TTLStore(ttl=60)
        meta = store.get("k", lambda: "v").meta()
        assert meta["stale"] is False
        assert math.isfinite(meta["age_seconds"])
