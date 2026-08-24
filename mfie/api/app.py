"""The HTTP layer: a JSON API plus the static frontend that consumes it.

    uvicorn mfie.api:app --host 0.0.0.0 --port 8000

One process serves both the API and the interface, which is what makes this
deployable as a single unit — no separate node server, no build step, no CORS
configuration in the common case. The frontend is plain files under
``mfie/web``; there is nothing to compile before it runs.

Errors are handled the way the rest of the engine handles provider failures:
one endpoint failing returns a typed envelope for that endpoint and leaves the
rest of the page working. A trading interface where a dead calendar feed blanks
the signal list is worse than useless — it is misleading.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from mfie.api.serializers import (
    analysis_to_dict,
    calibration_to_dict,
    candles_to_dict,
    clean,
    cycle_to_dict,
    macro_to_dict,
    plan_to_dict,
    regime_to_dict,
    signal_to_dict,
)
from mfie.api.service import get_service
from mfie.core.universe import CRYPTO, FOREX, UNIVERSE
from mfie.core.utils import configure_console, get_logger

log = get_logger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
API_VERSION = "1.0"

# Query parameters shared by every endpoint that can trigger an engine run.
_SYMBOLS = Query(None, description="Comma-separated symbols, e.g. BTCUSDT,EURUSD")
_TIMEFRAME = Query("1h", pattern="^(1m|5m|15m|30m|1h|2h|4h|12h|1d|1w)$")
_LIMIT = Query(500, ge=60, le=2000, description="Bars of history to load")
_FORCE = Query(False, description="Bypass the cache and recompute")


def _symbol_tuple(symbols: str | None) -> tuple[str, ...] | None:
    """Parse and validate a comma-separated symbol list.

    Unknown symbols are rejected with a 400 rather than silently dropped: a
    typo that quietly returns a smaller universe is the kind of bug that gets
    noticed only after someone trades on the result.
    """
    if not symbols:
        return None
    parsed: list[str] = []
    unknown: list[str] = []
    for raw in symbols.split(","):
        candidate = raw.strip()
        if not candidate:
            continue
        try:
            from mfie.core.universe import get_instrument

            parsed.append(get_instrument(candidate).symbol)
        except KeyError:
            unknown.append(candidate)
    if unknown:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unknown_symbols",
                "message": f"Unknown symbol(s): {', '.join(unknown)}",
                "known": sorted(UNIVERSE),
            },
        )
    return tuple(parsed) or None


api = APIRouter(prefix="/api")


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
@api.get("/health")
def health() -> dict[str, Any]:
    """Liveness probe. Deliberately cheap — it must not run the engine."""
    return {"status": "ok", "version": API_VERSION}


@api.get("/status")
def status() -> dict[str, Any]:
    """Which providers are configured, and what the cache is holding."""
    return get_service().status()


@api.get("/instruments")
def instruments() -> dict[str, Any]:
    def describe(instrument) -> dict[str, Any]:
        return {
            "symbol": instrument.symbol,
            "name": instrument.name,
            "asset_class": clean(instrument.asset_class),
            "base": instrument.base,
            "quote": instrument.quote,
        }

    return {
        "crypto": [describe(i) for i in CRYPTO.values()],
        "forex": [describe(i) for i in FOREX.values()],
    }


@api.get("/params")
def params() -> dict[str, Any]:
    """The econometric thresholds in force.

    The interface quotes budgets and limits next to live readings, and those
    labels must come from the same configuration the engine used rather than
    being duplicated in the frontend where they would drift.
    """
    from dataclasses import asdict

    from mfie.config import get_params

    return clean(asdict(get_params()))


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
@api.get("/analysis")
def analysis(
    symbols: str | None = _SYMBOLS,
    timeframe: str = _TIMEFRAME,
    limit: int = _LIMIT,
    force: bool = _FORCE,
) -> dict[str, Any]:
    """One scan: macro, per-instrument state, every signal, and the book.

    The interface's primary call. It is a single request on purpose — the
    panels on a page must agree with each other, and four separate endpoints
    could each land on a different cache generation and disagree about whether
    the curve is inverted.
    """
    cached = get_service().analysis(_symbol_tuple(symbols), timeframe, limit, force)
    return cached.envelope("analysis", analysis_to_dict(cached.value))


@api.get("/signals")
def signals(
    symbols: str | None = _SYMBOLS,
    timeframe: str = _TIMEFRAME,
    limit: int = _LIMIT,
    include_blocked: bool = Query(True),
    force: bool = _FORCE,
) -> dict[str, Any]:
    cached = get_service().analysis(_symbol_tuple(symbols), timeframe, limit, force)
    result = cached.value
    chosen = result.all_signals if include_blocked else result.tradable
    payload = {
        "signals": [signal_to_dict(s) for s in chosen],
        "counts": {
            "generated": len(result.all_signals),
            "tradable": len(result.tradable),
            "blocked": len(result.blocked),
        },
    }
    return cached.envelope("data", payload)


@api.get("/portfolio")
def portfolio(
    symbols: str | None = _SYMBOLS,
    timeframe: str = _TIMEFRAME,
    limit: int = _LIMIT,
    force: bool = _FORCE,
) -> dict[str, Any]:
    cached = get_service().analysis(_symbol_tuple(symbols), timeframe, limit, force)
    plan = plan_to_dict(getattr(cached.value, "plan", None))
    if plan is None:
        return {
            **cached.envelope("portfolio", None),
            "disabled": True,
            "message": "Portfolio allocation is disabled in config/params.yaml.",
        }
    return cached.envelope("portfolio", plan)


@api.get("/macro")
def macro(
    symbols: str | None = _SYMBOLS,
    timeframe: str = _TIMEFRAME,
    limit: int = _LIMIT,
    force: bool = _FORCE,
) -> dict[str, Any]:
    cached = get_service().analysis(_symbol_tuple(symbols), timeframe, limit, force)
    return cached.envelope("macro", macro_to_dict(cached.value.macro))


@api.get("/cycle/{domain}")
def cycle(domain: str, force: bool = _FORCE) -> dict[str, Any]:
    if domain not in {"crypto", "fx"}:
        raise HTTPException(
            status_code=400,
            detail={"error": "unknown_domain", "message": "Domain must be 'crypto' or 'fx'."},
        )
    cached = get_service().cycle(domain, force)
    return cached.envelope("cycle", cycle_to_dict(cached.value))


@api.get("/instrument/{symbol}")
def instrument_detail(
    symbol: str,
    timeframe: str = _TIMEFRAME,
    limit: int = _LIMIT,
    points: int = Query(300, ge=50, le=2000),
    force: bool = _FORCE,
) -> dict[str, Any]:
    """Price, indicators, regime and this instrument's signals.

    Price and the scan are cached separately: a chart redraw at a new zoom
    level should not be able to trigger a full macro rebuild.
    """
    service = get_service()
    resolved = _symbol_tuple(symbol)
    if not resolved:
        raise HTTPException(status_code=400, detail={"error": "missing_symbol"})
    resolved_symbol = resolved[0]

    cached = service.ohlcv(resolved_symbol, timeframe, limit, force)
    instrument, frame, indicators = cached.value
    payload = candles_to_dict(instrument, frame, indicators, points)

    # Signals come from the shared scan so the detail view cannot disagree with
    # the list view about the same instrument.
    try:
        scan = service.analysis(resolved, timeframe, limit, force=False)
        analysis_obj = scan.value.analyses.get(resolved_symbol)
        if analysis_obj is not None:
            payload["regime"] = regime_to_dict(analysis_obj.regime)
            payload["signals"] = [signal_to_dict(s) for s in analysis_obj.signals]
            payload["error"] = analysis_obj.error
    except Exception as exc:
        log.warning("Signals unavailable for %s: %s", resolved_symbol, exc)
        payload["signals"] = []
        payload["signals_error"] = str(exc)

    return cached.envelope("instrument", payload)


@api.get("/calibration")
def calibration(force: bool = _FORCE) -> dict[str, Any]:
    cached = get_service().calibration(force)
    return cached.envelope("calibration", calibration_to_dict(cached.value))


@api.post("/refresh")
def refresh() -> dict[str, Any]:
    """Drop every cached computation. The next request rebuilds from source."""
    dropped = get_service().store.invalidate()
    return {"status": "ok", "dropped": dropped}


@api.get("/export/analysis")
def export_analysis(
    symbols: str | None = _SYMBOLS,
    timeframe: str = _TIMEFRAME,
    limit: int = _LIMIT,
) -> dict[str, Any]:
    """The complete scan as one document, for archiving or downstream tooling."""
    cached = get_service().analysis(_symbol_tuple(symbols), timeframe, limit, force=False)
    return cached.envelope("analysis", analysis_to_dict(cached.value, include_history=True))


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_console()
    log.info("MFIE API ready — serving frontend from %s", WEB_DIR)
    yield
    log.info("MFIE API shutting down")


def create_app(serve_frontend: bool = True) -> FastAPI:
    app = FastAPI(
        title="MFIE",
        description="Macro-Informed Financial Intelligence Engine",
        version=API_VERSION,
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    # Payloads are JSON with long numeric arrays and compress by roughly 10x.
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    @app.middleware("http")
    async def timing(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - started) * 1000
        response.headers["X-Response-Time"] = f"{elapsed:.0f}ms"
        return response

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        """Return a typed envelope instead of an HTML traceback.

        The frontend renders ``error``/``message`` in place of the panel that
        failed, so one broken endpoint degrades one card.
        """
        log.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "message": str(exc) or exc.__class__.__name__,
                "path": request.url.path,
            },
        )

    app.include_router(api)

    if serve_frontend and WEB_DIR.is_dir():
        assets = WEB_DIR / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(WEB_DIR / "index.html")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            """Serve the app shell for client-side routes.

            Deep links like ``/portfolio`` must survive a page reload, so any
            unmatched path that is not an API call returns ``index.html`` and
            lets the router in the browser resolve it. API paths are excluded
            explicitly — a mistyped endpoint should 404 as JSON, not hand the
            caller a page of HTML.
            """
            if path.startswith("api/"):
                raise HTTPException(status_code=404, detail={"error": "not_found"})
            candidate = WEB_DIR / path
            if candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(WEB_DIR / "index.html")

    return app


app = create_app()
