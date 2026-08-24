"""Command line interface.

    python -m mfie --help

Every command works offline against synthetic data, so the tool is explorable
before any API key exists. Add keys to ``.env`` and the same commands switch to
live data with no other change.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mfie.config import PROJECT_ROOT, get_params, get_settings
from mfie.core.types import AssetClass
from mfie.core.universe import UNIVERSE, get_instrument, resolve
from mfie.core.utils import configure_console, get_logger

configure_console()

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Macro-Informed Financial Intelligence Engine — Forex & Crypto analysis.",
)
console = Console()
log = get_logger(__name__)


def _parse_symbols(symbols: str | None) -> list[str] | None:
    if not symbols:
        return None
    return [s.strip() for s in symbols.split(",") if s.strip()]


def _asset_class(name: str | None) -> AssetClass | None:
    if not name:
        return None
    try:
        return AssetClass(name.lower())
    except ValueError as exc:
        raise typer.BadParameter("asset class must be 'crypto' or 'forex'") from exc


# --------------------------------------------------------------------------- #
@app.command()
def analyze(
    symbols: str | None = typer.Option(None, "--symbols", "-s",
                                          help="Comma-separated, e.g. BTCUSDT,EURUSD"),
    timeframe: str = typer.Option("1h", "--timeframe", "-t"),
    limit: int = typer.Option(500, "--limit", "-l", help="Bars of history to load"),
    asset_class: str | None = typer.Option(None, "--class", "-c", help="crypto | forex"),
    audit: bool = typer.Option(True, "--audit/--no-audit", help="Show the filter chain"),
    save: bool = typer.Option(False, "--save", help="Persist signals to the database"),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
    strategies: str | None = typer.Option(None, "--strategies",
                                             help="Comma-separated strategy names"),
) -> None:
    """Run the full pipeline and print scored, filtered signals."""
    from mfie.interfaces.report import format_result
    from mfie.pipeline.engine import AnalysisEngine
    from mfie.strategies.registry import build_strategies

    strategy_objects = build_strategies(_parse_symbols(strategies)) if strategies else None
    engine = AnalysisEngine(strategies=strategy_objects)

    with console.status("Building macro context and scanning instruments..."):
        result = engine.run(
            symbols=_parse_symbols(symbols),
            timeframe=timeframe,
            limit=limit,
            asset_class=_asset_class(asset_class),
        )

    if as_json:
        payload = {
            "summary": result.summary(),
            "signals": [
                {
                    "symbol": s.instrument.symbol,
                    "direction": s.direction.value,
                    "strategy": s.raw.strategy,
                    "confidence": s.confidence,
                    "verdict": s.verdict,
                    "blocked": s.blocked,
                    "entry": s.raw.entry,
                    "stop": s.adjusted_stop or s.raw.stop,
                    "take_profit": s.adjusted_take_profit or s.raw.take_profit,
                    "risk_fraction": s.risk_fraction,
                    "units": s.units,
                    "audit": s.audit(),
                }
                for s in result.all_signals
            ],
        }
        console.print_json(json.dumps(payload, default=str))
    else:
        console.print(format_result(result, show_audit=audit))

    if save:
        from mfie.storage.repo import Repository

        repo = Repository()
        for signal in result.all_signals:
            repo.save_signal(signal)
        console.print(f"[green]Saved {len(result.all_signals)} signals to the database.[/green]")


@app.command()
def watch(
    symbols: str | None = typer.Option(None, "--symbols", "-s"),
    timeframe: str = typer.Option("1h", "--timeframe", "-t"),
    asset_class: str | None = typer.Option(None, "--class", "-c"),
) -> None:
    """One-line-per-instrument scan of the whole universe."""
    from mfie.interfaces.report import format_watchlist
    from mfie.pipeline.engine import AnalysisEngine

    with console.status("Scanning..."):
        result = AnalysisEngine().run(
            symbols=_parse_symbols(symbols),
            timeframe=timeframe,
            asset_class=_asset_class(asset_class),
        )
    console.print(format_watchlist(result))


@app.command()
def macro(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show data provenance"),
) -> None:
    """Print the current macro context: liquidity, curves, real yields, calendar."""
    from mfie.interfaces.report import format_macro_brief
    from mfie.pipeline.engine import AnalysisEngine

    engine = AnalysisEngine()
    with console.status("Fetching macro data..."):
        context = engine.build_macro_context(list(UNIVERSE.values()))
    console.print(format_macro_brief(context, verbose=verbose))

    table = Table(title="Real interest rates (policy rate - trailing CPI)")
    table.add_column("Currency")
    table.add_column("Policy rate", justify="right")
    table.add_column("Inflation", justify="right")
    table.add_column("Real rate", justify="right")
    for ccy in sorted(context.policy_rates):
        real = context.real_rate(ccy)
        table.add_row(
            ccy,
            f"{context.policy_rates[ccy]:.2%}",
            f"{context.inflation.get(ccy, float('nan')):.2%}",
            f"{real:+.2%}" if real is not None else "-",
        )
    console.print(table)


@app.command()
def cycle(
    domain: str = typer.Option("both", "--domain", "-d", help="crypto | fx | both"),
    validate: bool = typer.Option(False, "--validate",
                                  help="Measure whether the composite actually predicts returns"),
    factors: bool = typer.Option(True, "--factors/--no-factors", help="Show the factor breakdown"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Market Cycle Compass — the leading bull/bear state read."""
    from mfie.alpha.cycle import CycleEngine

    engine = CycleEngine()
    domains = ["crypto", "fx"] if domain == "both" else [domain]

    payload = {}
    for name in domains:
        with console.status(f"Building the {name} cycle panel..."):
            state = engine.evaluate(name)

        if as_json:
            payload[name] = state.as_dict()
            continue

        colour = {
            "expansion": "green", "early_recovery": "cyan",
            "late_expansion": "yellow", "contraction": "red", "neutral": "white",
        }[state.phase.value]

        console.print(
            Panel(
                "\n".join(state.narrative),
                title=f"[{colour}]{name.upper()} — {state.phase.label}  "
                      f"({state.score:+.2f})[/{colour}]",
                subtitle=f"confidence {state.confidence:.0%} · data {state.data_quality}",
            )
        )

        if factors and state.readings:
            table = Table(title="Factor breakdown")
            for column in ("Factor", "Score", "Weight", "Contribution", "Lead", "Trusted"):
                table.add_column(column, justify="right" if column != "Factor" else "left")
            for reading in state.readings:
                bar_colour = "green" if reading.score > 0 else "red"
                table.add_row(
                    reading.label,
                    f"[{bar_colour}]{reading.score:+.2f}[/{bar_colour}]",
                    f"{reading.weight:.0%}",
                    f"{reading.contribution:+.3f}",
                    f"{reading.lead_days}d",
                    "yes" if reading.trusted else "prior",
                )
            console.print(table)

        if validate:
            report = engine.validate(name)
            console.print(Panel("\n".join(report.lines()),
                                title=f"Validation — {name}"))

    if as_json:
        console.print_json(json.dumps(payload, default=str))


@app.command()
def backtest(
    symbol: str = typer.Argument(..., help="e.g. EURUSD or BTCUSDT"),
    timeframe: str = typer.Option("1h", "--timeframe", "-t"),
    bars: int = typer.Option(2000, "--bars", "-b"),
    warmup: int = typer.Option(250, "--warmup"),
    compare: bool = typer.Option(False, "--compare",
                                 help="Run with and without the macro filter chain"),
    save: bool = typer.Option(False, "--save", help="Persist trades and equity curve"),
) -> None:
    """Backtest the strategy library on one instrument."""
    from mfie.backtest.engine import BacktestEngine, compare_with_without_macro
    from mfie.backtest.metrics import compare_reports, evaluate_performance, strategy_attribution
    from mfie.data.hub import get_hub
    from mfie.pipeline.engine import AnalysisEngine

    instrument = get_instrument(symbol)
    hub = get_hub()
    df = hub.ohlcv(instrument, timeframe, bars)
    macro_context = AnalysisEngine(hub=hub).build_macro_context([instrument])

    source = hub.sources.get(f"ohlcv:{instrument.symbol}", "unknown")
    if source == "synthetic":
        console.print(
            "[yellow]Price data is synthetic (no live provider available). "
            "These results describe simulated data, not the market.[/yellow]"
        )

    if compare:
        with console.status("Running both configurations..."):
            results = compare_with_without_macro(instrument, df, macro_context, warmup=warmup)
        reports = {k: evaluate_performance(v) for k, v in results.items()}
        console.print(compare_reports(reports).to_string(float_format=lambda x: f"{x:,.4f}"))
        return

    with console.status(f"Backtesting {instrument.name} over {len(df)} bars..."):
        result = BacktestEngine().run(instrument, df, macro_context, warmup=warmup)
    report = evaluate_performance(result)

    console.print(Panel("\n".join(report.summary_lines()),
                        title=f"{instrument.name} {timeframe}"))
    attribution = strategy_attribution(result)
    if not attribution.empty:
        console.print("\nP&L by strategy:")
        console.print(attribution.to_string(float_format=lambda x: f"{x:,.2f}"))
    if report.block_reasons:
        console.print(f"\nBlocked by filter: {report.block_reasons}")

    if save:
        from mfie.storage.repo import Repository

        repo = Repository()
        run_id = repo.save_trades([t.as_row() for t in result.closed_trades], result.run_id)
        repo.save_equity_curve(run_id, result.equity)
        console.print(f"[green]Saved run {run_id}.[/green]")


@app.command()
def portfolio(
    symbols: str | None = typer.Option(None, "--symbols", "-s"),
    timeframe: str = typer.Option("1h", "--timeframe", "-t"),
    limit: int = typer.Option(500, "--limit", "-l"),
    asset_class: str | None = typer.Option(None, "--class", "-c", help="crypto | forex"),
    detail: bool = typer.Option(True, "--detail/--no-detail",
                                help="Show why each position was resized"),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
) -> None:
    """Allocate capital across the whole candidate set, correlations included.

    ``analyze`` sizes each signal on its own merits. This shows what the *book*
    can actually carry: which candidates are the same bet, which are dropped
    because their expected value does not survive costs, and how far the
    additive heat number overstates the risk being run.
    """
    from mfie.interfaces.report import format_portfolio_plan
    from mfie.pipeline.engine import AnalysisEngine

    engine = AnalysisEngine()
    if not engine.params.portfolio.enabled:
        console.print(
            "[yellow]Portfolio allocation is disabled in config/params.yaml "
            "(portfolio.enabled: false).[/yellow]"
        )
        raise typer.Exit(code=1)

    with console.status("Scoring candidates and allocating capital..."):
        result = engine.run(
            symbols=_parse_symbols(symbols),
            timeframe=timeframe,
            limit=limit,
            asset_class=_asset_class(asset_class),
        )

    plan = result.plan
    if plan is None or not plan.allocations:
        console.print("No candidate reached the portfolio layer.")
        return

    if as_json:
        payload = {
            "summary": plan.summary(),
            "positions": [
                {
                    "symbol": a.symbol,
                    "direction": a.signal.direction.value,
                    "strategy": a.signal.raw.strategy,
                    "standalone_risk": a.standalone_risk,
                    "allocated_risk": a.risk_fraction,
                    "units": a.units,
                    "cluster": a.cluster,
                    "risk_contribution": a.risk_contribution,
                    "dropped": a.dropped,
                    "reasons": a.reasons,
                    "edge": a.edge.as_dict() if a.edge else None,
                }
                for a in plan.allocations
            ],
        }
        console.print_json(json.dumps(payload, default=str))
        return

    console.print(format_portfolio_plan(plan, show_detail=detail))


@app.command()
def calibrate(
    symbol: str | None = typer.Option(None, "--symbol", "-s",
                                      help="Backtest this instrument to generate trades"),
    timeframe: str = typer.Option("1h", "--timeframe", "-t"),
    bars: int = typer.Option(3000, "--bars", "-b"),
    save: bool = typer.Option(False, "--save", help="Persist the trades for future runs"),
) -> None:
    """Measure whether the confidence score means anything.

    With ``--symbol`` it backtests that instrument and calibrates on the
    resulting trades. Without it, it reads whatever realised trades the
    database already holds. Either way the reliability table is the output that
    matters: it shows what actually happened at each confidence level, which is
    the check a confidence score never otherwise has to pass.
    """
    from mfie.portfolio.calibration import (
        ConfidenceCalibrator,
        load_calibrator,
        records_from_backtest,
    )

    if symbol:
        from mfie.backtest.engine import BacktestEngine
        from mfie.data.hub import get_hub
        from mfie.pipeline.engine import AnalysisEngine

        instrument = get_instrument(symbol)
        hub = get_hub()
        df = hub.ohlcv(instrument, timeframe, bars)
        macro_context = AnalysisEngine(hub=hub).build_macro_context([instrument])
        with console.status(f"Backtesting {instrument.name} over {len(df)} bars..."):
            result = BacktestEngine().run(instrument, df, macro_context)
        calibrator = ConfidenceCalibrator().fit(records_from_backtest(result))

        if hub.sources.get(f"ohlcv:{instrument.symbol}") == "synthetic":
            console.print(
                "[yellow]Price data is synthetic: this calibrates the engine "
                "against a random process, which is a code path test, not a "
                "measurement of edge.[/yellow]\n"
            )
        if save:
            from mfie.storage.repo import Repository

            repo = Repository()
            run_id = repo.save_trades([t.as_row() for t in result.closed_trades], result.run_id)
            console.print(f"[green]Saved run {run_id}.[/green]")
    else:
        calibrator = load_calibrator()

    summary = calibrator.summary()
    if not calibrator.fitted:
        console.print(
            "No realised trades available — the engine is running on its prior "
            "(0.35 + 0.25 x confidence). Run `mfie calibrate --symbol BTCUSDT --save` "
            "or `mfie backtest <symbol> --save` to give it something to learn from."
        )
        return

    table = Table(title=f"Reliability over {summary['observations']} realised trades")
    for column in ("Confidence", "Forecast", "Realised", "Trades", "Avg R"):
        table.add_column(column, justify="right")
    for _, row in calibrator.reliability_table().iterrows():
        table.add_row(
            str(row["confidence_range"]),
            f"{row['forecast']:.0%}",
            f"{row['realised']:.0%}",
            f"{int(row['trades'])}",
            f"{row['avg_r']:+.2f}",
        )
    console.print(table)

    skill = summary["skill"]
    verdict = (
        "the confidence score carries information"
        if skill > 0.01
        else "the confidence score is not beating its own base rate — "
             "treat it as an ordering, not a probability"
    )
    console.print(
        f"\nBrier {summary['brier']:.4f} vs base rate {summary['brier_baseline']:.4f} "
        f"(skill {skill:+.3f}) — {verdict}."
    )

    strategies = calibrator.strategy_table()
    if not strategies.empty:
        console.print("\nPer-strategy posteriors (shrunk toward the pooled rate):")
        console.print(strategies.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))


@app.command()
def pairs(
    asset_class: str = typer.Option("crypto", "--class", "-c", help="crypto | forex"),
    timeframe: str = typer.Option("1h", "--timeframe", "-t"),
    bars: int = typer.Option(1000, "--bars", "-b"),
    max_pvalue: float = typer.Option(0.05, "--max-pvalue"),
) -> None:
    """Scan for cointegrated pairs suitable for statistical arbitrage."""
    import pandas as pd

    from mfie.data.hub import get_hub
    from mfie.strategies.statarb import find_cointegrated_pairs

    instruments = resolve(None, _asset_class(asset_class))
    hub = get_hub()

    frames = {}
    with console.status(f"Loading {len(instruments)} instruments..."):
        for inst in instruments:
            df = hub.ohlcv(inst, timeframe, bars)
            if not df.empty:
                frames[inst.symbol] = df["close"]

    prices = pd.DataFrame(frames)
    with console.status("Testing for cointegration..."):
        found = find_cointegrated_pairs(prices, max_pvalue=max_pvalue)

    if not found:
        console.print("No cointegrated pairs found at this significance level.")
        return

    table = Table(title=f"Cointegrated pairs ({asset_class})")
    for column in ("Pair", "p-value", "Hedge ratio", "Half-life", "Spread z", "Corr"):
        table.add_column(column, justify="right" if column != "Pair" else "left")
    for row in found[:15]:
        table.add_row(
            f"{row['asset_a']} / {row['asset_b']}",
            f"{row['pvalue']:.4f}",
            f"{row['hedge_ratio']:.4f}",
            f"{row['half_life']:.1f}" if row["half_life"] == row["half_life"] else "-",
            f"{row['zscore']:+.2f}",
            f"{row['correlation']:.2f}",
        )
    console.print(table)
    console.print(
        f"[dim]{int(found[0]['n_tests'])} pairwise tests were run; at p<{max_pvalue} "
        f"roughly {found[0]['n_tests'] * max_pvalue:.0f} false positives are expected "
        "by chance alone.[/dim]"
    )


@app.command()
def train(
    symbol: str = typer.Argument(...),
    timeframe: str = typer.Option("1h", "--timeframe", "-t"),
    bars: int = typer.Option(2000, "--bars", "-b"),
    algorithm: str = typer.Option("gradient_boosting", "--algorithm", "-a",
                                  help="gradient_boosting | random_forest | svm"),
    horizon: int = typer.Option(12, "--horizon", help="Label horizon in bars"),
    save: bool = typer.Option(True, "--save/--no-save"),
) -> None:
    """Train and walk-forward validate the ML direction model."""
    from mfie.data.hub import get_hub
    from mfie.ml.model import DirectionModel
    from mfie.technical.indicators import compute_indicators

    instrument = get_instrument(symbol)
    df = get_hub().ohlcv(instrument, timeframe, bars)
    indicators = compute_indicators(df, get_params().technical)

    model = DirectionModel(horizon=horizon, algorithm=algorithm)
    model.threshold = float(max(indicators.last("atr_pct", 0.0) * 0.5, 0.0))

    with console.status(f"Training {algorithm} on {len(df)} bars..."):
        report = model.fit(indicators)

    table = Table(title=f"{instrument.name} {timeframe} — walk-forward validation")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Accuracy", f"{report.accuracy:.3f}")
    table.add_row("Majority baseline", f"{report.baseline:.3f}")
    table.add_row("Edge over baseline", f"{report.edge:+.3f}")
    table.add_row("ROC-AUC", f"{report.auc:.3f}")
    table.add_row("Folds", str(report.folds))
    table.add_row("Train / test rows", f"{report.n_train} / {report.n_test}")
    console.print(table)

    if report.feature_importance:
        console.print("\nTop features:")
        for name, importance in list(report.feature_importance.items())[:10]:
            console.print(f"  {name:<20} {importance:.4f}")

    if report.edge < 0.02:
        console.print(
            "[yellow]Edge is under 2 points over the baseline — this model would be "
            "rejected by the strategy at signal time.[/yellow]"
        )

    if save:
        path = model.save()
        console.print(f"[green]Saved model to {path}[/green]")


@app.command()
def ingest(
    symbols: str | None = typer.Option(None, "--symbols", "-s"),
    timeframe: str = typer.Option("1h", "--timeframe", "-t"),
    limit: int = typer.Option(500, "--limit", "-l"),
) -> None:
    """Fetch and persist market, macro and sentiment data to the database."""
    from mfie.data.hub import get_hub
    from mfie.pipeline.engine import AnalysisEngine
    from mfie.storage.repo import Repository

    hub = get_hub()
    repo = Repository()
    instruments = resolve(_parse_symbols(symbols))

    totals: dict[str, int] = {}
    with console.status("Ingesting instruments...") as status:
        for inst in instruments:
            status.update(f"Ingesting {inst.symbol}...")
            counts = repo.ingest_instrument(hub, inst, timeframe, limit)
            for key, value in counts.items():
                totals[key] = totals.get(key, 0) + value

        status.update("Ingesting macro series...")
        repo.save_macro_series("global_m2", hub.m2_series(), source="hub")
        repo.save_macro_series("stablecoin_supply", hub.stablecoin_supply(), source="hub")
        macro_context = AnalysisEngine(hub=hub).build_macro_context(instruments)
        for currency, series in macro_context.reer_history.items():
            repo.save_macro_series(f"reer_{currency}", series, source="hub", currency=currency)
        totals["events"] = repo.save_events(macro_context.events)

    table = Table(title="Rows written")
    table.add_column("Block")
    table.add_column("Rows", justify="right")
    for key, value in sorted(totals.items()):
        table.add_row(key, str(value))
    console.print(table)


@app.command(name="strategies")
def list_strategies() -> None:
    """List the strategy library."""
    from mfie.strategies.registry import describe_strategies

    table = Table(title="Strategies")
    for column in ("Name", "Family", "Assets", "Default", "Description"):
        table.add_column(column)
    for row in describe_strategies():
        table.add_row(
            row["name"], row["family"], row["asset_classes"], row["default"], row["description"]
        )
    console.print(table)


@app.command()
def doctor() -> None:
    """Check configuration, providers and database connectivity."""
    from mfie.data.hub import get_hub
    from mfie.storage.db import get_database

    settings = get_settings()
    hub = get_hub()

    table = Table(title="Data providers")
    table.add_column("Provider")
    table.add_column("Status")
    for name, available in hub.provider_status().items():
        table.add_row(name, "[green]configured[/green]" if available else "[yellow]missing key -> synthetic[/yellow]")
    console.print(table)

    console.print(f"\nDatabase URL: {settings.database_url}")
    try:
        db = get_database()
        console.print(
            f"Database: [green]reachable[/green] ({db.engine.dialect.name}"
            f"{', TimescaleDB-ready' if db.is_postgres else ''})"
        )
    except Exception as exc:
        console.print(f"Database: [red]unavailable[/red] — {exc}")

    console.print(f"Offline mode: {'[yellow]ON[/yellow]' if settings.offline else 'off'}")
    env_file = PROJECT_ROOT / ".env"
    console.print(
        f".env: {'found' if env_file.exists() else 'not found (copy .env.example to .env)'}"
    )


@app.command(name="init-db")
def init_db_command() -> None:
    """Create the database schema (and TimescaleDB hypertables on PostgreSQL)."""
    from mfie.storage.db import init_db

    db = init_db()
    console.print(f"[green]Schema created on {db.engine.dialect.name}.[/green]")
    if db.is_postgres:
        console.print("Hypertables and compression policies applied where TimescaleDB was available.")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h",
                             help="Bind address. Use 0.0.0.0 to expose on the network."),
    port: int = typer.Option(8000, "--port", "-p"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes (development)"),
    workers: int = typer.Option(1, "--workers", "-w",
                                help="Worker processes. Each keeps its own cache and engine."),
) -> None:
    """Serve the web interface and JSON API.

    This is the production interface: one process serves both the app at ``/``
    and the API under ``/api``. There is no build step and no second server.

    Binding to 0.0.0.0 exposes the interface to your whole network. There is no
    authentication in front of it, so put it behind a reverse proxy with auth,
    or a VPN, before doing that on anything but a trusted LAN.
    """
    try:
        import uvicorn
    except ImportError:
        console.print(
            "[red]uvicorn is not installed.[/red] Install the web extra:\n"
            "  pip install 'mfie[api]'   (or: pip install fastapi 'uvicorn[standard]')"
        )
        raise typer.Exit(code=1) from None

    if host == "0.0.0.0":  # noqa: S104 - deliberate, and warned about
        console.print(
            "[yellow]Binding to 0.0.0.0: the interface will be reachable from your "
            "network with no authentication in front of it.[/yellow]"
        )

    console.print(f"MFIE on [cyan]http://{'localhost' if host in {'127.0.0.1', '0.0.0.0'} else host}:{port}[/cyan]")
    console.print(f"API docs at [cyan]http://localhost:{port}/api/docs[/cyan]\n")

    # Workers require an import string rather than an app object, and reload is
    # incompatible with multiple workers in uvicorn.
    uvicorn.run(
        "mfie.api:app",
        host=host,
        port=port,
        reload=reload,
        workers=None if reload else (workers if workers > 1 else None),
        log_level="info",
    )


@app.command()
def dashboard(
    port: int = typer.Option(8501, "--port", "-p"),
) -> None:
    """Launch the legacy Streamlit dashboard (superseded by `serve`)."""
    console.print(
        "[yellow]`dashboard` is the old Streamlit prototype. `mfie serve` is the "
        "current interface — faster, deployable, and no Streamlit dependency.[/yellow]"
    )
    script = Path(__file__).parent / "interfaces" / "dashboard.py"
    console.print(f"Starting Streamlit dashboard on http://localhost:{port} ...")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(script), "--server.port", str(port)],
        check=False,
    )


@app.command()
def bot(
    interval: int = typer.Option(900, "--interval", "-i", help="Seconds between scans"),
    symbols: str | None = typer.Option(None, "--symbols", "-s"),
    once: bool = typer.Option(False, "--once", help="Send one report and exit"),
) -> None:
    """Run the Telegram alert bot."""
    from mfie.interfaces.telegram_bot import run_bot

    run_bot(symbols=_parse_symbols(symbols), interval=interval, once=once)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
