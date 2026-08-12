"""Streamlit dashboard.

    python -m mfie dashboard        (or: streamlit run mfie/interfaces/dashboard.py)

Six tabs: Signals, Cycle, Macro, Charts, Backtest, Data sources. Every chart
carries a legend or title naming its series and a matching table view, so
nothing depends on colour alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `streamlit run mfie/interfaces/dashboard.py` from a clean environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from mfie.config import get_params, get_settings  # noqa: E402
from mfie.core.types import Direction, FilterAction  # noqa: E402
from mfie.core.universe import CRYPTO, FOREX, UNIVERSE, get_instrument  # noqa: E402
from mfie.core.utils import configure_console  # noqa: E402
from mfie.interfaces.report import DISCLAIMER, format_macro_brief  # noqa: E402
from mfie.interfaces.theme import (  # noqa: E402
    base_layout,
    get_palette,
    line_style,
    status_color,
    verdict_status,
)

configure_console()

st.set_page_config(
    page_title="MFIE — Macro-Informed Market Intelligence",
    page_icon="📊",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Data plumbing
# --------------------------------------------------------------------------- #
@st.cache_resource
def _engine():
    from mfie.pipeline.engine import AnalysisEngine

    return AnalysisEngine()


@st.cache_data(ttl=300, show_spinner=False)
def run_analysis(symbols: tuple[str, ...], timeframe: str, limit: int):
    return _engine().run(symbols=list(symbols), timeframe=timeframe, limit=limit)


@st.cache_data(ttl=900, show_spinner=False)
def load_ohlcv(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    from mfie.data.hub import get_hub

    return get_hub().ohlcv(get_instrument(symbol), timeframe, limit)


def theme_mode() -> str:
    try:
        base = st.get_option("theme.base")
    except Exception:
        base = None
    return "light" if base == "light" else "dark"


PALETTE = get_palette(theme_mode())


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def price_chart(df: pd.DataFrame, indicators, title: str, signal=None) -> go.Figure:
    """Candlesticks with three overlays — the all-pairs-safe categorical cap."""
    ind = indicators.df
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            name="Price",
            increasing=dict(line=dict(color=PALETTE.good, width=1), fillcolor=PALETTE.good),
            decreasing=dict(line=dict(color=PALETTE.critical, width=1), fillcolor=PALETTE.critical),
        )
    )

    for slot, (column, label, dash) in enumerate(
        [("ema_fast", "EMA fast", None), ("ema_slow", "EMA slow", None),
         ("ema_trend", "EMA trend", "dot")]
    ):
        if column in ind.columns:
            fig.add_trace(
                go.Scatter(
                    x=ind.index, y=ind[column], name=label, mode="lines",
                    line=line_style(PALETTE, slot, dash),
                )
            )

    if signal is not None:
        entry = signal.raw.entry
        stop = signal.adjusted_stop or signal.raw.stop
        target = signal.adjusted_take_profit or signal.raw.take_profit
        for value, label, colour in (
            (entry, "Entry", PALETTE.ink_secondary),
            (stop, "Stop", PALETTE.critical),
            (target, "Target", PALETTE.good),
        ):
            if value:
                fig.add_hline(
                    y=value, line=dict(color=colour, width=1, dash="dash"),
                    annotation_text=f"{label} {value:,.5f}",
                    annotation_position="right",
                    annotation_font=dict(color=PALETTE.ink_secondary, size=11),
                )

    layout = base_layout(PALETTE, height=520, title=title)
    layout["xaxis"]["rangeslider"] = {"visible": False}
    fig.update_layout(**{k: v for k, v in layout.items() if v is not None})
    return fig


def oscillator_chart(indicators, title: str = "Momentum") -> go.Figure:
    ind = indicators.df
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=ind.index, y=ind["rsi"], name="RSI (14)", mode="lines",
                   line=line_style(PALETTE, 0))
    )
    for level, label in ((70, "Overbought"), (30, "Oversold")):
        fig.add_hline(
            y=level, line=dict(color=PALETTE.axis, width=1, dash="dot"),
            annotation_text=label, annotation_position="right",
            annotation_font=dict(color=PALETTE.ink_muted, size=10),
        )
    layout = base_layout(PALETTE, height=220, title=title)
    layout["yaxis"]["range"] = [0, 100]
    fig.update_layout(**{k: v for k, v in layout.items() if v is not None})
    return fig


def filter_impact_chart(signal) -> go.Figure:
    """Diverging bars: how each filter moved the signal's confidence.

    Polarity data (helped / hurt) gets the diverging pair with a neutral
    midpoint — not two arbitrary categorical hues.
    """
    rows = [
        (o.name, (o.multiplier - 1.0) * 100.0, o.reason)
        for o in signal.outcomes
        if not (o.action is FilterAction.PASS and o.reason == "not applicable")
    ]
    if not rows:
        return go.Figure()

    names = [r[0] for r in rows]
    impacts = [r[1] for r in rows]
    reasons = [r[2] for r in rows]
    colours = [PALETTE.diverging_high if v >= 0 else PALETTE.diverging_low for v in impacts]

    fig = go.Figure(
        go.Bar(
            x=impacts, y=names, orientation="h",
            marker=dict(color=colours, line=dict(color=PALETTE.surface, width=2)),
            customdata=reasons,
            hovertemplate="<b>%{y}</b><br>impact %{x:+.1f}%<br>%{customdata}<extra></extra>",
            name="Confidence impact",
            text=[f"{v:+.0f}%" for v in impacts],
            textposition="outside",
            textfont=dict(color=PALETTE.ink_secondary, size=11),
        )
    )
    layout = base_layout(PALETTE, height=max(240, 34 * len(rows)),
                         title="Filter impact on confidence")
    layout["showlegend"] = False
    layout["hovermode"] = "closest"
    layout["xaxis"]["zeroline"] = True
    layout["xaxis"]["zerolinecolor"] = PALETTE.axis
    fig.update_layout(**{k: v for k, v in layout.items() if v is not None})
    return fig


def cycle_chart(history: pd.DataFrame, params, title: str) -> go.Figure:
    """Composite cycle score over time.

    One series, so no legend box — the title names it. The reference lines carry
    the thresholds that drive the state machine, so the reader can see *why* a
    phase changed where it did rather than being told.
    """
    fig = go.Figure(
        go.Scatter(
            x=history.index, y=history["cycle"], mode="lines", name="Cycle score",
            line=line_style(PALETTE, 0),
            hovertemplate="%{x|%Y-%m-%d}<br>score %{y:+.3f}<extra></extra>",
        )
    )
    for level, label, colour in (
        (params.bull_entry, "Bull entry", PALETTE.good),
        (0.0, "", PALETTE.axis),
        (params.bear_entry, "Bear entry", PALETTE.critical),
    ):
        fig.add_hline(
            y=level,
            line=dict(color=colour, width=1, dash="dot" if label else "solid"),
            annotation_text=label,
            annotation_position="right",
            annotation_font=dict(color=PALETTE.ink_muted, size=10),
        )
    layout = base_layout(PALETTE, height=340, title=title)
    layout["showlegend"] = False
    layout["yaxis"]["range"] = [-1.05, 1.05]
    fig.update_layout(**{k: v for k, v in layout.items() if v is not None})
    return fig


def factor_contribution_chart(state) -> go.Figure:
    """Diverging bars: how much each factor pushes the composite, and which way."""
    readings = list(reversed(state.readings))
    if not readings:
        return go.Figure()

    values = [r.contribution for r in readings]
    colours = [PALETTE.diverging_high if v >= 0 else PALETTE.diverging_low for v in values]
    hover = [
        f"score {r.score:+.2f} · weight {r.weight:.0%} · lead ~{r.lead_days}d"
        f"{'' if r.trusted else ' (prior weight)'}"
        for r in readings
    ]

    fig = go.Figure(
        go.Bar(
            x=values, y=[r.label for r in readings], orientation="h",
            marker=dict(color=colours, line=dict(color=PALETTE.surface, width=2)),
            customdata=hover,
            hovertemplate="<b>%{y}</b><br>contribution %{x:+.3f}<br>%{customdata}<extra></extra>",
            text=[f"{v:+.3f}" for v in values],
            textposition="outside",
            textfont=dict(color=PALETTE.ink_secondary, size=11),
            name="Contribution",
        )
    )
    layout = base_layout(PALETTE, height=max(280, 38 * len(readings)),
                         title="What is driving the cycle score")
    layout["showlegend"] = False
    layout["hovermode"] = "closest"
    layout["xaxis"]["zeroline"] = True
    layout["xaxis"]["zerolinecolor"] = PALETTE.axis
    fig.update_layout(**{k: v for k, v in layout.items() if v is not None})
    return fig


def equity_chart(equity: pd.Series, title: str = "Equity curve") -> go.Figure:
    fig = go.Figure(
        go.Scatter(x=equity.index, y=equity.values, mode="lines", name="Equity",
                   line=line_style(PALETTE, 0))
    )
    layout = base_layout(PALETTE, height=360, title=title)
    layout["showlegend"] = False
    fig.update_layout(**{k: v for k, v in layout.items() if v is not None})
    return fig


def real_rate_chart(macro) -> go.Figure:
    rows = [
        (ccy, macro.real_rate(ccy) * 100)
        for ccy in sorted(macro.policy_rates)
        if macro.real_rate(ccy) is not None
    ]
    rows.sort(key=lambda kv: kv[1], reverse=True)
    if not rows:
        return go.Figure()

    names = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colours = [PALETTE.diverging_high if v >= 0 else PALETTE.diverging_low for v in values]

    fig = go.Figure(
        go.Bar(
            x=names, y=values,
            marker=dict(color=colours, line=dict(color=PALETTE.surface, width=2)),
            text=[f"{v:+.2f}%" for v in values], textposition="outside",
            textfont=dict(color=PALETTE.ink_secondary, size=11),
            hovertemplate="<b>%{x}</b><br>real rate %{y:+.2f}%<extra></extra>",
            name="Real rate",
        )
    )
    layout = base_layout(PALETTE, height=320,
                         title="Real interest rates (policy rate − trailing CPI)")
    layout["showlegend"] = False
    layout["hovermode"] = "closest"
    layout["yaxis"]["zeroline"] = True
    layout["yaxis"]["zerolinecolor"] = PALETTE.axis
    fig.update_layout(**{k: v for k, v in layout.items() if v is not None})
    return fig


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def sidebar() -> dict:
    st.sidebar.title("MFIE")
    st.sidebar.caption("Macro-Informed Financial Intelligence Engine")

    settings = get_settings()
    params = get_params()

    asset_class = st.sidebar.radio("Asset class", ["Crypto", "Forex", "Both"], index=2,
                                   horizontal=True)
    if asset_class == "Crypto":
        options = list(CRYPTO)
        default = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    elif asset_class == "Forex":
        options = list(FOREX)
        default = ["EURUSD", "GBPUSD", "USDJPY"]
    else:
        options = list(UNIVERSE)
        default = ["BTCUSDT", "ETHUSDT", "EURUSD", "GBPUSD"]

    symbols = st.sidebar.multiselect("Instruments", options, default=default)
    timeframe = st.sidebar.selectbox("Timeframe", ["15m", "30m", "1h", "4h", "1d"], index=2)
    limit = st.sidebar.slider("Bars of history", 200, 1500, 500, step=100)

    st.sidebar.divider()
    st.sidebar.subheader("Risk")
    equity = st.sidebar.number_input("Account equity", min_value=100.0,
                                     value=float(params.risk.account_equity), step=500.0)
    base_risk = st.sidebar.slider("Base risk per trade", 0.1, 3.0,
                                  float(params.risk.base_risk_per_trade * 100), step=0.1) / 100
    params.risk.account_equity = equity
    params.risk.base_risk_per_trade = base_risk

    st.sidebar.divider()
    if settings.offline:
        st.sidebar.warning("Offline mode — all data is simulated.")
    if st.sidebar.button("Refresh data", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    return {"symbols": tuple(symbols), "timeframe": timeframe, "limit": limit}


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
def tab_signals(result) -> None:
    macro = result.macro
    cols = st.columns(4)
    cols[0].metric("Liquidity regime", macro.gli_regime.value.title(), f"ΔGLI {macro.gli_delta:+.2%}")
    cols[1].metric("Macro regime", macro.macro_regime.value.title(),
                   f"US 10Y-2Y {macro.yield_spread('USD'):+.2%}" if macro.yield_spread("USD") else "")
    cols[2].metric("Fear & Greed", f"{macro.fear_greed:.0f}" if macro.fear_greed else "n/a")
    cols[3].metric("Actionable signals", len(result.tradable), f"{len(result.blocked)} blocked")

    st.divider()

    tradable = result.tradable
    if not tradable:
        st.info("No signal cleared the filter chain with a tradable size.")
    for signal in tradable:
        raw = signal.raw
        colour = status_color(PALETTE, verdict_status(signal.verdict))
        arrow = "▲" if raw.direction is Direction.LONG else "▼"
        with st.expander(
            f"{arrow} {raw.instrument.name} · {raw.direction.value.upper()} · "
            f"{raw.strategy} · {signal.verdict} ({signal.confidence:.0%})",
            expanded=(signal is tradable[0]),
        ):
            top = st.columns(4)
            top[0].metric("Confidence", f"{signal.confidence:.0%}")
            top[1].metric("Entry", f"{raw.entry:,.5f}")
            top[2].metric("Stop", f"{signal.adjusted_stop:,.5f}")
            top[3].metric("Risk", f"{signal.risk_fraction:.2%} of equity",
                          f"{signal.units:,.4f} units")

            st.markdown(
                f"<div style='height:3px;background:{colour};border-radius:2px;"
                "margin:4px 0 12px 0'></div>",
                unsafe_allow_html=True,
            )

            left, right = st.columns([1, 1])
            with left:
                st.markdown("**Setup**")
                for item in raw.rationale:
                    st.markdown(f"- {item}")
            with right:
                st.plotly_chart(filter_impact_chart(signal), width="stretch")

            # Table view of the same numbers — the relief rule for low-contrast
            # slots, and the accessible alternative to reading colour.
            st.markdown("**Filter chain**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Filter": o.name,
                            "Action": o.action.value,
                            "Multiplier": round(o.multiplier, 3),
                            "Reason": o.reason,
                        }
                        for o in signal.outcomes
                        if not (o.action is FilterAction.PASS and o.reason == "not applicable")
                    ]
                ),
                width="stretch",
                hide_index=True,
            )

    if result.blocked:
        st.subheader("Blocked signals")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Symbol": s.instrument.name,
                        "Direction": s.direction.value,
                        "Strategy": s.raw.strategy,
                        "Blocked by": s.block_reasons[0] if s.block_reasons else "unknown",
                    }
                    for s in result.blocked
                ]
            ),
            width="stretch",
            hide_index=True,
        )


def tab_macro(result) -> None:
    macro = result.macro
    st.code(format_macro_brief(macro, verbose=True), language=None)

    st.plotly_chart(real_rate_chart(macro), width="stretch")

    st.subheader("Rates and curves")
    rows = []
    for ccy in sorted(macro.policy_rates):
        rows.append(
            {
                "Currency": ccy,
                "Policy rate": macro.policy_rates.get(ccy),
                "Inflation": macro.inflation.get(ccy),
                "Real rate": macro.real_rate(ccy),
                "10Y": macro.yield_10y.get(ccy),
                "2Y": macro.yield_2y.get(ccy),
                "10Y-2Y": macro.yield_spread(ccy),
                "ESI": macro.esi.get(ccy, 0.0),
            }
        )
    st.dataframe(
        pd.DataFrame(rows).style.format(
            {c: "{:.2%}" for c in
             ("Policy rate", "Inflation", "Real rate", "10Y", "2Y", "10Y-2Y")}
            | {"ESI": "{:+.2f}"}
        ),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Economic calendar")
    upcoming = sorted([e for e in macro.events if e.ts >= macro.ts], key=lambda e: e.ts)[:25]
    if upcoming:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "When (UTC)": e.ts.strftime("%Y-%m-%d %H:%M"),
                        "In": f"{(e.ts - macro.ts).total_seconds() / 3600:.1f} h",
                        "Currency": e.currency,
                        "Event": e.name,
                        "Impact": e.impact,
                        "Forecast": e.forecast,
                        "Previous": e.previous,
                    }
                    for e in upcoming
                ]
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.caption("No upcoming events in the loaded window.")

    if macro.token_velocity:
        st.subheader("Crypto network economics")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Symbol": symbol,
                        "Velocity": macro.token_velocity.get(symbol),
                        "Velocity z": macro.token_velocity_z.get(symbol),
                        "NVT": macro.nvt.get(symbol),
                        "NVT z": macro.nvt_z.get(symbol),
                    }
                    for symbol in macro.token_velocity
                ]
            ),
            width="stretch",
            hide_index=True,
        )


def tab_cycle(result) -> None:
    """Market Cycle Compass — the leading bull/bear read."""
    states = result.macro.cycle_states
    if not states:
        st.info("The cycle compass did not run for this selection.")
        return

    params = get_params().cycle
    st.caption(
        "A composite of liquidity, credit, real rates, breadth, valuation and "
        "positioning — each entering as an impulse, weighted by its measured "
        "lead over price, and shrunk toward its economic prior when the data "
        "cannot justify more."
    )

    for domain, state in states.items():
        st.subheader(f"{domain.upper()} — {state.phase.label}")

        colour = status_color(
            PALETTE,
            {"expansion": "good", "early_recovery": "good",
             "late_expansion": "warning", "contraction": "critical",
             "neutral": "serious"}[state.phase.value],
        )
        st.markdown(
            f"<div style='height:3px;background:{colour};border-radius:2px;"
            "margin:2px 0 14px 0'></div>",
            unsafe_allow_html=True,
        )

        cols = st.columns(5)
        cols[0].metric("Cycle score", f"{state.score:+.2f}",
                       f"{state.momentum:+.2f} momentum")
        cols[1].metric("Bias", state.bias)
        cols[2].metric("Days in phase", state.days_in_phase)
        cols[3].metric("P(phase change)", f"{state.transition_probability:.0%}",
                       f"within {params.hazard_horizon}d")
        cols[4].metric("Confidence", f"{state.confidence:.0%}", state.data_quality)

        st.info(f"**{state.phase.label}** — {state.phase.stance}")

        if state.divergence_flag:
            message = (
                f"Bearish divergence ({state.divergence:+.2f} sd): price is rising while "
                "macro internals deteriorate — the distribution signature."
                if state.divergence < 0 else
                f"Bullish divergence ({state.divergence:+.2f} sd): internals improving while "
                "price still falls — historically an accumulation window."
            )
            (st.warning if state.divergence < 0 else st.success)(message)

        if state.history is not None and not state.history.empty:
            st.plotly_chart(
                cycle_chart(state.history, params, f"{domain.upper()} cycle score"),
                width="stretch",
            )

        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(factor_contribution_chart(state), width="stretch")
        with right:
            # Table view of the same numbers: the accessible alternative to
            # reading the bars, and the relief rule for low-contrast slots.
            st.markdown("**Factors**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Factor": r.label,
                            "Score": round(r.score, 3),
                            "Weight": f"{r.weight:.0%}",
                            "Contribution": round(r.contribution, 4),
                            "Lead (d)": r.lead_days,
                            "Weight source": "measured" if r.trusted else "prior",
                        }
                        for r in state.readings
                    ]
                ),
                width="stretch",
                hide_index=True,
            )

        with st.expander(f"Why — {domain} narrative and factor rationale"):
            for line in state.narrative:
                st.markdown(f"- {line}")
            st.markdown("---")
            for reading in state.readings:
                st.markdown(f"**{reading.label}** — {reading.rationale}")

        st.divider()


def tab_charts(result, config: dict) -> None:
    symbols = list(result.analyses)
    if not symbols:
        st.info("Select at least one instrument.")
        return
    symbol = st.selectbox("Instrument", symbols)
    analysis = result.analyses[symbol]
    if analysis.error or analysis.indicators is None:
        st.warning(f"No data: {analysis.error}")
        return

    regime = analysis.regime
    cols = st.columns(5)
    cols[0].metric("Price", f"{analysis.price:,.5f}")
    cols[1].metric("Regime", regime.regime.value.replace("_", " ").title(),
                   f"{regime.confidence:.0%} confidence")
    cols[2].metric("ADX", f"{regime.adx:.1f}")
    cols[3].metric("Hurst", f"{regime.hurst:.2f}",
                   "trending" if regime.hurst > 0.55 else "mean-reverting")
    cols[4].metric("Realised vol", f"{regime.realized_vol:.1%}")

    st.plotly_chart(
        price_chart(analysis.df.tail(300), analysis.indicators,
                    f"{analysis.instrument.name} · {config['timeframe']}", analysis.best),
        width="stretch",
    )
    st.plotly_chart(oscillator_chart(analysis.indicators), width="stretch")

    with st.expander("Indicator values (latest bar)"):
        ind = analysis.indicators.df.iloc[-1]
        interesting = [
            "close", "rsi", "macd", "macd_signal", "macd_hist", "atr", "atr_pct",
            "adx", "plus_di", "minus_di", "bb_percent_b", "bb_bandwidth",
            "zscore", "volume_z", "mfi", "realized_vol",
        ]
        st.dataframe(
            pd.DataFrame(
                [{"Indicator": k, "Value": float(ind[k])} for k in interesting if k in ind],
            ),
            width="stretch",
            hide_index=True,
        )


def tab_backtest(config: dict) -> None:
    from mfie.backtest.engine import BacktestEngine, compare_with_without_macro
    from mfie.backtest.metrics import (
        compare_reports,
        evaluate_performance,
        strategy_attribution,
    )

    st.caption(
        "Bar-close decisions, next-bar fills, costs on both sides, stop assumed "
        "to fill before target when a bar spans both."
    )

    left, right, third = st.columns(3)
    symbol = left.selectbox("Instrument", list(UNIVERSE), key="bt_symbol")
    bars = right.slider("Bars", 500, 5000, 2000, step=250)
    compare = third.checkbox("Compare with / without macro filters")

    if not st.button("Run backtest", type="primary"):
        return

    instrument = get_instrument(symbol)
    df = load_ohlcv(symbol, config["timeframe"], bars)
    macro = _engine().build_macro_context([instrument])

    with st.spinner("Running..."):
        if compare:
            results = compare_with_without_macro(instrument, df, macro)
            reports = {k: evaluate_performance(v) for k, v in results.items()}
            st.dataframe(compare_reports(reports), width="stretch")
            st.plotly_chart(
                equity_chart(results["macro_filtered"].equity, "Equity — macro filtered"),
                width="stretch",
            )
            return

        result = BacktestEngine().run(instrument, df, macro)

    report = evaluate_performance(result)
    cols = st.columns(5)
    cols[0].metric("Total return", f"{report.total_return:.2%}")
    cols[1].metric("Sortino", f"{report.sortino:.2f}")
    cols[2].metric("Max drawdown", f"{report.max_drawdown:.2%}")
    cols[3].metric("Trades", report.trades)
    cols[4].metric("Win rate", f"{report.win_rate:.1%}")

    st.plotly_chart(equity_chart(result.equity), width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("**Performance**")
        st.code("\n".join(report.summary_lines()), language=None)
    with right:
        st.markdown("**P&L by strategy**")
        attribution = strategy_attribution(result)
        if attribution.empty:
            st.caption("No closed trades.")
        else:
            st.dataframe(attribution, width="stretch")

    if not result.trade_frame.empty:
        with st.expander("Trades"):
            st.dataframe(result.trade_frame, width="stretch", hide_index=True)


def tab_sources(result) -> None:
    from mfie.data.hub import get_hub

    st.subheader("Provider configuration")
    status = get_hub().provider_status()
    st.dataframe(
        pd.DataFrame(
            [
                {"Provider": name, "Status": "configured" if ok else "no key — simulated"}
                for name, ok in status.items()
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    st.subheader("What served this run")
    sources = result.macro.sources
    if sources:
        st.dataframe(
            pd.DataFrame(
                [{"Data block": k, "Source": v} for k, v in sorted(sources.items())]
            ),
            width="stretch",
            hide_index=True,
        )
    simulated = sum(1 for v in sources.values() if v == "synthetic")
    if simulated:
        st.warning(
            f"{simulated} of {len(sources)} data blocks were simulated because no API key "
            "was configured or the request failed. Add keys to `.env` for live data."
        )


# --------------------------------------------------------------------------- #
def main() -> None:
    config = sidebar()
    st.title("Macro-Informed Market Intelligence")

    if not config["symbols"]:
        st.info("Pick at least one instrument in the sidebar.")
        return

    with st.spinner("Building macro context and scanning..."):
        result = run_analysis(config["symbols"], config["timeframe"], config["limit"])

    tabs = st.tabs(["Signals", "Cycle", "Macro", "Charts", "Backtest", "Data sources"])
    with tabs[0]:
        tab_signals(result)
    with tabs[1]:
        tab_cycle(result)
    with tabs[2]:
        tab_macro(result)
    with tabs[3]:
        tab_charts(result, config)
    with tabs[4]:
        tab_backtest(config)
    with tabs[5]:
        tab_sources(result)

    st.divider()
    st.caption(DISCLAIMER)


main()
