"""Human-readable rendering of analysis output.

The point of the whole engine is explainability, so these formatters always
show *why*: the macro state, the filters that fired, and the reason a signal was
blocked or shrunk — never a bare "BUY EURUSD".
"""

from __future__ import annotations

from datetime import datetime, timezone

from mfie.core.types import Direction, FilterAction, MacroContext, ScoredSignal

DISCLAIMER = (
    "Analysis only — not financial advice. Outputs are model estimates from public "
    "data and can be wrong. You are responsible for your own trading decisions."
)

_ARROW = {Direction.LONG: "LONG", Direction.SHORT: "SHORT", Direction.FLAT: "FLAT"}


def format_macro_brief(macro: MacroContext, verbose: bool = False) -> str:
    """One-screen summary of the macro state driving every filter this run."""
    lines = [
        "MACRO CONTEXT",
        f"  As of              {macro.ts:%Y-%m-%d %H:%M UTC}",
        f"  Liquidity regime   {macro.gli_regime.value.upper()} "
        f"(ΔGLI {macro.gli_delta:+.2%}; M2 {macro.m2_change:+.2%}, "
        f"stablecoins {macro.stablecoin_change:+.2%})",
        f"  Macro regime       {macro.macro_regime.value.upper()}",
    ]

    us_spread = macro.yield_spread("USD")
    if us_spread is not None:
        shape = "inverted" if us_spread <= 0 else "positive"
        lines.append(f"  US 10Y-2Y          {us_spread:+.2%} ({shape})")

    if macro.fear_greed is not None:
        lines.append(f"  Fear & Greed       {macro.fear_greed:.0f}/100")

    if macro.portfolio_cvar is not None:
        lines.append(f"  Portfolio CVaR95   {macro.portfolio_cvar:.2%}")

    real_rates = [
        (ccy, macro.real_rate(ccy))
        for ccy in sorted(macro.policy_rates)
        if macro.real_rate(ccy) is not None
    ]
    if real_rates:
        ranked = sorted(real_rates, key=lambda kv: kv[1], reverse=True)
        top = ", ".join(f"{c} {r:+.2%}" for c, r in ranked[:4])
        bottom = ", ".join(f"{c} {r:+.2%}" for c, r in ranked[-2:])
        lines.append(f"  Real yields (top)  {top}")
        lines.append(f"  Real yields (low)  {bottom}")

    upcoming = sorted(
        [e for e in macro.events if e.ts > macro.ts and e.impact == "high"],
        key=lambda e: e.ts,
    )[:3]
    if upcoming:
        lines.append("  Next high-impact releases:")
        for event in upcoming:
            minutes = (event.ts - macro.ts).total_seconds() / 60
            when = f"{minutes:.0f} min" if minutes < 180 else f"{minutes / 60:.1f} h"
            lines.append(f"    - {event.currency} {event.name} in {when}")

    if verbose and macro.sources:
        synthetic = sorted({k for k, v in macro.sources.items() if v == "synthetic"})
        live = sorted({v for v in macro.sources.values() if v != "synthetic"})
        lines.append(f"  Live sources       {', '.join(live) if live else 'none'}")
        if synthetic:
            lines.append(f"  Simulated blocks   {len(synthetic)} (no API key or request failed)")

    return "\n".join(lines)


def format_signal(signal: ScoredSignal, show_audit: bool = True) -> str:
    """Full render of one scored signal, including its audit trail."""
    raw = signal.raw
    inst = raw.instrument
    header = (
        f"{inst.name}  {_ARROW[raw.direction]}  [{raw.strategy}]  "
        f"{signal.verdict}  confidence {signal.confidence:.0%}"
    )
    lines = [header, "-" * len(header)]

    lines.append(f"  Timeframe        {raw.timeframe}   as of {raw.ts:%Y-%m-%d %H:%M UTC}")
    lines.append(f"  Entry            {raw.entry:,.5f}")
    stop = signal.adjusted_stop or raw.stop
    take_profit = signal.adjusted_take_profit or raw.take_profit
    stop_pct = abs(raw.entry - stop) / raw.entry if raw.entry else 0.0
    lines.append(f"  Stop             {stop:,.5f}  ({stop_pct:.2%} away)")
    if take_profit:
        rr = abs(take_profit - raw.entry) / max(abs(raw.entry - stop), 1e-12)
        lines.append(f"  Target           {take_profit:,.5f}  ({rr:.2f}R)")

    if signal.blocked:
        lines.append("  Position         BLOCKED")
        for reason in signal.block_reasons:
            lines.append(f"    ! {reason}")
    else:
        lines.append(
            f"  Position         {signal.risk_fraction:.2%} of equity at risk, "
            f"{signal.units:,.4f} units ({signal.size_fraction:.1%} notional)"
        )

    if raw.rationale:
        lines.append("  Setup:")
        for item in raw.rationale:
            lines.append(f"    - {item}")

    if show_audit and signal.outcomes:
        lines.append("  Macro filter chain:")
        for outcome in signal.outcomes:
            if outcome.action is FilterAction.PASS and outcome.reason == "not applicable":
                continue
            lines.append(f"    {outcome}")

    return "\n".join(lines)


def format_portfolio_plan(plan, show_detail: bool = True) -> str:
    """Render the book-level decision.

    The headline is the pair of numbers ``gross`` and ``effective``. When they
    are close the book is one bet under several names; when effective risk is
    far below gross, the diversification is real and the budget can carry it.
    """
    if plan is None:
        return ""

    held = plan.held
    lines = ["PORTFOLIO ALLOCATION"]

    if plan.book is not None:
        lines.append(f"  {plan.book.describe()}")
    lines.append(
        f"  Requested {plan.gross_requested:.2%} additive -> allocated "
        f"{plan.gross_risk:.2%} gross / {plan.effective_risk:.2%} correlated"
    )
    lines.append(
        f"  Hit rate from {plan.calibration_source} "
        f"({plan.calibration_observations} realised trades); "
        f"book expected value {plan.expected_r:+.2f}R per unit of risk"
    )

    for note in plan.notes:
        lines.append(f"  * {note}")

    if held:
        lines.append("")
        lines.append(
            f"  {'INSTRUMENT':<12}{'DIR':<6}{'STRATEGY':<22}"
            f"{'STANDALONE':>11}{'ALLOCATED':>11}{'OF BOOK':>9}  CLUSTER"
        )
        for allocation in sorted(held, key=lambda a: a.risk_fraction, reverse=True):
            signal = allocation.signal
            lines.append(
                f"  {signal.instrument.name:<12}{_ARROW[signal.direction]:<6}"
                f"{signal.raw.strategy:<22}"
                f"{allocation.standalone_risk:>10.2%} {allocation.risk_fraction:>10.2%} "
                f"{allocation.risk_contribution:>8.0%}  #{allocation.cluster}"
            )
            if show_detail:
                for reason in allocation.reasons:
                    lines.append(f"      - {reason}")
                if allocation.edge is not None:
                    lines.append(f"      - {allocation.edge.describe()}")
    else:
        lines.append("  No position survived allocation.")

    dropped = plan.dropped
    if dropped and show_detail:
        lines.append("")
        lines.append("  DROPPED BY THE PORTFOLIO LAYER")
        for allocation in dropped:
            reason = allocation.reasons[0] if allocation.reasons else "no reason recorded"
            lines.append(
                f"    {allocation.signal.instrument.name:<12}"
                f"{allocation.signal.raw.strategy:<22}{reason}"
            )

    return "\n".join(lines)


def format_result(result, show_audit: bool = True, include_blocked: bool = True) -> str:
    """Render a whole ``AnalysisResult``."""
    parts = [format_macro_brief(result.macro, verbose=True), ""]

    tradable = result.tradable
    blocked = result.blocked

    parts.append(
        f"SIGNALS  ({len(tradable)} actionable, {len(blocked)} blocked, "
        f"{len(result.all_signals)} generated across {len(result.analyses)} instruments)"
    )
    parts.append("")

    if tradable:
        for signal in tradable:
            parts.append(format_signal(signal, show_audit))
            parts.append("")
    else:
        parts.append("  No signal cleared the filter chain with a tradable size.")
        parts.append("")

    plan = getattr(result, "plan", None)
    if plan is not None and plan.allocations:
        parts.append(format_portfolio_plan(plan, show_detail=show_audit))
        parts.append("")

    # Signals that no rule vetoed but that sized to nothing. Without this
    # section they vanish from the report entirely and the totals stop adding up.
    below_threshold = [
        s for s in result.all_signals
        if not s.blocked and s.risk_fraction <= 0
    ]
    if below_threshold:
        parts.append("BELOW SIZING THRESHOLD (passed the filters, too weak to size)")
        for signal in sorted(below_threshold, key=lambda s: s.confidence, reverse=True):
            # The *last* note is the decisive one — sizing appends notes as it
            # goes and returns immediately on whichever constraint bites.
            reason = signal.block_reasons[-1] if signal.block_reasons else "no size allocated"
            parts.append(
                f"  {signal.instrument.name:<10} {_ARROW[signal.direction]:<5} "
                f"{signal.raw.strategy:<20} {signal.confidence:>4.0%}  {reason}"
            )
        parts.append("")

    if include_blocked and blocked:
        parts.append("BLOCKED SIGNALS")
        for signal in blocked:
            reason = signal.block_reasons[0] if signal.block_reasons else "unknown"
            parts.append(
                f"  {signal.instrument.name:<10} {_ARROW[signal.direction]:<5} "
                f"{signal.raw.strategy:<20} {reason}"
            )
        parts.append("")

    quiet = [
        analysis.instrument.name
        for analysis in result.analyses.values()
        if not analysis.signals and analysis.error is None
    ]
    if quiet:
        parts.append(f"No setup found: {', '.join(quiet)}")

    errors = [
        f"{a.instrument.name} ({a.error})"
        for a in result.analyses.values()
        if a.error
    ]
    if errors:
        parts.append(f"Skipped: {', '.join(errors)}")

    parts.append("")
    parts.append(DISCLAIMER)
    return "\n".join(parts)


def signal_to_telegram(signal: ScoredSignal, macro: MacroContext | None = None) -> str:
    """Compact Markdown message for the Telegram bot."""
    raw = signal.raw
    emoji = "🟢" if raw.direction is Direction.LONG else "🔴"
    if signal.blocked:
        emoji = "⛔"

    lines = [
        f"{emoji} *{raw.instrument.name}* — {_ARROW[raw.direction]} ({raw.strategy})",
        f"Verdict: *{signal.verdict}* · confidence {signal.confidence:.0%}",
    ]

    if signal.blocked:
        lines.append("")
        lines.append("*Blocked:*")
        for reason in signal.block_reasons[:3]:
            lines.append(f"• {reason}")
    else:
        stop = signal.adjusted_stop or raw.stop
        take_profit = signal.adjusted_take_profit or raw.take_profit
        lines.append("")
        lines.append(f"Entry `{raw.entry:,.5f}`  ·  Stop `{stop:,.5f}`")
        if take_profit:
            lines.append(f"Target `{take_profit:,.5f}`")
        lines.append(f"Risk {signal.risk_fraction:.2%} of equity · {signal.units:,.4f} units")

    notable = [
        o for o in signal.outcomes
        if o.action in (FilterAction.PENALIZE, FilterAction.BOOST, FilterAction.BLOCK)
    ]
    if notable:
        lines.append("")
        lines.append("*Macro filters:*")
        for outcome in notable[:4]:
            lines.append(f"• {outcome.name}: {outcome.reason}")

    if macro is not None:
        lines.append("")
        lines.append(
            f"_Liquidity {macro.gli_regime.value} (ΔGLI {macro.gli_delta:+.2%}) · "
            f"macro {macro.macro_regime.value}_"
        )

    lines.append("")
    lines.append(f"_{DISCLAIMER}_")
    return "\n".join(lines)


def format_watchlist(result) -> str:
    """One line per instrument — the quick scan view."""
    rows = ["SYMBOL      PRICE          REGIME              BEST SIGNAL"]
    for symbol, analysis in sorted(result.analyses.items()):
        if analysis.error:
            rows.append(f"{symbol:<11} {'-':<14} {analysis.error}")
            continue
        best = analysis.best
        regime = analysis.regime.regime.value if analysis.regime else "unknown"
        if best is None:
            verdict = "no setup"
        else:
            verdict = (
                f"{_ARROW[best.direction]} {best.raw.strategy} "
                f"({best.confidence:.0%})"
            )
        rows.append(f"{symbol:<11} {analysis.price:<14,.5f} {regime:<19} {verdict}")
    return "\n".join(rows)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
