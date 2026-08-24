"""Domain objects to JSON.

Kept apart from the route handlers on purpose. The engine's dataclasses carry
pandas frames, enums, numpy scalars and back-references to a shared
``MacroContext`` — none of which survive ``json.dumps`` — and burying that
conversion inside the routes would mean every new endpoint reinvents it and
gets a slightly different shape.

Two rules hold throughout:

* **Numbers are numbers.** No pre-formatted ``"1.25%"`` strings. Formatting is
  a presentation decision, and baking it into the payload makes the API useless
  to anything but this one UI — and makes locale support impossible later.
* **Every derived verdict travels with its inputs.** A signal ships its filter
  outcomes, a position ships the reasons it was resized. The interface must be
  able to show *why* without a second request, because the audit trail is the
  product.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


def clean(value: Any) -> Any:
    """Make a value JSON-safe, mapping non-finite floats to ``None``.

    NaN and infinity are legal in numpy and illegal in JSON. Emitting them
    produces a payload that ``JSON.parse`` rejects outright, so they become
    ``null`` and the interface renders them as "n/a" rather than crashing.
    """
    if value is None:
        return None
    if isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime().isoformat()
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    if hasattr(value, "value") and hasattr(value, "name"):   # enum
        return value.value
    return str(value)


# --------------------------------------------------------------------------- #
# Signals
# --------------------------------------------------------------------------- #
def outcome_to_dict(outcome) -> dict[str, Any]:
    return {
        "name": outcome.name,
        "action": clean(outcome.action),
        "multiplier": clean(outcome.multiplier),
        "stop_multiplier": clean(outcome.stop_multiplier),
        "reason": outcome.reason,
        "detail": clean(outcome.detail),
        # The confidence impact in percent, which is what the waterfall plots.
        "impact": clean((outcome.multiplier - 1.0) * 100.0),
    }


def signal_to_dict(signal) -> dict[str, Any]:
    raw = signal.raw
    instrument = signal.instrument
    edge = getattr(signal, "edge", None)

    return {
        "id": f"{instrument.symbol}:{raw.strategy}",
        "symbol": instrument.symbol,
        "name": instrument.name,
        "asset_class": clean(instrument.asset_class),
        "direction": clean(raw.direction),
        "strategy": raw.strategy,
        "timeframe": raw.timeframe,
        "verdict": signal.verdict,
        "blocked": bool(signal.blocked),
        "confidence": clean(signal.confidence),
        "strength": clean(raw.strength),
        "entry": clean(raw.entry),
        "stop": clean(signal.adjusted_stop or raw.stop),
        "original_stop": clean(raw.stop),
        "take_profit": clean(signal.adjusted_take_profit or raw.take_profit),
        "reward_risk": clean(raw.reward_risk),
        "risk_fraction": clean(signal.risk_fraction),
        "size_fraction": clean(signal.size_fraction),
        "units": clean(signal.units),
        "rationale": list(raw.rationale),
        "block_reasons": list(signal.block_reasons),
        "outcomes": [outcome_to_dict(o) for o in signal.outcomes],
        "edge": edge_to_dict(edge),
        "ts": clean(raw.ts),
    }


def edge_to_dict(edge) -> dict[str, Any] | None:
    if edge is None:
        return None
    hit_rate = edge.hit_rate
    return {
        "expected_r": clean(edge.expected_r),
        "expected_r_lower": clean(edge.expected_r_lower),
        "gross_expected_r": clean(edge.gross_expected_r),
        "cost_r": clean(edge.cost_r),
        "cost_drag": clean(edge.cost_drag),
        "reward_risk": clean(edge.reward_risk),
        "breakeven_hit_rate": clean(edge.breakeven_hit_rate),
        "edge_ratio": clean(edge.edge_ratio),
        "kelly": clean(edge.kelly),
        "tradable": bool(edge.tradable),
        "reason": edge.reason,
        "notes": list(edge.notes),
        "hit_rate": {
            "mean": clean(hit_rate.mean),
            "lower": clean(hit_rate.lower),
            "prior": clean(hit_rate.prior),
            "observations": int(hit_rate.observations),
            "credibility": clean(hit_rate.credibility),
            "source": hit_rate.source,
        },
    }


# --------------------------------------------------------------------------- #
# Portfolio
# --------------------------------------------------------------------------- #
def allocation_to_dict(allocation) -> dict[str, Any]:
    signal = allocation.signal
    return {
        "key": allocation.key,
        "symbol": allocation.symbol,
        "name": signal.instrument.name,
        "direction": clean(signal.direction),
        "strategy": signal.raw.strategy,
        "standalone_risk": clean(allocation.standalone_risk),
        "risk_fraction": clean(allocation.risk_fraction),
        "scale": clean(allocation.scale),
        "units": clean(allocation.units),
        "cluster": int(allocation.cluster),
        "cluster_rank": int(allocation.cluster_rank),
        "risk_contribution": clean(allocation.risk_contribution),
        "dropped": bool(allocation.dropped),
        "reasons": list(allocation.reasons),
        "edge": edge_to_dict(allocation.edge),
        "confidence": clean(signal.confidence),
        "entry": clean(signal.raw.entry),
        "stop": clean(signal.adjusted_stop or signal.raw.stop),
    }


def plan_to_dict(plan) -> dict[str, Any] | None:
    """Serialise a portfolio plan, including the correlation matrix.

    The matrix is sent as a dense list-of-lists with its labels, rather than as
    nested objects: it is square, it is small (bounded by ``max_positions``),
    and the heatmap wants it indexable by position.
    """
    if plan is None:
        return None

    book = plan.book
    correlation = None
    if plan.correlation is not None and not plan.correlation.matrix.empty:
        matrix = plan.correlation.matrix
        labels = list(matrix.index)
        correlation = {
            "labels": labels,
            "symbols": [label.split(":")[0] for label in labels],
            "matrix": [[clean(v) for v in row] for row in matrix.to_numpy().tolist()],
            "observations": int(plan.correlation.observations),
            "estimated_pairs": int(plan.correlation.estimated_pairs),
            "total_pairs": int(plan.correlation.total_pairs),
            "coverage": clean(plan.correlation.coverage),
        }

    return {
        "summary": clean(plan.summary()),
        "gross_requested": clean(plan.gross_requested),
        "gross_risk": clean(plan.gross_risk),
        "effective_risk": clean(plan.effective_risk),
        "diversification_ratio": clean(plan.diversification_ratio),
        "expected_r": clean(plan.expected_r),
        "scale_applied": clean(plan.scale_applied),
        "calibration_source": plan.calibration_source,
        "calibration_observations": int(plan.calibration_observations),
        "notes": list(plan.notes),
        "concentration": clean(book.concentration) if book else None,
        "clusters": len({a.cluster for a in plan.held}),
        "correlation": correlation,
        "held": [allocation_to_dict(a) for a in plan.held],
        "dropped": [allocation_to_dict(a) for a in plan.dropped],
    }


# --------------------------------------------------------------------------- #
# Macro
# --------------------------------------------------------------------------- #
def macro_to_dict(macro, limit_events: int = 40) -> dict[str, Any]:
    currencies = sorted(
        set(macro.policy_rates) | set(macro.inflation) | set(macro.yield_10y)
    )
    rates = [
        {
            "currency": ccy,
            "policy_rate": clean(macro.policy_rates.get(ccy)),
            "inflation": clean(macro.inflation.get(ccy)),
            "real_rate": clean(macro.real_rate(ccy)),
            "yield_10y": clean(macro.yield_10y.get(ccy)),
            "yield_2y": clean(macro.yield_2y.get(ccy)),
            "curve": clean(macro.yield_spread(ccy)),
            "esi": clean(macro.esi.get(ccy)),
        }
        for ccy in currencies
    ]

    upcoming = sorted([e for e in macro.events if e.ts >= macro.ts], key=lambda e: e.ts)
    events = [
        {
            "ts": clean(event.ts),
            "hours_away": clean((event.ts - macro.ts).total_seconds() / 3600.0),
            "currency": event.currency,
            "country": event.country,
            "name": event.name,
            "impact": event.impact,
            "forecast": clean(event.forecast),
            "previous": clean(event.previous),
            "actual": clean(event.actual),
        }
        for event in upcoming[:limit_events]
    ]

    tokenomics = [
        {
            "symbol": symbol,
            "velocity": clean(macro.token_velocity.get(symbol)),
            "velocity_z": clean(macro.token_velocity_z.get(symbol)),
            "nvt": clean(macro.nvt.get(symbol)),
            "nvt_z": clean(macro.nvt_z.get(symbol)),
        }
        for symbol in sorted(macro.token_velocity)
    ]

    live = sum(1 for v in macro.sources.values() if v != "synthetic")
    return {
        "ts": clean(macro.ts),
        "liquidity": {
            "regime": clean(macro.gli_regime),
            "gli_delta": clean(macro.gli_delta),
            "m2_change": clean(macro.m2_change),
            "stablecoin_change": clean(macro.stablecoin_change),
        },
        "macro_regime": clean(macro.macro_regime),
        "fear_greed": clean(macro.fear_greed),
        "portfolio_cvar": clean(macro.portfolio_cvar),
        "open_risk": clean(macro.open_risk),
        "rates": rates,
        "events": events,
        "tokenomics": tokenomics,
        "sources": clean(macro.sources),
        "sources_live": live,
        "sources_total": len(macro.sources),
    }


# --------------------------------------------------------------------------- #
# Cycle
# --------------------------------------------------------------------------- #
def cycle_to_dict(state, history_points: int = 400) -> dict[str, Any] | None:
    if state is None:
        return None

    history: list[dict[str, Any]] = []
    frame = getattr(state, "history", None)
    if frame is not None and not frame.empty:
        tail = frame.tail(history_points)
        for ts, row in tail.iterrows():
            point = {"ts": clean(ts)}
            for column in tail.columns:
                point[str(column)] = clean(row[column])
            history.append(point)

    return {
        "domain": state.domain,
        "ts": clean(state.ts),
        "phase": clean(state.phase),
        "bias": state.bias,
        "score": clean(state.score),
        "momentum": clean(state.momentum),
        "confidence": clean(state.confidence),
        "transition_probability": clean(state.transition_probability),
        "days_in_phase": int(state.days_in_phase),
        "divergence": clean(state.divergence),
        "divergence_flag": bool(state.divergence_flag),
        "previous_phase": clean(state.previous_phase),
        "data_quality": state.data_quality,
        "narrative": list(state.narrative),
        "factors": [
            {
                "name": reading.name,
                "label": reading.label,
                "score": clean(reading.score),
                "weight": clean(reading.weight),
                "contribution": clean(reading.contribution),
                "lead_days": int(reading.lead_days),
                "ic": clean(reading.ic),
                "trusted": bool(reading.trusted),
                "direction": reading.direction,
                "rationale": reading.rationale,
            }
            for reading in state.readings
        ],
        "history": history,
    }


# --------------------------------------------------------------------------- #
# Price and indicators
# --------------------------------------------------------------------------- #
_OVERLAY_COLUMNS = (
    "ema_fast", "ema_slow", "ema_trend", "bb_upper", "bb_lower", "bb_middle",
    "supertrend", "donchian_upper", "donchian_lower", "vwap",
)
_PANEL_COLUMNS = (
    "rsi", "macd", "macd_signal", "macd_hist", "adx", "plus_di", "minus_di",
    "atr", "volume_z",
)


def candles_to_dict(instrument, frame: pd.DataFrame, indicators, points: int = 300
                    ) -> dict[str, Any]:
    """Price history with the overlays and oscillators the chart draws.

    Only the columns the interface actually plots are sent. The indicator frame
    carries several dozen, and shipping all of them would multiply the payload
    for data no pixel depends on.
    """
    if frame is None or frame.empty:
        return {"symbol": instrument.symbol, "candles": [], "overlays": {}, "panels": {}}

    tail = frame.tail(points)
    candles = [
        {
            "ts": clean(ts),
            "open": clean(row["open"]),
            "high": clean(row["high"]),
            "low": clean(row["low"]),
            "close": clean(row["close"]),
            "volume": clean(row.get("volume")),
        }
        for ts, row in tail.iterrows()
    ]

    overlays: dict[str, list[float | None]] = {}
    panels: dict[str, list[float | None]] = {}
    if indicators is not None:
        full = getattr(indicators, "df", None)
        if full is not None and not full.empty:
            aligned = full.reindex(tail.index)
            for column in _OVERLAY_COLUMNS:
                if column in aligned:
                    overlays[column] = [clean(v) for v in aligned[column].tolist()]
            for column in _PANEL_COLUMNS:
                if column in aligned:
                    panels[column] = [clean(v) for v in aligned[column].tolist()]

    return {
        "symbol": instrument.symbol,
        "name": instrument.name,
        "asset_class": clean(instrument.asset_class),
        "candles": candles,
        "overlays": overlays,
        "panels": panels,
        "last": clean(tail["close"].iloc[-1]),
        "change": clean(
            tail["close"].iloc[-1] / tail["close"].iloc[0] - 1.0
            if len(tail) > 1 and tail["close"].iloc[0] else None
        ),
    }


def regime_to_dict(regime) -> dict[str, Any] | None:
    if regime is None:
        return None
    return {
        "regime": clean(regime.regime),
        "prefers": regime.prefers,
        "confidence": clean(regime.confidence),
        "adx": clean(regime.adx),
        "realized_vol": clean(regime.realized_vol),
        "vol_percentile": clean(regime.vol_percentile),
        "hurst": clean(regime.hurst),
        "drawdown": clean(regime.drawdown),
        "trend_slope": clean(regime.trend_slope),
    }


def analysis_to_dict(result, include_history: bool = False) -> dict[str, Any]:
    """The whole scan: macro, per-instrument state, signals and the book."""
    instruments = []
    for symbol, analysis in result.analyses.items():
        instruments.append(
            {
                "symbol": symbol,
                "name": analysis.instrument.name,
                "asset_class": clean(analysis.instrument.asset_class),
                "price": clean(analysis.price),
                "error": analysis.error,
                "regime": regime_to_dict(analysis.regime),
                "signal_count": len(analysis.signals),
                "best": signal_to_dict(analysis.best) if analysis.best else None,
            }
        )

    cycles = {
        domain: cycle_to_dict(state, history_points=200 if include_history else 0)
        for domain, state in result.macro.cycle_states.items()
    }

    return {
        "ts": clean(result.ts),
        "timeframe": result.timeframe,
        "summary": clean(result.summary()),
        "macro": macro_to_dict(result.macro),
        "cycles": cycles,
        "instruments": instruments,
        "signals": [signal_to_dict(s) for s in result.all_signals],
        "portfolio": plan_to_dict(getattr(result, "plan", None)),
    }


def calibration_to_dict(calibrator) -> dict[str, Any]:
    reliability = calibrator.reliability_table()
    strategies = calibrator.strategy_table()
    return {
        "fitted": bool(calibrator.fitted),
        "summary": clean(calibrator.summary()),
        "reliability": [clean(row) for row in reliability.to_dict("records")],
        "strategies": [clean(row) for row in strategies.to_dict("records")],
    }
