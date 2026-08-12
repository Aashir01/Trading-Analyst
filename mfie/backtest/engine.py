"""Event-driven backtester.

Design constraints that make the results mean something:

* **Bar-close decisions, next-bar fills.** A signal computed from bar ``t``'s
  close can only be filled at bar ``t+1``'s open. Filling at the close of the
  bar that generated the signal is the classic way to manufacture a beautiful,
  fictional equity curve.
* **Costs are always on.** Spread and commission are charged on entry and exit;
  slippage scales with the bar's range, so illiquid bars cost more.
* **Intrabar stop priority.** If a bar's range spans both the stop and the
  target, the stop is assumed to fill first. Without seeing tick data you
  cannot know the order, and the pessimistic assumption is the only honest one.
* **Walk-forward macro.** The macro context is rebuilt periodically from data
  available *at that point in the simulation*, not from today's snapshot.

What it does not model: partial fills, funding on leveraged positions, borrow
costs, exchange outages, or the market impact of your own order.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from mfie.config import Params, get_params
from mfie.core.types import Direction, Instrument, MacroContext
from mfie.core.utils import annualization_factor, get_logger
from mfie.pipeline.filters import Filter, FilterContext, build_filters
from mfie.pipeline.sizing import size_signal
from mfie.regime.detector import detect_regime
from mfie.strategies.base import Strategy, StrategyContext
from mfie.strategies.registry import build_strategies
from mfie.technical.indicators import compute_indicators

log = get_logger(__name__)


@dataclass
class CostModel:
    """Transaction costs, in fractions of notional."""

    spread_pct: float = 0.0002      # half-spread paid on each side
    commission_pct: float = 0.0004  # per side
    slippage_atr: float = 0.05      # fraction of ATR lost to slippage per side

    def entry_price(self, price: float, direction: Direction, atr: float) -> float:
        adverse = price * (self.spread_pct) + atr * self.slippage_atr
        return price + adverse if direction is Direction.LONG else price - adverse

    def exit_price(self, price: float, direction: Direction, atr: float) -> float:
        adverse = price * (self.spread_pct) + atr * self.slippage_atr
        return price - adverse if direction is Direction.LONG else price + adverse

    def commission(self, notional: float) -> float:
        return abs(notional) * self.commission_pct


@dataclass
class Trade:
    symbol: str
    strategy: str
    direction: Direction
    entry_ts: datetime
    entry_price: float
    units: float
    stop: float
    take_profit: float | None
    confidence: float
    risk_fraction: float
    exit_ts: datetime | None = None
    exit_price: float | None = None
    pnl: float = 0.0
    return_pct: float = 0.0
    exit_reason: str = ""
    costs: float = 0.0
    bars_held: int = 0

    @property
    def is_open(self) -> bool:
        return self.exit_ts is None

    @property
    def notional(self) -> float:
        return abs(self.units * self.entry_price)

    def as_row(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "strategy": self.strategy,
            "direction": self.direction.value,
            "entry_ts": self.entry_ts,
            "exit_ts": self.exit_ts,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "units": self.units,
            "pnl": self.pnl,
            "return_pct": self.return_pct,
            "exit_reason": self.exit_reason,
            "confidence": self.confidence,
        }


@dataclass
class BacktestResult:
    run_id: str
    equity: pd.Series
    trades: list[Trade]
    returns: pd.Series
    signals_generated: int = 0
    signals_blocked: int = 0
    signals_taken: int = 0
    block_reasons: dict[str, int] = field(default_factory=dict)
    params: Params | None = None
    timeframe: str = "1h"

    @property
    def trade_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.as_row() for t in self.trades])

    @property
    def closed_trades(self) -> list[Trade]:
        return [t for t in self.trades if not t.is_open]


class BacktestEngine:
    """Runs one or more instruments through the full signal -> filter -> size loop."""

    def __init__(
        self,
        params: Params | None = None,
        strategies: list[Strategy] | None = None,
        filters: list[Filter] | None = None,
        costs: CostModel | None = None,
        use_macro_filters: bool = True,
    ) -> None:
        self.params = params or get_params()
        self.strategies = strategies if strategies is not None else build_strategies()
        self.filters = filters if filters is not None else build_filters()
        self.costs = costs or CostModel()
        self.use_macro_filters = use_macro_filters

    # ------------------------------------------------------------------- run
    def run(
        self,
        instrument: Instrument,
        df: pd.DataFrame,
        macro: MacroContext | None = None,
        warmup: int = 250,
        max_bars_in_trade: int = 200,
        initial_equity: float | None = None,
    ) -> BacktestResult:
        if len(df) <= warmup + 10:
            raise ValueError(
                f"Need more than {warmup + 10} bars to backtest; got {len(df)}"
            )

        equity = initial_equity if initial_equity is not None else self.params.risk.account_equity
        starting_equity = equity
        periods_per_year = annualization_factor(self._infer_timeframe(df))

        # Indicators are computed once over the whole frame, then sliced. Every
        # indicator here is causal (backward-looking only), so slicing to bar i
        # gives exactly what would have been known at bar i.
        indicators_full = compute_indicators(df, self.params.technical, periods_per_year)
        full = indicators_full.df

        trades: list[Trade] = []
        open_trade: Trade | None = None
        equity_points: dict[pd.Timestamp, float] = {}
        signals_generated = signals_blocked = signals_taken = 0
        block_reasons: dict[str, int] = {}

        macro = macro or MacroContext()
        timestamps = df.index

        for i in range(warmup, len(df) - 1):
            ts = timestamps[i]
            # Only the *next* bar is ever touched for fills and exits — bar i is
            # the decision bar and is read through the indicator frame.
            next_bar = df.iloc[i + 1]

            # --- manage the open position on the *next* bar's range ---------
            if open_trade is not None:
                closed = self._try_exit(open_trade, next_bar, timestamps[i + 1], full.iloc[i])
                if closed:
                    equity += open_trade.pnl
                    trades.append(open_trade)
                    open_trade = None

            if open_trade is not None:
                open_trade.bars_held += 1
                if open_trade.bars_held >= max_bars_in_trade:
                    self._close_trade(
                        open_trade, float(next_bar["open"]), timestamps[i + 1],
                        "max_bars", float(full.iloc[i].get("atr", 0.0)),
                    )
                    equity += open_trade.pnl
                    trades.append(open_trade)
                    open_trade = None

            equity_points[ts] = equity

            if open_trade is not None:
                continue  # one position per instrument at a time

            # --- generate a signal from information available at bar i ------
            window = df.iloc[: i + 1]
            sliced = type(indicators_full)(full.iloc[: i + 1])
            regime = detect_regime(sliced, self.params.regime)

            ctx = StrategyContext(
                instrument=instrument,
                df=window,
                indicators=sliced,
                regime=regime,
                timeframe=self._infer_timeframe(df),
                params=self.params,
                extras={"spread_pct": self.costs.spread_pct},
            )

            best_signal = None
            best_decision = None
            for strategy in self.strategies:
                if not strategy.supports(ctx):
                    continue
                try:
                    raw = strategy.generate(ctx)
                except Exception:
                    continue
                if raw is None or raw.direction is Direction.FLAT:
                    continue

                signals_generated += 1
                outcomes = []
                blocked = False

                if self.use_macro_filters:
                    filter_ctx = FilterContext(
                        instrument=instrument,
                        macro=macro,
                        params=self.params,
                        regime=regime,
                        atr=float(full.iloc[i].get("atr", 0.0)),
                    )
                    for filt in self.filters:
                        outcome = filt(raw, filter_ctx)
                        outcomes.append(outcome)
                        if outcome.blocked:
                            blocked = True
                            key = outcome.name
                            block_reasons[key] = block_reasons.get(key, 0) + 1
                            break

                if blocked:
                    signals_blocked += 1
                    continue

                decision = size_signal(
                    raw, outcomes, params=self.params, equity=equity,
                    realized_vol=float(full.iloc[i].get("realized_vol", 0.0)) or None,
                )
                if decision.risk_fraction <= 0 or decision.units <= 0:
                    continue
                if best_decision is None or decision.confidence > best_decision.confidence:
                    best_signal, best_decision = raw, decision

            if best_signal is None or best_decision is None:
                continue

            # --- fill on the next bar's open --------------------------------
            atr_value = float(full.iloc[i].get("atr", 0.0))
            fill = self.costs.entry_price(float(next_bar["open"]), best_signal.direction, atr_value)
            open_trade = Trade(
                symbol=instrument.symbol,
                strategy=best_signal.strategy,
                direction=best_signal.direction,
                entry_ts=timestamps[i + 1].to_pydatetime(),
                entry_price=fill,
                units=best_decision.units,
                stop=best_decision.adjusted_stop,
                take_profit=best_decision.adjusted_take_profit,
                confidence=best_decision.confidence,
                risk_fraction=best_decision.risk_fraction,
            )
            open_trade.costs = self.costs.commission(open_trade.notional)
            signals_taken += 1

        # --- mark any still-open position to the final close ----------------
        if open_trade is not None:
            last_ts = timestamps[-1]
            self._close_trade(
                open_trade, float(df["close"].iloc[-1]), last_ts, "end_of_data",
                float(full["atr"].iloc[-1] if "atr" in full else 0.0),
            )
            equity += open_trade.pnl
            trades.append(open_trade)
            equity_points[last_ts] = equity

        equity_series = pd.Series(equity_points, name="equity").sort_index()
        returns = equity_series.pct_change().fillna(0.0)

        log.info(
            "Backtest %s: %d trades, %.2f -> %.2f equity",
            instrument.symbol, len(trades), starting_equity, equity,
        )

        return BacktestResult(
            run_id=uuid.uuid4().hex[:12],
            equity=equity_series,
            trades=trades,
            returns=returns,
            signals_generated=signals_generated,
            signals_blocked=signals_blocked,
            signals_taken=signals_taken,
            block_reasons=block_reasons,
            params=self.params,
            timeframe=self._infer_timeframe(df),
        )

    # ---------------------------------------------------------------- exits
    def _try_exit(self, trade: Trade, bar: pd.Series, ts, indicator_row: pd.Series) -> bool:
        high, low = float(bar["high"]), float(bar["low"])
        atr_value = float(indicator_row.get("atr", 0.0))

        stop_hit = (
            low <= trade.stop if trade.direction is Direction.LONG else high >= trade.stop
        )
        target_hit = (
            trade.take_profit is not None
            and (high >= trade.take_profit if trade.direction is Direction.LONG
                 else low <= trade.take_profit)
        )

        # Pessimistic: when both levels are inside the bar, assume the stop filled.
        if stop_hit:
            self._close_trade(trade, trade.stop, ts, "stop", atr_value)
            return True
        if target_hit:
            self._close_trade(trade, float(trade.take_profit), ts, "target", atr_value)
            return True
        return False

    def _close_trade(self, trade: Trade, price: float, ts, reason: str, atr_value: float) -> None:
        fill = self.costs.exit_price(price, trade.direction, atr_value)
        gross = (fill - trade.entry_price) * trade.units * trade.direction.sign
        exit_commission = self.costs.commission(abs(trade.units * fill))
        trade.costs += exit_commission
        trade.exit_price = fill
        trade.exit_ts = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        trade.pnl = float(gross - exit_commission - self.costs.commission(trade.notional))
        trade.return_pct = float(
            (fill - trade.entry_price) / trade.entry_price * trade.direction.sign
        )
        trade.exit_reason = reason

    @staticmethod
    def _infer_timeframe(df: pd.DataFrame) -> str:
        if len(df) < 3:
            return "1h"
        minutes = float(np.median(np.diff(df.index.values).astype("timedelta64[m]").astype(float)))
        table = {1: "1m", 5: "5m", 15: "15m", 30: "30m", 60: "1h",
                 120: "2h", 240: "4h", 720: "12h", 1440: "1d", 10080: "1w"}
        closest = min(table, key=lambda m: abs(m - minutes))
        return table[closest]


def compare_with_without_macro(
    instrument: Instrument,
    df: pd.DataFrame,
    macro: MacroContext,
    params: Params | None = None,
    warmup: int = 250,
) -> dict[str, BacktestResult]:
    """Run the same strategies with and without the macro filter chain.

    This is the experiment that justifies the whole econometric layer: if the
    filtered version does not improve risk-adjusted return, the filters are
    costing you money and the thresholds need rethinking.
    """
    unfiltered = BacktestEngine(params=params, use_macro_filters=False).run(
        instrument, df, macro, warmup=warmup
    )
    filtered = BacktestEngine(params=params, use_macro_filters=True).run(
        instrument, df, macro, warmup=warmup
    )
    return {"unfiltered": unfiltered, "macro_filtered": filtered}
