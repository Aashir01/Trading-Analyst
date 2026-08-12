"""Performance evaluation.

Reports downside-aware statistics first (Sortino, CVaR, max drawdown) because
those are what actually constrain a real book, and includes the trade-level
diagnostics — expectancy, profit factor, win rate — that tell you *why* a
strategy made or lost money rather than just how much.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from mfie.core.utils import annualization_factor, safe_div
from mfie.econ.risk import (
    calmar_ratio,
    conditional_value_at_risk,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    ulcer_index,
    value_at_risk,
)


@dataclass
class PerformanceReport:
    # Returns
    total_return: float = 0.0
    annual_return: float = 0.0
    annual_volatility: float = 0.0

    # Risk-adjusted
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0

    # Drawdown / tail
    max_drawdown: float = 0.0
    ulcer: float = 0.0
    var_95: float = 0.0     # daily
    cvar_95: float = 0.0    # daily

    # Trades
    trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_bars_held: float = 0.0
    exit_breakdown: dict[str, int] = field(default_factory=dict)

    # Pipeline
    signals_generated: int = 0
    signals_blocked: int = 0
    signals_taken: int = 0
    block_reasons: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def summary_lines(self) -> list[str]:
        return [
            f"Total return      {self.total_return:>10.2%}",
            f"Annualised return {self.annual_return:>10.2%}",
            f"Annualised vol    {self.annual_volatility:>10.2%}",
            f"Sharpe            {self.sharpe:>10.2f}",
            f"Sortino           {self.sortino:>10.2f}",
            f"Calmar            {self.calmar:>10.2f}",
            f"Max drawdown      {self.max_drawdown:>10.2%}",
            f"Daily CVaR 95%    {self.cvar_95:>10.2%}",
            f"Trades            {self.trades:>10d}",
            f"Win rate          {self.win_rate:>10.2%}",
            f"Profit factor     {self.profit_factor:>10.2f}",
            f"Expectancy        {self.expectancy:>10.4f}",
            f"Signals gen/blk/taken {self.signals_generated}/{self.signals_blocked}/{self.signals_taken}",
        ]


def evaluate_performance(result, periods_per_year: float | None = None) -> PerformanceReport:
    """Build a report from a ``BacktestResult``."""
    equity = pd.Series(result.equity).dropna()
    returns = pd.Series(result.returns).dropna()
    ppy = periods_per_year or annualization_factor(getattr(result, "timeframe", "1h"))

    report = PerformanceReport(
        signals_generated=getattr(result, "signals_generated", 0),
        signals_blocked=getattr(result, "signals_blocked", 0),
        signals_taken=getattr(result, "signals_taken", 0),
        block_reasons=dict(getattr(result, "block_reasons", {})),
    )

    if len(equity) >= 2:
        report.total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
        years = max(len(returns) / ppy, 1e-9)
        # Geometric annualisation; guards against a wiped-out equity curve.
        growth = max(equity.iloc[-1] / equity.iloc[0], 1e-9)
        report.annual_return = float(growth ** (1.0 / years) - 1.0)
        report.annual_volatility = float(returns.std(ddof=1) * np.sqrt(ppy))
        report.sharpe = sharpe_ratio(returns, ppy)
        report.sortino = sortino_ratio(returns, ppy)
        report.calmar = calmar_ratio(returns, equity, ppy)
        report.max_drawdown = max_drawdown(equity)
        report.ulcer = ulcer_index(equity)

        # Tail risk is measured on *daily* returns, not per-bar. On an hourly
        # backtest most bars have no position and contribute an exact zero;
        # a distribution padded with zeros understates the tail badly and would
        # not be comparable to the daily CVaR budget in RiskParams.
        daily = daily_returns(equity)
        tail_source = daily if len(daily) >= 20 else returns
        report.var_95 = value_at_risk(tail_source, 0.95)
        report.cvar_95 = conditional_value_at_risk(tail_source, 0.95)

    closed = [t for t in getattr(result, "trades", []) if getattr(t, "exit_ts", None) is not None]
    report.trades = len(closed)
    if closed:
        pnls = np.array([t.pnl for t in closed], dtype=float)
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]

        report.win_rate = float(len(wins) / len(pnls))
        gross_profit = float(wins.sum())
        gross_loss = float(-losses.sum())
        report.profit_factor = safe_div(gross_profit, gross_loss, float("inf") if gross_profit else 0.0)
        report.expectancy = float(pnls.mean())
        report.avg_win = float(wins.mean()) if len(wins) else 0.0
        report.avg_loss = float(losses.mean()) if len(losses) else 0.0
        report.largest_win = float(pnls.max())
        report.largest_loss = float(pnls.min())
        report.avg_bars_held = float(np.mean([t.bars_held for t in closed]))

        breakdown: dict[str, int] = {}
        for trade in closed:
            breakdown[trade.exit_reason] = breakdown.get(trade.exit_reason, 0) + 1
        report.exit_breakdown = breakdown

    return report


def daily_returns(equity: pd.Series) -> pd.Series:
    """Resample an equity curve of any bar size to calendar-day returns."""
    e = pd.Series(equity).dropna()
    if e.empty or not isinstance(e.index, pd.DatetimeIndex):
        return pd.Series(dtype=float)
    return e.resample("1D").last().ffill().pct_change().dropna()


def monthly_returns(equity: pd.Series) -> pd.Series:
    """Calendar-month returns, for the classic heat-map view."""
    e = pd.Series(equity).dropna()
    if e.empty:
        return pd.Series(dtype=float)
    monthly = e.resample("ME").last()
    return monthly.pct_change().dropna()


def rolling_sharpe(returns: pd.Series, window: int = 120, periods_per_year: float = 8760) -> pd.Series:
    r = pd.Series(returns).dropna()
    if len(r) < window:
        return pd.Series(dtype=float)
    mu = r.rolling(window).mean()
    sd = r.rolling(window).std(ddof=1).replace(0.0, np.nan)
    return (mu / sd * np.sqrt(periods_per_year)).dropna()


def strategy_attribution(result) -> pd.DataFrame:
    """P&L broken down by strategy — which parts of the library actually work."""
    closed = [t for t in getattr(result, "trades", []) if getattr(t, "exit_ts", None) is not None]
    if not closed:
        return pd.DataFrame()
    frame = pd.DataFrame(
        [{"strategy": t.strategy, "pnl": t.pnl, "win": 1 if t.pnl > 0 else 0} for t in closed]
    )
    grouped = frame.groupby("strategy").agg(
        trades=("pnl", "size"),
        total_pnl=("pnl", "sum"),
        avg_pnl=("pnl", "mean"),
        win_rate=("win", "mean"),
    )
    return grouped.sort_values("total_pnl", ascending=False)


def compare_reports(reports: dict[str, PerformanceReport]) -> pd.DataFrame:
    """Side-by-side table, e.g. filtered vs unfiltered."""
    rows = {}
    for label, report in reports.items():
        rows[label] = {
            "Total return": report.total_return,
            "Annual return": report.annual_return,
            "Annual vol": report.annual_volatility,
            "Sharpe": report.sharpe,
            "Sortino": report.sortino,
            "Max drawdown": report.max_drawdown,
            "Daily CVaR 95%": report.cvar_95,
            "Trades": report.trades,
            "Win rate": report.win_rate,
            "Profit factor": report.profit_factor,
            "Signals blocked": report.signals_blocked,
        }
    return pd.DataFrame(rows)
