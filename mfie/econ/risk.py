r"""Risk budgeting: downside-aware metrics and position sizing.

Standard deviation punishes upside and downside identically, which is wrong for
a directional book. The engine budgets risk with Conditional Value at Risk
instead:

.. math::

    VaR_\alpha = \inf\{ l \in \mathbb{R} : P(L > l) \le 1 - \alpha \}

    CVaR_\alpha = \mathbb{E}[\,L \mid L \ge VaR_\alpha\,]

CVaR is the *average* loss in the tail, not the threshold, so it sees the shape
of the tail rather than just its edge. Sortino replaces Sharpe for the same
reason: it divides by downside deviation only.

All functions take a return series (simple returns, per period) and are
sign-conventioned so that **losses are positive numbers** in VaR/CVaR output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mfie.config import RiskParams
from mfie.core.utils import clamp, safe_div


# --------------------------------------------------------------------------- #
# Tail risk
# --------------------------------------------------------------------------- #
def value_at_risk(returns: pd.Series, alpha: float = 0.95, method: str = "historical") -> float:
    """VaR at confidence ``alpha``, returned as a positive loss fraction.

    ``method``: ``historical`` (empirical quantile), ``parametric`` (normal), or
    ``cornish_fisher`` (normal adjusted for skew and kurtosis).
    """
    r = pd.Series(returns).dropna().astype(float)
    if len(r) < 20:
        return 0.0

    if method == "historical":
        return float(max(-np.quantile(r.to_numpy(), 1.0 - alpha), 0.0))

    mu, sigma = float(r.mean()), float(r.std(ddof=1))
    if sigma <= 0:
        return 0.0

    from scipy.stats import norm  # local import keeps module import cheap

    z = float(norm.ppf(1.0 - alpha))

    if method == "cornish_fisher":
        s = float(r.skew())
        k = float(r.kurtosis())  # pandas returns excess kurtosis
        z = (
            z
            + (z**2 - 1) * s / 6.0
            + (z**3 - 3 * z) * k / 24.0
            - (2 * z**3 - 5 * z) * (s**2) / 36.0
        )
    elif method != "parametric":
        raise ValueError(f"Unknown VaR method {method!r}")

    return float(max(-(mu + z * sigma), 0.0))


def conditional_value_at_risk(returns: pd.Series, alpha: float = 0.95,
                              method: str = "historical") -> float:
    """CVaR / expected shortfall at ``alpha``, as a positive loss fraction."""
    r = pd.Series(returns).dropna().astype(float)
    if len(r) < 20:
        return 0.0

    if method == "historical":
        threshold = np.quantile(r.to_numpy(), 1.0 - alpha)
        tail = r[r <= threshold]
        if tail.empty:
            return float(max(-threshold, 0.0))
        return float(max(-tail.mean(), 0.0))

    mu, sigma = float(r.mean()), float(r.std(ddof=1))
    if sigma <= 0:
        return 0.0

    from scipy.stats import norm

    z = float(norm.ppf(1.0 - alpha))
    # Closed form for the normal case: E[X | X <= q] = mu - sigma * phi(z)/(1-alpha)
    shortfall = mu - sigma * float(norm.pdf(z)) / (1.0 - alpha)
    return float(max(-shortfall, 0.0))


def scale_to_horizon(risk_value: float, periods: int) -> float:
    """Square-root-of-time scaling for a VaR/CVaR figure."""
    return float(risk_value * np.sqrt(max(periods, 1)))


# --------------------------------------------------------------------------- #
# Risk-adjusted performance
# --------------------------------------------------------------------------- #
def sharpe_ratio(returns: pd.Series, periods_per_year: float = 252,
                 risk_free: float = 0.0) -> float:
    r = pd.Series(returns).dropna().astype(float)
    if len(r) < 5:
        return 0.0
    excess = r - risk_free / periods_per_year
    sigma = float(excess.std(ddof=1))
    if sigma <= 0:
        return 0.0
    return float(excess.mean() / sigma * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: float = 252,
                  target: float = 0.0) -> float:
    """Excess return over downside deviation (only sub-target returns count)."""
    r = pd.Series(returns).dropna().astype(float)
    if len(r) < 5:
        return 0.0
    excess = r - target / periods_per_year
    downside = excess[excess < 0]
    if downside.empty:
        return float("inf") if excess.mean() > 0 else 0.0
    # Downside deviation uses the full sample in the denominator, per Sortino.
    dd = float(np.sqrt((downside**2).sum() / len(excess)))
    if dd <= 0:
        return 0.0
    return float(excess.mean() / dd * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline, as a positive fraction."""
    e = pd.Series(equity).dropna().astype(float)
    if e.empty:
        return 0.0
    running_max = e.cummax()
    drawdown = (e - running_max) / running_max.replace(0.0, np.nan)
    return float(abs(drawdown.min())) if not drawdown.dropna().empty else 0.0


def drawdown_series(equity: pd.Series) -> pd.Series:
    e = pd.Series(equity).dropna().astype(float)
    if e.empty:
        return pd.Series(dtype=float)
    return (e - e.cummax()) / e.cummax().replace(0.0, np.nan)


def calmar_ratio(returns: pd.Series, equity: pd.Series, periods_per_year: float = 252) -> float:
    r = pd.Series(returns).dropna().astype(float)
    if r.empty:
        return 0.0
    annual = float((1 + r).prod() ** (periods_per_year / len(r)) - 1)
    mdd = max_drawdown(equity)
    return safe_div(annual, mdd, 0.0)


def ulcer_index(equity: pd.Series) -> float:
    """RMS drawdown — penalises long, deep underwater periods."""
    dd = drawdown_series(equity).dropna()
    if dd.empty:
        return 0.0
    return float(np.sqrt((dd**2).mean()))


# --------------------------------------------------------------------------- #
# Sizing
# --------------------------------------------------------------------------- #
def kelly_fraction(win_rate: float, reward_risk: float, cap: float = 0.25) -> float:
    r"""Kelly: :math:`f^* = p - (1-p)/b`, capped.

    Full Kelly is far too aggressive on estimated (rather than known) edges, so
    the result is capped — quarter-Kelly is the usual practitioner choice.
    """
    p = clamp(win_rate, 0.0, 1.0)
    b = max(reward_risk, 1e-6)
    f = p - (1.0 - p) / b
    return float(clamp(f, 0.0, cap))


def volatility_target_scalar(realized_vol: float, target_vol: float,
                             max_scalar: float = 2.0) -> float:
    """Scale exposure so realised volatility approaches the target."""
    if realized_vol is None or realized_vol <= 0 or not np.isfinite(realized_vol):
        return 1.0
    return float(clamp(target_vol / realized_vol, 0.1, max_scalar))


def position_size(
    equity: float,
    risk_fraction: float,
    entry: float,
    stop: float,
) -> tuple[float, float]:
    """Units to trade and notional, from fixed-fractional risk.

    ``risk_fraction`` is the share of equity lost if the stop is hit.
    Returns ``(units, notional)``.
    """
    stop_distance = abs(entry - stop)
    if stop_distance <= 0 or entry <= 0 or equity <= 0:
        return 0.0, 0.0
    risk_capital = equity * max(risk_fraction, 0.0)
    units = risk_capital / stop_distance
    return float(units), float(units * entry)


@dataclass
class RiskBudget:
    equity: float
    cvar: float
    var: float
    sortino: float
    max_drawdown: float
    budget: float
    utilisation: float        # cvar / budget
    breached: bool
    scaler: float             # multiplier applied to every new position

    @property
    def headroom(self) -> float:
        return max(self.budget - self.cvar, 0.0)


def evaluate_risk_budget(
    portfolio_returns: pd.Series,
    params: RiskParams | None = None,
    periods_per_year: float = 252,
) -> RiskBudget:
    """Check portfolio tail risk against the macro risk budget.

    When CVaR exceeds the budget, every new position is scaled by
    ``cvar_breach_scaler`` — the automatic de-risking rule.
    """
    p = params or RiskParams()
    r = pd.Series(portfolio_returns).dropna().astype(float).tail(p.cvar_lookback)

    cvar = conditional_value_at_risk(r, p.cvar_alpha)
    var = value_at_risk(r, p.cvar_alpha)
    equity_curve = (1 + r).cumprod() if not r.empty else pd.Series(dtype=float)

    breached = cvar > p.cvar_budget
    if breached:
        # Scale down proportionally to the overshoot, floored at the configured
        # hard scaler so a 10x breach does not produce a 0.001x position.
        overshoot = safe_div(p.cvar_budget, cvar, 1.0)
        scaler = float(clamp(max(overshoot, p.cvar_breach_scaler), 0.1, 1.0))
    else:
        scaler = 1.0

    return RiskBudget(
        equity=p.account_equity,
        cvar=cvar,
        var=var,
        sortino=sortino_ratio(r, periods_per_year),
        max_drawdown=max_drawdown(equity_curve),
        budget=p.cvar_budget,
        utilisation=safe_div(cvar, p.cvar_budget, 0.0),
        breached=breached,
        scaler=scaler,
    )


def portfolio_heat(open_risk_fractions: list[float]) -> float:
    """Total fraction of equity at risk across open positions."""
    return float(sum(max(x, 0.0) for x in open_risk_fractions))


def correlation_adjusted_risk(returns_matrix: pd.DataFrame, weights: pd.Series) -> float:
    r"""Portfolio volatility :math:`\sqrt{w^T \Sigma w}`.

    Two 1%-risk trades in EURUSD and GBPUSD are not 2% of independent risk —
    this is what stops the book from being one trade wearing a disguise.
    """
    df = returns_matrix.dropna()
    if df.empty or weights.empty:
        return 0.0
    common = [c for c in df.columns if c in weights.index]
    if not common:
        return 0.0
    cov = df[common].cov().to_numpy()
    w = weights[common].to_numpy()
    variance = float(w @ cov @ w)
    return float(np.sqrt(max(variance, 0.0)))
