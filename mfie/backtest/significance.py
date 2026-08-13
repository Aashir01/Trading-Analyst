r"""Is the backtest result real, or is it the best of many guesses?

A Sharpe ratio is an estimate, and like any estimate it has a standard error.
Sharpe's own is roughly :math:`\sqrt{(1 + \tfrac12 SR^2)/n}` — on 250 daily
observations that is about 0.064, so a reported Sharpe of 1.0 carries a 95%
interval of roughly [0.87, 1.13] *if it was the only thing ever tested*.

It never is. The README already admits the honest version of this problem for
the Cycle Compass ("700 daily readings of a 63-day forward return are ~11
independent observations"). The same disease has a second strain here: the
project ships ten strategies, and ``config/params.yaml`` holds around fifty
tunable thresholds. Run the backtest, keep the best, and the winner's Sharpe
is not an estimate of its edge — it is an estimate of its edge *plus the
maximum of fifty draws of estimation noise*, and that maximum is large.

Under the null of no skill, the expected maximum Sharpe across :math:`N`
independent trials is approximately

.. math::

    E[\max SR] \approx \sqrt{V[SR]}\left[(1-\gamma)Z^{-1}\!\left(1-\tfrac1N\right)
                        + \gamma Z^{-1}\!\left(1-\tfrac{1}{Ne}\right)\right]

with :math:`\gamma` the Euler-Mascheroni constant. Twenty trials on a
zero-skill strategy produce an expected best Sharpe near 0.5 for free. The
Deflated Sharpe Ratio (Bailey & López de Prado, 2014) is the probability that
the observed Sharpe exceeds that benchmark, given the sample's own skew and
kurtosis — negative skew and fat tails, the signature of every trend
strategy's return distribution, make a given Sharpe less impressive, not more.

``min_track_record_length`` inverts the question: how many observations would
be needed before this result could be called significant at all? The answer is
usually much larger than the backtest, and that is the useful part.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mfie.core.utils import safe_div

EULER_MASCHERONI = 0.5772156649015329


def _normal_cdf(x: float) -> float:
    from scipy.stats import norm

    return float(norm.cdf(x))


def _normal_ppf(q: float) -> float:
    from scipy.stats import norm

    return float(norm.ppf(min(max(q, 1e-12), 1 - 1e-12)))


def sharpe_standard_error(sharpe: float, n: int, skew: float = 0.0,
                          excess_kurtosis: float = 0.0) -> float:
    r"""Standard error of a Sharpe estimate, adjusted for higher moments.

    .. math::

        \sigma_{SR} = \sqrt{\frac{1 - \gamma_3 SR
                      + \frac{\gamma_4 - 1}{4} SR^2}{n - 1}}

    The skew term is why a strategy that wins small and often but loses big is
    penalised: negative :math:`\gamma_3` inflates the standard error, which is
    the statistics agreeing with intuition for once.
    """
    if n < 3:
        return float("inf")
    variance = (
        1.0
        - skew * sharpe
        + ((excess_kurtosis + 3.0) - 1.0) / 4.0 * sharpe**2
    )
    return float(np.sqrt(max(variance, 1e-12) / (n - 1)))


def probabilistic_sharpe_ratio(
    sharpe: float,
    n: int,
    skew: float = 0.0,
    excess_kurtosis: float = 0.0,
    benchmark: float = 0.0,
) -> float:
    """P(true Sharpe > ``benchmark``), given the sample's shape.

    All quantities are per-period — deannualise an annualised Sharpe before
    passing it in, or the standard error will be wrong by that factor.
    """
    se = sharpe_standard_error(sharpe, n, skew, excess_kurtosis)
    if not np.isfinite(se) or se <= 0:
        return 0.0
    return _normal_cdf((sharpe - benchmark) / se)


def expected_max_sharpe(trials: int, sharpe_variance: float = 1.0) -> float:
    """Expected best Sharpe across ``trials`` independent zero-skill attempts.

    This is the bar a backtest has to clear to be interesting. It grows with
    the number of things tried, which is why "we tried a hundred parameter
    combinations and this one worked" is a statement about the hundred, not
    about the one.
    """
    n = max(int(trials), 1)
    if n == 1:
        return 0.0
    sigma = np.sqrt(max(sharpe_variance, 0.0))
    term = (1 - EULER_MASCHERONI) * _normal_ppf(1 - 1.0 / n) + EULER_MASCHERONI * _normal_ppf(
        1 - 1.0 / (n * np.e)
    )
    return float(sigma * term)


def deflated_sharpe_ratio(
    sharpe: float,
    n: int,
    trials: int,
    skew: float = 0.0,
    excess_kurtosis: float = 0.0,
    sharpe_variance: float | None = None,
) -> float:
    """Probability the observed Sharpe survives the multiple-testing correction.

    ``trials`` is the number of configurations actually tried — strategies,
    parameter sets, symbols, restarts. Under-report it and the test flatters
    you; the honest number is usually larger than anyone wants to admit.

    Above 0.95 the result is unlikely to be selection noise. Below 0.5 it very
    likely is.
    """
    if sharpe_variance is None:
        # Without the cross-trial variance, the estimator's own variance is the
        # best available proxy for how much the trials would have scattered.
        sharpe_variance = sharpe_standard_error(sharpe, n, skew, excess_kurtosis) ** 2
    benchmark = expected_max_sharpe(trials, sharpe_variance)
    return probabilistic_sharpe_ratio(sharpe, n, skew, excess_kurtosis, benchmark)


def min_track_record_length(
    sharpe: float,
    skew: float = 0.0,
    excess_kurtosis: float = 0.0,
    benchmark: float = 0.0,
    confidence: float = 0.95,
) -> float:
    """Observations needed before this Sharpe would be significant.

    Returns infinity when the observed Sharpe does not exceed the benchmark at
    all — no amount of further data makes a negative edge significant.
    """
    excess = sharpe - benchmark
    if excess <= 0:
        return float("inf")
    z = _normal_ppf(confidence)
    variance = 1.0 - skew * sharpe + ((excess_kurtosis + 3.0) - 1.0) / 4.0 * sharpe**2
    return float(1.0 + max(variance, 1e-12) * (z / excess) ** 2)


@dataclass
class SignificanceReport:
    """The multiple-testing view of a backtest."""

    sharpe_per_period: float = 0.0
    sharpe_annual: float = 0.0
    observations: int = 0
    skew: float = 0.0
    excess_kurtosis: float = 0.0
    trials: int = 1
    psr: float = 0.0                     # P(true Sharpe > 0)
    dsr: float = 0.0                     # after the multiple-testing haircut
    expected_max_sharpe: float = 0.0     # the null's best guess, per period
    min_track_record: float = 0.0        # observations needed for significance

    @property
    def verdict(self) -> str:
        if self.observations < 30:
            return "INSUFFICIENT DATA"
        if self.dsr >= 0.95:
            return "SIGNIFICANT"
        if self.dsr >= 0.75:
            return "SUGGESTIVE"
        return "NOT DISTINGUISHABLE FROM SELECTION NOISE"

    def summary_lines(self) -> list[str]:
        track = (
            "never at this Sharpe"
            if not np.isfinite(self.min_track_record)
            else f"{self.min_track_record:,.0f} observations"
        )
        return [
            f"Sharpe (annual)      {self.sharpe_annual:>10.2f}",
            f"Observations         {self.observations:>10d}",
            f"Skew / excess kurt   {self.skew:>10.2f} / {self.excess_kurtosis:.2f}",
            f"Trials assumed       {self.trials:>10d}",
            f"Probabilistic Sharpe {self.psr:>10.2%}",
            f"Deflated Sharpe      {self.dsr:>10.2%}",
            f"Needed for 95% conf. {track:>10}",
            f"Verdict              {self.verdict}",
        ]


def assess_significance(
    returns: pd.Series,
    trials: int = 1,
    periods_per_year: float = 252.0,
) -> SignificanceReport:
    """Full significance report for a return series.

    ``trials`` should count everything that was tried before this result was
    chosen — strategies in the library, parameter combinations swept, symbols
    scanned. The engine's own default counts the strategy library, because
    picking the best of ten strategies is ten trials whether or not anyone
    thought of it that way.
    """
    r = pd.Series(returns).dropna().astype(float)
    report = SignificanceReport(trials=max(int(trials), 1), observations=int(len(r)))
    if len(r) < 5:
        return report

    sigma = float(r.std(ddof=1))
    if sigma <= 0:
        return report

    sharpe = float(r.mean() / sigma)
    report.sharpe_per_period = sharpe
    report.sharpe_annual = float(sharpe * np.sqrt(periods_per_year))
    report.skew = float(r.skew())
    report.excess_kurtosis = float(r.kurtosis())

    variance = sharpe_standard_error(sharpe, len(r), report.skew, report.excess_kurtosis) ** 2
    report.expected_max_sharpe = expected_max_sharpe(report.trials, variance)
    report.psr = probabilistic_sharpe_ratio(sharpe, len(r), report.skew, report.excess_kurtosis, 0.0)
    report.dsr = deflated_sharpe_ratio(
        sharpe, len(r), report.trials, report.skew, report.excess_kurtosis, variance
    )
    report.min_track_record = min_track_record_length(
        sharpe, report.skew, report.excess_kurtosis, report.expected_max_sharpe
    )
    return report


def haircut_sharpe(sharpe: float, trials: int, n: int) -> float:
    """The Sharpe that survives the correction — what the number is 'really' worth.

    Useful as a single figure: a 1.4 Sharpe found across 40 trials on 500
    observations is worth roughly what a much smaller Sharpe found on the first
    attempt would have been.
    """
    if n < 3 or sharpe <= 0:
        return float(sharpe)
    variance = sharpe_standard_error(sharpe, n) ** 2
    return float(max(sharpe - expected_max_sharpe(trials, variance), 0.0))


def independent_observations(n: int, overlap: int) -> float:
    """Effective sample size when observations overlap by ``overlap`` periods.

    The Cycle Compass already corrects for this; backtests on overlapping
    windows need the same haircut, and rarely get it.
    """
    return float(safe_div(n, max(overlap, 1), float(n)))
