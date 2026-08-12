r"""Lead-lag estimation and weight shrinkage.

The question this module answers for each factor: **how far ahead does it see,
and how much should we believe it?**

For a lead :math:`k` and horizon :math:`H`, the information coefficient is the
rank correlation between the factor observed today and the return realised over
the window that starts :math:`k` days from now:

.. math::

    IC(k) = \operatorname{corr}_{\text{Spearman}}
            \Big( f_t,\; \frac{P_{t+k+H}}{P_{t+k}} - 1 \Big)

Rank correlation rather than Pearson because factor and return distributions are
both fat-tailed, and one 2020 makes a Pearson correlation say whatever that year
said.

The lead that maximises :math:`|IC|` is the factor's measured lead time. The
weight then blends the economic prior with the measured strength, in proportion
to statistical significance — and **refuses to flip a factor's sign**. If a
factor whose economics say "positive" measures negative, that is evidence the
factor is not working, not evidence that the economics are backwards. Sign
flipping on in-sample evidence is the single most reliable way to build a
composite indicator that backtests beautifully and then loses money.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mfie.core.utils import clamp, get_logger

log = get_logger(__name__)


@dataclass
class LeadLagResult:
    factor: str
    best_lead: int              # days ahead this factor appears to see
    best_ic: float              # information coefficient at that lead
    ic_curve: dict[int, float]  # lead -> IC, for plotting
    observations: int
    t_stat: float
    confidence: float           # 0..1 — how much to trust the measurement
    sign_agrees: bool           # does the measured sign match the economic prior

    @property
    def significant(self) -> bool:
        return self.sign_agrees and self.confidence > 0.0

    def summary(self) -> str:
        if not self.sign_agrees:
            return (
                f"{self.factor}: measured IC {self.best_ic:+.3f} has the wrong sign — "
                "falling back to the economic prior"
            )
        return (
            f"{self.factor}: leads by {self.best_lead}d, IC {self.best_ic:+.3f} "
            f"(t={self.t_stat:.1f}, confidence {self.confidence:.0%})"
        )


@dataclass
class ValidationReport:
    """Honest read on whether the composite actually predicts anything."""

    domain: str
    horizon: int
    composite_ic: float
    composite_t: float
    per_factor: list[LeadLagResult] = field(default_factory=list)
    hit_rate: float = 0.0
    n: int = 0
    note: str = ""

    @property
    def effective_n(self) -> float:
        return effective_sample_size(self.n, self.horizon)

    @property
    def verdict(self) -> str:
        if self.n < 200:
            return "INSUFFICIENT DATA"
        if abs(self.composite_t) < 2.0:
            return "NO MEASURABLE EDGE"
        if self.composite_ic > 0:
            return "PREDICTIVE"
        return "INVERTED — investigate before trusting"

    def lines(self) -> list[str]:
        out = [
            f"Domain            {self.domain}",
            f"Forward horizon   {self.horizon} days",
            f"Observations      {self.n} daily  ({self.effective_n:.0f} independent "
            f"after correcting for {self.horizon}-day window overlap)",
            f"Composite IC      {self.composite_ic:+.4f}  (t={self.composite_t:.2f})",
            f"Directional hits  {self.hit_rate:.1%}",
            f"Verdict           {self.verdict}",
            "",
            "Per factor:",
        ]
        out.extend(f"  {r.summary()}" for r in self.per_factor)
        if self.note:
            out += ["", self.note]
        return out


def forward_return(anchor: pd.Series, lead: int, horizon: int) -> pd.Series:
    """Return over the window ``[t+lead, t+lead+horizon]``, indexed at ``t``."""
    s = pd.Series(anchor).astype(float)
    start = s.shift(-lead)
    end = s.shift(-(lead + horizon))
    return (end / start.replace(0.0, np.nan)) - 1.0


def information_coefficient(factor: pd.Series, returns: pd.Series) -> tuple[float, int]:
    """Spearman rank correlation and the number of paired observations."""
    frame = pd.concat([pd.Series(factor).rename("f"), pd.Series(returns).rename("r")], axis=1)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 30 or frame["f"].nunique() < 5:
        return 0.0, len(frame)
    ic = frame["f"].corr(frame["r"], method="spearman")
    return (float(ic) if np.isfinite(ic) else 0.0), len(frame)


def effective_sample_size(n: int, horizon: int) -> float:
    r"""Independent observations in ``n`` daily readings of an ``H``-day forward return.

    Consecutive daily observations of a 63-day forward return share 62 of their
    63 days. They are very nearly the same observation counted 63 times, so the
    usual :math:`\sqrt{n}` in the t-statistic is wrong by a factor of about
    :math:`\sqrt{H}`.

    This is not a footnote. Measured naively on 700 daily readings, a factor in
    this project scored t = -7.3 — apparently overwhelming evidence. Corrected
    for overlap the same number is t ≈ -0.9: no evidence at all. Every published
    "leading indicator" that fails out of sample has some version of this
    mistake in it.
    """
    return max(float(n) / max(horizon, 1), 5.0)


def _t_stat(ic: float, n: int, horizon: int = 1) -> float:
    """t-statistic of a rank correlation, corrected for overlapping windows."""
    if abs(ic) >= 1.0:
        return 0.0
    n_eff = effective_sample_size(n, horizon)
    if n_eff < 5:
        return 0.0
    return float(ic * np.sqrt(max(n_eff - 2, 1)) / np.sqrt(max(1.0 - ic**2, 1e-9)))


def scan_leads(
    factor: pd.Series,
    anchor: pd.Series,
    horizon: int = 63,
    max_lead: int = 120,
    step: int = 10,
    min_tstat: float = 2.0,
    name: str = "factor",
) -> LeadLagResult:
    """Find the lead at which a factor is most informative about forward returns."""
    curve: dict[int, float] = {}
    best_lead, best_ic, best_n = 0, 0.0, 0

    for lead in range(0, max_lead + 1, max(step, 1)):
        returns = forward_return(anchor, lead, horizon)
        ic, n = information_coefficient(factor, returns)
        curve[lead] = ic
        if abs(ic) > abs(best_ic):
            best_lead, best_ic, best_n = lead, ic, n

    t = _t_stat(best_ic, best_n, horizon)
    sign_agrees = best_ic > 0  # all factors are constructed so higher = more bullish

    # Confidence ramps from 0 at the significance bar to 1 at twice the bar.
    confidence = clamp((abs(t) - min_tstat) / min_tstat, 0.0, 1.0) if sign_agrees else 0.0

    return LeadLagResult(
        factor=name,
        best_lead=best_lead,
        best_ic=best_ic,
        ic_curve=curve,
        observations=best_n,
        t_stat=t,
        confidence=confidence,
        sign_agrees=sign_agrees,
    )


def shrink_weights(
    priors: dict[str, float],
    results: dict[str, LeadLagResult],
    max_multiple: float = 2.0,
    min_multiple: float = 0.4,
) -> dict[str, float]:
    """Blend economic priors with measured strength, weighted by confidence.

    ``w = (1 - c) * prior + c * measured``, then clamped to
    ``[min_multiple, max_multiple] x prior``.

    At ``c = 0`` (no significant measurement, or the wrong sign) the prior
    survives untouched — the correct default, because the prior encodes decades
    of macro evidence and the measurement encodes one sample.

    The clamp matters as much as the blend. Left uncapped, whichever factor best
    fits this particular sample takes over the composite: in testing, valuation
    — the slowest and least timely factor of the seven — went from a 10% prior
    to a 44% measured weight on a single 720-day window. The cap keeps the
    composite recognisably the thing the priors describe.
    """
    total_prior = sum(priors.values()) or 1.0
    normalised_priors = {k: v / total_prior for k, v in priors.items()}

    strengths = {
        name: (abs(r.best_ic) if r.significant else 0.0) for name, r in results.items()
    }
    total_strength = sum(strengths.values())

    blended: dict[str, float] = {}
    for name, prior in normalised_priors.items():
        result = results.get(name)
        if result is None or total_strength <= 0:
            blended[name] = prior
            continue
        measured = strengths[name] / total_strength
        c = result.confidence
        raw = (1.0 - c) * prior + c * measured
        blended[name] = float(np.clip(raw, prior * min_multiple, prior * max_multiple))

    total = sum(blended.values()) or 1.0
    return {k: v / total for k, v in blended.items()}


def align_to_common_horizon(
    frame: pd.DataFrame,
    results: dict[str, LeadLagResult],
) -> pd.DataFrame:
    """Shift each factor so all of them speak to the same forecast horizon.

    A factor that leads by 90 days is, today, describing a point three months
    out. A factor leading by 10 days describes next fortnight. Averaging them
    raw blurs both. Shifting the fast factors *forward in time* is not possible
    (that would be lookahead), so instead the slow factors are delayed: each is
    shifted by ``lead - min_lead`` days, i.e. we read the 90-day factor as it
    stood 80 days ago so that it now describes the same moment the 10-day
    factor does.

    Only factors with a significant measured lead are shifted; the rest are left
    where they are, since delaying a factor on the strength of a noisy lead
    estimate throws away information for nothing.
    """
    leads = {
        name: results[name].best_lead
        for name in frame.columns
        if name in results and results[name].significant
    }
    if len(leads) < 2:
        return frame

    base = min(leads.values())
    shifted = frame.copy()
    for name, lead in leads.items():
        delay = lead - base
        if delay > 0:
            shifted[name] = frame[name].shift(delay)
    return shifted
