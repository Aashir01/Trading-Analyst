"""Market Cycle Compass — the leading bull/bear state engine.

This is the part of the project that is not a re-implementation of something
standard. The individual formulas are public; the design decisions that make
the composite *lead* rather than *lag* are where the work is:

1. **Impulses, not levels.** Every macro factor enters as a rate of change or an
   acceleration. The level of M2 tells you nothing; the second derivative of
   liquidity turns before risk assets do.

2. **A non-monotonic credit factor.** Nearly every model treats an inverting
   yield curve as the bear signal. Historically that is early by a year or
   more, and markets usually melt up through the inversion. The actual trigger
   is the *bull steepener* — the curve un-inverting from below. This factor is
   deliberately non-monotonic in the spread to capture that.

3. **Lead-aligned aggregation.** Each factor's lead time is estimated against
   forward returns, and factors are shifted so they all speak to the same
   forecast horizon before being combined. Averaging a 90-day-lead factor with
   a 10-day-lead factor without aligning them blurs both.

4. **Shrinkage toward economic priors, with no sign flipping.** Estimated
   weights are blended toward the priors in proportion to their statistical
   significance. If a factor's measured relationship has the *wrong sign*, the
   engine falls back to the prior and flags it, rather than flipping the sign
   and fitting the noise — which is how most composite indicators die.

5. **Asymmetric hysteresis.** Entering a bull state needs a higher bar than
   leaving one, because that is how markets actually behave.

Nothing here is a guarantee. See ``CycleEngine.validate`` — it reports the
measured information coefficient at every lead so you can see whether the
composite is predictive on your data, or is not.
"""

from mfie.alpha.cycle import CycleEngine, CyclePhase, CycleState, get_cycle_engine  # noqa: F401
from mfie.alpha.factors import FACTOR_SPECS, FactorPanel, FactorSpec  # noqa: F401
from mfie.alpha.leadlag import LeadLagResult, ValidationReport, scan_leads  # noqa: F401

__all__ = [
    "CycleEngine",
    "CycleState",
    "CyclePhase",
    "get_cycle_engine",
    "FACTOR_SPECS",
    "FactorPanel",
    "FactorSpec",
    "LeadLagResult",
    "ValidationReport",
    "scan_leads",
]
