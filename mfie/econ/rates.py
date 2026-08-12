r"""Interest rates: real yield differentials, the term premium, and carry.

Real Interest Rate Differential (RIRD) — currencies chase *real* yield:

.. math::

    R_A = I_A - \pi_A, \qquad R_B = I_B - \pi_B

    \Delta RIRD = R_A - R_B = (I_A - I_B) - (\pi_A - \pi_B)

where :math:`I` is the nominal policy rate and :math:`\pi` is trailing 12-month
CPI inflation. A positive :math:`\Delta RIRD` for pair ``A/B`` is a structural
bullish bias on ``A``.

Term premium — the classic recession signal:

.. math::  Spread_t = Y_{10Y,t} - Y_{2Y,t}

An inverted curve (:math:`Spread_t \le 0`) sets ``macro_regime = CONTRACTION``,
which throttles high-beta exposure.
"""

from __future__ import annotations

from dataclasses import dataclass

from mfie.config import RIRDParams, YieldParams
from mfie.core.types import Direction, Instrument, MacroRegime


def real_rate(nominal_rate: float | None, inflation: float | None) -> float | None:
    """R = I - pi. Returns ``None`` when either leg is unavailable."""
    if nominal_rate is None or inflation is None:
        return None
    return nominal_rate - inflation


def rird(
    policy_rates: dict[str, float],
    inflation: dict[str, float],
    base: str,
    quote: str,
) -> float | None:
    """ΔRIRD for pair ``base/quote``. Positive favours the base currency."""
    r_base = real_rate(policy_rates.get(base), inflation.get(base))
    r_quote = real_rate(policy_rates.get(quote), inflation.get(quote))
    if r_base is None or r_quote is None:
        return None
    return r_base - r_quote


def nominal_differential(policy_rates: dict[str, float], base: str, quote: str) -> float | None:
    """The raw carry — what you actually earn holding the pair overnight."""
    i_base, i_quote = policy_rates.get(base), policy_rates.get(quote)
    if i_base is None or i_quote is None:
        return None
    return i_base - i_quote


def yield_spread(yield_10y: dict[str, float], yield_2y: dict[str, float],
                 currency: str) -> float | None:
    y10, y2 = yield_10y.get(currency), yield_2y.get(currency)
    if y10 is None or y2 is None:
        return None
    return y10 - y2


def classify_macro_regime(
    yield_10y: dict[str, float],
    yield_2y: dict[str, float],
    params: YieldParams | None = None,
    anchor: str = "USD",
) -> tuple[MacroRegime, float | None]:
    """Regime from the anchor curve (USD by default — it prices global risk)."""
    p = params or YieldParams()
    spread = yield_spread(yield_10y, yield_2y, anchor)
    if spread is None:
        return MacroRegime.NEUTRAL, None
    if spread <= p.inversion_threshold:
        return MacroRegime.CONTRACTION, spread
    if spread >= p.steepening_threshold:
        return MacroRegime.EXPANSION, spread
    return MacroRegime.NEUTRAL, spread


@dataclass
class RateView:
    """Everything the rate-based filters need for one instrument."""

    base: str
    quote: str
    real_rate_base: float | None
    real_rate_quote: float | None
    rird: float | None
    nominal_carry: float | None
    spread_base: float | None
    spread_quote: float | None
    macro_regime: MacroRegime
    anchor_spread: float | None

    @property
    def favoured_direction(self) -> Direction:
        """Which way the real-yield differential leans, before any technicals."""
        if self.rird is None:
            return Direction.FLAT
        if self.rird > 0:
            return Direction.LONG
        if self.rird < 0:
            return Direction.SHORT
        return Direction.FLAT

    def alignment(self, direction: Direction, params: RIRDParams | None = None
                  ) -> tuple[str, float]:
        """Score a proposed trade against the differential.

        Returns ``(state, multiplier)`` where state is one of
        ``aligned`` / ``opposed`` / ``hard_opposed`` / ``neutral`` / ``unknown``.
        """
        p = params or RIRDParams()
        if self.rird is None or direction is Direction.FLAT:
            return "unknown", 1.0

        # Signed differential from the trade's point of view: positive means the
        # macro tailwind blows the same way as the trade.
        signed = self.rird * direction.sign

        if abs(signed) < p.significant_differential:
            return "neutral", 1.0
        if signed > 0:
            return "aligned", p.aligned_bonus
        if abs(signed) >= p.hard_block_differential:
            return "hard_opposed", p.opposed_penalty
        return "opposed", p.opposed_penalty


def compute_rate_view(
    instrument: Instrument,
    policy_rates: dict[str, float],
    inflation: dict[str, float],
    yield_10y: dict[str, float],
    yield_2y: dict[str, float],
    yield_params: YieldParams | None = None,
) -> RateView:
    regime, anchor_spread = classify_macro_regime(yield_10y, yield_2y, yield_params)
    return RateView(
        base=instrument.base,
        quote=instrument.quote,
        real_rate_base=real_rate(policy_rates.get(instrument.base), inflation.get(instrument.base)),
        real_rate_quote=real_rate(policy_rates.get(instrument.quote), inflation.get(instrument.quote)),
        rird=rird(policy_rates, inflation, instrument.base, instrument.quote),
        nominal_carry=nominal_differential(policy_rates, instrument.base, instrument.quote),
        spread_base=yield_spread(yield_10y, yield_2y, instrument.base),
        spread_quote=yield_spread(yield_10y, yield_2y, instrument.quote),
        macro_regime=regime,
        anchor_spread=anchor_spread,
    )


def carry_ranking(policy_rates: dict[str, float], inflation: dict[str, float]) -> list[tuple[str, float]]:
    """Currencies ranked by real yield — the classic carry-trade ladder."""
    rows = []
    for ccy in policy_rates:
        r = real_rate(policy_rates.get(ccy), inflation.get(ccy))
        if r is not None:
            rows.append((ccy, r))
    return sorted(rows, key=lambda x: x[1], reverse=True)
