# MFIE — Macro-Informed Financial Intelligence Engine

A Forex and Crypto analysis tool that treats a chart pattern as a *hypothesis*
and the macroeconomy as the *evidence*. A technical setup only becomes an
actionable signal after it survives a chain of econometric filters — global
liquidity, the yield curve, real interest-rate differentials, PPP valuation
bands, the economic calendar, on-chain tokenomics, market microstructure,
behavioural crowding, and a portfolio CVaR budget.

Surviving that chain earns a signal an opinion, not capital. Before anything is
sized, it must also clear its own **costs** in R, produce positive expected
value at a **calibrated** hit rate rather than an assumed one, and then compete
for room in a book whose risk is measured as `√(rᵀCr)` — so six correlated longs
are counted as the single bet they are.

Every output carries an audit trail. The tool never says "buy EURUSD"; it says
*why*, and shows you which macro rule cut the size in half — and which
correlation cut it in half again.

```
Ingestion ─► Storage ─► Technical signal ─► Econometric filter chain ─┐
                                                                      ▼
Interface ◄─ Portfolio allocation ◄─ Expected value ◄─ Risk sizing ◄──┘
             (correlation-aware)      (costs + calibration)
```

> **Analysis only — not financial advice.** Outputs are model estimates from
> public data and can be wrong. Nothing in this repository places an order.
> See [Legal and licensing](#legal-and-licensing).

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt

python -m mfie doctor                             # what is configured
python -m mfie analyze --symbols BTCUSDT,EURUSD   # scored, filtered signals
python -m mfie dashboard                          # Streamlit UI on :8501
```

It runs immediately with **no API keys**. Missing providers fall back to a
deterministic synthetic generator so the whole pipeline is explorable offline —
and the dashboard's *Data sources* tab always tells you which blocks were real
and which were simulated.

To go live, copy `.env.example` to `.env` and fill in whatever you have. The
single most valuable key is **FRED** (free): it powers the real-yield
differential, the yield curve and the PPP module.

---

## What it actually does

### 1. Ingestion

| Data | Provider | Key needed | Falls back to |
|---|---|---|---|
| Crypto OHLCV, order book, funding | Binance | no | synthetic |
| Crypto market cap, stablecoin supply | CoinGecko | optional (Pro) | synthetic |
| Forex OHLCV + retail positioning | OANDA v20 | yes | Alpha Vantage → synthetic |
| Forex OHLCV, US Treasury yields | Alpha Vantage | yes | synthetic |
| Policy rates, CPI, 10Y/2Y, M2, BIS REER | FRED | yes (free) | synthetic |
| Economic calendar | TradingEconomics | yes | synthetic |
| On-chain transfer volume, exchange flows | Glassnode | yes (paid) | CoinGecko volume → synthetic |
| Crypto Fear & Greed | alternative.me | no | synthetic |
| News headlines | NewsAPI | yes | synthetic |
| Social dominance | LunarCrush | yes | synthetic |

A provider never raises into the pipeline. A failed request is logged, tagged,
and replaced — an outage degrades quality, it does not stop the engine.

### 2. Storage

SQLAlchemy models that run unchanged on **SQLite** (default, zero setup) and on
**PostgreSQL/TimescaleDB**. Point `DATABASE_URL` at Postgres and the time-series
tables are promoted to hypertables with 30-day compression policies
automatically. All writes are idempotent upserts, so re-running ingestion over
overlapping windows is safe.

### 3. Technical layer

Every indicator is implemented in pandas/numpy — **TA-Lib is deliberately not a
dependency**, because its C build is the most common reason a Python trading
project fails to install on Windows. Wilder smoothing is used where the standard
definition calls for it, so values match TA-Lib to floating-point noise.

RSI · MACD · Bollinger · Keltner · TTM squeeze · ATR · ADX/DI · Supertrend ·
Donchian · Stochastic · CCI · Williams %R · OBV · MFI · VWAP · Parkinson
volatility · Hurst exponent · swing structure · RSI divergence.

### 4. Strategies

| Strategy | Family | Premise |
|---|---|---|
| `trend_following` | momentum | EMA stack + ADX + Supertrend + MACD must agree |
| `ma_cross` | momentum | Transparent baseline, gated by the 200-EMA |
| `bollinger_reversion` | mean reversion | Fade band extremes — vetoed when ADX says trend |
| `rsi_reversion` | mean reversion | RSI extreme **plus** divergence |
| `donchian_breakout` | breakout | Channel break with mandatory volume confirmation |
| `squeeze_breakout` | breakout | Volatility compression releasing with momentum |
| `pairs_trading` | stat arb | Engle-Granger cointegration, spread z-score, half-life |
| `funding_carry` | carry | Delta-neutral cash-and-carry on perpetual funding |
| `grid` | market making | Volatility-scaled grid, ranging regimes only |
| `ml_classifier` | ML | Gradient boosting, walk-forward validated (opt-in) |

Strategies know nothing about macro, sizing or the account. They propose; the
pipeline disposes.

### 4b. The Market Cycle Compass — the leading bull/bear engine

Everything else in this project judges *a trade*. This judges *the tide*:
whether the market is heading into a bull or bear phase over the next quarter.
It lives in [mfie/alpha/](mfie/alpha/) and runs as `python -m mfie cycle`.

Seven factors, each normalised to `[-1, +1]` where positive always means
risk-on, combined into one score and mapped to a four-phase cycle:

```
EARLY_RECOVERY  ──►  EXPANSION  ──►  LATE_EXPANSION  ──►  CONTRACTION
      ▲                                                        │
      └────────────────────────────────────────────────────────┘
```

| Factor | Prior weight | Typical lead | What it measures |
|---|---|---|---|
| Liquidity impulse | 26% | ~90d | *Acceleration* of M2 and stablecoin supply |
| Credit / curve regime | 20% | ~120d | Curve shape **and** direction (see below) |
| Real rate impulse | 16% | ~60d | Direction of the real policy rate |
| Participation breadth | 14% | ~21d | Share of the universe in an uptrend |
| Valuation stretch | 10% | ~120d | NVT (crypto) / REER (FX) |
| Positioning | 8% | ~10d | Funding + Fear & Greed; havens for FX |
| Trend confirmation | 6% | 0d | Deliberately lagging, deliberately small |

Five design decisions do the real work:

**1. Impulses, not levels.** Every macro factor enters as a rate of change or an
acceleration. The *level* of M2 tells you nothing — the second derivative of
liquidity turns before risk assets do.

**2. The credit factor is non-monotonic.** Almost every model treats an
inverting curve as *the* bear signal. Historically that is early by a year or
more and markets melt up through the inversion. The real trigger is the **bull
steepener** — the curve un-inverting from below, because the front end is
collapsing as the market prices cuts, and cuts get priced when something breaks.

| Curve state | Cycle phase | Score |
|---|---|---|
| Positive and steepening | early recovery | **+1.0** |
| Positive and flattening | mid expansion | +0.3 |
| Inverted and flattening | late cycle, melt-up | −0.2 |
| **Inverted and steepening** | **the trigger** | **−1.0** |

A curve that has recently crossed back above zero keeps the −1.0 for a year.

**3. Lead-aligned aggregation.** A 90-day-lead factor read today describes a
point three months out; a 10-day factor describes next fortnight. Averaging them
raw blurs both. Slow factors are *delayed* — the 90-day factor is read as it
stood 80 days ago — so all of them speak to one forecast horizon.

**4. Shrinkage toward priors, and no sign flipping.** Measured weights are
blended toward the economic priors in proportion to statistical significance,
and clamped to at most 2× their prior. If a factor measures with the **wrong
sign**, the engine falls back to the prior and says so, rather than flipping the
sign and fitting noise — the standard way composite indicators die.

**5. Overlap-corrected statistics.** 700 daily readings of a 63-day forward
return are ~11 independent observations, not 700. During development a factor
scored t = −7.3 uncorrected and t ≈ −0.9 corrected. Every "leading indicator"
that fails out of sample has some version of this mistake in it.

Output is a phase, a score, a per-factor breakdown, an empirical probability of
leaving the phase, and a **divergence flag** — price making highs while the
internals deteriorate, the distribution signature. Two configurations are hard
blocks in the filter chain: a high-beta long into `CONTRACTION`, and any long
into `LATE_EXPANSION` while the divergence flag is up.

```bash
python -m mfie cycle --domain both --validate
```

`--validate` measures whether the composite actually predicts returns on your
data and reports the information coefficient per factor per lead. On the
built-in synthetic data it correctly reports **NO MEASURABLE EDGE** — the macro
and price fixtures are independent, so there is nothing to find, and a model
that claimed otherwise would be lying to you.

### 5. The econometric filter chain

This is the part that makes it an *economist's* tool rather than another
indicator bot. Filters are **asymmetric** — they can veto freely but boost only
modestly (capped at 1.25×), because macro is far better at telling you when not
to trade than when to.

| Level | Filter | Rule |
|---|---|---|
| 1 | **Market Cycle** | Phase alignment from the Compass. Hard-blocks high-beta longs in `CONTRACTION` and longs into `LATE_EXPANSION` with divergence |
| 1 | Global Liquidity | ΔGLI = w₁·%ΔM2 + w₂·%Δstablecoins. Contraction penalises momentum longs, scaled by severity |
| 1 | Yield Curve | 10Y−2Y ≤ 0 ⇒ CONTRACTION. Blocks high-beta longs, penalises the rest |
| 2 | Event Blocker | Hard block ±30 min around a high-impact release (USD events reach crypto too) |
| 3 | Real Yield Differential | ΔRIRD = (Iₐ−πₐ)−(I_b−π_b). Aligned +15%, opposed −35%, >300 bps against ⇒ block |
| 3 | PPP Valuation | REER z-score. Beyond ±2σ suppresses trend continuation; beyond ±2.5σ blocks |
| 3 | Economic Surprise | Time-decayed, σ-standardised surprise index differential |
| 3 | Tokenomics | MV=PQ velocity divergence and NVT-signal z-score (crypto) |
| 4 | Microstructure | LF = spread/ATR. >0.30 halts; 0.10–0.30 widens the stop and shrinks size |
| 4 | Behavioural | Cₜ = 1 − \|(%long−0.5)/0.5\|^k, plus Fear & Greed and news polarity |
| 5 | Regime Alignment | Strategy family weighted by trending / ranging / crash regime |
| 5 | Risk Budget | Portfolio CVaR₉₅ over budget ⇒ every new position scaled down |
| 6 | **Expected Value** | E[R] = p(b−c) − (1−p)(1+c) at a calibrated *p*. Below the hurdle ⇒ block, with the breakeven hit rate quoted |
| 7 | **Portfolio** | Correlation clustering, redundancy decay, per-cluster caps, and a budget on √(rᵀCr) |

### 6. Sizing

```
confidence   = strategy strength × ∏ filter multipliers
risk         = base risk × confidence × CVaR scaler × volatility scalar
             ↳ capped by max risk/trade, a quarter-Kelly ceiling, and portfolio heat
units        = equity × risk / stop distance
```

Because risk is defined by the **stop distance**, widening a stop automatically
shrinks the position. That invariant is what keeps the microstructure filter
honest — and it is asserted in the test suite.

### 6b. The capital allocation layer — costs, calibration, and the book

Everything above this point judges **one trade at a time**. That leaves three
holes, and each of them is a way to lose money while every filter says yes. The
layer in [mfie/portfolio/](mfie/portfolio/) closes them; it runs as
`python -m mfie portfolio`.

**Hole 1 — costs were never priced.** The chain scored setups; nothing ever
subtracted what a round trip costs. Costs are converted to **R**, multiples of
the stop distance, because that is the only unit comparable to a 2R target:

```
c_R = (spread + 2·slippage + commission + carry) / |entry − stop|
```

The denominator is the point. Four basis points against a 3% stop is 0.02R and
irrelevant; against a 0.15% stop it is 0.4R, and the breakeven hit rate on a 2R
target moves from 33% to 47%. Two further details matter:

* **Carry is signed.** A short perp in a positive-funding market is *collecting*,
  and long the higher-yielding currency earns the differential. Treating that as
  a cost throws away the oldest real edge in either market, so `carry` may be
  negative and widen the trade's edge.
* **Holding time comes from diffusion, not a guess.** Expected time to travel a
  distance *D* with per-bar volatility σ scales as (D/σ)² — so a two-ATR stop is
  a ~4-bar trade and a ten-ATR stop a ~100-bar one. That drives the funding accrual.

**Hole 2 — the hit rate was invented.** Position size descends from Kelly,
Kelly needs *p*, and *p* came from `0.35 + 0.25 × confidence`. A guess, never
once checked against a realised trade, feeding the number that decides how much
money is at stake. It is now the **prior** of a Beta-Binomial credibility model:

```
p ~ Beta(κp₀ + w, κ(1−p₀) + l)      E[p] = z·p̂ + (1−z)·p₀,   z = n/(n+κ)
```

The shrinkage is not bolted on — it falls out of the posterior mean. At κ=40, a
strategy needs 40 trades before its own record outvotes the prior, so nine wins
from ten proves nothing and is sized as if it proved nothing. Cells are
hierarchical (`strategy × regime → strategy → prior`), so a thin cell inherits
its parent rather than inventing a rate from three observations.

**Sizing reads the posterior's lower tail, not its mean.** Two strategies both
measuring 55% — one over 400 trades, one over 12 — get very different sizes,
because the thin one's posterior is wide. Uncertainty shrinks the bet with no
rule written to say so, which a point estimate cannot express at all. Kelly on a
point estimate systematically overbets: estimation error is symmetric, but the
cost of overbetting is not, because drawdowns compound geometrically.

`python -m mfie calibrate` prints the check a confidence score has never had to
pass — what actually happened at each confidence level, and a Brier skill score
against the base rate. On the built-in synthetic data it reports **no skill**,
correctly.

**Hole 3 — portfolio heat is a count, not a measure.** Six crypto longs at 1%
each sum to 6% and clear the cap. In a liquidation they are one bet and the book
loses ~6% at once. The measure that knows the difference:

```
σ_book = √(rᵀ C r)        with   C̃ᵢⱼ = dᵢ dⱼ ρᵢⱼ
```

Signing by direction `d ∈ {+1,−1}` is what separates a risk model that
understands a pairs trade from one that double-counts it: long BTC + long ETH is
one bet, long BTC + **short** ETH is a spread whose risk is a fraction of either
leg. Six perfectly correlated trades return 6% and the cap bites; six independent
ones return 2.4% and the book is *allowed more risk* — which is the half that
makes money rather than merely saving it. A fixed additive cap has to be set low
enough to survive the worst case, so it under-allocates in every other case.

Unknown correlation is set to **+0.35, not zero**. Assuming independence when
the history is too short is the most expensive assumption in portfolio construction.

The allocator then, in order: drops trades whose expected value does not clear
its costs; clusters candidates on the HRP metric `d = √((1−ρ)/2)` with single
linkage (risk contagion is transitive — if A moves with B and B with C, all three
are one group); applies **redundancy decay**, so the *k*-th idea in a cluster
keeps `1/(1 + kλ)` of its size; caps each cluster; and scales the whole book to a
budget on `√(rᵀCr)`. Effective risk is homogeneous of degree 1 in the weights, so
one uniform scale lands exactly on the budget with no iteration.

| Question | Answered by | Where it bites |
|---|---|---|
| What does the round trip cost, in R? | `portfolio/costs.py` | Tight stops, wide spreads, funding |
| What hit rate has this actually earned? | `portfolio/calibration.py` | Unproven strategies size smaller |
| Does the arithmetic work after costs? | `portfolio/edge.py` | Positive-looking, negative-EV trades |
| How much of this book is one bet? | `portfolio/construction.py` | Correlated clones, hedges |
| What should the book hold? | `portfolio/allocator.py` | The final size, with reasons |

Every resize carries its reason into the same audit trail as the filters, so a
position that was cut says which step cut it and by how much.

```bash
python -m mfie portfolio --detail
python -m mfie calibrate --symbol BTCUSDT --save
```

### 6c. Was the backtest real, or the best of many guesses?

A Sharpe ratio with no multiple-testing correction attached is not a result. The
project ships ten strategies and ~50 tunable thresholds; run them all, keep the
best, and the winner's Sharpe estimates its edge *plus the maximum of fifty
draws of noise*. Under the null, the expected best Sharpe across *N* trials is

```
E[max SR] ≈ √V[SR] · [ (1−γ)·Z⁻¹(1 − 1/N) + γ·Z⁻¹(1 − 1/(Ne)) ]
```

so twenty trials on a zero-skill strategy produce an expected best Sharpe near
0.5 for free. `mfie/backtest/significance.py` reports the **Deflated Sharpe
Ratio** — the probability the observed Sharpe survives that benchmark, given the
sample's own skew and kurtosis. Negative skew and fat tails, the signature of
every trend strategy, make a given Sharpe *less* impressive, not more. Every
`PerformanceReport` now carries it, alongside the minimum track record length
needed for significance, which is usually far longer than the backtest.

### 7. Backtesting

Bar-close decisions with **next-bar fills**, costs on both sides, ATR-scaled
slippage, and the pessimistic assumption that a stop fills before a target when
one bar spans both. `--compare` runs the same strategies with and without the
macro chain, which is the experiment that justifies the filters existing at all.

---

## Command reference

```bash
python -m mfie analyze  --symbols EURUSD,BTCUSDT --timeframe 4h --audit
python -m mfie analyze  --class crypto --json          # machine-readable
python -m mfie watch                                   # one line per instrument
python -m mfie macro    --verbose                      # rates, curves, calendar
python -m mfie cycle    --domain both --validate        # bull/bear cycle read
python -m mfie portfolio --detail                      # correlation-aware allocation
python -m mfie calibrate --symbol BTCUSDT --save       # is 'confidence' a probability?
python -m mfie backtest EURUSD --bars 3000 --compare   # with vs without filters
python -m mfie pairs    --class crypto                 # cointegration scan
python -m mfie train    BTCUSDT --algorithm random_forest
python -m mfie ingest   --symbols BTCUSDT,ETHUSDT      # persist to the database
python -m mfie strategies                              # list the library
python -m mfie doctor                                  # config & connectivity
python -m mfie init-db                                 # create schema / hypertables
python -m mfie dashboard --port 8501
python -m mfie bot      --interval 900                 # Telegram alerts
```

---

## Project layout

```
mfie/
├── config.py            Settings (env) + Params (config/params.yaml)
├── core/                Domain types, universe, shared utilities
├── data/                Providers, synthetic fallback, DataHub
├── storage/             SQLAlchemy models, engine, repository
├── technical/           Indicator library (pure pandas/numpy)
├── econ/                liquidity · rates · valuation · surprise ·
│                        tokenomics · microstructure · behavioral · risk
├── alpha/               Market Cycle Compass: factors, lead-lag, cycle engine
├── regime/              Rule-based + Gaussian-mixture regime detection
├── strategies/          The strategy library and its registry
├── ml/                  Feature engineering + walk-forward direction model
├── pipeline/            Filter chain, sizing, orchestration engine
├── portfolio/           costs · calibration · expected value · construction ·
│                        allocator — the book-level decision
├── backtest/            Event-driven backtester, metrics, significance tests
├── interfaces/          Report formatter, Streamlit dashboard, Telegram bot
└── cli.py               Typer CLI

config/params.yaml       Every econometric threshold, version-controlled
tests/                   211 tests: formulas, causality, filter behaviour, risk,
                         cycle, costs, calibration, allocation, significance
```

---

## Tuning

`config/params.yaml` holds every threshold — GLI weights, PPP bands, CVaR
budget, Kelly cap, blackout window. Nothing there is sacred; the defaults are
plausible starting points, not fitted values. After changing one, run:

```bash
python -m mfie backtest EURUSD --compare
```

If the filtered run does not improve risk-adjusted return, that filter is
costing you money on that instrument and the threshold needs rethinking.

Two settings people usually want to change first:

* `risk.target_annual_vol` (default 15%) is a **portfolio-level** target applied
  per position, so a 65%-vol crypto pair gets sized far smaller than a 9%-vol FX
  pair. Raise it for a crypto-only book.
* `events.blackout_minutes_before/after` (default ±30) is aggressive. Widen it
  if you trade the majors; narrow it if you are explicitly trading releases.

Three more that change how much capital moves:

* `costs.commission_bps_per_side` (default 4 bps) must match **your** venue. It
  is the single most under-set number in retail trading: too low and the
  expected-value gate waves through trades that lose money slowly.
* `calibration.prior_strength` (default κ=40) is how sceptical the engine is of
  its own track record. Lower it only if you have years of trades and trust them.
* `portfolio.max_effective_risk` (default 4.5%) budgets `√(rᵀCr)`, not the sum,
  so it can safely exceed `risk.max_portfolio_risk` (6% additive). Correlated
  risk is never larger than additive risk — that is the whole point of measuring
  it. Set `portfolio.enabled: false` to go back to per-trade sizing only.

---

## Testing

```bash
python -m pytest tests -q          # 211 tests
```

The suite pins the mathematics (CVaR ≥ VaR, Kelly = p − (1−p)/b, RIRD
antisymmetry, contrarian penalty symmetry), asserts **no lookahead** in the
indicator layer by recomputing on truncated frames, and checks that each filter
blocks what it is supposed to block.

The allocation layer is tested as a set of statements about behaviour, because
that is what the money depends on:

* costs in R must double when the stop halves, and a short perp must *collect*
  positive funding;
* `E[R]` must be exactly zero at the breakeven hit rate — the two formulas are
  derived independently and have to agree, or the gate rejects the wrong trades;
* credibility must follow `z = n/(n+κ)` exactly, and a 12-trade cell must size
  smaller than a 600-trade cell with the same sample mean;
* two correlated longs must measure as ~one bet, a hedge as far less risk than
  either leg, and risk contributions must sum to total book risk (Euler's
  theorem — the decomposition is exact, and a fuzz test asserts effective risk
  never exceeds additive heat across random long/short books);
* clustering must be transitive, and two strategies on one instrument must not
  pass as diversification;
* units must be re-derived from the allocated risk, so the sizing invariant
  survives the portfolio pass;
* pure noise searched over 50 trials must **not** be reported as significant.

---

## Known limitations

Stated plainly, because a tool that hides these is worse than no tool:

* **Synthetic data is a fixture, not a simulator.** Backtest numbers produced
  offline describe a random process, not a market. They exercise the code; they
  say nothing about edge.
* **The filter thresholds are priors, not fitted parameters.** They encode
  standard macro relationships. They have not been optimised, and optimising
  them on your own history will overfit unless you are disciplined about
  out-of-sample testing.
* **Cointegration is unstable.** `pairs` re-tests on every call and reports how
  many pairwise tests were run, because at p<0.05 across 28 pairs you expect
  false positives by construction.
* **The backtester does not model** partial fills, funding on leveraged
  positions, borrow costs, exchange outages, or your own market impact.
* **The Cycle Compass needs history to be worth anything.** With under 250
  daily observations it uses the economic priors unchanged and measures nothing.
  It wants years of FRED data, not weeks.
* **Retail positioning is only real with an OANDA key.** Without it the
  behavioural filter is running on simulated data.
* **The ML model is opt-in and self-rejecting.** If walk-forward accuracy does
  not beat the majority-class baseline by 2 points and AUC 0.53, it emits
  nothing. Most of the time, on most instruments, it should emit nothing.
* **Calibration inherits its data's bias.** Fitted on backtest trades, it learns
  the backtester's fill assumptions as much as the market's behaviour, and
  survivorship in your own saved runs will flatter it. It is a measurement of
  the system's realised record, not a forecast — and on synthetic data it
  correctly reports no skill.
* **Correlations are unstable and rise exactly when it hurts.** The allocator
  measures the recent past; in a liquidation everything goes to 1.0 and the
  measured diversification evaporates. The `default_correlation` prior for thin
  pairs is pessimistic for this reason, but a correlation estimated over 250
  calm bars will still overstate diversification in the week it matters.
* **The cost model does not know your venue.** Tiered fees, maker rebates,
  borrow costs, and market impact at size are all absent. Impact in particular
  means the cost of a large position is understated, and it is understated worst
  in exactly the illiquid instruments where the microstructure filter is already
  nervous.
* **Expected holding time is a diffusion approximation.** Real trades exit on
  signals and time stops, not on first-passage of a driftless walk. It is right
  in shape and wrong in detail, which is fine for accruing funding and would not
  be fine for anything else.

---

## Legal and licensing

* **Not financial advice.** This is an analysis tool. It does not know your
  circumstances, and its outputs are model estimates that can be wrong.
* **Data licensing.** You may not redistribute data from OANDA, Glassnode,
  TradingEconomics or similar providers without a commercial redistribution
  licence. This project is architected so that **each user supplies their own
  API keys** and data stays local — that is the design that keeps personal use
  clearly inside the terms. If you ever expose this to other people, you need
  either their keys or your own redistribution licence.
* **Execution.** Nothing here places orders. If you add execution, add a kill
  switch and per-day loss limits before you add anything else.

MIT licensed.
