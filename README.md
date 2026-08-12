# MFIE — Macro-Informed Financial Intelligence Engine

A Forex and Crypto analysis tool that treats a chart pattern as a *hypothesis*
and the macroeconomy as the *evidence*. A technical setup only becomes an
actionable signal after it survives a chain of econometric filters — global
liquidity, the yield curve, real interest-rate differentials, PPP valuation
bands, the economic calendar, on-chain tokenomics, market microstructure,
behavioural crowding, and a portfolio CVaR budget.

Every output carries an audit trail. The tool never says "buy EURUSD"; it says
*why*, and shows you which macro rule cut the size in half.

```
Ingestion ─► Storage ─► Technical signal ─► Econometric filter chain ─► Risk sizing ─► Interface
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
├── backtest/            Event-driven backtester and performance metrics
├── interfaces/          Report formatter, Streamlit dashboard, Telegram bot
└── cli.py               Typer CLI

config/params.yaml       Every econometric threshold, version-controlled
tests/                   163 tests: formulas, causality, filter behaviour, risk, cycle
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

---

## Testing

```bash
python -m pytest tests -q          # 163 tests, ~1m45s
```

The suite pins the mathematics (CVaR ≥ VaR, Kelly = p − (1−p)/b, RIRD
antisymmetry, contrarian penalty symmetry), asserts **no lookahead** in the
indicator layer by recomputing on truncated frames, and checks that each filter
blocks what it is supposed to block.

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
