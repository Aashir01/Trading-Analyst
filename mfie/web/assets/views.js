/* ==========================================================================
   Views.

   Each export takes the current state and returns a DOM node. They are pure
   with respect to the store — no view reaches back and mutates shared state
   during render — which is what keeps a re-render safe to run at any time.

   The information hierarchy across every view is the same, and it is the
   argument the engine makes: what the book is doing, then why, then the
   evidence. A number never appears without the reasoning reachable in one
   click, because the audit trail is the product.
   ========================================================================== */

import {
  badge, callout, card, clusterColor, directionVariant, emptyState, frag, h, icon,
  metric, verdictVariant,
} from "./core.js";
import {
  candlestick, chartColor, cssVar, divergingBars, fmt, groupedBars, heatmap,
  lineChart, phaseDial, sparkline,
} from "./charts.js";

/* ---- Shared bits -------------------------------------------------------- */

const PHASE_LABEL = {
  early_recovery: "Early recovery",
  expansion: "Expansion",
  late_expansion: "Late expansion",
  contraction: "Contraction",
  // The Compass reports NEUTRAL when no phase has been confirmed — usually
  // because the history is too short for the state machine to commit.
  neutral: "Neutral",
  unknown: "Unknown",
};

const REGIME_LABEL = {
  trending_high_vol: "Trending · high vol",
  trending_low_vol: "Trending · low vol",
  ranging: "Ranging",
  crash: "Crash",
  unknown: "Unknown",
};

function phaseTone(phase) {
  if (phase === "early_recovery" || phase === "expansion") return "long";
  if (phase === "contraction") return "short";
  return "warn";
}

function riskTone(used, budget) {
  if (!budget) return "";
  const ratio = used / budget;
  if (ratio > 0.9) return "danger";
  if (ratio > 0.7) return "warn";
  return "";
}

function chartHost(className = "chart") {
  return h(`div.${className}`);
}

/* Charts are drawn after the node is in the document: several of them measure
   computed CSS custom properties, which do not resolve on a detached node. */
function draw(node, fn) {
  requestAnimationFrame(() => {
    if (node.isConnected) fn(node);
  });
  return node;
}

/* ==========================================================================
   Overview
   ========================================================================== */
export function overviewView(state) {
  const analysis = state.analysis;
  if (!analysis) return emptyState({ title: "No scan loaded yet", body: "Run a scan to populate the dashboard." });

  const { macro, portfolio, summary } = analysis;
  const params = state.params;
  const budget = params?.portfolio?.max_effective_risk ?? 0.045;
  const tradable = analysis.signals.filter((s) => !s.blocked && s.risk_fraction > 0);
  const blocked = analysis.signals.filter((s) => s.blocked);

  const cycle = analysis.cycles?.crypto || analysis.cycles?.fx || null;

  return h("div.view", {}, [
    /* --- Headline numbers ------------------------------------------------ */
    h("section.section", {}, [
      h("div.grid.grid--metrics", {}, [
        metric({
          label: "Actionable",
          value: String(tradable.length),
          meta: [`${analysis.signals.length} generated`, badge(`${blocked.length} blocked`, "neutral")],
        }),
        metric({
          label: "Correlated risk",
          value: fmt.pct(portfolio?.effective_risk ?? 0, 2),
          tone: riskTone(portfolio?.effective_risk ?? 0, budget),
          meta: [`of ${fmt.pct(budget, 1)} budget`],
          hint: "sqrt(r'Cr) — the book's risk once correlations are counted, not the sum of positions.",
        }),
        metric({
          label: "Diversification",
          value: `${fmt.num(portfolio?.diversification_ratio ?? 1, 2)}x`,
          meta: [`${portfolio?.clusters ?? 0} independent clusters`],
          hint: "Additive heat divided by correlated risk. 1.0x means the book is one trade wearing several names.",
        }),
        metric({
          label: "Liquidity",
          value: (macro.liquidity.regime || "").toUpperCase(),
          tone: macro.liquidity.regime === "contraction" ? "short" : macro.liquidity.regime === "expansion" ? "long" : "",
          meta: [`ΔGLI ${fmt.signedPct(macro.liquidity.gli_delta, 2)}`],
        }),
        metric({
          label: "Yield curve",
          value: fmt.signedPct(curveFor(macro, "USD"), 2),
          tone: (curveFor(macro, "USD") ?? 0) < 0 ? "short" : "",
          meta: [(curveFor(macro, "USD") ?? 0) < 0 ? "US 10Y-2Y inverted" : "US 10Y-2Y positive"],
        }),
        metric({
          label: "Fear & Greed",
          value: macro.fear_greed === null ? "—" : String(Math.round(macro.fear_greed)),
          tone: macro.fear_greed >= 80 ? "warn" : macro.fear_greed <= 20 ? "info" : "",
          meta: [fearGreedLabel(macro.fear_greed)],
        }),
      ]),
    ]),

    /* --- Warnings the reader must not have to hunt for ------------------- */
    ...overviewAlerts(analysis, budget),

    /* --- The book and the cycle side by side ----------------------------- */
    h("div.grid.grid--wide", {}, [
      card({
        title: "Top signals",
        hint: `${tradable.length} sized by the portfolio layer`,
        actions: [
          h("a.btn.btn--ghost", { href: "/signals", "data-route": "" }, "View all"),
        ],
        flush: true,
        body: tradable.length
          ? h("div.table-wrap", {}, [signalTable(tradable.slice(0, 8))])
          : emptyState({
              title: "Nothing cleared the chain",
              body: blocked.length
                ? `${blocked.length} signal${blocked.length === 1 ? " was" : "s were"} generated but every one was vetoed or failed its expected-value test. That is a normal outcome — the filters exist to say no.`
                : "No strategy proposed a setup on this universe and timeframe.",
            }),
      }),
      cycle
        ? card({
            title: "Market cycle",
            hint: cycle.domain === "crypto" ? "Crypto" : "FX",
            actions: [h("a.btn.btn--ghost", { href: "/cycle", "data-route": "" }, "Detail")],
            body: [
              draw(chartHost(), (node) =>
                phaseDial(node, cycle.score, {
                  size: 220,
                  label: PHASE_LABEL[cycle.phase] || cycle.phase,
                  sublabel: `${cycle.days_in_phase} days in phase · ${fmt.pct(cycle.transition_probability, 0)} chance of leaving`,
                })
              ),
              cycle.divergence_flag &&
                callout(
                  "Divergence flag: price is making highs while the internals deteriorate. Longs into late expansion are hard-blocked while this holds.",
                  "warn"
                ),
              h("div.dl", { style: { marginTop: "var(--space-4)" } }, [
                dlItem("Momentum", fmt.signed(cycle.momentum, 2)),
                dlItem("Confidence", fmt.pct(cycle.confidence, 0)),
                dlItem("Data", cycle.data_quality),
              ]),
            ],
          })
        : null,
    ]),

    /* --- Allocation ------------------------------------------------------ */
    portfolio && portfolio.held.length
      ? card({
          title: "Capital allocation",
          hint: "Requested by per-trade sizing vs allocated by the book",
          actions: [h("a.btn.btn--ghost", { href: "/portfolio", "data-route": "" }, "Detail")],
          body: draw(chartHost(), (node) =>
            groupedBars(node, allocationRows(portfolio), { label: "Allocation by position" })
          ),
        })
      : null,

    macroStrip(macro),
  ]);
}

function curveFor(macro, currency) {
  return macro.rates.find((r) => r.currency === currency)?.curve ?? null;
}

function fearGreedLabel(value) {
  if (value === null || value === undefined) return "unavailable";
  if (value >= 80) return "extreme greed";
  if (value >= 60) return "greed";
  if (value >= 40) return "neutral";
  if (value >= 20) return "fear";
  return "extreme fear";
}

function overviewAlerts(analysis, budget) {
  const alerts = [];
  const portfolio = analysis.portfolio;

  if (portfolio?.held?.length && portfolio.diversification_ratio < 1.2) {
    alerts.push(
      callout(
        `Diversification is ${fmt.num(portfolio.diversification_ratio, 2)}x — these positions are close to being one trade under several names. Additive heat would have read ${fmt.pct(portfolio.gross_risk, 2)} and missed it.`,
        "warn"
      )
    );
  }
  if (portfolio && portfolio.effective_risk > budget * 0.95) {
    alerts.push(
      callout(
        `Correlated risk ${fmt.pct(portfolio.effective_risk, 2)} is at the ${fmt.pct(budget, 1)} budget. New positions will be scaled down.`,
        "danger"
      )
    );
  }
  if (portfolio && portfolio.calibration_observations === 0) {
    alerts.push(
      callout(
        "Hit rates are running on the prior — no realised trades have been recorded yet. Run `mfie calibrate --symbol BTCUSDT --save` to give the sizer something measured to work from.",
        "info"
      )
    );
  }
  const synthetic = analysis.macro.sources_total - analysis.macro.sources_live;
  if (synthetic > 0 && analysis.macro.sources_live === 0) {
    alerts.push(
      callout(
        `Every data block is simulated (${synthetic} of ${analysis.macro.sources_total}). The pipeline is exercising its code paths against a deterministic fixture, not the market. Add API keys to see real readings.`,
        "warn"
      )
    );
  }
  return alerts.length ? [h("section.section", {}, alerts)] : [];
}

function macroStrip(macro) {
  const next = macro.events.slice(0, 4);
  return h("div.grid.grid--halves", {}, [
    card({
      title: "Real rates",
      hint: "Policy rate minus trailing CPI",
      flush: true,
      body: h("div.table-wrap", {}, [
        h("table.table", {}, [
          h("thead", {}, [
            h("tr", {}, [
              h("th", {}, "Currency"),
              h("th.align-right", {}, "Policy"),
              h("th.align-right", {}, "CPI"),
              h("th.align-right", {}, "Real"),
              h("th.align-right", {}, "10Y-2Y"),
            ]),
          ]),
          h("tbody", {}, macro.rates.map((row) =>
            h("tr", {}, [
              h("td.table__primary", {}, row.currency),
              h("td.num", {}, fmt.signedPct(row.policy_rate, 2)),
              h("td.num", {}, fmt.signedPct(row.inflation, 2)),
              h("td.num", { class: (row.real_rate ?? 0) < 0 ? "short" : "long" },
                fmt.signedPct(row.real_rate, 2)),
              h("td.num", { class: (row.curve ?? 0) < 0 ? "short" : "" },
                fmt.signedPct(row.curve, 2)),
            ])
          )),
        ]),
      ]),
    }),
    card({
      title: "Next releases",
      hint: "High-impact events blackout trading ±30 min",
      flush: true,
      body: next.length
        ? h("div.table-wrap", {}, [
            h("table.table", {}, [
              h("thead", {}, [
                h("tr", {}, [
                  h("th", {}, "Event"),
                  h("th", {}, "Currency"),
                  h("th", {}, "Impact"),
                  h("th.align-right", {}, "In"),
                ]),
              ]),
              h("tbody", {}, next.map((event) =>
                h("tr", {}, [
                  h("td.table__primary", {}, [
                    event.name,
                    h("span.table__sub", {}, fmt.time(event.ts, true)),
                  ]),
                  h("td", {}, event.currency),
                  h("td", {}, [badge(event.impact, event.impact === "high" ? "warn" : "neutral")]),
                  h("td.num", {}, `${fmt.num(event.hours_away, 1)}h`),
                ])
              )),
            ]),
          ])
        : emptyState({ title: "No scheduled events", body: "Nothing upcoming in the loaded calendar window." }),
    }),
  ]);
}

function dlItem(key, value) {
  return h("div.dl__item", {}, [h("span.dl__key", {}, key), h("span.dl__val", {}, value)]);
}

/* ==========================================================================
   Signals
   ========================================================================== */
export function signalsView(state, { onToggleBlocked }) {
  const analysis = state.analysis;
  if (!analysis) return emptyState({ title: "No scan loaded" });

  const tradable = analysis.signals.filter((s) => !s.blocked && s.risk_fraction > 0);
  const sized_out = analysis.signals.filter((s) => !s.blocked && s.risk_fraction <= 0);
  const blocked = analysis.signals.filter((s) => s.blocked);

  return h("div.view", {}, [
    h("section.section", {}, [
      h("div.grid.grid--metrics", {}, [
        metric({ label: "Generated", value: String(analysis.signals.length) }),
        metric({ label: "Actionable", value: String(tradable.length), tone: tradable.length ? "long" : "" }),
        metric({ label: "Below threshold", value: String(sized_out.length) }),
        metric({ label: "Blocked", value: String(blocked.length), tone: blocked.length ? "warn" : "" }),
      ]),
    ]),

    h("section.section", {}, [
      h("div.section__head", {}, [
        h("h2.section__title", {}, "Actionable"),
        h("span.section__hint", {}, "Sized by the portfolio layer, after costs and correlation"),
      ]),
      tradable.length
        ? frag(tradable.map((signal) => signalCard(signal)))
        : emptyState({
            title: "Nothing is tradable right now",
            body: "Every proposed setup was either vetoed by a macro rule, failed its expected-value test after costs, or sized below the minimum. That is the chain working, not failing.",
          }),
    ]),

    sized_out.length
      ? h("section.section", {}, [
          h("div.section__head", {}, [
            h("h2.section__title", {}, "Below sizing threshold"),
            h("span.section__hint", {}, "Passed every filter but sized to nothing"),
          ]),
          card({
            flush: true,
            body: h("div.table-wrap", {}, [
              h("table.table", {}, [
                h("thead", {}, [
                  h("tr", {}, [
                    h("th", {}, "Instrument"),
                    h("th", {}, "Direction"),
                    h("th", {}, "Strategy"),
                    h("th.align-right", {}, "Confidence"),
                    h("th", {}, "Why no position"),
                  ]),
                ]),
                h("tbody", {}, sized_out.map((signal) =>
                  h("tr", {}, [
                    h("td.table__primary", {}, signal.name),
                    h("td", {}, [badge(signal.direction, directionVariant(signal.direction))]),
                    h("td.mono", {}, signal.strategy),
                    h("td.num", {}, fmt.pct(signal.confidence, 0)),
                    h("td", {}, signal.block_reasons[signal.block_reasons.length - 1] || "no size allocated"),
                  ])
                )),
              ]),
            ]),
          }),
        ])
      : null,

    blocked.length
      ? h("section.section", {}, [
          h("div.section__head", {}, [
            h("h2.section__title", {}, "Blocked"),
            h("span.section__hint", {}, "Vetoed, with the rule that vetoed them"),
            h("div.section__actions", {}, [
              h("button.btn.btn--ghost", {
                type: "button",
                onclick: onToggleBlocked,
                "aria-pressed": String(state.showBlocked),
              }, state.showBlocked ? "Collapse" : "Expand"),
            ]),
          ]),
          state.showBlocked
            ? frag(blocked.map((signal) => signalCard(signal)))
            : card({
                flush: true,
                body: h("div.table-wrap", {}, [
                  h("table.table", {}, [
                    h("thead", {}, [
                      h("tr", {}, [
                        h("th", {}, "Instrument"),
                        h("th", {}, "Direction"),
                        h("th", {}, "Strategy"),
                        h("th", {}, "Blocked by"),
                      ]),
                    ]),
                    h("tbody", {}, blocked.map((signal) =>
                      h("tr", {}, [
                        h("td.table__primary", {}, signal.name),
                        h("td", {}, [badge(signal.direction, directionVariant(signal.direction))]),
                        h("td.mono", {}, signal.strategy),
                        h("td", {}, signal.block_reasons[0] || "unknown"),
                      ])
                    )),
                  ]),
                ]),
              }),
        ])
      : null,
  ]);
}

function signalTable(signals) {
  return h("table.table", {}, [
    h("thead", {}, [
      h("tr", {}, [
        h("th", {}, "Instrument"),
        h("th", {}, "Dir"),
        h("th", {}, "Strategy"),
        h("th.align-right", {}, "Conf"),
        h("th.align-right", {}, "E[R]"),
        h("th.align-right", {}, "Risk"),
        h("th", {}, "Verdict"),
      ]),
    ]),
    h("tbody", {}, signals.map((signal) =>
      h("tr", {}, [
        h("td.table__primary", {}, [
          signal.name,
          h("span.table__sub", {}, `entry ${fmt.price(signal.entry)}`),
        ]),
        h("td", {}, [badge(signal.direction, directionVariant(signal.direction))]),
        h("td.mono", { style: { fontSize: "var(--text-xs)" } }, signal.strategy),
        h("td.num", {}, fmt.pct(signal.confidence, 0)),
        h("td.num", { class: (signal.edge?.expected_r ?? 0) > 0 ? "long" : "short" },
          signal.edge ? fmt.signed(signal.edge.expected_r, 2) : "—"),
        h("td.num", {}, fmt.pct(signal.risk_fraction, 2)),
        h("td", {}, [badge(signal.verdict, verdictVariant(signal.verdict))]),
      ])
    )),
  ]);
}

function signalCard(signal) {
  const blocked = signal.blocked;
  const dirClass = blocked ? "blocked" : directionVariant(signal.direction);
  const arrow = signal.direction === "long" ? "▲" : signal.direction === "short" ? "▼" : "—";

  const details = h("details.signal", { dataset: { blocked: String(blocked) } }, [
    h("summary.signal__head", {}, [
      h(`div.signal__dir.signal__dir--${dirClass}`, {}, blocked ? "⨯" : arrow),
      h("div.signal__id", {}, [
        h("div.signal__symbol", {}, signal.name),
        h("div.signal__strategy", {}, signal.strategy),
      ]),
      h("div.signal__stats", {}, [
        stat("Confidence", fmt.pct(signal.confidence, 0)),
        signal.edge && stat("E[R]", fmt.signed(signal.edge.expected_r, 2),
          signal.edge.expected_r > 0 ? "long" : "short"),
        stat("Risk", blocked ? "—" : fmt.pct(signal.risk_fraction, 2)),
      ]),
      badge(signal.verdict, verdictVariant(signal.verdict)),
      icon("chevron", "signal__chevron"),
    ]),
    h("div.signal__body", {}, [
      blocked && signal.block_reasons.length
        ? callout(signal.block_reasons[0], "danger")
        : null,

      h("div.levels", {}, [
        level("Entry", signal.entry, "entry"),
        level("Stop", signal.stop, "stop"),
        signal.take_profit ? level("Target", signal.take_profit, "target") : null,
        signal.reward_risk
          ? h("div.level", {}, [
              h("div.level__label", {}, "Reward : risk"),
              h("div.level__value", {}, `${fmt.num(signal.reward_risk, 1)}R`),
            ])
          : null,
      ]),

      signal.edge ? edgeBlock(signal.edge) : null,

      signal.rationale.length
        ? h("div", {}, [
            h("div.label", { style: { marginBottom: "var(--space-2)" } }, "Strategy rationale"),
            h("ul", { style: { display: "flex", flexDirection: "column", gap: "var(--space-1)" } },
              signal.rationale.map((line) =>
                h("li.secondary", { style: { fontSize: "var(--text-sm)" } }, `— ${line}`)
              )),
          ])
        : null,

      h("div", {}, [
        h("div.label", { style: { marginBottom: "var(--space-2)" } }, "Filter chain"),
        auditTrail(signal.outcomes),
      ]),

      h("div", {}, [
        h("div.label", { style: { marginBottom: "var(--space-2)" } }, "Confidence impact"),
        draw(chartHost(), (node) =>
          divergingBars(node, impactRows(signal.outcomes), {
            unit: "%",
            format: (v) => fmt.signed(v, 0),
            label: "How each filter moved confidence",
            emptyMessage: "No filter changed this signal's confidence.",
          })
        ),
      ]),
    ]),
  ]);

  return details;
}

function stat(label, value, tone = "") {
  return h("div.signal__stat", {}, [
    h("span.signal__stat-label", {}, label),
    h("span.signal__stat-value", { class: tone }, value),
  ]);
}

function level(label, value, kind) {
  return h(`div.level.level--${kind}`, {}, [
    h("div.level__label", {}, label),
    h("div.level__value", {}, fmt.price(value)),
  ]);
}

function edgeBlock(edge) {
  const clears = edge.hit_rate.lower >= edge.breakeven_hit_rate;
  return h("div", {}, [
    h("div.label", { style: { marginBottom: "var(--space-2)" } }, "Expected value, after costs"),
    h("div.dl", {}, [
      dlItem("E[R] net", fmt.signed(edge.expected_r, 3)),
      dlItem("Round-trip cost", `${fmt.num(edge.cost_r, 3)}R`),
      dlItem("Breakeven hit rate", fmt.pct(edge.breakeven_hit_rate, 0)),
      dlItem("Calibrated hit rate", fmt.pct(edge.hit_rate.mean, 0)),
      dlItem("At sizing quantile", fmt.pct(edge.hit_rate.lower, 0)),
      dlItem("Kelly cap", fmt.pct(edge.kelly, 2)),
    ]),
    h("p.secondary", { style: { fontSize: "var(--text-sm)", marginTop: "var(--space-3)" } },
      `Hit rate from ${edge.hit_rate.source} (${edge.hit_rate.observations} realised trades, ` +
      `${fmt.pct(edge.hit_rate.credibility, 0)} credibility). ` +
      (clears
        ? `The lower bound clears breakeven, so the edge survives its own uncertainty.`
        : `The lower bound does not clear breakeven — the edge is not distinguishable from noise at this sample size.`)),
  ]);
}

function auditTrail(outcomes) {
  const rows = outcomes.filter((o) => !(o.action === "pass" && o.reason === "not applicable"));
  if (!rows.length) return emptyState({ title: "No filters applied" });

  return h("div.audit", {}, rows.map((outcome) =>
    h("div.audit__row", { dataset: { action: outcome.action } }, [
      h("span.audit__name", {}, outcome.name),
      h("span.audit__reason", {}, outcome.reason || "—"),
      h("span.audit__impact", {
        class: outcome.action === "block" ? "danger"
          : outcome.multiplier > 1.001 ? "info"
          : outcome.multiplier < 0.999 ? "warn" : "muted",
      }, outcome.action === "block" ? "BLOCK" : `${fmt.num(outcome.multiplier, 2)}x`),
    ])
  ));
}

function impactRows(outcomes) {
  return outcomes
    .filter((o) => !(o.action === "pass" && o.reason === "not applicable"))
    .filter((o) => Math.abs(o.impact) > 0.01 || o.action === "block")
    .map((o) => ({
      label: o.name,
      value: o.action === "block" ? -100 : o.impact,
      color: o.action === "block" ? cssVar("--status-danger") : undefined,
      tooltip: `<div class="tooltip__row"><span>${o.reason}</span></div>`,
    }));
}

/* ==========================================================================
   Portfolio
   ========================================================================== */
export function portfolioView(state) {
  const analysis = state.analysis;
  const portfolio = analysis?.portfolio;
  const budget = state.params?.portfolio?.max_effective_risk ?? 0.045;
  const additiveCap = state.params?.risk?.max_portfolio_risk ?? 0.06;

  if (!portfolio) {
    return emptyState({
      title: "Portfolio allocation is off",
      body: "Set `portfolio.enabled: true` in config/params.yaml to let the book-level layer size positions.",
    });
  }
  if (!portfolio.held.length && !portfolio.dropped.length) {
    return emptyState({
      title: "No candidates reached the portfolio layer",
      body: "Every signal was vetoed upstream or sized below the minimum, so there was nothing to allocate between.",
    });
  }

  return h("div.view", {}, [
    h("section.section", {}, [
      h("div.grid.grid--metrics", {}, [
        metric({
          label: "Additive heat",
          value: fmt.pct(portfolio.gross_risk, 2),
          meta: [`${fmt.pct(portfolio.gross_requested, 2)} requested`, `cap ${fmt.pct(additiveCap, 1)}`],
          hint: "The sum of position risks — the number that cannot tell six correlated longs from six independent ones.",
        }),
        metric({
          label: "Correlated risk",
          value: fmt.pct(portfolio.effective_risk, 2),
          tone: riskTone(portfolio.effective_risk, budget),
          meta: [`budget ${fmt.pct(budget, 1)}`],
          hint: "sqrt(r'Cr) with correlations signed by trade direction. This is the real exposure.",
        }),
        metric({
          label: "Diversification",
          value: `${fmt.num(portfolio.diversification_ratio, 2)}x`,
          tone: portfolio.diversification_ratio < 1.2 ? "warn" : "",
          meta: [`${portfolio.clusters} clusters`],
        }),
        metric({
          label: "Book E[R]",
          value: fmt.signed(portfolio.expected_r, 2),
          tone: portfolio.expected_r > 0 ? "long" : "short",
          meta: ["per unit of risk"],
        }),
        metric({
          label: "Concentration",
          value: fmt.pct(portfolio.concentration ?? 0, 0),
          tone: (portfolio.concentration ?? 0) > 0.5 ? "warn" : "",
          meta: ["largest single share of book risk"],
        }),
        metric({
          label: "Hit rate source",
          value: portfolio.calibration_source === "trades" ? "MEASURED" : "PRIOR",
          tone: portfolio.calibration_source === "trades" ? "" : "warn",
          meta: [`${portfolio.calibration_observations} realised trades`],
        }),
      ]),
    ]),

    /* Budget meter: the distance to the line is the decision, so draw it. */
    card({
      title: "Risk budget",
      hint: "Correlated risk against the diversified budget",
      body: [
        h("div.meter", {}, [
          h("div.meter__head", {}, [
            h("span.secondary", {}, `Correlated ${fmt.pct(portfolio.effective_risk, 2)}`),
            h("span.muted", {}, `Budget ${fmt.pct(budget, 2)}`),
          ]),
          h("div.meter__track", {}, [
            h("div.meter__fill", {
              class: riskTone(portfolio.effective_risk, budget) === "danger" ? "meter__fill--danger"
                : riskTone(portfolio.effective_risk, budget) === "warn" ? "meter__fill--warn" : "",
              style: { width: `${Math.min(100, (portfolio.effective_risk / budget) * 100)}%` },
            }),
          ]),
          h("div.meter__head", {}, [
            h("span.muted", { style: { fontSize: "var(--text-2xs)" } },
              `Additive heat reads ${fmt.pct(portfolio.gross_risk, 2)} — the gap between the two is the risk per-trade sizing cannot see.`),
          ]),
        ]),
        ...portfolio.notes.map((note) => callout(note, "neutral")),
      ],
    }),

    h("div.grid.grid--wide", {}, [
      card({
        title: "Allocation",
        hint: "Dashed outline is what per-trade sizing asked for",
        body: draw(chartHost(), (node) =>
          groupedBars(node, allocationRows(portfolio), { label: "Allocation by position" })
        ),
      }),
      card({
        title: "Correlation",
        hint: portfolio.correlation
          ? `${portfolio.correlation.observations} overlapping bars · ${fmt.pct(portfolio.correlation.coverage, 0)} measured`
          : "unavailable",
        body: [
          draw(chartHost("heatmap"), (node) =>
            heatmap(
              node,
              heatmapLabels(portfolio.correlation),
              portfolio.correlation?.matrix || [],
              { emptyMessage: "At least two positions are needed to correlate anything." }
            )
          ),
          h("div.heatmap__scale", {}, [
            h("span", {}, "hedge −1"),
            h("div.heatmap__gradient"),
            h("span", {}, "+1 same bet"),
          ]),
          portfolio.correlation?.estimated_pairs
            ? callout(
                `${portfolio.correlation.estimated_pairs} of ${portfolio.correlation.total_pairs} pairs had too little overlapping history and were assumed positively correlated rather than independent.`,
                "warn"
              )
            : null,
        ],
      }),
    ]),

    card({
      title: "Positions",
      hint: "Every resize, with the reason it happened",
      flush: true,
      body: h("div.table-wrap", {}, [
        h("table.table", {}, [
          h("thead", {}, [
            h("tr", {}, [
              h("th", {}, "Instrument"),
              h("th", {}, "Dir"),
              h("th", {}, "Strategy"),
              h("th.align-right", {}, "Requested"),
              h("th.align-right", {}, "Allocated"),
              h("th.align-right", {}, "Kept"),
              h("th.align-right", {}, "Of book"),
              h("th", {}, "Cluster"),
              h("th", {}, "Why resized"),
            ]),
          ]),
          h("tbody", {}, portfolio.held.map((position) =>
            h("tr", {}, [
              h("td.table__primary", {}, position.name),
              h("td", {}, [badge(position.direction, directionVariant(position.direction))]),
              h("td.mono", { style: { fontSize: "var(--text-xs)" } }, position.strategy),
              h("td.num.muted", {}, fmt.pct(position.standalone_risk, 2)),
              h("td.num.table__primary", {}, fmt.pct(position.risk_fraction, 2)),
              h("td.num", { class: position.scale < 0.999 ? "warn" : "" },
                `${fmt.num(position.scale, 2)}x`),
              h("td", {}, [
                h("div.bar-cell", {}, [
                  h("div.bar-cell__track", {}, [
                    h("div.bar-cell__fill", {
                      style: {
                        width: `${Math.min(100, (position.risk_contribution || 0) * 100)}%`,
                        background: clusterColor(position.cluster),
                      },
                    }),
                  ]),
                  h("span.num", {}, fmt.pct(position.risk_contribution, 0)),
                ]),
              ]),
              h("td", {}, [
                h("span.chip", { style: { borderLeft: `3px solid ${clusterColor(position.cluster)}` } },
                  `#${position.cluster}`),
              ]),
              h("td.secondary", { style: { fontSize: "var(--text-xs)", maxWidth: "320px" } },
                position.reasons.join("; ") || "full size held"),
            ])
          )),
        ]),
      ]),
    }),

    portfolio.dropped.length
      ? card({
          title: "Dropped by the portfolio layer",
          hint: "Signals that survived the macro chain and still did not earn capital",
          flush: true,
          body: h("div.table-wrap", {}, [
            h("table.table", {}, [
              h("thead", {}, [
                h("tr", {}, [
                  h("th", {}, "Instrument"),
                  h("th", {}, "Dir"),
                  h("th", {}, "Strategy"),
                  h("th.align-right", {}, "Wanted"),
                  h("th.align-right", {}, "E[R]"),
                  h("th", {}, "Reason"),
                ]),
              ]),
              h("tbody", {}, portfolio.dropped.map((position) =>
                h("tr", {}, [
                  h("td.table__primary", {}, position.name),
                  h("td", {}, [badge(position.direction, directionVariant(position.direction))]),
                  h("td.mono", { style: { fontSize: "var(--text-xs)" } }, position.strategy),
                  h("td.num.muted", {}, fmt.pct(position.standalone_risk, 2)),
                  h("td.num", { class: (position.edge?.expected_r ?? 0) > 0 ? "long" : "short" },
                    position.edge ? fmt.signed(position.edge.expected_r, 2) : "—"),
                  h("td.secondary", { style: { fontSize: "var(--text-xs)" } },
                    position.reasons[0] || "no reason recorded"),
                ])
              )),
            ]),
          ]),
        })
      : null,

    clusterBreakdown(portfolio),
  ]);
}

/**
 * Axis labels for the correlation matrix.
 *
 * Two strategies can fire on one instrument, and the allocator deliberately
 * keeps them as separate positions correlated at 1.0. Labelling both rows
 * "AVAXUSDT" makes the matrix look like a bug. Duplicated symbols therefore
 * carry a short strategy suffix; unique ones stay clean.
 */
function heatmapLabels(correlation) {
  if (!correlation?.labels?.length) return [];
  const counts = new Map();
  for (const symbol of correlation.symbols) {
    counts.set(symbol, (counts.get(symbol) || 0) + 1);
  }
  return correlation.labels.map((key, index) => {
    const symbol = correlation.symbols[index];
    if ((counts.get(symbol) || 0) < 2) return symbol;
    const strategy = key.split(":")[1] || "";
    // Initials, not truncation: "TF" reads as trend_following, "trefol" reads
    // as nothing at all.
    const initials = strategy
      .split("_")
      .map((part) => part.charAt(0).toUpperCase())
      .join("");
    return `${symbol} ${initials}`;
  });
}

function allocationRows(portfolio) {
  return portfolio.held.map((position) => ({
    label: position.name,
    sublabel: `${position.direction} · ${position.strategy}`,
    primary: position.risk_fraction,
    secondary: position.standalone_risk,
    color: clusterColor(position.cluster),
    tooltip: `
      <div class="tooltip__row"><span>Requested</span><strong>${fmt.pct(position.standalone_risk, 2)}</strong></div>
      <div class="tooltip__row"><span>Allocated</span><strong>${fmt.pct(position.risk_fraction, 2)}</strong></div>
      <div class="tooltip__row"><span>Share of book risk</span><strong>${fmt.pct(position.risk_contribution, 0)}</strong></div>
      <div class="tooltip__row"><span>Cluster</span><strong>#${position.cluster}</strong></div>
    `,
  }));
}

function clusterBreakdown(portfolio) {
  const groups = new Map();
  for (const position of portfolio.held) {
    if (!groups.has(position.cluster)) groups.set(position.cluster, []);
    groups.get(position.cluster).push(position);
  }
  if (groups.size <= 1 && portfolio.held.length <= 1) return null;

  return card({
    title: "Risk clusters",
    hint: "Positions inside one cluster are, for risk purposes, the same trade",
    body: h("div.clusters", {},
      [...groups.entries()].map(([cluster, members]) => {
        const total = members.reduce((sum, m) => sum + m.risk_fraction, 0);
        return h("div.cluster", { style: { "--cluster-color": clusterColor(cluster) } }, [
          h("div.cluster__head", {}, [
            h("strong", {}, `Cluster #${cluster}`),
            h("span.muted", {}, `${members.length} position${members.length === 1 ? "" : "s"} · ${fmt.pct(total, 2)} of equity at risk`),
          ]),
          h("div.cluster__members", {}, members.map((member) =>
            h("span.chip", {}, [
              member.name,
              h("span.muted", {}, member.direction),
              h("span", {}, fmt.pct(member.risk_fraction, 2)),
            ])
          )),
        ]);
      })
    ),
  });
}

/* ==========================================================================
   Cycle
   ========================================================================== */
export function cycleView(state, { domain, onDomain }) {
  const cycle = state.cycles[domain];

  if (state.cycleLoading && !cycle) {
    return h("div.view", {}, [
      card({ title: "Market Cycle Compass", body: h("div.skeleton.skeleton--chart") }),
    ]);
  }
  if (!cycle) {
    return h("div.view", {}, [
      domainSwitch(domain, onDomain),
      emptyState({
        title: "Cycle unavailable",
        body: "The Compass needs a long daily history for the whole domain. It could not be built from the data currently loaded.",
      }),
    ]);
  }

  const history = cycle.history || [];
  // The engine's history frame names the composite column "cycle"; the other
  // names are accepted so a rename upstream degrades to an empty chart rather
  // than a wrong one.
  const scoreSeries = history.map((p) => p.cycle ?? p.score ?? p.composite ?? null);
  const priceSeries = history.map((p) => p.anchor ?? p.price ?? null);

  return h("div.view", {}, [
    h("div.section__head", {}, [
      h("h2.section__title", {}, "Market Cycle Compass"),
      h("span.section__hint", {}, "The tide, not the trade — a leading read on the next quarter"),
      h("div.section__actions", {}, [domainSwitch(domain, onDomain)]),
    ]),

    /* Narrow column for the dial, wide for the factors: the dial is a single
       reading and the factor chart is eight labelled rows that need the room. */
    h("div.grid.grid--narrow-first", {}, [
      card({
        title: "Phase",
        hint: cycle.data_quality === "insufficient" ? "priors only — not enough history" : cycle.data_quality,
        body: [
          draw(chartHost(), (node) =>
            phaseDial(node, cycle.score, {
              size: 260,
              label: PHASE_LABEL[cycle.phase] || cycle.phase,
              sublabel: `${cycle.bias} · ${cycle.days_in_phase} days in phase`,
            })
          ),
          h("div.dl", { style: { marginTop: "var(--space-4)" } }, [
            dlItem("Momentum", fmt.signed(cycle.momentum, 2)),
            dlItem("Leaving phase", fmt.pct(cycle.transition_probability, 0)),
            dlItem("Confidence", fmt.pct(cycle.confidence, 0)),
            dlItem("Previous", PHASE_LABEL[cycle.previous_phase] || "—"),
          ]),
          cycle.divergence_flag
            ? callout(
                `Divergence ${fmt.num(cycle.divergence, 2)}σ: price is rising while the internals deteriorate. This is the distribution signature, and it hard-blocks longs into late expansion.`,
                "warn"
              )
            : null,
        ],
      }),
      card({
        title: "Factor contributions",
        hint: "Weight × score, positive means risk-on",
        body: draw(chartHost(), (node) =>
          divergingBars(node, cycle.factors.map((factor) => ({
            label: factor.label,
            value: factor.contribution,
            muted: !factor.trusted,
            tooltip: `
              <div class="tooltip__row"><span>Score</span><strong>${fmt.signed(factor.score, 2)}</strong></div>
              <div class="tooltip__row"><span>Weight</span><strong>${fmt.pct(factor.weight, 0)}</strong></div>
              <div class="tooltip__row"><span>Lead</span><strong>${factor.lead_days}d</strong></div>
              <div class="tooltip__row"><span>IC</span><strong>${fmt.num(factor.ic, 3)}</strong></div>
              <div class="tooltip__row"><span>${factor.trusted ? "Measured" : "Prior (not trusted)"}</span><strong></strong></div>
            `,
          })), {
            format: (v) => fmt.signed(v, 3),
            label: "Factor contributions to the composite",
          })
        ),
      }),
    ]),

    history.length > 5
      ? card({
          title: "Composite history",
          hint: "Bull entry +0.25, bear entry −0.25",
          body: draw(chartHost(), (node) =>
            lineChart(node, [{ label: "Cycle score", values: scoreSeries, color: cssVar("--accent") }], {
              height: 240,
              zeroLine: true,
              area: true,
              yFormat: (v) => fmt.num(v, 2),
              xLabelsFor: (i) => fmt.date(history[i]?.ts),
              bands: [
                { from: 0.25, to: 1, color: cssVar("--long") },
                { from: -1, to: -0.25, color: cssVar("--short") },
              ],
              label: "Cycle composite over time",
            })
          ),
        })
      : null,

    priceSeries.some(Number.isFinite)
      ? card({
          title: "Risk proxy",
          hint: "The domain's anchor price the composite is scored against",
          body: draw(chartHost(), (node) =>
            lineChart(node, [{ label: "Anchor", values: priceSeries, color: chartColor(1) }], {
              height: 200,
              yFormat: (v) => fmt.compact(v),
              xLabelsFor: (i) => fmt.date(history[i]?.ts),
              label: "Anchor price",
            })
          ),
        })
      : null,

    card({
      title: "Factors in detail",
      flush: true,
      body: h("div.table-wrap", {}, [
        h("table.table", {}, [
          h("thead", {}, [
            h("tr", {}, [
              h("th", {}, "Factor"),
              h("th.align-right", {}, "Score"),
              h("th.align-right", {}, "Weight"),
              h("th.align-right", {}, "Contribution"),
              h("th.align-right", {}, "Lead"),
              h("th.align-right", {}, "IC"),
              h("th", {}, "Source"),
              h("th", {}, "Reading"),
            ]),
          ]),
          h("tbody", {}, cycle.factors.map((factor) =>
            h("tr", {}, [
              h("td.table__primary", {}, factor.label),
              h("td.num", { class: factor.score > 0 ? "long" : factor.score < 0 ? "short" : "" },
                fmt.signed(factor.score, 2)),
              h("td.num", {}, fmt.pct(factor.weight, 0)),
              h("td.num", {}, fmt.signed(factor.contribution, 3)),
              h("td.num", {}, `${factor.lead_days}d`),
              h("td.num", {}, fmt.num(factor.ic, 3)),
              h("td", {}, [badge(factor.trusted ? "measured" : "prior",
                factor.trusted ? "info" : "neutral")]),
              h("td.secondary", { style: { fontSize: "var(--text-xs)", maxWidth: "340px" } },
                factor.rationale),
            ])
          )),
        ]),
      ]),
    }),

    cycle.narrative.length
      ? card({
          title: "Reading",
          body: h("ul", { style: { display: "flex", flexDirection: "column", gap: "var(--space-2)" } },
            cycle.narrative.map((line) =>
              h("li.secondary", { style: { fontSize: "var(--text-sm)" } }, `— ${line}`)
            )),
        })
      : null,
  ]);
}

function domainSwitch(domain, onDomain) {
  return h("div.segmented", { role: "group", "aria-label": "Cycle domain" }, [
    h("button.segmented__option", {
      type: "button", "aria-pressed": String(domain === "crypto"),
      onclick: () => onDomain("crypto"),
    }, "Crypto"),
    h("button.segmented__option", {
      type: "button", "aria-pressed": String(domain === "fx"),
      onclick: () => onDomain("fx"),
    }, "FX"),
  ]);
}

/* ==========================================================================
   Macro
   ========================================================================== */
export function macroView(state) {
  const analysis = state.analysis;
  if (!analysis) return emptyState({ title: "No scan loaded" });
  const macro = analysis.macro;

  const currencies = macro.rates.map((r) => r.currency);
  const realRates = macro.rates.map((r) => r.real_rate);

  return h("div.view", {}, [
    h("section.section", {}, [
      h("div.grid.grid--metrics", {}, [
        metric({
          label: "Liquidity regime",
          value: (macro.liquidity.regime || "").toUpperCase(),
          tone: macro.liquidity.regime === "contraction" ? "short" : macro.liquidity.regime === "expansion" ? "long" : "",
          meta: [`ΔGLI ${fmt.signedPct(macro.liquidity.gli_delta, 2)}`],
        }),
        metric({ label: "M2 change", value: fmt.signedPct(macro.liquidity.m2_change, 2) }),
        metric({ label: "Stablecoin supply", value: fmt.signedPct(macro.liquidity.stablecoin_change, 2) }),
        metric({
          label: "Macro regime",
          value: (macro.macro_regime || "").toUpperCase(),
          tone: macro.macro_regime === "contraction" ? "short" : "",
        }),
      ]),
    ]),

    card({
      title: "Real rate differentials",
      hint: "Policy rate minus trailing CPI. The engine's carry and RIRD filters read from here.",
      body: draw(chartHost(), (node) =>
        divergingBars(node, macro.rates
          .filter((r) => Number.isFinite(r.real_rate))
          .sort((a, b) => b.real_rate - a.real_rate)
          .map((r) => ({
            label: r.currency,
            value: r.real_rate * 100,
            tooltip: `
              <div class="tooltip__row"><span>Policy</span><strong>${fmt.signedPct(r.policy_rate, 2)}</strong></div>
              <div class="tooltip__row"><span>CPI</span><strong>${fmt.signedPct(r.inflation, 2)}</strong></div>
              <div class="tooltip__row"><span>10Y-2Y</span><strong>${fmt.signedPct(r.curve, 2)}</strong></div>
            `,
          })), {
          unit: "%",
          format: (v) => fmt.signed(v, 2),
          labelWidth: 90,
          label: "Real policy rates by currency",
        })
      ),
    }),

    card({
      title: "Rates and curves",
      flush: true,
      body: h("div.table-wrap", {}, [
        h("table.table", {}, [
          h("thead", {}, [
            h("tr", {}, [
              h("th", {}, "Currency"),
              h("th.align-right", {}, "Policy"),
              h("th.align-right", {}, "Inflation"),
              h("th.align-right", {}, "Real"),
              h("th.align-right", {}, "10Y"),
              h("th.align-right", {}, "2Y"),
              h("th.align-right", {}, "10Y-2Y"),
              h("th.align-right", {}, "ESI"),
            ]),
          ]),
          h("tbody", {}, macro.rates.map((row) =>
            h("tr", {}, [
              h("td.table__primary", {}, row.currency),
              h("td.num", {}, fmt.signedPct(row.policy_rate, 2)),
              h("td.num", {}, fmt.signedPct(row.inflation, 2)),
              h("td.num", { class: (row.real_rate ?? 0) < 0 ? "short" : "long" },
                fmt.signedPct(row.real_rate, 2)),
              h("td.num", {}, fmt.signedPct(row.yield_10y, 2)),
              h("td.num", {}, fmt.signedPct(row.yield_2y, 2)),
              h("td.num", { class: (row.curve ?? 0) < 0 ? "short" : "" },
                fmt.signedPct(row.curve, 2)),
              h("td.num", {}, fmt.signed(row.esi ?? 0, 2)),
            ])
          )),
        ]),
      ]),
    }),

    card({
      title: "Economic calendar",
      hint: "High-impact releases blackout trading ±30 minutes, USD events included for crypto",
      flush: true,
      body: macro.events.length
        ? h("div.table-wrap", {}, [
            h("table.table", {}, [
              h("thead", {}, [
                h("tr", {}, [
                  h("th", {}, "When"),
                  h("th.align-right", {}, "In"),
                  h("th", {}, "Currency"),
                  h("th", {}, "Event"),
                  h("th", {}, "Impact"),
                  h("th.align-right", {}, "Forecast"),
                  h("th.align-right", {}, "Previous"),
                ]),
              ]),
              h("tbody", {}, macro.events.map((event) =>
                h("tr", {}, [
                  h("td.mono", { style: { fontSize: "var(--text-xs)" } }, fmt.time(event.ts, true)),
                  h("td.num", { class: event.hours_away < 1 ? "warn" : "" },
                    `${fmt.num(event.hours_away, 1)}h`),
                  h("td", {}, event.currency),
                  h("td.table__primary", {}, event.name),
                  h("td", {}, [badge(event.impact, event.impact === "high" ? "warn" : "neutral")]),
                  h("td.num", {}, fmt.num(event.forecast, 2)),
                  h("td.num", {}, fmt.num(event.previous, 2)),
                ])
              )),
            ]),
          ])
        : emptyState({ title: "No upcoming events" }),
    }),

    macro.tokenomics.length
      ? card({
          title: "Crypto network economics",
          hint: "MV=PQ velocity and NVT, z-scored against their own history",
          flush: true,
          body: h("div.table-wrap", {}, [
            h("table.table", {}, [
              h("thead", {}, [
                h("tr", {}, [
                  h("th", {}, "Symbol"),
                  h("th.align-right", {}, "Velocity"),
                  h("th.align-right", {}, "Velocity z"),
                  h("th.align-right", {}, "NVT"),
                  h("th.align-right", {}, "NVT z"),
                ]),
              ]),
              h("tbody", {}, macro.tokenomics.map((row) =>
                h("tr", {}, [
                  h("td.table__primary", {}, row.symbol),
                  h("td.num", {}, fmt.num(row.velocity, 3)),
                  h("td.num", { class: (row.velocity_z ?? 0) < -1.5 ? "warn" : "" },
                    fmt.signed(row.velocity_z, 2)),
                  h("td.num", {}, fmt.num(row.nvt, 1)),
                  h("td.num", { class: (row.nvt_z ?? 0) > 2 ? "warn" : "" },
                    fmt.signed(row.nvt_z, 2)),
                ])
              )),
            ]),
          ]),
        })
      : null,
  ]);
}

/* ==========================================================================
   Instrument detail
   ========================================================================== */
export function instrumentView(state, { symbol, data, loading, error, onSelect }) {
  const universe = state.instruments;

  const picker = h("select.select", {
    "aria-label": "Instrument",
    onchange: (event) => onSelect(event.target.value),
  }, [
    universe
      ? frag([
          h("optgroup", { label: "Crypto" }, universe.crypto.map((i) =>
            h("option", { value: i.symbol, selected: i.symbol === symbol }, i.name))),
          h("optgroup", { label: "Forex" }, universe.forex.map((i) =>
            h("option", { value: i.symbol, selected: i.symbol === symbol }, i.name))),
        ])
      : h("option", {}, symbol || "Loading…"),
  ]);

  if (loading) {
    return h("div.view", {}, [
      h("div.section__head", {}, [
        h("h2.section__title", {}, symbol || "Instrument"),
        h("div.section__actions", {}, [picker]),
      ]),
      card({ body: h("div.skeleton.skeleton--chart") }),
    ]);
  }
  if (error) {
    return h("div.view", {}, [
      h("div.section__head", {}, [
        h("h2.section__title", {}, symbol || "Instrument"),
        h("div.section__actions", {}, [picker]),
      ]),
      callout(error.message, "danger"),
    ]);
  }
  if (!data) return emptyState({ title: "Pick an instrument" });

  const signals = data.signals || [];
  const best = signals.find((s) => !s.blocked && s.risk_fraction > 0) || signals[0] || null;
  const levels = best
    ? [
        { value: best.entry, label: "Entry", color: cssVar("--accent") },
        { value: best.stop, label: "Stop", color: cssVar("--short") },
        best.take_profit ? { value: best.take_profit, label: "Target", color: cssVar("--long") } : null,
      ].filter(Boolean)
    : [];

  const rsi = data.panels?.rsi;
  const macd = data.panels?.macd;

  return h("div.view", {}, [
    h("div.section__head", {}, [
      h("h2.section__title", {}, data.name || symbol),
      h("span.section__hint", {}, `${fmt.price(data.last)} · ${fmt.signedPct(data.change, 2)} over the window`),
      h("div.section__actions", {}, [picker]),
    ]),

    data.regime
      ? h("div.grid.grid--metrics", {}, [
          metric({
            label: "Regime",
            value: REGIME_LABEL[data.regime.regime] || data.regime.regime,
            meta: [`favours ${data.regime.prefers}`],
          }),
          metric({ label: "ADX", value: fmt.num(data.regime.adx, 1),
            meta: [data.regime.adx >= 25 ? "trending" : "no trend"] }),
          metric({ label: "Realised vol", value: fmt.pct(data.regime.realized_vol, 1),
            meta: [`${fmt.pct(data.regime.vol_percentile, 0)} percentile`] }),
          metric({ label: "Hurst", value: fmt.num(data.regime.hurst, 2),
            meta: [data.regime.hurst > 0.55 ? "persistent" : data.regime.hurst < 0.45 ? "mean-reverting" : "random walk"] }),
        ])
      : null,

    card({
      title: "Price",
      hint: best ? `${best.strategy} levels overlaid` : "No active setup",
      body: draw(chartHost(), (node) =>
        candlestick(node, data.candles, {
          height: 380,
          overlays: data.overlays,
          levels,
          label: `${data.name} price chart`,
        })
      ),
    }),

    (rsi || macd)
      ? h("div.grid.grid--halves", {}, [
          rsi
            ? card({
                title: "RSI",
                body: draw(chartHost(), (node) =>
                  lineChart(node, [{ label: "RSI", values: rsi, color: chartColor(0) }], {
                    height: 170,
                    yFormat: (v) => fmt.num(v, 0),
                    xLabelsFor: (i) => fmt.date(data.candles[i]?.ts),
                    bands: [
                      { from: 70, to: 100, color: cssVar("--short") },
                      { from: 0, to: 30, color: cssVar("--long") },
                    ],
                    label: "Relative strength index",
                  })
                ),
              })
            : null,
          macd
            ? card({
                title: "MACD",
                body: draw(chartHost(), (node) =>
                  lineChart(node, [
                    { label: "MACD", values: macd, color: chartColor(0) },
                    { label: "Signal", values: data.panels.macd_signal || [], color: chartColor(2) },
                  ], {
                    height: 170,
                    zeroLine: true,
                    yFormat: (v) => fmt.num(v, 2),
                    xLabelsFor: (i) => fmt.date(data.candles[i]?.ts),
                    label: "MACD",
                  })
                ),
              })
            : null,
        ])
      : null,

    h("section.section", {}, [
      h("div.section__head", {}, [
        h("h2.section__title", {}, "Signals on this instrument"),
        h("span.section__hint", {}, `${signals.length} proposed by the strategy library`),
      ]),
      signals.length
        ? frag(signals.map((signal) => signalCard(signal)))
        : emptyState({
            title: "No setup found",
            body: "No strategy in the library proposed a trade here on this timeframe.",
          }),
    ]),
  ]);
}

/* ==========================================================================
   Diagnostics
   ========================================================================== */
export function diagnosticsView(state, { onRefresh }) {
  const status = state.status;
  const analysis = state.analysis;
  const calibration = state.calibration;
  const sources = analysis?.macro?.sources || {};

  const live = Object.entries(sources).filter(([, v]) => v !== "synthetic");
  const synthetic = Object.entries(sources).filter(([, v]) => v === "synthetic");

  return h("div.view", {}, [
    h("section.section", {}, [
      h("div.grid.grid--metrics", {}, [
        metric({
          label: "Live data blocks",
          value: String(live.length),
          tone: live.length ? "long" : "warn",
          meta: [`${synthetic.length} simulated`],
        }),
        metric({
          label: "Providers configured",
          value: String(status?.configured ?? 0),
          meta: [`of ${Object.keys(status?.providers || {}).length}`],
        }),
        metric({
          label: "Offline mode",
          value: status?.offline ? "ON" : "OFF",
          tone: status?.offline ? "warn" : "",
        }),
        metric({
          label: "Cached computations",
          value: String(status?.cache?.entries ?? 0),
          meta: [`${status?.cache?.stale ?? 0} stale`],
        }),
      ]),
    ]),

    live.length === 0
      ? callout(
          "Every data block is simulated. The synthetic provider is a deterministic fixture, not a market simulator — numbers produced here exercise the code and say nothing about edge. Add an API key (FRED is free and the most valuable) to change that.",
          "warn"
        )
      : null,

    card({
      title: "Providers",
      hint: "A missing key degrades one block to synthetic; it never stops the engine",
      actions: [
        h("button.btn", { type: "button", onclick: onRefresh }, [
          icon("refresh", "btn__icon"), "Refresh",
        ]),
      ],
      body: h("div.providers", {},
        Object.entries(status?.providers || {}).map(([name, configured]) =>
          h("div.provider", { dataset: { live: String(configured) }, title: name }, [
            h("span.provider__dot"),
            h("span.provider__name", {}, name),
            // "available" rather than "key set": Binance and alternative.me
            // need no key at all, and labelling them "key set" is simply wrong.
            badge(configured ? "available" : "synthetic", configured ? "info" : "neutral"),
          ])
        )),
    }),

    Object.keys(sources).length
      ? card({
          title: "Data blocks",
          hint: "Which provider actually served each block on the last scan",
          flush: true,
          body: h("div.table-wrap", {}, [
            h("table.table", {}, [
              h("thead", {}, [
                h("tr", {}, [h("th", {}, "Block"), h("th", {}, "Source"), h("th", {}, "Status")]),
              ]),
              h("tbody", {}, Object.entries(sources).sort().map(([block, source]) =>
                h("tr", {}, [
                  h("td.mono", { style: { fontSize: "var(--text-xs)" } }, block),
                  h("td", {}, source),
                  h("td", {}, [badge(source === "synthetic" ? "simulated" : "live",
                    source === "synthetic" ? "neutral" : "info")]),
                ])
              )),
            ]),
          ]),
        })
      : null,

    calibrationCard(calibration),
  ]);
}

function calibrationCard(calibration) {
  if (!calibration) {
    return card({ title: "Hit-rate calibration", body: h("div.skeleton.skeleton--text") });
  }
  if (!calibration.fitted) {
    return card({
      title: "Hit-rate calibration",
      hint: "Is 'confidence' a probability?",
      body: emptyState({
        title: "Running on the prior",
        body: "No realised trades have been recorded, so position sizing uses the 0.35 + 0.25 × confidence prior. Run `mfie calibrate --symbol BTCUSDT --save` or `mfie backtest <symbol> --save` to give it something measured.",
      }),
    });
  }

  const summary = calibration.summary;
  const skill = summary.skill ?? 0;

  return card({
    title: "Hit-rate calibration",
    hint: `${summary.observations} realised trades`,
    flush: true,
    body: [
      h("div.card__body", {}, [
        h("div.grid.grid--metrics", {}, [
          metric({ label: "Realised hit rate", value: fmt.pct(summary.hit_rate, 1) }),
          metric({ label: "Average R", value: fmt.signed(summary.avg_r, 2),
            tone: summary.avg_r > 0 ? "long" : "short" }),
          metric({ label: "Brier score", value: fmt.num(summary.brier, 4),
            meta: [`base rate ${fmt.num(summary.brier_baseline, 4)}`] }),
          metric({ label: "Skill", value: fmt.signed(skill, 3),
            tone: skill > 0.01 ? "long" : "warn",
            meta: [skill > 0.01 ? "carries information" : "not beating its base rate"] }),
        ]),
        skill <= 0.01
          ? callout(
              "The confidence score is not beating its own base rate. Treat it as an ordering of setups, not as a probability — and note that on synthetic data this is the correct answer, since the fixtures are independent by construction.",
              "warn"
            )
          : null,
      ]),
      h("div.table-wrap", {}, [
        h("table.table", {}, [
          h("thead", {}, [
            h("tr", {}, [
              h("th", {}, "Confidence"),
              h("th.align-right", {}, "Forecast"),
              h("th.align-right", {}, "Realised"),
              h("th.align-right", {}, "Trades"),
              h("th.align-right", {}, "Avg R"),
            ]),
          ]),
          h("tbody", {}, calibration.reliability.map((row) =>
            h("tr", {}, [
              h("td.table__primary", {}, row.confidence_range),
              h("td.num.muted", {}, fmt.pct(row.forecast, 0)),
              h("td.num", { class: row.realised >= row.forecast ? "long" : "short" },
                fmt.pct(row.realised, 0)),
              h("td.num", {}, String(row.trades)),
              h("td.num", { class: row.avg_r > 0 ? "long" : "short" }, fmt.signed(row.avg_r, 2)),
            ])
          )),
        ]),
      ]),
      calibration.strategies.length
        ? h("div.table-wrap", { style: { borderTop: "1px solid var(--border-subtle)" } }, [
            h("table.table", {}, [
              h("thead", {}, [
                h("tr", {}, [
                  h("th", {}, "Strategy"),
                  h("th.align-right", {}, "Trades"),
                  h("th.align-right", {}, "Raw"),
                  h("th.align-right", {}, "Posterior"),
                  h("th.align-right", {}, "Avg R"),
                ]),
              ]),
              h("tbody", {}, calibration.strategies.map((row) =>
                h("tr", {}, [
                  h("td.mono", { style: { fontSize: "var(--text-xs)" } }, row.strategy),
                  h("td.num", {}, String(row.trades)),
                  h("td.num.muted", {}, fmt.pct(row.raw, 0)),
                  h("td.num.table__primary", {}, fmt.pct(row.posterior, 0)),
                  h("td.num", { class: row.avg_r > 0 ? "long" : "short" }, fmt.signed(row.avg_r, 2)),
                ])
              )),
            ]),
          ])
        : null,
    ],
  });
}
