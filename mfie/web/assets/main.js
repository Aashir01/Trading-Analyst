/* ==========================================================================
   Entry point: shell, routing, and the data lifecycle.
   ========================================================================== */

import {
  api, apiPost, applyTheme, badge, card, errorState, frag, h, icon,
  persistPrefs, queryFor, resolvedTheme, Router, store, toast,
} from "./core.js";
import { fmt, hideTooltip } from "./charts.js";
import {
  cycleView, diagnosticsView, instrumentView, macroView, overviewView,
  portfolioView, signalsView,
} from "./views.js";

/* ---- Routes ------------------------------------------------------------- */
const ROUTES = [
  { path: "/", id: "overview", title: "Overview", icon: "overview" },
  { path: "/signals", id: "signals", title: "Signals", icon: "signals" },
  { path: "/portfolio", id: "portfolio", title: "Portfolio", icon: "portfolio" },
  { path: "/cycle", id: "cycle", title: "Cycle", icon: "cycle" },
  { path: "/macro", id: "macro", title: "Macro", icon: "macro" },
  {
    path: "/instrument",
    id: "instrument",
    title: "Instruments",
    icon: "instrument",
    pattern: /^\/instrument\/(?<symbol>[A-Za-z0-9]+)$/,
  },
  { path: "/diagnostics", id: "diagnostics", title: "Diagnostics", icon: "diagnostics" },
];

/* Local view state that does not belong in the persisted store. */
const local = {
  cycleDomain: "crypto",
  instrument: { symbol: null, data: null, loading: false, error: null },
};

let router;

/* ==========================================================================
   Shell
   ========================================================================== */
function buildShell() {
  const state = store.get();

  const nav = h("nav.nav", { "aria-label": "Primary" },
    ROUTES.map((route) =>
      h("a.nav__item", {
        href: route.path === "/instrument" ? instrumentHref() : route.path,
        "data-route": "",
        "data-nav": route.id,
      }, [
        icon(route.icon),
        h("span", {}, route.title),
        h("span.nav__badge", { "data-badge": route.id, hidden: true }),
      ])
    )
  );

  const freshness = h("div.data-state", { id: "freshness", dataset: { state: "live" } }, [
    h("span.data-state__dot"),
    h("span.data-state__text", {}, "Loading…"),
  ]);

  const sidebar = h("aside.sidebar", {}, [
    h("div.brand", {}, [
      h("div.brand__mark", {}, "MF"),
      h("div.brand__text", {}, [
        h("span.brand__name", {}, "MFIE"),
        h("span.brand__sub", {}, "Macro-informed intelligence"),
      ]),
    ]),
    nav,
    h("div.sidebar__footer", {}, [freshness]),
  ]);

  const title = h("h1.topbar__title", { id: "view-title" }, "Overview");

  const symbolButton = h("button.btn", {
    type: "button",
    id: "symbol-button",
    "aria-haspopup": "true",
    "aria-expanded": "false",
  }, [icon("filter", "btn__icon"), h("span", { id: "symbol-label" }, "All instruments")]);

  const picker = h("div.picker", {}, [symbolButton]);

  const timeframe = h("select.select", {
    "aria-label": "Timeframe",
    onchange: (event) => {
      store.set({ timeframe: event.target.value });
      persistPrefs();
      loadAnalysis({ force: false });
    },
  }, ["15m", "30m", "1h", "4h", "12h", "1d"].map((value) =>
    h("option", { value, selected: value === state.timeframe }, value)
  ));

  const refreshButton = h("button.btn.btn--icon", {
    type: "button",
    id: "refresh",
    "aria-label": "Refresh data",
    title: "Recompute from source",
    onclick: () => refreshAll(),
  }, [icon("refresh", "btn__icon")]);

  const themeButton = h("button.btn.btn--icon", {
    type: "button",
    "aria-label": "Toggle colour theme",
    title: "Toggle theme",
    onclick: toggleTheme,
  }, [icon(resolvedTheme() === "dark" ? "sun" : "moon", "btn__icon")]);
  themeButton.id = "theme-toggle";

  const topbar = h("header.topbar", {}, [
    title,
    h("div.topbar__spacer"),
    h("div.topbar__controls", {}, [picker, timeframe, refreshButton, themeButton]),
  ]);

  const content = h("main.content", { id: "main", tabindex: "-1" }, [
    h("div.content__inner", { id: "view-root" }),
  ]);

  const app = h("div.app", {
    dataset: { navCollapsed: String(state.navCollapsed) },
  }, [sidebar, h("div.main", {}, [topbar, content])]);

  document.body.appendChild(
    h("a.skip-link", { href: "#main" }, "Skip to content")
  );
  document.body.appendChild(app);

  symbolButton.addEventListener("click", () => toggleSymbolPicker(picker, symbolButton));
  return { app, content };
}

function instrumentHref() {
  const symbol = store.get().selectedSymbol || local.instrument.symbol || "BTCUSDT";
  return `/instrument/${symbol}`;
}

/* ---- Theme -------------------------------------------------------------- */
function toggleTheme() {
  const next = resolvedTheme() === "dark" ? "light" : "dark";
  store.set({ theme: next });
  persistPrefs();
  applyTheme(next);

  const button = document.getElementById("theme-toggle");
  if (button) {
    button.innerHTML = "";
    button.appendChild(icon(next === "dark" ? "sun" : "moon", "btn__icon"));
  }
  // Charts read their colours from CSS variables at draw time, so a theme
  // change means redrawing rather than restyling.
  render();
}

/* ---- Symbol picker ------------------------------------------------------ */
function toggleSymbolPicker(host, button) {
  const existing = host.querySelector(".picker__panel");
  if (existing) {
    existing.remove();
    button.setAttribute("aria-expanded", "false");
    return;
  }

  const state = store.get();
  const universe = state.instruments;
  if (!universe) return;

  const selected = new Set(state.symbols || []);

  const option = (instrument) =>
    h("button.picker__option", {
      type: "button",
      role: "menuitemcheckbox",
      "aria-checked": String(selected.has(instrument.symbol)),
      onclick: (event) => {
        if (selected.has(instrument.symbol)) selected.delete(instrument.symbol);
        else selected.add(instrument.symbol);
        const row = event.currentTarget;
        row.setAttribute("aria-checked", String(selected.has(instrument.symbol)));
      },
    }, [
      h("span.picker__check", {}, [icon("check", "btn__icon")]),
      h("span", {}, instrument.name),
    ]);

  const panel = h("div.picker__panel", { role: "menu" }, [
    h("div.picker__group-label", {}, "Crypto"),
    ...universe.crypto.map(option),
    h("div.picker__group-label", {}, "Forex"),
    ...universe.forex.map(option),
    h("div.picker__foot", {}, [
      h("button.btn", {
        type: "button",
        style: { flex: "1" },
        onclick: () => {
          selected.clear();
          panel.querySelectorAll('[role="menuitemcheckbox"]').forEach((node) =>
            node.setAttribute("aria-checked", "false"));
        },
      }, "Clear"),
      h("button.btn.btn--primary", {
        type: "button",
        style: { flex: "1" },
        onclick: () => {
          store.set({ symbols: selected.size ? [...selected] : null });
          persistPrefs();
          panel.remove();
          button.setAttribute("aria-expanded", "false");
          updateSymbolLabel();
          loadAnalysis({ force: false });
        },
      }, "Apply"),
    ]),
  ]);

  host.appendChild(panel);
  button.setAttribute("aria-expanded", "true");

  const dismiss = (event) => {
    if (!host.contains(event.target)) {
      panel.remove();
      button.setAttribute("aria-expanded", "false");
      document.removeEventListener("click", dismiss);
    }
  };
  setTimeout(() => document.addEventListener("click", dismiss), 0);
}

function updateSymbolLabel() {
  const label = document.getElementById("symbol-label");
  if (!label) return;
  const symbols = store.get().symbols;
  label.textContent = !symbols?.length
    ? "All instruments"
    : symbols.length === 1
      ? symbols[0]
      : `${symbols.length} instruments`;
}

/* ==========================================================================
   Data
   ========================================================================== */
async function loadBootstrap() {
  try {
    const [instruments, params, status] = await Promise.all([
      api("/api/instruments"),
      api("/api/params"),
      api("/api/status"),
    ]);
    store.set({ instruments, params, status });
    updateSymbolLabel();
  } catch (error) {
    // Non-fatal: the app can still render a scan without the picker's contents.
    console.warn("Bootstrap failed", error);
  }
}

async function loadAnalysis({ force = false } = {}) {
  const state = store.get();
  store.set({ loading: true, error: null });
  render();

  try {
    const query = queryFor(state, force ? { force: true } : {});
    const payload = await api(`/api/analysis?${query}`, { dedupe: !force });
    store.set({
      analysis: payload.analysis,
      meta: payload.meta,
      loading: false,
      lastFetch: Date.now(),
    });
    updateFreshness();
    updateBadges();
  } catch (error) {
    store.set({ loading: false, error });
    toast(error.message, { tone: "danger", timeout: 10000 });
  }
  render();
}

async function loadCycle(domain, force = false) {
  const cached = store.get().cycles[domain];
  if (cached && !force) return;

  store.set({ cycleLoading: true });
  render();
  try {
    const payload = await api(`/api/cycle/${domain}${force ? "?force=true" : ""}`);
    store.set((s) => ({ cycles: { ...s.cycles, [domain]: payload.cycle }, cycleLoading: false }));
  } catch (error) {
    store.set({ cycleLoading: false });
    toast(`Cycle (${domain}): ${error.message}`, { tone: "warn" });
  }
  render();
}

async function loadInstrument(symbol, force = false) {
  if (!symbol) return;
  local.instrument = { symbol, data: null, loading: true, error: null };
  render();

  try {
    const state = store.get();
    const params = new URLSearchParams({
      timeframe: state.timeframe,
      limit: String(state.limit),
      points: "300",
    });
    if (force) params.set("force", "true");
    const payload = await api(`/api/instrument/${symbol}?${params}`);
    local.instrument = { symbol, data: payload.instrument, loading: false, error: null };
  } catch (error) {
    local.instrument = { symbol, data: null, loading: false, error };
  }
  render();
}

async function loadCalibration() {
  if (store.get().calibration) return;
  try {
    const payload = await api("/api/calibration");
    store.set({ calibration: payload.calibration });
  } catch (error) {
    console.warn("Calibration unavailable", error);
  }
  render();
}

async function refreshAll() {
  const button = document.getElementById("refresh");
  button?.classList.add("btn--loading");
  try {
    await apiPost("/api/refresh");
    store.set({ cycles: {}, calibration: null });
    local.instrument.data = null;
    await loadAnalysis({ force: true });
    const current = router.current;
    if (current?.route?.id === "cycle") await loadCycle(local.cycleDomain, true);
    if (current?.route?.id === "instrument") await loadInstrument(local.instrument.symbol, true);
    if (current?.route?.id === "diagnostics") {
      store.set({ status: await api("/api/status", { dedupe: false }) });
      await loadCalibration();
    }
    toast("Recomputed from source.", { tone: "info", timeout: 3000 });
  } catch (error) {
    toast(error.message, { tone: "danger" });
  } finally {
    button?.classList.remove("btn--loading");
  }
}

/* ---- Freshness ---------------------------------------------------------- */
function updateFreshness() {
  const node = document.getElementById("freshness");
  if (!node) return;
  const { meta, analysis } = store.get();
  if (!meta) return;

  const liveBlocks = analysis?.macro?.sources_live ?? 0;
  const state = meta.stale ? "stale" : liveBlocks === 0 ? "synthetic" : "live";
  node.dataset.state = state;

  const age = fmt.ago(meta.age_seconds);
  const label = state === "synthetic" ? `Simulated · ${age}` : meta.stale ? `Stale · ${age}` : `Live · ${age}`;
  node.querySelector(".data-state__text").textContent = label;
  node.title = `Computed ${new Date(meta.as_of).toLocaleString()}${
    meta.stale ? " — a refresh is running in the background" : ""
  }`;
}

function updateBadges() {
  const analysis = store.get().analysis;
  if (!analysis) return;

  const counts = {
    signals: analysis.signals.filter((s) => !s.blocked && s.risk_fraction > 0).length,
    portfolio: analysis.portfolio?.held?.length || 0,
  };

  for (const [id, count] of Object.entries(counts)) {
    const badgeNode = document.querySelector(`[data-badge="${id}"]`);
    if (!badgeNode) continue;
    badgeNode.textContent = String(count);
    badgeNode.hidden = count === 0;
  }
}

/* ==========================================================================
   Render
   ========================================================================== */
function render() {
  const root = document.getElementById("view-root");
  if (!root || !router?.current) return;

  const state = store.get();
  const { route, params } = router.current;

  document.getElementById("view-title").textContent = route.title;
  document.title = `${route.title} · MFIE`;

  for (const link of document.querySelectorAll(".nav__item")) {
    const isCurrent = link.dataset.nav === route.id;
    if (isCurrent) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }

  let view;
  if (state.error && !state.analysis) {
    view = errorState(state.error, () => loadAnalysis({ force: true }));
  } else if (state.loading && !state.analysis) {
    view = loadingView();
  } else {
    view = renderRoute(route, params, state);
  }

  root.innerHTML = "";
  root.appendChild(view);
  hideTooltip();
}

function renderRoute(route, params, state) {
  switch (route.id) {
    case "signals":
      return signalsView(state, {
        onToggleBlocked: () => {
          store.set({ showBlocked: !state.showBlocked });
          persistPrefs();
          render();
        },
      });
    case "portfolio":
      return portfolioView(state);
    case "cycle":
      return cycleView(state, {
        domain: local.cycleDomain,
        onDomain: (domain) => {
          local.cycleDomain = domain;
          loadCycle(domain);
          render();
        },
      });
    case "macro":
      return macroView(state);
    case "instrument":
      return instrumentView(state, {
        symbol: params.symbol || local.instrument.symbol,
        data: local.instrument.data,
        loading: local.instrument.loading,
        error: local.instrument.error,
        onSelect: (symbol) => {
          store.set({ selectedSymbol: symbol });
          persistPrefs();
          router.navigate(`/instrument/${symbol}`);
        },
      });
    case "diagnostics":
      return diagnosticsView(state, { onRefresh: refreshAll });
    default:
      return overviewView(state);
  }
}

function loadingView() {
  return h("div.view", {}, [
    h("div.grid.grid--metrics", {},
      Array.from({ length: 6 }, () => h("div.skeleton.skeleton--metric"))),
    card({ body: h("div.skeleton.skeleton--chart") }),
    card({ flush: true, body: frag(Array.from({ length: 6 }, () => h("div.skeleton.skeleton--row"))) }),
  ]);
}

/* ==========================================================================
   Boot
   ========================================================================== */
function boot() {
  applyTheme(store.get().theme);
  buildShell();

  router = new Router(ROUTES, {
    onNavigate: ({ route, params }) => {
      render();

      // Route-specific data is fetched lazily: the Compass reads years of
      // daily history and should not be paid for by someone who only opened
      // the signals list.
      if (route.id === "cycle") loadCycle(local.cycleDomain);
      if (route.id === "diagnostics") loadCalibration();
      if (route.id === "instrument") {
        const symbol = params.symbol || store.get().selectedSymbol || "BTCUSDT";
        if (local.instrument.symbol !== symbol || !local.instrument.data) {
          loadInstrument(symbol);
        }
      }
      document.getElementById("main")?.scrollTo({ top: 0 });
    },
  });

  store.subscribe(() => {
    const app = document.querySelector(".app");
    if (app) app.dataset.navCollapsed = String(store.get().navCollapsed);
  });

  router.resolve();
  loadBootstrap();
  loadAnalysis();

  // Keep the age indicator honest between fetches.
  setInterval(() => {
    const meta = store.get().meta;
    if (!meta) return;
    const elapsed = (Date.now() - store.get().lastFetch) / 1000;
    store.set({ meta: { ...meta, age_seconds: meta.age_seconds + 0 } });
    const node = document.getElementById("freshness");
    if (node) {
      const total = meta.age_seconds + elapsed;
      node.querySelector(".data-state__text").textContent =
        `${node.dataset.state === "synthetic" ? "Simulated" : node.dataset.state === "stale" ? "Stale" : "Live"} · ${fmt.ago(total)}`;
    }
  }, 15000);

  // Keyboard shortcuts, the way a terminal tool is expected to behave.
  document.addEventListener("keydown", (event) => {
    if (event.target.matches("input, select, textarea")) return;
    if (event.metaKey || event.ctrlKey || event.altKey) return;

    const index = ["1", "2", "3", "4", "5", "6", "7"].indexOf(event.key);
    if (index >= 0 && ROUTES[index]) {
      event.preventDefault();
      router.navigate(ROUTES[index].path === "/instrument" ? instrumentHref() : ROUTES[index].path);
      return;
    }
    if (event.key === "r") {
      event.preventDefault();
      refreshAll();
    }
    if (event.key === "t") {
      event.preventDefault();
      toggleTheme();
    }
  });

  window.addEventListener("scroll", hideTooltip, { passive: true, capture: true });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
