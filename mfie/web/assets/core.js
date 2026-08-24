/* ==========================================================================
   Application core: state, routing, data access, and DOM helpers.

   No framework. The whole interface is a handful of views that each render
   from one immutable snapshot of server state, so the machinery a framework
   would provide — reconciliation, change tracking, a component tree — buys
   very little here and costs a build step, which this project has gone out of
   its way not to need anywhere else.

   The pieces:
     store    — a tiny observable holding fetched data and UI preferences
     router   — hash-free history routing with deep links that survive reload
     api      — fetch wrapper with abort, timeout and typed errors
     h/frag   — DOM builders, so views read as structure rather than as strings
   ========================================================================== */

/* ==========================================================================
   DOM builders
   ========================================================================== */

/**
 * Build an element.
 *
 * `h("div.card", { onclick }, [child])`. The tag accepts a CSS-ish shorthand
 * for classes because nearly every node here is a div with classes on it and
 * spelling that out three hundred times obscures the structure.
 */
export function h(tag, attrs = {}, children = []) {
  const [name, ...classes] = String(tag).split(".");
  const node = document.createElement(name || "div");
  if (classes.length) node.className = classes.join(" ");

  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class" || key === "className") {
      node.className = [node.className, value].filter(Boolean).join(" ");
    } else if (key === "style" && typeof value === "object") {
      Object.assign(node.style, value);
    } else if (key === "dataset") {
      Object.assign(node.dataset, value);
    } else if (key.startsWith("on") && typeof value === "function") {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (key === "html") {
      node.innerHTML = value;
    } else if (value === true) {
      node.setAttribute(key, "");
    } else {
      node.setAttribute(key, String(value));
    }
  }

  append(node, children);
  return node;
}

export function append(parent, children) {
  for (const child of [].concat(children)) {
    // `cond && node` yields the falsy operand when cond is false — including
    // the 0 an empty `.length` produces, which would otherwise render as the
    // text "0". None of these are content anyone meant to render. A literal
    // zero that must appear is passed as the string "0".
    if (child === null || child === undefined || child === false ||
        child === "" || child === 0) continue;
    parent.appendChild(
      child instanceof Node ? child : document.createTextNode(String(child))
    );
  }
  return parent;
}

export function frag(children) {
  return append(document.createDocumentFragment(), children);
}

/** Inline SVG icon from the sprite below. Icons are paths, never emoji. */
export function icon(name, className = "nav__icon") {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("class", className);
  svg.setAttribute("aria-hidden", "true");
  svg.innerHTML = ICONS[name] || ICONS.dot;
  return svg;
}

export const ICONS = {
  overview: '<rect x="3" y="3" width="7" height="9" rx="1"/><rect x="14" y="3" width="7" height="5" rx="1"/><rect x="14" y="12" width="7" height="9" rx="1"/><rect x="3" y="16" width="7" height="5" rx="1"/>',
  signals: '<path d="M3 17l5-5 4 3 8-9"/><path d="M16 6h5v5"/>',
  portfolio: '<circle cx="12" cy="12" r="9"/><path d="M12 3v9l6 4"/>',
  cycle: '<path d="M21 12a9 9 0 1 1-3-6.7"/><path d="M21 4v5h-5"/>',
  macro: '<path d="M3 3v18h18"/><path d="M7 15l4-6 3 4 5-8"/>',
  instrument: '<path d="M4 20V9M9 20V4M14 20v-7M19 20V6"/>',
  diagnostics: '<path d="M12 2a7 7 0 0 0-4 12.7V17a2 2 0 0 0 2 2h4a2 2 0 0 0 2-2v-2.3A7 7 0 0 0 12 2z"/><path d="M10 22h4"/>',
  refresh: '<path d="M21 12a9 9 0 1 1-2.6-6.4"/><path d="M21 3v6h-6"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  moon: '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>',
  chevron: '<polyline points="6 9 12 15 18 9"/>',
  close: '<path d="M18 6L6 18M6 6l12 12"/>',
  check: '<polyline points="20 6 9 17 4 12"/>',
  warn: '<path d="M12 9v4M12 17h.01"/><path d="M10.3 3.9L1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 16v-4M12 8h.01"/>',
  block: '<circle cx="12" cy="12" r="9"/><path d="M5.6 5.6l12.8 12.8"/>',
  empty: '<path d="M3 7l9-4 9 4-9 4-9-4z"/><path d="M3 12l9 4 9-4M3 17l9 4 9-4"/>',
  filter: '<path d="M3 4h18l-7 8v6l-4 2v-8z"/>',
  layers: '<path d="M12 2l9 5-9 5-9-5 9-5z"/><path d="M3 12l9 5 9-5M3 17l9 5 9-5"/>',
  download: '<path d="M12 3v12"/><path d="M7 10l5 5 5-5"/><path d="M3 21h18"/>',
  menu: '<path d="M3 6h18M3 12h18M3 18h18"/>',
  dot: '<circle cx="12" cy="12" r="3"/>',
};

/* ==========================================================================
   Store
   ========================================================================== */
class Store {
  constructor(initial) {
    this.state = initial;
    this.listeners = new Set();
  }

  get() {
    return this.state;
  }

  set(patch) {
    const next = typeof patch === "function" ? patch(this.state) : patch;
    this.state = { ...this.state, ...next };
    for (const listener of this.listeners) listener(this.state);
    return this.state;
  }

  subscribe(listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }
}

/* Preferences survive reloads. Wrapped because storage throws outright in
   private windows and when site data is blocked, and a theme preference is
   never worth taking the app down for. */
const PREFS_KEY = "mfie.prefs.v1";

function loadPrefs() {
  try {
    return JSON.parse(localStorage.getItem(PREFS_KEY) || "{}");
  } catch {
    return {};
  }
}

export function savePrefs(prefs) {
  try {
    localStorage.setItem(PREFS_KEY, JSON.stringify(prefs));
  } catch {
    /* Non-fatal: the session simply will not remember. */
  }
}

const prefs = loadPrefs();

export const store = new Store({
  // Server data
  analysis: null,
  meta: null,
  instruments: null,
  params: null,
  status: null,
  cycles: {},
  calibration: null,

  // Request lifecycle
  loading: false,
  error: null,
  lastFetch: null,

  // UI preferences
  theme: prefs.theme || null,
  symbols: prefs.symbols || null,
  timeframe: prefs.timeframe || "1h",
  limit: prefs.limit || 500,
  navCollapsed: prefs.navCollapsed || false,
  showBlocked: prefs.showBlocked ?? true,
  selectedSymbol: prefs.selectedSymbol || null,
});

export function persistPrefs() {
  const s = store.get();
  savePrefs({
    theme: s.theme,
    symbols: s.symbols,
    timeframe: s.timeframe,
    limit: s.limit,
    navCollapsed: s.navCollapsed,
    showBlocked: s.showBlocked,
    selectedSymbol: s.selectedSymbol,
  });
}

/* ==========================================================================
   Theme
   ========================================================================== */
export function applyTheme(theme) {
  if (theme) {
    document.documentElement.setAttribute("data-theme", theme);
  } else {
    // No explicit choice: fall back to the OS via prefers-color-scheme.
    document.documentElement.removeAttribute("data-theme");
  }
}

export function resolvedTheme() {
  const explicit = store.get().theme;
  if (explicit) return explicit;
  return window.matchMedia?.("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

/* ==========================================================================
   API client
   ========================================================================== */
export class ApiError extends Error {
  constructor(message, { status = 0, code = "error", detail = null } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.detail = detail;
  }
}

const inflight = new Map();

/**
 * GET a JSON endpoint.
 *
 * Identical concurrent requests share one promise. Several panels asking for
 * the same scan at mount is the normal case, and without this the browser
 * opens four connections for one answer.
 */
export async function api(path, { timeout = 120000, signal = null, dedupe = true } = {}) {
  if (dedupe && inflight.has(path)) return inflight.get(path);

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  if (signal) signal.addEventListener("abort", () => controller.abort());

  const promise = (async () => {
    try {
      const response = await fetch(path, {
        signal: controller.signal,
        headers: { Accept: "application/json" },
      });

      let payload = null;
      try {
        payload = await response.json();
      } catch {
        throw new ApiError(
          `The server returned a non-JSON response (${response.status}).`,
          { status: response.status, code: "bad_response" }
        );
      }

      if (!response.ok) {
        const detail = payload?.detail ?? payload;
        throw new ApiError(
          detail?.message || payload?.message || `Request failed (${response.status}).`,
          { status: response.status, code: detail?.error || "http_error", detail }
        );
      }
      return payload;
    } catch (error) {
      if (error.name === "AbortError") {
        throw new ApiError(
          "The request timed out. The engine may still be building its first macro snapshot — that can take a minute on a cold start.",
          { code: "timeout" }
        );
      }
      if (error instanceof ApiError) throw error;
      throw new ApiError(
        "Could not reach the server. Check that the MFIE process is still running.",
        { code: "network" }
      );
    } finally {
      clearTimeout(timer);
      inflight.delete(path);
    }
  })();

  if (dedupe) inflight.set(path, promise);
  return promise;
}

export async function apiPost(path) {
  const response = await fetch(path, { method: "POST", headers: { Accept: "application/json" } });
  if (!response.ok) throw new ApiError(`Request failed (${response.status}).`, { status: response.status });
  return response.json();
}

/** Build the query string every data endpoint shares. */
export function queryFor(state, extra = {}) {
  const params = new URLSearchParams();
  if (state.symbols?.length) params.set("symbols", state.symbols.join(","));
  params.set("timeframe", state.timeframe);
  params.set("limit", String(state.limit));
  for (const [key, value] of Object.entries(extra)) {
    if (value !== null && value !== undefined) params.set(key, String(value));
  }
  return params.toString();
}

/* ==========================================================================
   Router
   ========================================================================== */
export class Router {
  constructor(routes, { onNavigate } = {}) {
    this.routes = routes;
    this.onNavigate = onNavigate;
    this.current = null;

    window.addEventListener("popstate", () => this.resolve());
    document.addEventListener("click", (event) => {
      const link = event.target.closest?.("a[data-route]");
      if (!link) return;
      // Let modified clicks open a new tab, as any real link would.
      if (event.metaKey || event.ctrlKey || event.shiftKey || event.button !== 0) return;
      event.preventDefault();
      this.navigate(link.getAttribute("href"));
    });
  }

  navigate(path, { replace = false } = {}) {
    if (path === window.location.pathname + window.location.search) return;
    window.history[replace ? "replaceState" : "pushState"]({}, "", path);
    this.resolve();
  }

  resolve() {
    const path = window.location.pathname.replace(/\/+$/, "") || "/";
    let matched = this.routes.find((route) => route.path === path);
    let params = {};

    if (!matched) {
      for (const route of this.routes) {
        if (!route.pattern) continue;
        const match = route.pattern.exec(path);
        if (match) {
          matched = route;
          params = match.groups || {};
          break;
        }
      }
    }

    matched = matched || this.routes.find((route) => route.path === "/");
    this.current = { route: matched, params, path };
    this.onNavigate?.(this.current);
    return this.current;
  }
}

/* ==========================================================================
   Toasts
   ========================================================================== */
let toastHost = null;

export function toast(message, { tone = "info", timeout = 6000 } = {}) {
  if (!toastHost) {
    toastHost = h("div.toasts", { "aria-live": "polite", "aria-atomic": "false" });
    document.body.appendChild(toastHost);
  }

  const close = h("button.toast__close", {
    type: "button",
    "aria-label": "Dismiss notification",
  }, [icon("close", "btn__icon")]);

  const node = h("div.toast", { dataset: { tone }, role: "status" }, [
    icon(tone === "warn" || tone === "danger" ? "warn" : "info", "callout__icon"),
    h("div", {}, message),
    close,
  ]);

  const dismiss = () => {
    node.style.opacity = "0";
    setTimeout(() => node.remove(), 200);
  };
  close.addEventListener("click", dismiss);
  toastHost.appendChild(node);
  if (timeout) setTimeout(dismiss, timeout);
  return dismiss;
}

/* ==========================================================================
   Shared view fragments
   ========================================================================== */
export function skeleton(kind = "text", count = 1) {
  return frag(
    Array.from({ length: count }, () => h(`div.skeleton.skeleton--${kind}`))
  );
}

export function emptyState({ title, body, iconName = "empty", action = null }) {
  return h("div.empty", {}, [
    icon(iconName, "empty__icon"),
    title && h("p.empty__title", {}, title),
    body && h("p.empty__body", {}, body),
    action,
  ]);
}

export function errorState(error, retry) {
  return h("div.empty", {}, [
    icon("warn", "empty__icon"),
    h("p.empty__title", {}, "Could not load this panel"),
    h("p.empty__body", {}, error?.message || String(error)),
    retry && h("button.btn.btn--primary", { type: "button", onclick: retry }, [
      icon("refresh", "btn__icon"), "Try again",
    ]),
  ]);
}

export function card({ title, hint, actions, body, foot, flush = false, className = "" }) {
  return h(`div.card${className ? `.${className}` : ""}`, {}, [
    (title || actions) &&
      h("div.card__head", {}, [
        title && h("h3.card__title", {}, title),
        hint && h("span.card__hint", {}, hint),
        actions && h("div.card__actions", {}, actions),
      ]),
    h(`div.card__body${flush ? ".card__body--flush" : ""}`, {}, body),
    foot && h("div.card__foot", {}, foot),
  ]);
}

export function metric({ label, value, meta, tone, hint, spark }) {
  // A value with no digits is a word, not a reading, and gets the UI face at a
  // smaller size — see .metric__value--text.
  const isText = typeof value === "string" && !/\d/.test(value);

  // Meta items must be elements for the flex gap to separate them; passing an
  // array of bare strings concatenates them into one run of text.
  const metaItems = []
    .concat(meta || [])
    .filter(Boolean)
    .map((item) => (item instanceof Node ? item : h("span", {}, String(item))));

  return h("div.metric", {}, [
    h("div.metric__label", {}, [
      label,
      hint &&
        h("span", {
          title: hint,
          "aria-label": hint,
          style: { cursor: "help", display: "inline-flex" },
        }, [icon("info", "btn__icon")]),
    ]),
    h("div.metric__value", {
      class: [tone || "", isText ? "metric__value--text" : ""].filter(Boolean).join(" "),
    }, value),
    // Ternary, not `&&`: an empty array short-circuits to the number 0, which
    // append() renders as the text "0" under every metric.
    metaItems.length ? h("div.metric__meta", {}, metaItems) : null,
    spark,
  ]);
}

export function badge(text, variant = "neutral", { large = false } = {}) {
  return h(`span.badge.badge--${variant}${large ? ".badge--lg" : ""}`, {}, text);
}

export function callout(text, tone = "info") {
  return h(`div.callout${tone !== "info" ? `.callout--${tone}` : ""}`, {}, [
    icon(tone === "warn" || tone === "danger" ? "warn" : "info", "callout__icon"),
    h("div", {}, text),
  ]);
}

/* ---- Semantics shared by every view ------------------------------------ */
export function directionVariant(direction) {
  return direction === "long" ? "long" : direction === "short" ? "short" : "neutral";
}

export function verdictVariant(verdict) {
  return {
    STRONG: "accent",
    MODERATE: "info",
    WEAK: "neutral",
    BLOCKED: "danger",
  }[verdict] || "neutral";
}

export function actionVariant(action) {
  return {
    block: "danger",
    penalize: "warn",
    boost: "info",
    pass: "neutral",
  }[action] || "neutral";
}

/** Colour a cluster consistently across every chart and table in a view. */
export function clusterColor(cluster) {
  const styles = getComputedStyle(document.documentElement);
  return styles.getPropertyValue(`--chart-${((cluster - 1 + 8) % 8) + 1}`).trim() || "#888";
}
