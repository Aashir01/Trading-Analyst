/* ==========================================================================
   Charts — hand-rolled SVG, no dependencies.

   Why not a charting library: the payloads here are small and the chart types
   are few and specific (candles with overlays, a diverging impact bar, a
   correlation matrix, a phase dial). A general-purpose library would be
   300-600KB to draw six shapes, would need a CDN or a build step — both of
   which the rest of this project deliberately avoids — and would still need
   overriding to match the design tokens.

   Every chart here:
     * reads its colours from CSS custom properties, so themes just work and
       there is one source of truth for colour;
     * scales to its container via viewBox rather than pixel width, so it is
       responsive without a resize observer;
     * renders numbers into a <title> or the shared tooltip so the data is
       reachable without hovering pixel-perfectly;
     * is given an accessible name and, where it carries information, a table
       equivalent elsewhere in the view. A chart is never the only place a
       number appears.
   ========================================================================== */

const SVG_NS = "http://www.w3.org/2000/svg";

/* ---- Colour access ------------------------------------------------------ */
export function cssVar(name, fallback = "#888") {
  const value = getComputedStyle(document.documentElement)
    .getPropertyValue(name)
    .trim();
  return value || fallback;
}

export function chartColor(index) {
  return cssVar(`--chart-${(index % 8) + 1}`);
}

/* ---- Element helpers ---------------------------------------------------- */
function el(tag, attrs = {}, children = []) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined) continue;
    node.setAttribute(key, String(value));
  }
  for (const child of [].concat(children)) {
    if (child === null || child === undefined) continue;
    node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return node;
}

function svgRoot(width, height, label) {
  const svg = el("svg", {
    viewBox: `0 0 ${width} ${height}`,
    preserveAspectRatio: "xMidYMid meet",
    role: "img",
    "aria-label": label || "chart",
  });
  return svg;
}

/**
 * Pick a viewBox width that matches how wide the chart will actually render.
 *
 * An SVG with `width: 100%` scales its whole coordinate system, font sizes
 * included. A fixed 1000-unit viewBox therefore renders 11px labels at 5.5px
 * inside a 500px card and at 13px inside a 1200px one — the same chart is
 * illegible in one column and oversized in another. Measuring the container
 * makes one unit equal one CSS pixel, so type stays the size it was specified
 * at no matter where the chart is placed.
 */
function measuredWidth(container, { min = 320, max = 1600, fallback = 900 } = {}) {
  const measured = container?.clientWidth || 0;
  if (!measured) return fallback;
  return Math.round(Math.max(min, Math.min(max, measured)));
}

/* ---- Scales ------------------------------------------------------------- */
export function linearScale(domain, range) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0;
  // A flat series has no span; centre it rather than dividing by zero and
  // producing NaN coordinates that silently blank the whole chart.
  if (!span || !Number.isFinite(span)) {
    const mid = (r0 + r1) / 2;
    return () => mid;
  }
  return (value) => r0 + ((value - d0) / span) * (r1 - r0);
}

export function extent(values) {
  let min = Infinity;
  let max = -Infinity;
  for (const value of values) {
    if (value === null || value === undefined || !Number.isFinite(value)) continue;
    if (value < min) min = value;
    if (value > max) max = value;
  }
  if (min === Infinity) return [0, 1];
  if (min === max) return [min - Math.abs(min || 1) * 0.1, max + Math.abs(max || 1) * 0.1];
  return [min, max];
}

/** Round a domain outward to human numbers so axis labels are readable. */
export function niceDomain([min, max], ticks = 5) {
  const span = max - min;
  if (!span || !Number.isFinite(span)) return [min - 1, max + 1];
  const raw = span / ticks;
  const magnitude = Math.pow(10, Math.floor(Math.log10(raw)));
  const normalized = raw / magnitude;
  const step =
    (normalized >= 5 ? 10 : normalized >= 2 ? 5 : normalized >= 1 ? 2 : 1) * magnitude;
  return [Math.floor(min / step) * step, Math.ceil(max / step) * step];
}

export function ticksFor([min, max], count = 5) {
  const step = (max - min) / count;
  if (!Number.isFinite(step) || step === 0) return [min];
  const out = [];
  for (let i = 0; i <= count; i += 1) out.push(min + step * i);
  return out;
}

/* ---- Formatting --------------------------------------------------------- */
export const fmt = {
  num(value, digits = 2) {
    if (value === null || value === undefined || !Number.isFinite(value)) return "—";
    return value.toLocaleString(undefined, {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    });
  },
  /** Price precision follows magnitude: 60000.00 and 1.08432 both read well. */
  price(value) {
    if (value === null || value === undefined || !Number.isFinite(value)) return "—";
    const abs = Math.abs(value);
    const digits = abs >= 1000 ? 2 : abs >= 1 ? 4 : abs >= 0.01 ? 5 : 8;
    return value.toLocaleString(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: digits,
    });
  },
  pct(value, digits = 1) {
    if (value === null || value === undefined || !Number.isFinite(value)) return "—";
    return `${(value * 100).toFixed(digits)}%`;
  },
  signedPct(value, digits = 1) {
    if (value === null || value === undefined || !Number.isFinite(value)) return "—";
    return `${value >= 0 ? "+" : ""}${(value * 100).toFixed(digits)}%`;
  },
  signed(value, digits = 2) {
    if (value === null || value === undefined || !Number.isFinite(value)) return "—";
    return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
  },
  compact(value) {
    if (value === null || value === undefined || !Number.isFinite(value)) return "—";
    const abs = Math.abs(value);
    if (abs >= 1e9) return `${(value / 1e9).toFixed(1)}B`;
    if (abs >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
    if (abs >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
    return value.toFixed(0);
  },
  time(iso, withDate = false) {
    if (!iso) return "—";
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return "—";
    return withDate
      ? date.toLocaleString(undefined, {
          month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
        })
      : date.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
  },
  date(iso) {
    if (!iso) return "—";
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return "—";
    return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  },
  ago(seconds) {
    if (!Number.isFinite(seconds)) return "—";
    if (seconds < 60) return `${Math.round(seconds)}s ago`;
    if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
    return `${Math.round(seconds / 3600)}h ago`;
  },
};

/* ---- Shared tooltip ----------------------------------------------------- */
let tooltipNode = null;

function tooltip() {
  if (!tooltipNode) {
    tooltipNode = document.createElement("div");
    tooltipNode.className = "tooltip";
    tooltipNode.setAttribute("role", "status");
    document.body.appendChild(tooltipNode);
  }
  return tooltipNode;
}

export function showTooltip(event, html) {
  const node = tooltip();
  node.innerHTML = html;
  node.dataset.visible = "true";
  const rect = node.getBoundingClientRect();
  // Flip near the viewport edges so the tooltip never leaves the screen.
  let x = event.clientX + 14;
  let y = event.clientY + 14;
  if (x + rect.width > window.innerWidth - 8) x = event.clientX - rect.width - 14;
  if (y + rect.height > window.innerHeight - 8) y = event.clientY - rect.height - 14;
  node.style.left = `${Math.max(8, x)}px`;
  node.style.top = `${Math.max(8, y)}px`;
}

export function hideTooltip() {
  if (tooltipNode) tooltipNode.dataset.visible = "false";
}

function tooltipRows(rows) {
  return rows
    .filter(Boolean)
    .map(([key, value]) => `<div class="tooltip__row"><span>${key}</span><strong>${value}</strong></div>`)
    .join("");
}

/* ---- Axis rendering ----------------------------------------------------- */
function yAxis(group, scale, domain, width, { ticks = 5, format = (v) => fmt.num(v), pad = 8 } = {}) {
  const values = ticksFor(domain, ticks);
  for (const value of values) {
    const y = scale(value);
    group.appendChild(
      el("line", {
        x1: 0, x2: width, y1: y, y2: y,
        stroke: cssVar("--chart-grid"), "stroke-width": 1,
      })
    );
    group.appendChild(
      el("text", {
        x: -pad, y: y + 3.5, "text-anchor": "end",
        fill: cssVar("--chart-axis"), "font-size": 10,
        "font-family": "var(--font-mono)",
      }, format(value))
    );
  }
}

function xLabels(group, positions, labels, y) {
  positions.forEach((x, i) => {
    group.appendChild(
      el("text", {
        x, y, "text-anchor": "middle",
        fill: cssVar("--chart-axis"), "font-size": 10,
        "font-family": "var(--font-mono)",
      }, labels[i])
    );
  });
}

/* ==========================================================================
   Candlestick chart with overlays
   ========================================================================== */
export function candlestick(container, data, options = {}) {
  const {
    height = 340,
    overlays = {},
    showVolume = true,
    levels = [],
  } = options;

  container.innerHTML = "";
  const candles = (data || []).filter((c) => Number.isFinite(c.close));
  if (!candles.length) {
    container.appendChild(emptyState("No price history available."));
    return;
  }

  const width = measuredWidth(container, { min: 480, fallback: 1000 });
  const margin = { top: 12, right: 64, bottom: 24, left: 8 };
  const volumeHeight = showVolume ? 46 : 0;
  const plotHeight = height - margin.top - margin.bottom - volumeHeight;
  const plotWidth = width - margin.left - margin.right;

  // The price axis must contain the overlays too, or a 200-EMA below the
  // visible range would be clipped and the chart would quietly lie.
  const priceValues = candles.flatMap((c) => [c.high, c.low]);
  for (const series of Object.values(overlays)) {
    if (Array.isArray(series)) priceValues.push(...series.filter(Number.isFinite));
  }
  for (const level of levels) {
    if (Number.isFinite(level.value)) priceValues.push(level.value);
  }

  const priceDomain = niceDomain(extent(priceValues), 5);
  const y = linearScale(priceDomain, [margin.top + plotHeight, margin.top]);
  const step = plotWidth / candles.length;
  const bodyWidth = Math.max(1, Math.min(9, step * 0.62));
  const x = (i) => margin.left + step * (i + 0.5);

  const svg = svgRoot(width, height, options.label || "Price chart");

  const grid = el("g", { transform: `translate(${margin.left},0)` });
  yAxis(grid, y, priceDomain, plotWidth, { format: (v) => fmt.price(v), pad: -6 });
  // Price labels sit on the right, where a trader's eye already is.
  for (const value of ticksFor(priceDomain, 5)) {
    grid.appendChild(
      el("text", {
        x: plotWidth + 8, y: y(value) + 3.5, "text-anchor": "start",
        fill: cssVar("--chart-axis"), "font-size": 10, "font-family": "var(--font-mono)",
      }, fmt.price(value))
    );
  }
  // Remove the left-hand duplicates that yAxis drew.
  grid.querySelectorAll("text[text-anchor='end']").forEach((n) => n.remove());
  svg.appendChild(grid);

  /* Volume behind the candles, scaled to its own band. */
  if (showVolume) {
    const volumes = candles.map((c) => c.volume || 0);
    const volumeScale = linearScale([0, Math.max(...volumes, 1)], [0, volumeHeight - 6]);
    const base = height - margin.bottom;
    const volumeGroup = el("g", { opacity: 0.4 });
    candles.forEach((candle, i) => {
      const h = volumeScale(candle.volume || 0);
      volumeGroup.appendChild(
        el("rect", {
          x: x(i) - bodyWidth / 2, y: base - h, width: bodyWidth, height: Math.max(h, 0.5),
          fill: candle.close >= candle.open ? cssVar("--long") : cssVar("--short"),
        })
      );
    });
    svg.appendChild(volumeGroup);
  }

  /* Overlays under the candles so price stays legible on top. */
  const overlayMeta = {
    ema_fast: { color: chartColor(0), label: "EMA fast" },
    ema_slow: { color: chartColor(1), label: "EMA slow" },
    ema_trend: { color: chartColor(2), label: "EMA 200", dash: "4 3" },
    bb_upper: { color: cssVar("--border-strong"), label: "Bollinger", dash: "2 3" },
    bb_lower: { color: cssVar("--border-strong"), label: null, dash: "2 3" },
    supertrend: { color: chartColor(7), label: "Supertrend", dash: "3 2" },
    vwap: { color: chartColor(4), label: "VWAP", dash: "1 3" },
  };

  const legend = [];
  for (const [key, series] of Object.entries(overlays)) {
    const meta = overlayMeta[key];
    if (!meta || !Array.isArray(series)) continue;
    const path = linePath(series, x, y);
    if (!path) continue;
    svg.appendChild(
      el("path", {
        d: path, fill: "none", stroke: meta.color,
        "stroke-width": 1.4, "stroke-dasharray": meta.dash || null,
        "stroke-linejoin": "round", opacity: 0.9,
      })
    );
    if (meta.label) legend.push({ label: meta.label, color: meta.color });
  }

  /* Candles. */
  const candleGroup = el("g");
  candles.forEach((candle, i) => {
    const up = candle.close >= candle.open;
    const color = up ? cssVar("--long") : cssVar("--short");
    const cx = x(i);
    candleGroup.appendChild(
      el("line", {
        x1: cx, x2: cx, y1: y(candle.high), y2: y(candle.low),
        stroke: color, "stroke-width": 1,
      })
    );
    const openY = y(candle.open);
    const closeY = y(candle.close);
    candleGroup.appendChild(
      el("rect", {
        x: cx - bodyWidth / 2,
        y: Math.min(openY, closeY),
        width: bodyWidth,
        height: Math.max(Math.abs(closeY - openY), 1),
        fill: color,
      })
    );
  });
  svg.appendChild(candleGroup);

  /* Trade levels: entry, stop, target. */
  for (const level of levels) {
    if (!Number.isFinite(level.value)) continue;
    const ly = y(level.value);
    svg.appendChild(
      el("line", {
        x1: margin.left, x2: margin.left + plotWidth, y1: ly, y2: ly,
        stroke: level.color || cssVar("--accent"),
        "stroke-width": 1.2, "stroke-dasharray": "5 4", opacity: 0.9,
      })
    );
    svg.appendChild(
      el("text", {
        x: margin.left + 6, y: ly - 5,
        fill: level.color || cssVar("--accent"),
        "font-size": 10, "font-weight": 600, "font-family": "var(--font-mono)",
      }, `${level.label} ${fmt.price(level.value)}`)
    );
  }

  /* Time axis. */
  const tickCount = Math.min(6, candles.length);
  const positions = [];
  const labels = [];
  for (let i = 0; i < tickCount; i += 1) {
    const index = Math.floor((candles.length - 1) * (i / Math.max(tickCount - 1, 1)));
    positions.push(x(index));
    labels.push(fmt.date(candles[index].ts));
  }
  xLabels(svg, positions, labels, height - 6);

  /* Crosshair. One transparent overlay rect, not one listener per candle. */
  const crosshair = el("g", { opacity: 0, "pointer-events": "none" });
  const vline = el("line", {
    y1: margin.top, y2: margin.top + plotHeight,
    stroke: cssVar("--chart-crosshair"), "stroke-width": 1, "stroke-dasharray": "3 3",
  });
  crosshair.appendChild(vline);
  svg.appendChild(crosshair);

  const hit = el("rect", {
    x: margin.left, y: margin.top, width: plotWidth, height: plotHeight + volumeHeight,
    fill: "transparent", style: "cursor:crosshair",
  });
  hit.addEventListener("mousemove", (event) => {
    const box = svg.getBoundingClientRect();
    const localX = ((event.clientX - box.left) / box.width) * width;
    const index = Math.max(0, Math.min(candles.length - 1, Math.round((localX - margin.left) / step - 0.5)));
    const candle = candles[index];
    if (!candle) return;
    crosshair.setAttribute("opacity", "1");
    vline.setAttribute("x1", x(index));
    vline.setAttribute("x2", x(index));
    const change = (candle.close - candle.open) / (candle.open || 1);
    showTooltip(event, `
      <div class="tooltip__title">${fmt.time(candle.ts, true)}</div>
      ${tooltipRows([
        ["Open", fmt.price(candle.open)],
        ["High", fmt.price(candle.high)],
        ["Low", fmt.price(candle.low)],
        ["Close", fmt.price(candle.close)],
        ["Change", fmt.signedPct(change)],
        candle.volume ? ["Volume", fmt.compact(candle.volume)] : null,
      ])}
    `);
  });
  hit.addEventListener("mouseleave", () => {
    crosshair.setAttribute("opacity", "0");
    hideTooltip();
  });
  svg.appendChild(hit);

  container.appendChild(svg);
  if (legend.length) container.appendChild(legendNode(legend));
}

function linePath(series, x, y) {
  let path = "";
  let started = false;
  series.forEach((value, i) => {
    if (value === null || value === undefined || !Number.isFinite(value)) {
      started = false;   // break the line across gaps rather than bridging them
      return;
    }
    path += `${started ? "L" : "M"}${x(i).toFixed(2)},${y(value).toFixed(2)}`;
    started = true;
  });
  return path || null;
}

function legendNode(items) {
  const wrap = document.createElement("div");
  wrap.className = "chart__legend";
  wrap.innerHTML = items
    .map(
      (item) =>
        `<span class="chart__legend-item"><span class="chart__swatch${
          item.box ? " chart__swatch--box" : ""
        }" style="background:${item.color}"></span>${item.label}</span>`
    )
    .join("");
  return wrap;
}

function emptyState(message) {
  const node = document.createElement("div");
  node.className = "empty";
  node.innerHTML = `<p class="empty__body">${message}</p>`;
  return node;
}

/* ==========================================================================
   Line / area chart
   ========================================================================== */
export function lineChart(container, series, options = {}) {
  const {
    height = 240,
    area = false,
    zeroLine = false,
    yFormat = (v) => fmt.num(v),
    xLabelsFor = null,
    bands = [],
  } = options;

  container.innerHTML = "";
  const active = series.filter((s) => s.values && s.values.some(Number.isFinite));
  if (!active.length) {
    container.appendChild(emptyState(options.emptyMessage || "No data for this window."));
    return;
  }

  const width = measuredWidth(container, { min: 360, fallback: 900 });
  const margin = { top: 12, right: 12, bottom: 24, left: 52 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;

  const allValues = active.flatMap((s) => s.values);
  let domain = niceDomain(extent(allValues), 4);
  if (zeroLine) domain = [Math.min(domain[0], 0), Math.max(domain[1], 0)];

  const count = Math.max(...active.map((s) => s.values.length));
  const y = linearScale(domain, [margin.top + plotHeight, margin.top]);
  const x = (i) => margin.left + (count > 1 ? (i / (count - 1)) * plotWidth : plotWidth / 2);

  const svg = svgRoot(width, height, options.label || "Line chart");

  /* Horizontal bands (e.g. bull / bear thresholds) sit behind everything.
     Band edges are clamped to the visible domain: a band declared from 0.25 to
     1.0 on a series that never exceeds 0.4 would otherwise be drawn far above
     the plot and, with overflow visible, paint over the card's own header. */
  for (const band of bands) {
    const from = Math.max(Math.min(band.from, domain[1]), domain[0]);
    const to = Math.max(Math.min(band.to, domain[1]), domain[0]);
    if (from === to) continue;
    const top = y(to);
    const bottom = y(from);
    svg.appendChild(
      el("rect", {
        x: margin.left, y: Math.min(top, bottom),
        width: plotWidth, height: Math.abs(bottom - top),
        fill: band.color, opacity: band.opacity ?? 0.08,
      })
    );
  }

  const grid = el("g", { transform: `translate(${margin.left},0)` });
  yAxis(grid, y, domain, plotWidth, { ticks: 4, format: yFormat });
  svg.appendChild(grid);

  if (zeroLine) {
    svg.appendChild(
      el("line", {
        x1: margin.left, x2: margin.left + plotWidth, y1: y(0), y2: y(0),
        stroke: cssVar("--chart-axis"), "stroke-width": 1, "stroke-dasharray": "4 3",
      })
    );
  }

  active.forEach((entry, index) => {
    const color = entry.color || chartColor(index);
    if (area) {
      const finite = entry.values
        .map((v, i) => ({ v, i }))
        .filter((p) => Number.isFinite(p.v));
      if (finite.length > 1) {
        const baseline = y(Math.max(domain[0], 0));
        let d = `M${x(finite[0].i)},${baseline}`;
        for (const point of finite) d += `L${x(point.i)},${y(point.v)}`;
        d += `L${x(finite[finite.length - 1].i)},${baseline}Z`;
        svg.appendChild(el("path", { d, fill: color, opacity: 0.12 }));
      }
    }
    const path = linePath(entry.values, x, y);
    if (path) {
      svg.appendChild(
        el("path", {
          d: path, fill: "none", stroke: color,
          "stroke-width": entry.width || 1.8,
          "stroke-dasharray": entry.dash || null,
          "stroke-linejoin": "round", "stroke-linecap": "round",
        })
      );
    }
  });

  if (xLabelsFor && count > 1) {
    const tickCount = Math.min(6, count);
    const positions = [];
    const labels = [];
    for (let i = 0; i < tickCount; i += 1) {
      const index = Math.floor((count - 1) * (i / Math.max(tickCount - 1, 1)));
      positions.push(x(index));
      labels.push(xLabelsFor(index));
    }
    xLabels(svg, positions, labels, height - 6);
  }

  /* Hover reads every series at the same index — comparing series at one
     moment is the whole reason they share an axis. */
  const marker = el("line", {
    y1: margin.top, y2: margin.top + plotHeight,
    stroke: cssVar("--chart-crosshair"), "stroke-width": 1,
    "stroke-dasharray": "3 3", opacity: 0,
  });
  svg.appendChild(marker);

  const hit = el("rect", {
    x: margin.left, y: margin.top, width: plotWidth, height: plotHeight,
    fill: "transparent", style: "cursor:crosshair",
  });
  hit.addEventListener("mousemove", (event) => {
    const box = svg.getBoundingClientRect();
    const localX = ((event.clientX - box.left) / box.width) * width;
    const index = Math.max(0, Math.min(count - 1, Math.round(((localX - margin.left) / plotWidth) * (count - 1))));
    marker.setAttribute("opacity", "1");
    marker.setAttribute("x1", x(index));
    marker.setAttribute("x2", x(index));
    showTooltip(event, `
      <div class="tooltip__title">${xLabelsFor ? xLabelsFor(index) : `#${index}`}</div>
      ${tooltipRows(active.map((s, i) => [
        `<span style="color:${s.color || chartColor(i)}">${s.label}</span>`,
        yFormat(s.values[index]),
      ]))}
    `);
  });
  hit.addEventListener("mouseleave", () => {
    marker.setAttribute("opacity", "0");
    hideTooltip();
  });
  svg.appendChild(hit);

  container.appendChild(svg);
  if (active.length > 1 || options.forceLegend) {
    container.appendChild(
      legendNode(active.map((s, i) => ({ label: s.label, color: s.color || chartColor(i) })))
    );
  }
}

/* ==========================================================================
   Diverging horizontal bars — filter impact, factor contribution
   ========================================================================== */
export function divergingBars(container, rows, options = {}) {
  const { rowHeight = 30, format = (v) => fmt.signed(v, 1), unit = "" } = options;

  container.innerHTML = "";
  if (!rows.length) {
    container.appendChild(emptyState(options.emptyMessage || "Nothing to show."));
    return;
  }

  const width = measuredWidth(container, { min: 340, fallback: 760 });
  const labelWidth = options.labelWidth ?? Math.min(210, Math.max(90, width * 0.26));
  const margin = { top: 8, right: 60, bottom: 8, left: labelWidth };
  const plotWidth = width - margin.left - margin.right;
  const height = margin.top + margin.bottom + rows.length * rowHeight;

  const magnitude = Math.max(...rows.map((r) => Math.abs(r.value)), 1e-9);
  const domain = [-magnitude, magnitude];
  const x = linearScale(domain, [margin.left, margin.left + plotWidth]);
  const zero = x(0);

  const svg = svgRoot(width, height, options.label || "Contribution chart");

  svg.appendChild(
    el("line", {
      x1: zero, x2: zero, y1: margin.top, y2: height - margin.bottom,
      stroke: cssVar("--chart-axis"), "stroke-width": 1,
    })
  );

  rows.forEach((row, index) => {
    const cy = margin.top + index * rowHeight + rowHeight / 2;
    const positive = row.value >= 0;
    const color = row.color || (positive ? cssVar("--diverge-high") : cssVar("--diverge-low"));
    const barX = positive ? zero : x(row.value);
    const barWidth = Math.max(Math.abs(x(row.value) - zero), 1);

    svg.appendChild(
      el("text", {
        x: margin.left - 12, y: cy + 4, "text-anchor": "end",
        fill: cssVar("--text-secondary"), "font-size": 11,
      }, row.label)
    );

    const bar = el("rect", {
      x: barX, y: cy - rowHeight * 0.3, width: barWidth, height: rowHeight * 0.6,
      fill: color, rx: 2, opacity: row.muted ? 0.4 : 0.85,
      style: "cursor:pointer",
    });
    if (row.tooltip) {
      bar.addEventListener("mousemove", (event) =>
        showTooltip(event, `<div class="tooltip__title">${row.label}</div>${row.tooltip}`)
      );
      bar.addEventListener("mouseleave", hideTooltip);
    }
    bar.appendChild(el("title", {}, `${row.label}: ${format(row.value)}${unit}`));
    svg.appendChild(bar);

    /* The value sits outside the bar by default, but a bar long enough to
       reach the label gutter would collide with the row label. Past that
       point the value moves inside the bar, where there is room and contrast. */
    const outsideX = positive ? barX + barWidth + 8 : barX - 8;
    const wouldCollide = !positive && outsideX < margin.left + 4;
    const inside = wouldCollide && barWidth > 46;

    svg.appendChild(
      el("text", {
        x: inside ? barX + 8 : outsideX,
        y: cy + 4,
        "text-anchor": inside ? "start" : positive ? "start" : "end",
        fill: inside ? "#fff" : cssVar("--text-muted"),
        "font-size": 10,
        "font-weight": inside ? 600 : 400,
        "font-family": "var(--font-mono)",
      }, `${format(row.value)}${unit}`)
    );
  });

  container.appendChild(svg);
}

/* ==========================================================================
   Grouped horizontal bars — standalone vs allocated size
   ========================================================================== */
export function groupedBars(container, rows, options = {}) {
  const { rowHeight = 38, format = (v) => fmt.pct(v, 2), labelWidth = 210 } = options;

  container.innerHTML = "";
  if (!rows.length) {
    container.appendChild(emptyState(options.emptyMessage || "No positions to show."));
    return;
  }

  const width = measuredWidth(container, { min: 360, fallback: 820 });
  const margin = { top: 8, right: 88, bottom: 8, left: Math.min(labelWidth, width * 0.32) };
  const plotWidth = width - margin.left - margin.right;
  const height = margin.top + margin.bottom + rows.length * rowHeight;

  const max = Math.max(...rows.flatMap((r) => [r.primary, r.secondary]), 1e-9);
  const x = linearScale([0, max], [margin.left, margin.left + plotWidth]);

  const svg = svgRoot(width, height, options.label || "Allocation chart");

  rows.forEach((row, index) => {
    const top = margin.top + index * rowHeight;
    const cy = top + rowHeight / 2;

    svg.appendChild(
      el("text", {
        x: margin.left - 12, y: cy - 2, "text-anchor": "end",
        fill: cssVar("--text-primary"), "font-size": 11, "font-weight": 500,
      }, row.label)
    );
    if (row.sublabel) {
      svg.appendChild(
        el("text", {
          x: margin.left - 12, y: cy + 10, "text-anchor": "end",
          fill: cssVar("--text-muted"), "font-size": 9.5,
        }, row.sublabel)
      );
    }

    /* The requested size is drawn as a hollow track; the allocated size fills
       it. The empty part of the track is the risk the book refused — which is
       the single thing this chart exists to show. */
    svg.appendChild(
      el("rect", {
        x: margin.left, y: cy - 9,
        width: Math.max(x(row.secondary) - margin.left, 1), height: 18,
        fill: "none", stroke: cssVar("--border-strong"),
        "stroke-width": 1, "stroke-dasharray": "3 2", rx: 3,
      })
    );

    const fill = el("rect", {
      x: margin.left, y: cy - 9,
      width: Math.max(x(row.primary) - margin.left, 1), height: 18,
      fill: row.color || cssVar("--accent"), rx: 3, opacity: 0.9,
      style: "cursor:pointer",
    });
    if (row.tooltip) {
      fill.addEventListener("mousemove", (event) =>
        showTooltip(event, `<div class="tooltip__title">${row.label}</div>${row.tooltip}`)
      );
      fill.addEventListener("mouseleave", hideTooltip);
    }
    fill.appendChild(
      el("title", {}, `${row.label}: allocated ${format(row.primary)} of ${format(row.secondary)} requested`)
    );
    svg.appendChild(fill);

    svg.appendChild(
      el("text", {
        x: margin.left + plotWidth + 12, y: cy + 4,
        fill: cssVar("--text-secondary"), "font-size": 11,
        "font-family": "var(--font-mono)",
      }, format(row.primary))
    );
  });

  container.appendChild(svg);
  container.appendChild(
    legendNode([
      { label: "Allocated", color: cssVar("--accent"), box: true },
      { label: "Requested by per-trade sizing", color: cssVar("--border-strong"), box: true },
    ])
  );
}

/* ==========================================================================
   Correlation heatmap
   ========================================================================== */
export function heatmap(container, labels, matrix, options = {}) {
  container.innerHTML = "";
  if (!labels?.length || !matrix?.length) {
    container.appendChild(emptyState(options.emptyMessage || "Not enough positions to correlate."));
    return;
  }

  const n = labels.length;
  const cell = Math.max(26, Math.min(56, 420 / n));

  // Reserve room for the longest label rather than a fixed guess. Row labels
  // are right-aligned into this gutter and column labels are rotated through
  // it, so a label longer than the reserve is clipped at the chart edge.
  const longest = labels.reduce((max, label) => Math.max(max, label.length), 0);
  const labelSpace = Math.min(200, Math.max(80, longest * 6.4 + 14));
  // The rotated column labels also lean past the right-hand column.
  const rightPad = Math.min(140, longest * 4.2 + 10);
  const width = labelSpace + n * cell + rightPad;
  const height = labelSpace + n * cell + 8;

  const svg = svgRoot(width, height, options.label || "Correlation matrix");

  const low = cssVar("--diverge-low");
  const mid = cssVar("--diverge-mid");
  const high = cssVar("--diverge-high");

  for (let row = 0; row < n; row += 1) {
    svg.appendChild(
      el("text", {
        x: labelSpace - 8, y: labelSpace + row * cell + cell / 2 + 4,
        "text-anchor": "end", fill: cssVar("--text-secondary"),
        "font-size": 10, "font-family": "var(--font-mono)",
      }, labels[row])
    );
    /* Column labels are rotated rather than truncated: a symbol is
       unrecognisable at four characters, and these are the primary key. */
    svg.appendChild(
      el("text", {
        x: labelSpace + row * cell + cell / 2, y: labelSpace - 8,
        "text-anchor": "start", fill: cssVar("--text-secondary"),
        "font-size": 10, "font-family": "var(--font-mono)",
        transform: `rotate(-55 ${labelSpace + row * cell + cell / 2} ${labelSpace - 8})`,
      }, labels[row])
    );
  }

  for (let row = 0; row < n; row += 1) {
    for (let col = 0; col < n; col += 1) {
      const value = matrix[row]?.[col];
      const shown = Number.isFinite(value) ? value : 0;
      const rect = el("rect", {
        x: labelSpace + col * cell, y: labelSpace + row * cell,
        width: cell - 1.5, height: cell - 1.5, rx: 2,
        fill: divergeColor(shown, low, mid, high),
        style: "cursor:pointer",
      });
      rect.appendChild(el("title", {}, `${labels[row]} / ${labels[col]}: ${shown.toFixed(2)}`));
      rect.addEventListener("mousemove", (event) =>
        showTooltip(event, `
          <div class="tooltip__title">${labels[row]} vs ${labels[col]}</div>
          ${tooltipRows([
            ["Signed correlation", shown.toFixed(3)],
            ["Reading", shown > 0.55 ? "same bet" : shown < -0.3 ? "hedge" : "independent"],
          ])}
        `)
      );
      rect.addEventListener("mouseleave", hideTooltip);
      svg.appendChild(rect);

      if (cell >= 34) {
        svg.appendChild(
          el("text", {
            x: labelSpace + col * cell + (cell - 1.5) / 2,
            y: labelSpace + row * cell + (cell - 1.5) / 2 + 3.5,
            "text-anchor": "middle",
            fill: Math.abs(shown) > 0.6 ? "#fff" : cssVar("--text-muted"),
            "font-size": 9, "font-family": "var(--font-mono)",
          }, shown.toFixed(2))
        );
      }
    }
  }

  container.appendChild(svg);
}

function divergeColor(value, low, mid, high) {
  const t = Math.max(-1, Math.min(1, value));
  return t >= 0 ? mixColor(mid, high, t) : mixColor(mid, low, -t);
}

function mixColor(from, to, t) {
  const a = parseHex(from);
  const b = parseHex(to);
  if (!a || !b) return from;
  const channel = (i) => Math.round(a[i] + (b[i] - a[i]) * t);
  return `rgb(${channel(0)},${channel(1)},${channel(2)})`;
}

function parseHex(color) {
  const match = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(color.trim());
  if (!match) return null;
  return [parseInt(match[1], 16), parseInt(match[2], 16), parseInt(match[3], 16)];
}

/* ==========================================================================
   Phase dial — the cycle score as an arc
   ========================================================================== */
export function phaseDial(container, score, options = {}) {
  const { size = 200, label = "", sublabel = "" } = options;
  container.innerHTML = "";

  const svg = svgRoot(size, size * 0.62, options.ariaLabel || `Cycle score ${score.toFixed(2)}`);
  const cx = size / 2;
  const cy = size * 0.52;
  const radius = size * 0.38;
  const thickness = size * 0.075;

  /* A 180-degree sweep: -1 at the left, +1 at the right, zero at the top. */
  const angleFor = (value) => Math.PI * (1 - (Math.max(-1, Math.min(1, value)) + 1) / 2);
  const point = (value, r) => {
    const angle = angleFor(value);
    return [cx + Math.cos(angle) * r, cy - Math.sin(angle) * r];
  };

  const arc = (from, to, r, color, width, opacity = 1) => {
    const [x1, y1] = point(from, r);
    const [x2, y2] = point(to, r);
    const large = Math.abs(to - from) > 2 ? 1 : 0;
    return el("path", {
      d: `M${x1},${y1}A${r},${r} 0 ${large} 1 ${x2},${y2}`,
      fill: "none", stroke: color, "stroke-width": width,
      "stroke-linecap": "round", opacity,
    });
  };

  /* Phase bands, coloured by what they mean for risk. */
  svg.appendChild(arc(-1, -0.25, radius, cssVar("--short"), thickness, 0.28));
  svg.appendChild(arc(-0.25, 0.25, radius, cssVar("--status-neutral"), thickness, 0.28));
  svg.appendChild(arc(0.25, 1, radius, cssVar("--long"), thickness, 0.28));

  const tone = score >= 0.25 ? cssVar("--long") : score <= -0.25 ? cssVar("--short") : cssVar("--status-neutral");
  if (Math.abs(score) > 0.01) svg.appendChild(arc(0, score, radius, tone, thickness));

  /* Needle. */
  const [nx, ny] = point(score, radius + thickness * 0.75);
  const [bx, by] = point(score, radius - thickness * 0.9);
  svg.appendChild(
    el("line", {
      x1: bx, y1: by, x2: nx, y2: ny,
      stroke: cssVar("--text-primary"), "stroke-width": 2.5, "stroke-linecap": "round",
    })
  );
  svg.appendChild(el("circle", { cx, cy, r: 4, fill: cssVar("--text-primary") }));

  svg.appendChild(
    el("text", {
      x: cx, y: cy - radius * 0.28, "text-anchor": "middle",
      fill: tone, "font-size": size * 0.15, "font-weight": 700,
      "font-family": "var(--font-mono)",
    }, fmt.signed(score, 2))
  );

  /* Scale ends, so the dial is readable without a caption. */
  svg.appendChild(
    el("text", { x: cx - radius - 2, y: cy + 16, "text-anchor": "middle",
      fill: cssVar("--text-muted"), "font-size": 9 }, "-1")
  );
  svg.appendChild(
    el("text", { x: cx + radius + 2, y: cy + 16, "text-anchor": "middle",
      fill: cssVar("--text-muted"), "font-size": 9 }, "+1")
  );

  const wrap = document.createElement("div");
  wrap.className = "dial";
  // The dial is a fixed-proportion instrument, not a plot: letting it stretch
  // to the card width turns a 200px gauge into a 900px one and swamps the
  // panel it is meant to summarise.
  svg.style.maxWidth = `${size}px`;
  svg.style.width = "100%";
  wrap.appendChild(svg);
  if (label) {
    const title = document.createElement("div");
    title.className = "dial__phase";
    title.style.color = tone;
    title.textContent = label;
    wrap.appendChild(title);
  }
  if (sublabel) {
    const sub = document.createElement("div");
    sub.className = "dial__meta";
    sub.textContent = sublabel;
    wrap.appendChild(sub);
  }
  container.appendChild(wrap);
}

/* ==========================================================================
   Sparkline
   ========================================================================== */
export function sparkline(container, values, options = {}) {
  const { height = 32, color = null, area = true } = options;
  container.innerHTML = "";

  const finite = values.filter(Number.isFinite);
  if (finite.length < 2) return;

  const width = 160;
  const domain = extent(finite);
  const y = linearScale(domain, [height - 2, 2]);
  const x = (i) => (i / (values.length - 1)) * width;

  const rising = finite[finite.length - 1] >= finite[0];
  const stroke = color || (rising ? cssVar("--long") : cssVar("--short"));

  const svg = svgRoot(width, height, options.label || "Trend");
  svg.setAttribute("preserveAspectRatio", "none");

  const path = linePath(values, x, y);
  if (!path) return;

  if (area) {
    svg.appendChild(
      el("path", { d: `${path}L${width},${height}L0,${height}Z`, fill: stroke, opacity: 0.12 })
    );
  }
  svg.appendChild(
    el("path", {
      d: path, fill: "none", stroke, "stroke-width": 1.5,
      "stroke-linejoin": "round", "stroke-linecap": "round",
      "vector-effect": "non-scaling-stroke",
    })
  );
  container.appendChild(svg);
}
