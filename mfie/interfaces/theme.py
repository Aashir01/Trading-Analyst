"""Chart palette and Plotly layout defaults.

The palette is the validated reference instance: eight categorical hues in a
fixed order, one sequential hue, a blue/red diverging pair, and a reserved
status palette. Colours are assigned by the *job* they do, never cycled.

Validation (OKLab ΔE ×100, adjacent pairlist, run with the palette validator):

* light surface ``#fcfcfb`` — worst CVD ΔE 9.2, worst normal-vision ΔE 27.6
* dark surface ``#1a1a19``  — worst CVD ΔE 9.4, worst normal-vision ΔE 26.5

Charts here use at most **three** categorical series, which is the documented
all-pairs-safe cap for this ordering (scatter and small multiples put every pair
on screen at once, not just adjacent ones). Beyond three, facet instead of
adding hues.

One light-mode slot (aqua, 2.74:1) sits below 3:1 against the light surface, so
the relief rule applies: every chart ships a legend and the dashboard exposes a
table view of the same numbers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    mode: str
    surface: str
    page: str
    ink_primary: str
    ink_secondary: str
    ink_muted: str
    grid: str
    axis: str
    border: str
    categorical: tuple[str, ...]
    sequential: tuple[str, ...]
    diverging_low: str
    diverging_mid: str
    diverging_high: str
    good: str = "#0ca30c"
    warning: str = "#fab219"
    serious: str = "#ec835a"
    critical: str = "#d03b3b"

    def series(self, index: int) -> str:
        """Categorical slot by position. Never wraps silently past slot 8."""
        if index >= len(self.categorical):
            raise IndexError(
                f"Categorical slot {index} requested but the palette has "
                f"{len(self.categorical)}. Fold extra series into 'Other' or facet."
            )
        return self.categorical[index]


LIGHT = Palette(
    mode="light",
    surface="#fcfcfb",
    page="#f9f9f7",
    ink_primary="#0b0b0b",
    ink_secondary="#52514e",
    ink_muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    border="rgba(11,11,11,0.10)",
    categorical=("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#008300", "#4a3aa7", "#e34948"),
    sequential=("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                "#2a78d6", "#256abf", "#184f95", "#0d366b"),
    diverging_low="#d03b3b",
    diverging_mid="#f0efec",
    diverging_high="#2a78d6",
)

DARK = Palette(
    mode="dark",
    surface="#1a1a19",
    page="#0d0d0d",
    ink_primary="#ffffff",
    ink_secondary="#c3c2b7",
    ink_muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    border="rgba(255,255,255,0.10)",
    categorical=("#3987e5", "#d95926", "#199e70", "#c98500",
                 "#d55181", "#008300", "#9085e9", "#e66767"),
    sequential=("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                "#2a78d6", "#256abf", "#184f95", "#0d366b"),
    diverging_low="#e66767",
    diverging_mid="#383835",
    diverging_high="#3987e5",
)

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def get_palette(mode: str = "dark") -> Palette:
    return DARK if mode == "dark" else LIGHT


def base_layout(palette: Palette, height: int = 420, title: str | None = None) -> dict:
    """Plotly layout with recessive chrome: hairline grid, muted axes, no frame."""
    return {
        "template": "plotly_dark" if palette.mode == "dark" else "plotly_white",
        "paper_bgcolor": palette.surface,
        "plot_bgcolor": palette.surface,
        "height": height,
        "title": {"text": title, "font": {"size": 15, "color": palette.ink_primary}} if title else None,
        "font": {"family": FONT_FAMILY, "color": palette.ink_secondary, "size": 12},
        "margin": {"l": 56, "r": 20, "t": 44 if title else 20, "b": 40},
        "xaxis": {
            "showgrid": False,
            "zeroline": False,
            "linecolor": palette.axis,
            "tickfont": {"color": palette.ink_muted, "size": 11},
        },
        "yaxis": {
            "showgrid": True,
            "gridcolor": palette.grid,
            "gridwidth": 1,
            "zeroline": False,
            "linecolor": palette.axis,
            "tickfont": {"color": palette.ink_muted, "size": 11},
        },
        # Crosshair + unified tooltip is the default reading affordance on any
        # time series; without it a chart is a picture, not an instrument.
        "hovermode": "x unified",
        "hoverlabel": {
            "bgcolor": palette.surface,
            "bordercolor": palette.border,
            "font": {"family": FONT_FAMILY, "color": palette.ink_primary, "size": 12},
        },
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0,
            "font": {"color": palette.ink_secondary, "size": 11},
            "bgcolor": "rgba(0,0,0,0)",
        },
        "showlegend": True,
    }


def line_style(palette: Palette, slot: int, dash: str | None = None) -> dict:
    """Thin 2px line in a categorical slot."""
    style = {"color": palette.series(slot), "width": 2}
    if dash:
        style["dash"] = dash
    return style


def status_color(palette: Palette, level: str) -> str:
    return {
        "good": palette.good,
        "warning": palette.warning,
        "serious": palette.serious,
        "critical": palette.critical,
    }.get(level, palette.ink_muted)


def verdict_status(verdict: str) -> str:
    """Map a signal verdict to a reserved status role."""
    return {
        "STRONG": "good",
        "MODERATE": "warning",
        "WEAK": "serious",
        "BLOCKED": "critical",
    }.get(verdict, "warning")
