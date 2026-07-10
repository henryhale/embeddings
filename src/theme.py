"""Chart palette and Plotly styling.

Dark mode is a *selected* set of steps for the dark surface, not an automatic
inversion of the light one. Both sets were run through the palette validator.

Categorical use here is deliberately capped at three slots: 3D scatter is an
"all pairs visible at once" form, where the full eight-slot order cannot hold
the colourblind-separation floor. Three can, in both modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    page: str
    text_primary: str
    text_secondary: str
    muted: str
    grid: str
    axis: str
    # Categorical identity — assigned in fixed order, never cycled.
    series: tuple[str, ...]
    # Sequential magnitude — one hue, light to dark.
    sequential: tuple[str, ...]
    # Diverging polarity — two poles with a neutral gray midpoint.
    diverging: tuple[str, str, str]
    highlight: str
    dim: str
    plotly_template: str = "plotly_white"


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    page="#f9f9f7",
    text_primary="#0b0b0b",
    text_secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    series=("#2a78d6", "#eb6834", "#1baf7a"),
    sequential=("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf",
                "#1c5cab", "#184f95", "#104281", "#0d366b"),
    diverging=("#2a78d6", "#f0efec", "#e34948"),
    highlight="#eb6834",
    dim="#c3c2b7",
    plotly_template="plotly_white",
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    page="#0d0d0d",
    text_primary="#ffffff",
    text_secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    series=("#3987e5", "#d95926", "#199e70"),
    sequential=("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf",
                "#1c5cab", "#184f95", "#104281", "#0d366b"),
    diverging=("#3987e5", "#383835", "#e66767"),
    highlight="#d95926",
    dim="#52514e",
    plotly_template="plotly_dark",
)


def current() -> Theme:
    """Best-effort detection of the active Streamlit theme."""
    try:
        import streamlit as st

        # Streamlit >= 1.39 exposes the resolved theme on st.context.
        base = getattr(getattr(st, "context", None), "theme", None)
        kind = getattr(base, "type", None)
        if kind is None:
            kind = st.get_option("theme.base")
        return DARK if str(kind).lower() == "dark" else LIGHT
    except Exception:
        return LIGHT


def sequential_scale(theme: Theme) -> list[list]:
    """Plotly colorscale spec from the sequential ramp."""
    n = len(theme.sequential)
    return [[i / (n - 1), c] for i, c in enumerate(theme.sequential)]


def diverging_scale(theme: Theme) -> list[list]:
    lo, mid, hi = theme.diverging
    return [[0.0, lo], [0.5, mid], [1.0, hi]]


def apply_layout(fig, theme: Theme, height: int = 620, legend: bool = True) -> None:
    """Common chrome: recessive grid, transparent surface, text-token ink."""
    fig.update_layout(
        template=theme.plotly_template,
        height=height,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family='system-ui, -apple-system, "Segoe UI", sans-serif',
            color=theme.text_secondary,
            size=12,
        ),
        showlegend=legend,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.01,
            xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=theme.text_secondary),
        ),
        hoverlabel=dict(
            bgcolor=theme.surface,
            font=dict(color=theme.text_primary, size=12),
            bordercolor=theme.axis,
        ),
    )
