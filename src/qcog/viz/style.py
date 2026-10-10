"""Figure style: accessible by default.

The palette is the Okabe-Ito colour-blind-safe set, the default font is a plain
sans-serif at a readable size, and every figure carries a plain-language
description in ``figure._qcog_description`` so that the same graphic can be offered
as text in the UI, in the HTML report and to a screen reader.  Making the
description part of the figure object means a chart can never be exported without
one.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless by default: the library is imported by scripts and servers

import matplotlib.pyplot as plt

__all__ = ["PALETTE", "SEQ_COLORS", "apply_style", "new_figure", "color_for", "annotate_figure", "describe"]

#: Okabe-Ito colour-blind-safe palette.
PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "grey": "#4D4D4D",
    "black": "#000000",
}

#: Colours for the recurring "quantum vs classical" contrast.
SEQ_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]

_STYLE = {
    "figure.dpi": 110,
    "figure.max_open_warning": 200,
    "savefig.dpi": 160,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#111111",
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.7,
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "font.family": "sans-serif",
    "font.size": 10,
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
}


def apply_style() -> None:
    """Install the Q-Cog matplotlib style (idempotent)."""
    plt.rcParams.update(_STYLE)


def new_figure(width: float = 6.4, height: float = 3.6):
    """Create a figure with the Q-Cog style applied."""
    apply_style()
    return plt.figure(figsize=(width, height))


def color_for(index: int) -> str:
    """Colour for the ``index``-th series in a figure."""
    return SEQ_COLORS[index % len(SEQ_COLORS)]


def annotate_figure(figure, description: str, *extra: str) -> None:
    """Attach an accessible description to a figure.

    The description is what :func:`qcog.viz.text_alternatives` returns, what the UI
    shows under the chart, and what the HTML report embeds in the ``alt``
    attribute.
    """
    figure._qcog_description = description
    figure._qcog_extra = [text for text in extra if text]


def describe(figure) -> str:
    """Return the accessible description of a figure (empty string if none)."""
    return getattr(figure, "_qcog_description", "")