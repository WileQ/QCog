"""Figure style: accessible by default.

The palette is the Okabe-Ito colour-blind-safe set, the default font is a plain
sans-serif at a readable size, and every figure carries a plain-language
description in ``figure._qcog_description`` so that the same graphic can be offered
as text in the UI, in the HTML report and to a screen reader.  Making the
description part of the figure object means a chart can never be exported without
one.
"""

from __future__ import annotations

import os
import sys

import matplotlib

# Headless by default: the library is imported by scripts, servers and reports.  Inside a
# notebook or any other interactive session the user's own backend is kept, so that
# ``display(figure)`` shows the chart inline instead of rendering to a file.
if not os.environ.get("MPLBACKEND") and "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

__all__ = [
    "PALETTE",
    "CATEGORICAL",
    "LINE_STYLES",
    "MARKERS",
    "SEQ_COLORS",
    "apply_style",
    "new_figure",
    "color_for",
    "series_style",
    "annotate_figure",
    "describe",
]

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

#: The categorical cycle used for series inside one figure.  Eight colours, all with
#: enough contrast against white to be usable as lines; the pale yellow of the Okabe-Ito
#: set is deliberately left out of the *line* cycle and kept in :data:`PALETTE` for areas
#: and annotations, because a pale line on a white page is not readable.
CATEGORICAL = ("blue", "orange", "green", "vermillion", "purple", "sky", "grey", "black")

#: Linestyles and markers, combined with the colours so that a figure with more series
#: than colours still separates them by shape rather than by colour alone (which is also
#: what makes a figure readable in greyscale or with a colour-vision deficiency).
LINE_STYLES = ("-", "--", "-.", ":")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")

#: Backwards-compatible list of the line colours, in cycle order.
SEQ_COLORS = [PALETTE[name] for name in CATEGORICAL]

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
    return PALETTE[CATEGORICAL[index % len(CATEGORICAL)]]


def series_style(index: int) -> dict:
    """Colour, linestyle and marker for the ``index``-th series of a figure.

    Colour cycles first, then linestyle, then marker, so that up to
    ``8 colours x 4 linestyles x 8 markers`` series stay distinguishable - and so that
    two series never differ *only* by hue, which is what breaks figures for readers with
    a colour-vision deficiency and in greyscale print.
    """
    palette_name = CATEGORICAL[index % len(CATEGORICAL)]
    style_index = index // len(CATEGORICAL)
    return {
        "color": PALETTE[palette_name],
        "linestyle": LINE_STYLES[style_index % len(LINE_STYLES)],
        "marker": MARKERS[(style_index // len(LINE_STYLES)) % len(MARKERS)],
        "label_style": f"{palette_name}{'' if style_index == 0 else ' ' + LINE_STYLES[style_index % len(LINE_STYLES)]}",
    }


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