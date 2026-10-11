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
from contextlib import contextmanager

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
    "MONO_PALETTE",
    "MONO_CATEGORICAL",
    "MONO_HATCHES",
    "LINE_STYLES",
    "MARKERS",
    "SEQ_COLORS",
    "apply_style",
    "new_figure",
    "color_for",
    "series_style",
    "bar_style",
    "monochrome",
    "is_monochrome",
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

#: The monochrome palette: black and white only, with the hierarchy carried by grey level
#: (opacity steps of the same ink) and by hatch, linestyle and marker.  Series never differ
#: by grey level alone, for the same reason they never differ by hue alone.
MONO_PALETTE = {
    "blue": "#000000",
    "orange": "#3f3f3f",
    "green": "#6e6e6e",
    "vermillion": "#1f1f1f",
    "purple": "#8f8f8f",
    "sky": "#b0b0b0",
    "yellow": "#d0d0d0",
    "grey": "#4d4d4d",
    "black": "#000000",
}

#: Series cycle in monochrome: darkest first, then the hatch and line style do the separating.
MONO_CATEGORICAL = ("blue", "vermillion", "green", "purple", "sky", "yellow")

#: Hatches cycled through when more series are drawn than there are usable greys.
MONO_HATCHES = ("", "//", "xx", "\\\\", "..", "++")

_MONO = False


def is_monochrome() -> bool:
    """True while figures are being drawn in the monochrome palette."""
    return _MONO


@contextmanager
def monochrome(enabled: bool = True):
    """Draw figures in black and white for the duration of the block.

    The palette names keep working (``PALETTE["blue"]`` is still the first series colour), they
    simply resolve to grey levels, so a figure written for colour output renders in monochrome
    without being changed.
    """
    global _MONO
    if not enabled:
        yield
        return
    previous_palette = dict(PALETTE)
    previous_colours = list(SEQ_COLORS)
    _MONO = True
    PALETTE.update(MONO_PALETTE)
    SEQ_COLORS[:] = [PALETTE[name] for name in MONO_CATEGORICAL]
    try:
        yield
    finally:
        PALETTE.clear()
        PALETTE.update(previous_palette)
        SEQ_COLORS[:] = previous_colours
        _MONO = False

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


def _cycle() -> tuple:
    return MONO_CATEGORICAL if _MONO else CATEGORICAL


def color_for(index: int) -> str:
    """Colour for the ``index``-th series in a figure."""
    names = _cycle()
    return PALETTE[names[index % len(names)]]


def series_style(index: int) -> dict:
    """Colour, linestyle and marker for the ``index``-th series of a figure.

    Colour cycles first, then linestyle, then marker, so that up to
    ``8 colours x 4 linestyles x 8 markers`` series stay distinguishable - and so that
    two series never differ *only* by hue, which is what breaks figures for readers with
    a colour-vision deficiency and in greyscale print.
    """
    names = _cycle()
    palette_name = names[index % len(names)]
    style_index = index // len(names)
    hatch = MONO_HATCHES[style_index % len(MONO_HATCHES)] if _MONO else ""
    return {
        "color": PALETTE[palette_name],
        "linestyle": LINE_STYLES[style_index % len(LINE_STYLES)],
        "marker": MARKERS[(style_index // len(LINE_STYLES)) % len(MARKERS)],
        "hatch": hatch,
        "label_style": f"{palette_name}{'' if style_index == 0 else ' ' + LINE_STYLES[style_index % len(LINE_STYLES)]}",
    }


def bar_style(index: int, *, hatch=None, edge_color: str = "none") -> dict:
    """Face, edge and hatch for the ``index``-th bar series.

    In colour the caller's own hatch and edge colour are used, so a chart looks exactly as it
    always did.  In monochrome the face is a grey level, the hatch carries the series and the edge
    is inked, so that the palest grey still reads as a bar.
    """
    style = series_style(index)
    if not _MONO:
        return {"color": style["color"], "hatch": hatch or "", "edgecolor": edge_color}
    return {"color": style["color"], "hatch": hatch if hatch else style["hatch"], "edgecolor": "#000000"}


def sequential_cmap() -> str:
    """The colour map for heat maps: greys in monochrome, viridis otherwise."""
    return "Greys" if _MONO else "viridis"


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