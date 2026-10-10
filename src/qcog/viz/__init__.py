"""Visualisation layer: accessible figures with mandatory text alternatives.

Every figure produced by this package carries a plain-language description on the
figure object itself, which the interactive explorer, the HTML report and the
screen-reader text export all read from the same place.  A chart therefore cannot
be published without an accessible alternative.
"""

from .figures import (
    bloch_sphere,
    circuit_figure,
    classical_comparison_figure,
    default_figures,
    heatmap_figure,
    histogram_figure,
    interference_figure,
    line_figure,
    marginal_bars,
    order_effect_figure,
    probability_bars,
    savefig,
    sweep_figure,
    text_alternatives,
)
from .style import PALETTE, SEQ_COLORS, annotate_figure, apply_style, color_for, describe, new_figure

__all__ = [
    "PALETTE",
    "SEQ_COLORS",
    "annotate_figure",
    "apply_style",
    "color_for",
    "describe",
    "new_figure",
    "bloch_sphere",
    "circuit_figure",
    "classical_comparison_figure",
    "default_figures",
    "heatmap_figure",
    "histogram_figure",
    "interference_figure",
    "line_figure",
    "marginal_bars",
    "order_effect_figure",
    "probability_bars",
    "savefig",
    "sweep_figure",
    "text_alternatives",
]