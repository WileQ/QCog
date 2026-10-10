"""Visualisation layer: accessible figures with mandatory text alternatives.

Every figure produced by this package carries a plain-language description on the
figure object itself, which the interactive explorer, the HTML report and the
screen-reader text export all read from the same place.  A chart therefore cannot
be published without an accessible alternative.
"""

from .checks import (
    catalogue_checks,
    figure_checks,
    has_drawn_marks,
    standalone_checks,
    structure_checks,
)
from .figures import (
    CIRCUIT_STYLES,
    VisualizationUnavailable,
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
from .style import (
    CATEGORICAL,
    LINE_STYLES,
    MARKERS,
    PALETTE,
    SEQ_COLORS,
    annotate_figure,
    apply_style,
    color_for,
    describe,
    new_figure,
    series_style,
)

__all__ = [
    "PALETTE",
    "CATEGORICAL",
    "LINE_STYLES",
    "MARKERS",
    "SEQ_COLORS",
    "series_style",
    "annotate_figure",
    "apply_style",
    "color_for",
    "describe",
    "new_figure",
    "bloch_sphere",
    "circuit_figure",
    "CIRCUIT_STYLES",
    "VisualizationUnavailable",
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
    "standalone_checks",
    "has_drawn_marks",
    "structure_checks",
    "catalogue_checks",
    "figure_checks",
]