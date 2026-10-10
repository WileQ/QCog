"""Quantities that make a model's behaviour measurable.

Analysis functions take the output of a simulation (:class:`SequentialResult`,
:class:`CognitiveState`) and answer a specific question about it: how much of the
prediction is interference, how large is the order effect, does the QQ equality
hold, how entangled is the state, and how does any of these change as a parameter
moves.
"""

from .entanglement import EntanglementReport, concurrence_and_entropy, entanglement_report
from .interference import InterferenceReport, interference_decomposition, which_path_table
from .order import (
    OrderEffectReport,
    QQReport,
    law_of_total_probability_test,
    order_effect,
    qq_equality,
    sure_thing_test,
)
from .sweeps import SweepResult, decoherence_sweep, sharpness_sweep, sweep

__all__ = [
    "EntanglementReport",
    "concurrence_and_entropy",
    "entanglement_report",
    "InterferenceReport",
    "interference_decomposition",
    "which_path_table",
    "OrderEffectReport",
    "QQReport",
    "law_of_total_probability_test",
    "order_effect",
    "qq_equality",
    "sure_thing_test",
    "SweepResult",
    "decoherence_sweep",
    "sharpness_sweep",
    "sweep",
]