"""Q-Cog - a Qiskit-native toolkit for quantum-probability models of cognition.

Q-Cog models, simulates, fits and explains quantum-probability accounts of human
judgement, decision making and collective behaviour.  The library is organised in
layers, and each layer can be used on its own:

``qcog.core``
    States, questions (projective and unsharp), quantum channels, the execution
    engine, classical baselines, metrics and fitting.
``qcog.backends``
    Exact evaluation, shot sampling, compiled circuits on Qiskit Aer, and an
    optional IonQ backend.
``qcog.analysis``
    Interference decompositions, order effects, the QQ equality, entanglement
    measures and parameter sweeps.
``qcog.experiments``
    Reusable designs: two questions in both orders, two-player quantum games,
    populations of agents.
``qcog.models``
    Twenty-three documented models in four modules.
``qcog.viz``
    Accessible figures, each carrying its own text description.
``qcog.report``
    Self-contained HTML and Markdown reports, compliant with the accessibility
    practices the library documents.

A complete run in four lines::

    from qcog import get_model
    model = get_model("question-order")
    result = model.run()
    print(result.summary)

Everything is MIT licensed, runs on a free local simulator, and never needs
network access.
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import analysis, backends, core, experiments, models, viz  # noqa: F401
from .core import (  # noqa: F401
    CognitiveState,
    Engine,
    NoiseSpec,
    POVM,
    ProjectiveMeasurement,
    SequentialResult,
    amplitudes,
    basis_state,
    bell_state,
    computational,
    depolarizing_channel,
    dichotomous,
    from_bloch,
    mixed_state,
    question,
    run_sequence,
    tensor,
    unsharp_question,
)
from .analysis import (  # noqa: F401
    entanglement_report,
    interference_decomposition,
    order_effect,
    qq_equality,
    sweep,
)
from .experiments import QuantumGame, QuantumOpinionModel, TwoQuestionExperiment  # noqa: F401
from .explain import (  # noqa: F401
    explain_backend,
    explain_interference,
    explain_measurement_pair,
    explain_order_effect,
    explain_qq,
    summary_table,
)
from .models import Model, ModelResult  # noqa: F401
from .registry import MODELS, MODULES, catalogue, get_model, list_models, models_by_module, search  # noqa: F401
from .report import write_catalogue_report, write_markdown_report, write_model_report  # noqa: F401
from .selftest import run_evidence  # noqa: F401
from .viz import default_figures, text_alternatives  # noqa: F401

__all__ = [
    "__version__",
    "analysis",
    "backends",
    "core",
    "experiments",
    "models",
    "viz",
    "CognitiveState",
    "Engine",
    "NoiseSpec",
    "POVM",
    "ProjectiveMeasurement",
    "SequentialResult",
    "amplitudes",
    "basis_state",
    "bell_state",
    "computational",
    "depolarizing_channel",
    "dichotomous",
    "from_bloch",
    "mixed_state",
    "question",
    "run_sequence",
    "tensor",
    "unsharp_question",
    "entanglement_report",
    "interference_decomposition",
    "order_effect",
    "qq_equality",
    "sweep",
    "QuantumGame",
    "QuantumOpinionModel",
    "TwoQuestionExperiment",
    "explain_backend",
    "explain_interference",
    "explain_measurement_pair",
    "explain_order_effect",
    "explain_qq",
    "summary_table",
    "Model",
    "ModelResult",
    "MODELS",
    "MODULES",
    "catalogue",
    "get_model",
    "list_models",
    "models_by_module",
    "search",
    "write_catalogue_report",
    "write_markdown_report",
    "write_model_report",
    "run_evidence",
    "default_figures",
    "text_alternatives",
]