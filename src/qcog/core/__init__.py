"""Numerical core of Q-Cog: states, questions, measurements, channels and results.

Everything in this sub-package is framework-agnostic apart from using Qiskit's
``Statevector``/``DensityMatrix`` for simulation and linear algebra, which is
what keeps a single definition of a model valid for both the exact and the
circuit execution paths.
"""

from .engine import Engine, run_sequence
from .instruments import (
    Channel,
    NoiseModel,
    NoiseSpec,
    amplitude_damping_channel,
    bit_flip_channel,
    depolarizing_channel,
    dephasing_channel,
    nonselective_channel,
    per_qubit_channel,
    phase_damping_channel,
)
from .measurement import (
    POVM,
    Measurement,
    ProjectiveMeasurement,
    computational,
    dichotomous,
    incompatibility,
    local_question,
    local_unsharp_question,
    measurements_commute,
    observable_measurement,
    question,
    unsharp_question,
    x_question,
    y_question,
    z_question,
)
from .results import SequentialResult, StepRecord
from .states import (
    CognitiveState,
    amplitudes,
    as_state,
    basis_state,
    bell_state,
    from_bloch,
    maximally_mixed,
    mixed_state,
    question_state,
    tensor,
    uniform_state,
)

__all__ = [
    "Engine",
    "run_sequence",
    "Channel",
    "NoiseModel",
    "NoiseSpec",
    "amplitude_damping_channel",
    "bit_flip_channel",
    "depolarizing_channel",
    "dephasing_channel",
    "nonselective_channel",
    "per_qubit_channel",
    "phase_damping_channel",
    "POVM",
    "Measurement",
    "ProjectiveMeasurement",
    "computational",
    "dichotomous",
    "incompatibility",
    "local_question",
    "local_unsharp_question",
    "measurements_commute",
    "observable_measurement",
    "question",
    "unsharp_question",
    "x_question",
    "y_question",
    "z_question",
    "SequentialResult",
    "StepRecord",
    "CognitiveState",
    "amplitudes",
    "as_state",
    "basis_state",
    "bell_state",
    "from_bloch",
    "maximally_mixed",
    "mixed_state",
    "question_state",
    "tensor",
    "uniform_state",
]