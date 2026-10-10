"""Execution on Qiskit Aer: the same model, compiled to a circuit and sampled.

Two things make this backend more than a re-implementation:

* the circuit is built from the *same* projectors the analytic backend uses, so
  agreement between the two is a genuine cross-check of both the basis-change
  compilation and the collapse semantics of mid-circuit measurement;
* it is the backend used by the Qollab / IonQ submission path, because it only
  needs standard gates, and ``mid_circuit_measurement=False`` produces a circuit
  that even hardware without mid-circuit measurement can execute.
"""

from __future__ import annotations

import time
from typing import Mapping, Sequence

import numpy as np

from ..circuits.builder import build_sequence_circuit, circuit_statistics, decode_counts
from ..core.linalg import num_qubits_for_dim
from ..core.measurement import Measurement
from ..core.results import SequentialResult, StepRecord
from ..core.states import as_state

__all__ = ["AerBackend", "AerExactBackend", "aer_noise_model", "AER_METHODS"]

AER_METHODS = ("automatic", "statevector", "density_matrix", "matrix_product_state", "stabilizer")


def aer_noise_model(kind: str = "depolarizing", strength: float = 0.02, num_qubits: int = 2):
    """Build a Qiskit Aer ``NoiseModel`` for one of the standard noise processes.

    This complements the Kraus-based decoherence in :mod:`qcog.core.instruments`:
    the Kraus path is exact and deterministic, the Aer noise model path samples
    device-like errors inside the simulator.
    """
    from qiskit_aer.noise import NoiseModel as AerNoiseModel
    from qiskit_aer.noise import depolarizing_error

    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must lie in [0, 1]")
    model = AerNoiseModel()
    if kind == "depolarizing":
        error = depolarizing_error(strength, 1)
        model.add_all_qubit_quantum_error(error, ["u", "u1", "u2", "u3", "rx", "ry", "rz", "h", "x"])
    elif kind == "readout":
        model.add_all_qubit_readout_error([[1 - strength, strength], [strength, 1 - strength]])
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown Aer noise kind {kind!r}; expected 'depolarizing' or 'readout'")
    return model


class AerBackend:
    """Sample a compiled interview on Qiskit's Aer simulator."""

    name = "aer"
    description = "compiled Qiskit circuit executed on the Aer simulator"

    def __init__(self, method: str = "automatic", label: str | None = None, **options) -> None:
        if method not in AER_METHODS:
            raise ValueError(f"unknown Aer method {method!r}; expected one of {AER_METHODS}")
        self.method = method
        self.label = label or f"aer[{method}]"
        self.options = options

    def build_circuit(self, state, measurements: Sequence[Measurement], mid_circuit_measurement: bool = True):
        """The compiled circuit for this interview (public: also used by the report generator)."""
        return build_sequence_circuit(
            state, measurements, mid_circuit_measurement=mid_circuit_measurement
        )

    def run(
        self,
        state,
        measurements: Sequence[Measurement],
        *,
        shots: int = 4096,
        seed: int | None = None,
        mid_circuit_measurement: bool = True,
        aer_noise=None,
        method: str | None = None,
        noise: object | None = None,
        between_steps: Mapping | None = None,
        **metadata,
    ) -> SequentialResult:
        """Run the interview and return shot-based joint statistics.

        ``noise`` (a :class:`~qcog.core.instruments.NoiseModel`) is accepted for
        interface compatibility; because hardware noise is applied by Aer rather
        than by Kraus operators, use ``aer_noise`` (see :func:`aer_noise_model`)
        when you want device-like errors inside the simulation.
        """
        from qiskit_aer import AerSimulator

        if shots <= 0:
            raise ValueError("shots must be a positive integer")
        confidence = as_state(state)
        questions = list(measurements)
        names = [question.name for question in questions]
        if len(set(names)) != len(names):
            raise ValueError(f"question names must be unique within one interview, got {names}")
        from qiskit import transpile

        circuit = self.build_circuit(confidence, questions, mid_circuit_measurement=mid_circuit_measurement)
        choose = method or self.method
        simulator = AerSimulator(noise_model=aer_noise, method=choose, seed_simulator=seed)
        # Aer executes gates, not Qiskit-level abstractions such as
        # ``StatePreparation`` or ``UnitaryGate``: transpile against the Aer
        # target so that only native instructions reach the simulator.
        compiled = transpile(
            circuit, simulator, optimization_level=self.options.get("optimization_level", 1),
            seed_transpiler=seed,
        )
        started = time.perf_counter()
        counts = simulator.run(compiled, shots=shots, seed_simulator=seed).result().get_counts()
        duration = time.perf_counter() - started
        joint_counts = decode_counts(
            counts, questions, num_qubits_for_dim(confidence.dim), circuit.metadata.get("qcog_layout")
        )
        joint = {key: value / shots for key, value in joint_counts.items()}
        return SequentialResult(
            order=tuple(names),
            outcomes={question.name: tuple(question.outcomes) for question in questions},
            joint=joint,
            steps=_conditional_steps(questions, joint),
            states={},
            backend=self.label,
            shots=shots,
            seed=seed,
            counts=joint_counts,
            duration_seconds=duration,
            noise=metadata.pop("noise_description", None),
            metadata={
                **metadata,
                "circuit": circuit_statistics(circuit),
                "compiled_circuit": circuit_statistics(compiled),
                "circuit_object": compiled,
                "method": choose,
                "mid_circuit_measurement": mid_circuit_measurement,
                "aer": True,
                "measurements": {question.name: question for question in questions},
            },
        )


def _conditional_steps(questions: Sequence[Measurement], joint: Mapping[tuple[str, ...], float]) -> list[StepRecord]:
    """Recover the prefix-conditional distributions from joint shot statistics."""
    records: list[StepRecord] = []
    prefixes: list[tuple[str, ...]] = [()]
    for step, question in enumerate(questions):
        next_prefixes: list[tuple[str, ...]] = []
        for prefix in prefixes:
            mass = sum(
                value
                for key, value in joint.items()
                if key[:step] == prefix
            )
            conditional = {}
            for outcome in question.outcomes:
                target = prefix + (outcome,)
                branch = sum(value for key, value in joint.items() if key[: step + 1] == target)
                conditional[outcome] = float(branch / mass) if mass > 0 else float("nan")
                next_prefixes.append(target)
            records.append(
                StepRecord(
                    step=step,
                    measurement=question.name,
                    outcomes=tuple(question.outcomes),
                    prefix=prefix,
                    conditional=conditional,
                )
            )
        prefixes = [prefix for prefix in next_prefixes if prefix[: step + 1] in {k[: step + 1] for k in joint}]
    return records


class AerExactBackend:
    """Exact probabilities obtained from *compiled circuits* rather than projectors.

    For every answer prefix the backend builds the small prefix circuit of
    :func:`qcog.circuits.builder.build_prefix_circuit`, saves the statevector just
    before the measurement and reads the outcome populations off it.  The joint
    distribution is the product of those conditional distributions.

    This is the honest cross-check of the whole compilation chain: it exercises
    state preparation, the basis-change unitaries and the frame bookkeeping,
    while producing exact numbers that can be compared with the analytic
    backend to machine precision.  It also needs no mid-circuit measurement and
    no ``reset``, so it maps directly onto restricted hardware.
    """

    name = "aer_exact"
    description = "exact outcome probabilities read from the statevectors of compiled prefix circuits"

    def __init__(self, method: str = "statevector", **options) -> None:
        if method not in {"statevector", "density_matrix"}:
            raise ValueError("AerExactBackend supports method='statevector' or 'density_matrix'")
        self.method = method
        self.options = options

    def run(
        self,
        state,
        measurements: Sequence[Measurement],
        *,
        noise=None,
        between_steps=None,
        shots=None,
        seed=None,
        **metadata,
    ) -> SequentialResult:
        from qiskit import transpile
        from qiskit_aer import AerSimulator

        from ..circuits.builder import build_prefix_circuits, outcome_populations

        if between_steps or (noise is not None and not getattr(noise, "is_trivial", True)):
            raise ValueError(
                "AerExactBackend evaluates the noiseless circuits; use backend='exact' for "
                "Kraus-level decoherence or backend='aer' for device-like noise."
            )
        confidence = as_state(state)
        questions = list(measurements)
        num_qubits = num_qubits_for_dim(confidence.dim)
        circuits = build_prefix_circuits(confidence, questions, measure=False)
        simulator = AerSimulator(method=self.method)
        started = time.perf_counter()
        conditionals: dict[tuple[str, ...], dict[str, float]] = {}
        for prefix, circuit in circuits.items():
            prepared = circuit.copy()
            prepared.save_statevector()
            compiled = transpile(prepared, simulator, optimization_level=1)
            statevector = np.asarray(simulator.run(compiled).result().get_statevector())
            question = questions[len(prefix)]
            conditionals[prefix] = outcome_populations(statevector, question)
        duration = time.perf_counter() - started
        joint: dict[tuple[str, ...], float] = {(): 1.0}
        for prefix in sorted(conditionals, key=len):
            parent = joint.get(prefix, 0.0)
            for outcome, probability in conditionals[prefix].items():
                joint[prefix + (outcome,)] = parent * probability
        joint_final = {
            key: value for key, value in joint.items() if len(key) == len(questions)
        }
        steps = [
            StepRecord(
                step=len(prefix),
                measurement=questions[len(prefix)].name,
                outcomes=tuple(questions[len(prefix)].outcomes),
                prefix=prefix,
                conditional=dict(values),
            )
            for prefix, values in conditionals.items()
        ]
        return SequentialResult(
            order=tuple(question.name for question in questions),
            outcomes={question.name: tuple(question.outcomes) for question in questions},
            joint=joint_final,
            steps=steps,
            states={},
            backend=self.name,
            shots=None,
            seed=seed,
            counts=None,
            duration_seconds=duration,
            noise=None,
            metadata={
                **metadata,
                "method": self.method,
                "num_prefix_circuits": len(circuits),
                "circuit": circuit_statistics(next(iter(circuits.values()))),
                "sources": [
                    circuit_statistics(circuit) for circuit in circuits.values()
                ][:8],
            },
        )
