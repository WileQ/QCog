"""Compile a Q-Cog interview into Qiskit circuits.

The compilation is deliberately boring, and that is the point: the *same*
projectors the analytic engine uses are turned into basis-change unitaries plus
computational-basis measurements, so every model in the library inherits a
circuit implementation for free.

Two equivalent compilations are provided, because they answer different needs.

``build_sequence_circuit``
    One circuit for the whole interview: rotate, measure, rotate back, rotate,
    measure, ...  A single shot yields a complete outcome tuple, which is what
    you want for estimating the joint distribution, and what makes order effects
    appear on real hardware.

``build_prefix_circuits``
    One small circuit per answer prefix.  These rely only on state preparation,
    a unitary and a measurement - no mid-circuit measurement and no ``reset`` -
    so they run on essentially any device.  They compute the *conditional*
    distribution of the next question exactly, and their product is the joint
    distribution.

Conventions
-----------
* Qubits are little-endian, matching ``qiskit.quantum_info.Statevector``: basis
  index ``k`` has qubit ``q`` in state ``(k >> q) & 1``.
* A model living in dimension ``d`` is embedded in ``2**ceil(log2 d)`` qubits;
  the unused amplitudes are exactly zero, so the embedding cannot change a
  prediction.
* Each question gets its own block of classical bits, so one shot produces one
  complete outcome tuple.

Why the "rotate back" step
--------------------------
If question ``i`` is compiled as ``U_i^dagger`` followed by a computational-basis
measurement, the simulator's state after the measurement lives in the *rotated*
frame, not in the physical frame.  Applying ``U_i`` again immediately afterwards
undoes the rotation without touching the recorded answer, so between questions
the simulator state is exactly the cognitive state the analytic engine works
with.  That keeps the circuits comparable to the analytic results, and it lets a
user save the statevector mid-interview to watch a belief change.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation, UnitaryGate

from ..core.linalg import (
    pvm_to_unitary,
    embed_matrix,
    embed_vector,
    num_qubits_for_dim,
    restrict_operators,
    trivial_qubits,
)
from ..core.measurement import Measurement, POVM
from ..core.states import CognitiveState, as_state

__all__ = [
    "build_state_preparation",
    "build_measurement_circuit",
    "build_sequence_circuit",
    "build_prefix_circuit",
    "build_prefix_circuits",
    "frame_unitaries",
    "measurement_qubits",
    "outcome_from_chunk",
    "decode_counts",
    "decode_bitstring",
    "circuit_statistics",
    "circuit_text_alternative",
]


def _preparable_state(state) -> CognitiveState:
    confidence = as_state(state)
    if confidence.vector is None:
        raise ValueError(
            "circuits can only be prepared for pure states; "
            "use the analytic backend for mixed states"
        )
    return confidence


def measurement_qubits(measurement: Measurement, num_qubits: int) -> tuple[int, ...]:
    """The qubits a question actually has to measure, in ascending order.

    A question about one subsystem leaves the others untouched, so only its own
    qubits may be measured; measuring the rest would collapse amplitudes the
    question does not involve.  If the question is genuinely non-local (a Bell-basis
    question, for instance) every qubit is required.
    """
    question = _require_projective(measurement)
    projectors = question.projectors
    trivial = trivial_qubits(projectors, num_qubits)
    required = tuple(qubit for qubit in range(num_qubits) if qubit not in trivial)
    return required or tuple(range(num_qubits))


def _require_projective(measurement: Measurement) -> Measurement:
    if isinstance(measurement, POVM) and not measurement.is_projective:
        raise ValueError(
            f"question {measurement.name!r} is an unsharp POVM and cannot be compiled directly; "
            "dilute it into a projective question or use the analytic backend"
        )
    if isinstance(measurement, POVM):
        return measurement.to_projective_limit()
    return measurement


def build_state_preparation(state, name: str = "state_prep") -> QuantumCircuit:
    """Circuit preparing ``state`` (a pure state) on ``num_qubits`` qubits."""
    confidence = _preparable_state(state)
    num_qubits = num_qubits_for_dim(confidence.dim)
    circuit = QuantumCircuit(num_qubits, name=name)
    circuit.append(StatePreparation(embed_vector(confidence.amplitudes), label="psi"), range(num_qubits))
    return circuit


def basis_change_gate(measurement: Measurement, num_qubits: int | None = None):
    """``(U_dagger, U, blocks, support)`` for a projective question.

    ``U_dagger`` rotates the state into the question's eigenbasis on the qubits the
    question actually concerns, ``U`` rotates back, ``blocks`` maps each outcome to
    the computational-basis indices it occupies, and ``support`` lists those qubits.
    Compiling a question onto its own qubits - rather than onto the whole register -
    keeps the circuit minimal and stops a local question from collapsing amplitudes
    it does not measure.
    """
    question = _require_projective(measurement)
    total_qubits = num_qubits if num_qubits is not None else num_qubits_for_dim(question.dim)
    support = measurement_qubits(question, total_qubits)
    if len(support) == total_qubits:
        unitary, blocks = question.basis_unitary()
    else:
        local = restrict_operators(question.projectors, support, total_qubits)
        unitary, blocks = pvm_to_unitary(local)
    forward = UnitaryGate(unitary.conj().T, label=f"{question.name}")
    backward = UnitaryGate(unitary, label=f"{question.name}:undo")
    return forward, backward, blocks, support


def build_measurement_circuit(
    measurement: Measurement,
    num_qubits: int | None = None,
    clbits: Sequence[int] | None = None,
    name: str = "measure",
) -> QuantumCircuit:
    """Standalone circuit that measures ``measurement`` on a fresh register."""
    num_qubits = num_qubits or num_qubits_for_dim(measurement.dim)
    clbits = list(clbits) if clbits is not None else list(range(num_qubits))
    forward, _, _, support = basis_change_gate(measurement, num_qubits)
    circuit = QuantumCircuit(num_qubits, len(support), name=name)
    circuit.append(forward, list(support))
    circuit.measure(list(support), list(range(len(support))))
    return circuit


def build_sequence_circuit(
    state,
    measurements: Sequence[Measurement],
    *,
    mid_circuit_measurement: bool = True,
    undo_rotation: bool = True,
    barriers: bool = True,
    name: str = "qcog_interview",
) -> QuantumCircuit:
    """One circuit for the whole interview.

    ``mid_circuit_measurement=True`` (default) measures after every question, so
    the simulator's collapse implements the Lueders update.  ``undo_rotation``
    restores the physical frame after each collapse (see the module docstring);
    turning it off reproduces the "kept in the rotated frame" behaviour, which is
    useful for demonstrating why the step matters.

    With ``mid_circuit_measurement=False`` only the last question is measured -
    that is the classical-like comparison circuit in which every question is put
    to an untouched belief state.
    """
    confidence = _preparable_state(state)
    questions = [_require_projective(measurement) for measurement in measurements]
    num_qubits = num_qubits_for_dim(confidence.dim)
    layout: list[tuple[int, ...]] = []
    planned = [measurement_qubits(question, num_qubits) for question in questions]
    total_clbits = sum(len(entry) for entry in planned)
    circuit = QuantumCircuit(num_qubits, total_clbits, name=name)
    circuit.append(StatePreparation(embed_vector(confidence.amplitudes), label="psi"), range(num_qubits))
    for step, question in enumerate(questions):
        forward, backward, _, support = basis_change_gate(question, num_qubits)
        is_last = step == len(questions) - 1
        if barriers:
            circuit.barrier()
        circuit.append(forward, list(support))
        if mid_circuit_measurement or is_last:
            offset = sum(len(entry) for entry in layout)
            targets = list(range(offset, offset + len(support)))
            circuit.measure(list(support), targets)
            layout.append(support)
            if undo_rotation and not is_last:
                circuit.append(backward, list(support))
    circuit.metadata["qcog_layout"] = [list(entry) for entry in layout]
    return circuit


def frame_unitaries(measurements: Sequence[Measurement]) -> list[np.ndarray]:
    """Accumulated frame unitaries ``W_i`` used by :func:`build_prefix_circuit`.

    ``W_1 = U_1^dagger`` and ``W_i = U_i^dagger U_{i-1}`` for ``i > 1``, with
    ``U_i`` the basis-change unitary of question ``i``.  The conditional
    probability of answer ``a_i`` given ``a_1 ... a_{i-1}`` is then
    ``|<a_i| W_i |a_{i-1}>|^2`` for ``i > 1`` and ``|<a_1| W_1 |psi>|^2`` for the
    first question.
    """
    questions = [_require_projective(measurement) for measurement in measurements]
    if not questions:
        raise ValueError("at least one question is required")
    num_qubits = num_qubits_for_dim(questions[0].dim)
    unitaries = []
    for question in questions:
        unitary, _ = question.basis_unitary()
        unitaries.append(embed_matrix(unitary, 2**num_qubits))
    frames = [unitaries[0].conj().T]
    for index in range(1, len(unitaries)):
        frames.append(unitaries[index].conj().T @ unitaries[index - 1])
    return frames


def build_prefix_circuit(
    state,
    measurements: Sequence[Measurement],
    prefix: Sequence[str] = (),
    *,
    measure: bool = True,
    name: str = "qcog_prefix",
) -> QuantumCircuit:
    """Circuit giving the conditional distribution of the next question.

    ``prefix`` lists the answers already obtained.  Passing ``()`` yields the
    circuit for the first question; passing ``("yes",)`` the circuit for the second
    question given that the first was answered "yes", and so on.

    The circuit prepares the projected, renormalised state explicitly and then
    applies the accumulated frame unitary, so it needs neither mid-circuit
    measurement nor ``reset``.  The projection is computed on the whole block that
    the answer corresponds to, which matters for questions with degenerate outcomes
    (a rank-2 answer is a subspace, not a single basis state).
    """
    confidence = _preparable_state(state)
    questions = [_require_projective(measurement) for measurement in measurements]
    if len(prefix) >= len(questions):
        raise ValueError("the prefix already answers every question")
    frames = frame_unitaries(questions)
    num_qubits = num_qubits_for_dim(confidence.dim)
    step = len(prefix)
    frame_state = embed_vector(confidence.amplitudes)
    for index in range(step):
        frame_state = frames[index] @ frame_state
        question = questions[index]
        _, start_index, stop_index = question.basis_unitary()[1][question.index(prefix[index])]
        mask = np.zeros_like(frame_state)
        mask[start_index:stop_index] = 1.0
        frame_state = mask * frame_state
        norm = float(np.linalg.norm(frame_state))
        if norm < 1e-12:
            raise ValueError(
                f"the answer prefix {tuple(prefix)!r} has probability zero for this state"
            )
        frame_state = frame_state / norm
    frame_state = frames[step] @ frame_state
    qubits = measurement_qubits(questions[step], num_qubits)
    circuit = QuantumCircuit(num_qubits, len(qubits), name=name)
    circuit.append(StatePreparation(frame_state, label="psi"), range(num_qubits))
    if measure:
        circuit.measure(list(qubits), range(len(qubits)))
    circuit.metadata["qcog_layout"] = [list(qubits)]
    return circuit


def build_prefix_circuits(
    state,
    measurements: Sequence[Measurement],
    *,
    name: str = "qcog_prefix",
    measure: bool = True,
) -> dict[tuple[str, ...], QuantumCircuit]:
    """Every prefix circuit needed for a complete joint distribution."""
    questions = [_require_projective(measurement) for measurement in measurements]
    circuits: dict[tuple[str, ...], QuantumCircuit] = {}
    frontier: list[tuple[str, ...]] = [()]
    for step, question in enumerate(questions):
        next_frontier: list[tuple[str, ...]] = []
        for prefix in frontier:
            circuits[prefix] = build_prefix_circuit(
                state, questions, prefix, measure=measure, name=f"{name}_{step}"
            )
            if step < len(questions) - 1:
                next_frontier.extend(prefix + (outcome,) for outcome in question.outcomes)
        frontier = next_frontier
    return circuits


_BLOCK_CACHE: dict[tuple, tuple] = {}


def outcome_from_chunk(measurement: Measurement, support: Sequence[int], chunk: str, num_qubits: int) -> str:
    """Decode one measured block of bits into an outcome label.

    The bits that were measured are read in the order the circuit used (the support
    qubits in ascending order), so the outcome is looked up in the *restricted*
    basis of the question - the same basis the circuit applied - rather than in the
    basis of the whole register.
    """
    question = _require_projective(measurement)
    support = tuple(sorted(int(qubit) for qubit in support))
    if len(support) == num_qubits:
        index = 0
        for position, qubit in enumerate(support):
            index |= int(chunk[position]) << qubit
        return question.decode(index)
    key = (question.name, question.dim, support, num_qubits)
    cached = _BLOCK_CACHE.get(key)
    if cached is None:
        local = restrict_operators(question.projectors, support, num_qubits)
        cached = pvm_to_unitary(local)[1]
        _BLOCK_CACHE[key] = cached
    local_index = int(chunk[::-1], 2) if len(chunk) > 1 else int(chunk)
    for outcome_index, start, stop in cached:
        if start <= local_index < stop:
            return question.outcomes[outcome_index]
    raise ValueError(f"measured bits {chunk!r} do not belong to any outcome of {question.name!r}")


def decode_bitstring(
    bitstring: str,
    measurements: Sequence[Measurement],
    num_qubits: int,
    layout: Sequence[Sequence[int]] | None = None,
) -> tuple[str, ...]:
    """Decode one full shot into one outcome per question.

    ``layout`` lists, for each question, the qubits that were measured for it (in
    ascending order).  Qubits that a question does not involve are reported to the
    decoder as 0: since the question acts trivially on them, the outcome it decodes
    is the same for every value they could take.
    """
    if layout is None:
        layout = [tuple(range(num_qubits))] * len(measurements)
    total_expected = sum(len(entry) for entry in layout)
    if len(bitstring) != total_expected:
        raise ValueError(
            f"bitstring of length {len(bitstring)} does not match the circuit layout "
            f"({total_expected} classical bits for {len(measurements)} questions)"
        )
    # Qiskit reports bitstrings with the highest classical bit on the left, so
    # reverse once and read each question's bits left to right.
    bits = bitstring[::-1]
    decoded: list[str] = []
    cursor = 0
    for measurement, qubits in zip(measurements, layout):
        chunk = bits[cursor : cursor + len(qubits)]
        cursor += len(qubits)
        decoded.append(outcome_from_chunk(measurement, qubits, chunk, num_qubits))
    return tuple(decoded)


def decode_counts(
    counts: dict[str, int],
    measurements: Sequence[Measurement],
    num_qubits: int,
    layout: Sequence[Sequence[int]] | None = None,
) -> dict[tuple[str, ...], int]:
    """Aggregate raw shot counts into joint outcome counts."""
    joint: dict[tuple[str, ...], int] = {}
    for bitstring, count in counts.items():
        if not all(character in "01" for character in bitstring):
            raise ValueError(f"message {bitstring!r} is not a computational-basis bitstring")
        key = decode_bitstring(bitstring, measurements, num_qubits, layout)
        joint[key] = joint.get(key, 0) + int(count)
    return joint


def outcome_populations(statevector, measurement: Measurement) -> dict[str, float]:
    """Populations of a statevector grouped by the outcomes of ``measurement``."""
    probabilities = np.abs(np.asarray(statevector)) ** 2
    question = _require_projective(measurement)
    _, blocks = question.basis_unitary()
    populations = {outcome: 0.0 for outcome in question.outcomes}
    for outcome_index, start, stop in blocks:
        populations[question.outcomes[outcome_index]] += float(probabilities[start:stop].sum())
    return populations


def circuit_statistics(circuit: QuantumCircuit) -> dict:
    """Small, UI-friendly circuit description (accessible alternative to a drawing)."""
    operations: dict[str, int] = {}
    for instruction in circuit.data:
        operations[instruction.operation.name] = operations.get(instruction.operation.name, 0) + 1
    return {
        "name": circuit.name,
        "qubits": circuit.num_qubits,
        "clbits": circuit.num_clbits,
        "depth": int(circuit.depth()) if circuit.num_qubits else 0,
        "size": int(circuit.size()),
        "operations": operations,
    }


def circuit_text_alternative(circuit: QuantumCircuit, measurements: Sequence[Measurement] | None = None) -> str:
    """Plain-language description of a circuit, for screen readers and reports."""
    stats = circuit_statistics(circuit)
    operations = ", ".join(f"{count}x {name}" for name, count in sorted(stats["operations"].items()))
    text = (
        f"Circuit {stats['name']}: {stats['qubits']} qubit(s), {stats['clbits']} classical bit(s), "
        f"depth {stats['depth']}, containing {operations}."
    )
    if measurements:
        text += " Questions, in order: " + "; ".join(measurement.describe() for measurement in measurements) + "."
    return text