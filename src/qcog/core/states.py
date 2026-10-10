"""Cognitive states: pure states, mixed states and the operations we need on them.

A :class:`CognitiveState` wraps a density matrix and, when the state is pure,
also keeps the amplitude vector.  Keeping the amplitudes matters: interference
terms, phase information and "which-path" decompositions are only well defined
for pure states, and those are exactly the quantities Q-Cog visualises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .linalg import (
    TOL,
    as_complex_array,
    is_hermitian,
    normalize,
    num_qubits_for_dim,
    purity as _purity,
    von_neumann_entropy,
)

__all__ = [
    "CognitiveState",
    "basis_state",
    "uniform_state",
    "amplitudes",
    "from_bloch",
    "question_state",
    "tensor",
    "bell_state",
    "mixed_state",
    "maximally_mixed",
    "as_state",
]


@dataclass
class CognitiveState:
    """A quantum state of a cognitive model.

    Parameters
    ----------
    density:
        ``(d, d)`` Hermitian, unit-trace, positive semi-definite matrix.
    vector:
        Amplitude vector of length ``d`` when the state is pure, else ``None``.
    label:
        Human-readable label used in figures, tables and explanations.
    basis_labels:
        Optional labels for the computational basis states, e.g.
        ``("bank teller and feminist", "bank teller and not feminist", ...)``.
    """

    density: np.ndarray
    vector: np.ndarray | None = None
    label: str = "psi"
    basis_labels: tuple[str, ...] | None = None
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ build
    def __post_init__(self) -> None:
        self.density = as_complex_array(self.density, 2)
        if self.density.shape[0] != self.density.shape[1]:
            raise ValueError("density matrix must be square")
        if not is_hermitian(self.density, 1e-8):
            raise ValueError("density matrix must be Hermitian")
        trace = float(np.real(np.trace(self.density)))
        if abs(trace - 1.0) > 1e-6:
            raise ValueError(f"density matrix must have unit trace, got {trace!r}")
        if self.vector is not None:
            self.vector = as_complex_array(self.vector, 1)
            if self.vector.size != self.density.shape[0]:
                raise ValueError("vector and density matrix dimensions disagree")

    @classmethod
    def from_vector(cls, vector, label: str = "psi", basis_labels=None, **metadata) -> "CognitiveState":
        """Build a pure state from (possibly unnormalised) amplitudes."""
        vec = normalize(vector)
        return cls(
            density=np.outer(vec, vec.conj()),
            vector=vec,
            label=label,
            basis_labels=tuple(basis_labels) if basis_labels else None,
            metadata=metadata,
        )

    @classmethod
    def from_density(cls, density, label: str = "rho", vector=None, basis_labels=None, **metadata) -> "CognitiveState":
        """Build a (possibly mixed) state from a density matrix."""
        return cls(
            density=as_complex_array(density, 2),
            vector=None if vector is None else normalize(vector),
            label=label,
            basis_labels=tuple(basis_labels) if basis_labels else None,
            metadata=metadata,
        )

    # -------------------------------------------------------------- properties
    @property
    def dim(self) -> int:
        """Dimension of the cognitive Hilbert space."""
        return int(self.density.shape[0])

    @property
    def num_qubits(self) -> int:
        return num_qubits_for_dim(self.dim)

    @property
    def is_pure(self) -> bool:
        return _purity(self.density) > 1.0 - 1e-8

    @property
    def amplitudes(self) -> np.ndarray:
        """Amplitude vector; raises for genuinely mixed states."""
        if self.vector is None:
            raise ValueError(
                f"state {self.label!r} is mixed: amplitudes (and interference terms) are not defined"
            )
        return self.vector

    @property
    def purity(self) -> float:
        return _purity(self.density)

    @property
    def entropy(self) -> float:
        """Von Neumann entropy of the state in bits."""
        return von_neumann_entropy(self.density)

    def probabilities(self) -> np.ndarray:
        """Population of the computational basis."""
        return np.real(np.diag(self.density))

    def probability(self, index: int) -> float:
        return float(np.real(self.density[index, index]))

    def bloch_vector(self) -> np.ndarray | None:
        """Bloch vector ``(x, y, z)`` for a qubit, else ``None``."""
        if self.dim != 2:
            return None
        rho = self.density
        return np.array(
            [
                np.real(rho[0, 1] + rho[1, 0]),
                np.real(1j * (rho[0, 1] - rho[1, 0])),
                np.real(rho[0, 0] - rho[1, 1]),
            ]
        )

    def basis_label(self, index: int) -> str:
        if self.basis_labels and index < len(self.basis_labels):
            return self.basis_labels[index]
        return f"|{index}>"

    # -------------------------------------------------------------- operations
    def evolve(self, unitary) -> "CognitiveState":
        """Apply a unitary ``U`` (the outer transformation of a context/question)."""
        from ..circuits.gates import as_unitary

        mat = as_unitary(unitary, self.dim)
        density = mat @ self.density @ mat.conj().T
        vector = mat @ self.vector if self.vector is not None else None
        return CognitiveState.from_density(density, label=self.label, vector=vector,
                                           basis_labels=self.basis_labels, **self.metadata)

    def apply_channel(self, kraus_operators: Sequence[np.ndarray]) -> "CognitiveState":
        """Apply a quantum channel given by Kraus operators."""
        operators = [as_complex_array(op, 2) for op in kraus_operators]
        density = np.zeros_like(self.density)
        for op in operators:
            density += op @ self.density @ op.conj().T
        trace = float(np.real(np.trace(density)))
        if trace > TOL:
            density = density / trace
        return CognitiveState.from_density(density, label=self.label, basis_labels=self.basis_labels,
                                           **self.metadata)

    def partial_trace(self, keep: Sequence[int], dims: Sequence[int] | None = None) -> "CognitiveState":
        """Reduced state of a subsystem.

        ``dims`` describes the tensor structure (defaults to qubits).
        """
        from qiskit.quantum_info import partial_trace as _pt

        if dims is None:
            size = num_qubits_for_dim(self.dim)
            if 2**size != self.dim:
                raise ValueError("partial trace requires a power-of-two dimension or explicit dims")
            dims = [2] * size
        reduced = _pt(self.density, qargs=[q for q in range(len(dims)) if q not in set(keep)])
        return CognitiveState.from_density(np.asarray(reduced.data), label=f"{self.label}|keep{tuple(keep)}")

    # -------------------------------------------------------------- serialisation
    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "dim": self.dim,
            "is_pure": self.is_pure,
            "purity": round(self.purity, 6),
            "amplitudes": None if self.vector is None else _round_complex_list(self.vector),
            "probabilities": [round(p, 6) for p in self.probabilities()],
            "basis_labels": list(self.basis_labels) if self.basis_labels else None,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        kind = "pure" if self.vector is not None else "mixed"
        return f"CognitiveState(dim={self.dim}, {kind}, label={self.label!r})"


def _round_complex_list(values, digits: int = 6) -> list:
    out = []
    for value in np.asarray(values).ravel():
        real = round(float(np.real(value)), digits)
        imag = round(float(np.imag(value)), digits)
        out.append([real, imag])
    return out


# --------------------------------------------------------------------- builders
def basis_state(index: int, dim: int, label: str | None = None, basis_labels=None) -> CognitiveState:
    """Computational basis state ``|index>``."""
    if not 0 <= index < dim:
        raise ValueError(f"index {index} outside 0..{dim - 1}")
    vector = np.zeros(dim, dtype=complex)
    vector[index] = 1.0
    return CognitiveState.from_vector(vector, label=label or f"|{index}>", basis_labels=basis_labels)


def uniform_state(dim: int, label: str = "uniform", basis_labels=None) -> CognitiveState:
    """Equal superposition over all basis states."""
    return CognitiveState.from_vector(np.ones(dim, dtype=complex), label=label, basis_labels=basis_labels)


def amplitudes(*values, label: str = "psi", basis_labels=None) -> CognitiveState:
    """Build a pure state from raw (unnormalised) amplitudes."""
    return CognitiveState.from_vector(np.array(values, dtype=complex), label=label, basis_labels=basis_labels)


def from_bloch(theta: float, phi: float = 0.0, label: str = "qubit") -> CognitiveState:
    """Qubit state with Bloch angles ``theta`` (from ``|0>``) and ``phi``."""
    vector = np.array(
        [np.cos(theta / 2.0), np.exp(1j * phi) * np.sin(theta / 2.0)], dtype=complex
    )
    return CognitiveState.from_vector(vector, label=label)


def question_state(theta: float, phi: float = 0.0, label: str = "belief") -> CognitiveState:
    """Alias of :func:`from_bloch` phrased in cognitive language."""
    return from_bloch(theta, phi, label=label)


def tensor(*states: CognitiveState) -> CognitiveState:
    """Kronecker product of states (a composite cognitive system).

    Follows numpy's Kronecker convention: the *first* argument becomes the most
    significant factor, i.e. the highest-numbered qubit in the little-endian
    convention used by Qiskit.  ``tensor(a, b)`` therefore puts ``a`` on qubit 1 and
    ``b`` on qubit 0.
    """
    if not states:
        raise ValueError("tensor() needs at least one state")
    density = np.array([[1.0 + 0.0j]])
    vector: np.ndarray | None = np.array([1.0 + 0.0j])
    labels: list[str] = []
    complete = all(state.basis_labels for state in states)
    for state in states:
        density = np.kron(density, state.density)
        if vector is not None and state.vector is not None:
            vector = np.kron(vector, state.vector)
        else:
            vector = None
        if complete:
            if not labels:
                labels = list(state.basis_labels or ())
            else:
                labels = [
                    f"{left} and {right}"
                    for left in labels
                    for right in (state.basis_labels or ())
                ]
    return CognitiveState.from_density(
        density,
        label=" x ".join(state.label for state in states),
        vector=vector,
        basis_labels=tuple(labels) if labels else None,
    )


def bell_state(kind: str = "phi+", label: str | None = None) -> CognitiveState:
    """One of the four Bell states on two qubits."""
    table = {
        "phi+": np.array([1, 0, 0, 1]),
        "phi-": np.array([1, 0, 0, -1]),
        "psi+": np.array([0, 1, 1, 0]),
        "psi-": np.array([0, 1, -1, 0]),
    }
    if kind not in table:
        raise ValueError(f"unknown Bell state {kind!r}; expected one of {sorted(table)}")
    return CognitiveState.from_vector(table[kind], label=label or f"|{kind}>")


def mixed_state(components: Iterable[tuple[float, CognitiveState]], label: str = "mixture") -> CognitiveState:
    """Convex mixture ``sum_i p_i rho_i`` (weights are renormalised)."""
    pairs = list(components)
    if not pairs:
        raise ValueError("mixture needs at least one component")
    weights = np.array([weight for weight, _ in pairs], dtype=float)
    total = weights.sum()
    if total <= 0:
        raise ValueError("mixture weights must be positive")
    weights = weights / total
    density = np.zeros_like(pairs[0][1].density)
    for weight, state in pairs:
        density += weight * state.density
    return CognitiveState.from_density(density, label=label)


def maximally_mixed(dim: int, label: str = "I/d") -> CognitiveState:
    """Maximally mixed state ``I / d`` (complete decoherence)."""
    return CognitiveState.from_density(np.eye(dim, dtype=complex) / dim, label=label)


def as_state(obj, label: str = "psi", basis_labels=None) -> CognitiveState:
    """Coerce vectors, Qiskit states or Q-Cog states into a :class:`CognitiveState`."""
    if isinstance(obj, CognitiveState):
        return obj
    if hasattr(obj, "data") and obj.__class__.__name__ in {"DensityMatrix", "Statevector"}:
        data = np.asarray(obj.data)
        if data.ndim == 1:
            return CognitiveState.from_vector(data, label=label, basis_labels=basis_labels)
        return CognitiveState.from_density(data, label=label, basis_labels=basis_labels)
    array = np.asarray(obj)
    if array.ndim == 1:
        return CognitiveState.from_vector(array, label=label, basis_labels=basis_labels)
    if array.ndim == 2:
        return CognitiveState.from_density(array, label=label, basis_labels=basis_labels)
    raise TypeError(f"cannot interpret {type(obj)!r} as a cognitive state")