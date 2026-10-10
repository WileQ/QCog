"""Quantum channels: non-selective measurement, noise and decoherence.

Two ideas are central to the scientific story of Q-Cog, and both live here:

*Non-selective measurement* -- if a question is asked but the answer is not
recorded, the state is dephased in the question's eigenbasis.  This is the
mechanism behind question-order effects, and it is also the reason those effects
disappear as soon as a "which question came first" record exists.  Q-Cog exposes
it explicitly as :func:`nonselective_channel`.

*Decoherence* -- depolarising, phase-damping and amplitude-damping channels let a
model answer the question "would this prediction survive a noisy device?".  The
answer is often informative: order effects shrink smoothly with the decoherence
strength, which is a testable statement rather than a slogan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

from .linalg import TOL, as_complex_array, check_unitary, gell_mann_basis
from .measurement import Measurement, POVM, ProjectiveMeasurement
from .states import CognitiveState, as_state

__all__ = [
    "Channel",
    "NoiseSpec",
    "NoiseModel",
    "kraus_channel",
    "unitary_channel",
    "identity_channel",
    "depolarizing_channel",
    "dephasing_channel",
    "nonselective_channel",
    "partial_dephasing_channel",
    "phase_damping_channel",
    "amplitude_damping_channel",
    "bit_flip_channel",
    "compose_channels",
    "per_qubit_channel",
    "tensor_channels",
    "apply_channels",
]


@dataclass(frozen=True)
class Channel:
    """A completely positive trace-preserving map given by Kraus operators."""

    name: str
    kraus: tuple[np.ndarray, ...]
    description: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        operators = tuple(as_complex_array(operator, 2) for operator in self.kraus)
        object.__setattr__(self, "kraus", operators)
        if not operators:
            raise ValueError("a channel needs at least one Kraus operator")
        dim = operators[0].shape[0]
        for operator in operators:
            if operator.shape != (dim, dim):
                raise ValueError("all Kraus operators must be square and equally sized")
        # Trace preservation: sum_i K_i^dagger K_i = I
        total = np.zeros((dim, dim), dtype=complex)
        for operator in operators:
            total += operator.conj().T @ operator
        deviation = float(np.max(np.abs(total - np.eye(dim))))
        if deviation > 1e-7:
            raise ValueError(f"Kraus operators are not trace preserving (max deviation {deviation:.2e})")

    @property
    def dim(self) -> int:
        return int(self.kraus[0].shape[0])

    @property
    def is_unitary(self) -> bool:
        """True when the channel is a single unitary (no decoherence)."""
        return len(self.kraus) == 1

    def apply(self, state) -> CognitiveState:
        """Apply the channel to a state."""
        confidence = as_state(state)
        return confidence.apply_channel(list(self.kraus))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dim": self.dim,
            "num_kraus": len(self.kraus),
            "is_unitary": self.is_unitary,
            "description": self.description,
            **({"metadata": self.metadata} if self.metadata else {}),
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Channel({self.name!r}, dim={self.dim}, kraus={len(self.kraus)})"


@dataclass
class NoiseModel:
    """Where decoherence enters a simulated conversation.

    ``apply_to``
        ``"each_question"`` (default) applies the channels after every question,
        ``"before_first"`` only before the first question, and
        ``"after_last"`` only after the last one.
    """

    channels: tuple[Channel, ...] = ()
    apply_to: str = "each_question"
    name: str = "noiseless"

    def __post_init__(self) -> None:
        self.channels = tuple(self.channels)
        if self.apply_to not in {"each_question", "before_first", "after_last", "never"}:
            raise ValueError(f"unknown apply_to={self.apply_to!r}")

    @property
    def is_trivial(self) -> bool:
        """True when the model contains no decoherence at all."""
        return not self.channels or self.apply_to == "never"

    def apply(self, state, step: int | None = None, num_steps: int | None = None) -> CognitiveState:
        """Apply the model's channels to ``state`` at the given question index."""
        if self.is_trivial:
            return as_state(state)
        if self.apply_to == "each_question":
            out = as_state(state)
            for channel in self.channels:
                out = channel.apply(out)
            return out
        if self.apply_to == "before_first" and step == 0:
            out = as_state(state)
            for channel in self.channels:
                out = channel.apply(out)
            return out
        if self.apply_to == "after_last" and num_steps is not None and step == num_steps:
            out = as_state(state)
            for channel in self.channels:
                out = channel.apply(out)
            return out
        return as_state(state)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "apply_to": self.apply_to,
            "channels": [channel.to_dict() for channel in self.channels],
        }

    @classmethod
    def from_spec(cls, spec: "NoiseSpec | None") -> "NoiseModel":
        """Build a noise model from a compact :class:`NoiseSpec` description."""
        if spec is None:
            return cls()
        if isinstance(spec, NoiseModel):
            return spec
        return cls(spec.channels(), spec.apply_to, spec.name)


# ------------------------------------------------------------------- builders
def kraus_channel(name: str, operators: Sequence[np.ndarray], description: str = "", **metadata) -> Channel:
    """Build a channel from explicit Kraus operators."""
    return Channel(name, tuple(np.asarray(op, dtype=complex) for op in operators), description, metadata)


def unitary_channel(name: str, unitary, description: str = "") -> Channel:
    """Build a unitary (noiseless) channel."""
    return Channel(name, (check_unitary(unitary, name=name),), description, {"unitary": True})


def identity_channel(dim: int = 2, name: str = "identity") -> Channel:
    """The no-op channel."""
    return Channel(name, (np.eye(dim, dtype=complex),), "identity")


def depolarizing_channel(p: float, dim: int = 2, name: str | None = None) -> Channel:
    """Depolarising channel ``rho -> (1 - p) rho + p I/d``.

    ``p = 1`` means the state is completely scrambled, which is the natural
    "the participant stopped using the model" limit.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0, 1]")
    eye = np.eye(dim, dtype=complex)
    # Kraus form of rho -> (1 - p) rho + p Tr(rho) I/d.  The identity term is not
    # sqrt(1 - p): the traceless part of the basis contributes p * rho / d^2 as well,
    # which is exactly what the tensor product of two half-boundary magnitudes must
    # cancel when d^2 half-boundary magnitudes meet at the axis.
    operators = [np.sqrt((1.0 - p) + p / (dim * dim)) * eye]
    # Remaining Kraus operators: a Hilbert-Schmidt-orthonormal traceless Hermitian
    # basis with Tr(B_j B_k) = 2 delta_jk, scaled so that sum_k K_k^dagger K_k = p * I.
    scale = p / (2.0 * dim)
    for basis in gell_mann_basis(dim):
        operators.append(np.sqrt(scale) * basis)
    return Channel(
        name or f"depolarizing(p={p:.3f})",
        tuple(operators),
        f"depolarising with p={p:.3f}",
        {"p": float(p), "dim": dim},
    )


def dephasing_channel(projectors: Sequence[np.ndarray], name: str | None = None) -> Channel:
    """Dephase in a chosen basis: ``rho -> sum_i P_i rho P_i``."""
    operators = tuple(as_complex_array(projector, 2) for projector in projectors)
    return Channel(
        name or "dephasing",
        operators,
        "dephasing in the eigenbasis of a question",
    )


def nonselective_channel(measurement: Measurement, name: str | None = None) -> Channel:
    """A question that is asked but *not recorded*.

    This is the Q-Cog implementation of the "hidden measurement" step that
    produces order effects: the state loses coherence in the basis of the
    question, exactly as if the participant had answered it silently.
    """
    if isinstance(measurement, POVM):
        operators = measurement.measurement_operators()
    else:
        operators = measurement.projectors
    return Channel(
        name or f"nonselective({measurement.name})",
        tuple(operators),
        f"question {measurement.name!r} asked but its answer not recorded",
        {"question": measurement.name},
    )


def partial_dephasing_channel(
    projectors: Sequence[np.ndarray], visibility: float, name: str | None = None
) -> Channel:
    """Dephase in a basis *with probability* ``visibility``.

    ``visibility = 1`` dephases completely (the "which path is known" case),
    ``visibility = 0`` leaves the state untouched (the "paths are combined" case).
    Everything in between is a smooth dial between the two-slit experiment and
    its classical version, and it is how Q-Cog asks how much distinguishability a
    phenomenon can tolerate.
    """
    if not 0.0 <= visibility <= 1.0:
        raise ValueError("visibility must lie in [0, 1]")
    operators = [np.sqrt(visibility) * np.asarray(projector, dtype=complex) for projector in projectors]
    if visibility < 1.0:
        operators.append(np.sqrt(1.0 - visibility) * np.eye(int(np.asarray(projectors[0]).shape[0])))
    return Channel(
        name or f"partial-dephasing(visibility={visibility:.3f})",
        tuple(operators),
        f"which-path information available with visibility {visibility:.3f}",
        {"visibility": float(visibility)},
    )


def phase_damping_channel(gamma: float, name: str | None = None) -> Channel:
    """Qubit phase damping with probability ``gamma`` (destroys superpositions, keeps populations)."""
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0, 1]")
    keep = np.sqrt(1.0 - gamma)
    return Channel(
        name or f"phase-damping(gamma={gamma:.3f})",
        (np.diag([1.0, keep]).astype(complex), np.diag([0.0, np.sqrt(gamma)]).astype(complex)),
        f"phase damping with gamma={gamma:.3f}",
        {"gamma": float(gamma)},
    )


def amplitude_damping_channel(gamma: float, name: str | None = None) -> Channel:
    """Qubit amplitude damping (relaxation toward ``|0>``) with probability ``gamma``."""
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0, 1]")
    first = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]], dtype=complex)
    second = np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=complex)
    return Channel(
        name or f"amplitude-damping(gamma={gamma:.3f})",
        (first, second),
        f"amplitude damping with gamma={gamma:.3f}",
        {"gamma": float(gamma)},
    )


def bit_flip_channel(p: float, name: str | None = None) -> Channel:
    """Classical bit-flip with probability ``p``."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0, 1]")
    return Channel(
        name or f"bit-flip(p={p:.3f})",
        (np.sqrt(1.0 - p) * np.eye(2), np.sqrt(p) * np.array([[0, 1], [1, 0]])),
        f"bit flip with p={p:.3f}",
        {"p": float(p)},
    )


def compose_channels(first: Channel, second: Channel, name: str | None = None) -> Channel:
    """Channel composition: apply ``first`` and then ``second``."""
    if first.dim != second.dim:
        raise ValueError("cannot compose channels of different dimension")
    operators = []
    for right in second.kraus:
        for left in first.kraus:
            operators.append(right @ left)
    return Channel(
        name or f"{second.name} o {first.name}",
        tuple(operators),
        f"{second.description} after {first.description}".strip(),
    )


def tensor_channels(first: Channel, second: Channel, name: str | None = None) -> Channel:
    """Act on two subsystems independently."""
    operators = [np.kron(left, right) for left in first.kraus for right in second.kraus]
    return Channel(name or f"{first.name} x {second.name}", tuple(operators), "tensor product channel")


def per_qubit_channel(factory: Callable[..., Channel], num_qubits: int, name: str | None = None, **kwargs) -> Channel:
    """Apply the same single-qubit channel independently to every qubit."""
    single = factory(**kwargs)
    operators = [np.array([[1.0 + 0.0j]])]
    for _ in range(num_qubits):
        operators = [np.kron(left, right) for left in operators for right in single.kraus]
    return Channel(
        name or f"{single.name}^x{num_qubits}",
        tuple(operators),
        f"{single.description} on each of {num_qubits} qubits",
        dict(single.metadata),
    )


def apply_channels(state, channels: Iterable[Channel]) -> CognitiveState:
    """Apply a sequence of channels to a state."""
    out = as_state(state)
    for channel in channels:
        out = channel.apply(out)
    return out


@dataclass(frozen=True)
class NoiseSpec:
    """Compact, serialisable description of a decoherence hypothesis.

    Kept deliberately small so that it can appear in a model's parameter schema
    (and therefore in the UI, the CLI and the generated documentation).
    """

    kind: str = "depolarizing"
    strength: float = 0.05
    num_qubits: int = 1
    apply_to: str = "each_question"
    name: str = ""

    KINDS = ("depolarizing", "phase_damping", "amplitude_damping", "bit_flip")

    def channels(self) -> tuple[Channel, ...]:
        """Concrete Kraus channels for this specification."""
        if self.kind not in self.KINDS:
            raise ValueError(f"unknown noise kind {self.kind!r}; expected one of {self.KINDS}")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("noise strength must lie in [0, 1]")
        if self.kind == "depolarizing":
            return (per_qubit_channel(depolarizing_channel, self.num_qubits, p=self.strength),)
        if self.kind == "phase_damping":
            return (per_qubit_channel(phase_damping_channel, self.num_qubits, gamma=self.strength),)
        if self.kind == "amplitude_damping":
            return (per_qubit_channel(amplitude_damping_channel, self.num_qubits, gamma=self.strength),)
        return (per_qubit_channel(bit_flip_channel, self.num_qubits, p=self.strength),)

    def to_model(self) -> NoiseModel:
        """The executable :class:`NoiseModel` for this specification."""
        return NoiseModel(self.channels(), self.apply_to, self.name or f"{self.kind}={self.strength:.3f}")

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "strength": float(self.strength),
            "num_qubits": int(self.num_qubits),
            "apply_to": self.apply_to,
        }
