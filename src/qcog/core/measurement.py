"""Questions as measurements.

In Q-Cog a *question* is a quantum measurement.  The library supports two kinds:

``ProjectiveMeasurement``
    A complete family of orthogonal projectors, ``sum_i P_i = I``.  These are the
    "sharp" questions used for the standard quantum-cognition models, and they
    can be compiled into circuits exactly.

``POVM``
    A complete family of positive operators, ``sum_i E_i = I``.  These model
    *unsharp* questions: ``E_yes = (I + lambda * n.sigma) / 2`` interpolates
    continuously between "the question carries no information" (``lambda = 0``,
    a classical non-question) and a fully projective question (``lambda = 1``).
    This dial is one of the most useful tools in the library because it lets a
    user ask *how much* quantum structure a phenomenon actually needs.

Both kinds expose the same interface: ``probabilities``, ``post_state`` and
``to_dict``, so models, backends, figures and explanations can be written once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .linalg import (
    TOL,
    as_complex_array,
    commutator_norm,
    is_complete_family,
    is_hermitian,
    is_orthogonal_family,
    is_projector,
    matrix_sqrt_psd,
    normalize,
    pvm_to_unitary,
)
from .states import CognitiveState, as_state

__all__ = [
    "ProjectiveMeasurement",
    "POVM",
    "Measurement",
    "as_measurement",
    "observable_measurement",
    "ray_measurement",
    "dichotomous",
    "question",
    "computational",
    "x_question",
    "y_question",
    "z_question",
    "unsharp_question",
    "depolarized_question",
    "incompatibility",
    "measurements_commute",
]

_CACHE: dict[tuple, tuple[np.ndarray, list[tuple[int, int, int]]]] = {}


def _state_or_vector(state) -> CognitiveState:
    return as_state(state, label="psi")


def _key(projectors: Sequence[np.ndarray]) -> tuple:
    return tuple(np.asarray(projector).round(12).tobytes() for projector in projectors)


@dataclass(frozen=True, eq=False)
class ProjectiveMeasurement:
    """A sharp question: orthogonal projectors summing to the identity.

    Parameters
    ----------
    name:
        Short identifier used as the column name in result tables (``"A"``, ``"Q1"``...).
    outcomes:
        Human-readable outcome labels in the same order as ``projectors``.
    projectors:
        Complete orthogonal family of Hermitian projectors.
    description:
        One-line description shown in the UI and generated documentation.
    """

    name: str
    outcomes: tuple[str, ...]
    projectors: tuple[np.ndarray, ...]
    description: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        projectors = tuple(as_complex_array(projector, 2) for projector in self.projectors)
        object.__setattr__(self, "projectors", projectors)
        object.__setattr__(self, "outcomes", tuple(str(outcome) for outcome in self.outcomes))
        if len(projectors) != len(self.outcomes):
            raise ValueError("every projector needs exactly one outcome label")
        if not projectors:
            raise ValueError("a measurement needs at least one outcome")
        dim = projectors[0].shape[0]
        for index, projector in enumerate(projectors):
            if projector.shape != (dim, dim):
                raise ValueError(f"projector {index} has shape {projector.shape}, expected {(dim, dim)}")
            if not is_projector(projector):
                raise ValueError(f"outcome {self.outcomes[index]!r} is not a projector")
        if not is_orthogonal_family(projectors):
            raise ValueError("projectors for different outcomes must be mutually orthogonal")
        if not is_complete_family(projectors):
            raise ValueError("projectors must sum to the identity (a complete question)")

    # -------------------------------------------------------------- properties
    @property
    def dim(self) -> int:
        """Dimension of the Hilbert space the question acts on."""
        return int(self.projectors[0].shape[0])

    @property
    def num_outcomes(self) -> int:
        return len(self.outcomes)

    @property
    def is_dichotomous(self) -> bool:
        """True for binary questions ("yes"/"no")."""
        return len(self.outcomes) == 2

    @property
    def ranks(self) -> tuple[int, ...]:
        return tuple(int(round(float(np.real(np.trace(projector))))) for projector in self.projectors)

    def observable(self, weights: Sequence[float] | None = None) -> np.ndarray:
        """Hermitian operator ``sum_i w_i P_i`` representing the question.

        With ``weights=None`` the outcome indices are used.  The operator is what
        the commutator-based incompatibility measures act on.
        """
        if weights is None:
            weights = list(range(1, len(self.projectors) + 1))
        if len(weights) != len(self.projectors):
            raise ValueError("need one weight per outcome")
        operator = np.zeros((self.dim, self.dim), dtype=complex)
        for weight, projector in zip(weights, self.projectors):
            operator += weight * projector
        return operator

    # ------------------------------------------------------------- evaluation
    def index(self, outcome: str) -> int:
        """Position of ``outcome`` in the outcome list."""
        try:
            return self.outcomes.index(str(outcome))
        except ValueError as error:  # pragma: no cover - defensive
            raise ValueError(f"{outcome!r} is not an outcome of question {self.name!r}") from error

    def probability(self, state, outcome: str) -> float:
        """Probability of ``outcome`` (Born rule), without collapsing the state."""
        index = self.index(outcome)
        confidence = _state_or_vector(state)
        value = float(np.real(np.trace(self.projectors[index] @ confidence.density)))
        return float(np.clip(value, 0.0, 1.0))

    def probabilities(self, state) -> np.ndarray:
        """Probability of every outcome, ordered like :attr:`outcomes`."""
        confidence = _state_or_vector(state)
        values = [
            float(np.real(np.trace(projector @ confidence.density)))
            for projector in self.projectors
        ]
        values = np.clip(np.array(values, dtype=float), 0.0, 1.0)
        total = values.sum()
        if abs(total - 1.0) > 1e-6 and total > 0:
            values = values / total
        return values

    def probabilities_dict(self, state) -> dict[str, float]:
        """Probabilities keyed by outcome label."""
        return {outcome: float(p) for outcome, p in zip(self.outcomes, self.probabilities(state))}

    def post_state(self, state, outcome: str) -> CognitiveState:
        """Lueders state after observing ``outcome`` (the projection postulate)."""
        index = self.index(outcome)
        confidence = _state_or_vector(state)
        projector = self.projectors[index]
        updated = projector @ confidence.density @ projector.conj().T
        trace = float(np.real(np.trace(updated)))
        if trace <= 1e-15:
            raise ValueError(
                f"outcome {outcome!r} of question {self.name!r} has zero probability for this state"
            )
        updated = updated / trace
        vector = None
        if confidence.vector is not None:
            candidate = projector @ confidence.vector
            norm = np.linalg.norm(candidate)
            if norm > TOL:
                vector = candidate / norm
        return CognitiveState.from_density(
            updated,
            label=f"{confidence.label}|{self.name}={outcome}",
            vector=vector,
            basis_labels=confidence.basis_labels,
        )

    # ---------------------------------------------------------------- circuits
    def basis_unitary(self):
        """``(U, blocks)`` realising this question as a computational-basis measurement.

        ``U^dagger`` rotates the state so that outcome ``i`` occupies the
        computational-basis indices in ``blocks[i]``.  The result is cached
        because it is recomputed for every simulated shot otherwise.
        """
        key = ("pvm", self.dim, _key(self.projectors))
        cached = _CACHE.get(key)
        if cached is None:
            cached = pvm_to_unitary(self.projectors)
            _CACHE[key] = cached
        return cached

    def decode(self, index: int) -> str:
        """Map a computational-basis index back to an outcome label."""
        _, blocks = self.basis_unitary()
        for outcome_index, start, stop in blocks:
            if start <= index < stop:
                return self.outcomes[outcome_index]
        raise ValueError(f"basis index {index} does not belong to any outcome")

    # ------------------------------------------------------------- comparison
    def incompatibility(self, other: "ProjectiveMeasurement") -> float:
        """How strongly this question fails to be compatible with ``other``.

        Defined as ``sum_ij || [P_i, Q_j] ||_F``.  It is zero exactly when the
        two questions share an eigenbasis, which is precisely the classical case
        where question order cannot matter.
        """
        if self.dim != other.dim:
            raise ValueError("questions act on different Hilbert spaces")
        total = 0.0
        for projector in self.projectors:
            for other_projector in other.projectors:
                total += commutator_norm(projector, other_projector)
        return float(total)

    def commutes_with(self, other: "ProjectiveMeasurement", tol: float = 1e-8) -> bool:
        """True when the two questions are jointly answerable (classical limit)."""
        return bool(self.incompatibility(other) < tol)

    # ---------------------------------------------------------- serialisation
    def to_dict(self) -> dict:
        """JSON-friendly description (projectors optional for readability)."""
        return {
            "kind": "projective",
            "name": self.name,
            "outcomes": list(self.outcomes),
            "dim": self.dim,
            "ranks": list(self.ranks),
            "description": self.description,
            "source": self.metadata.get("source", "model"),
        }

    def describe(self) -> str:
        ranks = ", ".join(f"{outcome} (rank {rank})" for outcome, rank in zip(self.outcomes, self.ranks))
        text = f"{self.name}: {len(self.outcomes)}-outcome question [{ranks}]"
        return f"{text} - {self.description}" if self.description else text

    def __eq__(self, other) -> bool:  # pragma: no cover - trivial
        if not isinstance(other, ProjectiveMeasurement):
            return NotImplemented
        return self.name == other.name and self.outcomes == other.outcomes and all(
            np.allclose(a, b, atol=1e-10) for a, b in zip(self.projectors, other.projectors)
        )

    def __hash__(self) -> int:
        return hash((self.name, self.outcomes, self.dim))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"ProjectiveMeasurement({self.name!r}, outcomes={list(self.outcomes)}, dim={self.dim})"


@dataclass(frozen=True, eq=False)
class POVM:
    """An unsharp question: effects ``E_i >= 0`` with ``sum_i E_i = I``."""

    name: str
    outcomes: tuple[str, ...]
    effects: tuple[np.ndarray, ...]
    description: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        effects = tuple(as_complex_array(effect, 2) for effect in self.effects)
        object.__setattr__(self, "effects", effects)
        object.__setattr__(self, "outcomes", tuple(str(outcome) for outcome in self.outcomes))
        if len(effects) != len(self.outcomes):
            raise ValueError("every effect needs exactly one outcome label")
        dim = effects[0].shape[0]
        for index, effect in enumerate(effects):
            if effect.shape != (dim, dim):
                raise ValueError(f"effect {index} has the wrong shape")
            if not is_hermitian(effect, 1e-8):
                raise ValueError(f"effect {self.outcomes[index]!r} is not Hermitian")
            if np.min(np.linalg.eigvalsh(effect)) < -1e-8:
                raise ValueError(f"effect {self.outcomes[index]!r} is not positive semi-definite")
        total = np.sum(np.stack(effects), axis=0)
        if not np.allclose(total, np.eye(dim), atol=1e-8):
            raise ValueError("effects must sum to the identity")

    @property
    def dim(self) -> int:
        return int(self.effects[0].shape[0])

    @property
    def num_outcomes(self) -> int:
        return len(self.outcomes)

    @property
    def is_projective(self) -> bool:
        """True when every effect is itself a projector (an unsharp POVM is not)."""
        return all(is_projector(effect) for effect in self.effects)

    @property
    def sharpness(self) -> float:
        """Bloch-length of the induced observable, ``1`` for a sharp qubit question."""
        return float(self.metadata.get("sharpness", 1.0 if self.is_projective else 0.0))

    def index(self, outcome: str) -> int:
        try:
            return self.outcomes.index(str(outcome))
        except ValueError as error:  # pragma: no cover - defensive
            raise ValueError(f"{outcome!r} is not an outcome of question {self.name!r}") from error

    def measurement_operators(self) -> tuple[np.ndarray, ...]:
        """Kraus operators ``M_i = sqrt(E_i)`` implementing the instrument."""
        return tuple(matrix_sqrt_psd(effect) for effect in self.effects)

    def probabilities(self, state) -> np.ndarray:
        confidence = _state_or_vector(state)
        values = [
            float(np.real(np.trace(effect @ confidence.density))) for effect in self.effects
        ]
        values = np.clip(np.array(values, dtype=float), 0.0, 1.0)
        total = values.sum()
        return values / total if total > 0 else values

    def probabilities_dict(self, state) -> dict[str, float]:
        return {outcome: float(p) for outcome, p in zip(self.outcomes, self.probabilities(state))}

    def probability(self, state, outcome: str) -> float:
        return float(self.probabilities(state)[self.index(outcome)])

    def post_state(self, state, outcome: str) -> CognitiveState:
        confidence = _state_or_vector(state)
        operator = self.measurement_operators()[self.index(outcome)]
        updated = operator @ confidence.density @ operator.conj().T
        trace = float(np.real(np.trace(updated)))
        if trace <= TOL:
            raise ValueError(f"outcome {outcome!r} of {self.name!r} has zero probability")
        return CognitiveState.from_density(
            updated / trace, label=f"{confidence.label}|{self.name}={outcome}"
        )

    def to_projective_limit(self, tol: float = 1e-8) -> ProjectiveMeasurement:
        """Return the underlying sharp question, if this POVM is projective."""
        if not self.is_projective:
            raise ValueError(f"{self.name!r} is unsharp and has no projective limit")
        return ProjectiveMeasurement(self.name, self.outcomes, self.effects, self.description, dict(self.metadata))

    def to_dict(self) -> dict:
        return {
            "kind": "povm",
            "name": self.name,
            "outcomes": list(self.outcomes),
            "dim": self.dim,
            "sharpness": round(self.sharpness, 4),
            "description": self.description,
            "source": self.metadata.get("source", "model"),
        }

    def describe(self) -> str:
        return (
            f"{self.name}: {len(self.outcomes)}-outcome unsharp question "
            f"(sharpness lambda={self.sharpness:.2f}) {self.description}"
        ).strip()

    def __hash__(self) -> int:
        return hash((self.name, self.outcomes, self.dim))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"POVM({self.name!r}, outcomes={list(self.outcomes)}, dim={self.dim}, sharpness={self.sharpness:.2f})"


#: Anything Q-Cog can present to a participant.
Measurement = ProjectiveMeasurement | POVM


def as_measurement(obj, name: str | None = None) -> Measurement:
    """Coerce matrices / Qiskit observables / measurements into a measurement."""
    if isinstance(obj, (ProjectiveMeasurement, POVM)):
        return obj
    if hasattr(obj, "to_matrix"):
        obj = np.asarray(obj.to_matrix())
    if hasattr(obj, "data"):
        obj = np.asarray(obj.data)
    array = as_complex_array(obj, 2)
    if is_projector(array):
        name = name or "observable"
        return ProjectiveMeasurement(name, ("yes", "no"), (array, np.eye(array.shape[0]) - array))
    if is_hermitian(array):
        return observable_measurement(name or "observable", array)
    raise TypeError("cannot interpret this object as a measurement")


# ------------------------------------------------------------------- builders
def observable_measurement(
    name: str,
    operator,
    outcomes: Sequence[str] | None = None,
    description: str = "",
    tol: float = 1e-8,
) -> ProjectiveMeasurement:
    """Turn a Hermitian observable into a sharp question via its spectral decomposition."""
    matrix = as_complex_array(operator, 2)
    if not is_hermitian(matrix, tol):
        raise ValueError("observable must be Hermitian")
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    groups: list[tuple[float, list[int]]] = []
    for index, value in enumerate(eigenvalues):
        if groups and abs(groups[-1][0] - value) < 1e-6:
            groups[-1][1].append(index)
        else:
            groups.append((value, [index]))
    groups.sort(key=lambda item: -item[0])
    projectors = []
    for _, indices in groups:
        basis = eigenvectors[:, indices]
        projectors.append(basis @ basis.conj().T)
    if outcomes is None:
        outcomes = [f"{value:.3g}" for value, _ in groups]
    return ProjectiveMeasurement(
        name, tuple(outcomes), tuple(projectors), description, {"source": "observable"}
    )


def ray_measurement(
    name: str,
    rays: Sequence[np.ndarray],
    outcomes: Sequence[str] | None = None,
    description: str = "",
) -> ProjectiveMeasurement:
    """Build a sharp question from an orthonormal family of rank-1 rays."""
    vectors = [normalize(np.asarray(ray, dtype=complex)) for ray in rays]
    projectors = [np.outer(vector, vector.conj()) for vector in vectors]
    if outcomes is None:
        outcomes = [f"o{index}" for index in range(len(vectors))]
    return ProjectiveMeasurement(
        name, tuple(outcomes), tuple(projectors), description, {"source": "rays"}
    )


def dichotomous(
    name: str,
    ray,
    outcomes: Sequence[str] = ("yes", "no"),
    description: str = "",
) -> ProjectiveMeasurement:
    """Binary question ``{P_ray, I - P_ray}``."""
    vector = normalize(np.asarray(ray, dtype=complex))
    projector = np.outer(vector, vector.conj())
    dim = vector.size
    return ProjectiveMeasurement(
        name,
        tuple(outcomes),
        (projector, np.eye(dim) - projector),
        description,
        {"source": "dichotomous"},
    )


def question(
    theta: float,
    phi: float = 0.0,
    name: str = "question",
    outcomes: Sequence[str] = ("yes", "no"),
    description: str = "",
) -> ProjectiveMeasurement:
    """Binary qubit question along the Bloch direction ``(theta, phi)``.

    ``theta = pi/2, phi = 0`` is the standard ``X`` question; ``theta = 0`` the
    ``Z`` question ("is the answer 0 or 1?").  Angles are the natural parameters
    for the fitted models in :mod:`qcog.models`.
    """
    yes = np.array(
        [np.cos(theta / 2.0), np.exp(1j * phi) * np.sin(theta / 2.0)], dtype=complex
    )
    no = np.array(
        [-np.exp(-1j * phi) * np.sin(theta / 2.0), np.cos(theta / 2.0)], dtype=complex
    )
    return ray_measurement(name, (yes, no), outcomes=outcomes, description=description)


def computational(name: str = "Z", outcomes: Sequence[str] = ("0", "1"), description: str = "") -> ProjectiveMeasurement:
    """The computational-basis question."""
    return ProjectiveMeasurement(
        name,
        tuple(outcomes),
        (np.diag([1, 0]).astype(complex), np.diag([0, 1]).astype(complex)),
        description or "computational-basis question",
        {"source": "computational"},
    )


def z_question(name: str = "Z", description: str = "") -> ProjectiveMeasurement:
    return computational(name=name, description=description or "Z question (states 0 / 1)")


def x_question(name: str = "X", description: str = "") -> ProjectiveMeasurement:
    return question(np.pi / 2, 0.0, name=name, description=description or "X question (+ / -)")


def y_question(name: str = "Y", description: str = "") -> ProjectiveMeasurement:
    return question(np.pi / 2, np.pi / 2, name=name, description=description or "Y question (+i / -i)")


def unsharp_question(
    theta: float,
    sharpness: float = 0.8,
    phi: float = 0.0,
    name: str = "question",
    outcomes: Sequence[str] = ("yes", "no"),
    description: str = "",
) -> POVM:
    """Unsharp binary question with controllable sharpness ``lambda in [0, 1]``.

    ``lambda = 1`` reproduces :func:`question`; ``lambda = 0`` gives ``E_i = I/2``,
    an uninformative question.  Interpolating between the two is how Q-Cog shows
    *how much* quantum coherence a phenomenon requires.
    """
    if not 0.0 <= sharpness <= 1.0:
        raise ValueError("sharpness must lie in [0, 1]")
    direction = np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ]
    )
    pauli = (
        direction[0] * np.array([[0, 1], [1, 0]])
        + direction[1] * np.array([[0, -1j], [1j, 0]])
        + direction[2] * np.array([[1, 0], [0, -1]])
    ).astype(complex)
    eye = np.eye(2, dtype=complex)
    yes = (eye + sharpness * pauli) / 2.0
    no = (eye - sharpness * pauli) / 2.0
    return POVM(
        name,
        tuple(outcomes),
        (yes, no),
        description or f"unsharp question (lambda={sharpness:.2f})",
        {"sharpness": float(sharpness), "theta": float(theta), "phi": float(phi), "source": "unsharp"},
    )


def depolarized_question(
    measurement: Measurement, visibility: float, name: str | None = None
) -> POVM:
    """Mix a question with the uninformative question: ``E -> v E + (1 - v) I/d``.

    ``visibility = 1`` keeps the question unchanged, ``visibility = 0`` removes
    it entirely.  This models a participant who answers partly at random.
    """
    if not 0.0 <= visibility <= 1.0:
        raise ValueError("visibility must lie in [0, 1]")
    dim = measurement.dim
    eye = np.eye(dim, dtype=complex)
    effects = []
    for effect in (
        measurement.effects if isinstance(measurement, POVM) else measurement.projectors
    ):
        effects.append(visibility * effect + (1.0 - visibility) * eye / dim)
    return POVM(
        name or measurement.name,
        measurement.outcomes,
        tuple(effects),
        f"{measurement.description} (visibility={visibility:.2f})".strip(),
        {"visibility": float(visibility), "source": "depolarized"},
    )


def measurements_commute(measurements: Iterable[Measurement], tol: float = 1e-8) -> bool:
    """True when every pair of questions in ``measurements`` is compatible."""
    items = list(measurements)
    for index, first in enumerate(items):
        for second in items[index + 1 :]:
            if _as_projectors(first) is None or _as_projectors(second) is None:
                return False
            if not _pairwise_commute(first, second, tol):
                return False
    return True


def incompatibility(measurements: Iterable[Measurement]) -> float:
    """Total incompatibility of a list of questions (0 in the classical limit)."""
    items = list(measurements)
    total = 0.0
    for index, first in enumerate(items):
        for second in items[index + 1 :]:
            total += _pairwise_incompatibility(first, second)
    return float(total)


def _as_projectors(measurement: Measurement):
    projectors = (
        measurement.projectors if isinstance(measurement, ProjectiveMeasurement) else measurement.effects
    )
    return projectors if all(is_projector(projector) for projector in projectors) else None


def _pairwise_incompatibility(first: Measurement, second: Measurement) -> float:
    first_ops = _as_projectors(first) or first.effects
    second_ops = _as_projectors(second) or second.effects
    total = 0.0
    for a in first_ops:
        for b in second_ops:
            total += commutator_norm(a, b)
    return float(total)


def _pairwise_commute(first: Measurement, second: Measurement, tol: float = 1e-8) -> bool:
    return _pairwise_incompatibility(first, second) < tol