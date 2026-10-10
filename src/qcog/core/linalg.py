"""Linear-algebra helpers shared by every Q-Cog component.

The module deliberately keeps the numerical layer small and explicit so that the
rest of the library can treat *projectors* (not circuits) as the single source of
truth.  Circuits are derived from projectors, never the other way round, which is
what allows the exact statevector path and the circuit path to be cross-checked
against each other (see ``tests/test_backend_parity.py``).
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

#: Numerical tolerance used for all structural checks (orthogonality, unitarity...).
TOL = 1e-9

__all__ = [
    "TOL",
    "as_complex_array",
    "normalize",
    "next_power_of_two",
    "num_qubits_for_dim",
    "embed_vector",
    "embed_matrix",
    "is_hermitian",
    "is_projector",
    "is_orthogonal_family",
    "is_complete_family",
    "check_unitary",
    "pvm_to_unitary",
    "matrix_sqrt_psd",
    "commutator",
    "anticommutator",
    "commutator_norm",
    "dephasing_projectors",
    "gell_mann_basis",
    "trivial_qubits",
    "acts_trivially_on",
    "restrict_operators",
    "trace_distance",
    "fidelity",
    "purity",
    "von_neumann_entropy",
]


def as_complex_array(value, ndim: int | None = None) -> np.ndarray:
    """Return ``value`` as a complex ``numpy`` array (no copy when possible)."""
    array = np.asarray(value, dtype=complex)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"expected a {ndim}-dimensional array, got shape {array.shape}")
    return array


def normalize(vector) -> np.ndarray:
    """Normalise a (possibly unnormalised) vector to unit length."""
    vec = as_complex_array(vector, 1)
    norm = np.linalg.norm(vec)
    if norm <= TOL:
        raise ValueError("cannot normalise the zero vector")
    return vec / norm


def next_power_of_two(value: int) -> int:
    """Smallest power of two that is greater than or equal to ``value``."""
    if value < 1:
        raise ValueError("value must be a positive integer")
    return 1 << (int(value) - 1).bit_length()


def num_qubits_for_dim(dim: int) -> int:
    """Number of qubits required to embed a ``dim``-dimensional system."""
    return next_power_of_two(dim).bit_length() - 1


def embed_vector(vector, qubit_dim: int | None = None) -> np.ndarray:
    """Zero-pad a state vector so that its dimension is a power of two."""
    vec = as_complex_array(vector, 1)
    target = qubit_dim or next_power_of_two(vec.size)
    if target < vec.size:
        raise ValueError(f"target dimension {target} is smaller than the state dimension {vec.size}")
    out = np.zeros(target, dtype=complex)
    out[: vec.size] = vec
    return out


def embed_matrix(matrix, qubit_dim: int | None = None) -> np.ndarray:
    """Block-diagonally embed a matrix with an identity on the padding subspace.

    The padding subspace is a *spectator*: it carries no population because
    :func:`embed_vector` puts zeros there, so this embedding never changes the
    predicted probabilities.
    """
    mat = as_complex_array(matrix, 2)
    if mat.shape[0] != mat.shape[1]:
        raise ValueError("matrix must be square")
    target = qubit_dim or next_power_of_two(mat.shape[0])
    if target < mat.shape[0]:
        raise ValueError("target dimension smaller than matrix dimension")
    out = np.eye(target, dtype=complex)
    out[: mat.shape[0], : mat.shape[0]] = mat
    return out


def is_hermitian(matrix, tol: float = TOL) -> bool:
    mat = as_complex_array(matrix, 2)
    return bool(np.allclose(mat, mat.conj().T, atol=tol))


def is_projector(matrix, tol: float = TOL) -> bool:
    """``P`` is a projector iff it is Hermitian and idempotent."""
    mat = as_complex_array(matrix, 2)
    if not is_hermitian(mat, tol):
        return False
    return bool(np.allclose(mat @ mat, mat, atol=tol))


def is_orthogonal_family(projectors: Sequence[np.ndarray], tol: float = TOL) -> bool:
    """``P_i P_j = 0`` for ``i != j``."""
    for i, first in enumerate(projectors):
        for j, second in enumerate(projectors):
            if i == j:
                continue
            if not np.allclose(first @ second, 0.0, atol=tol):
                return False
    return True


def is_complete_family(projectors: Sequence[np.ndarray], tol: float = TOL) -> bool:
    """``sum_i P_i = I``."""
    if not projectors:
        return False
    dim = projectors[0].shape[0]
    total = np.zeros((dim, dim), dtype=complex)
    for projector in projectors:
        total += projector
    return bool(np.allclose(total, np.eye(dim), atol=tol))


def check_unitary(matrix, tol: float = TOL, name: str = "matrix") -> np.ndarray:
    mat = as_complex_array(matrix, 2)
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"{name} must be square")
    if not np.allclose(mat.conj().T @ mat, np.eye(mat.shape[0]), atol=tol):
        raise ValueError(f"{name} is not unitary")
    return mat


def pvm_to_unitary(projectors: Sequence[np.ndarray], tol: float = TOL):
    """Return a unitary that rotates a projective measurement into the computational basis.

    Returns
    -------
    unitary, blocks
        ``unitary`` has columns grouped by outcome; ``blocks`` is a list of
        ``(outcome_index, start, stop)`` describing which computational basis
        indices belong to which outcome.  Because ``U`` maps ``P_i`` onto the
        diagonal block ``i``, measuring all qubits after ``U^dagger`` and
        decoding the integer outcome through ``blocks`` implements the PVM.
    """
    blocks: list[tuple[int, int, int]] = []
    columns: list[np.ndarray] = []
    start = 0
    for index, projector in enumerate(projectors):
        eigvals, eigvecs = np.linalg.eigh(as_complex_array(projector, 2))
        keep = np.abs(eigvals) > 1e-8
        basis = eigvecs[:, keep]
        if basis.shape[1] == 0:
            raise ValueError(f"projector {index} has empty support")
        columns.append(basis)
        stop = start + basis.shape[1]
        blocks.append((index, start, stop))
        start = stop
    unitary = np.hstack(columns)
    check_unitary(unitary, tol=tol, name="measurement basis change")
    return unitary, blocks


def matrix_sqrt_psd(matrix) -> np.ndarray:
    """Hermitian square root of a positive semi-definite matrix."""
    mat = as_complex_array(matrix, 2)
    eigvals, eigvecs = np.linalg.eigh(mat)
    if np.min(eigvals) < -1e-8:
        raise ValueError("matrix is not positive semi-definite")
    eigvals = np.clip(eigvals, 0.0, None)
    return (eigvecs * np.sqrt(eigvals)) @ eigvecs.conj().T


def commutator(first, second) -> np.ndarray:
    """``[A, B] = AB - BA``."""
    return as_complex_array(first, 2) @ as_complex_array(second, 2) - as_complex_array(
        second, 2
    ) @ as_complex_array(first, 2)


def anticommutator(first, second) -> np.ndarray:
    """``{A, B} = AB + BA``."""
    return as_complex_array(first, 2) @ as_complex_array(second, 2) + as_complex_array(
        second, 2
    ) @ as_complex_array(first, 2)


def commutator_norm(first, second) -> float:
    """Frobenius norm of the commutator of two observables.

    Used as a *unit-free* measure of how "incompatible" two questions are: if it
    is zero the questions share an eigenbasis, the model becomes classical, and
    order effects must vanish exactly.
    """
    return float(np.linalg.norm(commutator(first, second)))


def dephasing_projectors(projectors: Sequence[np.ndarray], state) -> np.ndarray:
    """Non-selective measurement: ``rho -> sum_i P_i rho P_i``."""
    rho = as_complex_array(state, 2)
    out = np.zeros_like(rho)
    for projector in projectors:
        out += projector @ rho @ projector.conj().T
    return out


def trace_distance(first, second) -> float:
    """Trace distance ``0.5 * ||rho - sigma||_1``."""
    from qiskit.quantum_info import state_fidelity  # noqa: F401  (documented dependency)

    diff = as_complex_array(first, 2) - as_complex_array(second, 2)
    eigenvalues = np.linalg.eigvalsh((diff + diff.conj().T) / 2.0)
    return float(0.5 * np.sum(np.abs(eigenvalues)))


def fidelity(first, second) -> float:
    """Uhlmann fidelity between two density matrices."""
    from qiskit.quantum_info import state_fidelity

    return float(np.real(state_fidelity(first, second)))


def purity(state) -> float:
    """``Tr(rho^2)``; 1 for a pure state."""
    rho = as_complex_array(state, 2)
    return float(np.real(np.trace(rho @ rho)))


def von_neumann_entropy(state) -> float:
    """Von Neumann entropy in bits."""
    rho = as_complex_array(state, 2)
    eigenvalues = np.linalg.eigvalsh((rho + rho.conj().T) / 2.0)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    if eigenvalues.size == 0:
        return 0.0
    return float(-np.sum(eigenvalues * np.log2(eigenvalues)))


def gell_mann_basis(dim: int) -> list[np.ndarray]:
    """Hilbert-Schmidt-orthonormal basis of traceless Hermitian ``dim x dim`` matrices.

    Normalised so that ``Tr(B_j B_k) = 2 delta_jk`` and, usefully,
    ``sum_k B_k^2 = 2 (dim^2 - 1) / dim * I``.  This is the generalised
    Gell-Mann basis used to build a *trace-preserving* depolarising channel in
    arbitrary dimension (see :func:`qcog.core.instruments.depolarizing_channel`).
    """
    if dim < 2:
        raise ValueError("dim must be at least 2")
    basis: list[np.ndarray] = []
    for j in range(dim):
        for k in range(j + 1, dim):
            symmetric = np.zeros((dim, dim), dtype=complex)
            symmetric[j, k] = 1.0
            symmetric[k, j] = 1.0
            basis.append(symmetric)
            antisymmetric = np.zeros((dim, dim), dtype=complex)
            antisymmetric[j, k] = -1j
            antisymmetric[k, j] = 1j
            basis.append(antisymmetric)
    for l in range(1, dim):
        diagonal = np.zeros(dim, dtype=complex)
        diagonal[:l] = 1.0
        diagonal[l] = -float(l)
        diagonal *= np.sqrt(2.0 / (l * (l + 1)))
        basis.append(np.diag(diagonal))
    assert len(basis) == dim * dim - 1
    return basis


def acts_trivially_on(matrix, qubit: int, num_qubits: int, tol: float = 1e-9) -> bool:
    """True when ``matrix`` equals ``A x I`` on ``qubit`` for some ``A``.

    The test is structural: reshape the operator so that the row and column indices
    are ``(higher qubits, this qubit, lower qubits)`` and require that it is
    diagonal and independent of the value of this qubit.  An operator that passes
    says nothing about this qubit's value, so the qubit does not have to be
    measured for the operator's outcome to be determined.
    """
    mat = as_complex_array(matrix, 2)
    size = 2**num_qubits
    if mat.shape != (size, size):
        raise ValueError(f"matrix of shape {mat.shape} does not match {num_qubits} qubits")
    lower = 2**qubit
    higher = 2 ** (num_qubits - qubit - 1)
    tensor = mat.reshape(higher, 2, lower, higher, 2, lower)
    # off-diagonal in this qubit must vanish
    off = tensor[:, 0, :, :, 1, :]
    if np.max(np.abs(off)) > tol:
        return False
    same = np.max(np.abs(tensor[:, 0, :, :, 0, :] - tensor[:, 1, :, :, 1, :]))
    return bool(same <= tol)


def trivial_qubits(projectors: Sequence[np.ndarray], num_qubits: int, tol: float = 1e-9) -> set[int]:
    """Qubits on which a whole family of operators acts as the identity.

    A question that only concerns one subsystem must not force a measurement of the
    others: measuring an uninvolved qubit would collapse amplitudes the question does
    not care about, which silently destroys coherence and changes predictions.
    """
    trivial: set[int] = set()
    for qubit in range(num_qubits):
        if all(acts_trivially_on(projector, qubit, num_qubits, tol) for projector in projectors):
            trivial.add(qubit)
    return trivial


def restrict_operators(operators: Sequence[np.ndarray], support: Sequence[int], num_qubits: int) -> list[np.ndarray]:
    """Restrict operators that act trivially on the other qubits.

    If an operator equals ``A x I`` on the qubits outside ``support``, then ``A`` is
    recovered by reading the entries whose indices have all other qubits set to zero.
    The result acts on ``len(support)`` qubits, with the support qubits in ascending
    order (the same order the circuit uses), which is what allows a question to be
    compiled onto just the qubits it concerns.
    """
    qubits = sorted(support)
    if not qubits:
        raise ValueError("the support must not be empty")
    size = 2 ** len(qubits)
    indices = []
    for local in range(size):
        full = 0
        for position, qubit in enumerate(qubits):
            if (local >> position) & 1:
                full |= 1 << qubit
        indices.append(full)
    restricted = []
    for operator in operators:
        matrix = as_complex_array(operator, 2)
        if matrix.shape != (2**num_qubits, 2**num_qubits):
            raise ValueError("operator does not match the register size")
        small = np.zeros((size, size), dtype=complex)
        for row in range(size):
            for column in range(size):
                small[row, column] = matrix[indices[row], indices[column]]
        restricted.append(small)
    return restricted
