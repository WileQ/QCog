"""Small gate helpers used by the state layer and the circuit builder.

These helpers keep a single, well-tested place where "a unitary in a cognitive
model" is defined, and where user input (matrix, Qiskit gate, callable) is
validated.
"""

from __future__ import annotations

import numpy as np

from ..core.linalg import TOL, as_complex_array, check_unitary

__all__ = [
    "as_unitary",
    "rotation_y",
    "rotation_z",
    "rotation_x",
    "phase_gate",
    "hadamard",
    "block_diagonal",
    "rotation_matrix",
]


def as_unitary(value, dim: int | None = None) -> np.ndarray:
    """Coerce ``value`` into a validated ``(d, d)`` unitary matrix."""
    from qiskit.quantum_info import Operator

    if isinstance(value, Operator):
        matrix = np.asarray(value.data)
    elif hasattr(value, "to_matrix"):
        matrix = np.asarray(value.to_matrix())
    else:
        matrix = as_complex_array(value, 2)
    if dim is not None and matrix.shape[0] != dim:
        raise ValueError(f"unitary has dimension {matrix.shape[0]}, expected {dim}")
    return check_unitary(matrix, tol=1e-8, name="unitary")


def rotation_y(theta: float) -> np.ndarray:
    """Rotation about the ``y`` axis of the Bloch sphere."""
    half = theta / 2.0
    return np.array(
        [[np.cos(half), -np.sin(half)], [np.sin(half), np.cos(half)]], dtype=complex
    )


def rotation_x(theta: float) -> np.ndarray:
    half = theta / 2.0
    return np.array(
        [[np.cos(half), -1j * np.sin(half)], [-1j * np.sin(half), np.cos(half)]], dtype=complex
    )


def rotation_z(phi: float) -> np.ndarray:
    return np.array([[np.exp(-1j * phi / 2), 0], [0, np.exp(1j * phi / 2)]], dtype=complex)


def phase_gate(phi: float) -> np.ndarray:
    """Diagonal phase gate ``diag(1, e^{i phi})`` (no global phase)."""
    return np.array([[1.0, 0.0], [0.0, np.exp(1j * phi)]], dtype=complex)


def hadamard() -> np.ndarray:
    return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def block_diagonal(blocks) -> np.ndarray:
    """Block-diagonal matrix from a sequence of square blocks."""
    matrices = [as_complex_array(block, 2) for block in blocks]
    size = sum(block.shape[0] for block in matrices)
    out = np.zeros((size, size), dtype=complex)
    offset = 0
    for block in matrices:
        width = block.shape[0]
        out[offset : offset + width, offset : offset + width] = block
        offset += width
    return out


def rotation_matrix(dim: int, i: int, j: int, angle: float) -> np.ndarray:
    """Real Givens rotation mixing basis states ``i`` and ``j`` in a ``dim`` space.

    This is the workhorse of the two-dimensional "belief rotation" models: it is
    a genuine unitary, so the same object can be run on a simulator, and it is
    easy to read as "the context turned the belief by ``angle``".
    """
    if not 0 <= i < dim or not 0 <= j < dim:
        raise ValueError("indices outside the Hilbert space")
    if i == j:
        raise ValueError("i and j must differ")
    matrix = np.eye(dim, dtype=complex)
    cos, sin = np.cos(angle), np.sin(angle)
    matrix[i, i] = cos
    matrix[i, j] = -sin
    matrix[j, i] = sin
    matrix[j, j] = cos
    return matrix