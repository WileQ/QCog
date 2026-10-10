"""Two-player quantum games (the Eisert-Wilkert-Lewenstein scheme).

Both players act on their own qubit with a unitary strategy; an entangling gate
``J`` couples them before and after, and the final computational-basis
measurement decides the classical outcome whose payoffs are then read off.  With
maximal entanglement this turns the prisoner's dilemma into a game whose unique
equilibrium is mutual cooperation - a clean, checkable, and famous result, which
makes it a good teaching example and a good test.

The implementation builds the actual unitary of the EWL circuit, so the same
object produces a circuit for Aer or IonQ and an exact statevector payoff.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Mapping, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, Statevector

__all__ = [
    "QuantumStrategy",
    "QuantumGame",
    "entangling_unitary",
    "strategy_unitary",
    "CLASSICAL_C",
    "CLASSICAL_D",
    "QUANTUM_Q",
]


def entangling_unitary(gamma: float) -> np.ndarray:
    """``J(gamma) = cos(gamma/2) I + i sin(gamma/2) X x X``.

    ``gamma = 0`` decouples the players (the classical game); ``gamma = pi/2`` is
    the maximally entangled case used by Eisert et al.
    """
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    identity = np.eye(4, dtype=complex)
    return np.cos(gamma / 2.0) * identity + 1j * np.sin(gamma / 2.0) * np.kron(x, x)


def strategy_unitary(theta: float, phi: float = 0.0) -> np.ndarray:
    """The Eisert strategy family ``U(theta, phi)``.

    ``U(0, 0) = I`` is cooperate, ``U(pi, 0)`` is defect and ``U(0, pi/2)`` is the
    "quantum" strategy ``Q``, which is what makes cooperation the equilibrium.
    """
    half = theta / 2.0
    return np.array(
        [
            [np.exp(1j * phi) * np.cos(half), np.sin(half)],
            [-np.sin(half), np.exp(-1j * phi) * np.cos(half)],
        ],
        dtype=complex,
    )


#: The named strategies used throughout the library and the UI.
CLASSICAL_C = {"name": "C  cooperate", "short": "C", "theta": 0.0, "phi": 0.0, "kind": "classical"}
CLASSICAL_D = {"name": "D  defect", "short": "D", "theta": np.pi, "phi": 0.0, "kind": "classical"}
QUANTUM_Q = {"name": "Q  quantum", "short": "Q", "theta": 0.0, "phi": np.pi / 2, "kind": "quantum"}


@dataclass(frozen=True)
class QuantumStrategy:
    """A strategy available to a player."""

    name: str
    theta: float = 0.0
    phi: float = 0.0
    description: str = ""

    @property
    def unitary(self) -> np.ndarray:
        return strategy_unitary(self.theta, self.phi)

    def to_dict(self) -> dict:
        return {"name": self.name, "theta": float(self.theta), "phi": float(self.phi),
                "description": self.description}

    @classmethod
    def cooperate(cls) -> "QuantumStrategy":
        return cls("C", 0.0, 0.0, "cooperate (the classical 'C')")

    @classmethod
    def defect(cls) -> "QuantumStrategy":
        return cls("D", float(np.pi), 0.0, "defect (the classical 'D')")

    @classmethod
    def quantum(cls) -> "QuantumStrategy":
        return cls("Q", 0.0, float(np.pi / 2), "the quantum strategy Q = iZ")


NASH_STRATEGIES = (QuantumStrategy.cooperate(), QuantumStrategy.defect(), QuantumStrategy.quantum())


@dataclass
class QuantumGame:
    """A two-player, two-strategy quantum game in the EWL scheme.

    Parameters
    ----------
    payoff_matrix:
        ``(2, 2)`` payoff to the row player, rows = own move, columns = opponent's
        move, in the order ``(cooperate, defect)``.
    entanglement:
        ``gamma`` in ``J(gamma)``; ``pi/2`` is maximal entanglement.
    noise:
        Optional depolarising probability applied to both qubits after ``J``;
        used to show how robust the quantum advantage is.
    """

    payoff_matrix: np.ndarray
    entanglement: float = float(np.pi / 2)
    noise: float = 0.0
    name: str = "quantum game"

    def __post_init__(self) -> None:
        self.payoff_matrix = np.asarray(self.payoff_matrix, dtype=float)
        if self.payoff_matrix.shape != (2, 2):
            raise ValueError("payoff_matrix must be 2x2")
        if not 0.0 <= self.noise <= 1.0:
            raise ValueError("noise must be a probability in [0, 1]")

    # ------------------------------------------------------------------ physics
    @property
    def entangling(self) -> np.ndarray:
        return entangling_unitary(self.entanglement)

    def circuit(self, first: QuantumStrategy, second: QuantumStrategy) -> QuantumCircuit:
        """The EWL circuit for one strategy profile."""
        circuit = QuantumCircuit(2, 2, name=f"ewl_{first.name}_{second.name}")
        circuit.append(UnitaryGate(self.entangling, label="J"), [0, 1])
        circuit.append(UnitaryGate(np.kron(first.unitary, second.unitary), label="U_A x U_B"), [0, 1])
        circuit.append(UnitaryGate(self.entangling.conj().T, label="J dagger"), [0, 1])
        circuit.measure([0, 1], [0, 1])
        return circuit

    def final_state(self, first: QuantumStrategy, second: QuantumStrategy):
        """Final state of the EWL circuit, before measurement.

        Returns a ``Statevector`` when ``noise == 0`` and a ``DensityMatrix``
        otherwise, so the same code reports coherent and noisy payoffs.
        """
        from qiskit.quantum_info import DensityMatrix

        state = Statevector.from_label("00")
        state = state.evolve(Operator(self.entangling))
        state = state.evolve(Operator(np.kron(first.unitary, second.unitary)))
        state = state.evolve(Operator(self.entangling.conj().T))
        if self.noise <= 0:
            return state
        from ..core.instruments import depolarizing_channel

        channel = depolarizing_channel(self.noise, 4)
        vector = np.asarray(state.data, dtype=complex)
        rho = np.outer(vector, vector.conj())
        mixed = np.zeros((4, 4), dtype=complex)
        for operator in channel.kraus:
            mixed += np.asarray(operator) @ rho @ np.asarray(operator).conj().T
        return DensityMatrix(mixed)

    def outcome_distribution(self, first: QuantumStrategy, second: QuantumStrategy) -> dict[tuple[str, str], float]:
        """Joint distribution over the classical moves ``(own, opponent)`` at the end of the game.

        Qubit ordering is little-endian and the first player owns the *most*
        significant qubit, so the basis index of the move pair ``(own, opponent)`` is
        ``opponent_bit + 2 * own_bit``.  With this convention the unentangled game
        (gamma = 0) reproduces the classical payoff matrix in both directions, which
        is checked by ``qcog evidence``.
        """
        state = self.final_state(first, second)
        probabilities = np.real(
            state.probabilities() if hasattr(state, "probabilities") else np.diag(np.asarray(state.data))
        )
        distribution: dict[tuple[str, str], float] = {}
        for own_bit, own in enumerate(("C", "D")):
            for opponent_bit, opponent in enumerate(("C", "D")):
                distribution[(own, opponent)] = float(probabilities[opponent_bit + 2 * own_bit])
        return distribution

    def expected_payoffs(self, first: QuantumStrategy, second: QuantumStrategy) -> dict:
        """Expected payoff of both players plus the joint move distribution."""
        distribution = self.outcome_distribution(first, second)
        payoff_first = 0.0
        payoff_second = 0.0
        for (own, opponent), probability in distribution.items():
            row = 0 if own == "C" else 1
            column = 0 if opponent == "C" else 1
            payoff_first += probability * self.payoff_matrix[row, column]
            payoff_second += probability * self.payoff_matrix[column, row]
        return {
            "first": float(payoff_first),
            "second": float(payoff_second),
            "moves": {f"{own}-{opponent}": round(value, 8) for (own, opponent), value in distribution.items()},
            "cooperation": float(sum(value for (own, opponent), value in distribution.items() if own == "C")),
        }

    # --------------------------------------------------------------- comparisons
    def classical_payoffs(self, first_move: str, second_move: str) -> tuple[float, float]:
        """Payoffs in the classical game (no entanglement)."""
        row = 0 if first_move.upper() == "C" else 1
        column = 0 if second_move.upper() == "C" else 1
        return float(self.payoff_matrix[row, column]), float(self.payoff_matrix[column, row])

    def payoff_table(self, strategies: Sequence[QuantumStrategy] = NASH_STRATEGIES):
        """Payoff matrix over a set of named strategies."""
        import pandas as pd

        rows = []
        for first in strategies:
            for second in strategies:
                payoffs = self.expected_payoffs(first, second)
                classical_first, classical_second = self.classical_payoffs(
                    first.name[0], second.name[0]
                )
                rows.append(
                    {
                        "first": first.name,
                        "second": second.name,
                        "quantum_first": round(payoffs["first"], 6),
                        "quantum_second": round(payoffs["second"], 6),
                        "classical_first": classical_first,
                        "classical_second": classical_second,
                        "cooperation": round(payoffs["cooperation"], 6),
                    }
                )
        return pd.DataFrame(rows)

    def best_response(
        self,
        opponent: QuantumStrategy,
        grid: int = 24,
        angle_range: tuple[float, float] = (0.0, float(np.pi)),
    ) -> dict:
        """Scan the payoff landscape for the first player's best response."""
        thetas = np.linspace(angle_range[0], angle_range[1], grid)
        phis = np.linspace(0.0, float(np.pi), grid)
        best = {"payoff": -np.inf, "theta": 0.0, "phi": 0.0}
        surface = np.zeros((grid, grid))
        for index, theta in enumerate(thetas):
            for column, phi in enumerate(phis):
                payoff = self.expected_payoffs(QuantumStrategy("probe", float(theta), float(phi)), opponent)[
                    "first"
                ]
                surface[index, column] = payoff
                if payoff > best["payoff"]:
                    best = {"payoff": float(payoff), "theta": float(theta), "phi": float(phi)}
        return {"best": best, "thetas": thetas.tolist(), "phis": phis.tolist(), "surface": surface.tolist()}

    def describe(self) -> str:
        return (
            f"{self.name}: entanglement gamma = {self.entanglement:.4f} rad, "
            f"depolarising noise = {self.noise:.3f}"
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "payoff_matrix": self.payoff_matrix.tolist(),
            "entanglement": float(self.entanglement),
            "noise": float(self.noise),
        }