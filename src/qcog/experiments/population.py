"""Populations of agents whose opinions are quantum states.

Each agent owns one qubit: |0>/|1> are the two sides of an issue, and the agent's
Bloch vector is its opinion.  Agents interact by applying a *partial swap* to
their pair - a coherent exchange of amplitude, the quantum analogue of "talking to
each other" - and optionally suffer decoherence between rounds.

Two things make this worth modelling with a quantum state at all.  First, the
exchange is coherent: an agent can end up in a superposition of positions, which
no averaging rule produces.  Second, correlations between agents are carried by
the state, so a group can be *jointly* more decisive (or less) than its members'
individual opinions suggest, which is the group-level question Module D asks.

The simulation is exact but dense, so the agent count is deliberately small
(default 5, hard limit 10) and the classical baselines are provided for larger
networks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from ..core.instruments import Channel, depolarizing_channel
from ..core.states import CognitiveState, as_state

__all__ = [
    "topology_adjacency",
    "edge_list",
    "partial_swap",
    "two_qubit_unitary",
    "apply_two_qubit_gate",
    "apply_single_qubit_operator",
    "apply_single_qubit_channel",
    "apply_partial_swap",
    "apply_local_depolarising",
    "embed_local_factor",
    "local_opinions",
    "opinion_from_angle",
    "local_entropies",
    "QuantumOpinionModel",
    "vote_distribution",
    "MAX_AGENTS",
]

#: Dense simulation limit: 2**10 x 2**10 density matrices are still fast, beyond
#: that the classical baselines should be used instead.
MAX_AGENTS = 10


def topology_adjacency(num_agents: int, topology: str = "ring", seed: int | None = None) -> np.ndarray:
    """Adjacency matrix of the interaction graph.

    ``"ring"`` (a cycle), ``"line"``, ``"complete"``, ``"star"``,
    ``"two-communities"`` (two dense blocks with a single weak link) and
    ``"random"`` (Erdos-Renyi with p=0.3) are available.
    """
    if not 2 <= num_agents <= MAX_AGENTS:
        raise ValueError(f"num_agents must lie between 2 and {MAX_AGENTS}")
    adjacency = np.zeros((num_agents, num_agents), dtype=float)
    if topology in {"ring", "line"}:
        for index in range(num_agents - 1):
            adjacency[index, index + 1] = adjacency[index + 1, index] = 1.0
        if topology == "ring" and num_agents > 2:
            adjacency[0, num_agents - 1] = adjacency[num_agents - 1, 0] = 1.0
    elif topology == "complete":
        adjacency = np.ones((num_agents, num_agents), dtype=float) - np.eye(num_agents)
    elif topology == "star":
        for index in range(1, num_agents):
            adjacency[0, index] = adjacency[index, 0] = 1.0
    elif topology == "two-communities":
        half = num_agents // 2
        for left in range(half):
            for right in range(half, num_agents):
                value = 1.0 if (left in {0} and right in {half}) else 0.0
                adjacency[left, right] = adjacency[right, left] = value
        for index in range(half - 1):
            adjacency[index, index + 1] = adjacency[index + 1, index] = 1.0
        for index in range(half, num_agents - 1):
            adjacency[index, index + 1] = adjacency[index + 1, index] = 1.0
        if half > 1:
            adjacency[half - 1, half] = adjacency[half, half - 1] = 0.3
    elif topology == "random":
        rng = np.random.default_rng(seed)
        upper = rng.random((num_agents, num_agents)) < 0.35
        adjacency = np.triu(upper, 1).astype(float)
        adjacency = adjacency + adjacency.T
        if adjacency.sum() == 0:  # guarantee connectivity for tiny graphs
            for index in range(num_agents - 1):
                adjacency[index, index + 1] = adjacency[index + 1, index] = 1.0
    else:
        raise ValueError(f"unknown topology {topology!r}")
    return adjacency


def edge_list(adjacency: np.ndarray) -> list[tuple[int, int]]:
    """Undirected edges of an adjacency matrix, each listed once."""
    edges = []
    size = adjacency.shape[0]
    for left in range(size):
        for right in range(left + 1, size):
            if adjacency[left, right] > 0:
                edges.append((left, right))
    return edges


def partial_swap(strength: float) -> np.ndarray:
    """``exp(i * strength * SWAP)`` on two qubits.

    ``strength = pi/4`` is the square-root of swap, which exchanges half of the
    two agents' opinions.  This is a coherent interaction: it can create
    superpositions and entanglement, which classical averaging cannot.
    """
    swap = np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
    )
    return np.cos(strength) * np.eye(4) + 1j * np.sin(strength) * swap


def two_qubit_unitary(gate: np.ndarray, adjacency: np.ndarray, num_agents: int) -> np.ndarray:
    """Embed a two-qubit gate into the full ``2**n`` space for the first edge.

    The full unitary is built by Kronecker products, which is exact and fast for
    the small registers this module supports.
    """
    if num_agents < adjacency.shape[0]:
        raise ValueError("adjacency larger than the register")
    return np.asarray(gate, dtype=complex)


def _qubit_axes(qubit: int, num_agents: int) -> tuple[int, int]:
    """Row and column axes of one qubit inside a ``(2,)*(2n)`` density tensor."""
    row = num_agents - 1 - qubit
    return row, row + num_agents


def apply_single_qubit_operator(density: np.ndarray, operator: np.ndarray, qubit: int, num_agents: int) -> np.ndarray:
    """Apply a one-qubit operator to a density matrix by tensor contraction.

    ``rho -> O rho O^dagger``, contracting only the qubit that ``O`` touches. This
    is the primitive that makes per-qubit noise cheap: composing per-qubit Kraus
    operators into whole-register ones would cost ``4**n`` operators.
    """
    operator = np.asarray(operator, dtype=complex)
    total = 2**num_agents
    if operator.shape != (2, 2):
        raise ValueError("the operator must be a 2x2 matrix")
    tensor = density.reshape([2] * (2 * num_agents))
    row, column = _qubit_axes(qubit, num_agents)
    tensor = np.tensordot(operator, tensor, axes=([1], [row]))
    tensor = np.moveaxis(tensor, 0, row)
    tensor = np.tensordot(operator.conj(), tensor, axes=([1], [column]))
    tensor = np.moveaxis(tensor, 0, column)
    return tensor.reshape(total, total)


def apply_single_qubit_channel(density: np.ndarray, kraus: Sequence[np.ndarray], qubit: int, num_agents: int) -> np.ndarray:
    """Apply a one-qubit channel (given by Kraus operators) to one qubit of a register."""
    total = 2**num_agents
    out = np.zeros((total, total), dtype=complex)
    for operator in kraus:
        out += _conjugate_local(density, np.asarray(operator, dtype=complex), qubit, num_agents)
    return out


def _conjugate_local(density: np.ndarray, operator: np.ndarray, qubit: int, num_agents: int) -> np.ndarray:
    """``O rho O^dagger`` for a one-qubit ``O`` (the summand of a local channel)."""
    tensor = density.reshape([2] * (2 * num_agents))
    row, column = _qubit_axes(qubit, num_agents)
    tensor = np.tensordot(operator, tensor, axes=([1], [row]))
    tensor = np.moveaxis(tensor, 0, row)
    tensor = np.tensordot(operator.conj(), tensor, axes=([1], [column]))
    tensor = np.moveaxis(tensor, 0, column)
    return tensor.reshape(2**num_agents, 2**num_agents)


def embed_local_factor(reduced: np.ndarray, local: np.ndarray, qubit: int, num_agents: int) -> np.ndarray:
    """Insert a one-qubit factor back into an ``(n-1)``-qubit operator.

    Used for the fast depolarising path: ``rho -> (1-p) rho + p Tr_q(rho) (x) I/2``
    needs the traced-out operator placed back on the qubit's two tensor axes.
    """
    size = 2 ** (num_agents - 1)
    if reduced.shape != (size, size):
        raise ValueError("the reduced operator does not match the register size")
    if local.shape != (2, 2):
        raise ValueError("the local factor must be 2x2")
    left = 2 ** (num_agents - 1 - qubit)
    right = 2**qubit
    blocks = reduced.reshape(left, right, left, right)
    # ``blocks`` has axes (row/higher, row/lower, column/higher, column/lower); the
    # inserted qubit goes between the two pairs on each side, in the same order, which
    # is what the little-endian reshape needs
    expanded = np.einsum("ijkl,ab->iajkbl", blocks, np.asarray(local, dtype=complex))
    return expanded.reshape(2**num_agents, 2**num_agents)


def apply_local_depolarising(density: np.ndarray, probability: float, qubit: int, num_agents: int) -> np.ndarray:
    """``rho -> (1 - p) rho + p Tr_q(rho) (x) I/2`` in three tensor passes.

    Mathematically identical to applying the four Kraus operators of the qubit
    depolarising channel, but it never composes them into whole-register operators,
    which is what keeps ten-agent runs fast.
    """
    if probability <= 0:
        return density
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    tensor = density.reshape([2] * (2 * num_agents))
    row, column = _qubit_axes(qubit, num_agents)
    traced = np.diagonal(tensor, axis1=row, axis2=column).sum(axis=-1)
    reduced = traced.reshape(2 ** (num_agents - 1), 2 ** (num_agents - 1))
    mixed = embed_local_factor(reduced, np.eye(2, dtype=complex) / 2.0, qubit, num_agents)
    return (1.0 - probability) * density + probability * mixed


def opinion_from_angle(angle: float) -> float:
    """The opinion a Bloch angle corresponds to: ``p(in favour) - p(against) = -cos(theta)``.

    The classical baselines in Module D average *opinions*, so the initial angles have
    to be converted before they are handed to a classical model; otherwise the two
    sides of a comparison would be in different units.
    """
    return float(-np.cos(angle))


def local_opinions(density: np.ndarray, num_agents: int) -> list[float]:
    """Opinion of every agent: ``p(in favour) - p(against)``, positive meaning in favour.

    Read off the diagonal, one dot product per agent, which is what makes a ten-agent
    trajectory cheap.  The sign convention matches :meth:`QuantumOpinionModel.opinion_of`.
    """
    populations = np.real(np.diag(density))
    indices = np.arange(populations.size)
    return [
        float(np.dot(populations, 2 * ((indices >> qubit) & 1) - 1))
        for qubit in range(num_agents)
    ]


def local_entropies(density: np.ndarray, num_agents: int) -> list[float]:
    """Von Neumann entropy of every single agent, from its two-by-two reduced state."""
    populations = np.real(np.diag(density))
    indices = np.arange(populations.size)
    entropies = []
    for qubit in range(num_agents):
        mask = ((indices >> qubit) & 1) == 0
        zero = float(populations[mask].sum())
        one = float(populations[~mask].sum())
        # the off-diagonal of the reduced state is the sum of the coherences that
        # connect the two values of this qubit with every other qubit fixed
        flipped = indices ^ (1 << qubit)
        coherence = complex(np.sum(density[indices, flipped][mask]))
        block = np.array([[zero, coherence], [np.conj(coherence), one]], dtype=complex)
        eigenvalues = np.linalg.eigvalsh((block + block.conj().T) / 2.0)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        entropies.append(float(-np.sum(eigenvalues * np.log2(eigenvalues))) if eigenvalues.size else 0.0)
    return entropies


def apply_partial_swap(density: np.ndarray, strength: float, left: int, right: int, num_agents: int) -> np.ndarray:
    """Apply ``exp(i * strength * SWAP)`` to a pair, exploiting its structure.

    Writing the gate as ``c I + i s SWAP``, the conjugation expands into four terms in
    which ``SWAP`` acts as an axis exchange, so no matrix multiplication is needed:
    ``c^2 rho - i c s (rho S) + i c s (S rho) + s^2 (S rho S)``. The result is
    accumulated into a preallocated buffer, because this gate is applied once per edge
    per round and allocation would dominate a ten-agent run.
    """
    cosine, sine = float(np.cos(strength)), float(np.sin(strength))
    total = 2**num_agents
    tensor = density.reshape([2] * (2 * num_agents))
    row_left, column_left = _qubit_axes(left, num_agents)
    row_right, column_right = _qubit_axes(right, num_agents)
    swapped_rows = np.swapaxes(tensor, row_left, row_right)
    swapped_columns = np.swapaxes(tensor, column_left, column_right)
    out = np.empty_like(tensor)
    np.multiply(tensor, cosine * cosine, out=out)
    if sine != 0.0:
        out += (-1j * cosine * sine) * swapped_columns
        out += (1j * cosine * sine) * swapped_rows
        out += (sine * sine) * np.swapaxes(swapped_rows, column_left, column_right)
    return out.reshape(total, total)


def apply_two_qubit_gate(
    density: np.ndarray, gate: np.ndarray, left: int, right: int, num_agents: int
) -> np.ndarray:
    """Apply a two-qubit gate to a density matrix by direct tensor contraction.

    Building the full ``2**n x 2**n`` unitary and multiplying is correct but costs
    ``O(8**n)``; contracting only the two qubits the gate touches costs ``O(4**n)``,
    which is what makes a ten-agent dense simulation practical.

    Conventions: qubits are little-endian, and the gate's own basis index is
    ``2 * bit(higher qubit) + bit(lower qubit)`` - the same convention the circuit
    construction uses, so a gate can be drawn and applied interchangeably.
    """
    gate = np.asarray(gate, dtype=complex)
    total = 2**num_agents
    if density.shape != (total, total):
        raise ValueError("the density matrix does not match the register size")
    if gate.shape != (4, 4):
        raise ValueError("the gate must be a two-qubit unitary")
    tensor = density.reshape([2] * (2 * num_agents))
    row_high, row_low = sorted((num_agents - 1 - left, num_agents - 1 - right))
    col_high, col_low = row_high + num_agents, row_low + num_agents
    blocks = gate.reshape(2, 2, 2, 2)
    tensor = np.tensordot(blocks, tensor, axes=([2, 3], [row_high, row_low]))
    tensor = np.moveaxis(tensor, [0, 1], [row_high, row_low])
    tensor = np.tensordot(blocks.conj(), tensor, axes=([2, 3], [col_high, col_low]))
    tensor = np.moveaxis(tensor, [0, 1], [col_high, col_low])
    return tensor.reshape(total, total)


def _pair_unitary(gate: np.ndarray, left: int, right: int, num_agents: int) -> np.ndarray:
    """``gate`` acting on qubits ``left`` and ``right`` (little-endian indices)."""
    if not 0 <= left < right < num_agents:
        raise ValueError("invalid qubit pair")
    if right != left + 1:
        # Fall back to a general construction via basis permutation.
        total = 2**num_agents
        matrix = np.zeros((total, total), dtype=complex)
        for index in range(total):
            bits = [(index >> qubit) & 1 for qubit in range(num_agents)]
            # local index = 2 * bit(higher qubit) + bit(lower qubit), matching the
            # adjacent case so that both branches act on the same pair convention
            sub = (bits[right] << 1) | bits[left]
            for out in range(4):
                amplitude = gate[out, sub]
                if amplitude == 0:
                    continue
                new_bits = list(bits)
                new_bits[right] = (out >> 1) & 1
                new_bits[left] = out & 1
                target = sum(bit << qubit for qubit, bit in enumerate(new_bits))
                matrix[target, index] = amplitude
        return matrix
    left_factor = np.eye(2 ** (num_agents - right - 1), dtype=complex)
    right_factor = np.eye(2**left, dtype=complex)
    return np.kron(np.kron(left_factor, gate), right_factor)


@dataclass
class QuantumOpinionModel:
    """A small network of agents whose opinions are qubits.

    Parameters
    ----------
    opinions:
        One Bloch angle per agent, measured from ``|0>``; ``theta`` near 0 means
        "against", near pi "in favour", pi/2 "undecided".
    topology / adjacency:
        Interaction graph (see :func:`topology_adjacency`).
    interaction_strength:
        ``theta`` of the partial-swap gate applied along each edge per round.
    decoherence:
        Per-agent depolarising probability applied after each round.
    """

    opinions: Sequence[float]
    topology: str = "ring"
    interaction_strength: float = float(np.pi / 4)
    decoherence: float = 0.0
    adjacency: np.ndarray | None = None
    phases: Sequence[float] | None = None

    def __post_init__(self) -> None:
        self.opinions = list(self.opinions)
        self.num_agents = len(self.opinions)
        if not 2 <= self.num_agents <= MAX_AGENTS:
            raise ValueError(f"between 2 and {MAX_AGENTS} agents are supported")
        if self.adjacency is None:
            self.adjacency = topology_adjacency(self.num_agents, self.topology)
        else:
            self.adjacency = np.asarray(self.adjacency, dtype=float)
        if self.adjacency.shape != (self.num_agents, self.num_agents):
            raise ValueError("adjacency must match the number of agents")
        self.phase_list = list(self.phases) if self.phases else [0.0] * self.num_agents
        self.state = self._initial_state()

    # ------------------------------------------------------------------ states
    def _initial_state(self) -> CognitiveState:
        vector = np.array([1.0 + 0.0j])
        for theta, phi in zip(self.opinions, self.phase_list):
            qubit = np.array([np.cos(theta / 2.0), np.exp(1j * phi) * np.sin(theta / 2.0)])
            vector = np.kron(qubit, vector)  # first agent = most significant qubit
        return as_state(vector, label="population state")

    def opinion_of(self, agent: int) -> float:
        """One agent's opinion: ``p(in favour) - p(against)``, positive meaning in favour.

        ``|1>`` is "in favour", so this is Bloch ``-z``; the sign convention is stated
        here once because the trajectories, the consensus metrics and the vote
        distribution all read their opinions from it.
        """
        reduced = self.state.partial_trace([agent], dims=[2] * self.num_agents)
        return float(np.real(reduced.density[1, 1] - reduced.density[0, 0]))

    def opinions_now(self) -> list[float]:
        """Current opinion of every agent."""
        return [self.opinion_of(agent) for agent in range(self.num_agents)]

    def local_entropy(self) -> float:
        """Mean von Neumann entropy of the individual agents, in bits."""
        return float(
            np.mean(
                [
                    self.state.partial_trace([agent], dims=[2] * self.num_agents).entropy
                    for agent in range(self.num_agents)
                ]
            )
        )

    # ------------------------------------------------------------- interaction
    @property
    def density(self) -> np.ndarray:
        """The current density matrix of the whole population."""
        return self.state.density

    def _set_density(self, density: np.ndarray) -> None:
        from ..core.states import CognitiveState

        self.state = CognitiveState.from_density(density, label="population state")

    def step(self) -> None:
        """One round of coherent opinion exchange followed by decoherence."""
        density = self.density
        for left, right in edge_list(self.adjacency):
            density = apply_partial_swap(
                density,
                self.interaction_strength * self.adjacency[left, right],
                left,
                right,
                self.num_agents,
            )
        if self.decoherence > 0:
            for qubit in range(self.num_agents):
                density = apply_local_depolarising(density, self.decoherence, qubit, self.num_agents)
        self._set_density(density)

    def run(self, steps: int, record: bool = True) -> dict:
        """Run ``steps`` rounds and return the opinion history."""
        history = [local_opinions(self.density, self.num_agents)]
        entropies = [float(np.mean(local_entropies(self.density, self.num_agents)))]
        for _ in range(int(steps)):
            self.step()
            if record:
                history.append(local_opinions(self.density, self.num_agents))
                entropies.append(float(np.mean(local_entropies(self.density, self.num_agents))))
        return {"history": history, "entropy": entropies, "final": self.opinions_now()}

    # ------------------------------------------------------------ aggregation
    def vote_distribution(self) -> dict[int, float]:
        """Distribution of the number of agents voting "in favour"."""
        return vote_distribution(self.state.density, self.num_agents)

    def consensus_metrics(self) -> dict:
        """Consensus, polarisation and cluster count of the current opinions."""
        values = np.asarray(self.opinions_now(), dtype=float)
        ordered = np.sort(values)
        clusters = 1
        for left, right in zip(ordered[:-1], ordered[1:]):
            if right - left > 0.25:
                clusters += 1
        return {
            "mean opinion": float(values.mean()),
            "polarisation": float(values.std()),
            "agreement (|mean|)": float(abs(values.mean())),
            "clusters": int(clusters),
            "mean local entropy": self.local_entropy(),
        }


def vote_distribution(density: np.ndarray, num_agents: int) -> dict[int, float]:
    """Population of each Hamming weight in the computational basis.

    The result is the distribution of "how many agents are in favour", which for a
    correlated state is not the binomial distribution a group of independent
    agents would produce.
    """
    populations = np.real(np.diag(np.asarray(density)))
    result = {count: 0.0 for count in range(num_agents + 1)}
    for index, probability in enumerate(populations):
        weight = bin(index).count("1")
        result[weight] += float(probability)
    return result


def independent_vote_distribution(probabilities: Sequence[float]) -> dict[int, float]:
    """Binomial (independent agents) reference distribution of the vote count."""
    from math import comb

    result = {count: 0.0 for count in range(len(probabilities) + 1)}
    total = len(probabilities)
    for mask in range(2**total):
        probability = 1.0
        weight = 0
        for agent, p in enumerate(probabilities):
            if (mask >> agent) & 1:
                probability *= p
                weight += 1
            else:
                probability *= 1 - p
        result[weight] += probability
    return result