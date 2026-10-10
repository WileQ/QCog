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

from ..core.instruments import Channel, depolarizing_channel, per_qubit_channel
from ..core.states import CognitiveState, as_state

__all__ = [
    "topology_adjacency",
    "edge_list",
    "partial_swap",
    "two_qubit_unitary",
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
            sub = ((bits[left]) << 1) | bits[right]
            for out in range(4):
                amplitude = gate[out, sub]
                if amplitude == 0:
                    continue
                new_bits = list(bits)
                new_bits[left] = (out >> 1) & 1
                new_bits[right] = out & 1
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
        """Bloch ``z`` component of one agent: negative is "against", positive "in favour"."""
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
    def step(self) -> None:
        """One round of coherent opinion exchange followed by decoherence."""
        for left, right in edge_list(self.adjacency):
            gate = partial_swap(self.interaction_strength * self.adjacency[left, right])
            unitary = _pair_unitary(gate, left, right, self.num_agents)
            self.state = self.state.evolve(unitary)
        if self.decoherence > 0:
            channel = per_qubit_channel(depolarizing_channel, self.num_agents, p=self.decoherence)
            self.state = self.state.apply_channel(channel.kraus)

    def run(self, steps: int, record: bool = True) -> dict:
        """Run ``steps`` rounds and return the opinion history."""
        history = [self.opinions_now()]
        entropies = [self.local_entropy()]
        for _ in range(int(steps)):
            self.step()
            if record:
                history.append(self.opinions_now())
                entropies.append(self.local_entropy())
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