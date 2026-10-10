"""Reusable experiment designs.

These are the handful of structures that most of the catalogue is built from: two
questions asked in both orders, a two-player quantum game in the
Eisert-Wilkert-Lewenstein scheme, and a population of interacting agents. Keeping
them here - rather than inside the models - is what lets a new model be a
configuration plus an explanation instead of a new simulator.
"""

from .games import (
    CLASSICAL_C,
    CLASSICAL_D,
    QUANTUM_Q,
    QuantumGame,
    QuantumStrategy,
    entangling_unitary,
    strategy_unitary,
)
from .population import (
    MAX_AGENTS,
    QuantumOpinionModel,
    edge_list,
    independent_vote_distribution,
    partial_swap,
    topology_adjacency,
    vote_distribution,
)
from .two_question import OrderEffectBundle, TwoQuestionExperiment

__all__ = [
    "CLASSICAL_C",
    "CLASSICAL_D",
    "QUANTUM_Q",
    "QuantumGame",
    "QuantumStrategy",
    "entangling_unitary",
    "strategy_unitary",
    "MAX_AGENTS",
    "QuantumOpinionModel",
    "edge_list",
    "independent_vote_distribution",
    "partial_swap",
    "topology_adjacency",
    "vote_distribution",
    "OrderEffectBundle",
    "TwoQuestionExperiment",
]
