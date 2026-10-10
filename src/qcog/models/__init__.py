"""The model catalogue: twenty-three models in four modules.

Import the individual modules (``qcog.models.module_a`` ...) for the classes, or
use :mod:`qcog.registry` for lookup by slug.
"""

from .base import MODULES, Model, ModelResult, make_engine
from .module_a import (
    BeliefRevision,
    DecisionTree,
    InterferenceParadox,
    QuestionOrder,
    SuperpositionBeliefs,
)
from .module_b import (
    AssociativeMemory,
    ConceptCombination,
    ConceptualEntanglement,
    ConjunctionFallacy,
    ContradictoryBeliefs,
    DisjunctionEffect,
    SemanticAmbiguity,
    SequentialCategorization,
)
from .module_c import (
    Auctions,
    CoordinationGames,
    EvolutionOfCooperation,
    NegotiationTrust,
    QuantumPrisonersDilemma,
)
from .module_d import (
    CollectiveJudgment,
    DeliberationVoting,
    InformationCascades,
    Polarization,
    QuantumOpinionDynamics,
)

__all__ = [
    "MODULES",
    "Model",
    "ModelResult",
    "make_engine",
    "SuperpositionBeliefs",
    "QuestionOrder",
    "InterferenceParadox",
    "DecisionTree",
    "BeliefRevision",
    "ConjunctionFallacy",
    "DisjunctionEffect",
    "ConceptCombination",
    "AssociativeMemory",
    "SemanticAmbiguity",
    "SequentialCategorization",
    "ConceptualEntanglement",
    "ContradictoryBeliefs",
    "QuantumPrisonersDilemma",
    "NegotiationTrust",
    "EvolutionOfCooperation",
    "Auctions",
    "CoordinationGames",
    "QuantumOpinionDynamics",
    "Polarization",
    "CollectiveJudgment",
    "InformationCascades",
    "DeliberationVoting",
]