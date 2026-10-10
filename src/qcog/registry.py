"""The model catalogue.

Importing this module gives access to every model in the library, plus the
lookups the CLI, the interactive explorer, the report generator and the generated
documentation all use.  Adding a model means adding one class and one registry
entry; nothing else in the library needs to change.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .models.base import MODULES, Model

#: Alias kept for the UI, which wants the module letters in order.
MODULE_ORDER = MODULES

__all__ = [
    "MODELS",
    "MODULES",
    "MODULE_ORDER",
    "get_model",
    "list_models",
    "models_by_module",
    "catalogue",
    "search",
]

MODELS: dict[str, type[Model]] = {}


def _register() -> None:
    from .models.module_a import (
        BeliefRevision,
        DecisionTree,
        InterferenceParadox,
        QuestionOrder,
        SuperpositionBeliefs,
    )
    from .models.module_b import (
        AssociativeMemory,
        ConceptCombination,
        ConceptualEntanglement,
        ConjunctionFallacy,
        ContradictoryBeliefs,
        DisjunctionEffect,
        SemanticAmbiguity,
        SequentialCategorization,
    )
    from .models.module_c import (
        Auctions,
        CoordinationGames,
        EvolutionOfCooperation,
        NegotiationTrust,
        QuantumPrisonersDilemma,
    )
    from .models.module_d import (
        CollectiveJudgment,
        DeliberationVoting,
        InformationCascades,
        Polarization,
        QuantumOpinionDynamics,
    )

    for model in (
        # Module A
        SuperpositionBeliefs,
        QuestionOrder,
        InterferenceParadox,
        DecisionTree,
        BeliefRevision,
        # Module B
        ConjunctionFallacy,
        DisjunctionEffect,
        ConceptCombination,
        AssociativeMemory,
        SemanticAmbiguity,
        SequentialCategorization,
        ConceptualEntanglement,
        ContradictoryBeliefs,
        # Module C
        QuantumPrisonersDilemma,
        NegotiationTrust,
        EvolutionOfCooperation,
        Auctions,
        CoordinationGames,
        # Module D
        QuantumOpinionDynamics,
        Polarization,
        CollectiveJudgment,
        InformationCascades,
        DeliberationVoting,
    ):
        MODELS[model.slug] = model


_register()


def get_model(slug: str, **parameters) -> Model:
    """Instantiate a model by slug.

    Parameters are passed straight to the constructor's keyword arguments (Q-Cog
    models take none) and then to :meth:`Model.run`, so ``get_model("question-order")``
    returns a ready object and ``get_model(slug).run(theta=...)`` runs it.
    """
    if slug not in MODELS:
        raise KeyError(
            f"unknown model {slug!r}; available: {', '.join(sorted(MODELS))}"
        )
    return MODELS[slug](**parameters)


def list_models() -> list[str]:
    """Slugs of every registered model, in catalogue order."""
    return list(MODELS)


def models_by_module() -> dict[str, list[type[Model]]]:
    """Models grouped by module letter, in catalogue order."""
    grouped: dict[str, list[type[Model]]] = {letter: [] for letter in MODULES}
    for model in MODELS.values():
        grouped.setdefault(model.module, []).append(model)
    return grouped


def catalogue() -> list[dict]:
    """Metadata for every model, used by the docs generator and the UI."""
    return [model.to_dict() for model in MODELS.values()]


def search(text: str) -> list[type[Model]]:
    """Models whose title, one-liner, keywords or question mention ``text``."""
    needle = text.strip().lower()
    if not needle:
        return list(MODELS.values())
    found = []
    for model in MODELS.values():
        haystack = " ".join(
            [model.slug, model.title, model.one_liner, model.cognitive_question,
             model.quantum_mechanism, " ".join(model.keywords)]
        ).lower()
        if needle in haystack:
            found.append(model)
    return found