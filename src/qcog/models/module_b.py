"""Module B - logic, concepts and memory.

Eight models about the content of thought: conjunctions and disjunctions, the
meaning of combined concepts, recall under competing cues, how a context fixes a
word's sense, how repeated categorisation stabilises, when joint structure cannot
be reduced to parts, and how beliefs that cannot be checked at the same time
coexist.  The recurring theme is that the *space* in which a judgement is
represented decides which classical identities have to hold.
"""

from __future__ import annotations

import numpy as np

from ..analysis.entanglement import entanglement_report
from ..analysis.interference import interference_decomposition
from ..analysis.sweeps import sweep
from ..core.classical import ClassicalBaseline, ClassicalConjunction
from ..core.measurement import ProjectiveMeasurement, computational, observable_measurement
from ..core.params import ParamSpec
from ..core.states import amplitudes, basis_state, mixed_state, tensor
from ..circuits.gates import rotation_matrix
from .base import Model, largest_outcome
from .common import (
    default_caveats,
    interference_sentence,
    local_question,
    measurement_for,
    percent,
    qubit,
    signed,
)
from ..experiments.two_question import TwoQuestionExperiment
from .shared import OrderBasedModel

__all__ = [
    "ConjunctionFallacy",
    "DisjunctionEffect",
    "ConceptCombination",
    "AssociativeMemory",
    "SemanticAmbiguity",
    "SequentialCategorization",
    "ConceptualEntanglement",
    "ContradictoryBeliefs",
]

PI = float(np.pi)

class ConjunctionFallacy(Model):
    """The Linda problem: a conjunction judged more probable than its conjunct.

    One qubit carries the belief.  "Is Linda a feminist?" is the ``Z`` question,
    "is Linda a bank teller?" is a question at an angle to it, and the belief state
    carries a relative phase.  The phase lets the two components of the belief
    interfere *destructively* in the direct bank-teller question, lowering p(bank
    teller); the conjunction is then evaluated after the feminist question has
    collapsed the state, where no interference remains.  The result is the classic
    pattern: p(feminist) high, p(bank teller) low, p(both) high - above the
    classical upper bound.

    The model also runs the same two questions on an *independent-attribute*
    representation to show that the fallacy then vanishes exactly.  That contrast is
    the point: the fallacy needs the two questions to be incompatible, not just
    uncertain.
    """

    slug = "conjunction-fallacy"
    title = "Conjunction fallacy (the Linda problem)"
    module = "B"
    one_liner = (
        "A relative phase in the belief lowers p(bank teller) while leaving the conjunction untouched, so "
        "'bank teller and feminist' is judged more probable than 'bank teller'."
    )
    cognitive_question = (
        "Why do people rate 'Linda is a bank teller and a feminist' as more probable than 'Linda is a bank "
        "teller', when no probability space allows it?"
    )
    quantum_mechanism = (
        "Destructive interference between the components of the belief state in the direct bank-teller "
        "question; the conjunction is evaluated after the feminist question has collapsed the state, so it "
        "keeps the higher conditional probability."
    )
    classical_baseline = (
        "The conjunction rule p(A and B) <= min(p(A), p(B)), which holds in every classical probability space "
        "and is also exact for commuting (independent-attribute) quantum questions."
    )
    references = (
        "Tversky & Kahneman (1983), Extensional versus intuitive reasoning, Psychological Review 90, 293-315.",
        "Busemeyer, Pothos, Franco & Trueblood (2011), A quantum theoretical explanation for probability judgment errors, Psychological Review 118, 193-218.",
        "Pothos & Busemeyer (2013), Can quantum probability provide a new direction for cognitive modeling? BBS 36, 255-274.",
    )
    caveats = default_caveats + (
        "The model reproduces the fallacy with a phase; it does not claim that beliefs literally have phases.",
        "The independent-attribute variant of the same model satisfies the classical bound exactly, so the "
        "fallacy is a property of incompatible questions, not of uncertainty as such.",
    )
    keywords = ("conjunction fallacy", "interference", "Linda", "probability judgment", "incompatible questions")
    difficulty = "intermediate"

    params = (
        ParamSpec("feminist_amplitude", "float", 1.2, "Feminist amplitude",
                  "Angle of the belief state towards the feminist answer (pi/2 = certain).", 0.0, PI),
        ParamSpec("teller_angle", "float", 2.0, "Bank-teller question angle",
                  "Angle of the bank-teller question relative to the feminist one; pi/2 makes them maximally "
                  "incompatible.", 0.0, PI, fittable=True),
        ParamSpec("relative_phase", "float", PI, "Relative phase",
                  "Phase of the belief state: pi gives maximal destructive interference in the direct "
                  "bank-teller question.", -PI, PI),
    )

    def build(self, **parameters) -> dict:
        angle = float(parameters["feminist_amplitude"])
        phase = float(parameters["relative_phase"])
        state = amplitudes(
            np.cos(angle),
            np.exp(1j * phase) * np.sin(angle),
            label="belief about Linda",
            basis_labels=("not feminist", "feminist"),
        )
        feminist = measurement_for(0.0, "feminist", ("no", "yes"), "is Linda a feminist?")
        # The first ray of a question is the "yes" ray, so asking about the bank teller
        # attribute at an angle uses the default outcome order and the feminist (Z)
        # question swaps its labels so that "yes" means |1> = feminist.
        teller = measurement_for(float(parameters["teller_angle"]), "bank teller", ("yes", "no"),
                                 "is Linda a bank teller?")
        return {
            "state": state,
            "feminist": feminist,
            "teller": teller,
            "parameters": dict(parameters),
        }

    def simulate(self, context: dict, engine) -> dict:
        state, feminist, teller = context["state"], context["feminist"], context["teller"]
        return {
            "feminist_then_teller": engine.run(state, [feminist, teller]),
            "teller_only": engine.run(state, [teller]),
            "feminist_only": engine.run(state, [feminist]),
        }

    def analyse(self, context: dict, results: dict) -> tuple[dict, list[str], str]:
        sequence = results["feminist_then_teller"]
        p_conjunction = sequence.probability({"feminist": "yes", "bank teller": "yes"})
        p_teller = results["teller_only"].marginal("bank teller")["yes"]
        p_feminist = results["feminist_only"].marginal("feminist")["yes"]
        bound = ClassicalConjunction()
        excess = p_conjunction - bound.bound(p_teller, p_feminist)
        # Reference representation with independent attributes: a product of the two
        # marginals.  There the classical bound is exact, so the excess is zero.
        independent_conjunction = p_teller * p_feminist
        independent_excess = independent_conjunction - min(p_teller, p_feminist)
        classical = ClassicalBaseline(
            name="classical conjunction rule",
            description="p(A and B) <= min(p(A), p(B)), exact for any single probability space.",
            prediction={"classical bound": bound.bound(p_teller, p_feminist)},
        )
        sweep_result = sweep(
            "relative_phase",
            [round(value, 3) for value in np.linspace(-PI, PI, 13)],
            lambda value: _conjunction_metrics(context, float(value)),
        )
        findings = [
            f"Model: p(feminist) = {p_feminist:.3f}, p(bank teller) = {p_teller:.3f}, "
            f"p(bank teller and feminist) = {p_conjunction:.3f}.",
            f"The classical upper bound is {bound.bound(p_teller, p_feminist):.3f}, so the model exceeds it by "
            f"{excess:+.3f}: the conjunction is judged more probable than one of its conjuncts.",
            "The whole effect comes from the phase of the belief state: it makes the two components interfere "
            "destructively in the direct bank-teller question, while the conjunction is asked of a collapsed "
            "state. The phase sweep crosses zero exactly where the fallacy disappears.",
            f"With independent attributes the joint probability would be p(feminist) x p(bank teller) = "
            f"{independent_conjunction:.3f}, which cannot exceed the bound ({independent_excess:+.3f}): the "
            "fallacy is a signature of incompatible questions, not of uncertainty as such.",
            f"Classical bound {bound.bound(p_teller, p_feminist):.3f} versus the |teller> conditional "
            f"{p_conjunction / max(p_feminist, 1e-12):.3f} - the gap a judge can check by hand.",
        ]
        summary = (
            f"p(conjunction) = {p_conjunction:.3f} exceeds the classical bound "
            f"{bound.bound(p_teller, p_feminist):.3f} by {excess:.3f}; independent attributes give "
            f"{independent_excess:+.3f}."
        )
        analysis = {
            "conjunction": bound.to_dict(p_conjunction, p_teller, p_feminist),
            "independent_check": {
                "p(conjunction)": independent_conjunction,
                "p(bank teller)": p_teller,
                "p(feminist)": p_feminist,
                "excess": independent_excess,
            },
            "classical": classical.to_dict(),
            "sweeps": {"relative phase": sweep_result},
        }
        return analysis, findings, summary

    def predict(self, parameters: dict | None = None) -> dict[str, float]:
        """Predicted probabilities for the two marginals and the conjunction."""
        values = self.defaults()
        values.update(parameters or {})
        context = self.build(**values)
        from ..core.engine import run_sequence

        sequence = run_sequence(context["state"], [context["feminist"], context["teller"]])
        teller_only = run_sequence(context["state"], [context["teller"]])
        feminist_only = run_sequence(context["state"], [context["feminist"]])
        return {
            "p(feminist)": feminist_only.marginal("feminist")["yes"],
            "p(bank teller)": teller_only.marginal("bank teller")["yes"],
            "p(feminist and bank teller)": sequence.probability(
                {"feminist": "yes", "bank teller": "yes"}
            ),
        }

    def context_metadata(self, context: dict) -> dict:
        return {"hilbert_dim": context["state"].dim, "representation": "single belief space"}


def _conjunction_metrics(context: dict, phase: float) -> dict:
    """Bound and conjunction as the relative phase is swept."""
    from ..core.engine import run_sequence

    parameters = dict(context["parameters"])
    parameters["relative_phase"] = phase
    rebuilt = ConjunctionFallacy().build(**parameters)
    sequence = run_sequence(rebuilt["state"], [rebuilt["feminist"], rebuilt["teller"]])
    teller_only = run_sequence(rebuilt["state"], [rebuilt["teller"]])
    feminist_only = run_sequence(rebuilt["state"], [rebuilt["feminist"]])
    conjunction = sequence.probability({"feminist": "yes", "bank teller": "yes"})
    teller = teller_only.marginal("bank teller")["yes"]
    feminist = feminist_only.marginal("feminist")["yes"]
    return {
        "p(bank teller)": teller,
        "p(conjunction)": conjunction,
        "classical bound": min(teller, feminist),
        "excess": conjunction - min(teller, feminist),
    }


class DisjunctionEffect(Model):
    """The disjunction effect, and why it needs a particular representation.

    Shafir and Tversky found that people who cooperate when they know the
    opponent's move, and also cooperate when they know the opposite move, behave
    differently when they know neither.  The model shows *when* a
    representation can produce this.  Any state of the form "world x action" - the
    opponent's move in one subsystem and one's own action in another, however
    entangled - must satisfy the sure-thing principle exactly, because the action
    marginal is the branch mixture.  The effect appears only in the representational
    alternative, where the two unresolved branches contribute coherent amplitudes
    to the *same* decision state.  That is a statement about how the uncertainty
    has to be represented, and it is checked numerically here.
    """

    slug = "disjunction-effect"
    title = "Disjunction effect and the representation of unresolved alternatives"
    module = "B"
    one_liner = (
        "Cooperation under an unknown opponent differs from the average of the two known cases - and the "
        "model proves that no 'world x action' state, entangled or not, can reproduce that."
    )
    cognitive_question = (
        "Why is the decision different only when the alternatives are unresolved, if the decision is the same "
        "in each resolved branch?"
    )
    quantum_mechanism = (
        "Coherent amplitudes for the two unresolved branches adding to the same decision state, versus the "
        "incoherent branch mixture produced by any tensor-product representation."
    )
    classical_baseline = (
        "The sure-thing principle: p(act) must equal the branch-weighted average of the conditional act "
        "probabilities.  It is also the exact prediction of any bipartite (world x action) state."
    )
    references = (
        "Shafir & Tversky (1992), Thinking through uncertainty: nonconsequential reasoning and choice, Cognitive Psychology 24, 449-474.",
        "Busemeyer & Bruza (2012), Quantum Models of Cognition and Decision, ch. 9.",
    )
    caveats = default_caveats + (
        "The model demonstrates that the effect requires a non-reducible representation; it does not show "
        "that people use one.",
    )
    keywords = ("disjunction effect", "sure-thing principle", "cooperation", "representation", "entanglement")
    difficulty = "advanced"

    params = (
        ParamSpec("cooperate_if_opponent_cooperates", "float", 0.8, "Cooperate when the opponent cooperates",
                  "Conditional cooperation rate in the resolved branch.", 0.0, 1.0, fittable=True),
        ParamSpec("cooperate_if_opponent_defects", "float", 0.6, "Cooperate when the opponent defects",
                  "Conditional cooperation rate in the other resolved branch.", 0.0, 1.0, fittable=True),
        ParamSpec("opponent_cooperation", "float", 0.5, "Belief that the opponent cooperates",
                  "Weight of the cooperating branch.", 0.05, 0.95),
        ParamSpec("relative_phase", "float", PI, "Relative phase",
                  "The coherence of the defecting branch relative to the cooperating one: pi makes the "
                  "defect amplitudes cancel when the opponent is unknown, which raises cooperation.",
                  -PI, PI),
    )

    def build(self, **parameters) -> dict:
        p_a = float(parameters["cooperate_if_opponent_cooperates"])
        p_b = float(parameters["cooperate_if_opponent_defects"])
        weight_a = float(parameters["opponent_cooperation"])
        weight_b = 1.0 - weight_a
        phase = float(parameters["relative_phase"])
        # Branch A: the opponent cooperates.  Branch B: the opponent defects.
        a_cooperate = np.sqrt(weight_a * p_a)
        a_defect = np.sqrt(weight_a * (1 - p_a))
        b_cooperate = np.sqrt(weight_b * p_b)
        b_defect = np.exp(1j * phase) * np.sqrt(weight_b * (1 - p_b))
        holistic = amplitudes(
            a_cooperate + b_cooperate,
            a_defect + b_defect,
            label="unresolved decision state",
            basis_labels=("cooperate", "defect"),
        )
        branch_a = amplitudes(a_cooperate, a_defect, label="opponent cooperates")
        branch_b = amplitudes(b_cooperate, b_defect, label="opponent defects")
        mixture = mixed_state([(weight_a, branch_a), (weight_b, branch_b)], label="resolved branch mixture")
        # The best possible tensor-product representation: an entangled
        # (world x action) state that reduces to exactly the same mixture.
        entangled = amplitudes(
            a_cooperate,
            a_defect,
            b_cooperate,
            b_defect,
            label="world x action (entangled)",
            basis_labels=(
                "opponent cooperates, cooperate",
                "opponent cooperates, defect",
                "opponent defects, cooperate",
                "opponent defects, defect",
            ),
        )
        decision_single = computational("decision", ("cooperate", "defect"), "the person's own action")
        decision_joint = local_question(0.0, 0, 2, "decision", ("cooperate", "defect"),
                                        description="the person's own action")
        world = local_question(0.0, 1, 2, "opponent", ("cooperates", "defects"),
                               description="the opponent's move")
        return {
            "holistic": holistic,
            "mixture": mixture,
            "entangled": entangled,
            "decision": decision_single,
            "decision_joint": decision_joint,
            "world": world,
            "parameters": dict(parameters),
        }

    def simulate(self, context: dict, engine) -> dict:
        decision, decision_joint, world = (
            context["decision"], context["decision_joint"], context["world"]
        )
        return {
            "unresolved": engine.run(context["holistic"], [decision]),
            "resolved_mixture": engine.run(context["mixture"], [decision]),
            "entangled_unknown": engine.run(context["entangled"], [decision_joint]),
            "entangled_known": engine.run(context["entangled"], [world, decision_joint]),
        }

    def analyse(self, context: dict, results: dict) -> tuple[dict, list[str], str]:
        unresolved = results["unresolved"].marginal("decision")["cooperate"]
        mixture = results["resolved_mixture"].marginal("decision")["cooperate"]
        entangled_unknown = results["entangled_unknown"].marginal("decision")["cooperate"]
        entangled_known = results["entangled_known"].marginal("decision")["cooperate"]
        guarantee = max(abs(entangled_unknown - mixture), abs(entangled_known - mixture))
        classical = ClassicalBaseline(
            name="sure-thing principle",
            description=(
                "The branch-weighted average of the conditional cooperation rates; also the exact prediction "
                "of any bipartite (world x action) state, entangled or not."
            ),
            prediction={"cooperation under an unknown opponent": mixture},
        )
        findings = [
            f"Cooperation when the opponent is unknown: {unresolved:.3f} (coherent representation) versus "
            f"{mixture:.3f} (branch mixture = sure-thing prediction). Difference {signed(unresolved - mixture)}.",
            f"The bipartite representation reproduces the sure-thing value to within {guarantee:.2e} both when "
            "the opponent's move is unknown and when it is revealed: no amount of entanglement between world "
            "and action can produce the effect.",
            "The relative-phase sweep moves the effect from constructive (more cooperation under uncertainty) "
            "through zero to destructive, which is the observable signature of coherence between the branches.",
            "Reading: the disjunction effect is not a failure of rationality to be patched by extra states - it "
            "is a constraint on how the uncertainty is represented.",
        ]
        summary = (
            f"Unknown opponent: {unresolved:.3f} versus the sure-thing prediction {mixture:.3f} "
            f"({signed(unresolved - mixture)}); the bipartite representation gives {entangled_unknown:.3f}."
        )
        analysis = {
            "classical": classical.to_dict(),
            "representation_guarantee": guarantee,
            "values": {
                "holistic unknown": unresolved,
                "branch mixture": mixture,
                "entangled unknown": entangled_unknown,
                "entangled after knowing": entangled_known,
            },
            "entanglement": entanglement_report(context["entangled"], label="world x action state"),
            "sweeps": {
                "relative phase": sweep(
                    "relative_phase",
                    [round(value, 3) for value in np.linspace(-PI, PI, 13)],
                    lambda value: _disjunction_metrics(context, float(value)),
                )
            },
        }
        return analysis, findings, summary

    def predict(self, parameters: dict | None = None) -> dict[str, float]:
        """Cooperation rate with a resolved, an unresolved and a bipartite representation."""
        values = self.defaults()
        values.update(parameters or {})
        context = self.build(**values)
        from ..core.engine import run_sequence

        holistic = run_sequence(context["holistic"], [context["decision"]])
        mixture = run_sequence(context["mixture"], [context["decision"]])
        entangled = run_sequence(context["entangled"], [context["decision"]])
        return {
            "p(cooperate | unresolved)": holistic.marginal("decision")["cooperate"],
            "p(cooperate | branch mixture)": mixture.marginal("decision")["cooperate"],
            "p(cooperate | bipartite state)": entangled.marginal("decision")["cooperate"],
        }

    def context_metadata(self, context: dict) -> dict:
        return {"representation_check": "bipartite vs holistic"}


def _disjunction_metrics(context: dict, phase: float) -> dict:
    """Cooperation under each representation as the relative phase is swept."""
    from ..core.engine import run_sequence

    parameters = dict(context["parameters"])
    parameters["relative_phase"] = phase
    rebuilt = DisjunctionEffect().build(**parameters)
    unresolved = run_sequence(rebuilt["holistic"], [rebuilt["decision"]])
    mixture = run_sequence(rebuilt["mixture"], [rebuilt["decision"]])
    return {
        "unresolved": unresolved.marginal("decision")["cooperate"],
        "sure-thing prediction": mixture.marginal("decision")["cooperate"],
        "difference": unresolved.marginal("decision")["cooperate"]
        - mixture.marginal("decision")["cooperate"],
    }

class ConceptCombination(Model):
    """Emergent and destroyed features in a combined concept.

    "Pet fish" is neither a typical pet nor a typical fish.  The model builds the
    combined concept by applying a combination operation to the two concepts and
    then asking feature questions of the result.  Features that are atypical in both
    parts can become typical in the whole (emergence), and typical features can be
    destroyed - the two effects the SCOP tradition reports.
    """

    slug = "concept-combination"
    title = "Concept combination: emergent and destroyed features"
    module = "B"
    one_liner = (
        "Combining two concepts with a unitary operation makes some features more typical than they are in "
        "either part, and others less - a joint structure the weighted-average baseline cannot produce."
    )
    cognitive_question = (
        "Why is a 'pet fish' neither a typical pet nor a typical fish, and why do combined concepts have "
        "features that neither part has?"
    )
    quantum_mechanism = (
        "A combination operation on the tensor product of two concepts; typicality is the projector of a "
        "feature question, and non-separability creates joint structure."
    )
    classical_baseline = (
        "Feature typicality of the combination as a weighted average of the typicalities of the parts "
        "(the standard extensional baseline)."
    )
    references = (
        "Aerts & Gabora (2005), A theory of concepts and their combinations, Kybernetes 34, 167-191.",
        "Hampton (1987), Inheritance of attributes in natural concept conjunctions, Memory & Cognition 15, 55-71.",
        "Blutner, Pothos & Bruza (2013), A quantum probability perspective on borderline vagueness, Topics in Cognitive Science 5, 711-736.",
    )
    caveats = default_caveats + (
        "Typicality ratings are modelled as probabilities of a feature question; the mapping from rating "
        "scales to probabilities is a modelling choice.",
    )
    keywords = ("concept combination", "emergence", "typicality", "SCOP", "non-separability")
    difficulty = "intermediate"

    params = (
        ParamSpec("pet_typicality", "float", 0.8, "Typicality of 'pet' features",
                  "Probability that a typical pet feature applies to the first concept.", 0.05, 0.95),
        ParamSpec("fish_typicality", "float", 0.25, "Typicality of 'fins' for 'fish'",
                  "How typical the feature is for the second concept alone.", 0.05, 0.95),
        ParamSpec("combination_strength", "float", 1.2, "Combination strength",
                  "Angle of the combination operation: 0 leaves the two concepts independent.", 0.0, PI),
        ParamSpec("feature", "choice", "fins", "Feature question",
                  "Which feature typicality to report on.", choices=("fins", "care", "colourful")),
    )

    def build(self, **parameters) -> dict:
        # qubit 0 = "needs care" (pet-flavoured), qubit 1 = "has fins" (fish-flavoured).
        # |0> means the feature is absent, |1> that it is present, so the state angle
        # encodes typicality directly and the feature questions are computational.
        pet_angle = 2.0 * np.arcsin(np.sqrt(parameters["pet_typicality"]))
        fish_angle = 2.0 * np.arcsin(np.sqrt(parameters["fish_typicality"]))
        pet_only = qubit(pet_angle, label="pet concept")
        fish_only = qubit(fish_angle, label="fish concept")
        # ``tensor`` uses numpy's Kronecker convention: the *first* argument is the
        # most significant qubit.  So the fish (feature B) factor comes first here, and
        # the pet factor ends up on qubit 0 (least significant).
        pet_state = tensor(basis_state(0, 2, label="no fish feature"), pet_only)
        fish_state = tensor(fish_only, basis_state(0, 2, label="no pet feature"))
        base = tensor(fish_only, pet_only)
        rotation = rotation_matrix(4, 1, 2, parameters["combination_strength"])
        combined = base.evolve(rotation)
        features = {
            "fins": local_question(0.0, 1, 2, "fins", ("no", "yes"), description="does it have fins?"),
            "care": local_question(0.0, 0, 2, "care", ("no", "yes"), description="does it need care?"),
            "colourful": local_question(PI / 2, 0, 2, "colourful", ("no", "yes"),
                                        description="is it colourful?"),
        }
        return {
            "pet_state": pet_state,
            "fish_state": fish_state,
            "combined": combined,
            "features": features,
            "rotation": rotation,
            "parameters": dict(parameters),
        }

    def simulate(self, context: dict, engine) -> dict:
        feature = context["features"][context["parameters"]["feature"]]
        return {
            "pet_alone": engine.run(context["pet_state"], [feature]),
            "fish_alone": engine.run(context["fish_state"], [feature]),
            "combined": engine.run(context["combined"], [feature]),
        }

    def analyse(self, context: dict, results: dict) -> tuple[dict, list[str], str]:
        feature = context["features"][context["parameters"]["feature"]]
        label = "yes"
        pet_alone = results["pet_alone"].marginal(feature.name)[label]
        fish_alone = results["fish_alone"].marginal(feature.name)[label]
        combined = results["combined"].marginal(feature.name)[label]
        average = 0.5 * (pet_alone + fish_alone)
        emergent = combined - max(pet_alone, fish_alone)
        report = entanglement_report(context["combined"], label="pet fish")
        classical = ClassicalBaseline(
            name="weighted-average typicality",
            description=(
                "Typicality of a feature for a combined concept as the weighted average of the typicalities "
                "for the parts; it can never exceed the maximum of the parts."
            ),
            prediction={"average of the parts": average, "maximum of the parts": max(pet_alone, fish_alone)},
        )
        findings = [
            f"Typicality of '{feature.name}' for the parts: {pet_alone:.3f} and {fish_alone:.3f}; for the "
            f"combination {combined:.3f} (their average is {average:.3f}).",
            f"Emergence: the combination exceeds the more typical part by {signed(emergent)}, which the "
            "weighted-average baseline forbids.",
            f"The combined state's entanglement is {report.entanglement:.4f} bits "
            f"(concurrence {report.concurrence:.4f}), so the joint structure is not a product of the parts.",
            "Features that are atypical in both parts but typical in the whole - and typical features that the "
            "combination destroys - are both produced by the same unitary, and the sweep over the combination "
            "strength moves between them.",
        ]
        summary = (
            f"'{feature.name}': parts {pet_alone:.2f}/{fish_alone:.2f} versus combination {combined:.2f} "
            f"(emergence {signed(emergent)})."
        )
        analysis = {
            "entanglement": report,
            "classical": classical.to_dict(),
            "typicality": {"pet": pet_alone, "fish": fish_alone, "combined": combined, "average": average,
                           "emergence": emergent},
            "sweeps": {
                "combination strength": sweep(
                    "combination_strength",
                    [round(value, 3) for value in np.linspace(0.0, PI, 12)],
                    lambda value: _combination_metrics(context, float(value)),
                )
            },
        }
        return analysis, findings, summary

    def context_metadata(self, context: dict) -> dict:
        return {"feature": context["parameters"]["feature"], "dim": context["combined"].dim}

def _combination_metrics(context: dict, strength: float) -> dict:
    """Feature typicalities and entanglement as the combination strength is swept."""
    from ..core.engine import run_sequence

    parameters = dict(context["parameters"])
    parameters["combination_strength"] = strength
    rebuilt = ConceptCombination().build(**parameters)
    feature = rebuilt["features"][parameters["feature"]]
    pet_alone = run_sequence(rebuilt["pet_state"], [feature]).marginal(feature.name)["yes"]
    fish_alone = run_sequence(rebuilt["fish_state"], [feature]).marginal(feature.name)["yes"]
    combined = run_sequence(rebuilt["combined"], [feature]).marginal(feature.name)["yes"]
    report = entanglement_report(rebuilt["combined"], label="combination")
    return {
        "combination": combined,
        "average of parts": 0.5 * (pet_alone + fish_alone),
        "maximum of parts": max(pet_alone, fish_alone),
        "concurrence": report.concurrence,
    }


class AssociativeMemory(OrderBasedModel):
    """Recall depends on the cue, and competing cues interfere.

    A memory trace is a state; a cue is a question put to it; recall is another
    question.  When two cues are similar, the two ways of reaching the same trace
    interfere, so the recall probability under *both* cues is not the sum of the
    two separate effects.  The mode switch chooses between the single-cue and the
    competing-cue narrative, both of which are the same two-question design.
    """

    slug = "associative-memory"
    title = "Associative memory and competing retrieval cues"
    module = "B"
    one_liner = (
        "Similar retrieval cues interfere, so presenting both cues moves recall away from the sum (or the "
        "average) of their separate effects."
    )
    cognitive_question = (
        "Why does adding a second, similar retrieval cue sometimes reduce rather than increase the chance of "
        "recall?"
    )
    quantum_mechanism = (
        "Cues as measurements with different bases: two similar cues are close in Hilbert space, so the "
        "paths they open to the same trace interfere."
    )
    classical_baseline = (
        "Independent-cue models: the probability of recall under both cues follows from the single-cue "
        "probabilities by the law of total probability."
    )
    references = (
        "Khrennikov (2006), Quantum-like brain: interference of minds, BioSystems 84, 225-241.",
        "Busemeyer & Bruza (2012), Quantum Models of Cognition and Decision, ch. 11 (memory).",
    )
    caveats = default_caveats + (
        "The model describes recall probabilities, not the neural mechanism of retrieval.",
    )
    keywords = ("memory", "recall", "cue interference", "similarity", "retrieval")
    difficulty = "intermediate"

    params = (
        ParamSpec("trace_angle", "float", 0.6, "Memory trace",
                  "Bloch angle of the stored trace: the strength of the encoded memory.", 0.0, PI),
        ParamSpec("cue_angle", "float", 0.7, "Cue angle",
                  "Angle of the retrieval cue relative to the trace.", 0.0, PI, fittable=True),
        ParamSpec("recall_angle", "float", PI / 2, "Recall question angle",
                  "Angle of the recall question; it is the question that can be interfered with.", 0.0, PI,
                  fittable=True),
        ParamSpec("mode", "choice", "competing cues", "Narrative",
                  "Which framing to report.", choices=("single cue", "competing cues")),
    )

    def make_experiment(self, **parameters) -> TwoQuestionExperiment:
        trace = qubit(parameters["trace_angle"], label="memory trace")
        cue = measurement_for(parameters["cue_angle"], "cue", ("yes", "no"), "retrieval cue")
        recall = measurement_for(parameters["recall_angle"], "recall", ("yes", "no"), "recall question")
        return TwoQuestionExperiment(trace, cue, recall, labels={"state": "trace", "a": "cue", "b": "recall"})

    def _framing(self, context: dict) -> str:
        """Which narrative this run reports (the design is the same either way)."""
        return str(context["parameters"].get("mode", "competing cues"))

    def extra_findings(self, context: dict, results: dict, bundle) -> list[str]:
        forward = results["A_then_B"]
        cue_first = forward.conditional("recall", {"cue": "yes"})["yes"]
        cue_absent = forward.marginal("recall")["yes"]
        if self._framing(context) == "single cue":
            return [
                f"With one cue, recall is {cue_first:.3f} against {cue_absent:.3f} unconditionally: the cue "
                "moves recall, and that is all a single retrieval path can say.",
                "The competing-cue framing reports the same numbers as interference between two retrieval "
                "paths; switch mode to 'competing cues' to read them that way.",
            ]
        return [
            f"Recall probability with the cue answered 'yes' is {cue_first:.3f}, against {cue_absent:.3f} "
            "unconditionally: the cue moves recall.",
            "Because the recall question is not compatible with the cue, the two retrieval paths interfere; "
            "the size of that interference is the number the library reports rather than the sum of the "
            "separate effects.",
        ]

    def extra_summary(self, context: dict, results: dict, bundle) -> str:
        forward = results["A_then_B"]
        cue_first = forward.conditional("recall", {"cue": "yes"})["yes"]
        cue_absent = forward.marginal("recall")["yes"]
        if self._framing(context) == "single cue":
            return (
                f"Single retrieval cue: recall {cue_first:.3f} with the cue, {cue_absent:.3f} without; the "
                "competing-cue reading of the same run is available as another mode."
            )
        return (
            f"Competing cues: recall {cue_first:.3f} with the cue versus {cue_absent:.3f} unconditionally; "
            "cue similarity drives the interference."
        )

    #: Sweeping the cue angle is the most informative single picture of this model.
    sweeps = (ParamSpec("cue_angle", "float", 0.7, "Cue angle", "", 0.0, PI, fittable=True),)

class SemanticAmbiguity(Model):
    """A word's sense is fixed by its context, and only before the question.

    A three-dimensional sense space, a context that rotates it, and a sense
    question.  The same word in two contexts is understood differently, and - the
    part a classical prior-updating account cannot produce - a sense that has
    already been resolved is immune to a later context.
    """

    slug = "semantic-ambiguity"
    title = "Semantic ambiguity: context fixes a word's sense"
    module = "B"
    one_liner = (
        "Sentence context rotates the meaning of an ambiguous word; once the sense has been resolved, the "
        "context can no longer change it."
    )
    cognitive_question = (
        "How does the surrounding sentence select one sense of an ambiguous word, and why does asking about "
        "the sense first lock the interpretation in?"
    )
    quantum_mechanism = (
        "Senses as basis states of a three-level system, context as a unitary transformation acting before "
        "the sense question is asked; the order matters because the sense question is a measurement."
    )
    classical_baseline = (
        "Context as prior re-weighting: the sense distribution is updated by the context, and the same "
        "posterior is available before or after any other judgement."
    )
    references = (
        "Kintsch (2001), Predication, Cognitive Science 25, 173-202.",
        "Blutner, Pothos & Bruza (2013), A quantum probability perspective on borderline vagueness, Topics in Cognitive Science 5, 711-736.",
    )
    caveats = default_caveats + (
        "Three senses and one rotation angle are a schematic; real lexical disambiguation needs a corpus.",
    )
    keywords = ("semantic ambiguity", "context", "sense selection", "order", "language")
    difficulty = "intermediate"

    params = (
        ParamSpec("sense_1_weight", "float", 0.7, "Weight of sense 1",
                  "Initial amplitude weight of the dominant sense.", 0.05, 0.95),
        ParamSpec("context_strength", "float", 0.9, "Context strength",
                  "How strongly the sentence context rotates the sense space.", 0.0, PI / 2),
        ParamSpec("context_scope", "choice", "narrow", "Context scope",
                  "Which pair of senses the context mixes.", choices=("narrow", "broad")),
    )

    def build(self, **parameters) -> dict:
        weight = float(parameters["sense_1_weight"])
        state = amplitudes(
            np.sqrt(weight), np.sqrt((1 - weight) / 2), np.sqrt((1 - weight) / 2),
            label="ambiguous word", basis_labels=("sense 1", "sense 2", "sense 3"),
        )
        pairs = (0, 1) if parameters["context_scope"] == "narrow" else (0, 2)
        context = rotation_matrix(3, pairs[0], pairs[1], parameters["context_strength"])
        context_reversed = rotation_matrix(3, pairs[0], pairs[1], -parameters["context_strength"])
        sense = ProjectiveMeasurement(
            "sense",
            ("sense 1", "sense 2", "sense 3"),
            tuple(
                np.diag([1.0 if index == position else 0.0 for index in range(3)]).astype(complex)
                for position in range(3)
            ),
            "which sense is intended?",
        )
        return {
            "state": state,
            "context": context,
            "context_reversed": context_reversed,
            "sense": sense,
            "parameters": dict(parameters),
        }

    def simulate(self, context: dict, engine) -> dict:
        state, sense = context["state"], context["sense"]
        return {
            "neutral": engine.run(state, [sense]),
            "context first": engine.run(state.evolve(context["context"]), [sense]),
            "opposite context first": engine.run(state.evolve(context["context_reversed"]), [sense]),
            "sense resolved first": engine.run(
                state, [sense, ProjectiveMeasurement("sense_again", sense.outcomes, sense.projectors,
                                                     "the same question asked again")]
            ),
        }

    def analyse(self, context: dict, results: dict) -> tuple[dict, list[str], str]:
        neutral = results["neutral"].marginal("sense")
        positive = results["context first"].marginal("sense")
        negative = results["opposite context first"].marginal("sense")
        pinned = results["sense resolved first"]
        dominant = largest_outcome(neutral)[0]
        shift = positive[dominant] - neutral[dominant]
        classical = ClassicalBaseline(
            name="context as prior re-weighting",
            description=(
                "A classical account updates the sense prior with the context; the posterior is available at "
                "any point, so asking about the sense first cannot make the context inert."
            ),
            prediction={"order invariance": 0.0},
        )
        pinned_shift = abs(
            pinned.conditional("sense_again", {"sense": dominant})[dominant] - 1.0
        )
        findings = [
            f"Without context the dominant sense '{dominant}' is chosen with probability {neutral[dominant]:.3f}; "
            f"the sentence context moves it to {positive[dominant]:.3f} ({signed(shift)}), and the opposite "
            f"context to {negative[dominant]:.3f}.",
            f"Once the sense has been resolved, the same question asked again reproduces it with probability "
            f"{pinned.conditional('sense_again', {'sense': dominant})[dominant]:.3f} (penalty {pinned_shift:.3f}): "
            "the interpretation is fixed and a later context cannot move it.",
            "A prior-updating account predicts that the context can be applied at any time with the same effect, "
            "so this ordering asymmetry is the model's testable content.",
        ]
        summary = (
            f"Context shifts the dominant sense by {signed(shift)}, and a resolved sense is immune to later "
            "context."
        )
        analysis = {
            "classical": classical.to_dict(),
            "sense_distributions": {"neutral": neutral, "context": positive, "opposite": negative},
            "incompatibility": float(1.0 - sum(positive[key] * neutral[key] for key in neutral)),
        }
        return analysis, findings, summary

    def context_metadata(self, context: dict) -> dict:
        return {"dim": context["state"].dim, "context_scope": context["parameters"]["context_scope"]}

class SequentialCategorization(Model):
    """Repeated categorisation stabilises the judgement.

    A sequence of labels is presented and the category judgement is read out each
    time.  Because each judgement is a measurement, repeated questioning makes the
    answer *more* stable rather than less - the quantum Zeno effect - while a
    classical model with independent judgements predicts a flat trajectory.
    """

    slug = "sequential-categorization"
    title = "Sequential categorisation: why repeated labels stabilise a judgement"
    module = "B"
    one_liner = (
        "Asking the same categorisation question repeatedly makes the answer more stable, not noisier; the "
        "model shows the trajectory and the classical baseline that predicts flatness."
    )
    cognitive_question = (
        "Does presenting several category labels in a row change how the category is judged, and in which "
        "direction?"
    )
    quantum_mechanism = (
        "Repeated projective or unsharp measurements of the same question: the state is driven towards the "
        "eigenstate of the answer, the quantum Zeno effect."
    )
    classical_baseline = (
        "Independence: each judgement is drawn from the same distribution, so the response trajectory is "
        "flat in expectation."
    )
    references = (
        "Busemeyer, Wang & Pothos (2015), Quantum models of cognition and decision, in The Oxford Handbook of Computational and Mathematical Psychology.",
        "Kvam, Pleskac, Yu & Busemeyer (2015), Interference effects of choice on confidence, PNAS 112, 10645-10650.",
    )
    caveats = default_caveats + (
        "The Zeno effect here is a property of repeated measurement; it does not require the labels to carry "
        "new information.",
        "The exact engine enumerates one path per label sequence (2 per label), so the parameter is capped at "
        "twelve labels; beyond that the sampled backend gives the same distribution with shot noise.",
    )
    keywords = ("categorisation", "repeated measurement", "Zeno effect", "stabilisation")
    difficulty = "intermediate"

    params = (
        ParamSpec("state_angle", "float", 0.5, "Initial belief angle",
                  "Bloch angle of the initial category belief.", 0.0, PI),
        ParamSpec("label_angle", "float", PI / 2, "Label question angle",
                  "Angle of the category question; pi/2 makes it maximally incompatible with the belief.", 0.0, PI),
        ParamSpec("num_labels", "int", 6, "Number of labels",
                  "How many category labels are presented in sequence. The exact engine enumerates one path "
                  "per label sequence, so the count is capped at twelve; longer sequences can use the "
                  "sampled backend.", 1, 12),
        ParamSpec("selection_strength", "float", 0.9, "Selection strength",
                  "Sharpness of each categorisation question (1 = fully decisive, 0 = uninformative).",
                  0.05, 1.0),
    )

    def build(self, **parameters) -> dict:
        state = qubit(parameters["state_angle"], label="category belief")
        count = int(parameters["num_labels"])
        questions = [
            measurement_for(parameters["label_angle"], f"label {index + 1}", ("yes", "no"),
                            f"category label {index + 1}", sharpness=parameters["selection_strength"])
            for index in range(count)
        ]
        return {"state": state, "questions": questions, "parameters": dict(parameters)}

    def simulate(self, context: dict, engine) -> dict:
        return {
            "sequence": engine.run(context["state"], context["questions"]),
            "single": engine.run(context["state"], [context["questions"][0]]),
        }

    def analyse(self, context: dict, results: dict) -> tuple[dict, list[str], str]:
        sequence = results["sequence"]
        questions = context["questions"]
        first_name = questions[0].name
        single = results["single"].marginal(first_name)["yes"]
        # After answering the first label "yes", how does the answer rate behave?
        anchored = [float(sequence.conditional(question.name, {first_name: "yes"})["yes"])
                    for question in questions]
        marginals = [float(sequence.marginal(question.name)["yes"]) for question in questions]
        agreement = float(
            sequence.marginal(first_name)["yes"] * anchored[1]
            + sequence.marginal(first_name)["no"] * sequence.conditional(questions[1].name, {first_name: "no"})["yes"]
        ) if len(questions) > 1 else float("nan")
        classical = ClassicalBaseline(
            name="independent judgements",
            description=(
                "Independent draws from the same distribution: consecutive answers agree with probability "
                "p^2 + (1-p)^2 and the answer rate stays at the initial value forever."
            ),
            prediction={
                "answer rate": single,
                "agreement between consecutive labels": single ** 2 + (1 - single) ** 2,
            },
        )
        stability = anchored[-1]
        findings = [
            f"After the first label has been answered 'yes', the same answer is reproduced with probability "
            f"{stability:.3f} on the last of {len(questions)} labels; the independent-judgement baseline is "
            f"{single:.3f}.",
            f"Consecutive labels agree with probability {agreement:.3f}, against the independent baseline "
            f"{single ** 2 + (1 - single) ** 2:.3f}: repeated questioning makes the judgement more stable, not "
            "noisier (a Zeno effect in a categorisation task).",
            "The unconditional answer rate stays flat, which is exactly why the stability has to be measured "
            "conditionally - a point the two series in the figure make visible.",
            "How much stabilisation appears is controlled by how decisive each label is: with weak labels the "
            "anchored series relaxes back towards the baseline, which is the classical limit.",
        ]
        summary = (
            f"Repeated labels reproduce the first answer with probability {stability:.3f} "
            f"(independent baseline {single:.3f})."
        )
        analysis = {
            "classical": classical.to_dict(),
            "stability": stability,
            "agreement": agreement,
            "trajectory": {
                "x": list(range(1, len(questions) + 1)),
                "xlabel": "label presented",
                "ylabel": "probability of answering 'yes'",
                "series": {
                    "after answering yes": anchored,
                    "unconditional": marginals,
                    "independent judgements": [single] * len(questions),
                },
            },
            "final_distribution": sequence.marginal(questions[-1].name),
        }
        return analysis, findings, summary

    def context_metadata(self, context: dict) -> dict:
        questions = len(context["questions"])
        return {
            "num_labels": questions,
            "reviewed_paths": 2**questions if questions <= 20 else "many",
            "cost note": (
                f"{questions} labels means the exact engine evaluates {2 ** questions} answer paths; "
                "twelve labels is the practical limit on a laptop"
                if questions <= 20
                else "this many labels is beyond exact enumeration; use the sampled backend"
            ),
        }

class ConceptualEntanglement(Model):
    """Joint structure that the parts do not contain.

    Two features are combined into a state of the form
    ``cos(theta)|both absent> + sin(theta)|both present>``.  The one-dimensional
    marginals reproduce the product prediction exactly, yet the joint probability
    of both features exceeds the product prediction by ``sin^2 cos^2`` - the whole
    of the entanglement, with concurrence ``|sin 2 theta|``.  Unlike the
    conjunction fallacy, the joint probability never exceeds the marginals, because
    commuting feature questions are still jointly answerable; what changes is the
    *correlation*.
    """

    slug = "conceptual-entanglement"
    title = "Conceptual entanglement: joint structure without interference"
    module = "B"
    one_liner = (
        "Two features can be perfectly correlated while each has the same marginal probability as in a "
        "product state: the whole carries structure the parts do not."
    )
    cognitive_question = (
        "How can a combined concept have joint structure - features that always co-occur - when neither part "
        "determines it?"
    )
    quantum_mechanism = (
        "Nonseparable (entangled) states: correlations in the joint distribution that no product state "
        "reproduces, quantified by concurrence and the entropy of entanglement."
    )
    classical_baseline = (
        "Statistical independence of the two features, p(A and B) = p(A) p(B), which is the best a product "
        "state can do."
    )
    references = (
        "Aerts & Gabora (2005), A theory of concepts and their combinations, Kybernetes 34, 167-191.",
        "Bruza, Kitto, Nelson & McEvoy (2009), Extracting spooky-activation-at-a-distance from considerations of entanglement, QI 2009.",
    )
    caveats = default_caveats + (
        "Entanglement here is a property of the modelled concept combination; it is not a claim that concepts "
        "are physically entangled.",
    )
    keywords = ("entanglement", "concepts", "correlation", "joint typicality", "non-separability")
    difficulty = "intermediate"

    params = (
        ParamSpec("combination_angle", "float", PI / 4, "Combination angle",
                  "pi/4 gives the maximally entangled combination; 0 or pi/2 is a product state.", 0.0, PI / 2),
        ParamSpec("decoherence", "float", 0.0, "Decoherence",
                  "Depolarising probability applied to both features (0 = coherent combination).", 0.0, 1.0),
    )

    def build(self, **parameters) -> dict:
        angle = float(parameters["combination_angle"])
        state = amplitudes(np.cos(angle), 0.0, 0.0, np.sin(angle), label="combined concept",
                           basis_labels=("neither", "A only", "B only", "both"))
        if float(parameters["decoherence"]) > 0:
            from ..core.instruments import per_qubit_channel

            channel = per_qubit_channel(_depolarizing_factory, 2, p=float(parameters["decoherence"]))
            state = state.apply_channel(channel.kraus)
        feature_a = local_question(0.0, 0, 2, "feature A", ("absent", "present"), description="is feature A present?")
        feature_b = local_question(0.0, 1, 2, "feature B", ("absent", "present"), description="is feature B present?")
        return {
            "state": state,
            "feature_a": feature_a,
            "feature_b": feature_b,
            "parameters": dict(parameters),
        }

    def simulate(self, context: dict, engine) -> dict:
        return {
            "joint": engine.run(context["state"], [context["feature_a"], context["feature_b"]]),
            "feature_a": engine.run(context["state"], [context["feature_a"]]),
            "feature_b": engine.run(context["state"], [context["feature_b"]]),
        }

    def analyse(self, context: dict, results: dict) -> tuple[dict, list[str], str]:
        joint = results["joint"]
        p_a = results["feature_a"].marginal("feature A")["present"]
        p_b = results["feature_b"].marginal("feature B")["present"]
        p_joint = joint.probability({"feature A": "present", "feature B": "present"})
        independence = p_a * p_b
        report = entanglement_report(context["state"], label="combined concept")
        classical = ClassicalBaseline(
            name="independent features",
            description="p(A and B) = p(A) p(B), the exact prediction of any product state.",
            prediction={"p(A and B)": independence},
        )
        findings = [
            f"Each feature alone is present with probability {p_a:.3f} and {p_b:.3f}; together they are present "
            f"with probability {p_joint:.3f}, against the independence prediction {independence:.3f} "
            f"(excess {signed(p_joint - independence)}).",
            f"Concurrence {report.concurrence:.3f} and entropy of entanglement "
            f"{report.entanglement:.3f} bits measure exactly this joint structure, and both are maximal at a "
            "combination angle of pi/4.",
            "The joint probability never exceeds the smaller marginal, unlike the conjunction fallacy: "
            "commuting feature questions are jointly answerable, so the effect is correlation rather than "
            "interference. The two mechanisms must not be conflated.",
            "Decoherence shrinks the correlation smoothly to zero, turning the combined concept back into a "
            "product of its parts.",
        ]
        summary = (
            f"Joint probability {p_joint:.3f} versus independence {independence:.3f} "
            f"(concurrence {report.concurrence:.2f})."
        )
        analysis = {
            "entanglement": report,
            "classical": classical.to_dict(),
            "joint_structure": {
                "p(A)": p_a,
                "p(B)": p_b,
                "p(A and B)": p_joint,
                "independence": independence,
                "excess": p_joint - independence,
            },
            "sweeps": {
                "combination angle": sweep(
                    "combination_angle",
                    [round(value, 3) for value in np.linspace(0.0, PI / 2, 11)],
                    lambda value: _entanglement_metrics(context, float(value)),
                )
            },
        }
        return analysis, findings, summary

    def context_metadata(self, context: dict) -> dict:
        return {"dim": context["state"].dim}


def _depolarizing_factory(p: float = 0.0):
    from ..core.instruments import depolarizing_channel

    return depolarizing_channel(p, 2)


def _entanglement_metrics(context: dict, angle: float) -> dict:
    """Correlation excess and concurrence as the combination angle is swept."""
    parameters = dict(context["parameters"])
    parameters["combination_angle"] = angle
    rebuilt = ConceptualEntanglement().build(**parameters)
    from ..core.engine import run_sequence

    report = entanglement_report(rebuilt["state"], label="combination")
    joint = run_sequence(rebuilt["state"], [rebuilt["feature_a"], rebuilt["feature_b"]])
    p_a = run_sequence(rebuilt["state"], [rebuilt["feature_a"]]).marginal("feature A")["present"]
    p_b = run_sequence(rebuilt["state"], [rebuilt["feature_b"]]).marginal("feature B")["present"]
    p_joint = joint.probability({"feature A": "present", "feature B": "present"})
    return {
        "p(A and B)": p_joint,
        "independence": p_a * p_b,
        "excess": p_joint - p_a * p_b,
        "concurrence": report.concurrence,
        "entropy of entanglement": report.entanglement,
    }

class ContradictoryBeliefs(OrderBasedModel):
    """Beliefs that cannot all be checked at once.

    A self-report question and a behavioural question are modelled as incompatible
    measurements on one belief state.  The model reports the size of the
    contradiction: the marginal of the self-report depends on whether the
    behavioural question was asked first, so no single joint distribution over the
    two answers can reproduce both orders.  The size of that violation is the
    quantitative content of "contradictory beliefs", rather than a diagnosis of the
    person.
    """

    slug = "contradictory-beliefs"
    title = "Contradictory beliefs and contextual measurements"
    module = "B"
    one_liner = (
        "A self-report and a behavioural question cannot be answered from one joint distribution: the model "
        "measures how far apart the two orders are."
    )
    cognitive_question = (
        "How can someone sincerely report one attitude and behave differently, without either report being a "
        "lie?"
    )
    quantum_mechanism = (
        "Incompatible (non-commuting) questions whose contexts cannot be jointly realised; a contextuality "
        "certificate replaces the assumption of a single joint distribution."
    )
    classical_baseline = (
        "Any classical account requires a joint distribution over self-report and behaviour, so the marginal "
        "of the report cannot depend on the order in which the two were elicited."
    )
    references = (
        "Bruza, Wang & Busemeyer (2015), Contextuality and probability in human cognition and decision, Phil. Trans. R. Soc. A 373.",
        "Wang, Solloway, Shiffrin & Busemeyer (2014), PNAS 111, 9431-9436.",
    )
    caveats = default_caveats + (
        "A contextuality certificate for the model is not a psychological diagnosis, and the model has no "
        "opinion about which of the two questions is the 'real' attitude.",
    )
    keywords = ("contradictory beliefs", "contextuality", "incompatible questions", "self-report")
    difficulty = "advanced"

    params = (
        ParamSpec("state_angle", "float", 0.0, "Belief angle",
                  "Where the state sits relative to the self-report question.", 0.0, PI),
        ParamSpec("report_angle", "float", 0.83, "Self-report question angle",
                  "Angle of the verbal self-report question.", 0.0, PI, fittable=True),
        ParamSpec("behaviour_angle", "float", PI / 2, "Behavioural question angle",
                  "Angle of the behavioural question; the gap to the report drives the contradiction.",
                  0.0, PI, fittable=True),
        ParamSpec("sharpness", "float", 1.0, "Question sharpness",
                  "How decisive each question is.", 0.0, 1.0),
    )

    sweeps = (ParamSpec("behaviour_angle", "float", PI / 2, "Behavioural question angle", "", 0.0, PI,
                        fittable=True),)

    def make_experiment(self, **parameters) -> TwoQuestionExperiment:
        state = qubit(parameters["state_angle"], label="belief")
        report = measurement_for(parameters["report_angle"], "self-report", ("yes", "no"),
                                 "what I say I value", sharpness=parameters["sharpness"])
        behaviour = measurement_for(parameters["behaviour_angle"], "behaviour", ("yes", "no"),
                                    "what I actually choose", sharpness=parameters["sharpness"])
        return TwoQuestionExperiment(state, report, behaviour,
                                     labels={"state": "belief", "a": "self-report", "b": "behaviour"})

    def extra_findings(self, context: dict, results: dict, bundle) -> list[str]:
        violation = max(abs(item.violation) for item in bundle.ltp)
        return [
            f"Contextuality certificate: the largest law-of-total-probability violation is {violation:.3f}. "
            "Any classical joint distribution over the two answers forces this number to be exactly 0.",
            f"The two orders are a total-variation distance of {bundle.report.total_variation:.3f} apart, which "
            "is the amount by which a single joint distribution would have to be wrong.",
            "The question sharpness sweep is the interesting comparison: turning both questions uninformative "
            "removes the contradiction and restores the classical picture.",
        ]

    def extra_summary(self, context: dict, results: dict, bundle) -> str:
        violation = max(abs(item.violation) for item in bundle.ltp)
        return (
            f"Self-report and behaviour differ by {violation:.3f} from what a joint distribution allows; "
            f"QQ difference {bundle.qq.difference:+.4f}."
        )

    def extra_analysis(self, context: dict, results: dict, bundle) -> dict:
        return {
            "contextuality_certificate": {
                "max_ltp_violation": max(abs(item.violation) for item in bundle.ltp),
                "total_variation_between_orders": bundle.report.total_variation,
                "incompatibility": bundle.report.incompatibility,
            }
        }

    def context_metadata(self, context: dict) -> dict:
        return {"incompatibility": round(context["experiment"].incompatibility, 6)}