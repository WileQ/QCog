"""Module A - individual cognition and decision making.

Five models about one person reasoning: what a belief state is, why question order
changes an answer, where interference shows up in a decision, how a sequence of
information shapes a choice, and how evidence revises a judgement.
"""

from __future__ import annotations

import numpy as np

from ..analysis.interference import interference_decomposition
from ..analysis.order import order_effect
from ..analysis.sweeps import sweep
from ..core.classical import ClassicalBaseline, law_of_total_probability
from ..core.instruments import NoiseSpec, depolarizing_channel, partial_dephasing_channel
from ..core.measurement import computational, unsharp_question
from ..core.params import ParamSpec
from ..core.states import amplitudes, maximally_mixed
from ..experiments.two_question import TwoQuestionExperiment
from .base import Model, largest_outcome
from .common import (
    classical_sentence,
    default_caveats,
    interference_sentence,
    measurement_for,
    order_effect_findings,
    percent,
    qubit,
    signed,
)

__all__ = [
    "SuperpositionBeliefs",
    "QuestionOrder",
    "InterferenceParadox",
    "DecisionTree",
    "BeliefRevision",
]

PI = float(np.pi)
class SuperpositionBeliefs(Model):
    """Genuine indeterminacy versus classical ignorance, made visible.

    A coherent superposition and an equal statistical mixture can give exactly the
    same answer to one question and completely different answers to another.  The
    model exhibits the pair: for ``|+>`` and for ``I/2`` the answer to "is it 0 or
    1?" is 50/50, but a question at right angles separates them perfectly.  It is
    the clearest demonstration that a response probability does not determine a
    person's state, and that a second, *different* question is needed to see the
    difference - asking the same question twice reveals nothing.
    """

    slug = "superposition-beliefs"
    title = "Superposition of beliefs versus classical uncertainty"
    module = "A"
    one_liner = (
        "A coherent belief and an equally uncertain but already settled belief give identical answers to one "
        "question and opposite answers to the next."
    )
    cognitive_question = (
        "When someone reports 'fifty-fifty', is the attitude genuinely unresolved, or settled but unknown? "
        "Response probabilities alone cannot tell the difference."
    )
    quantum_mechanism = (
        "Coherent superposition versus statistical mixture: coherence is invisible to the question whose "
        "answer distribution it defined, and visible to any question incompatible with it."
    )
    classical_baseline = (
        "A classical model identifies a 50/50 belief with the distribution (1/2, 1/2) and therefore predicts "
        "that no other question can distinguish the two situations."
    )
    references = (
        "Busemeyer & Bruza (2012), Quantum Models of Cognition and Decision, ch. 1-2.",
        "Pothos & Busemeyer (2013), Can quantum probability provide a new direction for cognitive modeling? BBS 36, 255-274.",
    )
    caveats = default_caveats + (
        "Coherence is a property of the model state; nothing here claims coherence has been measured in a person.",
        "With a single qubit, asking one sharp question erases exactly the coherence that a different sharp "
        "question would have revealed - which is why the model compares two separate questions rather than "
        "one question asked twice.",
    )
    keywords = ("superposition", "mixture", "coherence", "state", "probability")
    difficulty = "introductory"

    params = (
        ParamSpec("belief_angle", "float", PI / 2, "Belief angle",
                  "Bloch angle of the coherent belief: pi/2 is the balanced superposition.", 0.0, PI),
        ParamSpec("probe_angle", "float", PI / 2, "Probe question angle",
                  "Angle of the second question; pi/2 makes it maximally incompatible with the first.",
                  0.0, PI, fittable=True),
        ParamSpec("decoherence", "float", 0.15, "Decoherence",
                  "Depolarising probability applied to the coherent state (0 = pristine, 1 = fully mixed).",
                  0.0, 1.0),
    )

    def build(self, **parameters) -> dict:
        coherent = qubit(parameters["belief_angle"], label="coherent belief")
        return {
            "coherent": coherent,
            "decohered": coherent.apply_channel(
                depolarizing_channel(float(parameters["decoherence"]), 2).kraus
            ).with_label(
                "decohered beli"
                f"ef ({100 * float(parameters['decoherence']):.0f}% depolarised)"
            ),
            "ignorance": maximally_mixed(2, label="classical ignorance (I/2)"),
            "first": computational("Q1", ("0", "1"), "the question whose answer distribution is 50/50"),
            "probe": measurement_for(parameters["probe_angle"], "Q2", ("yes", "no"),
                                     "a second question at an angle to the first"),
        }

    def simulate(self, context: dict, engine) -> dict:
        first, probe = context["first"], context["probe"]
        return {
            "question_1": {
                "coherent": engine.run(context["coherent"], [first]),
                "decohered": engine.run(context["decohered"], [first]),
                "ignorance": engine.run(context["ignorance"], [first]),
            },
            "question_2": {
                "coherent": engine.run(context["coherent"], [probe]),
                "decohered": engine.run(context["decohered"], [probe]),
                "ignorance": engine.run(context["ignorance"], [probe]),
            },
            "sequential_check": engine.run(context["coherent"], [first, probe]),
        }

    def analyse(self, context: dict, results: dict) -> tuple[dict, list[str], str]:
        question_1 = results["question_1"]
        question_2 = results["question_2"]
        marginal_1 = question_1["coherent"].marginal("Q1")["0"]
        index = {
            name: question_2[name].marginal("Q2")["yes"]
            for name in ("coherent", "decohered", "ignorance")
        }
        base = index["ignorance"]
        distinguishability = index["coherent"] - base
        classical = ClassicalBaseline(
            name="classical ignorance model",
            description=(
                "A classical model that knows only the 50/50 answer distribution must predict the same "
                "distribution for every other question as well."
            ),
            prediction={"p(Q1=0)": 0.5, "p(Q2=yes)": 0.5},
        )
        sweep_result = sweep(
            "decoherence",
            [round(value, 2) for value in np.linspace(0.0, 1.0, 11)],
            lambda value: {"p(Q2=yes)": _probe_probability(context, float(value))},
        )
        sequential = results["sequential_check"]
        findings = [
            f"Both the coherent belief and the classical ignorance state answer Q1 with p(0) = "
            f"{marginal_1:.3f}: the marginal cannot tell them apart.",
            f"They are separated by Q2: p(Q2=yes) is {index['coherent']:.3f} for the coherent state, "
            f"{index['decohered']:.3f} after decoherence and {index['ignorance']:.3f} for the mixture "
            f"(distinguishability index {distinguishability:.3f}).",
            "The decoherence sweep turns the coherent state continuously into the ignorance state: the "
            "difference between the two is coherence, and nothing else.",
            "Asking Q1 first and Q2 second does not separate them, because the first sharp question erases "
            "the coherence; incompatibility has to be probed by asking the *other* question, not the same one again.",
        ]
        summary = (
            f"Same marginal, different states: p(Q2=yes) = {index['coherent']:.2f} (coherent) versus "
            f"{index['ignorance']:.2f} (classical ignorance)."
        )
        analysis = {
            "states": {
                "coherent": context["coherent"],
                "decohered": context["decohered"],
                "ignorance": context["ignorance"],
            },
            "distinguishability": index,
            "distinguishability_index": float(distinguishability),
            "sequential_check": sequential,
            "classical": classical.to_dict(),
            "sweeps": {"decoherence": sweep_result},
        }
        return analysis, findings, summary

    def context_metadata(self, context: dict) -> dict:
        return {"index": "difference in p(Q2=yes) between the coherent state and the equal mixture"}


def _probe_probability(context: dict, decoherence: float) -> float:
    """Probability of the probe question as a function of decoherence."""
    from ..core.engine import run_sequence

    state = context["coherent"].apply_channel(depolarizing_channel(decoherence, 2).kraus)
    return float(run_sequence(state, [context["probe"]]).marginal("Q2")["yes"])


class QuestionOrder(Model):
    """The flagship model: two questions, two orders, one falsifiable prediction.

    A state is prepared, two incompatible questions are asked, and the model
    reports the order effect, the violation of the law of total probability, the
    interference decomposition, and the QQ equality.  The QQ equality is the part
    that cannot be tuned: it says the probability of answering both questions the
    same way is identical in the two orders, with no free parameters.  That is what
    makes this model a test of the theory rather than a description of one study.
    """

    slug = "question-order"
    title = "Question-order effects and incompatible questions"
    module = "A"
    one_liner = (
        "Asking two questions in the two possible orders changes the first answer; the model predicts by how "
        "much, decomposes the change into interference, and satisfies a parameter-free QQ equality."
    )
    cognitive_question = (
        "Why does the answer to 'how satisfied are you with your life?' depend on whether 'how satisfied are "
        "you with your marriage?' was asked first?"
    )
    quantum_mechanism = (
        "Sequential projective (or unsharp) measurements of non-commuting questions: answering the first "
        "question projects the state, so the second question is put to a different belief state."
    )
    classical_baseline = (
        "A single joint distribution over responses (a classical sample space) forces the marginal of a "
        "question to be identical in both orders; any nonzero order effect rejects that family of models."
    )
    references = (
        "Wang & Busemeyer (2013), A quantum question order model supported by empirical tests of an a priori and precise prediction, Topics in Cognitive Science 5, 689-710.",
        "Wang, Solloway, Shiffrin & Busemeyer (2014), Context effects produced by question orders reveal quantum nature of human judgments, PNAS 111, 9431-9436.",
        "Moore (2002), Measuring new types of question-order effects, Public Opinion Quarterly 66, 80-91.",
    )
    caveats = default_caveats + (
        "The QQ equality requires two dichotomous questions; for more outcomes the model checks the "
        "corresponding multi-term identity numerically instead.",
    )
    keywords = ("question order", "non-commuting", "interference", "QQ equality", "contextuality")
    difficulty = "intermediate"

    params = (
        ParamSpec("state_angle", "float", 0.0, "Belief angle",
                  "Bloch angle of the initial belief state (0 = a definite answer to the first question).",
                  0.0, PI, fittable=True),
        ParamSpec("state_phase", "float", 0.0, "Belief phase",
                  "Relative phase of the belief state: it changes the interference term but not the first marginals.",
                  -PI, PI),
        ParamSpec("angle_a", "float", 0.955, "Question A angle",
                  "Angle of the first question; 0 would make it identical to the belief basis.",
                  0.0, PI, fittable=True),
        ParamSpec("angle_b", "float", PI / 2, "Question B angle",
                  "Angle of the second question; a large difference from A creates the effect.",
                  0.0, PI, fittable=True),
        ParamSpec("sharpness", "float", 1.0, "Question sharpness",
                  "1 = projective question, 0 = an uninformative question (the classical limit).", 0.0, 1.0),
        ParamSpec("decoherence", "float", 0.0, "Decoherence",
                  "Depolarising probability applied after each question, modelling forgetting or device noise.",
                  0.0, 1.0),
    )

    def build(self, **parameters) -> dict:
        state = qubit(parameters["state_angle"], parameters["state_phase"], label="belief")
        first = measurement_for(parameters["angle_a"], "A", ("yes", "no"), "first question")
        second = measurement_for(parameters["angle_b"], "B", ("yes", "no"), "second question",
                                 sharpness=parameters["sharpness"])
        first.metadata["theta"] = parameters["angle_a"]
        second.metadata["theta"] = parameters["angle_b"]
        experiment = TwoQuestionExperiment(
            state, first, second, labels={"state": "belief", "a": "A", "b": "B"}
        )
        return {"state": state, "experiment": experiment, "first": first, "second": second}

    def _noise(self, parameters):
        decoherence = float(parameters["decoherence"])
        return NoiseSpec("depolarizing", decoherence, 1) if decoherence > 0 else None

    def simulate(self, context: dict, engine) -> dict:
        return context["experiment"].run_both_orders(engine)

    def analyse(self, context: dict, results: dict) -> tuple[dict, list[str], str]:
        experiment = context["experiment"]
        bundle = experiment.analyse(results)
        sharpness_sweep = sweep(
            "sharpness",
            [round(value, 2) for value in np.linspace(0.05, 1.0, 11)],
            lambda value: {
                "order effect": _order_effect_at(context, float(value), "sharpness"),
                "interference": _interference_magnitude(context["state"], context["first"], context["second"]),
            },
        )
        decoherence_sweep = sweep(
            "decoherence",
            [round(value, 2) for value in np.linspace(0.0, 1.0, 11)],
            lambda value: {"order effect": _order_effect_at(context, float(value), "decoherence")},
        )
        findings = order_effect_findings(bundle)
        findings.append(classical_sentence(bundle))
        findings.append(
            "The sharpness sweep makes the classical limit explicit: as the questions become uninformative, "
            "both the order effect and the interference term go to zero, so the quantum structure comes from "
            "the questions being incompatible, not from the state alone."
        )
        summary = (
            f"Order effect {bundle.report.max_marginal_shift():.3f}; QQ difference "
            f"{signed(bundle.qq.difference)}; incompatibility {bundle.report.incompatibility:.3f}."
        )
        analysis = {
            "order_effect": bundle.report,
            "qq_equality": bundle.qq,
            "interference": bundle.interference,
            "ltp": bundle.ltp,
            "classical": bundle.classical,
            "bundle": bundle,
            "sweeps": {"sharpness": sharpness_sweep, "decoherence": decoherence_sweep},
        }
        return analysis, findings, summary

    def predict(self, parameters: dict | None = None) -> dict[str, float]:
        """Predicted response frequencies, in the format used by the fitting layer."""
        values = self.defaults()
        values.update(parameters or {})
        context = self.build(**values)
        from ..core.engine import Engine

        engine = Engine(backend="exact", noise=self._noise(values))
        results = self.simulate(context, engine)
        forward, backward = results["A_then_B"], results["B_then_A"]
        return {
            "AB: p(A=yes)": forward.marginal("A")["yes"],
            "AB: p(B=yes)": forward.marginal("B")["yes"],
            "AB: p(B=yes|A=yes)": forward.conditional("B", {"A": "yes"})["yes"],
            "AB: p(B=yes|A=no)": forward.conditional("B", {"A": "no"})["yes"],
            "BA: p(B=yes)": backward.marginal("B")["yes"],
            "BA: p(A=yes)": backward.marginal("A")["yes"],
            "BA: p(A=yes|B=yes)": backward.conditional("A", {"B": "yes"})["yes"],
            "BA: p(A=yes|B=no)": backward.conditional("A", {"B": "no"})["yes"],
            "QQ: agreement AB": forward.probability({"A": "yes", "B": "yes"})
            + forward.probability({"A": "no", "B": "no"}),
            "QQ: agreement BA": backward.probability({"A": "yes", "B": "yes"})
            + backward.probability({"A": "no", "B": "no"}),
        }

    def context_metadata(self, context: dict) -> dict:
        return {"incompatibility": round(context["experiment"].incompatibility, 6)}


def _order_effect_at(context: dict, value: float, kind: str) -> float:
    """Order-effect size at a given sharpness or decoherence level (used by sweeps)."""
    from ..core.engine import Engine

    if kind == "sharpness":
        state = context["state"]
        experiment = TwoQuestionExperiment(
            state,
            measurement_for(context["first"].metadata["theta"], "A", sharpness=value),
            measurement_for(context["second"].metadata["theta"], "B", sharpness=value),
        )
        engine = Engine(backend="exact")
    else:
        experiment = context["experiment"]
        engine = Engine(backend="exact", noise=NoiseSpec("depolarizing", value, 1))
    results = experiment.run_both_orders(engine)
    report = order_effect(results["A_then_B"], results["B_then_A"], "A", "B")
    return float(report.max_marginal_shift())


def _interference_magnitude(state, first, second) -> float:
    """Largest absolute interference term in the two-step design."""
    return float(interference_decomposition(state, first, second).max_abs_interference)

class InterferenceParadox(Model):
    """Distinguishing alternatives destroys the interference that decides them.

    A person weighs two alternatives that would lead to the same decision.  Asking
    *which* alternative applies should not change the decision - that is the
    sure-thing principle.  In the model it does: a recorded intermediate question
    dephases the state, and the interference term that was contributing to the
    decision disappears.  This is the mechanism behind the disjunction effect,
    shown as a difference between "alternatives combined" and "alternatives
    distinguished".
    """

    slug = "interference-paradox"
    title = "Interference between unresolved alternatives (disjunction effect)"
    module = "A"
    one_liner = (
        "Recording which of two alternatives applies removes the interference term that decided the outcome, "
        "so making a question 'more informative' changes the decision."
    )
    cognitive_question = (
        "Why do people who would accept a gamble conditional on winning and conditional on losing decline it "
        "while the outcome is still unknown (Tversky & Shafir's disjunction effect)?"
    )
    quantum_mechanism = (
        "Interference between the amplitudes of unresolved alternatives; a recorded intermediate question acts "
        "as a non-selective measurement that destroys the off-diagonal terms."
    )
    classical_baseline = (
        "The law of total probability / sure-thing principle requires "
        "p(decide yes) = p(win) p(yes|win) + p(lose) p(yes|lose) exactly."
    )
    references = (
        "Tversky & Shafir (1992), The disjunction effect in choice under uncertainty, Psychological Science 3, 305-309.",
        "Busemeyer & Bruza (2012), Quantum Models of Cognition and Decision, ch. 9.",
        "Kuperman & Perepelitsa (2016), A quantum version of the disjunction effect, Frontiers in Physics 4, 25.",
    )
    caveats = default_caveats + (
        "A violation of the sure-thing principle shows that respondents did not integrate both branches into "
        "one probability space; it does not by itself identify interference.",
    )
    keywords = ("interference", "disjunction effect", "sure-thing principle", "which path")
    difficulty = "intermediate"

    params = (
        ParamSpec("amplitude_angle", "float", PI / 4, "Alternative balance",
                  "How the belief splits between the two alternatives (pi/4 = equal amplitudes).", 0.0, PI / 2),
        ParamSpec("relative_phase", "float", 0.0, "Relative phase",
                  "Phase between the two alternatives: 0 and pi give maximal interference, pi/2 removes it.", -PI, PI),
        ParamSpec("visibility", "float", 0.0, "Which-path visibility",
                  "0 = alternatives combined, 1 = fully distinguished.", 0.0, 1.0),
        ParamSpec("decision_angle", "float", PI / 2, "Decision question angle",
                  "Angle of the final yes/no question relative to the alternatives.", 0.0, PI),
    )

    def build(self, **parameters) -> dict:
        alpha, phase = parameters["amplitude_angle"], parameters["relative_phase"]
        state = amplitudes(
            np.cos(alpha), np.exp(1j * phase) * np.sin(alpha), label="belief over two alternatives"
        )
        path = computational("which alternative", ("A", "B"), "asks which alternative applies")
        decision = measurement_for(parameters["decision_angle"], "decision", ("yes", "no"),
                                   "the yes/no decision")
        dephasing = partial_dephasing_channel(path.projectors, parameters["visibility"])
        return {
            "state": state,
            "path": path,
            "decision": decision,
            "dephasing": dephasing,
            "parameters": dict(parameters),
        }

    def simulate(self, context: dict, engine) -> dict:
        """Three conditions: paths combined, paths distinguished, and the requested partial case."""
        state, path, decision = context["state"], context["path"], context["decision"]
        fully = partial_dephasing_channel(path.projectors, 1.0)
        return {
            "combined": engine.run(state, [decision]),
            "distinguished": engine.run(state.apply_channel(fully.kraus), [decision]),
            "partial": engine.run(state.apply_channel(context["dephasing"].kraus), [decision]),
            "path_then_decision": engine.run(state, [path, decision]),
        }

    def analyse(self, context: dict, results: dict) -> tuple[dict, list[str], str]:
        combined, distinguished = results["combined"], results["distinguished"]
        partial = results["partial"]
        recorded = results["path_then_decision"]
        report = interference_decomposition(context["state"], context["path"], context["decision"])
        target = largest_outcome(combined.marginal("decision"))[0]
        classical = ClassicalBaseline(
            name="sure-thing principle (law of total probability)",
            description=(
                "If the decision is the same in both branches, the overall decision probability must equal "
                "the branch-weighted average of the conditional decisions."
            ),
            prediction={"p(decision=yes | combined)": float(distinguished.marginal("decision")["yes"])},
        )
        decision_name = context["decision"].name
        path_name = context["path"].name
        ltp = law_of_total_probability(
            decision_name,
            target,
            combined.marginal(decision_name)[target],
            recorded.marginal(path_name),
            recorded.conditional_table(path_name, decision_name),
            conditional_on=path_name,
        )
        shift = combined.marginal("decision")[target] - distinguished.marginal("decision")[target]
        findings = [
            f"With the alternatives combined, p(decision = {target}) = "
            f"{combined.marginal('decision')[target]:.3f}; with them fully distinguished it is "
            f"{distinguished.marginal('decision')[target]:.3f} (change {signed(shift)}), and with the "
            f"requested partial which-path visibility it is "
            f"{partial.marginal('decision')[target]:.3f}.",
            interference_sentence(report),
            f"Sure-thing check: the direct probability differs from the branch-weighted average by "
            f"{signed(ltp.violation)}, where the classical model requires exactly zero.",
            "The visibility parameter moves continuously between the two cases, which is why this is best "
            "read as a which-path experiment rather than as two parallel minds.",
        ]
        summary = (
            f"Interference supplies {percent(abs(report.interference[target]))} of p(decision = {target}); "
            f"distinguishing the alternatives shifts it by {signed(shift)}."
        )
        analysis = {
            "interference": report,
            "sure_thing": ltp,
            "classical": classical.to_dict(),
            "sweeps": {
                "visibility": sweep(
                    "visibility",
                    [round(value, 2) for value in np.linspace(0.0, 1.0, 11)],
                    lambda value: _visibility_metrics(context, float(value)),
                ),
                "relative phase": sweep(
                    "relative_phase",
                    [round(value, 3) for value in np.linspace(-PI, PI, 13)],
                    lambda value: {"interference": _phase_interference(context, float(value))},
                ),
            },
        }
        return analysis, findings, summary

    def context_metadata(self, context: dict) -> dict:
        return {"basis": "which-alternative question then a rotated decision question"}


def _phase_interference(context: dict, phase: float) -> float:
    """Largest interference term as the relative phase of the belief is varied."""
    parameters = context["parameters"]
    state = amplitudes(
        np.cos(parameters["amplitude_angle"]),
        np.exp(1j * phase) * np.sin(parameters["amplitude_angle"]),
        label="belief over two alternatives",
    )
    return _interference_magnitude(state, context["path"], context["decision"])


def _visibility_metrics(context: dict, visibility: float) -> dict:
    """Decision probability and interference strength as a function of visibility."""
    from ..core.engine import Engine

    state = context["state"]
    channel = partial_dephasing_channel(context["path"].projectors, visibility)
    resolved = state.apply_channel(channel.kraus)
    result = Engine(backend="exact").run(resolved, [context["decision"]])
    report = interference_decomposition(state, context["path"], context["decision"])
    target = largest_outcome(result.marginal("decision"))[0]
    return {
        "p(yes)": result.marginal("decision")["yes"],
        "target probability": result.marginal("decision")[target],
        "interference": report.max_abs_interference,
    }

class DecisionTree(Model):
    """Information order in a short sequence of decisions.

    Three cues arrive one after another and a final choice is read out.  Because
    the cues act as measurements, changing the order in which they arrive changes
    the final decision - and the model reports where the two orders diverge.  A
    classical decision tree with the same conditional probabilities is
    order-invariant, so the divergence is a quantum signature.
    """

    slug = "decision-tree"
    title = "Quantum decision tree: does the order of information matter?"
    module = "A"
    one_liner = (
        "Three cues, two arrival orders, one final decision: the model reports how much the decision depends "
        "on the order in which evidence arrived."
    )
    cognitive_question = (
        "Two people receive the same three pieces of evidence in different orders. Do they reach the same "
        "decision, and if not, which piece of evidence should have come first?"
    )
    quantum_mechanism = (
        "A sequence of measurements: each cue projects the belief state, so later cues act on a state that "
        "the earlier cues have already changed."
    )
    classical_baseline = (
        "Classical belief updating (Bayes' rule) is commutative: the posterior does not depend on the order "
        "in which the evidence arrived."
    )
    references = (
        "Busemeyer, Pothos, Franco & Trueblood (2011), A quantum theoretical explanation for probability judgment errors, Psychological Review 118, 193-218.",
        "Trueblood & Busemeyer (2011), A quantum probability account of order effects in inference, Cognitive Science 35, 1518-1552.",
    )
    caveats = default_caveats + (
        "Three cues are simulated; this demonstrates order dependence rather than calibrating a specific task.",
    )
    keywords = ("decision tree", "information order", "sequential measurement", "Bayes")
    difficulty = "advanced"

    params = (
        ParamSpec("state_angle", "float", 0.35, "Prior belief angle",
                  "Bloch angle of the prior belief about the decision.", 0.0, PI),
        ParamSpec("angle_cue_1", "float", 0.4, "Cue 1 angle", "Angle of the first cue.", 0.0, PI),
        ParamSpec("angle_cue_2", "float", PI / 2, "Cue 2 angle", "Angle of the second cue.", 0.0, PI),
        ParamSpec("angle_cue_3", "float", 2.4, "Cue 3 angle", "Angle of the third cue.", 0.0, PI),
        ParamSpec("sharpness", "float", 0.9, "Cue sharpness",
                  "How decisive each cue is (1 = fully decisive, 0 = uninformative).", 0.0, 1.0),
    )

    def build(self, **parameters) -> dict:
        state = qubit(parameters["state_angle"], label="prior belief")
        cues = {
            f"cue{index}": measurement_for(
                parameters[f"angle_cue_{index}"], f"cue{index}", ("positive", "negative"),
                f"evidence cue {index}", sharpness=parameters["sharpness"],
            )
            for index in (1, 2, 3)
        }
        return {"state": state, "cues": cues}

    def simulate(self, context: dict, engine) -> dict:
        cues = context["cues"]
        return {
            "forward": engine.run(context["state"], [cues["cue1"], cues["cue2"], cues["cue3"]]),
            "reversed": engine.run(context["state"], [cues["cue3"], cues["cue2"], cues["cue1"]]),
        }

    def analyse(self, context: dict, results: dict) -> tuple[dict, list[str], str]:
        forward, reversed_result = results["forward"], results["reversed"]
        final_forward = forward.marginal("cue3")
        final_reversed = reversed_result.marginal("cue1")
        divergence = float(abs(final_forward["positive"] - final_reversed["positive"]))
        classical = ClassicalBaseline(
            name="Bayesian decision tree",
            description=(
                "Sequential Bayesian updating multiplies likelihood ratios, and multiplication commutes, so "
                "the posterior after three cues is the same in both orders."
            ),
            prediction={"divergence": 0.0},
        )
        findings = [
            f"Order cue1 -> cue2 -> cue3 ends with p(positive) = {final_forward['positive']:.3f}; the reversed "
            f"order ends with {final_reversed['positive']:.3f}. Divergence = {divergence:.3f}.",
            "A Bayesian decision tree predicts a divergence of exactly zero for the same cues, so the whole "
            "effect comes from the non-commutativity of the cues.",
            "The divergence is largest when the most incompatible cue is asked of the state it least agrees "
            "with, which the per-cue tables show directly.",
            "Practical reading: when a decision is order-sensitive, the order that asks the most incompatible "
            "question last is the one that preserves the earlier judgements.",
        ]
        summary = f"Information order changes the final decision by {divergence:.3f}."
        table = [
            {
                "order": "cue1 -> cue2 -> cue3",
                "after first cue": round(forward.marginal("cue1")["positive"], 6),
                "after second cue": round(forward.conditional("cue2", {"cue1": "positive"})["positive"], 6),
                "final": round(final_forward["positive"], 6),
            },
            {
                "order": "cue3 -> cue2 -> cue1",
                "after first cue": round(reversed_result.marginal("cue3")["positive"], 6),
                "after second cue": round(
                    reversed_result.conditional("cue2", {"cue3": "positive"})["positive"], 6
                ),
                "final": round(final_reversed["positive"], 6),
            },
        ]
        analysis = {"classical": classical.to_dict(), "order_table": table, "divergence": divergence}
        return analysis, findings, summary

    def context_metadata(self, context: dict) -> dict:
        return {"num_cues": 3}

class BeliefRevision(Model):
    """How evidence revises a judgement, and how framing changes the revision.

    Evidence is modelled as an unsharp question: weak evidence moves the state a
    little, strong evidence moves it a lot, and repeating weak evidence converges.
    The model compares that trajectory with classical Bayesian updating of the same
    likelihoods, and shows that identically informative evidence produces different
    judgements when the *framing* of the belief state differs - something Bayesian
    updating, which depends only on the likelihoods, cannot produce.
    """

    slug = "belief-revision"
    title = "Ambiguity, evidence accumulation and framing"
    module = "A"
    one_liner = (
        "Repeated weak evidence moves a belief along a trajectory, and identically informative evidence given "
        "under a different framing lands somewhere else."
    )
    cognitive_question = (
        "Does the way a belief is framed change what the same evidence does to it, and does evidence "
        "accumulate the way Bayes' rule says it should?"
    )
    quantum_mechanism = (
        "Unsharp (POVM) measurements for weak evidence, repeated state updates for accumulation, and a "
        "unitary framing transformation that does not commute with the evidence."
    )
    classical_baseline = (
        "Bayesian updating: the posterior depends on the likelihood ratio and the prior, not on how the "
        "question was framed."
    )
    references = (
        "Busemeyer & Bruza (2012), Quantum Models of Cognition and Decision, ch. 10.",
        "Trueblood & Busemeyer (2012), A quantum probability model of causal reasoning, Frontiers in Psychology 3, 138.",
    )
    caveats = default_caveats + (
        "The evidence operator models 'how much this observation moves the judgement', not a statistical likelihood.",
    )
    keywords = ("belief revision", "updating", "POVM", "framing", "ambiguity")
    difficulty = "advanced"

    params = (
        ParamSpec("prior_angle", "float", PI / 2, "Prior belief angle",
                  "Bloch angle of the prior judgement.", 0.0, PI),
        ParamSpec("evidence_angle", "float", 0.6, "Evidence angle",
                  "Which proposition the evidence bears on.", 0.0, PI),
        ParamSpec("evidence_strength", "float", 0.8, "Evidence strength",
                  "Sharpness of the evidence measurement: 0.2 is weak evidence, 1 is conclusive.",
                  0.05, 1.0),
        ParamSpec("framing_angle", "float", 1.1, "Framing rotation",
                  "How much the framing of the question rotates the belief state before the evidence arrives.",
                  -PI, PI),
        ParamSpec("repetitions", "int", 8, "Evidence repetitions",
                  "How many times the same weak evidence is presented.", 1, 40),
    )

    def build(self, **parameters) -> dict:
        from ..circuits.gates import rotation_y

        prior = qubit(parameters["prior_angle"], label="prior judgement")
        evidence = unsharp_question(
            parameters["evidence_angle"], parameters["evidence_strength"], 0.0, "evidence",
            ("supports", "contradicts"), "observed evidence",
        )
        return {
            "prior": prior,
            "evidence": evidence,
            "framing": rotation_y(parameters["framing_angle"]),
            "query": measurement_for(PI / 2, "query", ("yes", "no"), "the judgement question"),
        }

    def simulate(self, context: dict, engine) -> dict:
        prior, evidence, query = context["prior"], context["evidence"], context["query"]
        framed = prior.evolve(context["framing"])
        return {
            "evidence_first": engine.run(prior, [evidence]),
            "framed_evidence": engine.run(framed, [evidence]),
            "framed_then_query": engine.run(framed, [evidence, query]),
            "unframed_then_query": engine.run(prior, [evidence, query]),
        }

    def analyse(self, context: dict, results: dict) -> tuple[dict, list[str], str]:
        prior, evidence = context["prior"], context["evidence"]
        framed = prior.evolve(context["framing"])
        trajectories = _accumulation_trajectories(prior, framed, evidence, 8)
        shift = float(
            results["framed_evidence"].marginal("evidence")["supports"]
            - results["evidence_first"].marginal("evidence")["supports"]
        )
        judgement_gap = float(
            results["framed_then_query"].conditional("query", {"evidence": "supports"})["yes"]
            - results["unframed_then_query"].conditional("query", {"evidence": "supports"})["yes"]
        )
        classical = ClassicalBaseline(
            name="Bayesian updating",
            description=(
                "With likelihoods fixed by the evidence strength, Bayes' rule predicts the same posterior for "
                "both framings, because framing is not part of a classical likelihood."
            ),
            prediction={"framing effect": 0.0},
        )
        findings = [
            f"The framing rotation changes the immediate response to the evidence by {signed(shift)}.",
            f"After the same evidence the follow-up judgement differs by {signed(judgement_gap)} between the "
            "two framings, where a likelihood-based classical model predicts exactly 0.",
            "Repeated weak evidence converges geometrically to the evidence eigenstate, but the path depends "
            "on the framing: the same number of observations can leave two people at different beliefs.",
            "The convergence is not a learning curve: no new information arrives, the state is projected "
            "again and again.",
        ]
        summary = f"Framing changes the effect of identical evidence by {signed(judgement_gap)}."
        analysis = {
            "classical": classical.to_dict(),
            "trajectory": trajectories,
            "framing_effect": {"immediate": shift, "judgement": judgement_gap},
        }
        return analysis, findings, summary

    def context_metadata(self, context: dict) -> dict:
        return {"evidence_model": "unsharp POVM", "framing_model": "unitary rotation"}


def _accumulation_trajectories(prior, framed, evidence, repetitions: int) -> dict:
    """Belief trajectories for repeated presentation of the same weak evidence."""
    outcomes = list(evidence.outcomes)
    target = outcomes[0]
    series: dict[str, list[float]] = {}
    for label, state in (("unframed", prior), ("framed", framed)):
        current = state
        values: list[float] = []
        for _ in range(int(repetitions)):
            current = evidence.post_state(current, target)
            values.append(1.0 - float(np.real(current.density[0, 0])))
        series[label] = values
    strength = float(evidence.metadata.get("sharpness", 0.5))
    likelihood = 0.5 + 0.5 * strength
    posterior = 0.5
    classical: list[float] = []
    for _ in range(int(repetitions)):
        posterior = (posterior * likelihood) / (
            posterior * likelihood + (1 - posterior) * (1 - likelihood)
        )
        classical.append(float(1.0 - posterior))
    series["classical Bayes"] = classical
    return {
        "x": list(range(1, int(repetitions) + 1)),
        "xlabel": "number of presentations of the same evidence",
        "ylabel": "probability of the second answer",
        "series": series,
    }