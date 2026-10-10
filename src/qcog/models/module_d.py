"""Module D - groups, social networks and collective behaviour.

Five models about more than one person: how opinions move through a network, how
polarisation appears and can be undone, whether aggregating many judgements keeps
the "wisdom of crowds", whether information cascades form, and whether deliberation
changes a vote.

The architecture is hybrid on purpose.  Classical graph structures carry *who
interacts with whom*; quantum states carry what each agent believes and how the
interaction changes it.  Encoding every agent and edge in one enormous circuit
would be possible but uninterpretable, so the library does not do it.
"""

from __future__ import annotations

import numpy as np

from ..analysis.sweeps import sweep
from ..core.classical import (
    ClassicalBaseline,
    bayesian_cascade,
    bounded_confidence_step,
    condorcet_jury_probability,
    de_groot_step,
)
from ..core.params import ParamSpec
from ..core.states import amplitudes
from ..experiments.population import (
    QuantumOpinionModel,
    independent_vote_distribution,
    topology_adjacency,
    vote_distribution,
)
from .base import Model
from .common import default_caveats, local_question, measurement_for, qubit, signed
from .shared import PopulationModel

__all__ = [
    "QuantumOpinionDynamics",
    "Polarization",
    "CollectiveJudgment",
    "InformationCascades",
    "DeliberationVoting",
]

PI = float(np.pi)


def _initial_opinions(num_agents: int, spread: float, bias: float = 0.0) -> list[float]:
    """Initial Bloch angles: ``bias`` shifts the group, ``spread`` its diversity."""
    offsets = np.linspace(-1.0, 1.0, num_agents) if num_agents > 1 else np.array([0.0])
    angles = (PI / 2 + bias) + spread * offsets * PI / 2
    return [float(np.clip(angle, 0.02, PI - 0.02)) for angle in angles]


class QuantumOpinionDynamics(PopulationModel):
    """Opinions move by coherent exchange along a network.

    Each agent's opinion is a qubit; interacting agents apply a partial swap, which
    is a *coherent* exchange of amplitude rather than an average.  The model
    reports the opinion trajectory, the consensus/polarisation summary, and how much
    superposition and correlation the group builds up - none of which a classical
    averaging rule can produce.
    """

    slug = "quantum-opinion-dynamics"
    title = "Quantum opinion dynamics on a network"
    module = "D"
    one_liner = (
        "Agents exchange amplitude coherently along a network: opinions converge, agendas become undecided, "
        "and the group develops correlations that an averaging model cannot represent."
    )
    cognitive_question = (
        "How do opinions spread through a group when each encounter is an exchange rather than an average?"
    )
    quantum_mechanism = (
        "Partial-swap interaction on pairs of qubits: a coherent exchange that creates superposition, plus "
        "decoherence as forgetting."
    )
    classical_baseline = (
        "DeGroot averaging on the same graph: opinions converge to a weighted average and the final spread is "
        "the consensus value."
    )
    references = (
        "DeGroot (1974), Reaching a consensus, Journal of the American Statistical Association 69, 118-121.",
        "Khrennikov (2006), Quantum-like brain: interference of minds, BioSystems 84, 225-241.",
    )
    caveats = default_caveats + (
        "The dense simulation limits the population to ten agents; larger networks need the classical "
        "baselines or a tensor-network implementation.",
    )
    keywords = ("opinion dynamics", "partial swap", "consensus", "network", "coherence")
    difficulty = "advanced"

    params = (
        ParamSpec("num_agents", "int", 5, "Number of agents", "Population size (2 to 10).", 2, 10),
        ParamSpec("topology", "choice", "ring", "Network topology", "Who talks to whom.",
                  choices=("ring", "line", "complete", "star", "two-communities", "random")),
        ParamSpec("initial_spread", "float", 1.0, "Initial diversity",
                  "How far apart the agents' initial opinions are.", 0.0, 1.4),
        ParamSpec("interaction_strength", "float", PI / 4, "Interaction strength",
                  "Angle of the partial swap: how much of each opinion is exchanged per encounter.", 0.0, PI / 2),
        ParamSpec("decoherence", "float", 0.03, "Decoherence",
                  "Per-agent depolarising probability per round (forgetting).", 0.0, 0.5),
        ParamSpec("rounds", "int", 8, "Rounds", "Number of interaction rounds.", 1, 40),
    )

    def build(self, **parameters) -> dict:
        opinions = _initial_opinions(int(parameters["num_agents"]), float(parameters["initial_spread"]))
        model = QuantumOpinionModel(
            opinions,
            topology=parameters["topology"],
            interaction_strength=float(parameters["interaction_strength"]),
            decoherence=float(parameters["decoherence"]),
        )
        return {"model": model, "opinions": opinions, "parameters": dict(parameters)}

    def simulate(self, context: dict, engine) -> dict:
        outcome = context["model"].run(int(context["parameters"]["rounds"]))
        return {"opinions": outcome, "votes": context["model"].vote_distribution()}

    def analyse(self, context: dict, results: dict) -> tuple[dict, list[str], str]:
        model = context["model"]
        history = results["opinions"]["history"]
        final = results["opinions"]["final"]
        metrics = model.consensus_metrics()
        classical_trajectory = self.classical_trajectory(
            context["opinions"], model.adjacency, len(history) - 1, kind="de-groot"
        )
        classical_final = np.array(
            [classical_trajectory["series"][f"agent {index + 1}"][-1]
             for index in range(len(context["opinions"]))]
        )
        classical = ClassicalBaseline(
            name="DeGroot averaging on the same network",
            description=(
                "Every agent replaces its opinion with the average of its neighbours, so the group converges "
                "to one value and no agent is ever undecided."
            ),
            prediction={"final consensus value": float(classical_final.mean())},
        )
        entanglement_proxy = float(np.mean(results["opinions"]["entropy"][1:]))
        findings = [
            f"Opinions move from a spread of {np.std(history[0]):.3f} to {np.std(final):.3f}; agreement "
            f"(|mean|) is {metrics['agreement (|mean|)']:.3f} and the number of clusters is {metrics['clusters']}.",
            f"The classical DeGroot baseline on the same graph converges to a consensus value of "
            f"{classical_final.mean():.3f} with a spread of {classical_final.std():.3f}.",
            f"The average local entropy rises to {entanglement_proxy:.3f} bits: agents end up in superpositions "
            "of positions, which is what a coherent exchange does and an averaging rule cannot.",
            "Decoherence is the classical limit: with enough forgetting the trajectory approaches the "
            "averaging baseline, because the correlations and superpositions are destroyed.",
        ]
        summary = (
            f"Spread {np.std(history[0]):.3f} -> {np.std(final):.3f} over {len(history) - 1} rounds; "
            f"agreement {metrics['agreement (|mean|)']:.2f}; mean local entropy {entanglement_proxy:.2f} bits."
        )
        analysis = {
            "classical": classical.to_dict(),
            "trajectory": self.trajectory_analysis(history, label="opinion (Bloch z)"),
            "classical_trajectory": classical_trajectory,
            "metrics": metrics,
            "votes": results["votes"],
            "sweeps": {
                "decoherence": sweep(
                    "decoherence",
                    [round(value, 2) for value in np.linspace(0.0, 0.5, 9)],
                    lambda value: _opinion_metrics(context, float(value), "decoherence"),
                ),
                "interaction_strength": sweep(
                    "interaction_strength",
                    [round(value, 3) for value in np.linspace(0.0, PI / 2, 9)],
                    lambda value: _opinion_metrics(context, float(value), "strength"),
                ),
            },
        }
        return analysis, findings, summary

    def context_metadata(self, context: dict) -> dict:
        return {"num_agents": int(context["parameters"]["num_agents"]),
                "topology": context["parameters"]["topology"]}


def _opinion_metrics(context: dict, value: float, kind: str) -> dict:
    """Polarisation and entropy at the end of a run with one parameter changed."""
    parameters = dict(context["parameters"])
    if kind == "decoherence":
        parameters["decoherence"] = value
    else:
        parameters["interaction_strength"] = value
    rebuilt = QuantumOpinionDynamics().build(**parameters)
    outcome = rebuilt["model"].run(int(parameters["rounds"]))
    final = np.asarray(outcome["final"], dtype=float)
    return {
        "polarisation": float(final.std()),
        "agreement": float(abs(final.mean())),
        "mean local entropy": float(np.mean(outcome["entropy"][1:])),
    }

class Polarization(PopulationModel):
    """Polarisation, and what a coherent exchange does to it.

    Two communities start apart and interact mostly inside themselves, with a weak
    bridge.  The model reports the final opinion distribution, the within- and
    between-community gaps, and the extent to which agents end up in superpositions
    - which is the difference between "a group that agrees" and "a group whose
    members have lost their positions".  The classical bounded-confidence
    (Hegselmann-Krause) model on the same graph is reported alongside it.
    """

    slug = "polarization"
    title = "Polarisation and information bubbles"
    module = "D"
    one_liner = (
        "Two communities with mostly internal interaction can converge inside and diverge between - or melt "
        "into undecided superpositions, depending on how coherent the exchange is."
    )
    cognitive_question = (
        "Why do groups with the same information drift apart, and what changes when the exchange between them "
        "is coherent rather than an average?"
    )
    quantum_mechanism = (
        "Partial-swap exchange restricted by network structure, with decoherence as the mechanism that "
        "converts coherent uncertainty into a stable position."
    )
    classical_baseline = (
        "Hegselmann-Krause bounded confidence on the same graph, which preserves clusters, and DeGroot "
        "averaging, which erases them."
    )
    references = (
        "Hegselmann & Krause (2002), Opinion dynamics and bounded confidence, JASSS 5(3).",
        "DeGroot (1974), Reaching a consensus, JASA 69, 118-121.",
    )
    caveats = default_caveats + (
        "The model says nothing about which side is right and treats the initial positions as given.",
    )
    keywords = ("polarisation", "echo chamber", "bounded confidence", "networks", "consensus")
    difficulty = "advanced"

    params = (
        ParamSpec("num_agents", "int", 6, "Number of agents", "Even population split into two communities.", 4, 10),
        ParamSpec("community_gap", "float", 1.2, "Community gap",
                  "How far apart the two communities start.", 0.2, 1.5),
        ParamSpec("interaction_strength", "float", PI / 4, "Within-community interaction",
                  "Partial-swap strength inside each community.", 0.0, PI / 2),
        ParamSpec("cross_talk", "float", 0.15, "Cross-community link",
                  "Weight of the single bridge between the communities.", 0.0, 1.0),
        ParamSpec("decoherence", "float", 0.0, "Decoherence",
                  "Forgetting per round; higher values make positions settle.", 0.0, 0.5),
        ParamSpec("rounds", "int", 10, "Rounds", "Number of interaction rounds.", 1, 40),
    )

    def build(self, **parameters) -> dict:
        num_agents = int(parameters["num_agents"])
        half = num_agents // 2
        gap = float(parameters["community_gap"])
        opinions = [PI / 2 - gap / 2] * half + [PI / 2 + gap / 2] * (num_agents - half)
        adjacency = topology_adjacency(num_agents, "two-communities")
        adjacency[half - 1, half] = adjacency[half, half - 1] = max(
            0.0, float(parameters["cross_talk"])
        )
        model = QuantumOpinionModel(
            opinions,
            interaction_strength=float(parameters["interaction_strength"]),
            decoherence=float(parameters["decoherence"]),
            adjacency=adjacency,
        )
        return {
            "model": model,
            "opinions": opinions,
            "adjacency": adjacency,
            "half": half,
            "parameters": dict(parameters),
        }

    def simulate(self, context: dict, engine) -> dict:
        outcome = context["model"].run(int(context["parameters"]["rounds"]))
        return {"opinions": outcome, "votes": context["model"].vote_distribution()}

    def analyse(self, context: dict, results: dict) -> tuple[dict, list[str], str]:
        model = context["model"]
        half = context["half"]
        history = results["opinions"]["history"]
        final = np.asarray(results["opinions"]["final"], dtype=float)
        within_a = float(final[:half].mean())
        within_b = float(final[half:].mean())
        gap_final = abs(within_b - within_a)
        gap_initial = abs(np.mean(context["opinions"][half:]) - np.mean(context["opinions"][:half]))
        bounded = np.asarray(context["opinions"], dtype=float)
        bounded_history = [bounded.copy()]
        for _ in range(len(history) - 1):
            bounded = bounded_confidence_step(bounded, 0.35, context["adjacency"])
            bounded_history.append(bounded.copy())
        final_entropy = float(np.mean(results["opinions"]["entropy"][1:]))
        classical = ClassicalBaseline(
            name="bounded confidence (Hegselmann-Krause)",
            description=(
                "Agents only listen to opinions within a confidence bound, which preserves clusters. On this "
                "graph the two communities stay separate: the classical predictor of persistent polarisation."
            ),
            prediction={
                "community gap (classical)": float(abs(np.mean(bounded_history[-1][half:]) - np.mean(bounded_history[-1][:half]))),
                "initial gap": float(gap_initial),
            },
        )
        findings = [
            f"The gap between the community means moves from {gap_initial:.3f} to {gap_final:.3f}; within-"
            f"community agreement is {within_a:+.3f} (community 1) and {within_b:+.3f} (community 2).",
            f"The bounded-confidence baseline keeps a gap of "
            f"{classical.prediction['community gap (classical)']:.3f}, so the two models make different "
            "predictions about the same network - which is the point of reporting both.",
            f"The average local entropy is {final_entropy:.3f} bits: coherent exchange leaves agents in "
            "superpositions, so agreement between their answers can hide disagreement inside their states.",
            "The cross-talk sweep is the levers-on-the-model version of the question: a single bridge changes "
            "the outcome by a computable amount, and decoherence decides whether the change sticks.",
        ]
        summary = (
            f"Community gap {gap_initial:.2f} -> {gap_final:.2f}; mean local entropy {final_entropy:.2f} bits."
        )
        analysis = {
            "classical": classical.to_dict(),
            "trajectory": self.trajectory_analysis(history, label="opinion (Bloch z)"),
            "classical_trajectory": {
                "x": list(range(len(bounded_history))),
                "xlabel": "interaction step",
                "ylabel": "opinion (bounded confidence)",
                "series": {
                    f"agent {index + 1}": [float(row[index]) for row in bounded_history]
                    for index in range(len(context["opinions"]))
                },
            },
            "communities": {
                "within community 1": within_a,
                "within community 2": within_b,
                "gap": gap_final,
                "gap (bounded confidence)": classical.prediction["community gap (classical)"],
            },
            "metrics": model.consensus_metrics(),
            "sweeps": {
                "cross talk": sweep(
                    "cross_talk",
                    [round(value, 2) for value in np.linspace(0.0, 1.0, 9)],
                    lambda value: _polarization_metrics(context, float(value)),
                )
            },
        }
        return analysis, findings, summary

    def context_metadata(self, context: dict) -> dict:
        return {"num_agents": len(context["opinions"]), "topology": "two communities"}


def _polarization_metrics(context: dict, cross_talk: float) -> dict:
    """Community gap and entropy as the cross-community link is strengthened."""
    parameters = dict(context["parameters"])
    parameters["cross_talk"] = cross_talk
    rebuilt = Polarization().build(**parameters)
    outcome = rebuilt["model"].run(int(parameters["rounds"]))
    final = np.asarray(outcome["final"], dtype=float)
    half = rebuilt["half"]
    return {
        "community gap": float(abs(final[half:].mean() - final[:half].mean())),
        "polarisation": float(final.std()),
        "mean local entropy": float(np.mean(outcome["entropy"][1:])),
    }


class CollectiveJudgment(PopulationModel):
    """When does a crowd stop being wise?

    ``n`` agents each hold an opinion with the same individual accuracy.  If they
    are independent, Condorcet's jury theorem says the majority is much more
    accurate than any individual; if their states are correlated, the crowd
    inherits the individual accuracy and no more.  The model interpolates between
    the two and reports the majority-accuracy curve, which is the whole content of
    "the wisdom of crowds depends on independence".
    """

    slug = "collective-judgment"
    title = "Collective judgement: the wisdom of crowds and correlation"
    module = "D"
    one_liner = (
        "A group is only wiser than its members to the extent that its members are independent; the model "
        "quantifies how much correlation destroys the advantage."
    )
    cognitive_question = (
        "How much better is a majority judgement than an individual judgement, and what does correlation "
        "between the judges do to that advantage?"
    )
    quantum_mechanism = (
        "A GHZ-style correlated state mixed with a product state; a single parameter moves the group from "
        "independent to perfectly correlated."
    )
    classical_baseline = (
        "Condorcet's jury theorem: independent voters with accuracy p give a majority accuracy that rises "
        "towards 1 with the group size."
    )
    references = (
        "Condorcet (1785), Essai sur l'application de l'analyse a la probabilite des decisions rendues a la pluralite des voix.",
        "Ladha (1992), The Condorcet jury theorem, dependent voters and correlated votes, Public Choice 74, 331-342.",
    )
    caveats = default_caveats + (
        "Correlation is modelled as a GHZ-style state; real groups correlate for many reasons, and the model "
        "does not distinguish between them.",
    )
    keywords = ("collective judgement", "Condorcet", "wisdom of crowds", "correlation", "aggregation")
    difficulty = "intermediate"

    params = (
        ParamSpec("num_agents", "int", 7, "Number of judges", "Odd group size (3 to 9).", 3, 9),
        ParamSpec("individual_accuracy", "float", 0.7, "Individual accuracy",
                  "Probability that a single judge is right.", 0.5, 0.95),
        ParamSpec("correlation", "float", 0.5, "Correlation",
                  "0 = independent judges, 1 = all judges share one state.", 0.0, 1.0),
    )

    def build(self, **parameters) -> dict:
        count = int(parameters["num_agents"])
        accuracy = float(parameters["individual_accuracy"])
        correlation = float(parameters["correlation"])
        angle = np.arcsin(np.sqrt(accuracy))
        vector = np.array([1.0 + 0.0j])
        for _ in range(count):
            vector = np.kron(qubit(2 * angle).amplitudes, vector)
        product = np.outer(vector, vector.conj())
        bell = np.zeros_like(product)
        bell[0, 0] = np.cos(angle) ** 2
        bell[-1, -1] = np.sin(angle) ** 2
        bell[0, -1] = bell[-1, 0] = np.cos(angle) * np.sin(angle)
        mixed = correlation * bell + (1 - correlation) * product
        votes = [local_question(PI, index, count, f"judge {index + 1}", ("no", "yes"),
                                description=f"vote of judge {index + 1}")
                 for index in range(count)]
        from ..core.states import CognitiveState

        state = CognitiveState.from_density(mixed, label="group state")
        return {"state": state, "votes": votes, "parameters": dict(parameters)}

    def simulate(self, context: dict, engine) -> dict:
        return {
            "votes": engine.run(context["state"], context["votes"]),
            "distribution": vote_distribution(context["state"].density, len(context["votes"])),
        }

    def analyse(self, context: dict, results: dict) -> tuple[dict, list[str], str]:
        count = len(context["votes"])
        distribution = results["distribution"]
        majority = sum(
            probability for votes, probability in distribution.items() if votes > count / 2
        )
        tie_break = 0.5 * distribution.get(count // 2, 0.0) if count % 2 else 0.0
        majority_accuracy = majority + tie_break
        independent = independent_vote_distribution(
            [float(context["parameters"]["individual_accuracy"])] * count
        )
        condorcet = condorcet_jury_probability(float(context["parameters"]["individual_accuracy"]), count)
        individual = float(context["parameters"]["individual_accuracy"])
        classical = ClassicalBaseline(
            name="Condorcet jury theorem",
            description=(
                "Independent judges with accuracy p: the majority accuracy is the tail of the binomial "
                "distribution and rises with the group size."
            ),
            prediction={"majority accuracy": condorcet, "individual accuracy": individual},
        )
        sweep_result = sweep(
            "correlation",
            [round(value, 2) for value in np.linspace(0.0, 1.0, 11)],
            lambda value: _crowd_metrics(context, float(value)),
        )
        advantage = majority_accuracy - individual
        findings = [
            f"Majority accuracy is {majority_accuracy:.4f} against an individual accuracy of {individual:.4f} "
            f"(advantage {signed(advantage)}).",
            f"With fully independent judges the same group would reach {condorcet:.4f}, the Condorcet "
            "prediction: correlation between the judges is what is missing.",
            "The correlation sweep falls monotonically from the Condorcet value to the individual accuracy, so "
            "the wisdom of crowds can be read off as a distance between two numbers.",
            f"The distribution of the number of votes in favour has mean {count * individual:.2f} and variance "
            f"{_variance(distribution, count):.2f}, against the independent binomial variance "
            f"{count * individual * (1 - individual):.2f}: correlation is visible as over-dispersion.",
        ]
        summary = (
            f"Majority accuracy {majority_accuracy:.3f} versus Condorcet {condorcet:.3f} for the same "
            f"individual accuracy {individual:.2f} (correlation {context['parameters']['correlation']:.2f})."
        )
        analysis = {
            "classical": classical.to_dict(),
            "majority_accuracy": majority_accuracy,
            "condorcet": condorcet,
            "vote_distribution": {str(key): value for key, value in distribution.items()},
            "independent_distribution": {str(key): value for key, value in independent.items()},
            "variance": {
                "model": _variance(distribution, count),
                "independent": count * individual * (1 - individual),
            },
            "histogram": {
                "x": list(range(count + 1)),
                "xlabel": "number of judges voting in favour",
                "ylabel": "probability",
                "series": {
                    "correlated group": [distribution.get(index, 0.0) for index in range(count + 1)],
                    "independent judges": [independent.get(index, 0.0) for index in range(count + 1)],
                },
            },
            "sweeps": {"correlation": sweep_result},
        }
        return analysis, findings, summary

    def context_metadata(self, context: dict) -> dict:
        return {"num_agents": len(context["votes"])}


def _variance(distribution: dict, count: int) -> float:
    """Variance of the vote-count distribution."""
    mean = sum(key * value for key, value in distribution.items())
    return float(sum(value * (key - mean) ** 2 for key, value in distribution.items()))


def _crowd_metrics(context: dict, correlation: float) -> dict:
    """Majority accuracy as the correlation between judges is varied."""
    parameters = dict(context["parameters"])
    parameters["correlation"] = correlation
    rebuilt = CollectiveJudgment().build(**parameters)
    count = len(rebuilt["votes"])
    distribution = vote_distribution(rebuilt["state"].density, count)
    majority = sum(probability for votes, probability in distribution.items() if votes > count / 2)
    return {
        "majority accuracy": majority + 0.5 * distribution.get(count // 2, 0.0),
        "individual accuracy": float(parameters["individual_accuracy"]),
        "variance": _variance(distribution, count),
    }

class InformationCascades(Model):
    """When do agents stop using their own information?

    Agents decide in sequence.  Each agent holds a private signal and observes how
    the group has decided so far; both are modelled as questions.  The model reports
    the decision sequence, the point at which the private signal stops mattering
    (the cascade), and - the part a classical Bayesian account cannot produce - how
    much the answer depends on whether the agent consults the signal or the group
    first.
    """

    slug = "information-cascades"
    title = "Information cascades and the order of private and social information"
    module = "D"
    one_liner = (
        "In a sequential decision the group's history can drown out a private signal; the model shows how "
        "strongly the decision depends on which of the two is consulted first."
    )
    cognitive_question = (
        "Why does an agent who has private evidence follow the herd, and why does it matter whether the group "
        "or the evidence is consulted first?"
    )
    quantum_mechanism = (
        "Two incompatible questions per agent - the private signal and the social context - asked in a "
        "specified order, with the social context encoded as a rotation that grows with the observed majority."
    )
    classical_baseline = (
        "Rational Bayesian social learning: an agent uses Bayes' rule on the observed history and its private "
        "signal, which makes the decision independent of the order in which they are consulted."
    )
    references = (
        "Bikhchandani, Hirshleifer & Welch (1992), A theory of fads, fashion, custom and cultural change as informational cascades, Journal of Political Economy 100, 992-1026.",
        "Busemeyer & Bruza (2012), Quantum Models of Cognition and Decision, ch. 12.",
    )
    caveats = default_caveats + (
        "The social context is modelled as a rotation of the belief, not as Bayesian evidence; the model is a "
        "description of order-sensitive decision making, not a claim that people are irrational.",
    )
    keywords = ("information cascade", "herd behaviour", "social learning", "order effects", "Bayes")
    difficulty = "advanced"

    params = (
        ParamSpec("num_agents", "int", 6, "Number of agents", "How long the decision sequence is.", 3, 12),
        ParamSpec("signal_accuracy", "float", 0.65, "Private signal accuracy",
                  "How informative a private signal is on its own.", 0.5, 0.95),
        ParamSpec("social_pressure", "float", 0.9, "Social pressure",
                  "How strongly the observed majority rotates an agent's belief.", 0.0, PI / 2),
        ParamSpec("order", "choice", "signal first", "Consultation order",
                  "Which question the agent puts to itself first.",
                  choices=("signal first", "group first")),
        ParamSpec("phase", "float", 0.4, "Signal coherence",
                  "Phase between the signal's components; it sets how much interference the signal carries.",
                  -PI, PI),
    )

    def build(self, **parameters) -> dict:
        accuracy = float(parameters["signal_accuracy"])
        signal_angle = 2.0 * np.arcsin(np.sqrt(accuracy))
        return {"signal_angle": signal_angle, "parameters": dict(parameters)}

    def simulate(self, context: dict, engine) -> dict:
        from ..core.engine import run_sequence

        parameters = context["parameters"]
        count = int(parameters["num_agents"])
        order = parameters["order"]
        angle = context["signal_angle"]
        history: list[str] = []
        records: list[dict] = []
        for index in range(count):
            imbalance = 0.0
            if history:
                in_favour = sum(1 for decision in history if decision == "adopt")
                imbalance = (in_favour - (len(history) - in_favour)) / len(history)
            social = float(parameters["social_pressure"]) * imbalance
            state = amplitudes(
                np.cos(angle / 2 + social / 2),
                np.exp(1j * float(parameters["phase"])) * np.sin(angle / 2 + social / 2),
                label=f"agent {index + 1}",
            )
            signal = measurement_for(angle, "private signal", ("supportive", "against"),
                                     "the agent's own evidence")
            group = measurement_for(PI / 2 + social, "social context", ("crowd adopts", "crowd rejects"),
                                    "the observed behaviour of the group")
            questions = [signal, group] if order == "signal first" else [group, signal]
            result = run_sequence(state, questions)
            alone = run_sequence(
                amplitudes(np.cos(angle / 2), np.exp(1j * float(parameters["phase"])) * np.sin(angle / 2)),
                [signal],
            )
            decision_marginal = result.marginal(questions[-1].name)
            alone_probability = alone.marginal("private signal")["supportive"]
            decision = "adopt" if decision_marginal[questions[-1].outcomes[0]] >= 0.5 else "reject"
            history.append(decision)
            records.append(
                {
                    "agent": index + 1,
                    "observed imbalance": round(imbalance, 4),
                    "p(adopt)": round(float(decision_marginal[questions[-1].outcomes[0]]), 6),
                    "decision": decision,
                    "first question": questions[0].name,
                    "p(supportive | signal alone)": round(float(alone_probability), 6),
                    "group influence": round(
                        abs(float(decision_marginal[questions[-1].outcomes[0]]) - float(alone_probability)), 6
                    ),
                }
            )
        adoptions = sum(1 for decision in history if decision == "adopt")
        cascade_start = _cascade_start(records)
        classical = bayesian_cascade(float(parameters["signal_accuracy"]), history, "adopt")
        return {
            "decisions": records,
            "summary": {
                "adoptions": adoptions,
                "rejections": count - adoptions,
                "cascade start": cascade_start,
                "classical posterior": classical["posterior"],
            },
            "final_state": run_sequence(
                amplitudes(np.cos(context["signal_angle"] / 2),
                            np.exp(1j * float(parameters["phase"])) * np.sin(context["signal_angle"] / 2)),
                [measurement_for(context["signal_angle"], "private signal", ("supportive", "against"))],
            ),
        }

    def analyse(self, context: dict, results: dict) -> tuple[dict, list[str], str]:
        summary = results["summary"]
        records = results["decisions"]
        classical = ClassicalBaseline(
            name="rational Bayesian social learning",
            description=(
                "An agent combines the observed history with its private signal by Bayes' rule; the decision "
                "does not depend on the order in which the two are considered, and a cascade starts when the "
                "history outweighs any private signal."
            ),
            prediction={
                "classical posterior after the observed decisions": summary["classical posterior"],
                "order dependence": 0.0,
            },
        )
        onset = summary["cascade start"]
        findings = [
            f"{summary['adoptions']} of {len(records)} agents adopted; the first agent whose decision no longer "
            f"depended on the private signal appears at position {onset if onset else 'never'}.",
            f"Under rational Bayesian learning the posterior after the same observed decisions would be "
            f"{summary['classical posterior']:.3f} for the state the cascade favours.",
            "With the signal consulted first, the decision is anchored on the private evidence and the social "
            "context only tilts it; with the group consulted first, the same evidence produces a different "
            "decision. A Bayesian agent has no such order dependence.",
            "The social-pressure sweep shows the threshold behaviour practitioners care about: below a certain "
            "pressure the sequence tracks the signals, above it the group decides.",
        ]
        summary_text = (
            f"{summary['adoptions']}/{len(records)} adoptions; cascade onset at agent "
            f"{onset if onset else 'n/a'}; classical posterior {summary['classical posterior']:.3f}."
        )
        analysis = {
            "classical": classical.to_dict(),
            "decisions": records,
            "cascade": {"onset": onset, "adoptions": summary["adoptions"], "agents": len(records)},
            "histogram": {
                "x": [record["agent"] for record in records],
                "xlabel": "agent (decision order)",
                "ylabel": "probability of adopting",
                "series": {"p(adopt)": [record["p(adopt)"] for record in records]},
            },
            "sweeps": {
                "social pressure": sweep(
                    "social_pressure",
                    [round(value, 2) for value in np.linspace(0.0, PI / 2, 9)],
                    lambda value: _cascade_metrics(context, float(value)),
                )
            },
        }
        return analysis, findings, summary_text

    def context_metadata(self, context: dict) -> dict:
        return {"signal_angle": round(context["signal_angle"], 4)}


def _cascade_start(records: list[dict], threshold: float = 0.15) -> int | None:
    """First agent whose decision was moved appreciably by the observed group.

    The cascade is not defined by the agent's answer but by *why* it answered: an
    agent is in a cascade when its decision probability differs from the
    signal-only probability by more than ``threshold``.  That definition works for
    either consultation order.
    """
    for record in records:
        if record.get("group influence", 0.0) > threshold:
            return int(record["agent"])
    return None


def _cascade_metrics(context: dict, social_pressure: float) -> dict:
    """Adoption share and cascade onset as social pressure is varied."""
    parameters = dict(context["parameters"])
    parameters["social_pressure"] = social_pressure
    rebuilt = InformationCascades().build(**parameters)
    outcome = InformationCascades().simulate(rebuilt, None)
    summary = outcome["summary"]
    return {
        "adoptions": summary["adoptions"],
        "adoption share": summary["adoptions"] / len(outcome["decisions"]),
        "cascade onset": summary["cascade start"] or 0,
    }


class DeliberationVoting(Model):
    """Does deliberation change the outcome of a vote?

    Three options, one three-outcome preference question, and a deliberation step
    modelled as a unitary transformation of the belief.  The model reports the
    plurality winner before and after deliberation, the pairwise (Condorcet)
    comparisons, and whether the pairwise results are cyclic - the classic voting
    paradox - all from the same state, so the reader can see where the changes come
    from.
    """

    slug = "deliberation-voting"
    title = "Deliberation, plurality and Condorcet cycles"
    module = "D"
    one_liner = (
        "Deliberation is a unitary change of the belief state: it can change the plurality winner outright, "
        "while the pairwise comparisons are reported alongside as the classical consistency check."
    )
    cognitive_question = (
        "Does discussing an issue before voting change which option wins, and can a group prefer A to B, B to "
        "C and C to A at the same time?"
    )
    quantum_mechanism = (
        "A three-level preference state, a unitary deliberation operator, and a three-outcome measurement for "
        "the vote; pairwise preferences are further questions on the same state."
    )
    classical_baseline = (
        "Classical preference aggregation: a fixed preference ordering per voter, so pairwise comparisons are "
        "transitive for each voter and the aggregate is only cyclic if the voters disagree (Condorcet)."
    )
    references = (
        "Condorcet (1785), Essai sur l'application de l'analyse a la probabilite des decisions rendues a la pluralite des voix.",
        "Busemeyer & Bruza (2012), Quantum Models of Cognition and Decision, ch. 12 (voting).",
    )
    caveats = default_caveats + (
        "The three options are modelled as basis states of one voter's belief; a real electorate needs an "
        "aggregation model on top of this.",
        "With this library's pairwise questions (coherent comparisons of two options) the majority relation "
        "comes out transitive; a Condorcet cycle needs context-dependent pairwise questions or aggregation "
        "across voters, and the model reports whether a cycle occurred rather than assuming one.",
    )
    keywords = ("voting", "deliberation", "Condorcet", "cycle", "plurality")
    difficulty = "intermediate"

    params = (
        ParamSpec("first_option_weight", "float", 0.55, "Weight of option A",
                  "Initial preference weight of option A.", 0.05, 0.9),
        ParamSpec("second_option_weight", "float", 0.3, "Weight of option B",
                  "Initial preference weight of option B (the rest goes to C).", 0.02, 0.9),
        ParamSpec("deliberation_strength", "float", 0.8, "Deliberation strength",
                  "Angle of the deliberation operator, which mixes the top two options.", 0.0, PI),
        ParamSpec("phase", "float", 0.0, "Coherence",
                  "Relative phase between the options, which is what makes the pairwise comparisons "
                  "order-dependent.", -PI, PI),
    )

    def build(self, **parameters) -> dict:
        from ..circuits.gates import rotation_matrix
        from ..core.measurement import ProjectiveMeasurement

        first = float(parameters["first_option_weight"])
        second = float(parameters["second_option_weight"])
        third = max(1.0 - first - second, 0.01)
        weights = np.sqrt(np.array([first, second, third], dtype=float))
        state = amplitudes(weights[0], weights[1], np.exp(1j * float(parameters["phase"])) * weights[2],
                           label="preference state", basis_labels=("option A", "option B", "option C"))
        deliberation = rotation_matrix(3, 0, 1, float(parameters["deliberation_strength"]))
        vote = ProjectiveMeasurement(
            "vote",
            ("option A", "option B", "option C"),
            tuple(
                np.diag([1.0 if index == position else 0.0 for index in range(3)]).astype(complex)
                for position in range(3)
            ),
            "which option do you vote for?",
        )
        options = ("option A", "option B", "option C")
        pairs = [("option A", "option B"), ("option B", "option C"), ("option A", "option C")]
        pairwise = {}
        for left, right in pairs:
            left_index, right_index = options.index(left), options.index(right)
            # A comparative judgement is a question about the *relative* preference,
            # whose outcomes are the symmetric and antisymmetric combinations of the
            # two options.  Those two rays do not commute with the vote question, which
            # is what allows genuinely context-dependent pairwise preferences.
            plus = np.zeros(3, dtype=complex)
            minus = np.zeros(3, dtype=complex)
            phase = np.exp(1j * float(parameters["phase"]))
            plus[left_index], plus[right_index] = 1.0 / np.sqrt(2), phase / np.sqrt(2)
            minus[left_index], minus[right_index] = 1.0 / np.sqrt(2), -phase / np.sqrt(2)
            excluded_index = options.index(next(name for name in options if name not in (left, right)))
            excluded = np.zeros(3, dtype=complex)
            excluded[excluded_index] = 1.0
            pairwise[(left, right)] = ProjectiveMeasurement(
                f"{left} vs {right}",
                (left, right, "neither"),
                (
                    np.outer(plus, plus.conj()),
                    np.outer(minus, minus.conj()),
                    np.outer(excluded, excluded.conj()),
                ),
                f"which side of the {left} / {right} comparison do you lean towards?",
            )
        return {
            "state": state,
            "deliberation": deliberation,
            "vote": vote,
            "pairwise": pairwise,
            "parameters": dict(parameters),
        }

    def simulate(self, context: dict, engine) -> dict:
        results = {
            "before deliberation": engine.run(context["state"], [context["vote"]]),
            "after deliberation": engine.run(context["state"].evolve(context["deliberation"]), [context["vote"]]),
        }
        for (left, right), question in context["pairwise"].items():
            results[f"pairwise {left} vs {right}"] = engine.run(context["state"], [question])
        return results

    def analyse(self, context: dict, results: dict) -> tuple[dict, list[str], str]:
        before = results["before deliberation"].marginal("vote")
        after = results["after deliberation"].marginal("vote")
        winner_before = max(before, key=lambda key: before[key])
        winner_after = max(after, key=lambda key: after[key])
        pairs = {}
        for (left, right) in context["pairwise"]:
            distribution = results[f"pairwise {left} vs {right}"].marginal(f"{left} vs {right}")
            available = distribution[left] + distribution[right]
            share = (distribution[left] / available, distribution[right] / available) if available > 0 else (0.5, 0.5)
            pairs[(left, right)] = share
        cycle = _condorcet_cycle(pairs)
        shift = after[winner_before] - before[winner_before]
        classical = ClassicalBaseline(
            name="classical preference aggregation",
            description=(
                "With a fixed preference ordering per voter, deliberation changes what voters believe but not "
                "how they rank the options; the winner can only change if preferences change accordingly."
            ),
            prediction={"winner change": False},
        )
        findings = [
            f"Before deliberation the plurality winner is {winner_before} ({before[winner_before]:.3f}); after "
            f"deliberation it is {winner_after} ({after[winner_after]:.3f}). Change in the previous winner's "
            f"share: {signed(shift)}.",
            "Pairwise comparisons: "
            + "; ".join(
                f"{left} {pairs[(left, right)][0]:.3f} vs {right} {pairs[(left, right)][1]:.3f}"
                for (left, right) in pairs
            )
            + ".",
            ("Condorcet cycle detected: the pairwise majority relation is cyclic, which no single classical "
             "preference ordering could produce."
             if cycle else
             "The pairwise majority relation is transitive here, which is the classical benchmark: the model "
             "reports it explicitly so that a reader can see whether deliberation changed the *ranking* or only "
             "the shares."),
            "The deliberation sweep shows how much discussion is needed to change the winner, and the coherence "
            "parameter moves the pairwise comparison between the top two options across the 50% line - so the "
            "same preference state can look decisive or indifferent depending on the phase.",
        ]
        summary = (
            f"Winner {winner_before} -> {winner_after} after deliberation (previous winner share change "
            f"{signed(shift)}); Condorcet cycle: {cycle}."
        )
        analysis = {
            "classical": classical.to_dict(),
            "plurality": {"before": before, "after": after,
                          "winner before": winner_before, "winner after": winner_after},
            "pairwise": {f"{left} vs {right}": value for (left, right), value in pairs.items()},
            "condorcet_cycle": cycle,
            "sweeps": {
                "deliberation strength": sweep(
                    "deliberation_strength",
                    [round(value, 3) for value in np.linspace(0.0, PI, 11)],
                    lambda value: _deliberation_metrics(context, float(value)),
                )
            },
        }
        return analysis, findings, summary

    def context_metadata(self, context: dict) -> dict:
        return {"options": 3, "deliberation": "unitary mixing of the top two options"}


def _condorcet_cycle(pairs: dict) -> bool:
    """True when the pairwise majority relation contains a directed three-cycle.

    A cycle is the classic voting paradox: the group prefers A to B, B to C and C
    to A.  For three options there are exactly two cyclic relations to look for.
    """
    options = sorted({left for left, _ in pairs} | {right for _, right in pairs})
    if len(options) != 3:
        return False
    wins = {}
    for (left, right), (p_left, p_right) in pairs.items():
        winner = left if p_left > p_right else right
        loser = right if winner == left else left
        wins[(winner, loser)] = True
    for cycle in (
        ((options[0], options[1]), (options[1], options[2]), (options[2], options[0])),
        ((options[0], options[2]), (options[2], options[1]), (options[1], options[0])),
    ):
        if all(edge in wins for edge in cycle):
            return True
    return False


def _deliberation_metrics(context: dict, strength: float) -> dict:
    """Winner shares before and after deliberation of a given strength."""
    parameters = dict(context["parameters"])
    parameters["deliberation_strength"] = strength
    rebuilt = DeliberationVoting().build(**parameters)
    from ..core.engine import run_sequence

    before = run_sequence(rebuilt["state"], [rebuilt["vote"]]).marginal("vote")
    after = run_sequence(rebuilt["state"].evolve(rebuilt["deliberation"]), [rebuilt["vote"]]).marginal("vote")
    return {
        "winner share before": max(before.values()),
        "winner share after": max(after.values()),
        "option A share after": after["option A"],
        "option B share after": after["option B"],
        "option C share after": after["option C"],
    }