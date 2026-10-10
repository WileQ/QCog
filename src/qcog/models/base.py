"""The model interface: metadata, parameters, simulation, analysis, figures.

A model in Q-Cog is a small, self-describing object.  It declares

* what cognitive question it explores and which quantum mechanism it uses,
* its parameters (once, as :class:`~qcog.core.params.ParamSpec` objects),
* how to simulate it, and
* which limitations a reader must keep in mind.

Everything else - the interactive explorer, the command line, the report
generator, the documentation and the tests - is driven from that declaration.
That is what makes twenty models maintainable instead of twenty scripts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping, Sequence

import numpy as np

from ..core.engine import Engine
from ..core.params import ParamSpec, coerce_params, parameter_table
from ..core.results import SequentialResult

__all__ = ["Model", "ModelResult", "MODULES", "make_engine"]

#: The four thematic modules of the library, in the order the proposal lists them.
MODULES = {
    "A": "Individual cognition and decision making",
    "B": "Logic, concepts and memory",
    "C": "Strategic decisions and game theory",
    "D": "Groups, social networks and collective behaviour",
}

#: Keyword arguments that configure execution rather than the model itself.
RESERVED = ("backend", "shots", "seed", "noise", "engine", "sweeps")


@dataclass
class ModelResult:
    """Everything one run of a model produces."""

    slug: str
    title: str
    module: str
    parameters: dict
    results: dict[str, Any] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    findings: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    backend: str = "exact"
    shots: int | None = None
    seed: int | None = None
    metadata: dict = field(default_factory=dict)
    classical: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ tables
    def tables(self) -> dict[str, Any]:
        """Named :class:`pandas.DataFrame` views of the result, for display and export."""
        frames: dict[str, Any] = {}
        for name, value in self.results.items():
            if isinstance(value, SequentialResult):
                frames[f"joint_{name}"] = value.joint_frame()
                frames[f"marginals_{name}"] = value.marginal_frame()
        return frames

    def sequential_results(self) -> dict[str, SequentialResult]:
        """Only the simulation results, without the analysis objects."""
        return {name: value for name, value in self.results.items() if isinstance(value, SequentialResult)}

    def report(self) -> dict:
        """The analysis section, with any report objects converted to dictionaries."""
        out = {}
        for name, value in self.analysis.items():
            if hasattr(value, "to_dict"):
                out[name] = value.to_dict()
            elif isinstance(value, Mapping):
                out[name] = {
                    key: (item.to_dict() if hasattr(item, "to_dict") else item) for key, item in value.items()
                }
            else:
                out[name] = value
        return out

    def to_dict(self) -> dict:
        return {
            "slug": self.slug,
            "title": self.title,
            "module": self.module,
            "parameters": {key: (value if isinstance(value, (int, float, str, bool)) else str(value))
                           for key, value in self.parameters.items()},
            "summary": self.summary,
            "findings": list(self.findings),
            "caveats": list(self.caveats),
            "references": list(self.references),
            "backend": self.backend,
            "shots": self.shots,
            "seed": self.seed,
            "results": {name: value.to_dict() for name, value in self.sequential_results().items()},
            "analysis": self.report(),
            "metadata": {key: value for key, value in self.metadata.items()
                         if isinstance(value, (int, float, str, bool, type(None), list, dict))},
        }

    @property
    def classical_comparison(self) -> list[dict]:
        """Model-versus-classical rows, as computed by :meth:`Model.classical_summary`."""
        return list(self.classical.get("rows", []))

    def headline(self) -> str:
        """One-sentence summary used in reports and the UI."""
        return self.summary or self.title


def make_engine(
    backend: str = "exact",
    shots: int | None = None,
    seed: int | None = None,
    noise=None,
    **options,
) -> Engine:
    """Build an :class:`~qcog.core.engine.Engine` from plain arguments."""
    if backend in {"sampled", "aer"} and not shots:
        shots = 4096
    return Engine(backend=backend, shots=shots, seed=seed, noise=noise, options=options)


# --------------------------------------------------------------------------- #
# Circuit view and classical comparison
#
# Two capabilities that every model inherits, so that "runnable and adjustable in
# Python", "show the circuit" and "always compare with classical cognition" hold
# for all twenty-three models without twenty-three copies of the same code.
# --------------------------------------------------------------------------- #

#: Dictionaries whose entries are alternative phrasings of one question rather than
#: a sequence of questions (used when deciding what a circuit should draw).
_VARIANT_DICTIONARIES = ("features", "pairwise", "questions_by_name")
#: Dictionaries whose entries really are a sequence of questions.
_SEQUENCE_DICTIONARIES = ("cues", "labels", "votes", "steps")
#: At most this many circuits are drawn per model: more would be noise.
MAX_CIRCUIT_VIEWS = 2
#: Circuits wider than this are not drawn readably.
MAX_CIRCUIT_QUBITS = 6


def _is_measurement(value) -> bool:
    """True for the two question types the library defines."""
    from ..core.measurement import POVM, ProjectiveMeasurement

    return isinstance(value, (ProjectiveMeasurement, POVM))


def _sharp_replacements(matching):
    """Replace unsharp questions with their sharp counterparts, when that is possible.

    ``unsharp_question`` records the angle and phase it was built from, so the
    projective question pointing in the same direction can be reconstructed exactly, and
    a circuit can be drawn for the sharp limit of the model.  A POVM without those
    parameters (a diluted or depolarised question) has no such counterpart, and the
    caller reports that instead of drawing something misleading.
    """
    from ..core.measurement import POVM, question as projective_question

    replaced = []
    for name, item in matching:
        if not (isinstance(item, POVM) and not item.is_projective):
            replaced.append((name, item))
            continue
        theta = item.metadata.get("theta")
        if theta is None:
            return None
        replaced.append(
            (
                name,
                projective_question(
                    float(theta), float(item.metadata.get("phi", 0.0)), item.name, item.outcomes,
                    item.description,
                ),
            )
        )
    return replaced


def _is_state(value) -> bool:
    """True for a cognitive state."""
    from ..core.states import CognitiveState

    return isinstance(value, CognitiveState)


class CircuitAndComparisonMixin:
    """The circuit view and the classical comparison.

    Kept as a mixin so that :class:`Model` stays readable, and so the two
    capabilities can be tested on their own.
    """

    # ------------------------------------------------------------------ circuits
    def circuit_designs(self, result) -> dict:
        """Named circuits for this model's design (empty when none applies).

        The scan is deliberately generic: it walks the model's own ``build`` output
        looking for a state and the questions that act on it, matches them by Hilbert
        space dimension, and compiles the pair or sequence.  A model can override this
        method when it wants to control what is drawn.
        """
        return self._circuit_scan(result)[0]

    def circuit_notes(self, result) -> list[str]:
        """Why some parts of the design have no circuit drawing."""
        return self._circuit_scan(result)[1]

    def extra_circuit_designs(self, result) -> dict:
        """Circuits a model wants to add beyond the generic scan (default: none).

        Used where the quantum mechanism is a specific state preparation that the
        generic scan cannot see, such as the GHZ preparation in the collective
        judgement model.
        """
        return {}

    def _circuit_scan(self, result, max_qubits: int = MAX_CIRCUIT_QUBITS):
        from ..circuits.builder import build_sequence_circuit
        from ..core.linalg import num_qubits_for_dim
        from ..core.measurement import POVM

        try:
            context = self.build(**result.parameters)
        except Exception as error:  # pragma: no cover - defensive
            return {}, [f"the design could not be rebuilt for the circuit view ({type(error).__name__})"]

        designs: dict[str, object] = {}
        notes: list[str] = []

        # 1. Games have their own circuit constructor: the entangling scheme is the
        #    quantum mechanism, so it is the most informative drawing available.
        game = context.get("game")
        if game is not None and hasattr(game, "circuit"):
            first, second = context.get("first"), context.get("second")
            if first is not None and second is not None:
                try:
                    designs["entangling game circuit"] = game.circuit(first, second)
                    notes.append(
                        "the game circuit shows the entangling gate J, the two strategy unitaries, J dagger "
                        "and the final measurement, which is the whole quantum mechanism of this model"
                    )
                except Exception as error:  # pragma: no cover - defensive
                    notes.append(f"the game circuit could not be built ({type(error).__name__})")

        # 2. Networks: one interaction step is the quantum ingredient.
        network = context.get("model")
        if network is not None and hasattr(network, "adjacency") and hasattr(network, "interaction_strength"):
            try:
                from qiskit import QuantumCircuit
                from qiskit.circuit.library import UnitaryGate
                from ..experiments.population import partial_swap

                circuit = QuantumCircuit(2, name="one interaction step")
                circuit.append(
                    UnitaryGate(partial_swap(float(network.interaction_strength)),
                                label="partial swap"),
                    [0, 1],
                )
                designs["one interaction step"] = circuit
                notes.append(
                    "the population itself is evaluated as a density matrix over all agents, so the drawing "
                    "shows one network interaction: the coherent partial swap that agents apply to a pair"
                )
            except Exception:  # pragma: no cover - defensive
                pass

        # 3. Two-question designs: both orders are worth drawing, because the model
        #    is about the difference between them.
        experiment = context.get("experiment")
        if experiment is not None and getattr(experiment, "question_a", None) is not None:
            if experiment.state.vector is not None:
                for label, questions in (
                    (f"{experiment.question_a.name} then {experiment.question_b.name}",
                     [experiment.question_a, experiment.question_b]),
                    (f"{experiment.question_b.name} then {experiment.question_a.name}",
                     [experiment.question_b, experiment.question_a]),
                ):
                    try:
                        designs[label] = build_sequence_circuit(experiment.state, questions)
                    except Exception as error:  # pragma: no cover - defensive
                        notes.append(f"{label}: {type(error).__name__}")
            else:
                notes.append(
                    "the model state is mixed (a density matrix), so there is no state-vector circuit to "
                    "draw; the exact backend evaluates it directly"
                )

        # 4. Everything else: match each state with the questions of the same dimension.
        #    Skipped when the model is a game: its circuit already ends with the
        #    measurement of the moves, so a second drawing would be a duplicate.
        states = (
            [] if game is not None else [(name, value) for name, value in context.items() if _is_state(value)]
        )
        top_level = [
            (name, value)
            for name, value in context.items()
            if _is_measurement(value) and not name.startswith("_")
        ]
        selected = context.get("parameters", {}).get("feature") if isinstance(context.get("parameters"), dict) else None
        for name, value in context.items():
            if isinstance(value, dict) and name in _SEQUENCE_DICTIONARIES:
                top_level.extend((f"{name}[{key}]", item) for key, item in value.items())
            elif isinstance(value, dict) and name in _VARIANT_DICTIONARIES:
                if selected is not None and selected in value:
                    top_level.append((f"{name}[{selected}]", value[selected]))
            elif isinstance(value, (list, tuple)) and value and all(_is_measurement(item) for item in value):
                top_level.extend((f"{name}[{index}]", item) for index, item in enumerate(value))
        best = None
        for state_name, state in states:
            matching = [(name, item) for name, item in top_level if item.dim == state.dim]
            if matching and (best is None or len(matching) > len(best[2])):
                best = (state_name, state, matching)
        if best is not None:
            state_name, state, matching = best
            if state.vector is None:
                notes.append(
                    f"the state '{state_name}' is mixed (a density matrix), so it cannot be prepared as a "
                    "circuit; the exact backend evaluates it directly"
                )
            elif num_qubits_for_dim(state.dim) > max_qubits:
                notes.append(
                    f"the state '{state_name}' needs {num_qubits_for_dim(state.dim)} qubits, which is too wide "
                    "to draw readably"
                )
            else:
                unsharp = [
                    name
                    for name, item in matching
                    if isinstance(item, POVM) and not item.is_projective
                ]
                sharp = _sharp_replacements(matching) if unsharp else matching
                if sharp is None:
                    notes.append(
                        "the question(s) " + ", ".join(unsharp) + " are unsharp (POVMs) with no sharp "
                        "counterpart, so no projective circuit reproduces them; the exact backend evaluates "
                        "the effects directly"
                    )
                else:
                    if unsharp:
                        notes.append(
                            "the drawing uses the sharp version of each question (lambda = 1); this run "
                            "dilutes them towards unsharp questions, which the exact backend evaluates directly"
                        )
                    matching = sharp
                    questions = [item for _, item in matching]
                    truncated = len(questions) > 3
                    questions = questions[:3]
                    label = " then ".join(name for name, _ in matching[:3])
                    if truncated:
                        notes.append(
                            f"the circuit view shows the first three of {len(matching)} questions; the rest are "
                            "evaluated by the exact backend"
                        )
                    try:
                        designs[label] = build_sequence_circuit(state, questions)
                        if len(questions) == 2:
                            reversed_label = " then ".join(
                                name for name, _ in reversed(matching[:2])
                            )
                            designs[reversed_label] = build_sequence_circuit(
                                state, list(reversed(questions))
                            )
                    except Exception as error:  # pragma: no cover - defensive
                        notes.append(f"{label}: {type(error).__name__}")
        # 5. Model-specific extras (a GHZ preparation, for instance).
        for name, design in (self.extra_circuit_designs(result) or {}).items():
            designs.setdefault(name, design)

        if not designs and not notes:
            notes.append(
                "this model has no single-state circuit: it is evaluated as a density matrix over several "
                "agents, or it builds its questions inside the simulation itself"
            )
        return dict(list(designs.items())[:MAX_CIRCUIT_VIEWS]), notes

    # ---------------------------------------------------------------- comparison
    def classical_summary(self, result) -> dict:
        """The model-versus-classical comparison: rows, baseline name and a verdict.

        The numbers come from what the model already computed, so this cannot drift
        from the simulation.  Every row is a pair of values on the same scale: what
        the model predicts, and what a classical account of the same situation
        predicts.  Where a classical model *must* produce zero - an order effect, an
        interference contribution, a conjunction excess - the baseline value is
        exactly zero, which is why the chart is informative rather than decorative.
        """
        rows = self.comparison_rows(result)
        analysis = result.analysis
        embedded = analysis.get("classical") if isinstance(analysis.get("classical"), dict) else {}
        name = embedded.get("name") or self.classical_baseline or "classical baseline"
        description = embedded.get("description") or self.classical_baseline
        zero_rows = [row for row in rows if row.get("baseline") == 0.0]
        if rows and len(zero_rows) == len(rows):
            verdict = (
                "Every row compares the model with the value a classical account is forced to give - exactly "
                "zero - so the bars are the model's own content."
            )
        elif rows:
            verdict = (
                "The classical reference values are shown next to the model's; the difference between the two "
                "bars is the model's content, and where they agree the classical account is adequate."
            )
        else:
            verdict = (
                "This model's classical comparison is reported in its findings and tables rather than as a "
                "single pair of numbers."
            )
        return {
            "name": name,
            "description": description,
            "rows": rows,
            "verdict": verdict,
        }

    def comparison_rows(self, result) -> list[dict]:
        """Numeric model-versus-classical rows for this model.

        Implemented generically over the analysis dictionaries the models produce, and
        overridable by a model that can make a sharper comparison.
        """
        analysis = result.analysis
        rows: list[dict] = []

        def add(metric: str, model, baseline, note: str = "") -> None:
            if model is None or baseline is None:
                return
            try:
                model_value = float(model)
                baseline_value = float(baseline)
            except (TypeError, ValueError):
                return
            if model_value != model_value or baseline_value != baseline_value:  # NaN guard
                return
            rows.append(
                {
                    "metric": metric,
                    "model": round(model_value, 6),
                    "baseline": round(baseline_value, 6),
                    "note": note,
                }
            )

        report = analysis.get("order_effect")
        if report is not None and hasattr(report, "max_marginal_shift"):
            add("order effect on the first question", report.max_marginal_shift(), 0.0,
                "a classical joint distribution forces this to zero")
            add("total variation between the two orders", report.total_variation, 0.0,
                "classical: the two orders are one distribution")
        qq = analysis.get("qq_equality")
        if qq is not None and getattr(qq, "applicable", False):
            add("QQ difference (agreement across orders)", abs(qq.difference), 0.0,
                "a classical joint distribution also predicts zero here")
        ltp_rows = analysis.get("ltp") or []
        violations = [abs(item.violation) for item in ltp_rows if hasattr(item, "violation")]
        if violations:
            add("law-of-total-probability violation", max(violations), 0.0,
                "a classical model satisfies the law exactly")
        interference = analysis.get("interference")
        if interference is not None and hasattr(interference, "max_abs_interference"):
            shares = [abs(value) for value in interference.relative_interference().values()]
            add("interference share of the second question", max(shares) if shares else 0.0, 0.0,
                "a classical account has no interference term")

        conjunction = analysis.get("conjunction")
        if isinstance(conjunction, dict):
            add("p(A and B)", conjunction.get("p_conjunction"), conjunction.get("classical_bound"),
                "the classical bound is min(p(A), p(B))")
            add("excess over the bound", conjunction.get("excess"), 0.0, "classical: never positive")
        independent = analysis.get("independent_check")
        if isinstance(independent, dict) and isinstance(analysis.get("conjunction"), dict):
            add("excess with independent attributes", independent.get("excess"), 0.0,
                "with independent attributes the bound is exact")

        values = analysis.get("values")
        if isinstance(values, dict) and "branch mixture" in values:
            add("decision when the alternative is unresolved", values.get("holistic unknown"),
                values.get("branch mixture"), "the branch mixture is the sure-thing prediction")

        structure = analysis.get("joint_structure")
        if isinstance(structure, dict):
            add("p(A and B)", structure.get("p(A and B)"), structure.get("independence"),
                "independence is the product-state prediction")
            add("correlation excess", structure.get("excess"), 0.0, "classical: independent features")

        typicality = analysis.get("typicality")
        if isinstance(typicality, dict):
            add("feature typicality of the combination", typicality.get("combined"),
                typicality.get("average"), "the weighted average of the parts")
            add("emergence over the parts", typicality.get("emergence"), 0.0,
                "an extensional account cannot exceed the parts")

        index = analysis.get("distinguishability")
        if isinstance(index, dict):
            add("distinguishability of the two states", index.get("coherent"), index.get("ignorance"),
                "classical ignorance cannot distinguish them")

        accuracy = analysis.get("majority_accuracy")
        if accuracy is not None and analysis.get("condorcet") is not None:
            add("majority accuracy", accuracy, analysis.get("condorcet"),
                "Condorcet's theorem assumes independent judges")

        if analysis.get("coordination") is not None:
            add("coordination probability", analysis.get("coordination"), 0.5,
                "the symmetric classical mixed equilibrium")

        agreement = analysis.get("agreement") if "agreement" in analysis else analysis.get("agreement probability")
        if isinstance(agreement, (int, float)):
            classical_agreement = 0.0
            if isinstance(analysis.get("classical"), dict):
                prediction = analysis["classical"].get("prediction", {})
                for key in ("agreement probability", "coordination probability"):
                    if isinstance(prediction, dict) and key in prediction:
                        classical_agreement = float(prediction[key])
            add("agreement between consecutive judgements", agreement, classical_agreement,
                "the independent-judgement baseline")

        shares = analysis.get("final_shares")
        if isinstance(shares, dict):
            trajectory = analysis.get("trajectory", {})
            classical_defection = 0.0
            series = trajectory.get("series", {}) if isinstance(trajectory, dict) else {}
            for key, values in series.items():
                if key.startswith("D (classical") and values:
                    classical_defection = float(values[-1])
            add("defection in the evolved population", shares.get("D"), classical_defection,
                "the replicator dynamics on the classical payoff matrix")

        communities = analysis.get("communities")
        if isinstance(communities, dict):
            add("gap between the communities", communities.get("gap"),
                communities.get("gap (bounded confidence)"), "bounded-confidence opinion dynamics")

        divergence = analysis.get("divergence")
        if isinstance(divergence, (int, float)):
            add("decision divergence between the two orders", divergence, 0.0,
                "a Bayesian decision tree is order-invariant")

        framing = analysis.get("framing_effect")
        if isinstance(framing, dict):
            add("framing effect on the judgement", framing.get("judgement"), 0.0,
                "a likelihood-based account is framing-invariant")

        order_shift = analysis.get("order_shift")
        if isinstance(order_shift, (int, float)):
            add("order effect on the valuation marginal", order_shift, 0.0,
                "a risk-neutral bidder is order-invariant")
        rival_shift = analysis.get("rival_shift")
        if isinstance(rival_shift, (int, float)):
            add("shift caused by information about rivals", rival_shift, 0.0,
                "a classical value estimate does not depend on the framing")

        sure_thing = analysis.get("sure_thing")
        if sure_thing is not None and hasattr(sure_thing, "violation"):
            add("sure-thing violation", abs(sure_thing.violation), 0.0,
                "the sure-thing principle is the classical baseline")

        stability = analysis.get("stability")
        if isinstance(stability, (int, float)):
            classical_stability = 0.0
            if isinstance(analysis.get("classical"), dict):
                prediction = analysis["classical"].get("prediction", {})
                classical_stability = float(prediction.get("answer rate", 0.0)) if isinstance(prediction, dict) else 0.0
            add("answer stability under repeated labels", stability, classical_stability,
                "independent judgements stay at the initial answer rate")

        cascade = analysis.get("cascade")
        if isinstance(cascade, dict) and cascade.get("agents"):
            adoption = cascade.get("adoptions", 0) / max(int(cascade.get("agents", 1)), 1)
            classical_posterior = 0.0
            if isinstance(analysis.get("classical"), dict):
                prediction = analysis["classical"].get("prediction", {})
                if isinstance(prediction, dict):
                    classical_posterior = float(
                        prediction.get("classical posterior after the observed decisions", 0.0)
                    )
            add("adoption share", adoption, classical_posterior,
                "the rational Bayesian posterior for the same history")

        plurality = analysis.get("plurality")
        if isinstance(plurality, dict):
            before = plurality.get("before", {})
            after = plurality.get("after", {})
            if before and after:
                winner = plurality.get("winner before")
                add("share of the winner before deliberation", before.get(winner), before.get(winner))
                add("change in the winner's share", (after.get(winner, 0.0) - before.get(winner, 0.0)), 0.0,
                    "classical preferences only move when they individually change")

        senses = analysis.get("sense_distributions")
        if isinstance(senses, dict) and "neutral" in senses and "context" in senses:
            neutral, contextual = senses["neutral"], senses["context"]
            dominant = max(neutral, key=lambda key: neutral[key]) if neutral else None
            if dominant:
                add("p(dominant sense) with no context", neutral[dominant], neutral[dominant])
                add("change in p(dominant sense) caused by context",
                    contextual.get(dominant, 0.0) - neutral[dominant], 0.0,
                    "prior re-weighting is order-invariant")

        # Games: compare with the classical equilibrium of the same payoff matrix.
        joint = analysis.get("joint_moves")
        matrix = analysis.get("payoff_matrix")
        if matrix is None:
            for source in (analysis.get("classical"), analysis.get("classical_report")):
                if isinstance(source, dict) and "payoff_matrix" in source:
                    matrix = source["payoff_matrix"]
                    break
        if isinstance(joint, dict) and joint and matrix is not None:
            separator = "/" if any("/" in str(key) for key in joint) else "-"
            labels: list[str] = []
            for key in joint:
                for part in str(key).split(separator):
                    if part not in labels:
                        labels.append(part)
            diagonal = [key for key in joint if len(set(str(key).split(separator))) == 1]
            if diagonal and len(labels) >= 2:
                from ..core.classical import symmetric_mixed_equilibrium

                equilibrium = symmetric_mixed_equilibrium(matrix)
                add(
                    "probability of the mutually cooperative outcome",
                    joint[diagonal[0]],
                    equilibrium["cooperation probability"],
                    "the mixed-strategy equilibrium of the same classical game",
                )
                try:
                    payload = np.asarray(matrix, dtype=float)
                    expected = 0.0
                    for key, probability in joint.items():
                        parts = str(key).split(separator)
                        if len(parts) == 2 and parts[0] in labels and parts[1] in labels:
                            expected += float(probability) * float(
                                payload[labels.index(parts[0]), labels.index(parts[1])]
                            )
                    add(
                        "expected payoff for the first player",
                        expected,
                        equilibrium["value"],
                        "the value of the classical equilibrium",
                    )
                except Exception:  # pragma: no cover - defensive
                    pass

        # Networks: compare the final configuration with classical averaging.
        metrics = analysis.get("metrics")
        classical_trajectory = analysis.get("classical_trajectory")
        if isinstance(metrics, dict) and isinstance(classical_trajectory, dict):
            series = classical_trajectory.get("series", {}) or {}
            finals = [values[-1] for values in series.values() if isinstance(values, list) and values]
            if finals:
                add(
                    "polarisation at the end",
                    metrics.get("polarisation"),
                    float(np.std(finals)),
                    "the classical baseline on the same network",
                )
                add(
                    "agreement (absolute mean opinion)",
                    metrics.get("agreement (|mean|)"),
                    abs(float(np.mean(finals))),
                    "the classical baseline on the same network",
                )
            add(
                "mean local entropy",
                metrics.get("mean local entropy"),
                0.0,
                "an averaging model has no superposition",
            )

        # Vote counts: over-dispersion relative to independent judges.
        variance = analysis.get("variance")
        if isinstance(variance, dict):
            add(
                "variance of the vote count",
                variance.get("model"),
                variance.get("independent"),
                "independent judges give the binomial variance",
            )

        contextuality = analysis.get("contextuality_certificate")
        if isinstance(contextuality, dict):
            add("contextuality certificate", contextuality.get("max_ltp_violation"), 0.0,
                "no joint distribution can host the two orders")
        return rows


class Model(CircuitAndComparisonMixin, ABC):
    """Base class for every Q-Cog model.

    Subclasses declare metadata as class attributes, parameters as the ``params``
    tuple, and implement :meth:`simulate` and :meth:`analyse`.  :meth:`run` adds
    parameter handling, execution and the analytical side-effects, so a subclass
    never has to repeat that boilerplate.
    """

    slug: ClassVar[str] = ""
    title: ClassVar[str] = ""
    module: ClassVar[str] = "A"
    one_liner: ClassVar[str] = ""
    cognitive_question: ClassVar[str] = ""
    quantum_mechanism: ClassVar[str] = ""
    classical_baseline: ClassVar[str] = ""
    references: ClassVar[tuple[str, ...]] = ()
    caveats: ClassVar[tuple[str, ...]] = ()
    keywords: ClassVar[tuple[str, ...]] = ()
    difficulty: ClassVar[str] = "intermediate"
    params: ClassVar[tuple[ParamSpec, ...]] = ()

    # ----------------------------------------------------------------- metadata
    @classmethod
    def module_title(cls) -> str:
        """Human-readable name of the module this model belongs to."""
        return MODULES.get(cls.module, "Unfiled")

    @classmethod
    def defaults(cls) -> dict:
        """Default value of every parameter."""
        return coerce_params(cls.params, {})

    @classmethod
    def parameter_table(cls) -> list[dict]:
        """Rows describing the model's parameters."""
        return parameter_table(cls.params)

    @classmethod
    def describe(cls) -> str:
        """Multi-line description used by ``qcog info`` and the documentation."""
        lines = [
            f"{cls.title} ({cls.slug}, module {cls.module})",
            f"  one-liner  : {cls.one_liner}",
            f"  question   : {cls.cognitive_question}",
            f"  mechanism  : {cls.quantum_mechanism}",
            f"  baseline   : {cls.classical_baseline}",
        ]
        if cls.params:
            lines.append("  parameters :")
            for spec in cls.params:
                bounds = ""
                if spec.low is not None or spec.high is not None:
                    bounds = f" in [{spec.low}, {spec.high}]"
                lines.append(
                    f"    - {spec.name} ({spec.kind}{bounds}, default {spec.default!r}): {spec.description}"
                )
        if cls.caveats:
            lines.append("  caveats    :")
            lines.extend(f"    - {caveat}" for caveat in cls.caveats)
        if cls.references:
            lines.append("  references :")
            lines.extend(f"    - {reference}" for reference in cls.references)
        return "\n".join(lines)

    @classmethod
    def to_dict(cls) -> dict:
        return {
            "slug": cls.slug,
            "title": cls.title,
            "module": cls.module,
            "module_title": cls.module_title(),
            "one_liner": cls.one_liner,
            "cognitive_question": cls.cognitive_question,
            "quantum_mechanism": cls.quantum_mechanism,
            "classical_baseline": cls.classical_baseline,
            "difficulty": cls.difficulty,
            "keywords": list(cls.keywords),
            "references": list(cls.references),
            "caveats": list(cls.caveats),
            "parameters": cls.parameter_table(),
            "defaults": cls.defaults(),
        }

    # ----------------------------------------------------------------- execution
    def run(self, **kwargs) -> ModelResult:
        """Simulate the model and analyse the outcome.

        Reserved keyword arguments (``backend``, ``shots``, ``seed``, ``noise``,
        ``engine``, ``sweeps``) configure execution; everything else is a model
        parameter and is validated against :attr:`params`.  Passing ``sweeps=False``
        skips the parameter sweeps and returns the headline analysis only, which is
        what to use while exploring a model interactively.
        """
        engine_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key in RESERVED}
        parameters = coerce_params(self.params, kwargs)
        engine = engine_kwargs.pop("engine", None) or make_engine(
            **{key: value for key, value in engine_kwargs.items() if key != "sweeps"}
        )
        context = self.build(**parameters)
        results = self.simulate(context, engine)
        from ..analysis.sweeps import suppress_sweeps

        with suppress_sweeps(not engine_kwargs.get("sweeps", True)):
            analysis, findings, summary = self.analyse(context, results)
        durations = [
            value.duration_seconds
            for value in results.values()
            if isinstance(value, SequentialResult)
        ]
        metadata = {
            "engine_backend": engine.backend,
            "engine_shots": engine.shots,
            "engine_seed": engine.seed,
            "runtime_seconds": float(sum(durations)),
            "difficulty": self.difficulty,
            "module_title": self.module_title(),
        }
        metadata.update(self.context_metadata(context))
        result = ModelResult(
            slug=self.slug,
            title=self.title,
            module=self.module,
            parameters=parameters,
            results=results,
            analysis=analysis,
            summary=summary,
            findings=findings,
            caveats=list(self.caveats),
            references=list(self.references),
            backend=engine.backend,
            shots=engine.shots,
            seed=engine.seed,
            metadata=metadata,
        )
        notes = self.circuit_notes(result)
        if notes:
            result.metadata["circuit_notes"] = notes
        result.classical = self.classical_summary(result)
        return result

    # ------------------------------------------------------------ subclass hooks
    @abstractmethod
    def build(self, **parameters) -> dict:
        """Construct the model objects (state, questions, network...) from parameters."""

    @abstractmethod
    def simulate(self, context: dict, engine: Engine) -> dict[str, SequentialResult]:
        """Run the simulations the model is built around."""

    @abstractmethod
    def analyse(self, context: dict, results: dict) -> tuple[dict, list[str], str]:
        """Return ``(analysis, findings, summary)`` for the simulated results."""

    def context_metadata(self, context: dict) -> dict:
        """Extra run metadata worth recording (measured quantities, circuit sizes...)."""
        return {}

    def predict(self, parameters: Mapping[str, Any] | None = None) -> dict[str, float]:
        """The model's predicted probabilities, used for fitting.

        Keys are stable response labels such as ``"AB:yes:yes"``.  Models that are
        not meant to be fitted raise :class:`NotImplementedError`, which the
        fitting layer reports as "this model has no closed-form predictor".
        """
        raise NotImplementedError(f"{self.slug} does not provide closed-form predictions")

    def figures(self, result: ModelResult) -> dict:
        """Named :class:`matplotlib.figure.Figure` objects for this result.

        Always includes a model-versus-classical comparison chart, plus a drawing of
        the quantum circuit wherever one exists for the model's design (a mixed state
        or an unsharp question means there is no circuit; :meth:`circuit_notes` says
        so in words).
        """
        from ..viz.figures import circuit_figure, classical_comparison_figure, default_figures

        figures = default_figures(self, result)
        rows = result.classical_comparison
        if rows:
            figures["model versus classical baseline"] = classical_comparison_figure(
                rows, result.self_title if hasattr(result, "self_title") else result.title
            )
        for name, design in self.circuit_designs(result).items():
            try:
                figures[f"circuit: {name}"] = circuit_figure(design, title=name)
            except Exception:  # pragma: no cover - a drawing failure must not break a run
                continue
        return figures

    def text_alternatives(self, result: ModelResult) -> dict[str, str]:
        """Plain-language descriptions of every figure, keyed by figure name."""
        from ..viz.figures import text_alternatives

        return text_alternatives(self.figures(result))

    # ------------------------------------------------------------------ helpers
    def _engine(self, backend: str = "exact", shots: int | None = None, seed: int | None = None, noise=None) -> Engine:
        return make_engine(backend=backend, shots=shots, seed=seed, noise=noise)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} slug={self.slug!r}>"


def combine_findings(*groups: Sequence[str]) -> list[str]:
    """Flatten groups of finding strings, dropping empties."""
    out: list[str] = []
    for group in groups:
        out.extend(text for text in group if text)
    return out


def format_probability(value: float) -> str:
    """Format a probability for human-readable output."""
    return f"{value:.3f}"


def largest_outcome(distribution: Mapping[str, float]) -> tuple[str, float]:
    """Most likely outcome of a distribution."""
    return max(distribution.items(), key=lambda item: item[1])


def numeric(value) -> float:
    """Coerce numpy scalars to plain floats for reporting."""
    return float(np.asarray(value).ravel()[0])
