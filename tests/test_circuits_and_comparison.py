"""Every model must be runnable, adjustable, compared with a classical baseline, and
must draw its circuit where a circuit exists.

These tests are the executable form of three promises the library makes:
that each of the twenty-three models can be driven from Python and reconfigured
parameter by parameter, that every model reports what a classical account of the
same situation predicts, and that the quantum circuit is drawn wherever the model
has one - with an explanation in words when it does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from qcog.core.instruments import depolarizing_channel, per_qubit_channel
from qcog.experiments.population import (
    QuantumOpinionModel,
    apply_local_depolarising,
    apply_partial_swap,
    apply_single_qubit_channel,
    apply_two_qubit_gate,
    local_entropies,
    local_opinions,
    partial_swap,
    _pair_unitary,
)
from qcog.registry import MODELS, get_model
from qcog.viz import circuit_figure, classical_comparison_figure


@pytest.fixture(scope="module")
def runs():
    return {slug: model().run() for slug, model in MODELS.items()}


# ---------------------------------------------------------------- adjustability
def test_every_model_runs_and_returns_normalised_results(runs):
    for slug, result in runs.items():
        assert result.summary, slug
        for name, item in result.sequential_results().items():
            assert np.isclose(sum(item.joint.values()), 1.0, atol=1e-6), (slug, name)


def test_every_parameter_can_be_changed():
    """Each parameter is exercised at both bounds (or every choice) from Python."""
    for slug, model_class in MODELS.items():
        model = model_class()
        for spec in model_class.params:
            if spec.kind in ("float", "int"):
                values = [value for value in (spec.low, spec.high) if value is not None]
            elif spec.kind == "choice":
                values = list(spec.choices or ())
            elif spec.kind == "bool":
                values = [True, False]
            else:
                values = [spec.default]
            for value in values:
                result = model.run(sweeps=False, **{spec.name: value})
                for item in result.sequential_results().values():
                    assert np.isclose(sum(item.joint.values()), 1.0, atol=1e-6), (slug, spec.name, value)


def test_unknown_parameters_are_rejected():
    with pytest.raises(ValueError):
        get_model("question-order").run(not_a_parameter=1)


def test_parameter_bounds_are_enforced():
    model = get_model("quantum-opinion-dynamics")
    assert model.run(num_agents=99).parameters["num_agents"] <= 8
    assert model.run(num_agents=-5).parameters["num_agents"] >= 2


def test_the_execution_noise_keyword_is_not_a_model_parameter():
    """``noise`` belongs to the execution layer, and saying so must be clear."""
    with pytest.raises(TypeError) as error:
        get_model("question-order").run(noise=0.5)
    assert "execution layer" in str(error.value)
    # the models' own decoherence parameters are separate and do work
    noisy = get_model("quantum-prisoners-dilemma").run(decoherence=0.3)
    assert noisy.sequential_results()
    assert noisy.parameters["decoherence"] == 0.3


# ------------------------------------------------------------- classical rows
def test_every_model_reports_a_classical_comparison(runs):
    for slug, result in runs.items():
        rows = result.classical_comparison
        assert rows, f"{slug} has no classical comparison"
        for row in rows:
            assert row["metric"], slug
            assert np.isfinite(row["model"]) and np.isfinite(row["baseline"]), (slug, row)
            assert "note" in row
        assert result.classical["name"], slug
        assert result.classical["verdict"], slug


def test_flagship_comparisons_are_the_expected_ones(runs):
    metrics = {row["metric"] for row in runs["question-order"].classical_comparison}
    assert "order effect on the first question" in metrics
    assert "interference share of the second question" in metrics
    assert any(row["baseline"] == 0.0 for row in runs["question-order"].classical_comparison)

    conjunction = {row["metric"]: row for row in runs["conjunction-fallacy"].classical_comparison}
    assert conjunction["p(A and B)"]["model"] > conjunction["p(A and B)"]["baseline"]

    prisoners = {row["metric"]: row for row in runs["quantum-prisoners-dilemma"].classical_comparison}
    assert prisoners["expected payoff for the first player"]["model"] > prisoners[
        "expected payoff for the first player"
    ]["baseline"]

    crowd = {row["metric"]: row for row in runs["collective-judgment"].classical_comparison}
    assert crowd["majority accuracy"]["baseline"] > crowd["majority accuracy"]["model"]


def test_comparison_figure_carries_a_description(runs):
    figure = classical_comparison_figure(runs["question-order"].classical_comparison)
    description = getattr(figure, "_qcog_description")
    assert "classical" in description
    assert "order effect" in description


def test_models_can_override_the_comparison(runs):
    """A model may sharpen its own comparison; the default must still be usable."""
    for slug, result in runs.items():
        assert result.classical["rows"] == result.classical_comparison


# ------------------------------------------------------------------- circuits
def test_every_model_has_a_circuit_or_a_reason(runs):
    for slug, result in runs.items():
        model = MODELS[slug]()
        designs = model.circuit_designs(result)
        notes = model.circuit_notes(result)
        assert designs or notes, f"{slug} neither draws a circuit nor explains why not"
        for name, design in designs.items():
            from qiskit import QuantumCircuit

            assert isinstance(design, QuantumCircuit), (slug, name)
            assert design.num_qubits >= 1


def test_circuit_figures_are_accessible(runs):
    from qiskit import QuantumCircuit

    model = MODELS["question-order"]()
    designs = model.circuit_designs(runs["question-order"])
    assert len(designs) == 2  # both orders of the same two questions
    for name, circuit in designs.items():
        assert isinstance(circuit, QuantumCircuit)
        figure = circuit_figure(circuit, title=name)
        description = getattr(figure, "_qcog_description")
        assert description.startswith("Circuit ")
        assert "qubit" in description
        assert len(getattr(figure, "_qcog_extra")) >= 1


def test_models_with_unsharp_questions_draw_their_sharp_limit(runs):
    model = MODELS["decision-tree"]()
    notes = " ".join(model.circuit_notes(runs["decision-tree"]))
    assert "sharp version" in notes
    assert model.circuit_designs(runs["decision-tree"])


def test_models_with_smaller_spaces_draw_their_circuit(runs):
    """A three-outcome question on two qubits must still compile."""
    for slug in ("auctions", "semantic-ambiguity", "deliberation-voting"):
        model = MODELS[slug]()
        designs = model.circuit_designs(runs[slug])
        assert designs, slug


def test_mixed_state_models_say_why_there_is_no_state_vector_circuit(runs):
    model = MODELS["collective-judgment"]()
    notes = " ".join(model.circuit_notes(runs["collective-judgment"]))
    assert "density matrix" in notes
    designs = model.circuit_designs(runs["collective-judgment"])
    assert "GHZ preparation of the correlated component" in designs


def test_game_models_draw_the_entangling_circuit(runs):
    for slug in ("quantum-prisoners-dilemma", "coordination-games", "negotiation-trust",
                 "evolution-of-cooperation"):
        model = MODELS[slug]()
        designs = model.circuit_designs(runs[slug])
        assert "entangling game circuit" in designs, slug
        assert designs["entangling game circuit"].num_qubits == 2


def test_figures_include_the_comparison_and_the_circuit(runs):
    model = MODELS["question-order"]()
    figures = model.figures(runs["question-order"])
    assert "model versus classical baseline" in figures
    assert any(name.startswith("circuit: ") for name in figures)
    for name, figure in figures.items():
        assert len(getattr(figure, "_qcog_description", "")) > 30, name


# --------------------------------------------------- numerical fast paths
def test_fast_partial_swap_matches_the_dense_construction():
    rng = np.random.default_rng(3)
    for agents in (2, 3, 4):
        total = 2**agents
        vector = rng.normal(size=total) + 1j * rng.normal(size=total)
        vector /= np.linalg.norm(vector)
        density = np.outer(vector, vector.conj())
        pairs = [(0, 1)] if agents == 2 else [(0, 1), (1, 2), (0, 2)]
        for left, right in pairs:
            gate = partial_swap(0.41)
            unitary = _pair_unitary(gate, left, right, agents)
            reference = unitary @ density @ unitary.conj().T
            assert np.allclose(apply_partial_swap(density, 0.41, left, right, agents), reference, atol=1e-10)
            assert np.allclose(apply_two_qubit_gate(density, gate, left, right, agents), reference, atol=1e-10)


def test_fast_local_channel_matches_the_composed_kraus_channel():
    rng = np.random.default_rng(4)
    for agents in (2, 3, 4):
        total = 2**agents
        vector = rng.normal(size=total) + 1j * rng.normal(size=total)
        vector /= np.linalg.norm(vector)
        density = np.outer(vector, vector.conj())
        fast = density
        for qubit in range(agents):
            fast = apply_local_depolarising(fast, 0.31, qubit, agents)
        composed = per_qubit_channel(depolarizing_channel, agents, p=0.31)
        reference = np.zeros_like(density)
        for operator in composed.kraus:
            reference += operator @ density @ np.asarray(operator).conj().T
        assert np.allclose(fast, reference, atol=1e-10)
        # and the Kraus-based local application agrees with both
        single = density
        kraus = depolarizing_channel(0.31, 2).kraus
        for qubit in range(agents):
            single = apply_single_qubit_channel(single, kraus, qubit, agents)
        assert np.allclose(single, reference, atol=1e-10)


def test_fast_observables_match_the_reference_partial_traces():
    from qcog.core.states import CognitiveState

    rng = np.random.default_rng(5)
    for agents in (2, 3):
        total = 2**agents
        vector = rng.normal(size=total) + 1j * rng.normal(size=total)
        vector /= np.linalg.norm(vector)
        density = np.outer(vector, vector.conj())
        state = CognitiveState.from_density(density)
        reduced = [state.partial_trace([qubit], dims=[2] * agents) for qubit in range(agents)]
        assert np.allclose(
            local_opinions(density, agents), [item.density[1, 1].real - item.density[0, 0].real for item in reduced]
        )
        assert np.allclose(local_entropies(density, agents), [item.entropy for item in reduced], atol=1e-8)


def test_population_trajectory_uses_the_opinion_scale():
    model = QuantumOpinionModel([0.0, np.pi], topology="line", interaction_strength=np.pi / 4)
    assert np.allclose(model.opinions_now(), [-1.0, 1.0], atol=1e-9)
    outcome = model.run(1)
    assert np.allclose(outcome["history"][0], [-1.0, 1.0], atol=1e-9)
    assert np.allclose(outcome["final"], [0.0, 0.0], atol=1e-9)
