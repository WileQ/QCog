"""Tests of the visualisation layer, the population model and the game experiment."""

import numpy as np
import pytest

from qcog.core import amplitudes, basis_state, question, run_sequence
from qcog.experiments import (
    QuantumGame,
    QuantumOpinionModel,
    TwoQuestionExperiment,
    independent_vote_distribution,
    topology_adjacency,
    vote_distribution,
)
from qcog.experiments.games import QuantumStrategy, entangling_unitary, strategy_unitary
from qcog.viz import (
    bloch_sphere,
    interference_figure,
    line_figure,
    order_effect_figure,
    probability_bars,
    sweep_figure,
    text_alternatives,
)


def test_two_question_experiment_runs_both_orders():
    experiment = TwoQuestionExperiment(
        basis_state(0, 2), question(0.9, 0.0, "A"), question(np.pi / 2, 0.0, "B")
    )
    results = experiment.run_both_orders(__import__("qcog.core", fromlist=["Engine"]).Engine())
    assert set(results) == {"A_then_B", "B_then_A"}
    bundle = experiment.analyse(results)
    assert bundle.report.max_marginal_shift() > 0.1
    assert bundle.qq.applicable


def test_game_unentangled_limit_is_classical():
    from qcog.core.classical import ClassicalPrisonersDilemma

    dilemma = ClassicalPrisonersDilemma()
    game = QuantumGame(dilemma.payoff_matrix(), entanglement=0.0)
    assert np.isclose(game.expected_payoffs(QuantumStrategy.defect(), QuantumStrategy.cooperate())["first"], 5.0)
    assert np.isclose(game.expected_payoffs(QuantumStrategy.defect(), QuantumStrategy.defect())["first"], 1.0)


def test_game_payoff_table_and_best_response():
    from qcog.core.classical import ClassicalPrisonersDilemma

    game = QuantumGame(dilemma_matrix := ClassicalPrisonersDilemma().payoff_matrix(), entanglement=np.pi / 2)
    table = game.payoff_table()
    assert len(table) == 9
    cooperative = table[(table["first"] == "C") & (table["second"] == "C")]["quantum_first"].iloc[0]
    temptation = table[(table["first"] == "D") & (table["second"] == "C")]["quantum_first"].iloc[0]
    assert np.isclose(cooperative, 3.0)
    # the entangler is what removes the classical temptation to defect on a cooperator
    assert np.isclose(temptation, 0.0)
    del dilemma_matrix
    best = game.best_response(QuantumStrategy.cooperate(), grid=7)
    # so cooperating is the best reply to cooperating: the dilemma is closed
    assert np.isclose(best["best"]["payoff"], cooperative)


def test_strategy_unitaries_are_unitary():
    for theta, phi in ((0.0, 0.0), (np.pi, 0.0), (0.0, np.pi / 2)):
        matrix = strategy_unitary(theta, phi)
        assert np.allclose(matrix.conj().T @ matrix, np.eye(2))
    assert np.allclose(entangling_unitary(0.0), np.eye(4))


def test_population_topologies_and_interaction():
    for topology in ("ring", "line", "complete", "star", "two-communities", "random"):
        adjacency = topology_adjacency(5, topology)
        assert adjacency.shape == (5, 5)
        assert np.allclose(adjacency, adjacency.T)
    model = QuantumOpinionModel([0.0, np.pi], topology="line", interaction_strength=np.pi / 4)
    model.step()
    assert np.allclose(model.opinions_now(), [0.0, 0.0], atol=1e-9)


def test_population_rejects_too_many_agents():
    with pytest.raises(ValueError):
        QuantumOpinionModel([0.0] * 12)


def test_vote_distributions_sum_to_one():
    state = amplitudes(np.cos(0.6), 0.0, 0.0, np.sin(0.6))
    distribution = vote_distribution(state.density, 2)
    assert np.isclose(sum(distribution.values()), 1.0)
    assert np.isclose(distribution[0], np.cos(0.6) ** 2)
    binomial = independent_vote_distribution([0.5, 0.5])
    assert np.isclose(sum(binomial.values()), 1.0)
    assert np.isclose(binomial[1], 0.5)


def test_figures_carry_descriptions():
    result = run_sequence(basis_state(0, 2), [question(0.9, 0.0, "A")])
    figures = {
        "bars": probability_bars(result),
        "order": order_effect_figure(
            __import__("qcog.analysis", fromlist=["order_effect"]).order_effect(
                run_sequence(basis_state(0, 2), [question(0.9, 0.0, "A"), question(np.pi / 2, 0.0, "B")]),
                run_sequence(basis_state(0, 2), [question(np.pi / 2, 0.0, "B"), question(0.9, 0.0, "A")]),
                "A",
                "B",
            )
        ),
        "sphere": bloch_sphere(basis_state(0, 2)),
        "lines": line_figure({"x": [0, 1], "xlabel": "step", "ylabel": "value", "series": {"a": [0.0, 1.0]}}),
    }
    descriptions = text_alternatives(figures)
    for name, description in descriptions.items():
        assert len(description) > 20, name


def test_interference_and_sweep_figures():
    from qcog.analysis import interference_decomposition, sweep

    report = interference_decomposition(basis_state(0, 2), question(0.955, 0.0, "A"), question(np.pi / 2, 0.0, "B"))
    figure = interference_figure(report)
    assert "interference" in getattr(figure, "_qcog_description")
    table = sweep("x", [0.0, 1.0, 2.0], lambda value: {"metric": value, "other": 2 * value})
    assert "metric" in getattr(sweep_figure(table), "_qcog_description")