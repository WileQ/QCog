"""Tests of the classical baselines, the metrics and the fitting layer."""

import numpy as np
import pytest

from qcog.core.classical import (
    ClassicalConjunction,
    ClassicalPrisonersDilemma,
    bayesian_cascade,
    bounded_confidence_step,
    condorcet_jury_probability,
    de_groot_step,
    law_of_total_probability,
)
from qcog.core.fitting import compare_fits, fit, fittable_parameters
from qcog.core.metrics import (
    agreement_index,
    aic,
    bic,
    chi_square_test,
    hellinger,
    jensen_shannon,
    kl_divergence,
    proportion_z_test,
    total_variation,
)
from qcog.core.params import ParamSpec, coerce_params, fit_bounds
from qcog.registry import MODELS


def test_total_variation_and_kl():
    first = {"a": 0.5, "b": 0.5}
    second = {"a": 1.0, "b": 0.0}
    assert np.isclose(total_variation(first, second), 0.5)
    assert kl_divergence(first, second) > 0
    assert np.isclose(jensen_shannon(first, first), 0.0, atol=1e-12)
    assert 0 < hellinger(first, second) < 1


def test_chi_square_and_information_criteria():
    outcome = chi_square_test([10, 20, 30], [20, 20, 20])
    assert outcome["statistic"] > 0 and 0 <= outcome["p_value"] <= 1
    assert aic(-10.0, 2) > aic(-5.0, 2)
    assert bic(-10.0, 2, 100) < bic(-10.0, 5, 100)


def test_proportion_z_test_and_agreement_index():
    assert proportion_z_test(0.5, 0.5, 100)["p_value"] > 0.9
    assert np.isclose(agreement_index({("yes", "yes"): 0.4, ("yes", "no"): 0.6}), 0.4)


def test_classical_baselines():
    dilemma = ClassicalPrisonersDilemma()
    assert dilemma.payoffs("D", "C") == (5.0, 0.0)
    assert dilemma.nash_equilibrium()["pure_nash"] == ("D", "D")
    assert np.isclose(condorcet_jury_probability(0.7, 7), 0.8740, atol=5e-4)
    assert np.isclose(condorcet_jury_probability(0.5, 7), 0.5)
    bound = ClassicalConjunction()
    assert bound.is_violated(0.8, 0.3, 0.9)
    assert not bound.is_violated(0.2, 0.3, 0.9)


def test_classical_dynamics_helpers():
    opinions = np.array([0.0, 0.0, 1.0])
    adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    assert np.allclose(de_groot_step(opinions, adjacency), [0.0, 0.5, 0.0])
    clustered = bounded_confidence_step(np.array([0.0, 0.05, 1.0, 1.05]), epsilon=0.3)
    assert clustered[0] == pytest.approx(0.025) and clustered[2] == pytest.approx(1.025)


def test_law_of_total_probability_report():
    report = law_of_total_probability(
        "A", "yes", 0.8, {"yes": 0.5, "no": 0.5},
        {"yes": {"yes": 1.0, "no": 0.0}, "no": {"yes": 0.0, "no": 1.0}},
    )
    assert np.isclose(report.indirect, 0.5)
    assert np.isclose(report.violation, 0.3)
    assert "0.8000" in report.summary()


def test_bayesian_cascade_helper():
    outcome = bayesian_cascade(0.6, ["A", "A"], "B", ("A", "B"))
    assert 0.0 <= outcome["posterior"] <= 1.0
    assert outcome["decision_counts"] == {"A": 2, "B": 0}


def test_parameter_specification_coercion_and_bounds():
    specs = (
        ParamSpec("theta", "float", 1.0, low=0.0, high=2.0, fittable=True),
        ParamSpec("count", "int", 3, low=1, high=5),
        ParamSpec("mode", "choice", "a", choices=("a", "b")),
        ParamSpec("flag", "bool", False),
    )
    values = coerce_params(specs, {"theta": 5.0, "count": 100, "mode": "b", "flag": "yes"})
    assert values["theta"] == 2.0 and values["count"] == 5 and values["mode"] == "b" and values["flag"] is True
    assert fit_bounds(specs) == [(0.0, 2.0)]
    assert len(fittable_parameters(specs)) == 1
    with pytest.raises(ValueError):
        coerce_params(specs, {"nope": 1})


def test_model_predictions_can_be_fitted():
    model = MODELS["question-order"]()
    observed = model.predict({"angle_a": 1.05, "angle_b": 1.4})
    result = fit(lambda parameters: model.predict(parameters), observed, model.params)
    assert result.success
    assert result.r_squared > 0.999


def test_fitting_recovers_a_known_probability():
    observed = {"p": 0.25, "q": 0.75}
    specs = (ParamSpec("x", "float", 0.5, low=0.05, high=0.95, fittable=True),)

    def predict(parameters):
        return {"p": parameters["x"], "q": 1 - parameters["x"]}

    result = fit(predict, observed, specs)
    assert result.success
    assert np.isclose(result.parameters["x"], 0.25, atol=1e-6)
    assert result.r_squared > 0.999
    assert "fitted" in result.summary()


def test_fitting_a_model_with_counts_gives_information_criteria():
    observed = {"p": 0.3, "q": 0.7}
    counts = {"p": 30.0, "q": 70.0}
    specs = (ParamSpec("x", "float", 0.5, low=0.01, high=0.99, fittable=True),)

    def predict(parameters):
        return {"p": parameters["x"], "q": 1 - parameters["x"]}

    quantum = fit(predict, observed, specs, counts=counts)
    classical = fit(lambda _: {"p": 0.5, "q": 0.5}, observed, (), counts=counts)
    comparison = compare_fits(quantum, classical)
    assert quantum.aic is not None and classical.aic is not None
    assert "verdict" in comparison


def test_model_predictions_are_fittable():
    model = MODELS["question-order"]()
    predictions = model.predict({"angle_a": 1.0})
    assert len(predictions) >= 8
    assert all(0.0 <= value <= 1.0 for value in predictions.values())
    assert np.isclose(predictions["QQ: agreement AB"], predictions["QQ: agreement BA"], atol=1e-9)