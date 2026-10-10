"""Tests of the analysis layer: interference, order effects, QQ, entanglement, sweeps."""

import numpy as np
import pytest

from qcog.analysis import (
    entanglement_report,
    interference_decomposition,
    order_effect,
    qq_equality,
    sweep,
)
from qcog.analysis.order import law_of_total_probability_test
from qcog.core import (
    amplitudes,
    basis_state,
    bell_state,
    computational,
    depolarizing_channel,
    mixed_state,
    question,
    run_sequence,
    tensor,
    unsharp_question,
)
from qcog.core.measurement import incompatibility


@pytest.fixture()
def pair():
    state = basis_state(0, 2)
    return state, question(0.955, 0.0, "A"), question(np.pi / 2, 0.0, "B")


def test_interference_decomposition_is_exact(pair):
    state, first, second = pair
    report = interference_decomposition(state, first, second)
    assert report.check()
    single = run_sequence(state, [second])
    for outcome, total in report.totals.items():
        assert np.isclose(total, single.marginal("B")[outcome])


def test_commuting_questions_have_no_interference():
    state = amplitudes(1, 0)
    first = computational("path")
    second = question(0.4, 0.0, "B")
    report = interference_decomposition(state, first, second)
    assert report.max_abs_interference < 1e-9
    assert incompatibility([first, second]) > 0.0  # the questions themselves are incompatible


def test_order_effect_and_law_of_total_probability(pair):
    state, first, second = pair
    forward = run_sequence(state, [first, second])
    backward = run_sequence(state, [second, first])
    report = order_effect(forward, backward, "A", "B")
    assert report.max_marginal_shift() > 0.2
    violations = law_of_total_probability_test(forward, backward, "A", "B")
    assert max(abs(item.violation) for item in violations) > 0.2


def test_qq_equality_holds_for_incompatible_questions(pair):
    state, first, second = pair
    report = qq_equality(
        run_sequence(state, [first, second]), run_sequence(state, [second, first]), "A", "B"
    )
    assert report.applicable
    assert abs(report.difference) < 1e-9
    assert report.satisfied


def test_qq_equality_reports_scalar_differences_when_labels_differ():
    state = basis_state(0, 2)
    two_outcome = question(np.pi / 2, 0.0, "X", ("yes", "no"))
    three_outcome = __import__("qcog.core", fromlist=["question"]).question
    from qcog.core.measurement import ProjectiveMeasurement

    odd = ProjectiveMeasurement(
        "odd",
        ("low", "high"),
        (np.diag([1.0, 0.0]).astype(complex), np.diag([0.0, 1.0]).astype(complex)),
    )
    report = qq_equality(run_sequence(state, [two_outcome, odd]), run_sequence(state, [odd, two_outcome]), "X", "odd")
    assert not report.applicable
    assert "not applicable" in report.summary()


def test_entanglement_report_calibration():
    bell = entanglement_report(bell_state())
    assert np.isclose(bell.concurrence, 1.0, atol=1e-9)
    assert np.isclose(bell.entanglement, 1.0, atol=1e-9)
    product = entanglement_report(tensor(basis_state(0, 2), basis_state(0, 2)))
    assert product.is_product


def test_sweep_collects_metrics():
    result = sweep("theta", [0.0, 1.0, 2.0], lambda value: {"metric": value * 2, "other": -value})
    assert len(result) == 3
    assert result.column("metric") == [0.0, 2.0, 4.0]
    assert result.peak("metric")[1] == 4.0


def test_sharpness_and_decoherence_limits():
    state = basis_state(0, 2)
    soft_first = unsharp_question(0.955, 0.0, 0.0, "A")
    soft_second = unsharp_question(np.pi / 2, 0.0, 0.0, "B")
    report = order_effect(
        run_sequence(state, [soft_first, soft_second]),
        run_sequence(state, [soft_second, soft_first]),
        "A",
        "B",
    )
    assert report.max_marginal_shift() < 1e-12


def test_mixed_state_interference_uses_effects():
    state = mixed_state([(0.5, amplitudes(1, 0)), (0.5, amplitudes(0, 1))])
    first = computational("path")
    second = question(np.pi / 2, 0.0, "B")
    report = interference_decomposition(state, first, second)
    assert report.max_abs_interference < 1e-9  # an incoherent state has no interference