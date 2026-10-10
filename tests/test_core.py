"""Tests of the numerical core: states, questions, channels, results."""

import numpy as np
import pytest

from qcog.core import (
    POVM,
    ProjectiveMeasurement,
    amplitudes,
    basis_state,
    bell_state,
    computational,
    depolarizing_channel,
    maximally_mixed,
    mixed_state,
    question,
    run_sequence,
    tensor,
    unsharp_question,
)
from qcog.core.linalg import embed_matrix, embed_vector, gell_mann_basis, is_projector, pvm_to_unitary
from qcog.core.measurement import incompatibility, measurements_commute


def test_state_normalisation_and_purity():
    state = amplitudes(1, 2, label="test")
    assert state.is_pure
    assert np.isclose(np.linalg.norm(state.amplitudes), 1.0)
    assert np.isclose(state.probabilities().sum(), 1.0)


def test_mixed_state_is_not_pure_and_has_no_amplitudes():
    state = mixed_state([(0.5, basis_state(0, 2)), (0.5, basis_state(1, 2))])
    assert not state.is_pure
    assert np.allclose(state.density, np.eye(2) / 2)
    with pytest.raises(ValueError):
        _ = state.amplitudes


def test_bloch_vector_of_basis_states():
    assert np.allclose(basis_state(0, 2).bloch_vector(), [0, 0, 1])
    assert np.allclose(basis_state(1, 2).bloch_vector(), [0, 0, -1])


def test_measurement_validation():
    with pytest.raises(ValueError):
        ProjectiveMeasurement("bad", ("a", "b"), (np.diag([1, 0]).astype(complex), np.diag([1, 0]).astype(complex)))
    with pytest.raises(ValueError):
        POVM("bad", ("a", "b"), (np.eye(2) / 4, np.eye(2) / 4))


def test_born_rule_and_lueders_update():
    state = amplitudes(1, 1)
    measurement = computational("Z", ("zero", "one"))
    assert np.allclose(measurement.probabilities(state), [0.5, 0.5])
    post = measurement.post_state(state, "zero")
    assert np.allclose(post.density, np.diag([1.0, 0.0]))


def test_unsharp_question_interpolates_to_the_sharp_one():
    state = amplitudes(1, 0)
    sharp = question(0.7, 0.0, "q")
    soft = unsharp_question(0.7, 1.0, 0.0, "q")
    assert np.allclose(sharp.probabilities(state), soft.probabilities(state))
    flat = unsharp_question(0.7, 0.0, 0.0, "q", ("yes", "no"))
    assert np.allclose(flat.probabilities(state), [0.5, 0.5])


def test_commuting_and_non_commuting_questions():
    z = computational("Z")
    x = question(np.pi / 2, 0.0, "X")
    assert not measurements_commute([z, x])
    assert incompatibility([z, x]) > 0.5
    assert measurements_commute([z, computational("Z2")])


def test_embedding_and_basis_unitary_round_trip():
    measurement = question(0.9, 0.3, "q")
    unitary, blocks = pvm_to_unitary(measurement.projectors)
    assert unitary.shape == (2, 2)
    assert all(is_projector(measurement.projectors[index]) for index, _, _ in blocks)
    padded = embed_matrix(unitary, 4)
    assert np.allclose(padded[:2, :2], unitary)
    assert np.allclose(padded[2:, 2:], np.eye(2))
    assert np.allclose(np.linalg.norm(embed_vector(np.ones(2))), np.sqrt(2))


def test_gell_mann_basis_is_complete_and_orthonormal():
    for dim in (2, 3, 4):
        basis = gell_mann_basis(dim)
        assert len(basis) == dim * dim - 1
        for left in basis:
            assert np.allclose(np.trace(left), 0.0, atol=1e-12)
        total = sum(matrix @ matrix for matrix in basis)
        assert np.allclose(total, 2 * (dim * dim - 1) / dim * np.eye(dim))


def test_depolarising_channel_limits():
    for probability, expected in ((0.0, np.diag([1.0, 0.0])), (1.0, np.eye(2) / 2)):
        channel = depolarizing_channel(probability, 2)
        outcome = basis_state(0, 2).apply_channel(channel.kraus)
        assert np.allclose(outcome.density, expected, atol=1e-12)


def test_depolarising_channel_is_trace_preserving_in_every_dimension():
    for dim in (2, 3, 4):
        channel = depolarizing_channel(0.35, dim)
        total = sum(operator.conj().T @ operator for operator in channel.kraus)
        assert np.allclose(total, np.eye(dim), atol=1e-12)


def test_tensor_puts_the_first_argument_on_the_high_qubit():
    state = tensor(basis_state(1, 2), basis_state(0, 2))
    assert np.isclose(state.probability(2), 1.0)  # index = qubit0 + 2*qubit1 = 2


def test_results_marginals_conditionals_and_error_bars():
    state = basis_state(0, 2)
    first = question(0.955, 0.0, "A")
    second = question(np.pi / 2, 0.0, "B")
    result = run_sequence(state, [first, second], backend="sampled", shots=40000, seed=3)
    assert result.is_exact is False
    assert np.isclose(sum(result.marginal("A").values()), 1.0)
    conditional = result.conditional("B", {"A": "yes"})
    assert np.isclose(sum(conditional.values()), 1.0)
    errors = result.sampling_error("A")
    assert all(0.0 <= value <= 0.05 for value in errors.values())
    assert "A" in result.summary()


def test_duplicate_question_names_are_rejected():
    measurement = question(0.5, 0.0, "A")
    with pytest.raises(ValueError):
        run_sequence(basis_state(0, 2), [measurement, measurement])


def test_bell_state_entropy():
    state = bell_state()
    reduced = state.partial_trace([0], dims=[2, 2])
    assert np.isclose(reduced.entropy, 1.0)


def test_maximally_mixed_has_maximal_entropy():
    assert np.isclose(maximally_mixed(2).entropy, 1.0)