"""Backend tests: the analytic engine and the compiled circuits must agree."""

import numpy as np
import pytest

from qcog.backends import AerBackend, AerExactBackend, get_backend, list_backends
from qcog.core import basis_state, amplitudes, NoiseSpec, question, run_sequence
from qcog.core.linalg import num_qubits_for_dim


@pytest.fixture(scope="session")
def aer():
    """Qiskit Aer, or a skip with the reason.

    Aer ships compiled extension modules that load on first use, and on some machines - a
    Windows box under load, for instance - the loader fails with "DLL load failed while
    importing controller_wrappers".  The library treats Aer as optional for exactly this
    reason: it retries, and then says so in a sentence.  The tests that need it follow the
    same rule, so an unusable Aer is a skip with the reason attached rather than a red suite.
    """
    try:
        run_sequence(basis_state(0, 2), [question(0.5, 0.0, "Q")], backend="aer_exact")
    except RuntimeError as error:  # pragma: no cover - depends on the machine
        if "could not be loaded" in str(error):
            pytest.skip(str(error))
        raise
    return True


@pytest.fixture()
def design():
    state = basis_state(0, 2)
    first = question(0.955, 0.0, "A")
    second = question(np.pi / 2, 0.0, "B")
    return state, [first, second]


def test_purge_modules_clears_a_partial_import():
    """A half-initialised package must be droppable, so a retry re-imports it cleanly.

    The test uses a throwaway package name: mutating ``sys.modules`` for a real dependency
    would leak into every other test in the process, which is exactly the failure mode being
    guarded against.
    """
    import sys
    import types

    from qcog.backends import purge_modules

    prefix = "qcog_test_partial_package"
    sys.modules[f"{prefix}.sub"] = types.ModuleType(f"{prefix}.sub")
    sys.modules[prefix] = types.ModuleType(prefix)
    try:
        removed = purge_modules(prefix)
        assert sorted(removed) == [prefix, f"{prefix}.sub"]
        assert prefix not in sys.modules and f"{prefix}.sub" not in sys.modules
        from qcog.backends import aer as aer_module

        assert "purge_modules" in aer_module.__all__
    finally:
        sys.modules.pop(prefix, None)
        sys.modules.pop(f"{prefix}.sub", None)


def test_backend_registry():
    assert "exact" in list_backends()
    assert isinstance(get_backend("aer[statevector]"), AerBackend)
    with pytest.raises(ValueError):
        get_backend("does-not-exist")


def test_exact_matches_compiled_prefix_circuits(aer, design):
    state, measurements = design
    exact = run_sequence(state, measurements)
    circuit = AerExactBackend().run(state, measurements)
    difference = max(abs(exact.joint[key] - circuit.joint[key]) for key in exact.joint)
    assert difference < 1e-12
    assert circuit.is_exact


def test_shot_based_circuit_agrees_within_sampling_error(aer, design):
    state, measurements = design
    exact = run_sequence(state, measurements)
    sampled = AerBackend().run(state, measurements, shots=30000, seed=99)
    worst = max(abs(exact.joint[key] - sampled.joint[key]) for key in exact.joint)
    assert worst < 5.0 * np.sqrt(0.25 / 30000)
    assert sampled.counts is not None


def test_circuit_reproduces_the_order_effect(aer, design):
    state, measurements = design
    forward = AerExactBackend().run(state, measurements)
    backward = AerExactBackend().run(state, list(reversed(measurements)))
    assert abs(forward.marginal("A")["yes"] - backward.marginal("A")["yes"]) > 0.2


def test_sampled_backend_marks_results_as_inexact(design):
    state, measurements = design
    result = run_sequence(state, measurements, backend="sampled", shots=1000, seed=1)
    assert result.shots == 1000 and result.seed == 1
    assert np.isclose(sum(result.joint.values()), 1.0)


def test_mixed_states_cannot_be_compiled(aer, design):
    _, measurements = design
    state = amplitudes(1, 1).apply_channel(
        __import__("qcog.core", fromlist=["depolarizing_channel"]).depolarizing_channel(0.2, 2).kraus
    )
    with pytest.raises(ValueError):
        AerBackend().build_circuit(state, measurements)


def test_partial_measurement_does_not_disturb_the_other_qubit(aer):
    """A question about one subsystem must not collapse the other one."""
    from qcog.core import tensor
    from qcog.models.common import local_question

    state = tensor(amplitudes(0.6, 0.8), basis_state(0, 2))
    measurements = [local_question(0.7, 0, 2, "A"), local_question(np.pi / 2, 1, 2, "B")]
    exact = run_sequence(state, measurements)
    compiled = AerExactBackend().run(state, measurements)
    for key in exact.joint:
        assert abs(exact.joint[key] - compiled.joint.get(key, 0.0)) < 1e-12
    sampled = AerBackend().run(state, measurements, shots=20000, seed=4)
    worst = max(abs(exact.joint[key] - sampled.joint.get(key, 0.0)) for key in exact.joint)
    assert worst < 5.0 * np.sqrt(0.25 / 20000)


def test_multi_qubit_sequence_matches_exact(aer):
    from qcog.core import tensor
    from qcog.models.common import local_question

    state = tensor(amplitudes(0.6, 0.8, label="two level"), basis_state(0, 2))
    measurements = [local_question(0.7, 0, 2, "A"), local_question(np.pi / 2, 1, 2, "B")]
    exact = run_sequence(state, measurements)
    circuit = AerExactBackend().run(state, measurements)
    for key in exact.joint:
        assert abs(exact.joint[key] - circuit.joint.get(key, 0.0)) < 1e-12


def test_between_question_decoherence_removes_conditional_structure(design):
    state, measurements = design
    noisy = run_sequence(state, measurements, noise=NoiseSpec("depolarizing", 1.0, 1, "each_question"))
    gap = abs(
        noisy.conditional("B", {"A": "yes"})["yes"] - noisy.conditional("B", {"A": "no"})["yes"]
    )
    assert gap < 1e-12


def test_num_qubits_for_dim():
    assert num_qubits_for_dim(2) == 1
    assert num_qubits_for_dim(3) == 2
    assert num_qubits_for_dim(4) == 2