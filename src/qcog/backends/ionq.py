"""Optional execution on IonQ backends through the official Qiskit provider.

Q-Cog is fully runnable on a free local simulator; this module exists because the
same circuits should be portable to IonQ without a rewrite, and
because the re-preparation compilation in
:func:`qcog.circuits.builder.sequence_circuit_with_repreparation` avoids relying
on hardware mid-circuit measurement.

Usage
-----
>>> from qcog.backends import IonQBackend            # doctest: +SKIP
>>> backend = IonQBackend("ionq_simulator")          # doctest: +SKIP
>>> backend.available()                              # doctest: +SKIP
True
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

from ..circuits.builder import build_sequence_circuit, circuit_statistics, decode_counts
from ..core.linalg import num_qubits_for_dim
from ..core.measurement import Measurement
from ..core.results import SequentialResult
from ..core.states import as_state

__all__ = ["IonQBackend", "ionq_available"]


def ionq_available() -> bool:
    """True when ``qiskit-ionq`` is installed and an API key is configured."""
    try:
        import qiskit_ionq  # noqa: F401
    except Exception:
        return False
    return bool(os.environ.get("IONQ_API_KEY"))


class IonQBackend:
    """Run a compiled interview on an IonQ simulator or QPU."""

    name = "ionq"
    description = "IonQ simulator or hardware via the qiskit-ionq provider"

    def __init__(self, backend_name: str = "ionq_simulator", token: str | None = None, **options) -> None:
        self.backend_name = backend_name
        self.token = token or os.environ.get("IONQ_API_KEY")
        self.options = options

    def available(self) -> bool:
        """Whether this backend can be constructed in the current environment."""
        try:
            import qiskit_ionq  # noqa: F401
        except Exception:
            return False
        return bool(self.token)

    def _provider_backend(self):
        if not self.available():
            raise RuntimeError(
                "IonQ backend unavailable: install the optional dependency with "
                '`pip install -e ".[ionq]"` and set the IONQ_API_KEY environment variable.'
            )
        from qiskit_ionq import IonQProvider

        return IonQProvider(self.token).get_backend(self.backend_name)

    def run(
        self,
        state,
        measurements: Sequence[Measurement],
        *,
        shots: int = 1024,
        seed: int | None = None,
        mid_circuit_measurement: bool = False,
        **metadata,
    ) -> SequentialResult:
        """Submit the compiled circuit and decode the returned counts.

        ``mid_circuit_measurement=False`` is the safe default for hardware: the
        circuit then prepares each post-measurement state explicitly.
        """
        from qiskit import transpile

        confidence = as_state(state)
        questions = list(measurements)
        circuit = build_sequence_circuit(
            confidence, questions, mid_circuit_measurement=mid_circuit_measurement
        )
        device = self._provider_backend()
        compiled = transpile(circuit, backend=device, optimization_level=3, seed_transpiler=seed)
        job = device.run(compiled, shots=shots)
        counts = job.result().get_counts()
        joint_counts = decode_counts(
            counts, questions, num_qubits_for_dim(confidence.dim), circuit.metadata.get("qcog_layout")
        )
        joint = {key: value / shots for key, value in joint_counts.items()}
        return SequentialResult(
            order=tuple(question.name for question in questions),
            outcomes={question.name: tuple(question.outcomes) for question in questions},
            joint=joint,
            backend=f"ionq[{self.backend_name}]",
            shots=shots,
            seed=seed,
            counts=joint_counts,
            noise=None,
            metadata={**metadata, "circuit": circuit_statistics(compiled), "ionq": True,
                      "backend_name": self.backend_name},
        )