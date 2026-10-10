"""Execution backends: analytic, sampled, Aer and (optionally) IonQ.

A backend is anything that can turn ``(state, [questions])`` into a
:class:`~qcog.core.results.SequentialResult`.  Models never care which one is
used, which is what makes ``qcog run question_order --backend aer`` a one-flag
experiment.
"""

from __future__ import annotations

from .aer import AER_METHODS, AerBackend, AerExactBackend, aer_noise_model
from .exact import ExactBackend, SampledBackend
from .ionq import IonQBackend, ionq_available

__all__ = [
    "AER_METHODS",
    "AerBackend",
    "AerExactBackend",
    "aer_noise_model",
    "ExactBackend",
    "SampledBackend",
    "IonQBackend",
    "ionq_available",
    "BACKENDS",
    "get_backend",
    "list_backends",
    "backend_table",
]

BACKENDS = {
    "exact": ExactBackend,
    "sampled": SampledBackend,
    "aer": AerBackend,
    "aer_exact": AerExactBackend,
    "ionq": IonQBackend,
}


def get_backend(name: str = "exact", **options):
    """Instantiate a backend by name.

    Accepts the short names in :data:`BACKENDS`, plus ``aer[statevector]`` style
    labels and the aliases ``"aer_statevector"`` / ``"aer_density_matrix"``.
    """
    key = str(name or "exact").strip().lower()
    if key in {"aer_statevector", "aer-statevector"}:
        return AerBackend(method="statevector")
    if key in {"aer_density_matrix", "aer-density-matrix"}:
        return AerBackend(method="density_matrix")
    if key.startswith("aer[") and key.endswith("]"):
        return AerBackend(method=key[4:-1])
    if key not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; available: {sorted(BACKENDS)}")
    return BACKENDS[key](**options)


def list_backends(available_only: bool = False) -> list[str]:
    """Names of the available backends."""
    names = ["exact", "sampled", "aer", "aer_exact", "aer_statevector", "aer_density_matrix"]
    try:
        if ionq_available():
            names.append("ionq")
    except Exception:  # pragma: no cover - defensive
        pass
    return names if not available_only else names


def backend_table() -> list[dict]:
    """Description of every backend, for the CLI, the UI and the docs."""
    rows = [
        {
            "name": "exact",
            "class": "ExactBackend",
            "kind": "analytic",
            "shots": False,
            "notes": "Born rule + Lueders update on the projectors; the reference implementation.",
        },
        {
            "name": "sampled",
            "class": "SampledBackend",
            "kind": "analytic + sampling",
            "shots": True,
            "notes": "Analytic distribution resampled with a multinomial draw; models finite participant pools.",
        },
        {
            "name": "aer",
            "class": "AerBackend",
            "kind": "circuit",
            "shots": True,
            "notes": "Compiled Qiskit circuit on AerSimulator; cross-checks the entire compilation path.",
        },
        {
            "name": "aer_exact",
            "class": "AerExactBackend",
            "kind": "circuit (exact)",
            "shots": False,
            "notes": "Exact probabilities read from the statevectors of the compiled prefix circuits; no shot noise.",
        },
        {
            "name": "aer_statevector",
            "class": "AerBackend(method='statevector')",
            "kind": "circuit",
            "shots": True,
            "notes": "Same circuit with the statevector simulation method (mid-circuit collapse included).",
        },
        {
            "name": "ionq",
            "class": "IonQBackend",
            "kind": "device",
            "shots": True,
            "notes": "Optional: IonQ simulator or hardware through qiskit-ionq; needs IONQ_API_KEY.",
        },
    ]
    return rows