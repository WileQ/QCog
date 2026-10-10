"""Shared building blocks for the model catalogue.

Two things live here: small helpers that turn numbers into sentences (so that
every model explains itself in the same voice), and the recurring designs that
several models reuse.  Keeping the wording in one place also keeps the caveats
consistent, which matters: a library that models human judgement has to be
careful not to over-claim, in every one of its twenty models.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..core.measurement import Measurement, ProjectiveMeasurement, unsharp_question
from ..core.states import CognitiveState, amplitudes, from_bloch

__all__ = [
    "qubit",
    "measurement_for",
    "percent",
    "signed",
    "interference_sentence",
    "order_effect_findings",
    "classical_sentence",
    "default_caveats",
    "amplitude_state",
]

#: The caveat every model in the library carries.
default_caveats = (
    "This is a model of response probabilities, not a claim about what the brain does.",
    "A quantum model only means the data are described by a quantum probability space; "
    "classical models may fit equally well, and the classical baseline is reported for that reason.",
)


def percent(value: float) -> str:
    """Format a probability as a percentage with one decimal."""
    return f"{100.0 * float(value):.1f}%"


def signed(value: float) -> str:
    """Format a number with an explicit sign."""
    return f"{float(value):+.3f}"


def qubit(angle: float = 0.0, phi: float = 0.0, label: str = "belief", basis_labels=None) -> CognitiveState:
    """A qubit belief state at the given Bloch angles."""
    return from_bloch(angle, phi, label=label)


def amplitude_state(a: complex, b: complex, label: str = "belief", basis_labels=None) -> CognitiveState:
    """A qubit state from two amplitudes."""
    return amplitudes(a, b, label=label, basis_labels=basis_labels)


def measurement_for(
    angle: float,
    name: str,
    outcomes: Sequence[str] = ("yes", "no"),
    description: str = "",
    sharpness: float = 1.0,
    phi: float = 0.0,
) -> Measurement:
    """A binary qubit question, optionally unsharp.

    ``sharpness < 1`` returns a POVM: the question is asked, but with a blurred
    answer key.  Sweeping the sharpness from 1 to 0 is how the library shows how
    much "quantum question" a phenomenon really needs.
    """
    from ..core.measurement import question as projective_question

    if sharpness >= 1.0:
        return projective_question(angle, phi, name, outcomes, description)
    return unsharp_question(angle, sharpness, phi, name, outcomes, description)


def interference_sentence(report) -> str:
    """Turn an :class:`InterferenceReport` into one explanatory sentence."""
    if report is None:
        return ""
    maxima = max(report.totals, key=lambda key: report.totals[key])
    share = report.relative_interference()[maxima]
    return (
        f"The classical (law-of-total-probability) part of p({report.second}={maxima}) is "
        f"{report.direct[maxima]:.3f} and the interference term is {signed(report.interference[maxima])}, "
        f"i.e. interference supplies {percent(share)} of the probability. Interference can be negative, "
        "which is what makes the two question orders disagree."
    )


def order_effect_findings(bundle) -> list[str]:
    """Findings that every two-question model can report."""
    findings: list[str] = []
    report = bundle.report
    shift = report.max_marginal_shift()
    top = max(report.marginal_first_in_first_order, key=lambda key: report.marginal_first_in_first_order[key])
    if shift < 1e-9:
        findings.append(
            f"Asking {bundle.backward.order[0]} first leaves p({report.first}) unchanged: the two questions "
            f"behave as compatible (classical) questions. Incompatibility = {bundle.report.incompatibility:.2e}."
        )
    else:
        findings.append(
            f"Asking the other question first moves p({report.first} = {top}) by "
            f"{signed(report.marginal_shift(top))} (TV distance {report.total_variation:.4f}). "
            f"Question incompatibility = {report.incompatibility:.4f}."
        )
    findings.append(bundle.qq.summary())
    for item in bundle.ltp:
        if abs(item.violation) > 1e-9:
            findings.append(f"Law of total probability violated: {item.summary()}")
    if bundle.interference is not None:
        findings.append(interference_sentence(bundle.interference))
    return findings


def classical_sentence(bundle) -> str:
    """One sentence about what a classical model requires."""
    report = bundle.report
    shift = report.max_marginal_shift()
    return (
        "A classical model answers both questions from a single joint distribution, so it predicts a "
        f"marginal shift of exactly 0; the model predicts {shift:.4f}. "
        "The (unfitted) classical baseline is therefore rejected by construction whenever this is nonzero - "
        "which is a statement about the model, not yet about people."
    )

def local_question(
    theta: float,
    qubit: int,
    num_qubits: int,
    name: str,
    outcomes: Sequence[str] = ("yes", "no"),
    phi: float = 0.0,
    description: str = "",
):
    """A single-qubit question embedded in a larger register.

    Qubits are little-endian (matching Qiskit), so the operator for the least
    significant qubit is the rightmost Kronecker factor.
    """
    from ..core.measurement import ProjectiveMeasurement
    from ..core.measurement import question as _question

    base = _question(theta, phi, name, outcomes, description)
    projectors = []
    for projector in base.projectors:
        left = np.eye(2 ** (num_qubits - qubit - 1), dtype=complex)
        right = np.eye(2**qubit, dtype=complex)
        projectors.append(np.kron(np.kron(left, projector), right))
    return ProjectiveMeasurement(name, outcomes, tuple(projectors), description,
                                 {"source": "local", "qubit": qubit, "theta": float(theta)})


def local_unsharp_question(
    theta: float,
    qubit: int,
    num_qubits: int,
    sharpness: float,
    name: str,
    outcomes: Sequence[str] = ("yes", "no"),
    description: str = "",
):
    """An unsharp single-qubit question embedded in a larger register."""
    from ..core.measurement import POVM, unsharp_question

    base = unsharp_question(theta, sharpness, 0.0, name, outcomes, description)
    effects = []
    for effect in base.effects:
        left = np.eye(2 ** (num_qubits - qubit - 1), dtype=complex)
        right = np.eye(2**qubit, dtype=complex)
        effects.append(np.kron(np.kron(left, effect), right))
    return POVM(name, outcomes, tuple(effects), description,
                {"source": "local-unsharp", "qubit": qubit, "sharpness": float(sharpness)})


def joint_ray_question(name: str, rays: Sequence[np.ndarray], outcomes: Sequence[str], description: str = ""):
    """A dichotomous-or-more question whose outcomes are given rays."""
    from ..core.measurement import ray_measurement

    return ray_measurement(name, rays, outcomes, description)