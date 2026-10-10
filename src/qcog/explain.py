"""Turning numbers into sentences.

The explanation layer takes what a model computed and produces short, plain
sentences that say what happened, why, and what a classical account would have
said instead.  It is deliberately rule-based rather than generative: the text is a
direct function of the numbers (probabilities, interference terms, commutators), so
it can be unit-tested, it cannot hallucinate a number, and it is identical in the
CLI, the UI and the report.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .analysis.interference import InterferenceReport
from .analysis.order import OrderEffectReport, QQReport
from .core.measurement import Measurement, incompatibility
from .core.results import SequentialResult

__all__ = [
    "explain_order_effect",
    "explain_interference",
    "explain_qq",
    "explain_measurement_pair",
    "explain_backend",
    "summary_table",
]


def _percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def explain_order_effect(report: OrderEffectReport) -> str:
    """One paragraph explaining an order effect, or its absence."""
    if report.max_marginal_shift() < 1e-9:
        return (
            f"Answering {report.first} has the same distribution whether it is asked before or after "
            f"{report.second}. The two questions behave like classical questions that can be answered from a "
            f"single joint distribution, which is what the zero order effect means. Their incompatibility "
            f"measure is {report.incompatibility:.2e}."
        )
    top = max(report.marginal_first_in_first_order, key=lambda key: report.marginal_first_in_first_order[key])
    first = report.marginal_first_in_first_order[top]
    second = report.marginal_first_in_second_order[top]
    direction = "lower" if second < first else "higher"
    if not np.isfinite(report.incompatibility):
        return (
            f"p({report.first} = {top}) is {first:.3f} when {report.first} is asked first and {second:.3f} "
            f"when {report.second} is asked first, so asking the other question first makes this answer "
            f"{direction} by {abs(second - first):.3f}. A classical model with one joint distribution over the "
            f"two answers would force this difference to be zero."
        )
    return (
        f"p({report.first} = {top}) is {first:.3f} when {report.first} is asked first and {second:.3f} when "
        f"{report.second} is asked first, so asking the other question first makes this answer {direction} by "
        f"{abs(second - first):.3f}. The two questions do not commute, which is exactly why the answer can "
        f"depend on the context: their incompatibility measure is {report.incompatibility:.3f}. A classical "
        f"model with one joint distribution over the two answers would force this difference to be zero."
    )


def explain_interference(report: InterferenceReport) -> str:
    """One paragraph explaining where a probability came from."""
    if report is None:
        return ""
    outcome = max(report.totals, key=lambda key: report.totals[key])
    total = report.totals[outcome]
    direct = report.direct[outcome]
    cross = report.interference[outcome]
    share = abs(cross) / total if total > 0 else 0.0
    sign = "adds" if cross > 0 else "removes"
    return (
        f"p({report.second} = {outcome}) is {total:.3f}. Applying the law of total probability to the "
        f"{report.first} question accounts for {direct:.3f} of it; the remaining {cross:+.3f} is interference, "
        f"which {sign} probability. In other words {_percent(share)} of this answer comes from the two "
        f"{report.first} paths being able to reach the same {report.second} answer coherently rather than "
        f"alternatively. If the {report.first} answer were recorded - or measured unsharply, or lost to "
        f"decoherence - the interference term would shrink and the classical value {direct:.3f} would be what "
        f"is left."
    )


def explain_qq(report: QQReport) -> str:
    """One paragraph explaining the QQ equality test."""
    if not getattr(report, "applicable", True):
        return (
            "The QQ equality is not defined for these two questions because their outcome labels are not "
            "comparable; it applies to two dichotomous questions with the same two answers."
        )
    verdict = "consistent with the parameter-free quantum prediction" if report.satisfied else "violated"
    return (
        f"Answering both questions the same way happens with probability {report.agreement_first_order:.3f} in "
        f"one order and {report.agreement_second_order:.3f} in the other, a difference of "
        f"{report.difference:+.4f}. The quantum model predicts a difference of exactly zero with no free "
        f"parameters, and this run is {verdict}"
        + (f" (z = {report.z_score:+.2f})." if report.standard_error > 0 else ".")
    )


def explain_measurement_pair(first: Measurement, second: Measurement) -> str:
    """Explain what two questions have to do with each other."""
    value = incompatibility([first, second])
    if value < 1e-9:
        return (
            f"{first.name} and {second.name} commute, so they are jointly answerable: the model is classical "
            "for this pair and no order effect is possible."
        )
    return (
        f"{first.name} and {second.name} do not commute (incompatibility {value:.3f}), so they cannot be "
        "answered from one joint distribution. Whatever order effect the model predicts comes from this "
        "incompatibility and not from any additional assumption."
    )


def explain_backend(result: SequentialResult) -> str:
    """Explain how a result was produced."""
    if result.is_exact:
        return (
            f"These numbers are exact: the Born rule and the Lueders update were evaluated on the projectors "
            f"with Qiskit's statevector and density-matrix classes ({result.backend})."
        )
    return (
        f"These numbers come from {result.shots} shots with seed {result.seed} on the {result.backend} "
        f"backend. Each marginal therefore carries a standard error of about "
        f"{np.sqrt(0.25 / max(result.shots, 1)):.4f}, which is the right scale for judging whether a "
        f"difference is real."
    )


def summary_table(model_result) -> list[dict]:
    """Rows for the "what did this run show?" table in reports and the UI."""
    rows = [
        {"item": "model", "value": model_result.title},
        {"item": "summary", "value": model_result.summary},
        {"item": "backend", "value": model_result.backend},
        {"item": "parameters", "value": ", ".join(
            f"{key}={value}" for key, value in model_result.parameters.items() if not isinstance(value, float)
            or abs(value) > 1e-12
        )},
    ]
    for name, item in model_result.sequential_results().items():
        rows.append({"item": f"order: {name}", "value": item.summary()})
    return rows