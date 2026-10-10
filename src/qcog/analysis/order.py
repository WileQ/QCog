"""Order effects, the law of total probability, and the QQ equality.

The tools here are what turn "the two orders give different answers" into a
number, and what let a user check a *parameter-free* prediction of the quantum
model against data.

The QQ equality
---------------
Write ``A`` and ``B`` for two dichotomous questions with outcomes ``yes``/``no``,
and let ``p_AB`` be the joint distribution observed when ``A`` is asked first.
The quantum model predicts

    p_AB(A=yes, B=yes) + p_AB(A=no, B=no)
        = p_BA(B=yes, A=yes) + p_BA(B=no, A=no)

that is: *the probability of answering both questions the same way does not
depend on the order*.  Classical models have no reason to obey this, and unlike a
fit statistic it involves no free parameters, so it is a genuine test rather than
a description.  :func:`qq_equality` reports the two sides, their difference, and -
for shot-based results - a standard error so that a user can see whether a
discrepancy is real or noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ..core.measurement import Measurement, incompatibility
from ..core.metrics import agreement_index, total_variation
from ..core.results import SequentialResult
from ..core.states import as_state

__all__ = [
    "OrderEffectReport",
    "QQReport",
    "order_effect",
    "qq_equality",
    "law_of_total_probability_test",
    "sure_thing_test",
]


@dataclass
class OrderEffectReport:
    """Everything Q-Cog knows about one pair of questions asked in both orders."""

    first: str
    second: str
    marginal_first_in_first_order: dict[str, float]
    marginal_first_in_second_order: dict[str, float]
    conditional_first_order: dict[str, dict[str, float]]
    conditional_second_order: dict[str, dict[str, float]]
    joint_first_order: dict[tuple, float]
    joint_second_order: dict[tuple, float]
    incompatibility: float
    metadata: dict = field(default_factory=dict)

    @property
    def total_variation(self) -> float:
        """Total-variation distance between the two marginal distributions of the first question."""
        return total_variation(self.marginal_first_in_first_order, self.marginal_first_in_second_order)

    def marginal_shift(self, outcome: str) -> float:
        """Change in ``p(first = outcome)`` caused purely by asking the other question first."""
        return float(
            self.marginal_first_in_second_order.get(outcome, 0.0)
            - self.marginal_first_in_first_order.get(outcome, 0.0)
        )

    def max_marginal_shift(self) -> float:
        """Largest absolute marginal shift over the outcomes of the first question."""
        return float(
            max(
                (abs(self.marginal_shift(outcome)) for outcome in self.marginal_first_in_first_order),
                default=0.0,
            )
        )

    @property
    def delta_first_order(self) -> float:
        """``Delta_AB``: how strongly the second answer depends on the first."""
        return _delta(self.conditional_first_order)

    @property
    def delta_second_order(self) -> float:
        """``Delta_BA``: the same quantity computed in the reversed order."""
        return _delta(self.conditional_second_order)

    def to_dict(self) -> dict:
        return {
            "first": self.first,
            "second": self.second,
            "total_variation": round(self.total_variation, 8),
            "max_marginal_shift": round(self.max_marginal_shift(), 8),
            "marginal_first_in_first_order": {k: round(v, 8) for k, v in self.marginal_first_in_first_order.items()},
            "marginal_first_in_second_order": {k: round(v, 8) for k, v in self.marginal_first_in_second_order.items()},
            "delta_first_order": round(self.delta_first_order, 8),
            "delta_second_order": round(self.delta_second_order, 8),
            "incompatibility": round(self.incompatibility, 8),
            **({"metadata": self.metadata} if self.metadata else {}),
        }

    def summary(self) -> str:
        shift = self.max_marginal_shift()
        if shift < 1e-9:
            return (
                f"No order effect: asking {self.second} before {self.first} leaves the marginal of "
                f"{self.first} unchanged (questions are compatible, incompatibility "
                f"{self.incompatibility:.2e})."
            )
        return (
            f"Order effect of {shift:.4f} in p({self.first} = most likely outcome): "
            f"the {self.first}-then-{self.second} and {self.second}-then-{self.first} orders disagree "
            f"(TV distance {self.total_variation:.4f}, question incompatibility {self.incompatibility:.4f})."
        )


def _delta(conditional: Mapping[str, Mapping[str, float]]) -> float:
    """``p(second = yes | first = yes) - p(second = yes | first = no)``.

    Falls back to the most likely outcome of the second question when the
    outcomes are not literally labelled "yes"/"no", so the same measure works for
    any dichotomous pair of questions.
    """
    first_outcomes = sorted(conditional)
    if len(first_outcomes) != 2:
        return float("nan")
    second_outcomes = conditional[first_outcomes[0]]
    if not second_outcomes:
        return float("nan")
    target = "yes" if "yes" in second_outcomes else max(second_outcomes, key=lambda key: second_outcomes[key])
    high = conditional[first_outcomes[0]].get(target, float("nan"))
    low = conditional[first_outcomes[1]].get(target, float("nan"))
    return float(high - low)


@dataclass
class QQReport:
    """The QQ equality: same-answer agreement across the two question orders.

    "The same way" means the first answer to both questions, or the second answer to both -
    pairing by position within each question, so the two questions may be worded differently
    ("satisfied / dissatisfied" and "would vote / would not vote", for instance).
    """

    first: str
    second: str
    agreement_first_order: float
    agreement_second_order: float
    standard_error: float = 0.0
    applicable: bool = True

    @property
    def difference(self) -> float:
        """``QQ = agreement(A then B) - agreement(B then A)``; the quantum prediction is 0."""
        return float(self.agreement_first_order - self.agreement_second_order)

    @property
    def satisfied(self) -> bool:
        """True when the difference is within two standard errors."""
        if not self.applicable:
            return False
        if self.standard_error <= 0:
            return abs(self.difference) < 1e-8
        return bool(abs(self.difference) <= 2.0 * self.standard_error)

    @property
    def z_score(self) -> float:
        """Difference in units of its standard error (0 when exact)."""
        if self.standard_error <= 0:
            return 0.0
        return float(self.difference / self.standard_error)

    def to_dict(self) -> dict:
        return {
            "first": self.first,
            "second": self.second,
            "agreement_first_order": round(self.agreement_first_order, 8),
            "agreement_second_order": round(self.agreement_second_order, 8),
            "applicable": bool(self.applicable),
            "difference": round(self.difference, 8),
            "standard_error": round(self.standard_error, 8),
            "z_score": round(self.z_score, 4),
            "satisfied": self.satisfied,
        }

    def summary(self) -> str:
        if not self.applicable:
            return (
                "QQ equality not applicable: it is defined for two dichotomous questions, and these two do "
                "not both have exactly two answers."
            )
        verdict = "holds" if self.satisfied else "is violated"
        return (
            f"QQ equality {verdict}: same-answer agreement is {self.agreement_first_order:.4f} in one "
            f"order and {self.agreement_second_order:.4f} in the other (difference "
            f"{self.difference:+.4f}, z = {self.z_score:+.2f})."
        )


def order_effect(
    result_first_order: SequentialResult,
    result_second_order: SequentialResult,
    first: str | None = None,
    second: str | None = None,
) -> OrderEffectReport:
    """Compare the same two questions asked in both orders."""
    first = first or result_first_order.order[0]
    second = second or result_first_order.order[1]
    if result_second_order.order[0] != second:  # pragma: no cover - defensive
        raise ValueError("the second result must ask the same two questions in the reversed order")
    return OrderEffectReport(
        first=first,
        second=second,
        marginal_first_in_first_order=result_first_order.marginal(first),
        marginal_first_in_second_order=result_second_order.marginal(first),
        conditional_first_order=result_first_order.conditional_table(first, second),
        conditional_second_order=result_second_order.conditional_table(second, first),
        joint_first_order=result_first_order.joint,
        joint_second_order=result_second_order.joint,
        incompatibility=_incompatibility_from_results(result_first_order, first, second),
        metadata={
            "backend": result_first_order.backend,
            "shots": result_first_order.shots,
        },
    )


def _incompatibility_from_results(result: SequentialResult, first: str, second: str) -> float:
    measures = result.metadata.get("measurements")
    if measures:
        return float(incompatibility([measures[first], measures[second]]))
    return float("nan")


def qq_equality(
    result_first_order: SequentialResult,
    result_second_order: SequentialResult,
    first: str | None = None,
    second: str | None = None,
) -> QQReport:
    """Evaluate the QQ equality between two question orders."""
    first = first or result_first_order.order[0]
    second = second or result_first_order.order[1]
    first_outcomes = tuple(result_first_order.outcomes[first])
    second_outcomes = tuple(result_first_order.outcomes[second])
    # the equality is about two dichotomous questions, paired answer by answer; the two
    # questions may be worded completely differently
    applicable = len(first_outcomes) == 2 and len(second_outcomes) == 2
    left = (
        _agreement(result_first_order, first, second, first_outcomes, second_outcomes)
        if applicable
        else float("nan")
    )
    right = (
        _agreement(result_second_order, first, second, first_outcomes, second_outcomes)
        if applicable
        else float("nan")
    )
    shots = result_first_order.shots
    standard_error = 0.0
    if shots and applicable:
        variance = (left * (1 - left) + right * (1 - right)) / shots
        standard_error = float(np.sqrt(max(variance, 0.0)))
    return QQReport(first, second, float(left), float(right), standard_error, applicable)


def _agreement(result, first: str, second: str, first_outcomes: tuple, second_outcomes: tuple) -> float:
    """Probability of giving the first answer to both questions, or the second to both.

    The two questions are paired answer by answer - the first answer of one with the first
    answer of the other - and the pairing is looked up *by question name*, so it does not
    matter in which order the two were asked.  This is what lets the QQ equality be checked
    for questions that share no wording at all.
    """
    first_axis = result.order.index(first)
    second_axis = result.order.index(second)
    total = 0.0
    for key, value in result.joint.items():
        both_first = key[first_axis] == first_outcomes[0] and key[second_axis] == second_outcomes[0]
        both_second = key[first_axis] == first_outcomes[1] and key[second_axis] == second_outcomes[1]
        if both_first or both_second:
            total += float(value)
    return float(total)


def law_of_total_probability_test(
    direct_result: SequentialResult,
    reversed_result: SequentialResult,
    question: str | None = None,
    other: str | None = None,
) -> list:
    """Law-of-total-probability violations for every outcome of ``question``.

    ``direct_result`` asks ``question`` first; ``reversed_result`` asks ``other``
    first.  A classical model must satisfy the identity for every outcome, so any
    nonzero value is a signature of contextuality.
    """
    from ..core.classical import law_of_total_probability

    question = question or direct_result.order[0]
    other = other or direct_result.order[1]
    direct_marginal = direct_result.marginal(question)
    other_marginal = reversed_result.marginal(other)
    conditional = reversed_result.conditional_table(other, question)
    return [
        law_of_total_probability(
            question, outcome, direct_marginal[outcome], other_marginal, conditional, conditional_on=other
        )
        for outcome in direct_marginal
    ]


def sure_thing_test(
    joint_with_condition: Mapping[str, float],
    conditioned: str,
    unconditional: float,
    positive: str = "yes",
    negative: str = "no",
) -> dict:
    """The sure-thing principle: ``p = p(positive) p_yes + p(negative) p_no``.

    A violation means the decision changed merely because an unresolved
    alternative was made definite - the disjunction effect.
    """
    positive_probability = float(joint_with_condition.get(f"{conditioned}={positive}", joint_with_condition.get(positive, 0.0)))
    negative_probability = float(joint_with_condition.get(f"{conditioned}={negative}", joint_with_condition.get(negative, 0.0)))
    predicted = positive_probability + negative_probability
    if positive_probability <= 0 and negative_probability <= 0:
        total = sum(joint_with_condition.values())
        predicted = predicted if predicted else total
    return {
        "sure_thing_prediction": predicted,
        "unconditional_choice": float(unconditional),
        "violation": float(unconditional - predicted),
        "violated": abs(unconditional - predicted) > 1e-9,
    }