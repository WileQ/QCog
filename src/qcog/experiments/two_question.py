"""The workhorse design: two questions, asked in both orders.

A surprising share of quantum-cognition results - question-order effects, the
conjunction fallacy, framing effects, the disjunction effect, sequential
categorisation, contradictory beliefs - are the same experiment with different
dressing: a state, two incompatible questions, and a comparison between the two
orders.  Implementing that design once, well tested, means a new model is a
configuration plus an explanation rather than a new simulation engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ..analysis.interference import InterferenceReport, interference_decomposition
from ..analysis.order import (
    OrderEffectReport,
    QQReport,
    law_of_total_probability_test,
    order_effect,
    qq_equality,
)
from ..core.classical import law_of_total_probability
from ..core.engine import Engine
from ..core.measurement import Measurement, incompatibility
from ..core.results import SequentialResult
from ..core.states import CognitiveState, as_state

__all__ = ["TwoQuestionExperiment", "OrderEffectBundle"]


@dataclass
class OrderEffectBundle:
    """All comparisons between the two orders of one two-question experiment."""

    forward: SequentialResult
    backward: SequentialResult
    report: OrderEffectReport
    qq: QQReport
    ltp: list
    interference: InterferenceReport | None = None
    classical: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "order_effect": self.report.to_dict(),
            "qq_equality": self.qq.to_dict(),
            "ltp_violations": [item.to_dict() for item in self.ltp],
            "interference": None if self.interference is None else self.interference.to_dict(),
            "classical": self.classical,
        }

    def summary(self) -> str:
        return f"{self.report.summary()} {self.qq.summary()}"


@dataclass
class TwoQuestionExperiment:
    """A belief state and two questions that can be asked in either order.

    Parameters
    ----------
    state:
        The belief state of the participant (pure or mixed).
    question_a, question_b:
        The two questions.  Their incompatibility is what allows the order to
        matter: if they commute, the model is classical and the order effect is
        exactly zero.
    labels:
        Optional ``{"state": ..., "a": ..., "b": ...}`` labels used in figures.
    """

    state: CognitiveState
    question_a: Measurement
    question_b: Measurement
    labels: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.state = as_state(self.state)
        if self.question_a.dim != self.state.dim or self.question_b.dim != self.state.dim:
            raise ValueError("questions and state must live in the same Hilbert space")

    # ------------------------------------------------------------------ running
    def run(self, engine: Engine, order: Sequence[str] = ("A", "B")) -> SequentialResult:
        """Simulate the two questions in the requested order."""
        questions = {"A": self.question_a, "B": self.question_b}
        if tuple(order) not in {("A", "B"), ("B", "A")}:
            raise ValueError("order must be ('A', 'B') or ('B', 'A')")
        return engine.run(self.state, [questions[name] for name in order], measurements=questions)

    def run_both_orders(self, engine: Engine) -> dict[str, SequentialResult]:
        """Simulate both orders."""
        forward = engine.run(self.state, [self.question_a, self.question_b])
        backward = engine.run(self.state, [self.question_b, self.question_a])
        lookup = {self.question_a.name: self.question_a, self.question_b.name: self.question_b}
        lookup.update({"A": self.question_a, "B": self.question_b})
        forward.metadata["measurements"] = lookup
        backward.metadata["measurements"] = lookup
        return {"A_then_B": forward, "B_then_A": backward}

    # ------------------------------------------------------------------ analysis
    @property
    def incompatibility(self) -> float:
        """How incompatible the two questions are (zero for a classical pair)."""
        return float(incompatibility([self.question_a, self.question_b]))

    def interference(self) -> InterferenceReport:
        """Interference decomposition of the second question's probabilities."""
        return interference_decomposition(self.state, self.question_a, self.question_b)

    def analyse(self, results: Mapping[str, SequentialResult]) -> OrderEffectBundle:
        """Compare the two orders and decompose the difference."""
        forward = results["A_then_B"]
        backward = results["B_then_A"]
        report = order_effect(forward, backward, self.question_a.name, self.question_b.name)
        qq = qq_equality(forward, backward, self.question_a.name, self.question_b.name)
        ltp = law_of_total_probability_test(forward, backward, self.question_a.name, self.question_b.name)
        interference = None
        if self.state.is_pure:
            interference = self.interference()
        classical = self.classical_comparison(forward, backward)
        return OrderEffectBundle(forward, backward, report, qq, ltp, interference, classical)

    def classical_comparison(
        self, forward: SequentialResult, backward: SequentialResult
    ) -> dict:
        """What a classical model predicts for the same two marginals.

        A classical participant answers both questions from one sample space, so
        the marginal of the first question must be the same in both orders and the
        joint must factor through the conditionals.  The reported numbers are the
        size of the resulting contradiction.
        """
        first = self.question_a.name
        second = self.question_b.name
        direct = forward.marginal(first)
        reversed_marginal = backward.marginal(first)
        classical_joint = {
            f"{key[0]}|{key[1]}": value for key, value in backward.joint.items()
        }
        return {
            "name": "single sample space (commuting questions)",
            "description": (
                "A classical model answers both questions from one joint distribution, so the "
                "marginal of a question cannot depend on which question came first."
            ),
            "marginal_difference": {
                outcome: round(float(reversed_marginal[outcome] - direct[outcome]), 8)
                for outcome in direct
            },
            "predicted_total_variation": 0.0,
            "classical_joint_for_reference": classical_joint,
        }

    # ------------------------------------------------------------------- helpers
    def describe(self) -> str:
        """Human-readable description of the design."""
        state = self.labels.get("state", self.state.label)
        first = self.labels.get("a", self.question_a.name)
        second = self.labels.get("b", self.question_b.name)
        return (
            f"State {state!r}, questions {first!r} and {second!r}; "
            f"incompatibility {self.incompatibility:.4f}."
        )