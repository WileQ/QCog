"""Classical baselines.

Every Q-Cog model reports what a *classical* account of the same situation
predicts, for two reasons.  First, it is the honest way to present a quantum
model: if the classical account fits just as well, say so.  Second, the
difference between the two is what "quantum" means quantitatively here - usually
an interference term or a failure of the law of total probability.

The baselines are deliberately simple and well documented so that a reader can
reproduce any of them with pen and paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

__all__ = [
    "law_of_total_probability",
    "LTPReport",
    "markov_joint",
    "sample_space_joint",
    "ClassicalConjunction",
    "de_groot_step",
    "bounded_confidence_step",
    "ClassicalPrisonersDilemma",
    "condorcet_jury_probability",
    "bayesian_cascade",
    "ClassicalBaseline",
]


@dataclass
class LTPReport:
    """Result of the law-of-total-probability test for one question.

    ``indirect`` is the prediction a classical, order-free account makes for the
    marginal of the first question, given the conditional structure observed when
    the other question is asked first.  A nonzero ``violation`` is the standard
    signature of question-order effects.
    """

    question: str
    outcome: str
    direct: float
    indirect: float
    conditional_on: str

    @property
    def violation(self) -> float:
        """``direct - indirect``; zero for any classical model."""
        return float(self.direct - self.indirect)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "outcome": self.outcome,
            "direct": round(self.direct, 8),
            "indirect": round(self.indirect, 8),
            "violation": round(self.violation, 8),
            "conditional_on": self.conditional_on,
        }

    def summary(self) -> str:
        return (
            f"p({self.question}={self.outcome}) measured directly = {self.direct:.4f}, "
            f"but computed from the other order = {self.indirect:.4f} "
            f"(violation {self.violation:+.4f})"
        )


def law_of_total_probability(
    question: str,
    outcome: str,
    direct: float,
    other_marginal: Mapping[str, float],
    conditional: Mapping[str, Mapping[str, float]],
    conditional_on: str = "other question",
) -> LTPReport:
    """Compare a directly measured marginal with the classical prediction.

    Parameters
    ----------
    direct:
        ``p(question = outcome)`` measured when ``question`` is asked first.
    other_marginal:
        Marginal of the other question in the *reversed* order.
    conditional:
        ``p(question = outcome | other = value)`` from the reversed order.
    """
    indirect = float(
        sum(
            other_marginal[value] * conditional[value].get(outcome, 0.0)
            for value in other_marginal
        )
    )
    return LTPReport(question, outcome, float(direct), indirect, conditional_on)


def markov_joint(
    first_marginal: Mapping[str, float], conditional: Mapping[str, Mapping[str, float]]
) -> dict[tuple[str, str], float]:
    """Classical joint distribution ``p(a, b) = p(a) p(b | a)``."""
    joint: dict[tuple[str, str], float] = {}
    for a, p_a in first_marginal.items():
        for b, p_b in conditional.get(a, {}).items():
            joint[(a, b)] = float(p_a) * float(p_b)
    return joint


def sample_space_joint(marginal: Mapping[str, float]) -> dict[tuple[str, str], float]:
    """A *classical* joint for two questions with the given identical marginals.

    In a classical model the two questions are answered from a single sample
    space, so the marginal of the first question cannot depend on whether it was
    asked first or second.  This function makes that constraint explicit, and its
    disagreement with the simulated data is what :func:`qcog.analysis.order_effect`
    reports as the order effect.
    """
    outcomes = list(marginal)
    joint: dict[tuple[str, str], float] = {}
    for left in outcomes:
        for right in outcomes:
            joint[(left, right)] = float(marginal[left] * marginal[right])
    return joint


@dataclass
class ClassicalConjunction:
    """The classical bound violated by the conjunction fallacy.

    ``p(A and B) <= min(p(A), p(B))`` holds in every Kolmogorov probability
    space.  A model that predicts otherwise is not classical - whether it is
    quantum is a separate question.
    """

    name: str = "classical conjunction bound"

    def bound(self, p_first: float, p_second: float) -> float:
        return float(min(p_first, p_second))

    def is_violated(self, p_conjunction: float, p_first: float, p_second: float, tol: float = 0.0) -> bool:
        return bool(p_conjunction - self.bound(p_first, p_second) > tol)

    def to_dict(self, p_conjunction: float, p_first: float, p_second: float) -> dict:
        bound = self.bound(p_first, p_second)
        return {
            "model": self.name,
            "p_conjunction": round(float(p_conjunction), 6),
            "classical_bound": round(bound, 6),
            "excess": round(float(p_conjunction) - bound, 6),
            "violated": self.is_violated(p_conjunction, p_first, p_second),
        }


def de_groot_step(opinions: np.ndarray, adjacency: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """One step of DeGroot learning: every agent averages its neighbours' opinions.

    The classical reference model for opinion dynamics: opinions converge to the
    same value on a connected graph, so the final spread measures consensus.
    """
    opinions = np.asarray(opinions, dtype=float)
    adjacency = np.asarray(adjacency, dtype=float)
    if weights is not None:
        adjacency = adjacency * np.asarray(weights, dtype=float)
    row_sums = adjacency.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return adjacency @ opinions / row_sums.ravel()


def bounded_confidence_step(opinions: np.ndarray, epsilon: float = 0.3, adjacency: np.ndarray | None = None) -> np.ndarray:
    """Hegselmann-Krause bounded-confidence update.

    Agents only listen to other agents whose opinion is within ``epsilon``.  It is
    the standard classical model of polarisation: clusters form whose size
    depends on the confidence bound.
    """
    opinions = np.asarray(opinions, dtype=float)
    count = opinions.size
    updated = np.zeros_like(opinions)
    for index in range(count):
        close = np.abs(opinions - opinions[index]) <= epsilon
        if adjacency is not None:
            close &= np.asarray(adjacency, dtype=float)[index] > 0
        close[index] = True
        updated[index] = opinions[close].mean()
    return updated


@dataclass
class ClassicalPrisonersDilemma:
    """The classical prisoner's dilemma, used as the baseline for the quantum version.

    Payoff matrix (player, opponent): ``(C, C) = 3``, ``(D, D) = 1``,
    ``(D, C) = 5``, ``(C, D) = 0``.  Defection is the unique Nash equilibrium,
    with payoff ``1`` each, whereas cooperating would give ``3`` each.
    """

    temptation: float = 5.0
    reward: float = 3.0
    punishment: float = 1.0
    sucker: float = 0.0

    name: str = "classical prisoner's dilemma"

    def payoffs(self, strategy_first: str, strategy_second: str) -> tuple[float, float]:
        """Payoffs ``(first, second)`` for the two pure strategies ``"C"``/``"D"``."""
        table = {
            ("C", "C"): (self.reward, self.reward),
            ("D", "D"): (self.punishment, self.punishment),
            ("D", "C"): (self.temptation, self.sucker),
            ("C", "D"): (self.sucker, self.temptation),
        }
        key = (strategy_first.upper(), strategy_second.upper())
        if key not in table:
            raise ValueError(f"unknown strategy profile {key}")
        return table[key]

    def payoff_matrix(self) -> np.ndarray:
        """``(2, 2)`` payoff matrix for the row player (``C`` then ``D``)."""
        return np.array(
            [
                [self.reward, self.sucker],
                [self.temptation, self.punishment],
            ],
            dtype=float,
        )

    def nash_equilibrium(self) -> dict:
        """The classical mixed-strategy analysis: mutual defection is dominant."""
        return {
            "strategies": ("C", "D"),
            "pure_nash": ("D", "D"),
            "payoffs": {"D": self.punishment, "C": self.reward},
            "cooperation": 0.0,
            "comment": (
                "Any mixture with positive defection probability weakly dominates cooperation, "
                "so mutual defection is the unique Nash equilibrium."
            ),
        }

    def to_dict(self) -> dict:
        return {
            "model": self.name,
            "payoff_matrix": self.payoff_matrix().tolist(),
            "nash_equilibrium": self.nash_equilibrium(),
        }


def condorcet_jury_probability(individual_accuracy: float, jurors: int) -> float:
    """Probability a majority of ``jurors`` independent voters is correct.

    The classical aggregation baseline used by the collective-judgement model.
    """
    if not 0.0 <= individual_accuracy <= 1.0:
        raise ValueError("individual_accuracy must lie in [0, 1]")
    if jurors <= 0 or jurors % 2 == 0:
        raise ValueError("jurors must be a positive odd number")
    from math import comb

    total = 0.0
    for correct in range(jurors // 2 + 1, jurors + 1):
        total += comb(jurors, correct) * individual_accuracy ** correct * (1 - individual_accuracy) ** (
            jurors - correct
        )
    return float(total)


def bayesian_cascade(
    private_signal_accuracy: float,
    decisions: Sequence[str],
    signal: str,
    labels: Sequence[str] = ("A", "B"),
) -> dict:
    """Classical Bayesian information cascade.

    Given the decisions already made, compute the posterior over the true state
    and the probability that the next agent follows the private signal.  A
    cascade occurs when the private signal no longer changes the decision.
    """
    first_label, second_label = labels
    if first_label == second_label:
        raise ValueError("the two labels must differ")
    prior = 0.5
    follows_cascade = True
    for decision in decisions:
        if decision == signal:
            follows_cascade = False
        prior = prior  # decisions are treated as observations in the exact version below
    counts_first = sum(1 for decision in decisions if decision == first_label)
    counts_second = len(decisions) - counts_first
    likelihood = private_signal_accuracy
    odds = (prior / (1 - prior)) * (likelihood / (1 - likelihood)) ** (counts_first - counts_second)
    posterior_first = odds / (1 + odds)
    return {
        "posterior": posterior_first if signal == first_label else 1 - posterior_first,
        "cascade_broken_by_private_signal": not follows_cascade,
        "decision_counts": {first_label: counts_first, second_label: counts_second},
    }


@dataclass
class ClassicalBaseline:
    """Container describing the classical account a model is compared against."""

    name: str
    description: str
    prediction: dict = field(default_factory=dict)
    references: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "prediction": {key: round(value, 8) if isinstance(value, float) else value
                           for key, value in self.prediction.items()},
            "references": list(self.references),
        }