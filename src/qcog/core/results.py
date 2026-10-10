"""Result containers shared by every backend.

A :class:`SequentialResult` is the single object the rest of the library works
with: models consume it to write explanations, the visualisation layer plots it,
the fitting layer compares it with data, and the CLI/UI serialise it.  Because
backends never leak their own data structures, an exact statevector run, a
shot-based run and a circuit run of the same model are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .states import CognitiveState

__all__ = ["StepRecord", "SequentialResult", "OutcomeTable", "result_from_diagonal"]


@dataclass
class StepRecord:
    """Conditional response distribution at one point in a simulated interview."""

    step: int
    measurement: str
    outcomes: tuple[str, ...]
    prefix: tuple[str, ...]
    conditional: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "measurement": self.measurement,
            "prefix": list(self.prefix),
            "outcomes": list(self.outcomes),
            "conditional": {key: round(value, 8) for key, value in self.conditional.items()},
        }


@dataclass
class SequentialResult:
    """Joint and conditional response distributions for one simulated interview.

    Parameters
    ----------
    order:
        The question order that was simulated, e.g. ``("A", "B")``.
    outcomes:
        Mapping from question name to its outcome labels.
    joint:
        Probability of every complete outcome tuple.
    steps:
        Conditional distribution at every reachable prefix (used for the
        "why did the answer change?" explanations and for plotting paths).
    states:
        Density matrix keyed by prefix, for inspecting how beliefs evolve.
    """

    order: tuple[str, ...]
    outcomes: dict[str, tuple[str, ...]]
    joint: dict[tuple[str, ...], float]
    steps: list[StepRecord] = field(default_factory=list)
    states: dict[tuple[str, ...], np.ndarray] = field(default_factory=dict)
    backend: str = "exact"
    shots: int | None = None
    seed: int | None = None
    counts: dict[tuple[str, ...], int] | None = None
    duration_seconds: float = 0.0
    noise: dict | None = None
    metadata: dict = field(default_factory=dict)

    # -------------------------------------------------------------- accessors
    @property
    def is_exact(self) -> bool:
        """True when the distribution is analytic rather than shot-sampled."""
        return self.counts is None

    @property
    def question_axis(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.order)}

    def outcome_combinations(self) -> list[tuple[str, ...]]:
        """All complete outcome tuples, in simulation order."""
        return [tuple(entry) for entry in self.joint]

    def probability(self, assignment: dict[str, str] | Sequence[str]) -> float:
        """Probability of a complete (or partially specified) outcome assignment."""
        if isinstance(assignment, dict):
            key = tuple(assignment[name] for name in self.order)
        else:
            key = tuple(assignment)
        return float(self.joint.get(key, 0.0))

    def marginal(self, measurement: str) -> dict[str, float]:
        """Marginal distribution of a single question."""
        index = self._axis(measurement)
        totals: dict[str, float] = {outcome: 0.0 for outcome in self.outcomes[measurement]}
        for key, value in self.joint.items():
            totals[key[index]] += value
        return totals

    def marginals(self) -> dict[str, dict[str, float]]:
        """Marginal distribution of every question."""
        return {name: self.marginal(name) for name in self.order}

    def conditional(self, measurement: str, given: dict[str, str] | None = None) -> dict[str, float]:
        """``p(measurement | given)``, renormalised from the joint distribution."""
        given = dict(given or {})
        index = self._axis(measurement)
        given_indices = {self._axis(name): value for name, value in given.items()}
        totals: dict[str, float] = {outcome: 0.0 for outcome in self.outcomes[measurement]}
        mass = 0.0
        for key, value in self.joint.items():
            if any(key[axis] != outcome for axis, outcome in given_indices.items()):
                continue
            totals[key[index]] += value
            mass += value
        if mass <= 0:
            return {outcome: float("nan") for outcome in self.outcomes[measurement]}
        return {outcome: value / mass for outcome, value in totals.items()}

    def conditional_table(self, first: str, second: str) -> dict[str, dict[str, float]]:
        """``p(second | first = outcome)`` for every outcome of ``first``."""
        return {
            outcome: self.conditional(second, {first: outcome})
            for outcome in self.outcomes[first]
        }

    def expectation(self, measurement: str, weights: Sequence[float] | None = None) -> float:
        """Expected value of the question's observable (default weights ``0, 1, 2, ...``)."""
        outcomes = self.outcomes[measurement]
        if weights is None:
            weights = list(range(len(outcomes)))
        if len(weights) != len(outcomes):
            raise ValueError("need one weight per outcome")
        marginal = self.marginal(measurement)
        return float(sum(weight * marginal[outcome] for weight, outcome in zip(weights, outcomes)))

    def sampling_error(self, measurement: str) -> dict[str, float]:
        """Standard error of each marginal probability under shot sampling."""
        if self.shots in (None, 0):
            return {outcome: 0.0 for outcome in self.outcomes[measurement]}
        marginal = self.marginal(measurement)
        shots = float(self.shots)
        return {
            outcome: float(np.sqrt(max(value * (1.0 - value), 0.0) / shots))
            for outcome, value in marginal.items()
        }

    # ---------------------------------------------------------------- exports
    def joint_frame(self):
        """Joint distribution as a :class:`pandas.DataFrame`."""
        import pandas as pd

        rows = []
        for key, value in self.joint.items():
            row = {name: outcome for name, outcome in zip(self.order, key)}
            row["probability"] = value
            if self.shots:
                row["count"] = int(round(value * self.shots))
            rows.append(row)
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame = frame.sort_values("probability", ascending=False).reset_index(drop=True)
        return frame

    def marginal_frame(self):
        """Marginal distributions as a tidy :class:`pandas.DataFrame`."""
        import pandas as pd

        rows = []
        for name in self.order:
            for outcome, value in self.marginal(name).items():
                rows.append({"question": name, "outcome": outcome, "probability": value})
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Short human-readable summary used by the CLI and the UI."""
        parts = []
        for index, name in enumerate(self.order):
            marginals = self.marginal(name)
            top = max(marginals.items(), key=lambda item: item[1])
            word = "marginal" if index == 0 else "conditional marginal"
            parts.append(f"{word} {name}: {top[0]} = {top[1]:.3f}")
        suffix = "" if self.is_exact else f" (shots={self.shots}, seed={self.seed})"
        return f"{' -> '.join(self.order)} [{self.backend}{suffix}] :: " + "; ".join(parts)

    def to_dict(self) -> dict:
        return {
            "order": list(self.order),
            "outcomes": {name: list(values) for name, values in self.outcomes.items()},
            "joint": {"|".join(key): round(value, 10) for key, value in self.joint.items()},
            "marginals": {
                name: {outcome: round(value, 10) for outcome, value in self.marginal(name).items()}
                for name in self.order
            },
            "backend": self.backend,
            "shots": self.shots,
            "seed": self.seed,
            "is_exact": self.is_exact,
            "duration_seconds": round(self.duration_seconds, 6),
            "noise": self.noise,
            "metadata": self.metadata,
        }

    # ---------------------------------------------------------------- helpers
    def _axis(self, measurement: str) -> int:
        if measurement not in self.order:
            raise ValueError(f"question {measurement!r} is not part of this result ({self.order})")
        return self.order.index(measurement)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"SequentialResult(order={self.order}, backend={self.backend!r}, outcomes={len(self.joint)})"


@dataclass
class OutcomeTable:
    """A tidy table of observed or predicted frequencies (used by the fitters)."""

    keys: tuple[str, ...]
    values: dict[str, float]
    source: str = "model"

    def as_dict(self) -> dict[str, float]:
        return {key: round(value, 10) for key, value in self.values.items()}

    def __iter__(self) -> Iterable[str]:
        return iter(self.keys)

    def __len__(self) -> int:
        return len(self.keys)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"OutcomeTable({len(self.keys)} entries, source={self.source!r})"


def result_from_diagonal(density: np.ndarray, measurements, *, backend: str = "exact", label: str = "") -> SequentialResult:
    """Joint distribution of a set of *commuting* questions, read off the diagonal.

    When every question is diagonal in the same basis - a group of yes/no votes, for
    instance - the joint distribution is exactly the computational-basis population of
    each outcome tuple, so there is no need to enumerate one path per answer sequence.
    The result is identical to what the sequential engine would produce, at a cost that
    is linear in the number of basis states rather than exponential in the number of
    questions; the reduction is exact, not an approximation.

    Raises
    ------
    ValueError
        If the questions do not commute with each other diagonally, in which case the
        caller must use the sequential engine.
    """
    import numpy as np

    from .measurement import POVM

    questions = list(measurements)
    outcomes = {question.name: tuple(question.outcomes) for question in questions}
    populations = np.real(np.diag(np.asarray(density)))
    total_states = populations.size
    # map each basis index to the outcome tuple it realises
    mapping: dict[int, tuple[str, ...]] = {}
    for index in range(total_states):
        entry = []
        for question in questions:
            effects = question.effects if isinstance(question, POVM) else question.projectors
            found = None
            for position, effect in enumerate(effects):
                if abs(effect[index, index]) > 0.5:
                    found = question.outcomes[position]
                    break
            if found is None:
                entry = []
                break
            entry.append(found)
        if entry:
            mapping[index] = tuple(entry)
    if not mapping:
        raise ValueError("the questions are not all diagonal in the computational basis")
    joint: dict[tuple[str, ...], float] = {}
    for index, key in mapping.items():
        joint[key] = joint.get(key, 0.0) + float(populations[index])
    total = sum(joint.values())
    if total > 0:
        joint = {key: value / total for key, value in joint.items()}
    steps: list[StepRecord] = []
    prefixes: list[tuple[str, ...]] = [()]
    for position, question in enumerate(questions):
        records = []
        for prefix in prefixes:
            mass = sum(value for key, value in joint.items() if key[:position] == prefix)
            conditional = {}
            for outcome in outcomes[question.name]:
                branch = sum(
                    value for key, value in joint.items() if key[: position + 1] == prefix + (outcome,)
                )
                conditional[outcome] = float(branch / mass) if mass > 0 else float("nan")
            records.append(
                StepRecord(position, question.name, tuple(question.outcomes), prefix, conditional)
            )
        steps.extend(records)
        prefixes = [record.prefix + (outcome,) for record in records for outcome in outcomes[question.name]]
    return SequentialResult(
        order=tuple(question.name for question in questions),
        outcomes=outcomes,
        joint=joint,
        steps=steps,
        states={},
        backend=backend,
        metadata={"joint_from": "diagonal of the density matrix (commuting questions)", "label": label},
    )
