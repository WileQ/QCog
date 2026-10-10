"""Where does a probability come from?  Classic versus interference terms.

For a first question ``A`` and a second question ``B`` the Born rule splits into
two pieces whose sum need not be one probability:

    p(b) = sum_a p(a) p(b | a)   +   sum_{a != a'} <psi|P_a P_b P_a'|psi>
           \_____ direct ______/     \__________ interference ___________/

The first sum is exactly what a classical model would predict (it is the law of
total probability applied to the A-then-B order).  The second is the interference
term; it can be negative, and it is what makes the predictions of the two question
orders differ.

Q-Cog reports the split numerically and, with
:func:`qcog.viz.interference_decomposition_figure`, visually, because "the
phenomenon is caused by interference" is a much weaker statement than "the
interference term is -0.11 and accounts for the whole gap".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ..core.measurement import Measurement, POVM, ProjectiveMeasurement
from ..core.states import CognitiveState, as_state

__all__ = ["InterferenceReport", "interference_decomposition", "which_path_table"]


def _measurement_operators(measurement: Measurement) -> tuple[np.ndarray, ...]:
    if isinstance(measurement, POVM):
        return measurement.measurement_operators()
    return tuple(measurement.projectors)


def _effects(measurement: Measurement) -> tuple[np.ndarray, ...]:
    if isinstance(measurement, POVM):
        return tuple(measurement.effects)
    return tuple(measurement.projectors)


@dataclass
class InterferenceReport:
    """Classical and interference contributions to the second question's probabilities."""

    first: str
    second: str
    state_label: str
    totals: dict[str, float]
    direct: dict[str, float]
    interference: dict[str, float]
    cross_terms: dict[tuple[str, str, str], complex] = field(default_factory=dict)
    paths: dict[tuple[str, str], float] = field(default_factory=dict)

    def check(self, tol: float = 1e-9) -> bool:
        """True when ``direct + interference`` reproduces the total (a self-test)."""
        return all(
            abs(self.direct[outcome] + self.interference[outcome] - self.totals[outcome]) < tol
            for outcome in self.totals
        )

    @property
    def max_abs_interference(self) -> float:
        """Largest absolute interference contribution (a quick 'how quantum' number)."""
        return float(max((abs(value) for value in self.interference.values()), default=0.0))

    def relative_interference(self) -> dict[str, float]:
        """Interference as a fraction of the total probability, per outcome."""
        return {
            outcome: (self.interference[outcome] / self.totals[outcome] if self.totals[outcome] else 0.0)
            for outcome in self.totals
        }

    def to_dict(self) -> dict:
        return {
            "first": self.first,
            "second": self.second,
            "state": self.state_label,
            "totals": {key: round(value, 8) for key, value in self.totals.items()},
            "direct": {key: round(value, 8) for key, value in self.direct.items()},
            "interference": {key: round(value, 8) for key, value in self.interference.items()},
            "cross_terms": {
                "|".join(key): [round(float(np.real(value)), 8), round(float(np.imag(value)), 8)]
                for key, value in self.cross_terms.items()
            },
            "paths": {"|".join(key): round(value, 8) for key, value in self.paths.items()},
        }

    def summary(self) -> str:
        parts = []
        for outcome, total in self.totals.items():
            parts.append(
                f"{self.second}={outcome}: p={total:.4f} = direct {self.direct[outcome]:.4f} "
                f"+ interference {self.interference[outcome]:+.4f}"
            )
        return "; ".join(parts)


def interference_decomposition(state, first: Measurement, second: Measurement) -> InterferenceReport:
    """Split ``p(second)`` into classical and interference contributions.

    Works for projective and unsharp (POVM) questions.  For a pure state the
    individual complex cross terms are returned as well, which is what the phasor
    visualisation uses.
    """
    confidence: CognitiveState = as_state(state)
    first_operators = _measurement_operators(first)
    second_effects = _effects(second)
    rho = confidence.density
    direct: dict[str, float] = {}
    interference: dict[str, float] = {}
    totals: dict[str, float] = {}
    paths: dict[tuple[str, str], float] = {}
    cross: dict[tuple[str, str, str], complex] = {}
    for outcome_index, outcome in enumerate(second.outcomes):
        effect = second_effects[outcome_index]
        diagonal = 0.0
        off_diagonal = 0.0
        for index, operator in enumerate(first_operators):
            diagonal += float(np.real(np.trace(effect @ operator @ rho @ operator.conj().T)))
            paths[(first.outcomes[index], outcome)] = diagonal
        ordered: dict[tuple[str, str], complex] = {}
        for left, left_operator in enumerate(first_operators):
            for right, right_operator in enumerate(first_operators):
                if left == right:
                    continue
                value = complex(np.trace(effect @ left_operator @ rho @ right_operator.conj().T))
                ordered[(first.outcomes[left], first.outcomes[right])] = value
                off_diagonal += float(np.real(value))
        for left in range(len(first_operators)):
            for right in range(left + 1, len(first_operators)):
                cross[(first.outcomes[left], first.outcomes[right], outcome)] = ordered[
                    (first.outcomes[left], first.outcomes[right])
                ]
        direct[outcome] = diagonal
        interference[outcome] = off_diagonal
        totals[outcome] = diagonal + off_diagonal
    return InterferenceReport(
        first=first.name,
        second=second.name,
        state_label=confidence.label,
        totals=totals,
        direct=direct,
        interference=interference,
        cross_terms=cross,
        paths=paths,
    )


def which_path_table(report: InterferenceReport) -> list[dict]:
    """Tidy rows for the "which path" contribution table."""
    return [
        {
            "first_outcome": first_outcome,
            "second_outcome": second_outcome,
            "direct_contribution": round(value, 8),
        }
        for (first_outcome, second_outcome), value in sorted(report.paths.items())
    ]