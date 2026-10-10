"""Parameter sweeps: the quickest way to turn a model into an experiment.

A sweep answers questions of the form "how does the order effect depend on how
incompatible the questions are?", "how much decoherence can the prediction
tolerate?", or "where does the model stop being distinguishable from its classical
baseline?".  Every sweep returns a tidy :class:`pandas.DataFrame` and, optionally,
the raw results, so it can be plotted, exported, or fitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

from ..core.states import as_state

__all__ = ["SweepResult", "sweep", "sharpness_sweep", "decoherence_sweep"]


@dataclass
class SweepResult:
    """The outcome of a parameter sweep."""

    parameter: str
    values: list
    table: "object"  # pandas.DataFrame
    raw: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.values)

    def column(self, name: str) -> list:
        """One column of the sweep table as a list."""
        return list(self.table[name])

    def peak(self, metric: str) -> tuple:
        """Value of the swept parameter at which ``metric`` is largest."""
        frame = self.table
        index = int(np.argmax(frame[metric].to_numpy()))
        return frame[self.parameter].iloc[index], float(frame[metric].iloc[index])

    def to_dict(self) -> dict:
        return {
            "parameter": self.parameter,
            "values": list(self.values),
            "table": self.table.to_dict(orient="records") if hasattr(self.table, "to_dict") else self.table,
        }

    def summary(self) -> str:
        return (
            f"Swept {self.parameter} over {len(self.values)} values; "
            f"metrics: {', '.join(self.table.columns)}"
        )


def sweep(
    parameter: str,
    values: Sequence,
    measure: Callable[[object], Mapping[str, float]],
    raw: bool = False,
) -> SweepResult:
    """Run ``measure(value)`` for every value and collect the metrics in a table.

    ``measure`` may return any flat mapping of metric name to number; it is the
    model author's job to choose metrics that are worth plotting.
    """
    import pandas as pd

    rows = []
    collected = []
    for value in values:
        metrics = dict(measure(value))
        row = {parameter: value}
        row.update(
            {key: float(value) if np.isscalar(value) and not isinstance(value, str) else value
             for key, value in metrics.items()}
        )
        rows.append(row)
        if raw:
            collected.append(metrics)
    return SweepResult(parameter, list(values), pd.DataFrame(rows), collected)


def sharpness_sweep(
    measure: Callable[[float], Mapping[str, float]],
    values: Sequence[float] | None = None,
) -> SweepResult:
    """Sweep the sharpness (``lambda``) of an unsharp question.

    ``lambda = 1`` is a fully projective question and ``lambda = 0`` an
    uninformative one, so this sweep shows how much "quantum question" a
    phenomenon actually needs.
    """
    if values is None:
        values = [round(value, 3) for value in np.linspace(0.0, 1.0, 11)]
    return sweep("sharpness", list(values), measure)


def decoherence_sweep(
    measure: Callable[[float], Mapping[str, float]],
    values: Sequence[float] | None = None,
) -> SweepResult:
    """Sweep a decoherence strength from 0 (coherent) to 1 (fully scrambled)."""
    if values is None:
        values = [round(value, 3) for value in np.linspace(0.0, 1.0, 11)]
    return sweep("decoherence", list(values), measure)