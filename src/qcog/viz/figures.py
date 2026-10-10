"""Figures built from model results.

Every function takes plain library objects (a :class:`SequentialResult`, an
analysis report, a sweep table) and returns a matplotlib figure with an attached
text description.  The model base class assembles the standard set from whatever
a model put in its analysis dictionary, so a new model gets a dashboard without
writing any plotting code - and cannot accidentally ship a chart without an
accessible alternative.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

from .style import PALETTE, annotate_figure, color_for, new_figure

__all__ = [
    "probability_bars",
    "marginal_bars",
    "order_effect_figure",
    "interference_figure",
    "sweep_figure",
    "line_figure",
    "histogram_figure",
    "bloch_sphere",
    "heatmap_figure",
    "default_figures",
    "text_alternatives",
    "savefig",
]


def probability_bars(result, title: str | None = None, top: int | None = 10):
    """Joint outcome probabilities of one simulation as a bar chart."""
    frame = result.joint_frame()
    if top:
        frame = frame.head(top)
    figure = new_figure(max(5.0, 0.9 * len(frame) + 2.0), 3.4)
    axis = figure.add_subplot(111)
    labels = [" + ".join(str(value) for value in row) for row in frame[list(result.order)].to_numpy()]
    values = frame["probability"].to_numpy()
    axis.bar(range(len(values)), values, color=PALETTE["blue"])
    axis.set_xticks(range(len(values)))
    axis.set_xticklabels(labels, rotation=30, ha="right")
    axis.set_ylabel("probability")
    axis.set_ylim(0, max(values.max() * 1.15, 0.05))
    axis.set_title(title or f"Joint response distribution ({' -> '.join(result.order)})")
    for index, value in enumerate(values):
        axis.text(index, value + 0.005, f"{value:.3f}", ha="center", fontsize=8)
    figure.tight_layout()
    annotate_figure(
        figure,
        f"Bar chart of the joint response probabilities for the question order "
        f"{' then '.join(result.order)}. "
        + "; ".join(f"{label}: {value:.3f}" for label, value in zip(labels[:6], values[:6]))
        + ("; ..." if len(labels) > 6 else "")
        + f". Backend {result.backend}.",
    )
    return figure


def marginal_bars(result, title: str | None = None):
    """Marginal distributions of every question in one simulation."""
    frame = result.marginal_frame()
    questions = list(result.order)
    figure = new_figure(max(5.0, 1.4 * len(questions) + 1.6), 3.2)
    axis = figure.add_subplot(111)
    width = 0.8 / max(len(set(frame["question"])), 1)
    for position, question in enumerate(questions):
        subset = frame[frame["question"] == question]
        axis.bar(
            np.arange(len(subset)) + position * width,
            subset["probability"],
            width=width,
            label=question,
            color=color_for(position),
        )
    axis.set_xticks(np.arange(len(frame[frame["question"] == questions[0]])) + width)
    axis.set_xticklabels(list(frame[frame["question"] == questions[0]]["outcome"]), rotation=20, ha="right")
    axis.set_ylabel("probability")
    axis.set_ylim(0, 1.05)
    axis.set_title(title or "Marginal answer distributions")
    axis.legend()
    figure.tight_layout()
    annotate_figure(
        figure,
        "Grouped bar chart of the marginal answer distribution of each question. "
        + "; ".join(
            f"{name}: "
            + ", ".join(f"{outcome} {value:.3f}" for outcome, value in result.marginal(name).items())
            for name in questions
        ),
    )
    return figure


def order_effect_figure(report, title: str | None = None):
    """The marginal of the first question in each of the two question orders."""
    outcomes = list(report.marginal_first_in_first_order)
    first = [report.marginal_first_in_first_order[outcome] for outcome in outcomes]
    second = [report.marginal_first_in_second_order[outcome] for outcome in outcomes]
    figure = new_figure(5.6, 3.4)
    axis = figure.add_subplot(111)
    positions = np.arange(len(outcomes))
    axis.bar(positions - 0.19, first, width=0.38, color=PALETTE["blue"],
             label=f"{report.first} asked first")
    axis.bar(positions + 0.19, second, width=0.38, color=PALETTE["orange"],
             label=f"{report.first} asked second")
    for position, (left, right) in enumerate(zip(first, second)):
        axis.text(position, max(left, right) + 0.02, f"{right - left:+.3f}", ha="center", fontsize=8)
    axis.set_xticks(positions)
    axis.set_xticklabels(outcomes)
    axis.set_ylabel(f"p({report.first} = outcome)")
    axis.set_ylim(0, 1.05)
    axis.set_title(title or f"Order effect on {report.first}")
    axis.legend()
    figure.tight_layout()
    annotate_figure(
        figure,
        f"Bar chart comparing the marginal distribution of {report.first} when it is asked before "
        f"{report.second} and after it. "
        + "; ".join(
            f"{outcome}: {left:.3f} asked first versus {right:.3f} asked second"
            for outcome, left, right in zip(outcomes, first, second)
        )
        + f". Largest shift {report.max_marginal_shift():.3f}; total variation distance "
        f"{report.total_variation:.3f}.",
    )
    return figure


def interference_figure(report, title: str | None = None):
    """Direct (classical) and interference contributions to the second question."""
    outcomes = list(report.totals)
    direct = [report.direct[outcome] for outcome in outcomes]
    interference = [report.interference[outcome] for outcome in outcomes]
    figure = new_figure(5.8, 3.4)
    axis = figure.add_subplot(111)
    positions = np.arange(len(outcomes))
    axis.bar(positions, direct, color=PALETTE["blue"], label="classical contribution")
    axis.bar(positions, interference, bottom=direct, color=PALETTE["vermillion"],
             label="interference contribution")
    axis.axhline(0.0, color="#333333", linewidth=1)
    for position, outcome in enumerate(outcomes):
        axis.text(position, report.totals[outcome] + 0.02, f"{report.totals[outcome]:.3f}",
                  ha="center", fontsize=8)
    axis.set_xticks(positions)
    axis.set_xticklabels(outcomes)
    axis.set_ylabel(f"contribution to p({report.second})")
    axis.set_title(title or "Where the probability comes from")
    axis.legend()
    figure.tight_layout()
    annotate_figure(
        figure,
        f"Stacked bar chart decomposing p({report.second}) into the classical (law of total probability) "
        "contribution and the interference contribution. "
        + "; ".join(
            f"{outcome}: classical {report.direct[outcome]:.3f} plus interference "
            f"{report.interference[outcome]:+.3f} equals {report.totals[outcome]:.3f}"
            for outcome in outcomes
        )
        + ".",
    )
    return figure


def sweep_figure(sweep_result, title: str | None = None):
    """All numeric metrics of a sweep as lines against the swept parameter."""
    frame = sweep_result.table
    parameter = sweep_result.parameter
    metrics = [
        column
        for column in frame.columns
        if column != parameter and np.issubdtype(frame[column].dtype, np.number)
    ]
    figure = new_figure(6.0, 3.4)
    axis = figure.add_subplot(111)
    for index, metric in enumerate(metrics):
        axis.plot(frame[parameter], frame[metric], marker="o", color=color_for(index), label=metric)
    axis.set_xlabel(parameter.replace("_", " "))
    axis.set_ylabel("value")
    axis.set_title(title or f"How the model responds to {parameter.replace('_', ' ')}")
    axis.legend()
    figure.tight_layout()
    annotate_figure(
        figure,
        f"Line chart of {', '.join(metrics)} as {parameter.replace('_', ' ')} varies from "
        f"{frame[parameter].iloc[0]} to {frame[parameter].iloc[-1]}. "
        + "; ".join(
            f"{metric} goes from {frame[metric].iloc[0]:.3f} to {frame[metric].iloc[-1]:.3f}"
            for metric in metrics
        )
        + ".",
    )
    return figure


def line_figure(spec: Mapping, title: str | None = None):
    """Generic multi-series line chart from the ``{x, xlabel, ylabel, series}`` convention."""
    figure = new_figure(6.2, 3.6)
    axis = figure.add_subplot(111)
    for index, (label, values) in enumerate(spec["series"].items()):
        axis.plot(spec["x"], values, marker="o" if len(spec["x"]) < 12 else None,
                  color=color_for(index), label=label)
    axis.set_xlabel(spec.get("xlabel", "step"))
    axis.set_ylabel(spec.get("ylabel", "value"))
    axis.set_title(title or spec.get("title", "Trajectory"))
    if len(spec["series"]) > 1:
        axis.legend()
    figure.tight_layout()
    last = {label: values[-1] for label, values in spec["series"].items()}
    annotate_figure(
        figure,
        f"Line chart of {len(spec['series'])} series over {spec.get('xlabel', 'steps')}. "
        + "; ".join(f"{label} ends at {value:.3f}" for label, value in last.items())
        + ".",
    )
    return figure


def histogram_figure(spec: Mapping, title: str | None = None):
    """Grouped bars for count distributions (votes, decisions)."""
    figure = new_figure(6.0, 3.4)
    axis = figure.add_subplot(111)
    series = list(spec["series"].items())
    width = 0.8 / max(len(series), 1)
    positions = np.arange(len(spec["x"]))
    for index, (label, values) in enumerate(series):
        axis.bar(positions + index * width, values, width=width, label=label, color=color_for(index))
    axis.set_xticks(positions + width * (len(series) - 1) / 2)
    axis.set_xticklabels([str(value) for value in spec["x"]])
    axis.set_xlabel(spec.get("xlabel", ""))
    axis.set_ylabel(spec.get("ylabel", "probability"))
    axis.set_title(title or "Distribution comparison")
    if len(series) > 1:
        axis.legend()
    figure.tight_layout()
    annotate_figure(
        figure,
        f"Grouped bar chart comparing {len(series)} distributions over {spec.get('xlabel', 'counts')}. "
        + "; ".join(
            f"{label}: peak at {spec['x'][int(np.argmax(values))]} with probability {max(values):.3f}"
            for label, values in series
        )
        + ".",
    )
    return figure


def bloch_sphere(state, title: str | None = None):
    """Bloch sphere with the state vector and its projections."""
    if state.dim != 2:
        raise ValueError("the Bloch sphere is only defined for a qubit")
    vector = state.bloch_vector()
    figure = new_figure(4.6, 4.4)
    axis = figure.add_subplot(111, projection="3d")
    u = np.linspace(0, 2 * np.pi, 32)
    v = np.linspace(0, np.pi, 16)
    axis.plot_wireframe(
        np.outer(np.cos(u), np.sin(v)),
        np.outer(np.sin(u), np.sin(v)),
        np.outer(np.ones_like(u), np.cos(v)),
        color="#CCCCCC", linewidth=0.4, alpha=0.6,
    )
    axis.quiver(0, 0, 0, vector[0], vector[1], vector[2], color=PALETTE["blue"], arrow_length_ratio=0.12)
    axis.plot([-1, 1], [0, 0], [0, 0], color="#999999", linewidth=0.8)
    axis.plot([0, 0], [-1, 1], [0, 0], color="#999999", linewidth=0.8)
    axis.plot([0, 0], [0, 0], [-1, 1], color="#999999", linewidth=0.8)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.set_box_aspect((1, 1, 1))
    axis.set_title(title or f"Belief state ({state.label})")
    figure.tight_layout()
    length = float(np.linalg.norm(vector))
    annotate_figure(
        figure,
        f"Bloch sphere with the belief state {state.label}. The Bloch vector points to "
        f"x = {vector[0]:+.2f}, y = {vector[1]:+.2f}, z = {vector[2]:+.2f}, with length {length:.2f}. "
        + (
            "A length of 1 means a definite (pure) belief; shorter vectors mean the belief is mixed or "
            "uncertain."
            if length < 0.999
            else "The belief is pure."
        ),
    )
    return figure


def heatmap_figure(frame, title: str | None = None, row_column: str | None = None):
    """Heatmap of a tidy table with a numeric value column."""
    numeric = [column for column in frame.columns if np.issubdtype(frame[column].dtype, np.number)]
    if not numeric:
        raise ValueError("the table has no numeric column to plot")
    column = numeric[-1]
    if row_column is None:
        row_column = frame.columns[0]
    rows = list(dict.fromkeys(frame[row_column]))
    columns = list(dict.fromkeys(frame[frame.columns[1]]))
    grid = np.full((len(rows), len(columns)), np.nan)
    for _, entry in frame.iterrows():
        grid[rows.index(entry[row_column]), columns.index(entry[frame.columns[1]])] = entry[column]
    figure = new_figure(1.5 + 0.8 * len(columns), 1.6 + 0.6 * len(rows))
    axis = figure.add_subplot(111)
    image = axis.imshow(grid, cmap="viridis")
    axis.set_xticks(range(len(columns)))
    axis.set_xticklabels(columns, rotation=20, ha="right")
    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels(rows)
    for i in range(len(rows)):
        for j in range(len(columns)):
            if not np.isnan(grid[i, j]):
                axis.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    axis.set_title(title or column.replace("_", " "))
    figure.colorbar(image, ax=axis, shrink=0.8, label=column)
    axis.grid(False)
    figure.tight_layout()
    annotate_figure(
        figure,
        f"Heatmap of {column} by {row_column} and {frame.columns[1]}. "
        + "; ".join(
            f"{rows[i]} vs {columns[j]}: {grid[i, j]:.2f}"
            for i in range(len(rows))
            for j in range(len(columns))
            if not np.isnan(grid[i, j])
        )
        + ".",
    )
    return figure


def default_figures(model, result) -> dict:
    """The standard figure set for a model result."""
    figures: dict[str, object] = {}
    for name, item in result.sequential_results().items():
        if len(item.joint) <= 16:
            figures[f"probabilities: {name}"] = probability_bars(item)
        else:
            figures[f"marginals: {name}"] = marginal_bars(item)
    analysis = result.analysis
    if "order_effect" in analysis:
        figures["order effect"] = order_effect_figure(analysis["order_effect"])
    if analysis.get("interference") is not None:
        figures["interference"] = interference_figure(analysis["interference"])
    for name, sweep_result in (analysis.get("sweeps") or {}).items():
        figures[f"sweep: {name}"] = sweep_figure(sweep_result)
    for name, state in (analysis.get("states") or {}).items():
        if getattr(state, "dim", 0) == 2:
            figures[f"state: {name}"] = bloch_sphere(state, title=f"Belief state ({name})")
    if "trajectory" in analysis:
        figures["trajectory"] = line_figure(analysis["trajectory"])
    if "classical_trajectory" in analysis:
        figures["classical trajectory"] = line_figure(analysis["classical_trajectory"])
    if "histogram" in analysis:
        figures["distribution"] = histogram_figure(analysis["histogram"])
    for key in ("payoff_table",):
        frame = analysis.get(key)
        if frame is not None and hasattr(frame, "columns"):
            try:
                figures[key.replace("_", " ")] = heatmap_figure(frame)
            except Exception:  # pragma: no cover - a table without numeric columns
                pass
    return figures


def text_alternatives(figures: Mapping) -> dict[str, str]:
    """Accessible text description of every figure."""
    return {name: getattr(figure, "_qcog_description", "") for name, figure in figures.items()}


def savefig(figure, path, description: bool = False) -> str:
    """Save a figure, optionally appending its text description."""
    figure.savefig(path, bbox_inches="tight")
    return str(path)