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
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from .style import PALETTE, annotate_figure, color_for, new_figure, series_style


def _is_number(column) -> bool:
    """True for numbers, and False for text and for booleans.

    ``np.issubdtype(column.dtype, np.number)`` used to answer this, but a pandas extension
    dtype - the ``StringDtype`` that pandas 3 gives a text column, for instance - cannot be
    interpreted by numpy at all, so the question is put to pandas instead.
    """
    return bool(is_numeric_dtype(column)) and not bool(is_bool_dtype(column))


__all__ = [
    "VisualizationUnavailable",
    "probability_bars",
    "classical_comparison_figure",
    "circuit_figure",
    "CIRCUIT_STYLES",
    "marginal_bars",
    "order_effect_figure",
    "interference_figure",
    "sweep_figure",
    "line_figure",
    "histogram_figure",
    "bloch_sphere",
    "heatmap_figure",
    "payoff_table_figure",
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
    """Marginal distributions of the questions in one simulation.

    When every question has the *same* outcome labels the bars are grouped side by side,
    which invites comparison.  As soon as the labels differ - a yes/no question and a
    three-way choice, say - grouping them would put labels under the wrong bars, so each
    question gets its own panel instead.  The two cases are drawn differently on purpose:
    a figure that mislabels its own bars is worse than no figure.
    """
    frame = result.marginal_frame()
    questions = list(result.order)
    outcome_sets = [tuple(result.outcomes[name]) for name in questions]
    same_labels = len(set(outcome_sets)) == 1
    shared = same_labels and len(questions) <= 4
    reason = (
        "the questions have different outcome labels"
        if not same_labels
        else f"{len(questions)} questions would be unreadable side by side"
    )

    if shared:
        labels = list(outcome_sets[0])
        figure = new_figure(max(5.0, 1.2 * len(questions) + 2.4), 3.2)
        axis = figure.add_subplot(111)
        width = 0.8 / len(questions)
        for position, question in enumerate(questions):
            subset = frame[frame["question"] == question]
            axis.bar(
                np.arange(len(labels)) + position * width,
                subset["probability"],
                width=width,
                label=question,
                color=color_for(position),
                hatch=(None if position == 0 else "//"),
                edgecolor="white",
                linewidth=0.6,
            )
            for index, value in enumerate(subset["probability"]):
                axis.text(index + position * width, value + 0.02, f"{value:.2f}", ha="center", fontsize=8)
        axis.set_xticks(np.arange(len(labels)) + width * (len(questions) - 1) / 2)
        axis.set_xticklabels(labels)
        axis.set_ylabel("probability")
        axis.set_ylim(0, 1.08)
        axis.set_title(title or "Marginal answer distributions")
        axis.legend()
    else:
        figure = new_figure(5.6, max(2.4, 2.1 * len(questions)))
        for position, question in enumerate(questions):
            axis = figure.add_subplot(len(questions), 1, position + 1)
            subset = frame[frame["question"] == question]
            labels = list(subset["outcome"])
            values = list(subset["probability"])
            axis.barh(list(range(len(labels)))[::-1], values, color=color_for(position))
            for index, value in enumerate(values):
                axis.text(value + 0.02, len(labels) - 1 - index, f"{value:.2f}", va="center", fontsize=8)
            axis.set_yticks(list(range(len(labels)))[::-1])
            axis.set_yticklabels(labels)
            axis.set_xlim(0, 1.12)
            axis.set_xlabel("probability")
            axis.set_title(f"{question}  (outcomes: {', '.join(labels)})", fontsize=10)
    figure.tight_layout()
    annotate_figure(
        figure,
        ("Grouped bar chart" if shared else f"Panel chart, one panel per question, because {reason}")
        + " of the marginal answer distribution of each question. "
        + "; ".join(
            f"{name}: " + ", ".join(f"{outcome} {value:.3f}" for outcome, value in result.marginal(name).items())
            for name in questions
        )
        + ".",
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
        if column != parameter and _is_number(frame[column])
    ]
    figure = new_figure(6.0, 3.4)
    axis = figure.add_subplot(111)
    for index, metric in enumerate(metrics):
        style = series_style(index)
        axis.plot(
            frame[parameter], frame[metric], label=metric,
            color=style["color"], linestyle=style["linestyle"], marker=style["marker"], markersize=4,
        )
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
        style = series_style(index)
        axis.plot(
            spec["x"], values, label=label,
            color=style["color"], linestyle=style["linestyle"],
            marker=style["marker"] if len(spec["x"]) < 16 else None,
            markevery=max(1, len(spec["x"]) // 12), markersize=4,
        )
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
        style = series_style(index)
        axis.bar(
            positions + index * width, values, width=width, label=label,
            color=style["color"], hatch=("//" if index % 2 else None), edgecolor="white", linewidth=0.6,
        )
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
    length = float(np.linalg.norm(vector))
    if length < 1e-6:
        # a maximally mixed state sits at the centre: an arrow of length zero would simply
        # be invisible, so mark the point and say what it means
        axis.scatter([0], [0], [0], color=PALETTE["blue"], s=45, zorder=5)
        axis.text(0, 0, 0.2, "maximally mixed", ha="center", fontsize=8)
    else:
        axis.quiver(0, 0, 0, vector[0], vector[1], vector[2], color=PALETTE["blue"],
                    arrow_length_ratio=0.12)
    axis.plot([-1, 1], [0, 0], [0, 0], color="#999999", linewidth=0.8)
    axis.plot([0, 0], [-1, 1], [0, 0], color="#999999", linewidth=0.8)
    axis.plot([0, 0], [0, 0], [-1, 1], color="#999999", linewidth=0.8)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.set_box_aspect((1, 1, 1))
    axis.set_title(title or f"Belief state ({state.label})")
    figure.tight_layout()
    annotate_figure(
        figure,
        f"Bloch sphere with the belief state {state.label}. The Bloch vector points to "
        f"x = {vector[0]:+.2f}, y = {vector[1]:+.2f}, z = {vector[2]:+.2f}, with length {length:.2f}. "
        + (
            "The vector has length zero, so the state is maximally mixed: no question can tell it apart from a "
            "fair coin. A length of 1 means a definite (pure) belief; anything in between is partly mixed."
            if length < 1e-6
            else (
                "Length 1 means a definite (pure) belief; shorter vectors are partly mixed, and the distance from "
                "the centre is the state's purity."
                if length < 0.999
                else "The belief is pure: it sits on the surface."
            )
        ),
    )
    return figure


def heatmap_figure(
    frame,
    title: str | None = None,
    row_column: str | None = None,
    column_column: str | None = None,
    value_column: str | None = None,
):
    """Heatmap of a tidy table: one numeric value per (row label, column label) pair.

    The label and value columns can be named explicitly, which is what a caller should do
    when a table has several numeric columns; otherwise the last numeric column is plotted
    and the first two text columns become the axes, and the title names the value so the
    chart cannot be misread.
    """
    numeric = [column for column in frame.columns if _is_number(frame[column])]
    if not numeric:
        raise ValueError("the table has no numeric column to plot")
    text_columns = [column for column in frame.columns if column not in numeric]
    column = value_column or numeric[-1]
    if column not in frame.columns:
        raise ValueError(f"no column named {column!r} in the table")
    if row_column is None:
        row_column = text_columns[0] if text_columns else frame.columns[0]
    if column_column is None:
        candidates = [name for name in text_columns if name != row_column]
        column_column = candidates[0] if candidates else frame.columns[1]
    rows = list(dict.fromkeys(frame[row_column]))
    columns = list(dict.fromkeys(frame[column_column]))
    grid = np.full((len(rows), len(columns)), np.nan)
    for _, entry in frame.iterrows():
        grid[rows.index(entry[row_column]), columns.index(entry[column_column])] = entry[column]
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
    axis.set_xlabel(column_column)
    axis.set_ylabel(row_column)
    axis.set_title((title or "heatmap") + f"  ({column})")
    figure.colorbar(image, ax=axis, shrink=0.8, label=column)
    axis.grid(False)
    figure.tight_layout()
    annotate_figure(
        figure,
        f"Heatmap of {column} by {row_column} and {column_column}. "
        + "; ".join(
            f"{rows[i]} vs {columns[j]}: {grid[i, j]:.2f}"
            for i in range(len(rows))
            for j in range(len(columns))
            if not np.isnan(grid[i, j])
        )
        + ".",
    )
    return figure


def _first_column(frame, names: Sequence[str]):
    """The first of ``names`` that the table has, or ``None``."""
    for name in names:
        if name in frame.columns:
            return name
    return None


def payoff_table_figure(
    frame,
    title: str | None = None,
    row_column: str | None = None,
    column_column: str | None = None,
    quantum_column: str | None = None,
    classical_column: str | None = None,
):
    """Two heatmaps of a game table side by side: the quantum payoff and the classical one.

    A payoff table holds several numbers per outcome, so a single heatmap of it cannot be
    honest - the last numeric column wins, and in this catalogue that is the cooperation flag.
    The model's payoff to the first player is drawn on the left and the classical account's
    payoff to the same player on the right, on one shared colour scale, so the two accounts can
    be read against each other cell by cell.  The numbers themselves stay in the data table
    that accompanies every chart, and the description says how many cells differ.
    """
    numeric = [column for column in frame.columns if _is_number(frame[column])]
    if not numeric:
        raise ValueError("the table has no numeric column to plot")
    quantum_column = quantum_column or _first_column(frame, ("quantum_first", "quantum_payoff", "payoff"))
    classical_column = classical_column or _first_column(frame, ("classical_first", "classical_payoff"))
    if quantum_column is None or classical_column is None:
        raise ValueError("the table has no quantum/classical payoff pair to compare")
    for name in (quantum_column, classical_column):
        if name not in frame.columns:
            raise ValueError(f"no column named {name!r} in the table")
    text_columns = [column for column in frame.columns if column not in numeric]
    if row_column is None:
        row_column = text_columns[0] if text_columns else frame.columns[0]
    if column_column is None:
        candidates = [name for name in text_columns if name != row_column]
        column_column = candidates[0] if candidates else frame.columns[1]
    rows = list(dict.fromkeys(frame[row_column]))
    columns = list(dict.fromkeys(frame[column_column]))
    panels: list[np.ndarray] = []
    for name in (quantum_column, classical_column):
        grid = np.full((len(rows), len(columns)), np.nan)
        for _, entry in frame.iterrows():
            grid[rows.index(entry[row_column]), columns.index(entry[column_column])] = entry[name]
        panels.append(grid)
    drawn = np.concatenate([panel.ravel() for panel in panels])
    drawn = drawn[np.isfinite(drawn)]
    if drawn.size:
        low, high = float(drawn.min()), float(drawn.max())
    else:
        low, high = 0.0, 1.0
    if high <= low:
        low, high = low - 0.5, high + 0.5          # a flat table still needs a scale
    figure = new_figure(3.4 + 1.4 * len(columns), 2.2 + 0.7 * len(rows))
    figure.set_layout_engine("constrained")     # the shared colorbar needs a layout engine
    image = None
    for position, (name, label, grid) in enumerate(
        ((quantum_column, "quantum model", panels[0]), (classical_column, "classical account", panels[1])),
        start=1,
    ):
        axis = figure.add_subplot(1, 2, position)
        image = axis.imshow(grid, cmap="viridis", vmin=low, vmax=high)
        axis.set_xticks(range(len(columns)))
        axis.set_xticklabels(columns, rotation=20, ha="right")
        axis.set_yticks(range(len(rows)))
        axis.set_yticklabels(rows)
        for i in range(len(rows)):
            for j in range(len(columns)):
                if not np.isnan(grid[i, j]):
                    axis.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
        axis.set_xlabel(column_column)
        axis.set_ylabel(row_column)
        axis.set_title(f"{label}  ({name})")
        axis.grid(False)
    assert image is not None
    figure.colorbar(image, ax=list(figure.axes), shrink=0.8, label="payoff")
    comparable = np.isfinite(panels[0]) & np.isfinite(panels[1])
    differences = int(np.count_nonzero(~np.isclose(panels[0][comparable], panels[1][comparable]))) if comparable.any() else 0
    annotate_figure(
        figure,
        f"Two heatmaps of the payoff table on one shared colour scale ({low:.2f} to {high:.2f}): "
        f"{quantum_column} on the left, {classical_column} on the right. Rows are {row_column}, "
        f"columns are {column_column}. "
        + ("The two accounts agree in every cell." if differences == 0 else
           f"The two accounts differ in {differences} of the {int(np.count_nonzero(comparable))} cells.")
        + " "
        + "; ".join(
            f"{rows[i]} vs {columns[j]}: quantum {panels[0][i, j]:.2f}, classical {panels[1][i, j]:.2f}"
            for i in range(len(rows))
            for j in range(len(columns))
            if comparable[i, j]
        )
        + ".",
    )
    return figure


def _payoff_figure(frame):
    """The payoff chart for a tidy game table: the quantum account beside the classical one."""
    quantum = _first_column(frame, ("quantum_first", "quantum_payoff", "payoff"))
    classical = _first_column(frame, ("classical_first", "classical_payoff"))
    if quantum and classical:
        return payoff_table_figure(frame, quantum_column=quantum, classical_column=classical)
    if quantum:
        return heatmap_figure(frame, value_column=quantum)
    return heatmap_figure(frame)


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
        table = getattr(sweep_result, "table", None)
        metrics = [
            column
            for column in getattr(table, "columns", [])
            if column != sweep_result.parameter
        ]
        if not metrics or len(table.index) <= 1:
            continue  # a suppressed or empty sweep has nothing to draw
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
                figures[key.replace("_", " ")] = _payoff_figure(frame)
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

def classical_comparison_figure(rows: Sequence[Mapping], title: str | None = None):
    """Grouped bars: what the model predicts against what a classical account predicts.

    Every Q-Cog model produces these rows, so this chart - the model-versus-classical
    comparison - is present for all twenty-three of them.  Where the classical value
    is exactly zero the pair of bars reads as "the classical account is silent here,
    the model is not".
    """
    rows = [row for row in rows if row.get("model") is not None and row.get("baseline") is not None]
    if not rows:
        raise ValueError("no comparison rows to plot")
    labels = [str(row["metric"]) for row in rows]
    model_values = [float(row["model"]) for row in rows]
    baseline_values = [float(row["baseline"]) for row in rows]
    figure = new_figure(max(6.4, 1.1 * len(rows) + 3.0), 3.8)
    axis = figure.add_subplot(111)
    positions = np.arange(len(rows))
    axis.bar(positions - 0.2, model_values, width=0.4, color=PALETTE["blue"], label="quantum model")
    axis.bar(positions + 0.2, baseline_values, width=0.4, color="white", edgecolor=PALETTE["grey"],
             hatch="//", linewidth=1.0, label="classical baseline")
    for position, (model_value, baseline_value) in enumerate(zip(model_values, baseline_values)):
        axis.text(position - 0.2, model_value + 0.01, f"{model_value:.3f}", ha="center", fontsize=8)
        axis.text(position + 0.2, baseline_value + 0.01, f"{baseline_value:.3f}", ha="center", fontsize=8)
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [label if len(label) <= 34 else label[:31] + "..." for label in labels],
        rotation=18, ha="right", fontsize=9,
    )
    axis.set_ylabel("value")
    axis.set_title(title or "Model versus classical baseline")
    axis.legend()
    figure.tight_layout()
    annotate_figure(
        figure,
        "Grouped bar chart comparing the quantum model with the classical baseline on "
        f"{len(rows)} quantities. "
        + "; ".join(
            f"{label}: model {model_value:.3f} versus classical {baseline_value:.3f}"
            for label, model_value, baseline_value in zip(labels, model_values, baseline_values)
        )
        + ". Where the classical value is zero the classical account is forced to predict nothing there, "
        "so the model's bar is the whole content of that comparison.",
    )
    return figure


class VisualizationUnavailable(RuntimeError):
    """Raised when a figure cannot be drawn.

    The library never substitutes a text panel for a chart: an image is a graph or there is no
    image.  When a drawing is impossible - a circuit the drawer cannot render, a state with no
    Bloch vector - this error is raised, callers skip the figure, and the explanation stays
    where text belongs: in the description, the caption, the report and the CLI.
    """


#: Styles the Qiskit matplotlib drawer ships with.  A full Qiskit style dictionary is
#: accepted wherever a style name is, so a user can bring their own.
CIRCUIT_STYLES = ("default", "iqp", "clifford", "bw")


def circuit_figure(
    design,
    title: str | None = None,
    measurements: Sequence | None = None,
    *,
    style: str | dict | None = None,
    scale: float = 0.85,
    fold: int = 28,
    initial_state: bool = False,
    filename=None,
    dpi: int = 160,
):
    """Draw a quantum circuit, with a text alternative in place of the diagram.

    ``design`` may be a Qiskit ``QuantumCircuit`` or the ``(state, measurements)`` pair
    a model declares.  If the matplotlib drawer is unavailable (or the circuit cannot be
    rendered) the figure degrades to the plain-language description rather than failing,
    so a report can always be produced.

    The image itself is adjustable: ``style`` (one of :data:`CIRCUIT_STYLES`, or a full
    Qiskit style dictionary), ``scale`` (gate size), ``fold`` (how many gates before the
    drawing wraps), ``initial_state`` (show the ``|0>`` inputs) and ``filename`` (write
    the image to that path - the suffix chooses the format, so ``.png``, ``.svg`` and
    ``.pdf`` all work).
    """
    from qiskit import QuantumCircuit

    from ..circuits.builder import build_sequence_circuit, circuit_statistics, circuit_text_alternative

    if isinstance(design, QuantumCircuit):
        circuit = design
    elif isinstance(design, (tuple, list)) and len(design) == 2:
        state, questions = design
        try:
            circuit = build_sequence_circuit(state, questions)
        except Exception as error:
            raise VisualizationUnavailable(
                f"these questions cannot be compiled into a circuit ({type(error).__name__}: {error})"
            ) from error
        measurements = questions
    else:
        raise VisualizationUnavailable(
            "a circuit drawing needs a QuantumCircuit or a (state, measurements) pair; "
            f"got {type(design).__name__}"
        )
    statistics = circuit_statistics(circuit)
    operations = ", ".join(f"{name} x{count}" for name, count in sorted(statistics["operations"].items()))
    description = (
        circuit_text_alternative(circuit, measurements)
        + f" It contains {statistics['size']} operations at depth {statistics['depth']} on "
        + f"{statistics['qubits']} qubit(s): {operations}."
    )
    if isinstance(style, str) and style not in ("", "default"):
        style_spec = {"name": style}
    elif isinstance(style, dict):
        style_spec = style
    else:
        style_spec = None  # the drawer's own default, without a spurious warning
    try:
        from qiskit.visualization import circuit_drawer

        figure = circuit_drawer(
            circuit, output="mpl", scale=float(scale), fold=int(fold),
            initial_state=bool(initial_state), justify="left", style=style_spec,
        )
    except Exception as error:
        raise VisualizationUnavailable(
            f"the circuit of {circuit.name!r} could not be drawn ({type(error).__name__}: {error}); "
            "the description is available as text instead: " + description
        ) from error
    annotate_figure(
        figure,
        description,
        "Reading the drawing: each horizontal line is one qubit, time runs left to right, and every box is "
        "labelled with the operation or question it implements. The boxes at the right measure the qubits into "
        "the classical bits that record each answer.",
    )
    if title and getattr(figure, "axes", None):
        figure.axes[0].set_title(title, fontsize=10)
    if filename is not None:
        figure.savefig(filename, bbox_inches="tight", dpi=dpi)
    return figure
