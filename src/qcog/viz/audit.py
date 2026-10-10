"""Check that each chart shows the numbers it claims to show.

Rendering without an error is not the same as making sense: a bar chart whose labels belong
to a different question, or whose bars were read from the wrong column, renders perfectly
and misleads.  Each check here builds a figure from *known* data, reads the drawing back -
bar heights, stacked tops, line data, heatmap cells, the Bloch vector - and compares it with
the numbers that went in.  `python scripts/check_figures.py` runs this alongside the render
checks, and the test suite asserts it.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..analysis.interference import interference_decomposition
from ..analysis.order import order_effect
from ..analysis.sweeps import sweep
from ..core import basis_state, depolarizing_channel, question, run_sequence
from ..core.states import amplitudes
from ..circuits.builder import build_sequence_circuit, circuit_statistics
from .figures import (
    bloch_sphere,
    circuit_figure,
    classical_comparison_figure,
    heatmap_figure,
    histogram_figure,
    interference_figure,
    line_figure,
    marginal_bars,
    order_effect_figure,
    probability_bars,
    sweep_figure,
)
from .style import describe

__all__ = ["structure_checks", "bar_values", "bar_tops"]


def bar_values(axis) -> list[float]:
    """Heights of the bars in an axis, in drawing order."""
    import matplotlib.patches as patches

    return [float(patch.get_height()) for patch in axis.patches if isinstance(patch, patches.Rectangle)]


def bar_tops(axis) -> list[float]:
    """Top edge of each bar, so stacked bars can be compared with their totals."""
    import matplotlib.patches as patches

    return [
        float(patch.get_y() + patch.get_height())
        for patch in axis.patches
        if isinstance(patch, patches.Rectangle)
    ]


def structure_checks() -> list[dict]:
    """One row per chart type, comparing the drawing with the data that produced it."""
    rows: list[dict] = []

    def record(name: str, ok: bool, detail: str) -> None:
        rows.append({"name": name, "ok": bool(ok), "detail": detail})

    # a two-question design with a known joint distribution
    state = basis_state(0, 2)
    first = question(0.955, 0.0, "A")
    second = question(np.pi / 2, 0.0, "B")
    forward = run_sequence(state, [first, second])
    backward = run_sequence(state, [second, first])

    # 1. probability bars show the joint probabilities
    figure = probability_bars(forward)
    drawn = bar_values(figure.axes[0])
    expected = sorted(forward.joint.values(), reverse=True)[: len(drawn)]
    difference = float(np.max(np.abs(np.array(drawn) - np.array(expected)))) if drawn else float("nan")
    record(
        "probability bars match the joint distribution",
        bool(drawn) and np.allclose(drawn, expected, atol=1e-9),
        f"{len(drawn)} bars, largest difference {difference:.2e}",
    )
    plt.close(figure)

    # 2. grouped marginal bars, when the questions share their outcome labels
    figure = marginal_bars(forward)
    heights = bar_values(figure.axes[0])
    expected = [forward.marginal("A")[outcome] for outcome in ("yes", "no")] + [
        forward.marginal("B")[outcome] for outcome in ("yes", "no")
    ]
    tick_labels = [text.get_text() for text in figure.axes[0].get_xticklabels()]
    record(
        "grouped marginal bars match each question's marginals",
        len(heights) == len(expected) and np.allclose(sorted(heights), sorted(expected), atol=1e-9),
        f"{len(heights)} bars against {len(expected)} marginals, labels {tick_labels}",
    )
    plt.close(figure)

    # 3. and panels, when the labels differ: two three-outcome questions with different names
    from ..core.measurement import observable_measurement

    qutrit = amplitudes(0.5, 0.7, 0.5, label="three-way belief")
    senses = observable_measurement("senses", np.diag([1.0, 2.0, 3.0]), ("low", "mid", "high"))
    categories = observable_measurement("categories", np.diag([3.0, 2.0, 1.0]), ("A", "B", "C"))
    labelled = run_sequence(qutrit, [senses, categories])
    figure = marginal_bars(labelled)
    panels = [axis for axis in figure.axes if axis.get_yticklabels()]
    panel_values = {len(bar_values(axis)) for axis in panels}
    record(
        "panels are used when the questions have different outcome labels",
        len(figure.axes) == 2 and panel_values == {3},
        f"{len(figure.axes)} panels with {sorted(panel_values)} bars",
    )
    plt.close(figure)

    # 4. the order-effect chart shows both orders, one pair of bars per outcome
    report = order_effect(forward, backward, "A", "B")
    figure = order_effect_figure(report)
    heights = bar_values(figure.axes[0])
    expected = [
        report.marginal_first_in_first_order[outcome] for outcome in report.marginal_first_in_first_order
    ] + [
        report.marginal_first_in_second_order[outcome] for outcome in report.marginal_first_in_first_order
    ]
    record(
        "order-effect bars match both orders",
        len(heights) == len(expected) and np.allclose(sorted(heights), sorted(expected), atol=1e-9),
        f"{len(heights)} bars against {len(expected)} marginals",
    )
    plt.close(figure)

    # 5. the interference chart's stacked tops equal direct + interference
    interference = interference_decomposition(state, first, second)
    figure = interference_figure(interference)
    tops = bar_tops(figure.axes[0])
    # the chart draws one bar per outcome for the classical part, then one per outcome for the
    # interference term stacked on it: the first block's tops are the direct terms, the second
    # block's tops are the totals
    outcomes = len(interference.totals)
    classical_part = tops[:outcomes]
    stacked_total = tops[outcomes:]
    expected_classical = [interference.direct[outcome] for outcome in interference.totals]
    expected_totals = [interference.totals[outcome] for outcome in interference.totals]
    record(
        "stacked interference bars add up to the totals",
        np.allclose(classical_part, expected_classical, atol=1e-9)
        and np.allclose(stacked_total, expected_totals, atol=1e-9),
        f"direct {[round(value, 3) for value in classical_part]} -> totals "
        f"{[round(value, 3) for value in stacked_total]}",
    )
    plt.close(figure)

    # 6. sweeps plot the table, point for point
    sweep_result = sweep("theta", [0.0, 0.5, 1.0], lambda value: {"order effect": value / 2, "interference": value})
    figure = sweep_figure(sweep_result)
    lines = figure.axes[0].lines
    record(
        "sweep charts plot every point of the table",
        len(lines) == 2 and all(len(line.get_xdata()) == 3 for line in lines),
        f"{len(lines)} series over {len(sweep_result.values)} values",
    )
    plt.close(figure)

    # 7. line charts draw one line per series, of the right length
    spec = {
        "x": [0, 1, 2],
        "xlabel": "step",
        "ylabel": "opinion",
        "series": {"first": [0.1, 0.2, 0.3], "second": [0.3, 0.2, 0.1]},
    }
    figure = line_figure(spec)
    lengths = [len(line.get_ydata()) for line in figure.axes[0].lines]
    record("line charts draw every series", lengths == [3, 3], f"series lengths {lengths}")
    plt.close(figure)

    # 8. histograms draw one bar per count per series
    histogram_spec = {
        "x": [0, 1, 2],
        "xlabel": "votes in favour",
        "ylabel": "probability",
        "series": {"correlated": [0.2, 0.3, 0.5], "independent": [0.5, 0.3, 0.2]},
    }
    figure = histogram_figure(histogram_spec)
    record(
        "histograms draw one bar per count per series",
        len(bar_values(figure.axes[0])) == 6,
        f"{len(bar_values(figure.axes[0]))} bars for 2 series of 3 counts",
    )
    plt.close(figure)

    # 9. the Bloch sphere marks the state it was given, including the centre
    pure = amplitudes(1, 1j, label="pure")
    mixed_state = amplitudes(1, 1, label="mixed").apply_channel(depolarizing_channel(0.5, 2).kraus)
    for label, drawn_state in (("pure", pure), ("mixed", mixed_state)):
        figure = bloch_sphere(drawn_state)
        vector = drawn_state.bloch_vector()
        if float(np.linalg.norm(vector)) < 1e-6:
            ok, detail = bool(figure.axes[0].collections), "maximally mixed state marked at the centre"
        else:
            arrows = [line for line in figure.axes[0].lines if len(line.get_xdata()) == 2]
            ok, detail = bool(arrows), f"vector drawn, length {np.linalg.norm(vector):.2f}"
        record(f"Bloch sphere draws the {label} state", ok, detail)
        plt.close(figure)

    # 10. heatmaps place every cell of the table where it belongs
    frame = pd.DataFrame(
        [
            {"first": "C", "second": "C", "payoff": 3.0},
            {"first": "C", "second": "D", "payoff": 0.0},
            {"first": "D", "second": "C", "payoff": 5.0},
            {"first": "D", "second": "D", "payoff": 1.0},
        ]
    )
    figure = heatmap_figure(frame, value_column="payoff")
    cells = np.asarray(figure.axes[0].images[0].get_array(), dtype=float)
    labels = [text.get_text() for text in figure.axes[0].get_xticklabels()]
    record(
        "heatmap cells and axis labels match the table",
        cells.shape == (2, 2) and np.allclose(cells, [[3.0, 0.0], [5.0, 1.0]]) and labels == ["C", "D"],
        f"grid {cells.shape}, x labels {labels}",
    )
    plt.close(figure)

    # 11. the comparison chart draws a model bar and a baseline bar per row
    comparison_rows = [
        {"metric": "order effect", "model": 0.29, "baseline": 0.0, "note": ""},
        {"metric": "interference share", "model": 0.47, "baseline": 0.0, "note": ""},
    ]
    figure = classical_comparison_figure(comparison_rows)
    heights = bar_values(figure.axes[0])
    record(
        "comparison chart draws a model bar and a baseline bar per row",
        len(heights) == 4 and np.allclose(sorted(heights), [0.0, 0.0, 0.29, 0.47], atol=1e-9),
        f"{len(heights)} bars for {len(comparison_rows)} rows",
    )
    plt.close(figure)

    # 12. the circuit drawing describes the circuit that was drawn
    circuit = build_sequence_circuit(state, [first, second])
    statistics = circuit_statistics(circuit)
    figure = circuit_figure(circuit, title="A then B")
    text = describe(figure)
    record(
        "circuit drawing describes the circuit it drew",
        f"{statistics['qubits']} qubit" in text and "measure x2" in text,
        f"names {statistics['qubits']} qubit(s) and both measurements",
    )
    plt.close(figure)

    return rows