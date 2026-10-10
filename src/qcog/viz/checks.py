"""Exercise every visualization the library can produce.

"All the figures work" is a claim, so it is checked rather than assumed.  Two passes:

* :func:`standalone_checks` builds every figure *function* on synthetic input, so a
  chart type cannot break unnoticed while no model happens to use it;
* :func:`catalogue_checks` renders every figure of every model in the catalogue - results,
  the classical comparison, the circuits, sweeps, Bloch spheres, trajectories, histograms,
  heatmaps - and verifies that each one carries an accessible description.

Both return rows of ``{name, ok, detail}``, which the CLI prints and the test suite
asserts, and both can write the images out for inspection.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .audit import structure_checks

__all__ = ["standalone_checks", "catalogue_checks", "figure_checks", "STANDALONE_FIGURES"]


def _rows(**kwargs) -> dict:
    return {"name": kwargs["name"], "ok": bool(kwargs["ok"]), "detail": kwargs.get("detail", "")}


def has_drawn_marks(figure) -> bool:
    """True when a figure actually contains something drawn.

    The library's rule is that an image is a graph or there is no image: a panel of text, or an
    empty frame, is not a visualization.  This is the check behind that rule.
    """
    for axis in getattr(figure, "axes", []):
        if axis.patches or axis.lines or axis.collections or axis.images:
            return True
    return False


def standalone_figures() -> dict:
    """One figure per visualization function, built from library objects.

    The inputs are the smallest ones that exercise the function: a two-question
    simulation for the distribution charts, its analysis reports for the order-effect and
    interference charts, a state for the Bloch sphere, a sweep for the line charts, a
    table for the heatmap and a circuit for the drawing.
    """
    import numpy as np

    from ..core import basis_state, question, run_sequence, unsharp_question
    from ..core.states import amplitudes
    from ..analysis.interference import interference_decomposition
    from ..analysis.order import order_effect
    from ..analysis.sweeps import sweep
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

    state = basis_state(0, 2)
    first = question(0.955, 0.0, "A")
    second = question(np.pi / 2, 0.0, "B")
    soft = unsharp_question(np.pi / 2, 0.9, 0.0, "B")
    forward = run_sequence(state, [first, soft])
    backward = run_sequence(state, [soft, first])
    report = order_effect(forward, backward, "A", "B")
    interference = interference_decomposition(state, first, soft)
    sweep_result = sweep("theta", [0.0, 0.5, 1.0, 1.5], lambda value: {"order effect": value / 2, "interference": value})
    frame = __import__("pandas").DataFrame(
        [{"first": "C", "second": "C", "payoff": 3.0}, {"first": "C", "second": "D", "payoff": 0.0},
         {"first": "D", "second": "C", "payoff": 5.0}, {"first": "D", "second": "D", "payoff": 1.0}]
    )
    from ..circuits.builder import build_sequence_circuit

    return {
        "probability_bars": lambda: probability_bars(forward),
        "marginal_bars": lambda: marginal_bars(forward),
        "order_effect_figure": lambda: order_effect_figure(report),
        "interference_figure": lambda: interference_figure(interference),
        "sweep_figure": lambda: sweep_figure(sweep_result),
        "line_figure": lambda: line_figure(
            {"x": [0, 1, 2, 3], "xlabel": "step", "ylabel": "opinion",
             "series": {f"agent {index + 1}": [0.1 * index + 0.05 * step for step in range(4)] for index in range(10)}}
        ),
        "histogram_figure": lambda: histogram_figure(
            {"x": list(range(5)), "xlabel": "votes in favour", "ylabel": "probability",
             "series": {"correlated": [0.3, 0.0, 0.0, 0.0, 0.7], "independent": [0.1, 0.2, 0.4, 0.2, 0.1]}}
        ),
        "bloch_sphere": lambda: bloch_sphere(amplitudes(1, 1j, label="belief")),
        "heatmap_figure": lambda: heatmap_figure(frame, title="payoff"),
        "circuit_figure": lambda: circuit_figure(build_sequence_circuit(state, [first, second]), title="A then B"),
        "classical_comparison_figure": lambda: classical_comparison_figure(
            [{"metric": "order effect", "model": 0.29, "baseline": 0.0, "note": ""},
             {"metric": "interference share", "model": 0.47, "baseline": 0.0, "note": ""}]
        ),
    }


STANDALONE_FIGURES = tuple(sorted(standalone_figures()))


def standalone_checks(directory: str | Path | None = None) -> list[dict]:
    """Build every figure function on synthetic input; return one row per function."""
    rows = []
    for name, builder in sorted(standalone_figures().items()):
        try:
            figure = builder()
            description = getattr(figure, "_qcog_description", "")
            described = len(description) >= 20
            drawn = has_drawn_marks(figure)
            ok = described and drawn
            detail = (
                f"description {len(description)} characters"
                if ok
                else ("no accessible description" if not described else "the figure contains no drawn marks")
            )
            if ok and directory is not None:
                path = Path(directory)
                path.mkdir(parents=True, exist_ok=True)
                figure.savefig(path / f"standalone__{name}.png", bbox_inches="tight")
                detail += f", written to {path / f'standalone__{name}.png'}"
            rows.append(_rows(name=name, ok=ok, detail=detail))
            plt.close(figure)
        except Exception as error:  # pragma: no cover - reported, not raised
            rows.append(_rows(name=name, ok=False, detail=f"{type(error).__name__}: {error}"))
    return rows


def catalogue_checks(slugs=None, directory: str | Path | None = None, sweeps: bool = True) -> list[dict]:
    """Render every figure of every model; one row per model.

    ``directory`` writes the images out; ``sweeps`` keeps the models' parameter sweeps
    (the slowest part) - the figures themselves are checked either way.
    """
    from ..registry import MODELS

    chosen = list(slugs) if slugs else list(MODELS)
    rows = []
    for slug in chosen:
        model = MODELS[slug]()
        try:
            result = model.run(sweeps=sweeps)
            figures = model.figures(result)
            descriptions = {
                name: len(getattr(figure, "_qcog_description", "")) for name, figure in figures.items()
            }
            missing = [name for name, length in descriptions.items() if length < 20]
            empty = [name for name, figure in figures.items() if not has_drawn_marks(figure)]
            if missing or empty:
                rows.append(
                    _rows(
                        name=slug,
                        ok=False,
                        detail=f"undescribed figures: {missing}; figures with nothing drawn: {empty}",
                    )
                )
                continue
            written = 0
            if directory is not None:
                path = Path(directory)
                path.mkdir(parents=True, exist_ok=True)
                for name, figure in figures.items():
                    safe = "".join(character if character.isalnum() else "_" for character in name)[:56]
                    figure.savefig(path / f"{slug}__{safe}.png", bbox_inches="tight")
                    written += 1
            for figure in figures.values():
                plt.close(figure)  # a check pass renders and releases
            kinds = sorted({name.split(":")[0].split("__")[0] for name in figures})
            rows.append(
                _rows(
                    name=slug,
                    ok=True,
                    detail=f"{len(figures)} figures ({', '.join(kinds)})"
                    + (f", {written} written" if directory is not None else ""),
                )
            )
        except Exception as error:  # pragma: no cover - reported, not raised
            rows.append(_rows(name=slug, ok=False, detail=f"{type(error).__name__}: {error}"))
    return rows


def figure_checks(directory: str | Path | None = None, slugs=None, sweeps: bool = True) -> list[dict]:
    """Every check, in one list: the figure functions, the drawings, then the whole catalogue."""
    return standalone_checks(directory) + structure_checks() + catalogue_checks(slugs, directory, sweeps)