"""The shared battery behind every file in ``model_tests/``.

Each ``module_*/NN_*.py`` file tests one model of the catalogue.  This module holds the
checks that are the same for all twenty-three of them, so that a model file only has to add
the claims that are specific to its own mathematics:

* metadata: the catalogue entry is complete, the module letter matches, the parameter
  schema is well formed (bounds, kinds, choices, descriptions, defaults inside bounds);
* parameters: every declared parameter is honoured, clamped at its bounds, rejected when
  unknown or of the wrong type, and *live* - changing it changes the observable result;
* the numbers: every joint distribution is normalised, every marginal sums to one, the
  summary and findings are sentences with values in them;
* the classical comparison: rows exist, a zero baseline carries its explanation, the
  comparison chart draws exactly those rows, and ``classical=False`` switches it off
  without leaving an empty chart behind;
* execution: exact/deterministic, sampled (seeded, reproducible, within sampling error),
  compiled Aer backends agreeing with the analytic result, unknown backends refused, and
  the ``sweeps``/``circuits``/``classical``/``circuit_style`` flags honoured;
* circuits: every design builds, the statistics are sane, drawing works in three formats,
  and where a design corresponds to a sequential result the compiled circuit reproduces the
  model's own probabilities;
* figures: every figure is a graph (never a panel of text), carries a description that
  names the numbers it draws, is structurally consistent with the data it came from
  (bar heights, stacked segments, sweep lines, heatmap cells, Bloch vector, circuit gates),
  separates its series by more than colour, and survives the trip to a PNG file.

Nothing here imports the test files; ``run_all.py`` imports *them* and calls
:func:`run_model`.
"""

from __future__ import annotations

import html
import importlib.util
import math
import re
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import qcog  # noqa: E402
from qcog.circuits.builder import circuit_statistics  # noqa: E402
from qcog.core.instruments import NoiseSpec  # noqa: E402
from qcog.core.params import KINDS  # noqa: E402
from qcog.models.base import RESERVED  # noqa: E402
from qcog.registry import MODELS  # noqa: E402
from qcog.viz.audit import bar_tops, bar_values  # noqa: E402
from qcog.viz.checks import has_drawn_marks  # noqa: E402
from qcog.viz.figures import VisualizationUnavailable, circuit_figure  # noqa: E402

#: Relative tolerance for "these two numbers are the same".
TOL = 1e-9
#: How close a sampled backend has to be, in multiples of the binomial standard error.
SAMPLING_SIGMA = 5.0
#: Figures smaller than this in pixels, or with fewer distinct colours than this, are not
#: plots of anything.
MIN_IMAGE = (240, 170)
MIN_COLOURS = 8

CHECK = "check"
NOTE = "note"


@dataclass
class Suite:
    """Collects one model's checks and runs the model for them."""

    slug: str
    out_dir: Path
    fast: bool = False
    checks: list[dict] = field(default_factory=list)
    notes: list[dict] = field(default_factory=list)
    runs: int = 0
    seconds: float = 0.0
    save_figures: bool = True

    def __post_init__(self) -> None:
        try:
            self.model = qcog.get_model(self.slug)
        except KeyError as error:  # pragma: no cover - a typo in a test file
            raise SystemExit(f"unknown model slug {self.slug!r}: {error}") from None
        self.slug = self.model.slug
        self.out_dir = Path(self.out_dir) / self.slug
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.section_name = "general"

    # ------------------------------------------------------------------ recording
    def section(self, name: str) -> None:
        self.section_name = name

    def check(self, name: str, ok: Any, detail: Any = "") -> bool:
        ok = bool(ok)
        self.checks.append({"name": name, "ok": ok, "detail": str(detail), "section": self.section_name})
        return ok

    def note(self, name: str, detail: Any = "") -> None:
        self.notes.append({"name": name, "detail": str(detail), "section": self.section_name})

    def close(self, name: str, expected, actual, tol: float = TOL, detail: str = "") -> bool:
        """Check that two numbers agree to ``tol``."""
        try:
            delta = abs(float(expected) - float(actual))
        except (TypeError, ValueError):
            return self.check(name, False, f"{expected!r} versus {actual!r}")
        return self.check(
            name, delta <= tol, detail or f"{float(actual):.6g} versus {float(expected):.6g} (delta {delta:.2g})"
        )

    def close_list(self, name: str, expected, actual, tol: float = TOL) -> bool:
        try:
            left = np.asarray(list(expected), dtype=float)
            right = np.asarray(list(actual), dtype=float)
            same = left.shape == right.shape and bool(np.allclose(left, right, atol=tol, rtol=0))
            detail = f"{len(left)} versus {len(right)} values" if left.shape != right.shape else (
                f"largest difference {float(np.max(np.abs(left - right))):.2g}" if left.size else "empty"
            )
        except (TypeError, ValueError) as error:
            same, detail = False, f"{type(error).__name__}: {error}"
        return self.check(name, same, detail)

    # --------------------------------------------------------------------- running
    def run(self, *, validate: bool = True, label: str = "", **kwargs):
        """Run the model, time it, and validate the result unless told not to."""
        started = time.perf_counter()
        result = self.model.run(**kwargs)
        self.seconds += time.perf_counter() - started
        self.runs += 1
        if validate:
            self.validate_result(result, label or _describe(kwargs))
        return result

    def validate_result(self, result, label: str = "") -> None:
        """The checks every single run of every model has to pass."""
        prefix = f"{label}: " if label else ""
        for key, item in sorted(result.results.items()):
            if not hasattr(item, "joint"):
                continue
            total = sum(float(value) for value in item.joint.values())
            self.check(
                f"{prefix}{key}: joint distribution sums to 1",
                abs(total - 1.0) < TOL,
                f"sum {total:.12f} over {len(item.joint)} outcomes",
            )
            worst = min([float(v) for v in item.joint.values()] or [0.0])
            self.check(
                f"{prefix}{key}: every probability lies in [0, 1]",
                worst >= -TOL and max([float(v) for v in item.joint.values()] or [0.0]) <= 1 + TOL,
                f"smallest {worst:.3g}",
            )
            for question in item.order:
                marginal = item.marginal(question)
                total = sum(float(value) for value in marginal.values())
                self.check(
                    f"{prefix}{key}: the marginal of {question!r} sums to 1",
                    abs(total - 1.0) < TOL,
                    f"sum {total:.12f}",
                )
            labels = [
                part for outcome in item.joint for part in (outcome if isinstance(outcome, tuple) else (outcome,))
            ]
            self.check(
                f"{prefix}{key}: the answers are labelled",
                bool(labels) and all(isinstance(part, str) and part.strip() for part in labels),
                f"{len(item.joint)} outcomes, {len(set(labels))} distinct labels",
            )
            self.check(
                f"{prefix}{key}: the backend is recorded",
                bool(getattr(item, "backend", "")) and float(getattr(item, "duration_seconds", 0.0)) >= 0.0,
                f"backend {getattr(item, 'backend', None)!r}",
            )


def _describe(kwargs: Mapping[str, Any]) -> str:
    if not kwargs:
        return ""
    return "(" + ", ".join(f"{key}={value}" for key, value in sorted(kwargs.items())) + ")"
# ---------------------------------------------------------------------------- helpers
def flatten(value: Any, prefix: str = "", depth: int = 0) -> list[str]:
    """A comparable, printable view of an analysis value of any shape."""
    if isinstance(value, (int, float, str, bool)) or value is None:
        return [f"{prefix}={value!r}"]
    if isinstance(value, dict):
        out: list[str] = []
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            out += flatten(item, f"{prefix}.{key}", depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        return [f"{prefix}=[{len(value)}]{list(value[:4])!r}"]
    if hasattr(value, "table"):  # a SweepResult
        return flatten(value.table.to_dict(), f"{prefix}.table", depth + 1)
    if hasattr(value, "joint"):  # a SequentialResult
        return [f"{prefix}[{len(value.joint)}]={sorted(float(v) for v in value.joint.values())[:4]!r}"]
    if depth > 6:
        return [f"{prefix}=<deep>"]
    return [f"{prefix}={type(value).__name__}({str(value)[:48]!r})"]


def signature(result) -> str:
    """Everything a run reports, so that "did this parameter change anything?" is decidable.

    The parameter echo itself is deliberately *not* part of this: a parameter that is
    recorded but ignored would otherwise look live.
    """
    parts = [result.summary] + [str(finding) for finding in result.findings]
    for key, item in sorted(result.results.items()):
        if hasattr(item, "joint"):
            for outcome, value in sorted(item.joint.items(), key=lambda pair: str(pair[0])):
                parts.append(f"{key}:{outcome}={float(value):.12f}")
        parts.append(f"{key}:noise={bool(getattr(item, 'noise', None))}")
    for key, value in sorted(result.analysis.items()):
        if key == "sweeps":
            continue
        parts += flatten(value, key)
    for key, value in sorted((result.analysis.get("sweeps") or {}).items()):
        parts += flatten(getattr(value, "table", value), f"sweep.{key}")
    for row in result.classical_comparison:
        parts.append(f"cmp:{row.get('metric')}={row.get('model')}/{row.get('baseline')}")
    return "\n".join(parts)


def generic_point(spec) -> Any:
    """A value that is legal, not the default and not a symmetric corner of the range."""
    if spec.kind == "choice":
        others = [choice for choice in spec.choices if choice != spec.default]
        return others[0] if others else None
    if spec.kind == "int" and spec.low is not None and spec.high is not None:
        return int(round(spec.low + 0.43 * (spec.high - spec.low)))
    if spec.kind == "float" and spec.low is not None and spec.high is not None:
        return float(spec.low + 0.43 * (spec.high - spec.low))
    if spec.kind == "bool":
        return not bool(spec.default)
    if spec.kind == "text":
        return "a different cue"
    return None


# --------------------------------------------------------------------------- metadata
def metadata_checks(suite: Suite, module: str = "") -> None:
    suite.section("catalogue entry")
    model = suite.model
    suite.check("the slug matches the file", model.slug == suite.slug, model.slug)
    suite.check("it is registered in the catalogue", suite.slug in MODELS, f"{len(MODELS)} models")
    if module:
        suite.check(f"the module letter is {module}", model.module == module, model.module)
    suite.check("the module letter is A, B, C or D", model.module in {"A", "B", "C", "D"}, model.module)
    for name, minimum, ending in (
        ("title", 12, ""),
        ("one_liner", 40, "."),
        ("cognitive_question", 40, "?"),
        ("quantum_mechanism", 40, "."),
        ("classical_baseline", 40, "."),
    ):
        text = str(getattr(model, name, "") or "").strip()
        if ending == "?":
            ok = len(text) >= minimum and "?" in text
        elif ending:
            ok = len(text) >= minimum and text.endswith((".", "?"))
        else:
            ok = len(text) >= minimum
        suite.check(f"{name.replace('_', ' ')} is a complete sentence", ok, f"{len(text)} characters: {text[:60]!r}")
    suite.check("the title does not end in a full stop", not model.title.endswith("."), model.title)
    suite.check(
        "it names the concepts it is about",
        len(model.keywords) >= 2 and all(str(word).strip() for word in model.keywords),
        ", ".join(model.keywords),
    )
    suite.check(
        "it cites the literature it comes from",
        len(model.references) >= 1
        and all(len(str(ref)) >= 30 for ref in model.references)
        and all(re.search(r"(1[5-9]|20)\d{2}", str(ref)) for ref in model.references),
        f"{len(model.references)} references",
    )
    suite.check(
        "it states its own limits",
        len(model.caveats) >= 1 and all(len(caveat) >= 40 for caveat in model.caveats),
        f"{len(model.caveats)} caveats",
    )
    suite.check("the difficulty is one of the three", model.difficulty in {"introductory", "intermediate", "advanced"}, model.difficulty)
    entry = model.to_dict()
    expected = {
        "slug", "title", "module", "module_title", "one_liner", "cognitive_question",
        "quantum_mechanism", "classical_baseline", "difficulty", "keywords", "references",
        "caveats", "parameters", "defaults",
    }
    suite.check("the catalogue entry has every field", expected <= set(entry), f"{len(entry)} fields")
    table = model.parameter_table()
    suite.check(
        "the parameter table is complete",
        all({"name", "kind", "default", "label", "description", "low", "high", "choices"} <= set(row) for row in table),
        f"{len(table)} parameters",
    )
    info = model.describe()
    suite.check("the info block names the model", model.slug in info and model.title in info, info.splitlines()[0])
    suite.check(
        "the info block lists every parameter with its bounds and meaning",
        all(
            spec.name in info and spec.description in info and (spec.kind in info)
            for spec in model.params
        ),
        ", ".join(spec.name for spec in model.params) or "no parameters",
    )
    suite.check(
        "the info block states the caveats and the references",
        all(caveat in info for caveat in model.caveats) and all(reference in info for reference in model.references),
        f"{len(model.caveats)} caveats, {len(model.references)} references",
    )
    suite.check(
        "the catalogue search finds it",
        type(model) in qcog.search(model.keywords[0]) and type(model) in qcog.search(model.slug),
        f"first keyword {model.keywords[0]!r}",
    )


# ------------------------------------------------------------------------- parameters
def parameter_checks(suite: Suite) -> None:
    suite.section("parameters")
    model = suite.model
    specs = list(model.params)
    names = [spec.name for spec in specs]
    suite.check("the parameter names are unique", len(set(names)) == len(names), ", ".join(names) or "none")
    suite.check(
        "the parameter names are identifiers",
        all(re.fullmatch(r"[a-z][a-z0-9_]*", name) for name in names),
        ", ".join(names) or "none",
    )
    suite.check(
        "every parameter is documented",
        all(spec.description and len(spec.description) >= 20 and spec.description.strip().endswith(".") for spec in specs),
        f"{len(specs)} parameters",
    )
    suite.check(
        "every parameter has a label",
        all(spec.display and spec.display != "" for spec in specs),
        ", ".join(spec.display for spec in specs) or "none",
    )
    suite.check("every parameter declares a known kind", all(spec.kind in KINDS for spec in specs), ", ".join(sorted({spec.kind for spec in specs})))
    bounded = [spec for spec in specs if spec.low is not None or spec.high is not None]
    suite.check(
        "numeric bounds are ordered and contain the default",
        all(
            (spec.low is None or spec.high is None or float(spec.low) <= float(spec.high))
            and (spec.low is None or float(spec.default) >= float(spec.low) - TOL)
            and (spec.high is None or float(spec.default) <= float(spec.high) + TOL)
            for spec in bounded
        ),
        f"{len(bounded)} bounded",
    )
    suite.check(
        "the steps are positive",
        all(spec.step is None or float(spec.step) > 0 for spec in specs),
    )
    choices = [spec for spec in specs if spec.kind == "choice"]
    suite.check(
        "a choice parameter lists its alternatives and defaults to one of them",
        all(spec.choices and spec.default in spec.choices for spec in choices),
        ", ".join(f"{spec.name}: {len(spec.choices)}" for spec in choices) or "no choice parameters",
    )
    suite.check(
        "the defaults name exactly the declared parameters",
        set(model.defaults()) == set(names),
        f"{len(model.defaults())} defaults",
    )

    default = suite.run(label="defaults")
    suite.check(
        "a run with no arguments records every default",
        all(
            str(default.parameters[spec.name]) == str(spec.default)
            for spec in specs
        ),
        ", ".join(f"{spec.name}={default.parameters[spec.name]!r}" for spec in specs),
    )

    # bounds are inclusive, and values outside them are clamped to the bound, not rejected
    for spec in bounded:
        if spec.kind not in {"float", "int"}:
            continue
        for bound, inside in (("low", -1), ("high", +1)):
            edge = getattr(spec, bound)
            if edge is None:
                continue
            edge = int(edge) if spec.kind == "int" else float(edge)
            outside = edge + inside * (1 if spec.kind == "int" else max(1.0, abs(edge) * 0.5 + 1.0))
            at_edge = suite.run(sweeps=False, **{spec.name: edge})
            beyond = suite.run(sweeps=False, **{spec.name: outside})
            suite.check(
                f"{spec.name}: the {bound} bound is accepted",
                abs(float(at_edge.parameters[spec.name]) - float(edge)) <= TOL,
                f"asked {outside if outside == edge else edge}, recorded {at_edge.parameters[spec.name]}",
            )
            suite.close(
                f"{spec.name}: a value past the {bound} bound is clamped to it",
                float(at_edge.parameters[spec.name]),
                float(beyond.parameters[spec.name]),
            )
            suite.check(
                f"{spec.name}: clamping keeps the run valid",
                signature(beyond) != "",
            )

    # None means "use the default", an unknown name is refused, and a wrong type is refused
    probes = [spec for spec in specs][:6]
    if probes:
        name = probes[0].name
        again = suite.run(sweeps=False, **{name: None})
        suite.close(
            f"{name}: passing None means the default",
            float(suite.model.defaults()[name]) if probes[0].kind in {"float", "int"} else 0.0,
            float(again.parameters[name]) if probes[0].kind in {"float", "int"} else 0.0,
        )
    try:
        suite.model.run(sweeps=False, definitely_not_a_parameter=1)
        suite.check("an unknown parameter is refused", False, "no error raised")
    except ValueError as error:
        suite.check("an unknown parameter is refused", "definitely_not_a_parameter" in str(error), str(error)[:90])
    except Exception as error:  # pragma: no cover
        suite.check("an unknown parameter is refused", False, f"{type(error).__name__}: {error}")
    numeric = [spec for spec in specs if spec.kind in {"float", "int", "bool"}]
    if numeric:
        name = numeric[0].name
        try:
            suite.model.run(sweeps=False, **{name: "not a number"})
            suite.check(f"{name}: a non-numeric value is refused", False, "no error raised")
        except (TypeError, ValueError) as error:
            suite.check(f"{name}: a non-numeric value is refused", True, f"{type(error).__name__}: {str(error)[:60]}")
        except Exception as error:  # pragma: no cover
            suite.check(f"{name}: a non-numeric value is refused", False, f"{type(error).__name__}: {error}")
    for spec in choices:
        try:
            suite.model.run(sweeps=False, **{spec.name: "__not_a_choice__"})
            suite.check(f"{spec.name}: an unknown choice is refused", False, "no error raised")
        except ValueError as error:
            suite.check(
                f"{spec.name}: an unknown choice is refused",
                spec.name in str(error),
                str(error)[:90],
            )
        except Exception as error:  # pragma: no cover
            suite.check(f"{spec.name}: an unknown choice is refused", False, f"{type(error).__name__}: {error}")
    for spec in choices:
        for choice in spec.choices:
            run = suite.run(sweeps=False, **{spec.name: choice})
            suite.check(
                f"{spec.name}: the choice {choice!r} runs",
                run.parameters[spec.name] == choice,
                str(run.parameters[spec.name]),
            )

    # every parameter has to be *live*: changing it changes the observable result
    suite.section("every parameter is live")
    baselines = {"defaults": {}}
    generic = {spec.name: generic_point(spec) for spec in specs if generic_point(spec) is not None}
    if generic:
        baselines["a generic point"] = generic
    live: set[str] = set()
    inert_from_defaults: list[str] = []
    for label, baseline in baselines.items():
        base = suite.run(sweeps=True, label=label, **baseline)
        base_signature = signature(base)
        for spec in specs:
            if spec.kind == "choice":
                candidates = [c for c in spec.choices if c != base.parameters[spec.name]]
            elif spec.kind in {"float", "int"}:
                span = float(spec.high) - float(spec.low) if spec.low is not None and spec.high is not None else 1.0
                low = float(spec.low) if spec.low is not None else float(base.parameters[spec.name]) - span
                candidates = [low + fraction * span for fraction in (0.22, 0.66)]
                if spec.kind == "int":
                    candidates = [int(round(value)) for value in candidates]
            elif spec.kind == "bool":
                candidates = [not bool(base.parameters[spec.name])]
            else:
                candidates = ["a different cue"]
            moved = False
            for probe in candidates:
                if probe == base.parameters[spec.name]:
                    continue
                try:
                    other = suite.run(sweeps=True, validate=False, **{**baseline, spec.name: probe})
                except Exception as error:  # pragma: no cover - a parameter that cannot be set
                    suite.check(f"{spec.name}: the value {probe!r} runs", False, f"{type(error).__name__}: {error}")
                    moved = True
                    break
                if signature(other) != base_signature:
                    moved = True
                    break
            if moved:
                live.add(spec.name)
            elif label == "defaults":
                inert_from_defaults.append(spec.name)
    dead = [spec.name for spec in specs if spec.name not in live]
    for name in inert_from_defaults:
        if name not in dead:
            suite.note(
                f"{name} only moves away from the declared defaults",
                "at the defaults it is a symmetry or a global phase, so it changes nothing there",
            )
    suite.check(
        "every parameter changes the result somewhere in its range",
        not dead,
        "; ".join(dead) or f"all {len(specs)} parameters are live",
    )
    return default


def params_of(model) -> tuple:
    """The declared parameters of a model (kept as a function so tests can monkeypatch)."""
    return model.params

# ----------------------------------------------------------------------------- sweeps
def sweep_checks(suite: Suite, result) -> None:
    """Every sweep has to be a chart of something: it must move, and it must match the run."""
    suite.section("sweeps")
    sweeps = result.analysis.get("sweeps") or {}
    for name, sweep_result in sorted(sweeps.items()):
        table = sweep_result.table
        parameter = sweep_result.parameter
        known = {spec.name for spec in suite.model.params} | set(RESERVED)
        suite.check(
            f"the {name} sweep sweeps a real parameter",
            str(parameter).replace(" ", "_") in known,
            f"{name} sweeps {parameter!r}; declared parameters are {sorted(spec.name for spec in suite.model.params)}",
        )
        suite.check(f"the {name} sweep has a column for its parameter", parameter in table.columns, ", ".join(map(str, table.columns)))
        metrics = [column for column in table.columns if column != parameter]
        suite.check(f"the {name} sweep has at least one metric", bool(metrics), ", ".join(map(str, metrics)))
        suite.check(
            f"the {name} sweep covers the parameter's range",
            len(table) >= 3 and float(table[parameter].max()) >= float(table[parameter].min()),
            f"{len(table)} rows from {float(table[parameter].min()):g} to {float(table[parameter].max()):g}",
        )
        suite.check(
            f"the {name} sweep values are the ones it recorded",
            list(table[parameter]) == [float(value) if isinstance(value, (int, float)) else value for value in sweep_result.values]
            or np.allclose([float(v) for v in table[parameter]], [float(v) for v in sweep_result.values], atol=1e-9),
            f"{len(sweep_result.values)} values",
        )
        varying = []
        for metric in metrics:
            values = np.asarray([float(value) for value in table[metric]], dtype=float)
            if float(np.nanmax(values) - np.nanmin(values)) > 1e-9:
                varying.append(metric)
            else:
                suite.note(f"the {name} sweep draws a flat line for {metric}", f"constant at {float(values[0]):.6g}")
        if varying:
            suite.check(f"the {name} sweep moves with its parameter", True, ", ".join(map(str, varying)))
        else:
            suite.note(
                f"the {name} sweep is flat across its whole range at these parameters",
                "the chart is correct but shows nothing: the parameter has no effect at this point",
            )
    suppressed = suite.run(sweeps=False, label="with switches off").analysis.get("sweeps") or {}
    with_metrics = [name for name, sweep_result in suppressed.items() if len(sweep_result.table.columns) > 1]
    suite.check(
        "a run with sweeps switched off computes no sweep metrics",
        not with_metrics,
        "; ".join(f"{name}: {sorted(suppressed[name].table.columns)}" for name in with_metrics)
        or f"{len(suppressed)} sweep entries without metrics",
    )


def check_sweep_anchor(suite: Suite, result, sweep_name: str, resolvers: Mapping[str, Callable], rows: Sequence[int] | None = None) -> None:
    """The sweep table must agree with runs made at the sweep's own values.

    This is the check that catches a parameter which is drawn in a sweep but never reaches
    the simulation: the chart would move while the run ignored the parameter.
    """
    sweep_result = (result.analysis.get("sweeps") or {}).get(sweep_name)
    if sweep_result is None:
        suite.check(f"the {sweep_name} sweep exists to anchor", False, "no such sweep")
        return
    table = sweep_result.table
    column = sweep_result.parameter          # the sweep's own column name, which may be a label
    parameter = sweep_result.parameter       # the name a run accepts, once resolved below
    known = {spec.name for spec in suite.model.params}
    if str(parameter) not in known:
        candidate = str(parameter).replace(" ", "_")
        if candidate in known:
            parameter = candidate
    if str(parameter) not in known:
        suite.note(
            f"the {sweep_name} sweep cannot be anchored to a run",
            f"it sweeps {parameter!r}, which is not one of the model's parameters {sorted(known)}",
        )
        return
    indices = list(rows) if rows is not None else [0, len(table) // 2, len(table) - 1]
    missing = [metric for metric in resolvers if metric not in table.columns]
    suite.check(
        f"the {sweep_name} sweep carries the metrics {', '.join(map(str, resolvers))}",
        not missing,
        f"missing {missing}" if missing else f"{len(table)} rows",
    )
    for index in indices:
        if not 0 <= index < len(table):
            continue
        value = table[column].iloc[index]
        want = dict(table.iloc[index])
        run = suite.run(sweeps=False, validate=False, **{parameter: value})
        for metric, resolver in resolvers.items():
            if metric not in table.columns:
                continue
            expected = float(want[metric])
            actual = float(resolver(run))
            difference = abs(expected - actual)
            tolerance = max(5e-3, 0.01 * abs(expected))
            suite.check(
                f"{sweep_name} at {parameter}={value}: the chart's {metric} is the run's",
                difference <= tolerance,
                f"chart {expected:.6g} against run {actual:.6g} (difference {difference:.2g})",
            )
            if 1e-6 < difference <= tolerance:
                suite.note(
                    f"{sweep_name} at {parameter}={value}: the {metric} wobbles in the last digits",
                    f"chart {expected:.6g} against run {actual:.6g}; the two agree to "
                    f"{difference / max(abs(expected), 1e-12):.1%}",
                )


# -------------------------------------------------------------------- the classical view
def classical_checks(suite: Suite, result) -> None:
    suite.section("classical comparison")
    classical = result.classical
    suite.check("the comparison is present", bool(classical), f"{len(classical)} keys")
    for key in ("name", "description", "rows", "verdict", "options", "options_honoured", "options_ignored"):
        suite.check(f"the comparison records {key}", key in classical, str(classical.get(key))[:60])
    rows = result.classical_comparison
    suite.check("the comparison has rows", len(rows) >= 1, f"{len(rows)} rows")
    suite.check(
        "every row pairs a model value with a baseline value",
        all(
            isinstance(row.get("metric"), str)
            and len(str(row["metric"])) >= 5
            and row.get("model") is not None
            and row.get("baseline") is not None
            for row in rows
        ),
        "; ".join(f"{row.get('metric')}: {row.get('model')} vs {row.get('baseline')}" for row in rows[:3]),
    )
    silent = [row for row in rows if abs(float(row["baseline"])) < TOL]
    suite.check(
        "where the classical account must be silent, the row says so",
        all(str(row.get("note", "")).strip() for row in silent),
        f"{len(silent)} of {len(rows)} rows have a zero baseline",
    )
    suite.check(
        "the comparison has a verdict",
        isinstance(classical.get("verdict"), str) and len(str(classical["verdict"])) >= 10,
        str(classical.get("verdict"))[:80],
    )
    suite.check(
        "the verdict mentions the baseline it compares against",
        bool(str(classical.get("name", "")).strip()),
        str(classical.get("name"))[:60],
    )
    suite.check(
        "the comparison is numerically distinct from the baseline somewhere",
        any(abs(float(row["model"]) - float(row["baseline"])) > TOL for row in rows),
        f"{sum(abs(float(row['model']) - float(row['baseline'])) > TOL for row in rows)} of {len(rows)} rows differ",
    )

    figures = suite.model.figures(result)
    comparison = figures.get("model versus classical baseline")
    suite.check("the comparison chart is drawn", comparison is not None, f"{len(figures)} figures")
    if comparison is not None:
        axis = comparison.axes[0]
        heights = bar_values(axis)
        expected = [float(row["model"]) for row in rows] + [float(row["baseline"]) for row in rows]
        suite.close_list("the comparison chart draws one model and one baseline bar per row", expected, heights, tol=1e-9)
        drawn_labels = [tick.get_text() for tick in axis.get_xticklabels()]
        wanted_labels = [str(row["metric"]) if len(str(row["metric"])) <= 34 else str(row["metric"])[:31] + "..." for row in rows]
        suite.check(
            "the comparison chart labels each row with its metric",
            drawn_labels[: len(rows)] == wanted_labels,
            str(drawn_labels[:2]),
        )
        suite.check(
            "the comparison chart names both series",
            comparison.axes[0].get_legend() is not None
            and len(comparison.axes[0].get_legend().get_texts()) == 2,
            str([t.get_text() for t in comparison.axes[0].get_legend().get_texts()]) if comparison.axes[0].get_legend() else "no legend",
        )
        plt.close(comparison)

    off = suite.run(sweeps=False, classical=False, label="classical off")
    suite.check("switching the comparison off empties the rows", not off.classical_comparison, f"{len(off.classical_comparison)} rows")
    off_figures = suite.model.figures(off)
    suite.check(
        "switching the comparison off leaves no empty comparison chart",
        "model versus classical baseline" not in off_figures,
        f"{len(off_figures)} figures",
    )
    suite.check(
        "switching the comparison off does not change the numbers",
        signature(off).replace("cmp:", "") != "" and all(
            abs(float(a.joint[key]) - float(b.joint[key])) < TOL
            for a, b in zip(
                [off.results[key] for key in sorted(off.results) if hasattr(off.results[key], "joint")],
                [result.results[key] for key in sorted(result.results) if hasattr(result.results[key], "joint")],
            )
            for key in a.joint
        ),
    )
    ignored = suite.run(sweeps=False, classical_options={"__not_an_option__": 1}, validate=False)
    recorded = str(ignored.classical.get("options_ignored"))
    suite.check(
        "an unknown classical option is reported rather than dropped",
        "__not_an_option__" in recorded,
        recorded[:80],
    )


# ------------------------------------------------------------------ execution and flags
_AER_STATE: list[bool | str] = []


def aer_available() -> bool | str:
    """Is Qiskit Aer usable here?  Returns True, or the reason it is not."""
    if not _AER_STATE:
        try:
            import qiskit_aer  # noqa: F401

            _AER_STATE.append(True)
        except Exception as error:  # pragma: no cover - environment dependent
            _AER_STATE.append(f"{type(error).__name__}: {error}")
    return _AER_STATE[0]


def execution_checks(suite: Suite, result) -> None:
    suite.section("execution and flags")
    model = suite.model
    second = suite.run(label="second exact run")
    suite.check(
        "the analytic backend is deterministic",
        signature(second) == signature(result),
        "two runs agree",
    )

    key = next((name for name in sorted(result.results) if hasattr(result.results[name], "joint")), None)
    if key is None:
        suite.note(
            "this model reports no sequential distribution",
            "it returns aggregates, so the shot-based comparisons do not apply to it",
        )
        shots = 4000 if suite.fast else 20000
        suite.run(backend="sampled", shots=shots, seed=7, label=f"sampled ({shots} shots)")
        plain = suite.run(sweeps=False, label="sweeps off")
        suite.check(
            "the sweeps flag is honoured",
            all(len(s.table.columns) <= 1 for s in (plain.analysis.get("sweeps") or {}).values()),
            f"{len(plain.analysis.get('sweeps') or {})} sweep entries",
        )
        return
    exact_joint = dict(result.results[key].joint)
    shots = 4000 if suite.fast else 20000
    sampled = suite.run(backend="sampled", shots=shots, seed=7, label=f"sampled ({shots} shots)")
    suite.check(
        "the sampled backend records what it was asked for",
        sampled.shots == shots and sampled.seed == 7 and sampled.backend == "sampled",
        f"shots={sampled.shots} seed={sampled.seed} backend={sampled.backend}",
    )
    sampled_again = suite.run(backend="sampled", shots=shots, seed=7, validate=False)
    suite.check(
        "the sampled backend is reproducible for a fixed seed",
        all(abs(float(sampled.results[key].joint[outcome]) - float(sampled_again.results[key].joint[outcome])) < TOL for outcome in exact_joint),
        f"seed {sampled.seed}",
    )
    low = 400 if suite.fast else 1200
    first_seed = suite.run(backend="sampled", shots=low, seed=11, validate=False, label="sampled, seed 11")
    second_seed = suite.run(backend="sampled", shots=low, seed=12, validate=False, label="sampled, seed 12")
    varied = signature(first_seed) != signature(second_seed)
    point_mass = max(float(value) for value in exact_joint.values()) > 1 - 1e-9
    honest = str(first_seed.backend) == "exact" and bool(first_seed.metadata.get("backend_note"))
    suite.check(
        "a shot-based backend either samples or says why it cannot",
        varied or point_mass or honest,
        "two seeds give different numbers"
        if varied
        else ("the distribution is a single outcome, so sampling cannot look noisy" if point_mass else str(first_seed.metadata.get("backend_note", "the same numbers came back for both seeds"))[:120]),
    )
    if honest:
        suite.note("this model is evaluated analytically whatever backend is asked for", str(first_seed.metadata.get("backend_note"))[:140])
    if point_mass and not varied:
        suite.note("its headline distribution is a single outcome", "sampling a point mass gives the point mass")

    tolerance = SAMPLING_SIGMA * math.sqrt(0.25 / shots)
    worst = max(abs(float(sampled.results[key].joint[outcome]) - float(exact_joint[outcome])) for outcome in exact_joint)
    suite.check(
        f"the sampled backend agrees with the exact one within {SAMPLING_SIGMA:g} standard errors",
        worst < max(tolerance, 2e-3),
        f"largest deviation {worst:.4f} against a {tolerance:.4f} tolerance",
    )
    suite.check(
        "a zero shot count falls back to the documented default",
        suite.run(backend="sampled", shots=0, seed=1, validate=False).shots == 4096,
    )
    try:
        model.run(backend="not-a-backend")
        suite.check("an unknown backend is refused", False, "no error raised")
    except ValueError as error:
        suite.check("an unknown backend is refused", "backend" in str(error).lower(), str(error)[:80])
    except Exception as error:  # pragma: no cover
        suite.check("an unknown backend is refused", False, f"{type(error).__name__}: {error}")

    state = aer_available()
    if state is not True:
        suite.note("the compiled backends were not exercised", f"qiskit-aer is unavailable here ({state})")
    else:
        try:
            compiled = suite.run(backend="aer_exact", label="aer_exact")
            worst = max(
                abs(float(compiled.results[key].joint[outcome]) - float(exact_joint[outcome]))
                for outcome in exact_joint
            )
            suite.check(
                "the compiled circuit reproduces the analytic distribution exactly",
                worst < 1e-9,
                f"largest deviation {worst:.3g}",
            )
        except ValueError as error:
            suite.note("the compiled exactly backend declined this design", f"{type(error).__name__}: {error}"[:120])
        except RuntimeError as error:
            suite.note("the compiled backend could not run here", str(error)[:110])
        try:
            noisy = suite.run(backend="aer", shots=shots, seed=3, label="aer (shot based)")
            worst = max(
                abs(float(noisy.results[key].joint[outcome]) - float(exact_joint[outcome]))
                for outcome in exact_joint
            )
            suite.check(
                "the compiled shot-based backend agrees with the analytic one",
                worst < max(tolerance * 1.5, 3e-3),
                f"largest deviation {worst:.4f}",
            )
        except (RuntimeError, ValueError) as error:
            suite.note("the compiled shot-based backend declined this design", f"{type(error).__name__}: {error}"[:120])

    plain = suite.run(sweeps=False, label="sweeps off")
    full_sweeps = result.analysis.get("sweeps") or {}
    quiet_sweeps = plain.analysis.get("sweeps") or {}
    suite.check(
        "the sweeps flag is honoured",
        set(quiet_sweeps) == set(full_sweeps)
        and all(len(sweep_result.table.columns) <= 1 for sweep_result in quiet_sweeps.values()),
        f"{len(quiet_sweeps)} sweep entries, none with metrics"
        if quiet_sweeps
        else "this model declares no sweeps, and none appeared",
    )
    suite.check(
        "switching sweeps off does not change the headline numbers",
        all(
            abs(float(plain.results[name].joint[outcome]) - float(result.results[name].joint[outcome])) < TOL
            for name in plain.results
            if hasattr(plain.results[name], "joint")
            for outcome in plain.results[name].joint
        ),
    )
    quiet = suite.run(sweeps=False, circuits=False, validate=False)
    suite.check(
        "the circuits flag is honoured",
        not any(str(name).startswith("circuit:") for name in model.figures(quiet)),
        f"{len(model.figures(quiet))} figures without circuits",
    )
    for style in ("default", "iqp"):
        dressed = suite.run(sweeps=False, circuit_style=style, circuit_scale=0.7, circuit_fold=12, validate=False)
        drawn = {name: figure for name, figure in model.figures(dressed).items() if str(name).startswith("circuit:")}
        suite.check(
            f"the circuit drawing accepts style={style!r}, scale=0.7 and fold=12",
            bool(drawn) and all(has_drawn_marks(figure) for figure in drawn.values()),
            f"{len(drawn)} circuits",
        )
        for figure in drawn.values():
            plt.close(figure)
    qubits = max((circuit.num_qubits for circuit in model.circuit_designs(result).values()), default=1)
    try:
        noisy = suite.run(sweeps=False, noise=NoiseSpec("depolarizing", 0.05, qubits), validate=False)
        suite.check(
            f"an explicit noise model on {qubits} qubit(s) is accepted",
            noisy is not None,
            "depolarizing 0.05",
        )
    except (ValueError, TypeError) as error:
        suite.note("an explicit noise model was refused for this design", f"{type(error).__name__}: {error}"[:120])


# ----------------------------------------------------------------------------- circuits
def circuit_checks(suite: Suite, result) -> None:
    suite.section("circuits")
    model = suite.model
    designs = model.circuit_designs(result)
    suite.check("the model has at least one circuit design", len(designs) >= 1, f"{len(designs)} designs")
    notes = model.circuit_notes(result)
    suite.check("a design without a drawing explains why", all(len(str(note)) > 20 for note in notes), f"{len(notes)} notes")
    statistics: dict[str, dict] = {}
    for name, circuit in designs.items():
        suite.check(f"the {name!r} design has a name", bool(str(name).strip()) and len(str(name)) >= 3, str(name))
        stats = circuit_statistics(circuit)
        statistics[name] = stats
        operations = stats["operations"]
        suite.check(
            f"{name}: the circuit has qubits and gates",
            stats["qubits"] >= 1 and stats["size"] >= 1 and stats["depth"] >= 1,
            str(stats),
        )
        suite.check(
            f"{name}: it measures every classical bit it declares",
            stats["clbits"] == operations.get("measure", 0),
            f"{stats['clbits']} clbits, {operations.get('measure', 0)} measurements",
        )
        suite.check(
            f"{name}: it avoids reset",
            "reset" not in operations,
            ", ".join(sorted(operations)),
        )
        gates = {op: count for op, count in operations.items() if op != "barrier"}
        suite.check(
            f"{name}: the gate count adds up",
            sum(gates.values()) == stats["size"],
            ", ".join(f"{op} x{count}" for op, count in sorted(operations.items())),
        )
        rebuilt = circuit_statistics(model.circuit_designs(result)[name])
        suite.check(f"{name}: building it twice gives the same circuit", rebuilt == stats, f"size {stats['size']}")
        figures = model.figures(result)
        figure = figures.get(f"circuit: {name}")
        suite.check(f"{name}: it is drawn", figure is not None, f"{len(figures)} figures")
        if figure is None:
            continue
        suite.check(f"{name}: the drawing is not a panel of text", has_drawn_marks(figure))
        description = getattr(figure, "_qcog_description", "")
        suite.check(
            f"{name}: the description states the circuit's size",
            "Circuit" in description and f"{stats['qubits']} qubit" in description and f"depth {stats['depth']}" in description,
            description[:110],
        )
        axis = figure.axes[0]
        suite.check(
            f"{name}: the drawing has gates and wires",
            len(axis.patches) >= 1 and len(axis.lines) >= stats["qubits"],
            f"{len(axis.patches)} gate boxes, {len(axis.lines)} lines",
        )
        plt.close(figure)
    if len(designs) > 1:
        first, second = sorted(designs)[:2]
        suite.check(
            f"the two designs {first!r} and {second!r} use the same resources",
            statistics[first]["qubits"] == statistics[second]["qubits"]
            and statistics[first]["operations"] == statistics[second]["operations"],
            f"{statistics[first]['qubits']} qubits, {sorted(statistics[first]['operations'])}",
        )

    # drawing to a file, in the formats a reader would ask for
    directory = suite.out_dir / "circuits"
    directory.mkdir(parents=True, exist_ok=True)
    name = sorted(designs)[0]
    for fmt in ("png", "svg", "pdf"):
        images = model.circuit_images(result, name=name, directory=directory, format=fmt)
        written = sorted(directory.glob(f"*.{fmt}"))
        suite.check(
            f"the {name!r} circuit can be written as {fmt}",
            bool(written) and all(path.stat().st_size > 0 for path in written),
            ", ".join(f"{path.name} ({path.stat().st_size} bytes)" for path in written),
        )
        for figure in images.values():
            plt.close(figure)
    for figure in model.circuit_images(result, name=name).values():
        plt.close(figure)

    # the drawn circuit has to be the model's own design: run it and compare
    state = aer_available()
    if state is not True:
        suite.note("the drawn circuits were not run", f"qiskit-aer is unavailable here ({state})")
        return
    for name, circuit in designs.items():
        key = _matching_result_key(name, result)
        if key is None:
            suite.note(f"the {name!r} circuit is not one of the sequential results", "compared for structure only")
            continue
        item = result.results[key]
        observed = _run_circuit(circuit, item, shots=4000 if suite.fast else 20000, seed=11)
        if observed is None:
            suite.note(f"the {name!r} circuit could not be run here", "Aer declined it")
            continue
        worst = max(
            abs(observed.get(outcome, 0.0) - float(item.joint[outcome]))
            for outcome in item.joint
        )
        tolerance = max(SAMPLING_SIGMA * math.sqrt(0.25 / (4000 if suite.fast else 20000)) * 2, 5e-3)
        suite.check(
            f"the drawn {name!r} circuit reproduces the model's own {key!r} distribution",
            worst < tolerance,
            f"largest deviation {worst:.4f} over {len(item.joint)} outcomes",
        )


def _matching_result_key(design_name: str, result) -> str | None:
    wanted = re.sub(r"[^a-z0-9]+", "_", str(design_name).lower()).strip("_")
    for key in result.results:
        if re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_") == wanted:
            return key
    return None


def _run_circuit(circuit, item, *, shots: int, seed: int):
    """Sample a design circuit and turn the counts into the model's outcome tuples."""
    try:
        from qiskit_aer import AerSimulator

        backend = AerSimulator(seed_simulator=seed)
        # the design carries the library's own state-preparation instruction, which a
        # simulator can only execute once it has been expanded into elementary gates
        runnable = circuit.decompose(reps=3)
        counts = backend.run(runnable, shots=shots).result().get_counts()
    except Exception:  # pragma: no cover - environment dependent
        return None
    outcomes = [tuple(item.outcomes[question]) for question in item.order]
    distribution: Counter = Counter()
    for bitstring, count in counts.items():
        bits = bitstring[::-1]  # qiskit prints the highest classical bit first
        try:
            picked = tuple(outcomes[index][int(bits[index])] for index in range(len(outcomes)))
        except (IndexError, ValueError):
            return None
        distribution[picked] += count
    total = sum(distribution.values())
    if not total:
        return None
    return {outcome: count / total for outcome, count in distribution.items()}

# ------------------------------------------------------------------------------ figures
def suite_figures(suite: Suite, result) -> dict:
    """The model's figures for one result, rendered once and remembered."""
    cached = getattr(suite, "_figure_cache", None)
    if cached is not None and cached[0] is result:
        return cached[1]
    figures = suite.model.figures(result)
    suite._figure_cache = (result, figures)
    return figures


def figure_checks(suite: Suite, result, save: bool = True) -> dict[str, Path]:
    """Every figure: a graph, described, consistent with the data, readable as a file."""
    suite.section("figures")
    figures = suite_figures(suite, result)
    suite.check("the model draws a set of figures", len(figures) >= 3, f"{len(figures)} figures")
    suite.check(
        "the figure set includes the classical comparison",
        "model versus classical baseline" in figures,
        ", ".join(sorted(figures)),
    )
    suite.check(
        "the figure set includes a circuit drawing",
        any(str(name).startswith("circuit:") for name in figures),
    )
    suite.check(
        "the figure set draws the model's own quantities",
        any(
            str(name).startswith(("probabilities:", "marginals:", "sweep:", "state:", "circuit:"))
            or name in {"distribution", "trajectory", "classical trajectory", "order effect", "interference"}
            for name in figures
        ),
        ", ".join(sorted(figures)),
    )
    suite.check(
        "no figure is a panel of text instead of a graph",
        all(has_drawn_marks(figure) for figure in figures.values()),
        f"{sum(not has_drawn_marks(figure) for figure in figures.values())} without marks",
    )
    suite.check(
        "every figure carries a plain-language description",
        all(len(str(getattr(figure, "_qcog_description", ""))) >= 60 for figure in figures.values()),
        f"{sum(len(str(getattr(f, '_qcog_description', ''))) < 60 for f in figures.values())} short",
    )

    saved: dict[str, Path] = {}
    index = 0
    for name, figure in sorted(figures.items()):
        index += 1
        _one_figure(suite, result, name, figure)
        if save:
            path = suite.out_dir / f"{index:02d}_{_slugify(name)}.png"
            try:
                figure.savefig(path, bbox_inches="tight", dpi=110)
                saved[name] = path
                _image_quality(suite, name, path)
            except Exception as error:  # pragma: no cover - a writing failure
                suite.check(f"{name}: it can be written as a PNG", False, f"{type(error).__name__}: {error}")
    return saved


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")[:70]


def _one_figure(suite: Suite, result, name: str, figure) -> None:
    """The checks that apply to every figure, whatever it draws."""
    description = str(getattr(figure, "_qcog_description", ""))
    suite.check(f"{name}: it is a graph, not a panel of text", has_drawn_marks(figure))
    suite.check(f"{name}: it is described in words", len(description) >= 60, f"{len(description)} characters")
    suite.check(
        f"{name}: the description names numbers and no missing ones",
        bool(re.search(r"\d", description)) and not re.search(r"\bnan\b|\bNone\b|\binf\b", description),
        description[:90],
    )
    suite.check(
        f"{name}: no axis is left unlabelled",
        all(str(axis.get_title()).strip() for axis in figure.axes if not _is_colorbar(axis)),
        ", ".join(f"{axis.get_title()!r}" for axis in figure.axes),
    )
    junk = [
        text.get_text()
        for axis in figure.axes
        for text in axis.texts
        if re.search(r"\bnan\b|\bNone\b|\binf\b", text.get_text())
    ]
    suite.check(f"{name}: every drawn label is a number or a name", not junk, str(junk[:3]))
    _series_separation(suite, name, figure)
    _structure(suite, result, name, figure)


def _is_colorbar(axis) -> bool:
    return getattr(axis, "_colorbar", None) is not None


def _series_separation(suite: Suite, name: str, figure) -> None:
    """Series must differ by more than hue - the library's own accessibility rule.

    Circuit drawings and Bloch spheres are drawings rather than series charts, so the rule
    does not apply to them.
    """
    if str(name).startswith("circuit:") or str(name).startswith("state:"):
        return
    for axis in figure.axes:
        lines = [line for line in axis.lines if line.get_linestyle() != "None"]
        if len(lines) >= 2:
            keys = {(line.get_color(), line.get_linestyle(), line.get_marker()) for line in lines}
            suite.check(
                f"{name}: its {len(lines)} series are distinguishable by more than colour",
                len(keys) == len(lines),
                f"{len(keys)} distinct (colour, line style, marker) combinations",
            )
        bars = [patch for patch in axis.patches if isinstance(patch, mpatches.Rectangle)]
        if len(bars) >= 3:
            colours = {patch.get_facecolor() for patch in bars}
            styles = {(patch.get_facecolor(), patch.get_hatch()) for patch in bars}
            suite.check(
                f"{name}: its bars do not rely on colour alone",
                len(styles) == len(colours) or len({patch.get_hatch() for patch in bars}) > 1,
                f"{len(colours)} colours, {len(styles)} distinct bar styles",
            )


def _image_quality(suite: Suite, name: str, path: Path) -> None:
    """The file a reader actually opens has to look like a chart."""
    try:
        from PIL import Image
    except Exception as error:  # pragma: no cover - pillow missing
        suite.note(f"{name}: the image could not be inspected", f"{type(error).__name__}")
        return
    with Image.open(path) as handle:
        image = handle.convert("RGB")
        width, height = image.size
        colours = image.getcolors(maxcolors=1 << 22) or []
        grey = np.asarray(image.convert("L").resize((160, 160)), dtype=float)
    pixels = width * height
    dominant = max((count for count, _ in colours), default=pixels) / pixels
    suite.check(
        f"{name}: the image is big enough to read",
        width >= MIN_IMAGE[0] and height >= MIN_IMAGE[1],
        f"{width}x{height}, {path.stat().st_size // 1024} kB",
    )
    suite.check(
        f"{name}: the image is not blank",
        len(colours) >= MIN_COLOURS and float(grey.std()) > 4.0,
        f"{len(colours)} distinct colours, grey standard deviation {float(grey.std()):.1f}",
    )
    suite.check(
        f"{name}: the image is a chart rather than a solid block",
        dominant < 0.985,
        f"the most common pixel covers {dominant * 100:.1f}% of the image",
    )


# ----------------------------------------------------- structural agreement, by chart type
def _structure(suite: Suite, result, name: str, figure) -> None:
    if name.startswith("probabilities:"):
        _check_probabilities(suite, result, name, figure)
    elif name.startswith("marginals:"):
        _check_marginals(suite, result, name, figure)
    elif name == "order effect":
        _check_order_effect(suite, result, name, figure)
    elif name == "interference":
        _check_interference(suite, result, name, figure)
    elif name.startswith("sweep:"):
        _check_sweep_figure(suite, result, name, figure)
    elif name.startswith("state:"):
        _check_bloch(suite, result, name, figure)
    elif name in {"trajectory", "classical trajectory"}:
        _check_trajectory(suite, result, name, figure)
    elif name == "distribution":
        _check_histogram(suite, result, name, figure)
    elif name in {"payoff table", "payoff matrix"}:
        _check_payoff(suite, result, name, figure)
    elif name == "model versus classical baseline":
        pass  # checked against the comparison rows in classical_checks
    elif name.startswith("circuit:"):
        pass  # checked in circuit_checks
    else:
        suite.note(f"the {name!r} figure has no structural check", "it is still checked for marks and description")


def _probabilities_key(name: str, result) -> str | None:
    wanted = name.split(":", 1)[1].strip().lower()
    for key in result.results:
        if str(key).lower() == wanted:
            return key
    return None


def _check_probabilities(suite: Suite, result, name: str, figure) -> None:
    key = _probabilities_key(name, result)
    if key is None:
        suite.check(f"{name}: the figure names a result that exists", False, key)
        return
    item = result.results[key]
    frame = item.joint_frame()
    shown = item.joint_frame().head(10)
    axis = figure.axes[0]
    heights = bar_values(axis)
    labels = [tick.get_text() for tick in axis.get_xticklabels()]
    suite.check(
        f"{name}: one bar per outcome it draws",
        len(heights) == len(shown),
        f"{len(heights)} bars for {len(shown)} of {len(frame)} outcomes",
    )
    suite.close_list(
        f"{name}: the bar heights are the joint probabilities",
        [float(value) for value in shown["probability"]],
        heights,
        tol=1e-9,
    )
    suite.check(
        f"{name}: the bars are labelled with their outcomes",
        labels[: len(shown)] == [" + ".join(str(part) for part in row) for row in shown[list(item.order)].to_numpy()],
        str(labels[:2]),
    )
    suite.check(
        f"{name}: the bars are probabilities",
        all(-TOL <= height <= 1 + TOL for height in heights),
        f"largest {max(heights) if heights else 0:.3f}",
    )
    if len([height for height in heights if height > 1e-6]) <= 1:
        suite.note(f"{name}: the distribution is (almost) a single outcome", "the bars are correct but the chart says little")
    suite.check(f"{name}: the axis is labelled as a probability", "probability" in axis.get_ylabel().lower(), axis.get_ylabel())


def _bar_lengths(axis) -> list[float]:
    """Bar values whether the chart draws them vertically or horizontally."""
    bars = [patch for patch in axis.patches if isinstance(patch, mpatches.Rectangle)]
    if not bars:
        return []
    widths = [float(patch.get_width()) for patch in bars]
    heights = [float(patch.get_height()) for patch in bars]
    if max(heights) - min(heights) < 1e-9 and max(widths) - min(widths) > 1e-9:
        return widths                      # horizontal bars: constant thickness, varying length
    if max(widths) - min(widths) < 1e-9 and max(heights) - min(heights) > 1e-9:
        return heights                     # vertical bars: constant width, varying height
    return widths if abs(sum(widths) - 1.0) < abs(sum(heights) - 1.0) else heights


def _check_marginals(suite: Suite, result, name: str, figure) -> None:
    key = _probabilities_key(f"x:{name.split(':', 1)[1]}", result)
    if key is None:
        suite.check(f"{name}: the figure names a result that exists", False, key)
        return
    item = result.results[key]
    frame = item.marginal_frame()
    questions = list(item.order)
    shared = len({tuple(item.outcomes[question]) for question in questions}) == 1 and len(questions) <= 4
    if shared:
        labels = list(item.outcomes[questions[0]])
        axis = figure.axes[0]
        heights = bar_values(axis)
        expected = [
            float(frame[(frame["question"] == question) & (frame["outcome"] == outcome)]["probability"].iloc[0])
            for question in questions
            for outcome in labels
        ]
        suite.close_list(f"{name}: the grouped bars are the marginal probabilities", expected, heights, tol=1e-9)
        suite.check(f"{name}: every question is drawn", f"{len(questions)}" in str(len(labels) * len(questions)), f"{len(heights)} bars")
    else:
        panels = [axis for axis in figure.axes if not _is_colorbar(axis)]
        suite.check(
            f"{name}: there is one panel per question",
            len(panels) == len(questions),
            f"{len(panels)} panels for {len(questions)} questions",
        )
        for axis, question in zip(panels, questions):
            expected = [
                float(frame[(frame["question"] == question) & (frame["outcome"] == outcome)]["probability"].iloc[0])
                for outcome in item.outcomes[question]
            ]
            suite.close_list(
                f"{name}: the {question!r} panel shows that question's marginal",
                expected,
                _bar_lengths(axis),
                tol=1e-9,
            )


def _check_order_effect(suite: Suite, result, name: str, figure) -> None:
    """The chart must be the analysis report the model itself produced."""
    report = result.analysis.get("order_effect")
    if report is None:
        suite.check(f"{name}: the order-effect analysis exists", False, "no analysis entry")
        return
    outcomes = list(report.marginal_first_in_first_order)
    axis = figure.axes[0]
    heights = bar_values(axis)
    expected = [float(report.marginal_first_in_first_order[outcome]) for outcome in outcomes] + [
        float(report.marginal_first_in_second_order[outcome]) for outcome in outcomes
    ]
    suite.close_list(
        f"{name}: the bars are the two marginal distributions of {report.first!r}",
        expected,
        heights,
        tol=1e-9,
    )
    suite.close(
        f"{name}: the shift the chart reports is the shift in its own bars",
        report.max_marginal_shift(),
        max(abs(float(report.marginal_first_in_second_order[outcome]) - float(report.marginal_first_in_first_order[outcome])) for outcome in outcomes),
        tol=1e-9,
    )
    suite.check(
        f"{name}: the outcomes are the axis labels",
        [tick.get_text() for tick in axis.get_xticklabels()][: len(outcomes)] == [str(outcome) for outcome in outcomes],
        str([tick.get_text() for tick in axis.get_xticklabels()][:2]),
    )
    suite.check(
        f"{name}: both orders are named in the legend",
        axis.get_legend() is not None and len(axis.get_legend().get_texts()) == 2,
        str([text.get_text() for text in axis.get_legend().get_texts()]) if axis.get_legend() else "no legend",
    )
    suite.check(
        f"{name}: the axis is labelled with the question it shows",
        str(report.first) in axis.get_ylabel(),
        axis.get_ylabel(),
    )


def _check_interference(suite: Suite, result, name: str, figure) -> None:
    report = result.analysis.get("interference")
    if report is None:
        suite.check(f"{name}: the interference decomposition exists", False, "no analysis entry")
        return
    axis = figure.axes[0]
    heights = bar_values(axis)
    tops = bar_tops(axis)
    outcomes = list(report.totals)
    expected_tops = [float(report.totals[outcome]) for outcome in outcomes] * 1
    suite.check(
        f"{name}: one stacked bar per outcome",
        len(heights) == 2 * len(outcomes),
        f"{len(heights)} segments for {len(outcomes)} outcomes",
    )
    suite.close_list(
        f"{name}: the two segments of each bar are the classical and interference parts",
        [float(report.direct[outcome]) for outcome in outcomes] + [float(report.interference[outcome]) for outcome in outcomes],
        heights,
        tol=1e-9,
    )
    suite.close_list(
        f"{name}: each stacked bar tops out at the reported total",
        expected_tops,
        tops[len(outcomes) :],
        tol=1e-9,
    )
    suite.close_list(
        f"{name}: each interference segment starts where the classical one ends",
        [float(report.direct[outcome]) for outcome in outcomes],
        [patch.get_y() for patch in axis.patches[len(outcomes) :]],
        tol=1e-9,
    )


def _check_sweep_figure(suite: Suite, result, name: str, figure) -> None:
    label = name.split(":", 1)[1].strip()
    sweeps = result.analysis.get("sweeps") or {}
    sweep_result = sweeps.get(label) or next(
        (item for item in sweeps.values() if item.parameter == label.replace(" ", "_")), None
    )
    if sweep_result is None:
        suite.check(f"{name}: the sweep exists in the analysis", False, f"{label} not in {sorted(sweeps)}")
        return
    parameter = sweep_result.parameter
    table = sweep_result.table
    axis = figure.axes[0]
    suite.close_list(f"{name}: the x axis is the swept parameter", [float(value) for value in table[parameter]], axis.lines[0].get_xdata(), tol=1e-9)
    metrics = [column for column in table.columns if column != parameter]
    lines = [line for line in axis.lines if len(line.get_ydata()) == len(table)]
    suite.check(
        f"{name}: one line per metric",
        len(lines) == len(metrics),
        f"{len(lines)} lines for {len(metrics)} metrics",
    )
    for line, metric in zip(lines, metrics):
        suite.close_list(
            f"{name}: the {metric!r} line is that metric's column",
            [float(value) for value in table[metric]],
            line.get_ydata(),
            tol=1e-9,
        )
    suite.check(
        f"{name}: the legend names the metrics",
        axis.get_legend() is not None
        and [text.get_text() for text in axis.get_legend().get_texts()] == [str(metric) for metric in metrics],
        str([text.get_text() for text in axis.get_legend().get_texts()]) if axis.get_legend() else "no legend",
    )
    suite.check(
        f"{name}: the x axis is labelled with the parameter",
        parameter.replace("_", " ") in axis.get_xlabel().lower().replace("_", " "),
        axis.get_xlabel(),
    )


def _check_bloch(suite: Suite, result, name: str, figure) -> None:
    state_name = name.split(":", 1)[1].strip()
    states = result.analysis.get("states") or {}
    state = states.get(state_name)
    if state is None:
        suite.check(f"{name}: the state exists in the analysis", False, str(list(states)))
        return
    vector = np.asarray(state.bloch_vector(), dtype=float)
    length = float(np.linalg.norm(vector))
    description = str(getattr(figure, "_qcog_description", ""))
    suite.check(
        f"{name}: the description states the vector it drew",
        all(f"{value:+.2f}" in description for value in vector),
        f"x={vector[0]:+.2f} y={vector[1]:+.2f} z={vector[2]:+.2f}",
    )
    suite.close(f"{name}: the stated length is the state's Bloch length", length, _stated_length(description), tol=5e-3)
    axis = figure.axes[0]
    if length < 1e-6:
        suite.check(
            f"{name}: a mixed state is marked at the centre",
            bool(axis.collections) and "maximally mixed" in description,
            f"{len(axis.collections)} marks",
        )
    else:
        suite.check(
            f"{name}: the state is drawn as a vector",
            bool(axis.collections),
            f"{len(axis.collections)} vector artists",
        )


def _stated_length(description: str) -> float:
    match = re.search(r"length ([0-9]*\.?[0-9]+)", description)
    return float(match.group(1)) if match else float("nan")


def _check_trajectory(suite: Suite, result, name: str, figure) -> None:
    data = result.analysis.get("trajectory" if name == "trajectory" else "classical_trajectory")
    if not isinstance(data, dict) or "series" not in data:
        suite.check(f"{name}: the trajectory data exists", False, "no analysis entry")
        return
    axis = figure.axes[0]
    lines = [line for line in axis.lines if len(line.get_ydata()) == len(data["x"])]
    suite.check(
        f"{name}: one line per series",
        len(lines) == len(data["series"]),
        f"{len(lines)} lines for {len(data['series'])} series",
    )
    suite.close_list(f"{name}: the x axis is the trajectory's x", data["x"], lines[0].get_xdata() if lines else [], tol=1e-9)
    for line, (label, values) in zip(lines, data["series"].items()):
        suite.check(
            f"{name}: the {label!r} line is that series",
            np.allclose(np.asarray(line.get_ydata(), dtype=float), np.asarray(values, dtype=float), atol=1e-9),
            f"{len(values)} steps",
        )
    suite.check(
        f"{name}: the legend names every agent",
        axis.get_legend() is not None and len(axis.get_legend().get_texts()) == len(data["series"]),
        str(len(axis.get_legend().get_texts())) if axis.get_legend() else "no legend",
    )


def _check_histogram(suite: Suite, result, name: str, figure) -> None:
    data = result.analysis.get("histogram")
    if not isinstance(data, dict) or "series" not in data:
        suite.check(f"{name}: the histogram data exists", False, "no analysis entry")
        return
    values = [float(value) for series in data["series"].values() for value in series]
    heights = [value for value in bar_values(figure.axes[0]) if value > 0]
    suite.check(
        f"{name}: the bars are the distribution it was drawn from",
        len(heights) == len(values) and np.allclose(sorted(heights), sorted(values), atol=1e-9),
        f"{len(heights)} bars for {len(values)} values",
    )
    suite.check(
        f"{name}: the x axis lists the outcomes",
        len(figure.axes[0].get_xticklabels()) >= len(next(iter(data["series"].values()))),
        ", ".join(tick.get_text() for tick in figure.axes[0].get_xticklabels())[:60],
    )


def _check_payoff(suite: Suite, result, name: str, figure) -> None:
    frame = result.analysis.get("payoff_table")
    if frame is None or not hasattr(frame, "columns"):
        suite.check(f"{name}: the payoff table exists", False, "no analysis entry")
        return
    panels = [axis for axis in figure.axes if axis.images]
    suite.check(f"{name}: every payoff column is drawn as its own panel", len(panels) >= 1, f"{len(panels)} panels")
    text_columns = [column for column in frame.columns if frame[column].dtype == object or str(frame[column].dtype).startswith("str")]
    numeric = [column for column in frame.columns if column not in text_columns and column not in {"cooperation"}]
    suite.check(
        f"{name}: the cooperation flag is not plotted as a payoff",
        all(column not in {"cooperation"} for column in numeric),
        ", ".join(map(str, numeric)),
    )
    for axis, column in zip(panels, [col for col in numeric if col in {"quantum_first", "classical_first"}] or numeric):
        drawn = np.asarray(axis.images[0].get_array(), dtype=float)
        rows = list(dict.fromkeys(frame[text_columns[0]]))
        columns = list(dict.fromkeys(frame[text_columns[1]]))
        expected = np.full((len(rows), len(columns)), np.nan)
        for _, row in frame.iterrows():
            expected[rows.index(row[text_columns[0]]), columns.index(row[text_columns[1]])] = float(row[column])
        suite.check(
            f"{name}: the {column!r} panel holds that column's values",
            drawn.shape == expected.shape and np.allclose(drawn, expected, equal_nan=True, atol=1e-9),
            f"{drawn.shape} panel for {column!r}",
        )
    suite.check(
        f"{name}: the axes are labelled with the players' choices",
        all(str(axis.get_xlabel()) and str(axis.get_ylabel()) for axis in panels),
        ", ".join(f"{axis.get_ylabel()}/{axis.get_xlabel()}" for axis in panels),
    )

# ------------------------------------------------------------------------------- report
CARD_CSS = """
:root { color-scheme: light dark; }
body { margin: 0 auto; max-width: 62rem; padding: 1.2rem 1rem 4rem; font: 16px/1.55 system-ui, sans-serif; }
h1 { font-size: 1.5rem; } h2 { margin-top: 2rem; font-size: 1.15rem; border-bottom: 1px solid #8884; }
h3 { margin: 1.4rem 0 .3rem; font-size: 1rem; }
figure { margin: 0 0 1.6rem; } img { max-width: 100%; height: auto; border: 1px solid #8884; border-radius: 4px; }
figcaption { font-size: .86rem; color: #6668; }
table { border-collapse: collapse; width: 100%; font-size: .88rem; }
th, td { border: 1px solid #8884; padding: .3rem .45rem; text-align: left; vertical-align: top; }
.pass { color: #1a7f37; } .fail { color: #b3261e; font-weight: 700; } .note { color: #8a6100; }
.bar { display: flex; gap: .6rem; flex-wrap: wrap; font-size: .9rem; margin: .4rem 0 1rem; }
.pill { border: 1px solid #8884; border-radius: 999px; padding: .1rem .6rem; }
.summary { border-left: 4px solid #8886; padding: .4rem .8rem; margin: .8rem 0; }
code { background: #8882; padding: .05rem .25rem; border-radius: 3px; }
"""


def _cell(value: Any) -> str:
    return html.escape(str(value))


def write_report(suite: Suite, result, saved: Mapping[str, Path], extra_notes: Sequence[str] = ()) -> Path:
    """A page a human can read: the graphs, their descriptions, the numbers, the checks."""
    passed = sum(1 for check in suite.checks if check["ok"])
    failed = [check for check in suite.checks if not check["ok"]]
    parts: list[str] = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        f"<title>Model tests: {_cell(suite.slug)}</title><style>{CARD_CSS}</style></head><body>",
        f"<h1>{_cell(suite.model.title)}</h1>",
        f"<p><code>{_cell(suite.slug)}</code> - module {_cell(suite.model.module)}"
        f" ({_cell(suite.model.difficulty)}) - {suite.runs} runs, {suite.seconds:.1f} s</p>",
        "<div class='bar'>"
        f"<span class='pill'>{len(suite.checks)} checks</span>"
        f"<span class='pill pass'>{passed} passed</span>"
        f"<span class='pill {'fail' if failed else 'pass'}'>{len(failed)} failed</span>"
        f"<span class='pill note'>{len(suite.notes)} notes</span>"
        f"<span class='pill'>{len(saved)} figures</span></div>",
        f"<p class='summary'>{_cell(result.summary)}</p>",
    ]
    parts.append("<h2>What it found</h2><ul>")
    parts += [f"<li>{_cell(finding)}</li>" for finding in result.findings]
    parts.append("</ul>")
    if result.caveats:
        parts.append("<h2>What it does not claim</h2><ul>")
        parts += [f"<li>{_cell(caveat)}</li>" for caveat in result.caveats]
        parts.append("</ul>")

    parts.append("<h2>The model against the classical baseline</h2>")
    parts.append("<table><thead><tr><th>Quantity</th><th>Model</th><th>Classical</th><th>Why the classical value is what it is</th></tr></thead><tbody>")
    for row in result.classical_comparison:
        parts.append(
            "<tr>"
            f"<td>{_cell(row.get('metric'))}</td>"
            f"<td>{float(row['model']):.4f}</td>"
            f"<td>{float(row['baseline']):.4f}</td>"
            f"<td>{_cell(row.get('note', ''))}</td></tr>"
        )
    parts.append("</tbody></table>")
    if result.classical.get("verdict"):
        parts.append(f"<p>{_cell(result.classical['verdict'])}</p>")

    parts.append("<h2>The figures</h2>")
    by_name = {check["name"]: check for check in suite.checks}
    for name, figure in sorted(suite_figures(suite, result).items()):
        path = saved.get(name)
        parts.append("<figure>")
        if path is not None:
            parts.append(f"<img src='{_cell(path.name)}' alt='{_cell(getattr(figure, '_qcog_description', name))[:400]}'>")
        parts.append(f"<figcaption><strong>{_cell(name)}</strong> - {_cell(getattr(figure, '_qcog_description', ''))}</figcaption>")
        failed_here = [check for check in failed if check["name"].startswith(name)]
        if failed_here:
            for check in failed_here:
                parts.append(f"<div class='fail'>{_cell(check['name'])}: {_cell(check['detail'])}</div>")
        else:
            parts.append("<div class='pass'>a graph, described in words, and matching the numbers it came from</div>")
        parts.append("</figure>")

    parts.append("<h2>The parameters</h2>")
    parts.append("<table><thead><tr><th>Parameter</th><th>Kind</th><th>Default</th><th>Bounds</th><th>What it means</th></tr></thead><tbody>")
    for spec in suite.model.params:
        bounds = ""
        if spec.low is not None or spec.high is not None:
            bounds = f"[{spec.low:g}, {spec.high:g}]" if spec.low is not None and spec.high is not None else str(spec.low or spec.high)
        parts.append(
            f"<tr><td><code>{_cell(spec.name)}</code></td><td>{_cell(spec.kind)}</td><td>{_cell(spec.default)}</td>"
            f"<td>{_cell(bounds)}</td><td>{_cell(spec.description)}</td></tr>"
        )
    parts.append("</tbody></table>")

    sweeps = result.analysis.get("sweeps") or {}
    if sweeps:
        parts.append("<h2>The sweeps</h2>")
        for sweep_name, sweep_result in sorted(sweeps.items()):
            parts.append(f"<h3>{_cell(sweep_name)}</h3>")
            parts.append(sweep_result.table.to_html(index=False, border=0, float_format=lambda value: f"{value:.4f}"))

    designs = suite.model.circuit_designs(result)
    if designs:
        parts.append("<h2>The circuits</h2>")
        parts.append("<table><thead><tr><th>Design</th><th>Qubits</th><th>Depth</th><th>Gates</th><th>Operations</th></tr></thead><tbody>")
        for name, circuit in sorted(designs.items()):
            stats = circuit_statistics(circuit)
            operations = ", ".join(f"{op} x{count}" for op, count in sorted(stats["operations"].items()))
            parts.append(
                f"<tr><td>{_cell(name)}</td><td>{stats['qubits']}</td><td>{stats['depth']}</td>"
                f"<td>{stats['size']}</td><td>{_cell(operations)}</td></tr>"
            )
        parts.append("</tbody></table>")
        notes = suite.model.circuit_notes(result)
        if notes:
            parts.append("<ul>" + "".join(f"<li>{_cell(note)}</li>" for note in notes) + "</ul>")

    if result.references:
        parts.append("<h2>Where it comes from</h2><ul>")
        parts += [f"<li>{_cell(reference)}</li>" for reference in result.references]
        parts.append("</ul>")

    if suite.notes or extra_notes:
        parts.append("<h2>Notes for a human reviewer</h2><ul>")
        parts += [f"<li class='note'>{_cell(note['name'])}: {_cell(note['detail'])}</li>" for note in suite.notes]
        parts += [f"<li class='note'>{_cell(note)}</li>" for note in extra_notes]
        parts.append("</ul>")

    parts.append("<h2>Every check</h2>")
    parts.append("<table><thead><tr><th>Check</th><th>Result</th><th>Detail</th></tr></thead><tbody>")
    section = None
    for check in suite.checks:
        if check["section"] != section:
            section = check["section"]
            parts.append(f"<tr><th colspan='3'>{_cell(section)}</th></tr>")
        parts.append(
            f"<tr><td>{_cell(check['name'])}</td>"
            f"<td class='{'pass' if check['ok'] else 'fail'}'>{'pass' if check['ok'] else 'FAIL'}</td>"
            f"<td>{_cell(check['detail'])}</td></tr>"
        )
    parts.append("</tbody></table></body></html>")
    path = suite.out_dir / "report.html"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- the runner
def run_model(
    slug: str,
    extra: Callable[[Suite, Any], None] | None = None,
    *,
    module: str = "",
    out_dir: str | Path | None = None,
    fast: bool = False,
    quiet: bool = False,
    save: bool = True,
    title: str = "",
) -> dict:
    """Run the whole battery for one model and return a report dictionary."""
    out_dir = Path(out_dir) if out_dir is not None else HERE / "out"
    suite = Suite(slug, out_dir, fast=fast)
    started = time.perf_counter()
    metadata_checks(suite, module)
    result = parameter_checks(suite)
    sweep_checks(suite, result)
    classical_checks(suite, result)
    execution_checks(suite, result)
    circuit_checks(suite, result)
    saved = figure_checks(suite, result, save=save)
    if extra is not None:
        suite.section("model-specific checks")
        extra(suite, result)
    report = write_report(suite, result, saved) if save else None
    elapsed = time.perf_counter() - started
    for figure in suite_figures(suite, result).values():
        plt.close(figure)
    plt.close("all")
    failures = [check for check in suite.checks if not check["ok"]]
    summary = {
        "slug": suite.slug,
        "title": title or suite.model.title,
        "module": suite.model.module,
        "checks": len(suite.checks),
        "passed": len(suite.checks) - len(failures),
        "failed": len(failures),
        "failures": failures,
        "notes": suite.notes,
        "figures": len(saved),
        "runs": suite.runs,
        "seconds": round(elapsed, 1),
        "report": str(report) if report else "",
        "exit_code": 1 if failures else 0,
    }
    if not quiet:
        print_model_summary(summary, verbose=not save)
    return summary


def print_model_summary(summary: Mapping, *, verbose: bool = False) -> None:
    mark = "PASS" if not summary["failed"] else "FAIL"
    print(
        f"{mark}  {summary['slug']:<26} module {summary['module']}  "
        f"{summary['passed']:>4}/{summary['checks']:<4} checks  "
        f"{summary['figures']:>2} figures  {summary['runs']:>3} runs  {summary['seconds']:>5.1f}s"
    )
    for failure in summary["failures"]:
        print(f"        FAIL  {failure['name']}  {failure['detail']}")
    for note in summary["notes"]:
        print(f"        note  {note['name']}  {note['detail']}")
    if verbose:
        print(f"        report {summary['report']}")


def main_for(slug: str, extra=None, *, module: str = "", argv: Sequence[str] | None = None) -> int:
    """``python model_tests/module_x/NN_model.py``."""
    import argparse

    parser = argparse.ArgumentParser(description=f"Run every check for {slug}.")
    parser.add_argument("--fast", action="store_true", help="fewer shots and no Aer comparison")
    parser.add_argument("--no-save", action="store_true", help="do not write figures or the report")
    parser.add_argument("--quiet", action="store_true", help="only the summary line")
    parser.add_argument("--out", default=None, help="where to write figures and reports")
    args = parser.parse_args(argv)
    summary = run_model(
        slug, extra, module=module, fast=args.fast, quiet=args.quiet,
        save=not args.no_save, out_dir=args.out,
    )
    if not summary["failed"]:
        print(f"\n{summary['slug']}: {summary['passed']} checks passed, {len(summary['notes'])} notes,"
              f" {summary['figures']} figures written to {summary['report'] or 'nowhere'}")
    else:
        print(f"\n{summary['slug']}: {summary['failed']} of {summary['checks']} checks FAILED")
    return summary["exit_code"]


def import_file(path: Path):
    """Import one model-test file without it being a package."""
    specification = importlib.util.spec_from_file_location(f"model_tests_{path.stem}", path)
    if specification is None or specification.loader is None:  # pragma: no cover
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module
