"""Command line interface: ``qcog <command>``.

Commands
--------
``qcog list``              every model, grouped by module
``qcog info SLUG``         one model's documentation
``qcog run SLUG``          run a model, with ``--param value`` overrides
``qcog circuit SLUG``      draw the model's quantum circuit (text, plus images on request;
                           --which, --style, --scale, --fold, --format, --out)
``qcog sweep SLUG``        print the model's own sweeps as tables
``qcog fit SLUG DATA``     fit a model to observed frequencies (CSV or JSON)
``qcog report``            a full accessible HTML report of every model
``qcog evidence``          run the cross-checks the library relies on
``qcog backends``          the available execution backends

Parameters are passed as ``--name value``, using the names from ``qcog info``.
Execution is configured with ``--backend``, ``--shots`` and ``--seed``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core.params import coerce_params
from .registry import MODELS, catalogue, get_model, list_models, models_by_module

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    """The argument parser (kept public so tests and the docs can introspect it)."""
    parser = argparse.ArgumentParser(
        prog="qcog",
        description="Q-Cog: simulate, explain and visualise quantum-probability models of cognition.",
    )
    parser.add_argument("--version", action="store_true", help="print the Q-Cog version and exit")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="list every model")
    sub.add_parser("backends", help="list the execution backends")

    info = sub.add_parser("info", help="describe one model")
    info.add_argument("slug")

    run = sub.add_parser("run", help="run a model")
    run.add_argument("slug")
    run.add_argument("--backend", default="exact", help="exact, sampled, aer, aer_exact or ionq")
    run.add_argument("--shots", type=int, default=None, help="number of shots for sampled backends")
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--format", default="text", choices=("text", "json"))
    run.add_argument("--param", nargs="*", default=[], help="parameter overrides as name=value")
    run.add_argument("--save", default=None, help="directory for an HTML report of this run")
    run.add_argument("--fast", action="store_true", help="skip the parameter sweeps (headline numbers only)")
    run.add_argument("--no-classical", action="store_true",
                     help="skip the model-versus-classical comparison")
    run.add_argument("--classical-option", action="append", default=[], metavar="NAME=VALUE",
                     help="adjust the classical baseline (rows=..., max_rows=..., accuracy=..., epsilon=..., "
                          "kind=..., signal_accuracy=...); repeatable")
    run.add_argument("--no-circuits", action="store_true", help="do not draw the circuit figures")
    run.add_argument("--circuit-style", default="default", help="circuit drawing style: default, iqp, clifford, bw")
    run.add_argument("--circuit-scale", type=float, default=0.85, help="circuit drawing scale (gate size)")
    run.add_argument("--circuit-fold", type=int, default=28, help="gates per line before the drawing wraps")

    circuit = sub.add_parser("circuit", help="draw the quantum circuit of a model")
    circuit.add_argument("slug")
    circuit.add_argument("--param", nargs="*", default=[])
    circuit.add_argument("--out", default=None, help="directory to write image files into")
    circuit.add_argument("--which", default=None, help="draw only this design (see the listing)")
    circuit.add_argument("--all", action="store_true", help="draw every design (default)")
    circuit.add_argument("--format", default="png", choices=("png", "svg", "pdf"), help="image format")
    circuit.add_argument("--style", default="default", help="drawing style: default, iqp, clifford, bw")
    circuit.add_argument("--scale", type=float, default=0.85, help="gate size")
    circuit.add_argument("--fold", type=int, default=28, help="gates per line before the drawing wraps")

    sweep = sub.add_parser("sweep", help="print the sweeps of a model")
    sweep.add_argument("slug")
    sweep.add_argument("--param", nargs="*", default=[])

    fit = sub.add_parser("fit", help="fit a model to observed frequencies")
    fit.add_argument("slug")
    fit.add_argument("data", help="JSON or CSV file with response keys and observed frequencies")
    fit.add_argument("--param", nargs="*", default=[])

    report = sub.add_parser("report", help="write an accessible HTML report")
    report.add_argument("--out", default="out/qcog_report.html")
    report.add_argument("--max-models", type=int, default=None)
    report.add_argument("--include-figures", action="store_true", default=True)
    report.add_argument("--markdown", action="store_true", help="also write a Markdown summary")

    sub.add_parser("evidence", help="run the library's own cross-checks and print the evidence table")
    return parser


def _parse_overrides(items: list[str]) -> dict:
    """Turn ``["theta=0.9", "mode=ring"]`` into a parameter dictionary."""
    values: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"parameter override {item!r} must look like name=value")
        name, raw = item.split("=", 1)
        text = raw.strip()
        try:
            values[name.strip()] = json.loads(text)
        except json.JSONDecodeError:
            values[name.strip()] = text
    return values


def _print(message: str = "") -> None:
    sys.stdout.write(message + "\n")


def cmd_list() -> int:
    """List every model."""
    _print("Q-Cog model catalogue")
    _print("=" * 60)
    for letter, models in models_by_module().items():
        _print(f"\nModule {letter} - {models[0].module_title() if models else ''}")
        for model in models:
            _print(f"  {model.slug:32s} {model.one_liner}")
    _print(f"\n{len(list_models())} models in total. Use `qcog info SLUG` for the details.")
    return 0


def cmd_info(slug: str) -> int:
    """Describe one model."""
    model = get_model(slug)
    _print(model.describe())
    return 0


def cmd_run(slug: str, backend: str, shots: int | None, seed: int | None, fmt: str,
            overrides: list[str], save: str | None, fast: bool = False,
            classical: bool = True, classical_options: list[str] | None = None,
            circuits: bool = True, circuit_style: str = "default",
            circuit_scale: float = 0.85, circuit_fold: int = 28) -> int:
    """Run a model and print or save its results."""
    model = MODELS[slug]()
    values = _parse_overrides(overrides)
    baseline_options = _parse_overrides(classical_options or [])
    result = model.run(
        backend=backend, shots=shots, seed=seed, sweeps=not fast, classical=classical,
        classical_options=baseline_options, circuits=circuits, circuit_style=circuit_style,
        circuit_scale=circuit_scale, circuit_fold=circuit_fold, **values,
    )
    if fmt == "json":
        _print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        _print(f"{result.title}  [{result.module}]")
        _print("-" * 60)
        _print(f"backend     : {result.backend}" + (f", shots {result.shots}" if result.shots else ""))
        _print(f"parameters  : {', '.join(f'{k}={v}' for k, v in result.parameters.items())}")
        _print(f"summary     : {result.summary}")
        _print("\nFindings")
        for finding in result.findings:
            _print(f"  * {finding}")
        rows = result.classical_comparison
        if rows:
            _print("\nAgainst the classical baseline: " + result.classical["name"])
            _print(f"  {'quantity':52s} {'model':>9s} {'classical':>10s}")
            for row in rows:
                _print(f"  {row['metric'][:52]:52s} {row['model']:9.3f} {row['baseline']:10.3f}")
            _print(f"  {result.classical['verdict']}")
        elif not classical:
            _print("\nThe classical comparison is switched off for this run (--no-classical).")
        designs = model.circuit_designs(result)
        if designs:
            _print("\nQuantum circuit(s): " + ", ".join(designs))
            _print(f"  draw them with `qcog circuit {slug}`")
        for note in model.circuit_notes(result):
            _print(f"  circuit note: {note}")
        _print("\nLimit what this does not say")
        for caveat in result.caveats:
            _print(f"  - {caveat}")
        for name, item in result.sequential_results().items():
            _print(f"\n{name}: {item.summary()}")
            _print(item.marginal_frame().to_string(index=False))
    if save:
        from .report import write_model_report

        path = write_model_report(model, result, save)
        _print(f"\nHTML report written to {path}")
    return 0


def cmd_circuit(slug: str, overrides: list[str], out: str | None, which: str | None = None,
                style: str = "default", scale: float = 0.85, fold: int = 28,
                suffix: str = "png") -> int:
    """Describe (and optionally draw) the circuits of a model."""
    from .circuits.builder import circuit_statistics, circuit_text_alternative

    model = MODELS[slug]()
    values = _parse_overrides(overrides)
    result = model.run(sweeps=False, **values)
    designs = model.circuit_designs(result)
    if which is not None:
        if which not in designs:
            _print(f"{slug} has no circuit named {which!r}. Available: {sorted(designs) or 'none'}")
            return 1
        designs = {which: designs[which]}
    if not designs:
        _print(f"{slug} has no state-vector circuit for these parameters:")
        for note in model.circuit_notes(result):
            _print(f"  - {note}")
        return 0
    for name, design in designs.items():
        statistics = circuit_statistics(design)
        _print(f"=== {name} ===")
        _print("  " + circuit_text_alternative(design))
        _print(f"  qubits {statistics['qubits']}, classical bits {statistics['clbits']}, "
               f"depth {statistics['depth']}, operations {statistics['operations']}")
    for note in model.circuit_notes(result):
        _print(f"note: {note}")
    if out:
        from .viz import VisualizationUnavailable

        try:
            # draw exactly what was listed: --which picks one design, otherwise every design
            written = model.circuit_images(result, name=which, directory=out, style=style, scale=scale,
                                           fold=fold, format=suffix)
        except VisualizationUnavailable as error:
            _print(f"no image written: {error}")
            return 1
        for name in written:
            _print(f"wrote {name} -> {out}")
    return 0


def cmd_sweep(slug: str, overrides: list[str]) -> int:
    """Print the sweep tables of a model."""
    model = MODELS[slug]()
    values = _parse_overrides(overrides)
    result = model.run(**values)
    sweeps = result.analysis.get("sweeps") or {}
    if not sweeps:
        _print(f"{slug} declares no sweeps; run it with `qcog run {slug}` instead.")
        return 0
    for name, sweep_result in sweeps.items():
        if len(sweep_result.table.columns) <= 1 or len(sweep_result.table.index) <= 1:
            continue  # suppressed or empty
        _print(f"\n=== {name} ===")
        _print(sweep_result.table.to_string(index=False))
    return 0


def _is_dataset(value) -> bool:
    """True for a dataset object: a mapping that carries its own frequencies."""
    return isinstance(value, dict) and any(isinstance(value.get(key), dict) for key in ("frequencies", "observed"))


def _frequencies_from_json(payload, slug: str, name: str) -> dict[str, float]:
    """The observed response frequencies in a JSON file, or a clear reason why there are none.

    Three shapes are accepted, because all three are things a person actually has: a flat
    ``{response: frequency}`` mapping, one dataset object with a ``frequencies`` (or ``observed``)
    mapping inside it, and a bundle of such datasets - which is what ``qcog/data/benchmarks.json``
    ships - in which case the dataset whose name matches the model is used, or the only one.
    """
    if not isinstance(payload, dict) or not payload:
        raise SystemExit(f"{name} does not contain a set of response frequencies")
    if _is_dataset(payload):
        dataset, chosen = payload, name
    else:
        datasets = {key: value for key, value in payload.items() if _is_dataset(value)}
        if not datasets:
            dataset, chosen = payload, name
        else:
            wanted = slug.replace("-", "_")
            match = next((key for key in sorted(datasets) if wanted in key.replace("-", "_")), None)
            if match is None and len(datasets) == 1:
                match = next(iter(datasets))
            if match is None:
                raise SystemExit(
                    f"{name} holds several datasets ({', '.join(sorted(datasets))}); pass a file with a "
                    "single set of response frequencies"
                )
            dataset, chosen = datasets[match], f"{name} ({match})"
    frequencies = dataset.get("frequencies") or dataset.get("observed") or dataset
    if not isinstance(frequencies, dict):
        raise SystemExit(f"{chosen} does not contain a set of response frequencies")
    observed: dict[str, float] = {}
    for key, value in frequencies.items():
        try:
            observed[str(key)] = float(value)
        except (TypeError, ValueError):
            raise SystemExit(
                f"{chosen}: {key!r} is not a number, so this is not a table of response frequencies"
            ) from None
    if not observed:
        raise SystemExit(f"{chosen} contains no response frequencies")
    return observed


def cmd_fit(slug: str, data_path: str, overrides: list[str]) -> int:
    """Fit a model to observed frequencies."""
    from .core.fitting import fit

    model = MODELS[slug]()
    path = Path(data_path)
    if not path.exists():
        raise SystemExit(f"data file {data_path} not found")
    if path.suffix.lower() == ".json":
        observed = _frequencies_from_json(json.loads(path.read_text(encoding="utf-8")), slug, path.name)
    else:
        observed = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lower().startswith("response"):
                continue
            key, value = line.split(",")[:2]
            observed[key.strip()] = float(value)
    values = _parse_overrides(overrides)
    defaults = model.defaults()
    defaults.update(values)

    def predict(parameters: dict) -> dict:
        return model.predict({**defaults, **parameters})

    outcome = fit(predict, observed, model.params)
    _print(f"Fitting {slug} to {len(observed)} observed responses")
    _print(outcome.summary())
    _print("")
    _print(outcome.table().to_string(index=False))
    return 0


def cmd_report(out: str, max_models: int | None, include_figures: bool, markdown: bool) -> int:
    """Write the accessible HTML report."""
    from .report import write_catalogue_report

    path = write_catalogue_report(out, max_models=max_models, include_figures=include_figures)
    _print(f"Report written to {path}")
    if markdown:
        from .report import write_markdown_report

        other = write_markdown_report(str(Path(out).with_suffix(".md")), max_models=max_models)
        _print(f"Markdown summary written to {other}")
    return 0


def cmd_evidence() -> int:
    """Run the cross-checks and print what was verified."""
    from .selftest import run_evidence

    for row in run_evidence():
        _print(f"{row['check']:52s} {row['result']:>12s}   {row['detail']}")
    return 0


def cmd_backends() -> int:
    """List the execution backends."""
    from .backends import backend_table

    _print(f"{'name':16s} {'kind':18s} {'shots':6s} notes")
    for row in backend_table():
        _print(f"{row['name']:16s} {row['kind']:18s} {str(row['shots']):6s} {row['notes']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``qcog`` command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "version", False) or args.command is None:
        from . import __version__

        _print(f"Q-Cog {__version__}")
        if args.command is None and not getattr(args, "version", False):
            parser.print_help()
        return 0
    try:
        return _dispatch(parser, args)
    except SystemExit:
        raise
    except (ValueError, KeyError, TypeError, FileNotFoundError, OSError) as error:
        # a typo in a backend name, a parameter or a file path is a message, not a traceback
        print(f"error: {error}", file=sys.stderr)
        return 2


def _dispatch(parser, args) -> int:
    """Run the command the parser selected."""
    if args.command == "list":
        return cmd_list()
    if args.command == "info":
        return cmd_info(args.slug)
    if args.command == "run":
        return cmd_run(
            args.slug, args.backend, args.shots, args.seed, args.format, args.param, args.save,
            getattr(args, "fast", False), not args.no_classical, args.classical_option,
            not args.no_circuits, args.circuit_style, args.circuit_scale, args.circuit_fold,
        )
    if args.command == "circuit":
        return cmd_circuit(args.slug, args.param, args.out, args.which, args.style, args.scale,
                           args.fold, args.format)
    if args.command == "sweep":
        return cmd_sweep(args.slug, args.param)
    if args.command == "fit":
        return cmd_fit(args.slug, args.data, args.param)
    if args.command == "report":
        return cmd_report(args.out, args.max_models, args.include_figures, args.markdown)
    if args.command == "evidence":
        return cmd_evidence()
    if args.command == "backends":
        return cmd_backends()
    parser.error(f"unknown command {args.command!r}")  # pragma: no cover - argparse handles it
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())