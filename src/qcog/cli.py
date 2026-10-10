"""Command line interface: ``qcog <command>``.

Commands
--------
``qcog list``              every model, grouped by module
``qcog info SLUG``         one model's documentation
``qcog run SLUG``          run a model, with ``--param value`` overrides
``qcog circuit SLUG``      draw the model's quantum circuit (text, plus PNGs on request)
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

    circuit = sub.add_parser("circuit", help="draw the quantum circuit of a model")
    circuit.add_argument("slug")
    circuit.add_argument("--param", nargs="*", default=[])
    circuit.add_argument("--out", default=None, help="directory to write PNG files into")

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
            overrides: list[str], save: str | None, fast: bool = False) -> int:
    """Run a model and print or save its results."""
    model = MODELS[slug]()
    values = _parse_overrides(overrides)
    result = model.run(backend=backend, shots=shots, seed=seed, sweeps=not fast, **values)
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


def cmd_circuit(slug: str, overrides: list[str], out: str | None) -> int:
    """Describe (and optionally draw) the circuits of a model."""
    from .circuits.builder import circuit_statistics, circuit_text_alternative

    model = MODELS[slug]()
    values = _parse_overrides(overrides)
    result = model.run(sweeps=False, **values)
    designs = model.circuit_designs(result)
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
        from .viz import circuit_figure, savefig

        directory = Path(out)
        directory.mkdir(parents=True, exist_ok=True)
        for name, design in designs.items():
            safe = "".join(character if character.isalnum() else "_" for character in name)[:40]
            path = directory / f"{slug}_{safe}.png"
            savefig(circuit_figure(design, title=name), path)
            _print(f"wrote {path}")
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


def cmd_fit(slug: str, data_path: str, overrides: list[str]) -> int:
    """Fit a model to observed frequencies."""
    from .core.fitting import fit

    model = MODELS[slug]()
    path = Path(data_path)
    if not path.exists():
        raise SystemExit(f"data file {data_path} not found")
    if path.suffix.lower() == ".json":
        observed = json.loads(path.read_text(encoding="utf-8"))
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
    if args.command == "list":
        return cmd_list()
    if args.command == "info":
        return cmd_info(args.slug)
    if args.command == "run":
        return cmd_run(
            args.slug, args.backend, args.shots, args.seed, args.format, args.param, args.save,
            getattr(args, "fast", False),
        )
    if args.command == "circuit":
        return cmd_circuit(args.slug, args.param, args.out)
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