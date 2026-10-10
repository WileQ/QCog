#!/usr/bin/env python
"""Run Q-Cog models, print what each one shows, and save its figures and report.

This is the demo showcase: it deliberately lives *outside* the package, because it
is an example of *using* Q-Cog rather than part of it.  Nothing here is imported by the
library, and the script works from a plain checkout with no installation.

    python demos/showcase/run_showcase.py                       # the four models in MODELS
    python demos/showcase/run_showcase.py --models question-order auctions
    python demos/showcase/run_showcase.py --all                  # every model in the catalogue
    python demos/showcase/run_showcase.py --param correlation=0  # extra parameter overrides
    python demos/showcase/run_showcase.py > demos/showcase/output.txt  # capture the output

For each model it prints: the cognitive question, the quantum mechanism and the classical
baseline; every parameter with its bounds (all adjustable from Python); the summary; the
findings; the model-versus-classical comparison; whether a circuit was drawn (and what
was drawn); and the model's own caveats.  It then saves every figure - including the
classical-comparison chart and the circuit drawings - into ``demos/showcase/figures/``.

Options
-------
--models SLUG [SLUG ...]   which models to run (default: the four in MODELS)
--all                      run the whole catalogue
--param NAME=VALUE [...]   parameter overrides applied to every model that accepts them
--out DIR                  where to write figures (default: demos/showcase/figures)
--reports                  also write an accessible HTML report per model
--no-figures               print only, save nothing
--fast                     skip the parameter sweeps inside each model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# The package lives in src/; this makes the script runnable from a checkout.
ROOT = Path(__file__).resolve().parents[2]
if (ROOT / "src").exists() and str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qcog.registry import MODELS as CATALOGUE, get_model  # noqa: E402

#: The models this showcase runs by default: one per module, chosen because between them
#: they exercise every feature - a coherence-versus-ignorance contrast, a three-level
#: question, an order effect in an economic setting, and a group mechanism.  (The whole
#: catalogue is available with ``--all``.)
DEFAULT_MODELS = ["superposition-beliefs", "semantic-ambiguity", "auctions", "collective-judgment"]

#: Hand-picked "now change something" variations, chosen because each one has a clear
#: meaning.  Models without an entry get automatic variations across their numeric
#: parameters, so this script works for any model in the catalogue.
CURATED_VARIATIONS = {
    "superposition-beliefs": [("no decoherence at all", {"decoherence": 0.0})],
    "semantic-ambiguity": [("a context that does nothing", {"context_strength": 0.0})],
    "auctions": [("a bidder who ignores the rivals", {"rival_pressure": 0.0})],
    "collective-judgment": [
        ("judges who are fully independent", {"correlation": 0.0}),
        ("judges who all share one state", {"correlation": 1.0}),
    ],
}

RULE = "=" * 92
THIN = "-" * 92


def auto_variations(model) -> list[tuple[str, dict]]:
    """Fall back to one variation per numeric parameter, at its upper bound."""
    variations = []
    for spec in model.params:
        if spec.kind in {"float", "int"} and spec.high is not None:
            variations.append((f"{spec.name} at its upper bound ({spec.high})", {spec.name: spec.high}))
    return variations[:2]


def show_model(slug: str, overrides: dict, fast: bool) -> None:
    """Print everything one model has to say, and return nothing."""
    model = get_model(slug)
    result = model.run(sweeps=not fast, **overrides)

    print()
    print(RULE)
    print(f"{model.title}")
    print(f"module {model.module} - {model.module_title()}   |   slug: {slug}")
    print(RULE)
    print(f"QUESTION  : {model.cognitive_question}")
    print(f"MECHANISM : {model.quantum_mechanism}")
    print(f"CLASSICAL : {model.classical_baseline}")
    print()
    print("PARAMETERS (every one adjustable from Python: model.run(name=value))")
    for spec in model.params:
        bounds = f"[{spec.low}, {spec.high}]" if spec.low is not None or spec.high is not None else ""
        choices = f"   one of {list(spec.choices)}" if spec.choices else ""
        current = result.parameters[spec.name]
        print(f"    {spec.name:24s} = {str(current):22s} {spec.kind:6s} {bounds}{choices}")
    if result.shots:
        print(f"    (executed on the {result.backend} backend with {result.shots} shots, seed {result.seed})")
    else:
        print(f"    (executed exactly on the {result.backend} backend)")
    print()
    print(f"SUMMARY: {result.summary}")
    print()
    print("FINDINGS")
    for finding in result.findings:
        print(f"  * {finding}")

    print()
    print(f"AGAINST THE CLASSICAL BASELINE: {result.classical['name']}")
    print(f"    {result.classical['description']}")
    print(f"    {'quantity':48s} {'quantum':>9s} {'classical':>10s}")
    for row in result.classical_comparison:
        print(f"    {row['metric'][:48]:48s} {row['model']:9.3f} {row['baseline']:10.3f}")
    print(f"    -> {result.classical['verdict']}")

    print()
    print("QUANTUM CIRCUIT")
    designs = model.circuit_designs(result)
    if designs:
        for name, circuit in designs.items():
            qubits = "qubit" if circuit.num_qubits == 1 else "qubits"
            bits = "bit" if circuit.num_clbits == 1 else "bits"
            print(f"    {name}: {circuit.num_qubits} {qubits}, {circuit.num_clbits} classical {bits}, "
                  f"depth {circuit.depth()}, {circuit.size()} operations")
    for note in model.circuit_notes(result):
        print(f"    note: {note}")

    print()
    print("WHAT THIS DOES NOT CLAIM")
    for caveat in result.caveats:
        print(f"  - {caveat}")
    return result


def save_artefacts(slug: str, model, result, figures_dir: Path | None, reports_dir: Path | None) -> list[Path]:
    """Save the model's figures and (optionally) its HTML report."""
    written: list[Path] = []
    if figures_dir is not None:
        from qcog.viz import savefig

        figures_dir.mkdir(parents=True, exist_ok=True)
        for name, figure in model.figures(result).items():
            safe = "".join(character if character.isalnum() else "_" for character in name)[:56]
            path = figures_dir / f"{slug}__{safe}.png"
            savefig(figure, path)
            written.append(path)
    if reports_dir is not None:
        from qcog.report import write_model_report

        # the report embeds its figures, so no separate PNGs are needed here
        written.append(write_model_report(model, result, reports_dir, save_figures=False))
    return written


def show_variations(slug: str, overrides: dict, fast: bool) -> None:
    """Print what changes when a parameter is moved - the point of a simulator."""
    model = get_model(slug)
    curated = CURATED_VARIATIONS.get(slug) or auto_variations(model)
    if not curated:
        return
    print()
    print(THIN)
    print(f"ADJUSTING IT: {model.title}")
    print(THIN)
    for label, parameters in curated:
        merged = {**overrides, **parameters}
        result = model.run(sweeps=False, **merged)
        values = ", ".join(f"{key}={value}" for key, value in parameters.items())
        print(f"    {values:52s} -> {result.summary}")


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the models, print and save."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=None, help="model slugs to run")
    parser.add_argument("--all", action="store_true", help="run every model in the catalogue")
    parser.add_argument("--param", nargs="*", default=[], help="extra parameter overrides as name=value")
    parser.add_argument("--out", default=str(ROOT / "demos" / "showcase" / "figures"), help="figure directory")
    parser.add_argument("--reports", action="store_true", help="also write an HTML report per model")
    parser.add_argument("--no-figures", action="store_true", help="do not save anything")
    parser.add_argument("--fast", action="store_true", help="skip the models' parameter sweeps")
    args = parser.parse_args(argv)

    overrides: dict[str, object] = {}
    for item in args.param:
        if "=" not in item:
            parser.error(f"parameter override {item!r} must look like name=value")
        name, raw = item.split("=", 1)
        try:
            overrides[name] = float(raw)
        except ValueError:
            overrides[name] = raw

    slugs = list(CATALOGUE) if args.all else (args.models or DEFAULT_MODELS)
    unknown = [slug for slug in slugs if slug not in CATALOGUE]
    if unknown:
        parser.error(f"unknown model(s) {unknown}; use --all or one of: {', '.join(sorted(CATALOGUE))}")

    figures_dir = None if args.no_figures else Path(args.out)
    reports_dir = (ROOT / "demos" / "showcase" / "reports") if args.reports else None

    print("Q-Cog showcase")
    print(f"   models: {', '.join(slugs)}")
    print(f"   overrides: {overrides or 'none'}   sweeps: {'off' if args.fast else 'on'}")
    print(f"   figures: {figures_dir if figures_dir else 'not saved'}")

    written: list[Path] = []
    for slug in slugs:
        model = get_model(slug)
        result = show_model(slug, overrides, args.fast)
        written.extend(save_artefacts(slug, model, result, figures_dir, reports_dir))

    print()
    print(THIN)
    print("ADJUST THEM: the same models with different parameters")
    print(THIN)
    for slug in slugs:
        show_variations(slug, overrides, args.fast)

    if written:
        print()
        print(f"{len(written)} files written:")
        for path in written:
            try:
                shown = path.relative_to(ROOT)
            except ValueError:  # pragma: no cover - only if --out points outside the repo
                shown = path
            print(f"    {shown}")
    print()
    print("Next: `qcog list` for the catalogue, `qcog run <slug> --param name=value`, "
          "`qcog circuit <slug>`, or the website: `python scripts/build_website.py`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())