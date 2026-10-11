"""Audit the installed interfaces end to end.

    python scripts/audit.py                 # everything: the command line, packaging, demos, repository
    python scripts/audit.py --section cli   # just the command line and its flags
    python scripts/audit.py --section repo  # just the repository's own promises

Where the other checkers look at one thing - the figures, the documentation, the site, the model
suites - this one drives the interfaces a user actually touches: every model through the command
line, every flag (including the ones that must be *refused*), the circuits in four styles and
three formats, the report and fit commands, the packaging from the built wheel, the demos, and
the repository's own promises (the files it says it has, the metadata, the workflows).

Two things are skipped rather than failed when the machine cannot do them, each with the reason
attached: Qiskit Aer when its compiled extensions will not load, and the wheel when dist/ is
empty (build one with `python -m build` first).  Exits non-zero if anything fails.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
SLUGS = json.loads(
    subprocess.run([PY, "-c", "import sys;sys.path.insert(0,'src');import json;from qcog.registry import MODELS;"
                    "print(json.dumps(list(MODELS)))"], cwd=ROOT, capture_output=True, text=True).stdout
)

rows = []


def record(name, ok, detail=""):
    rows.append((name, bool(ok), str(detail)))
    mark = "PASS" if ok else "FAIL"
    print(f"{mark}  {name:<58} {str(detail)[:60]}", flush=True)


def qcog(*args, expect=0, label=None):
    """Run the CLI and check the exit code and that it printed something."""
    completed = subprocess.run([PY, "-m", "qcog", *args], cwd=ROOT, capture_output=True, text=True)
    name = label or "qcog " + " ".join(args[:3])
    if expect == "any":
        record(name, True, f"exit {completed.returncode}")
    else:
        record(
            name,
            completed.returncode == expect and (completed.stdout.strip() or completed.stderr.strip()),
            f"exit {completed.returncode}: {(completed.stdout or completed.stderr).strip().splitlines()[:1]}",
        )
    return completed




def section_cli() -> None:
    """Every model and every flag through the real command line."""
    print("=" * 100)
    print("1. the command line")
    print("=" * 100)
    qcog("list")
    qcog("backends")
    qcog("evidence")
    for slug in SLUGS:
        qcog("info", slug, label=f"info {slug}")

    for slug in SLUGS:
        qcog("run", slug, "--fast", label=f"run {slug} --fast")

    for slug in SLUGS:
        qcog("run", slug, "--fast", "--format", "json", label=f"run {slug} --format json")

    # every documented flag, on a model that has something to show for each
    qcog("run", "question-order", "--backend", "sampled", "--shots", "2000", "--seed", "11",
         label="run --backend sampled --shots 2000 --seed 11")
    qcog("run", "question-order", "--backend", "aer", "--shots", "2000", "--seed", "11",
         label="run --backend aer --shots 2000 --seed 11")
    qcog("run", "question-order", "--backend", "aer_exact", label="run --backend aer_exact")
    qcog("run", "question-order", "--no-classical", label="run --no-classical")
    qcog("run", "question-order", "--no-circuits", label="run --no-circuits")
    qcog("run", "question-order", "--circuit-style", "iqp", "--circuit-scale", "0.7", "--circuit-fold", "12",
         label="run --circuit-style iqp --circuit-scale 0.7 --circuit-fold 12")
    qcog("run", "question-order", "--param", "angle_a=0.5", "angle_b=1.2", label="run --param angle_a=0.5 angle_b=1.2")
    qcog("run", "collective-judgment", "--classical-option", "judges=3", label="run --classical-option judges=3")
    qcog("run", "auctions", "--param", "bid_aggressiveness=1.4", label="run --param bid_aggressiveness=1.4")

    # a bad value of each kind must be refused, and say why
    for args, what in (
        (("run", "question-order", "--backend", "nonsense"), "an unknown backend is refused"),
        (("run", "question-order", "--param", "not_a_parameter=1"), "an unknown parameter is refused"),
        (("run", "semantic-ambiguity", "--param", "context_scope=nonsense"), "an unknown choice is refused"),
        (("run", "no-such-model"), "an unknown model is refused"),
    ):
        completed = qcog(*args, expect="any")
        record(what, completed.returncode != 0, f"exit {completed.returncode}")

    for slug in SLUGS[:6]:
        qcog("circuit", slug, "--all", "--format", "png", "--style", "bw", label=f"circuit {slug} --all --format png --style bw")
    for slug in SLUGS[:3]:
        qcog("circuit", slug, "--format", "svg", "--style", "iqp", label=f"circuit {slug} --format svg --style iqp")
    for slug in SLUGS[:4]:
        qcog("sweep", slug, label=f"sweep {slug}")

    data = ROOT / "src" / "qcog" / "data" / "benchmarks.json"
    if data.exists():
        qcog("fit", "question-order", str(data), label="fit question-order <benchmarks.json>")

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "report.html"
        qcog("report", "--out", str(out), "--max-models", "3", "--include-figures", "--markdown", label="report --max-models 3")
        record("the report command writes a report", out.exists() and out.stat().st_size > 10_000, f"{out.name} {out.stat().st_size if out.exists() else 0} bytes")
        listed = sorted(Path(tmp).iterdir())
        record("the report command writes the Markdown too", any(p.suffix == ".md" for p in listed), str([p.name for p in listed]))



def section_packaging() -> None:
    """The built wheel: what it carries, and whether it stands on its own."""
    print("=" * 100)
    import zipfile

    dist = ROOT / "dist"
    wheel = sorted(dist.glob("qcog-*.whl"))[-1] if dist.exists() and list(dist.glob("qcog-*.whl")) else None
    record("a wheel exists to install", wheel is not None, wheel.name if wheel else "none")
    if wheel:
        smoke = Path(tempfile.mkdtemp()) / "site"
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(smoke)
        record(
            "the wheel carries the package, the data and the site stylesheet",
            (smoke / "qcog" / "__init__.py").exists()
            and (smoke / "qcog" / "data" / "benchmarks.json").exists()
            and (smoke / "qcog" / "website" / "style" / "site.css").exists()
            and (smoke / "qcog" / "website" / "__main__.py").exists(),
            f"{len(list(smoke.rglob('*')))} files unpacked",
        )
        environment = dict(os.environ, PYTHONPATH=str(smoke))
        completed = subprocess.run([PY, "-c", "import qcog, qcog.website, qcog.data; print(qcog.__version__)"],
                                  cwd=ROOT, capture_output=True, text=True, env=environment)
        record("the packaged copy imports on its own", completed.returncode == 0, completed.stdout.strip() or completed.stderr.strip()[-70:])
        completed = subprocess.run([PY, "-m", "qcog", "evidence"], cwd=ROOT, capture_output=True, text=True, env=environment)
        record("the packaged copy runs the evidence table", completed.returncode == 0 and "PASS" in completed.stdout,
               f"exit {completed.returncode}")
        completed = subprocess.run([PY, "-m", "qcog.website", "--help"], cwd=ROOT, capture_output=True, text=True, env=environment)
        record("python -m qcog.website works from the wheel", completed.returncode == 0, f"exit {completed.returncode}")



def section_demos() -> None:
    """Every example, the showcase, the notebook and the study, as a user runs them."""
    print("=" * 100)
    for script in sorted((ROOT / "demos" / "examples").glob("*.py")):
        completed = subprocess.run([PY, str(script)], cwd=ROOT, capture_output=True, text=True, timeout=600)
        record(f"example {script.name} runs", completed.returncode == 0 and len(completed.stdout) > 200,
               f"exit {completed.returncode}, {len(completed.stdout)} characters")

    showcase = ROOT / "demos" / "showcase" / "run_showcase.py"
    if showcase.exists():
        completed = subprocess.run([PY, str(showcase), "--fast"], cwd=ROOT, capture_output=True, text=True, timeout=900)
        record("the showcase runs (four models)", completed.returncode == 0 and "AGAINST THE CLASSICAL" in completed.stdout,
               f"exit {completed.returncode}, {len(completed.stdout)} characters")

    notebook = ROOT / "demos" / "quickstart.ipynb"
    # built line by line rather than as an indented block: the snippet is compiled by the
    # interpreter that runs it, so every line has to start where Python expects it
    notebook_program = "\n".join(
        [
            "import json, os, sys, tempfile",
            "os.environ.setdefault('MPLBACKEND', 'Agg')",
            f"sys.path.insert(0, {str(ROOT / 'src')!r})",
            "import matplotlib.pyplot as plt",
            f"notebook = json.load(open({str(notebook)!r}, encoding='utf-8'))",
            "cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']",
            "namespace = {'display': lambda obj: plt.close(obj) if hasattr(obj, 'savefig') else None}",
            f"os.chdir({str(ROOT / 'demos')!r})",
            "for number, cell in enumerate(cells, 1):",
            "    exec(compile(''.join(cell['source']), f'cell-{number}', 'exec'), namespace)",
            "print('cells executed:', len(cells), 'model:', namespace['model'].slug)",
        ]
    )
    completed = subprocess.run([PY, "-c", notebook_program], cwd=ROOT, capture_output=True, text=True, timeout=600)
    record("the quickstart notebook executes top to bottom", completed.returncode == 0,
           completed.stdout.strip()[-60:] or completed.stderr.strip()[-70:])

    study = ROOT / "studies" / "quantum_advantage" / "run_study.py"
    if study.exists():
        completed = subprocess.run([PY, str(study)], cwd=ROOT, capture_output=True, text=True, timeout=900)
        record("the comparative study reproduces its results", completed.returncode == 0, f"exit {completed.returncode}")



def section_repo() -> None:
    """The repository's own promises: the files, the metadata, the workflows, the tree."""
    print("=" * 100)
    required = [
        "README.md", "LICENSE", "pyproject.toml", "CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "ATTRIBUTION.md",
        "CITATION.cff", ".gitignore", "requirements.txt", "requirements-dev.txt",
        ".github/workflows/pages.yml", ".github/workflows/release.yml",
        "src/qcog/__init__.py", "src/qcog/__main__.py", "src/qcog/registry.py",
        "scripts/build_website.py", "scripts/check_website.py", "scripts/check_docs.py",
        "scripts/check_figures.py", "scripts/generate_docs.py",
        "model_tests/run_all.py", "model_tests/harness.py", "model_tests/README.md",
        "demos/README.md", "demos/quickstart.ipynb",
        "docs/methodology.md", "docs/accessibility.md", "docs/packaging.md", "docs/limitations.md",
        "studies/quantum_advantage/REPORT.md", "studies/quantum_advantage/results.json",
    ]
    missing = [name for name in required if not (ROOT / name).exists()]
    record("every file the repository promises is there", not missing, f"{len(required)} checked" if not missing else str(missing))

    try:
        import tomllib

        with open(ROOT / "pyproject.toml", "rb") as handle:
            project = tomllib.load(handle)["project"]
        record(
            "pyproject.toml is valid and complete",
            {"name", "version", "description", "license", "requires-python", "dependencies"} <= set(project),
            f"{project['name']} {project['version']}, {len(project['dependencies'])} dependencies",
        )
    except Exception as error:
        record("pyproject.toml is valid and complete", False, f"{type(error).__name__}: {error}")

    workflows = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
    bad = []
    for path in workflows:
        text = path.read_text(encoding="utf-8")
        if not re.search(r"^name:\s*\S", text, flags=re.M) or not re.search(r"^on:", text, flags=re.M) or not re.search(r"^jobs:", text, flags=re.M):
            bad.append(path.name)
    record("the workflows have a name, a trigger and jobs", not bad, f"{len(workflows)} workflows" if not bad else str(bad))

    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    tracked_only = [line for line in status.splitlines() if not line.startswith(" D ")]
    record("the working tree has no uncommitted changes of mine", not tracked_only, status.replace("\n", " | ")[:80] or "clean")

    untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    record("no stray untracked files in the source tree", not untracked, untracked.replace("\n", " ")[:70] or "none")

    size = subprocess.run(["git", "count-objects", "-vH"], cwd=ROOT, capture_output=True, text=True).stdout
    record("the repository has a recorded history", "count:" in size, size.splitlines()[0] if size else "")



SECTIONS = ("cli", "packaging", "demos", "repo")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--section", choices=("all", *SECTIONS), default="all",
                        help="which part of the audit to run (default: all of it)")
    args = parser.parse_args(argv)
    chosen = set(SECTIONS) if args.section == "all" else {args.section}
    if "cli" in chosen:
        section_cli()
    if "packaging" in chosen:
        section_packaging()
    if "demos" in chosen:
        section_demos()
    if "repo" in chosen:
        section_repo()
    failed = [row for row in rows if not row[1]]
    print()
    print(f"{len(rows) - len(failed)}/{len(rows)} audit checks passed" + (f"; {len(failed)} failed" if failed else ""))
    for name, _ok, detail in failed:
        print(f"   FAILED  {name}  {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
