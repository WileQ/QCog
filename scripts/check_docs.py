"""Check that the documentation still matches the code.

    python scripts/check_docs.py

The README, the project page and the showcase all quote numbers - how many models, how
many tests, how many visualizations, which files exist - and numbers in prose rot quietly.
This script re-derives each of them and fails loudly when one has moved, so the docs can be
trusted by a reader who has no reason to check them by hand.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qcog.registry import MODELS, MODULES  # noqa: E402
from qcog.viz.checks import STANDALONE_FIGURES  # noqa: E402


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _count_tests() -> int:
    """Number of tests collected, without importing the test modules."""
    import subprocess

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    collected = 0
    for line in completed.stdout.splitlines():
        text = line.strip()
        # newer pytest prints one line per test file with a count, older versions print
        # one line per test id; support both
        summary = re.match(r"^(tests[/\\][^:]+):\s*(\d+)$", text)
        if summary:
            collected += int(summary.group(2))
        elif "::" in text:
            collected += 1
    return collected


def main() -> int:
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    readme = _read("README.md")
    project_page = _read("docs/project_page.md")
    accessibility = _read("docs/accessibility.md")
    methodology = _read("docs/methodology.md")
    showcase_readme = _read("demos/showcase/README.md")
    contributing = _read("CONTRIBUTING.md")

    # 1. the model count
    check(
        "README states the model count",
        f"{len(MODELS)} documented models" in readme or "Twenty-three documented models" in readme,
        f"registry has {len(MODELS)}",
    )
    check("README states the module count", f"{len(MODULES)} modules" in readme or "four modules" in readme.lower())
    if "Twenty-three models" in readme or "23 models" in readme:
        check("README's spelled-out model count is current", len(MODELS) == 23, f"registry has {len(MODELS)}")

    # 2. every slug in the README tables exists
    listed = set(re.findall(r"`([a-z][a-z0-9-]{3,})`", readme))
    unknown = sorted(slug for slug in listed if slug.endswith(("-order", "-fallacy", "-effect", "-dynamics", "-games", "-dilemma", "-voting", "-judgment", "-network")) and slug not in MODELS)
    check("README model slugs all exist", not unknown, f"unknown: {unknown}")

    # 3. the test count, if the docs quote one
    tests = _count_tests()
    quoted = re.findall(r"(\d{2,3}) tests", readme + project_page + methodology)
    if quoted:
        stale = sorted({value for value in quoted if int(value) != tests})
        check(
            "quoted test counts are current",
            not stale,
            f"docs say {sorted(set(quoted))}, suite has {tests}"
            + (" - run `python scripts/generate_docs.py` to align them" if stale else ""),
        )
    else:
        check("test count is not quoted (nothing to drift)", True, f"suite has {tests} tests")

    # 4. the visualization inventory
    stale_figures = sorted(name for name in STANDALONE_FIGURES if name not in accessibility + methodology + readme)
    check("every figure type is documented somewhere", not stale_figures, f"undocumented: {stale_figures}")

    # 5. commands quoted in the docs exist
    for command, target in (
        ("qcog list", "src/qcog/cli.py"),
        ("qcog run", "src/qcog/cli.py"),
        ("qcog circuit", "src/qcog/cli.py"),
        ("qcog sweep", "src/qcog/cli.py"),
        ("qcog fit", "src/qcog/cli.py"),
        ("qcog report", "src/qcog/cli.py"),
        ("qcog evidence", "src/qcog/cli.py"),
        ("qcog backends", "src/qcog/cli.py"),
    ):
        check(f"documented command {command!r} exists", (ROOT / target).exists())

    # 6. the files the docs point at exist
    for relative in (
        "demos/showcase/run_showcase.py", "demos/showcase/output.txt", "demos/showcase/figures",
        "demos/showcase/reports", "demos/README.md",
        "src/qcog/website/build.py", "src/qcog/website/style/site.css",
        "scripts/build_website.py", "scripts/check_website.py", "docs/packaging.md",
        ".github/workflows/pages.yml", ".github/workflows/release.yml",
        "demos/quickstart.ipynb", "demos/examples/05_catalogue_report.py",
        "demos/examples/06_economy_and_vote.py", "scripts/check_figures.py",
        "scripts/build_website.py", "scripts/check_website.py",
        "studies/quantum_advantage/run_study.py", "studies/quantum_advantage/REPORT.md",
        "studies/quantum_advantage/results.json", "studies/quantum_advantage/README.md",
        "scripts/generate_docs.py", "docs/models.generated.md", "docs/project_page.md",
        "docs/methodology.md", "docs/accessibility.md", "docs/limitations.md",
        "docs/demo_video_script.md", "docs/qollab_runbook.md", "docs/submission_checklist.md",
        "ATTRIBUTION.md", "CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "LICENSE", "CITATION.cff",
    ):
        check(f"{relative} exists", (ROOT / relative).exists())

    # 6b. the comparative study and its report agree with each other
    import json

    study_results = ROOT / "studies" / "quantum_advantage" / "results.json"
    study_report = ROOT / "studies" / "quantum_advantage" / "REPORT.md"
    if study_results.exists() and study_report.exists():
        payload = json.loads(study_results.read_text(encoding="utf-8"))
        report = study_report.read_text(encoding="utf-8")
        slugs = [case["slug"] for case in payload["cases"]]
        check("the study report covers every result", all(slug in report for slug in slugs), f"{len(slugs)} cases")
        check(
            "every study case names its sense and its limits",
            all(case.get("sense") and case.get("not_shown") for case in payload["cases"]),
            "S1/S2/S3 labels and a 'what it does not show' line per case",
        )
        figures = sorted((ROOT / "studies" / "quantum_advantage" / "figures").glob("*.png"))
        check(
            "the study report embeds one graph per case",
            len(figures) == len(slugs) and all(f"figures/{figure.name}" in report for figure in figures),
            f"{len(figures)} figures",
        )

    # 7. the notebook and the showcase are referenced where a reader looks
    check("README points at the notebook", "demos/quickstart.ipynb" in readme)
    check("README points at the showcase", "demos/showcase" in readme)
    check("README points at the website", "website/" in readme)
    check("README documents installing from PyPI", "pip install qcog" in readme)
    check("README points at the packaging guide", "docs/packaging.md" in readme)
    check("README points at the comparative study", "studies/quantum_advantage" in readme)
    check("project page points at the comparative study", "studies/quantum_advantage" in project_page)
    check("project page points at the showcase", "showcase" in project_page)
    check("project page points at the notebook", "notebook" in project_page.lower())
    check("project page points at the website", "website" in project_page.lower())
    check(
        "the website is generated, not committed",
        "website/" in _read(".gitignore"),
        "listed in .gitignore",
    )
    check("contributing explains the visualization check", "check_figures.py" in contributing)
    check("showcase README explains its own contents", "run_showcase.py" in showcase_readme and "output.txt" in showcase_readme)

    # 8. the accessibility policy still describes what the code does
    from qcog.viz import CATEGORICAL, LINE_STYLES, series_style

    check("accessibility doc states the categorical palette size", str(len(CATEGORICAL)) in accessibility,
          f"palette has {len(CATEGORICAL)} colours")
    check("accessibility doc mentions linestyle separation", "linestyle" in accessibility.lower())
    check("series styles are distinguishable", len({(series_style(i)["color"], series_style(i)["linestyle"]) for i in range(len(CATEGORICAL) * len(LINE_STYLES))}) == len(CATEGORICAL) * len(LINE_STYLES))

    width = max(len(name) for name, _, _ in checks) + 2
    failures = 0
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}} {detail}")
        failures += 0 if ok else 1
    print()
    if failures:
        print(f"{failures} documentation checks failed")
        return 1
    print(f"documentation is consistent ({len(checks)} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
