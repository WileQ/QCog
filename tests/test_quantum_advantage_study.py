"""The quantum-advantage study must keep running and keep reporting honest results."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies" / "quantum_advantage"


@pytest.fixture(scope="module")
def study_results(tmp_path_factory):
    """Run the study (figures off, for speed) and read its results."""
    out = tmp_path_factory.mktemp("study")
    completed = subprocess.run(
        [sys.executable, str(STUDY / "run_study.py"), "--out", str(out), "--no-figures"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=1800,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads((out / "results.json").read_text(encoding="utf-8")), completed.stdout


def test_study_runs_and_covers_every_case(study_results):
    results, output = study_results
    cases = results["cases"]
    assert len(cases) == 5
    for case in cases:
        assert case["sense"].startswith(("S1", "S2", "S3")), case["slug"]
        assert case["not_shown"], case["slug"]
        assert case["metrics"], case["slug"]
        assert case["evidence"], case["slug"]


def test_every_case_shows_the_quantum_account_ahead(study_results):
    results, _ = study_results
    for case in results["cases"]:
        favoured = [row for row in case["metrics"] if row["favours"] == "quantum"]
        assert favoured, f"{case['slug']} reports no advantage at all"
        for row in favoured:
            quantum, classical = float(row["quantum"]), float(row["classical"])
            assert quantum == quantum and classical == classical  # no NaNs in the results
            assert row["direction"] in {"lower", "higher"}, row
            if row["direction"] == "lower":
                assert quantum < classical, (case["slug"], row)
            else:
                assert quantum > classical, (case["slug"], row)


def test_study_reports_where_the_classical_model_survives(study_results):
    """A study that only ever concludes 'quantum wins' is not evidence of anything."""
    results, output = study_results
    caveats = " ".join(case["not_shown"] for case in results["cases"])
    assert "beta-binomial" in caveats
    assert "joints" in caveats


def test_the_reported_report_and_results_exist():
    assert (STUDY / "REPORT.md").exists()
    assert (STUDY / "results.json").exists()
    figures = sorted((STUDY / "figures").glob("*.png"))
    assert len(figures) == 5, [path.name for path in figures]
    report = (STUDY / "REPORT.md").read_text(encoding="utf-8")
    for figure in figures:
        assert f"figures/{figure.name}" in report, figure.name


def test_study_figures_are_graphs():
    from qcog.viz import has_drawn_marks
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    for path in (STUDY / "figures").glob("*.png"):
        image = mpimg.imread(path)
        assert image.shape[0] > 100 and image.shape[1] > 100, path.name
    assert has_drawn_marks  # the checker used across the library
    plt.close("all")
