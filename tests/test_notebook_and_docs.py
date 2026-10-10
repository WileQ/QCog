"""The notebook must run, and the documentation must not drift from the code.

Two promises are checked here: the quickstart notebook's code cells execute from top to
bottom, and the numbers and file lists quoted in the documentation still match the
library (``scripts/check_docs.py`` does the same from the command line).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def notebook_cells():
    path = ROOT / "notebooks" / "quickstart.ipynb"
    assert path.exists(), "the quickstart notebook is missing"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_notebook_is_valid_json_with_cells(notebook_cells):
    assert len(notebook_cells) >= 6
    for cell in notebook_cells:
        assert cell["source"], "an empty code cell"


def test_notebook_cells_run_from_top_to_bottom(notebook_cells, tmp_path):
    """Execute the notebook's cells in one interpreter, exactly as a reader would."""

    def display(obj):  # the only IPython function the notebook uses
        if hasattr(obj, "savefig"):
            plt.close(obj)

    import os
    import shutil

    namespace = {"display": display, "plt": plt}
    cwd = Path.cwd()
    report_dir = ROOT / "out" / "notebook_report"
    shutil.rmtree(report_dir, ignore_errors=True)
    try:
        # a notebook runs with its own directory as the working directory
        os.chdir(ROOT / "notebooks")
        for index, cell in enumerate(notebook_cells, 1):
            source = "".join(cell["source"])
            exec(compile(source, f"notebook-cell-{index}", "exec"), namespace)  # noqa: S102
        assert "result" in namespace and namespace["result"].findings
        assert "model" in namespace and namespace["model"].slug == "question-order"
        # the last cell wrote a report, relative to the notebook's directory
        assert (ROOT / "out" / "notebook_report" / "question-order.html").exists()
    finally:
        os.chdir(cwd)
        shutil.rmtree(report_dir, ignore_errors=True)


def test_documentation_matches_the_code():
    """Run the documentation check the same way a reader would."""
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_docs.py")],
        cwd=str(ROOT), capture_output=True, text=True, timeout=900,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "documentation is consistent" in completed.stdout
