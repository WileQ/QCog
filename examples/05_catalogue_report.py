"""Run every model, print the self-check table, and write the full report.

``python examples/05_catalogue_report.py [--max-models N]``
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qcog import run_evidence, write_catalogue_report, write_markdown_report  # noqa: E402
from qcog.registry import MODELS, models_by_module  # noqa: E402

limit = None
if "--max-models" in sys.argv:
    limit = int(sys.argv[sys.argv.index("--max-models") + 1])

print("Q-Cog catalogue")
print("=" * 78)
for letter, models in models_by_module().items():
    print(f"Module {letter}: {models[0].module_title()} ({len(models)} models)")

print()
print("Evidence: what the library checks about itself")
print("=" * 78)
started = time.time()
for row in run_evidence():
    print(f"{row['result']:4s} {row['check']}")
print(f"({time.time() - started:.1f} s)")

print()
started = time.time()
html_path = write_catalogue_report("out/qcog_report.html", max_models=limit, include_figures=True)
markdown_path = write_markdown_report("out/qcog_report.md", max_models=limit)
print(f"HTML report   : {html_path} ({html_path.stat().st_size / 1024:.0f} KB)")
print(f"Markdown report: {markdown_path} ({markdown_path.stat().st_size / 1024:.0f} KB)")
print(f"({time.time() - started:.1f} s for {limit or len(MODELS)} models)")