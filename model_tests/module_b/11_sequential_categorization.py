"""Sequential categorization - module B, model 11 of 23.

A sequence of category labels presented one after another.  The checks confirm that
consecutive judgements agree more often than independent draws would, that the answer rate stays
flat (which is why stability has to be conditional), and that the number of labels drives the
figure.

Run it on its own:

    python module_b/11_sequential_categorization.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _extra in (ROOT / "src", ROOT / "model_tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import numpy as np  # noqa: E402
import qcog  # noqa: E402
from harness import Suite, check_sweep_anchor, main_for  # noqa: E402

SLUG = "sequential-categorization"
MODULE = "B"
TITLE = "Sequential categorization"


def specific(suite: Suite, result) -> None:
    """The checks that only make sense for this model."""
    model = suite.model
    agreement = float(result.analysis["agreement"])
    suite.check("consecutive judgements agree more often than not", agreement > 0.6, f"{agreement:.4f}")
    stability = float(result.analysis["stability"])
    suite.check("an answer is repeated more often than independent judgement predicts", stability > agreement, f"{stability:.4f} against {agreement:.4f}")

    rows = {row["metric"]: row for row in result.classical_comparison}
    suite.check(
        "the classical baseline is the independent-judgement prediction",
        float(rows["answer stability under repeated labels"]["baseline"]) > 0.0,
        str(rows["answer stability under repeated labels"]),
    )
    suite.check(
        "the model beats that baseline",
        float(rows["answer stability under repeated labels"]["model"])
        > float(rows["answer stability under repeated labels"]["baseline"]),
    )

    final = result.analysis["final_distribution"]
    suite.close("the final distribution sums to one", 1.0, sum(float(value) for value in final.values()), tol=1e-9)
    trajectory = result.analysis["trajectory"]
    suite.check(
        "the trajectory follows every label the model was asked for",
        len(trajectory["x"]) >= 3,
        f"{len(trajectory['x'])} labels",
    )
    fewer = suite.run(sweeps=False, num_labels=2)
    suite.check(
        "the number of labels changes the sequence the figure draws",
        len(fewer.analysis["trajectory"]["x"]) < len(trajectory["x"]),
        f"2 labels give {len(fewer.analysis['trajectory']['x'])} steps",
    )
    suite.check(
        "the label parameter really changes how many questions are asked",
        len(fewer.results["sequence"].order) < len(result.results["sequence"].order),
        f"{len(result.results['sequence'].order)} questions",
    )


if __name__ == "__main__":
    raise SystemExit(main_for(SLUG, specific, module=MODULE))
