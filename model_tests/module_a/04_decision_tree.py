"""Decision tree - module A, model 4 of 23.

Three cues asked in two orders.  A Bayesian tree predicts the same decision either way; this
model does not.  The checks measure the divergence, confirm it closes when the cues are made
equally incompatible, and read the per-cue table the model prints.

Run it on its own:

    python module_a/04_decision_tree.py
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

SLUG = "decision-tree"
MODULE = "A"
TITLE = "Decision tree"


def specific(suite: Suite, result) -> None:
    """The checks that only make sense for this model."""
    model = suite.model
    divergence = float(result.analysis["divergence"])
    suite.check("the two cue orders end in different decisions", divergence > 0.01, f"divergence {divergence:.4f}")
    table = result.analysis["order_table"]
    suite.check("the model tables the state after every cue", len(table) >= 2, f"{len(table)} rows")
    suite.check(
        "the table names the two orders it compares",
        all("order" in row for row in table),
        str([row["order"] for row in table]),
    )
    forward, reversed_order = result.results["forward"], result.results["reversed"]
    suite.close(
        "the divergence is the gap between the two final distributions",
        max(
            abs(float(forward.marginal(forward.order[-1]).get(outcome, 0.0)) - float(reversed_order.marginal(reversed_order.order[-1]).get(outcome, 0.0)))
            for outcome in set(forward.marginal(forward.order[-1])) | set(reversed_order.marginal(reversed_order.order[-1]))
        ),
        divergence,
        tol=1e-6,
    )
    equal = suite.run(sweeps=False, angle_cue_1=1.0, angle_cue_2=1.0, angle_cue_3=1.0)
    suite.check(
        "asking three identical cues removes the order dependence",
        float(equal.analysis["divergence"]) < 1e-6,
        f"{float(equal.analysis['divergence']):.2e}",
    )
    suite.check(
        "the cue angles are the model's own dials",
        abs(
            float(suite.run(sweeps=False, angle_cue_2=0.1).analysis["divergence"])
            - divergence
        )
        > 1e-6,
        "angle_cue_2=0.1",
    )
    suite.check(
        "the unsharp questions mean the drawing shows the sharp version",
        any("sharp" in note for note in model.circuit_notes(result)),
        str(model.circuit_notes(result)),
    )


if __name__ == "__main__":
    raise SystemExit(main_for(SLUG, specific, module=MODULE))
