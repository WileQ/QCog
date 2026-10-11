"""Belief revision - module A, model 5 of 23.

Ambiguous evidence, framing and repeated weak evidence.  The checks confirm that framing
moves the follow-up judgement while the classical likelihood account cannot, and that the
trajectory really has as many steps as the model was asked for (the figure follows the
parameter).

Run it on its own:

    python module_a/05_belief_revision.py
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

SLUG = "belief-revision"
MODULE = "A"
TITLE = "Belief revision"


def specific(suite: Suite, result) -> None:
    """The checks that only make sense for this model."""
    model = suite.model
    effect = result.analysis["framing_effect"]
    suite.check("framing changes the immediate response to the evidence", abs(float(effect["immediate"])) > 0.1, f"{float(effect['immediate']):+.4f}")
    suite.check("framing changes the later judgement", abs(float(effect["judgement"])) > 0.01, f"{float(effect['judgement']):+.4f}")
    suite.check(
        "the classical likelihood account predicts no framing effect",
        abs(float(result.classical_comparison[0]["baseline"])) < 1e-9,
        str(result.classical_comparison[0]),
    )

    trajectory = result.analysis["trajectory"]
    expected = int(result.parameters["repetitions"])
    suite.check(
        "the trajectory has one step per presentation of the evidence",
        len(trajectory["x"]) == expected and all(len(series) == expected for series in trajectory["series"].values()),
        f"{len(trajectory['x'])} steps for repetitions={expected}",
    )
    suite.check(
        "the trajectory compares the framed and the unframed belief",
        {"framed", "unframed"} <= set(trajectory["series"]),
        str(sorted(trajectory["series"])),
    )
    shorter = suite.run(sweeps=False, repetitions=3)
    suite.check(
        "the number of presentations changes the trajectory the figure draws",
        len(shorter.analysis["trajectory"]["x"]) == 3,
        f"{shorter.analysis['trajectory']['x']}",
    )
    weak = suite.run(sweeps=False, evidence_strength=0.1)
    suite.check(
        "weak evidence moves the belief less than strong evidence",
        abs(float(weak.analysis["framing_effect"]["immediate"])) <= abs(float(effect["immediate"])) + 1e-9,
        f"{float(weak.analysis['framing_effect']['immediate']):+.4f} against {float(effect['immediate']):+.4f}",
    )
    suite.check(
        "the frames converge to the same evidence eigenstate",
        abs(
            float(list(result.analysis["trajectory"]["series"]["framed"])[-1])
            - float(list(result.analysis["trajectory"]["series"]["unframed"])[-1])
        )
        < 0.05,
        "the last steps agree",
    )
    suite.check("the trajectory figure is drawn", "trajectory" in model.figures(result))


if __name__ == "__main__":
    raise SystemExit(main_for(SLUG, specific, module=MODULE))
