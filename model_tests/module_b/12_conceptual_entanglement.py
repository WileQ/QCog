"""Conceptual entanglement - module B, model 12 of 23.

Two features of one concept, jointly answerable but correlated.  The checks confirm the
correlation excess over independence, that the concurrence measures exactly that structure, and -
the contrast with the conjunction fallacy - that the joint never exceeds the smaller marginal.

Run it on its own:

    python module_b/12_conceptual_entanglement.py
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

SLUG = "conceptual-entanglement"
MODULE = "B"
TITLE = "Conceptual entanglement"


def specific(suite: Suite, result) -> None:
    """The checks that only make sense for this model."""
    model = suite.model
    structure = result.analysis["joint_structure"]
    p_a = float(structure["p(A)"])
    p_b = float(structure["p(B)"])
    joint = float(structure["p(A and B)"])
    suite.close("independence is the product of the marginals", p_a * p_b, float(structure["independence"]))
    suite.close("the excess is the joint minus independence", joint - p_a * p_b, float(structure["excess"]))
    suite.check("the features are correlated", float(structure["excess"]) > 0.1, f"excess {float(structure['excess']):.4f}")
    suite.check(
        "the joint never exceeds the smaller marginal, unlike the conjunction fallacy",
        joint <= min(p_a, p_b) + 1e-9,
        f"p(A and B) {joint:.3f}, min(p(A), p(B)) {min(p_a, p_b):.3f}",
    )
    entanglement = result.analysis["entanglement"]
    suite.check(
        "the default combination angle gives a maximally entangled pair",
        float(entanglement.concurrence) > 0.99,
        f"concurrence {float(entanglement.concurrence):.4f}",
    )
    suite.check(
        "the features are individually fair coins",
        abs(p_a - 0.5) < 1e-9 and abs(p_b - 0.5) < 1e-9,
        f"p(A) {p_a:.4f}, p(B) {p_b:.4f}",
    )
    uncorrelated = suite.run(sweeps=False, combination_angle=0.0)
    suite.check(
        "at zero combination angle the features become independent",
        abs(float(uncorrelated.analysis["joint_structure"]["excess"])) < 1e-9,
        f"excess {float(uncorrelated.analysis['joint_structure']['excess']):.2e}",
    )
    decohered = suite.run(sweeps=False, decoherence=1.0)
    suite.check(
        "decoherence destroys the correlation",
        float(decohered.analysis["joint_structure"]["excess"]) < float(structure["excess"]),
        f"{float(decohered.analysis['joint_structure']['excess']):.4f} against {float(structure['excess']):.4f}",
    )
    check_sweep_anchor(
        suite,
        result,
        "combination angle",
        {
            "p(A and B)": lambda run: run.analysis["joint_structure"]["p(A and B)"],
            "independence": lambda run: run.analysis["joint_structure"]["independence"],
            "excess": lambda run: run.analysis["joint_structure"]["excess"],
            "concurrence": lambda run: run.analysis["entanglement"].concurrence,
            "entropy of entanglement": lambda run: run.analysis["entanglement"].entropy_of_entanglement,
        },
    )


if __name__ == "__main__":
    raise SystemExit(main_for(SLUG, specific, module=MODULE))
