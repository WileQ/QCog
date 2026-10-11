"""Associative memory - module B, model 9 of 23.

Recall under competing retrieval cues: two similar cues open two paths that interfere.  The
checks read the order effect and the interference, and confirm what the mode switch is supposed to
do - change the framing, not the numbers.

Run it on its own:

    python module_b/09_associative_memory.py
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

SLUG = "associative-memory"
MODULE = "B"
TITLE = "Associative memory"


def specific(suite: Suite, result) -> None:
    """The checks that only make sense for this model."""
    model = suite.model
    report = result.analysis["order_effect"]
    suite.check("the cue and the recall question interfere", float(report.max_marginal_shift()) > 0.1, f"{float(report.max_marginal_shift()):.4f}")
    suite.check("the questions are incompatible", float(report.incompatibility) > 0.5, f"{float(report.incompatibility):.4f}")
    qq = result.analysis["qq_equality"]
    suite.close("QQ holds for the two cues", float(qq.agreement_first_order), float(qq.agreement_second_order), tol=1e-9)

    forward = result.results["A_then_B"]
    conditional = float(forward.conditional("recall", {"cue": "yes"})["yes"])
    unconditional = float(forward.marginal("recall")["yes"])
    suite.check(
        "answering the cue raises recall above its unconditional rate",
        conditional > unconditional,
        f"{conditional:.3f} with the cue against {unconditional:.3f} without",
    )
    suite.check(
        "the interference term explains the gap",
        float(result.analysis["interference"].max_abs_interference) > 1e-6,
        f"{float(result.analysis['interference'].max_abs_interference):.4f}",
    )

    other = suite.run(sweeps=False, mode="single cue")
    same_numbers = all(
        abs(float(forward.joint[outcome]) - float(other.results["A_then_B"].joint[outcome])) < 1e-12
        for outcome in forward.joint
    )
    suite.check(
        "the mode switch changes the framing without changing the numbers",
        same_numbers and other.summary != result.summary,
        f"single cue: {other.summary[:70]!r}",
    )
    suite.check(
        "the single-cue framing is written as one retrieval path",
        "single" in other.summary.lower() and "interference" in result.summary.lower(),
        other.summary[:80],
    )
    check_sweep_anchor(
        suite,
        result,
        "cue_angle",
        {
            "order effect": lambda run: run.analysis["order_effect"].max_marginal_shift(),
            "interference": lambda run: run.analysis["interference"].max_abs_interference,
        },
    )


if __name__ == "__main__":
    raise SystemExit(main_for(SLUG, specific, module=MODULE))
