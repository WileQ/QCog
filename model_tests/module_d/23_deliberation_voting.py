"""Deliberation voting - module D, model 23 of 23.

Preference aggregation with three options, before and after deliberation.  The checks read the
plurality winner, the pairwise majorities, the Condorcet cycle question, and confirm that
deliberation - not mere aggregation - is what changes the winner.

Run it on its own:

    python module_d/23_deliberation_voting.py
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

SLUG = "deliberation-voting"
MODULE = "D"
TITLE = "Deliberation voting"


def specific(suite: Suite, result) -> None:
    """The checks that only make sense for this model."""
    model = suite.model
    plurality = result.analysis["plurality"]
    suite.check(
        "the plurality winner changes with deliberation",
        plurality["winner before"] != plurality["winner after"],
        f"{plurality['winner before']} -> {plurality['winner after']}",
    )
    before = plurality["before"]
    after = plurality["after"]
    suite.close("the shares before deliberation sum to one", 1.0, sum(float(v) for v in before.values()))
    suite.close("the shares after deliberation sum to one", 1.0, sum(float(v) for v in after.values()))
    suite.check(
        "the winner before deliberation is the largest share before",
        max(before, key=lambda key: before[key]) == plurality["winner before"],
        str({key: round(float(value), 4) for key, value in before.items()}),
    )
    suite.check(
        "the winner after deliberation is the largest share after",
        max(after, key=lambda key: after[key]) == plurality["winner after"],
        str({key: round(float(value), 4) for key, value in after.items()}),
    )
    pairwise = result.analysis["pairwise"]
    suite.check("every pair of options is compared", len(pairwise) == 3, str(sorted(pairwise)))
    suite.check(
        "each pairwise comparison is a pair of complementary shares",
        all(abs(float(pair[0]) + float(pair[1]) - 1.0) < 1e-9 for pair in pairwise.values()),
        str({key: tuple(round(float(value), 4) for value in pair) for key, pair in pairwise.items()}),
    )
    quiet = suite.run(sweeps=False, deliberation_strength=0.0)
    suite.check(
        "without deliberation nothing moves",
        quiet.analysis["plurality"]["winner before"] == quiet.analysis["plurality"]["winner after"],
        str(quiet.analysis["plurality"]["winner after"]),
    )
    suite.check(
        "the model reports whether the pairwise majorities cycle",
        isinstance(result.analysis["condorcet_cycle"], bool),
        f"cycle: {result.analysis['condorcet_cycle']}",
    )
    suite.check(
        "the comparison row is about the winner's share",
        any("winner" in str(row.get("metric", "")) for row in result.classical_comparison),
        str([row.get("metric") for row in result.classical_comparison]),
    )
    check_sweep_anchor(
        suite,
        result,
        "deliberation strength",
        {"option A share after": lambda run: run.analysis["plurality"]["after"]["option A"]},
    )


if __name__ == "__main__":
    raise SystemExit(main_for(SLUG, specific, module=MODULE))
