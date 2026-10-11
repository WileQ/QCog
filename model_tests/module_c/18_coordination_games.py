"""Coordination games - module C, model 18 of 23.

Two players choosing a convention without talking.  The checks confirm perfect coordination
against the classical coin flip, read the payoff table, and record what the shared-context
parameter does to those numbers.

Run it on its own:

    python module_c/18_coordination_games.py
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

SLUG = "coordination-games"
MODULE = "C"
TITLE = "Coordination games"


def specific(suite: Suite, result) -> None:
    """The checks that only make sense for this model."""
    model = suite.model
    coordination = float(result.analysis["coordination"])
    suite.close("the players coordinate with certainty here", 1.0, coordination, tol=1e-9)
    joint = result.analysis["joint_moves"]
    suite.check("coordination means the two agreeing outcomes", abs(float(joint["convention A/convention A"]) + float(joint["convention B/convention B"]) - 1.0) < 1e-9, str(joint))
    rows = {row["metric"]: row for row in result.classical_comparison}
    row = rows["coordination probability"]
    suite.close("the classical mixed equilibrium coordinates half the time", 0.5, float(row["baseline"]), tol=1e-9)
    suite.check("the model beats it", float(row["model"]) > float(row["baseline"]), str(row))

    table = result.analysis["payoff_table"]
    suite.check("the payoff table covers the three strategies", len(table) >= 4, f"{len(table)} rows")
    zero_context = suite.run(sweeps=False, shared_context=0.0)
    suite.check(
        "a matched pair of conventions is left alone by the entangler",
        abs(float(zero_context.analysis["coordination"]) - coordination) < 1e-9,
        f"coordination {float(zero_context.analysis['coordination']):.3f} with no shared context, "
        f"{coordination:.3f} with maximal",
    )
    # what does move this model is the noise: the efficient convention is not robust to it
    noise_sweep = result.analysis["sweeps"]["decoherence"].table
    suite.check(
        "decoherence takes coordination back to the classical coin flip",
        abs(float(noise_sweep["coordination"].iloc[0]) - 1.0) < 1e-9
        and abs(float(noise_sweep["coordination"].iloc[-1]) - 0.5) < 1e-9,
        f"{float(noise_sweep['coordination'].iloc[0]):.3f} -> {float(noise_sweep['coordination'].iloc[-1]):.3f}",
    )
    check_sweep_anchor(
        suite,
        result,
        "decoherence",
        {"coordination": lambda run: run.analysis["coordination"]},
    )
    suite.check(
        "the biases do change the outcome",
        abs(float(suite.run(sweeps=False, first_bias=0.7).analysis["coordination"]) - coordination) > 1e-9
        or abs(float(suite.run(sweeps=False, first_bias=0.7).analysis["joint_moves"]["convention A/convention A"]) - float(joint["convention A/convention A"])) > 1e-9,
        "first_bias=0.7",
    )
    suite.check("a decoherence sweep is offered as the failure mode", "decoherence" in (result.analysis.get("sweeps") or {}), str(sorted((result.analysis.get("sweeps") or {}))))
    check_sweep_anchor(suite, result, "shared context", {"coordination": lambda run: run.analysis["coordination"]})
    check_sweep_anchor(suite, result, "decoherence", {"coordination": lambda run: run.analysis["coordination"]})


if __name__ == "__main__":
    raise SystemExit(main_for(SLUG, specific, module=MODULE))
