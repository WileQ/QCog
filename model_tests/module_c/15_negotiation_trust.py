"""Negotiation trust - module C, model 15 of 23.

A bargaining game where commitment is profitable but risky.  The checks read the agreement
probability against the classical mixed equilibrium, the effect of suspicion, and the payoff
table's two panels.

Run it on its own:

    python module_c/15_negotiation_trust.py
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

SLUG = "negotiation-trust"
MODULE = "C"
TITLE = "Negotiation trust"


def specific(suite: Suite, result) -> None:
    """The checks that only make sense for this model."""
    model = suite.model
    joint = result.analysis["joint_moves"]
    agreement = float(joint["commit-commit"])
    suite.check("the parties reach the profitable outcome most of the time", agreement > 0.5, f"{agreement:.4f}")
    suite.check("all four outcomes are reported", len(joint) == 4, str(sorted(joint)))
    suite.close("the joint distribution over the four outcomes sums to one", 1.0, sum(float(v) for v in joint.values()))
    rows = {row["metric"]: row for row in result.classical_comparison}
    if "probability of the mutually cooperative outcome" in rows:
        row = rows["probability of the mutually cooperative outcome"]
        suite.check("the model beats the classical mixed equilibrium", float(row["model"]) > float(row["baseline"]), str(row))
    suite.check(
        "the payoff matrix is 2x2 and symmetric",
        len(result.analysis["payoff_matrix"]) == 2 and float(result.analysis["payoff_matrix"][0][1]) == 0.0,
        str(result.analysis["payoff_matrix"]),
    )
    doubtful = suite.run(sweeps=False, suspicion=1.0)
    suite.check(
        "suspicion lowers the agreement probability",
        float(doubtful.analysis["joint_moves"]["commit-commit"]) < agreement,
        f"{float(doubtful.analysis['joint_moves']['commit-commit']):.4f} against {agreement:.4f}",
    )
    eager = suite.run(sweeps=False, first_stance=1.0, second_stance=1.0)
    suite.check("the stances are the players' own dials", eager is not None, "both stances at 1.0")
    # sharing a context is what makes committing safe: what hedging against a committer pays
    # falls from the classical 3 to 0
    table = result.analysis["sweeps"]["shared context"].table
    parameter = result.analysis["sweeps"]["shared context"].parameter
    own = float(result.parameters["shared_context"])
    index = min(range(len(table)), key=lambda row: abs(float(table[parameter].iloc[row]) - own))
    suite.check(
        "the shared-context sweep reports what hedging costs",
        "hedging against commitment" in table.columns,
        ", ".join(map(str, table.columns)),
    )
    suite.close(
        "with no shared context hedging against a committer pays the classical 3",
        3.0,
        float(table["hedging against commitment"].iloc[0]),
        tol=0.05,      # the sweep carries the model's own suspicion, which costs a little
    )
    suite.check(
        "with maximal shared context hedging is nearly worthless, so committing is safe",
        float(table["hedging against commitment"].iloc[-1]) < 0.2,
        f"{float(table['hedging against commitment'].iloc[-1]):.3f} against a commitment payoff of 4.000",
    )
    suite.close(
        "the sweep is the run: the same agreement at the model's own shared context",
        agreement,
        float(table["agreement"].iloc[index]),
        tol=1e-6,
    )
    check_sweep_anchor(
        suite,
        result,
        "shared context",
        {
            "agreement": lambda run: run.analysis["joint_moves"]["commit-commit"],
        },
    )
    check_sweep_anchor(
        suite,
        result,
        "suspicion",
        {"agreement": lambda run: run.analysis["joint_moves"]["commit-commit"]},
    )


if __name__ == "__main__":
    raise SystemExit(main_for(SLUG, specific, module=MODULE))
