"""Collective judgment - module D, model 21 of 23.

A group of judges voting on the same question.  The checks reproduce Condorcet's theorem at
zero correlation, show what correlation does to the majority and to the variance of the vote
count, and read the GHZ circuit that prepares the correlated component.

Run it on its own:

    python module_d/21_collective_judgment.py
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

SLUG = "collective-judgment"
MODULE = "D"
TITLE = "Collective judgment"


def specific(suite: Suite, result) -> None:
    """The checks that only make sense for this model."""
    model = suite.model
    majority = float(result.analysis["majority_accuracy"])
    condorcet = float(result.analysis["condorcet"])
    suite.check("the group beats the individual judge", majority > float(result.parameters["individual_accuracy"]) + 1e-9, f"{majority:.4f} against {result.parameters['individual_accuracy']}")
    suite.check("correlation costs the group accuracy", majority < condorcet, f"{majority:.4f} against Condorcet's {condorcet:.4f}")
    independent = suite.run(sweeps=False, correlation=0.0)
    suite.close(
        "with independent judges the model reproduces Condorcet's theorem exactly",
        float(independent.analysis["condorcet"]),
        float(independent.analysis["majority_accuracy"]),
        tol=1e-9,
    )
    unanimous = suite.run(sweeps=False, correlation=1.0)
    suite.close(
        "with perfectly correlated judges the group is no better than one judge",
        float(result.parameters["individual_accuracy"]),
        float(unanimous.analysis["majority_accuracy"]),
        tol=1e-6,
    )
    variance = result.analysis["variance"]
    suite.check("the vote count varies more than independence allows", float(variance["model"]) > float(variance["independent"]), str(variance))
    suite.close(
        "with independence the variance is the binomial one",
        float(independent.analysis["variance"]["independent"]),
        float(independent.analysis["variance"]["model"]),
        tol=1e-6,
    )
    distribution = result.analysis["vote_distribution"]
    suite.close("the vote distribution sums to one", 1.0, sum(float(v) for v in distribution.values()), tol=1e-9)
    suite.check("the model reports one marginal per judge", len(result.analysis["histogram"]["series"]) >= 1, str(sorted(result.analysis["histogram"]["series"])))
    votes = result.results["votes"]
    suite.check(
        "every judge is asked the same question",
        len(votes.order) == int(result.parameters["num_agents"]),
        f"{len(votes.order)} judges",
    )
    suite.check(
        "the correlated component is drawn as a GHZ preparation",
        any("GHZ" in name for name in model.circuit_designs(result)),
        str(sorted(model.circuit_designs(result))),
    )
    suite.check(
        "the model explains why the group state cannot be compiled",
        any("mixed" in note for note in model.circuit_notes(result)),
        str(model.circuit_notes(result))[:120],
    )
    check_sweep_anchor(
        suite,
        result,
        "correlation",
        {
            "majority accuracy": lambda run: run.analysis["majority_accuracy"],
            "variance": lambda run: run.analysis["variance"]["model"],
        },
    )


if __name__ == "__main__":
    raise SystemExit(main_for(SLUG, specific, module=MODULE))
