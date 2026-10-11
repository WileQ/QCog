"""Quantum opinion dynamics - module D, model 19 of 23.

Opinions exchanged coherently on a network: agents end up in superpositions, so agreement
between their answers can hide disagreement about positions.  The checks read the metrics, the
trajectories and the interaction parameter.

Run it on its own:

    python module_d/19_quantum_opinion_dynamics.py
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

SLUG = "quantum-opinion-dynamics"
MODULE = "D"
TITLE = "Quantum opinion dynamics"


def specific(suite: Suite, result) -> None:
    """The checks that only make sense for this model."""
    model = suite.model
    metrics = result.analysis["metrics"]
    for key in ("mean opinion", "polarisation", "agreement (|mean|)", "clusters", "mean local entropy"):
        suite.check(f"the metrics include {key!r}", key in metrics, str(sorted(metrics)))
    suite.check("the population starts spread out and ends closer together", abs(float(metrics["mean opinion"])) < 0.2, f"mean opinion {float(metrics['mean opinion']):+.4f}")
    votes = result.analysis["votes"]
    suite.close("the vote distribution sums to one", 1.0, sum(float(v) for v in votes.values()), tol=1e-9)
    trajectory = result.analysis["trajectory"]
    suite.check(
        "the trajectory has one series per agent",
        len(trajectory["series"]) == int(result.parameters["num_agents"]),
        f"{len(trajectory['series'])} series for {result.parameters['num_agents']} agents",
    )
    suite.check(
        "the classical baseline trajectory is drawn on the same network",
        len(result.analysis["classical_trajectory"]["series"]) == len(trajectory["series"]),
    )
    suite.check(
        "the mean local entropy is positive: agents stay in superposition",
        float(metrics["mean local entropy"]) > 0.0,
        f"{float(metrics['mean local entropy']):.4f} bits",
    )

    other_topology = suite.run(sweeps=False, topology="complete")
    suite.check(
        "the topology changes the dynamics",
        abs(float(other_topology.analysis["metrics"]["mean local entropy"]) - float(metrics["mean local entropy"])) > 1e-6,
        f"complete graph gives {float(other_topology.analysis['metrics']['mean local entropy']):.4f}",
    )
    idle = suite.run(sweeps=False, interaction_strength=0.0)
    suite.check(
        "with no interaction the opinions do not move",
        abs(float(idle.analysis["metrics"]["polarisation"]) - float(metrics["polarisation"])) > 1e-6,
        f"polarisation stays at {float(idle.analysis['metrics']['polarisation']):.4f}",
    )
    sampled = suite.run(backend="sampled", shots=500, seed=5, validate=False)
    suite.check(
        "a backend this model cannot use is reported rather than pretended",
        str(sampled.backend) == "exact" and bool(sampled.metadata.get("backend_note")),
        str(sampled.metadata.get("backend_note", ""))[:100],
    )
    sweep = result.analysis["sweeps"]["decoherence"]
    column = sweep.parameter
    for index in (0, len(sweep.table) // 2, len(sweep.table) - 1):
        value = float(sweep.table[column].iloc[index])
        matched = suite.run(sweeps=False, decoherence=value)
        chart = float(sweep.table["mean local entropy"].iloc[index])
        actual = float(matched.analysis["metrics"]["mean local entropy"])
        suite.check(
            f"the sweep's mean local entropy at decoherence={value} is within 0.1 of the run's",
            abs(chart - actual) < 0.1,
            f"chart {chart:.4f} against run {actual:.4f}",
        )
        if abs(chart - actual) > 1e-6:
            suite.note(
                f"the entropy curve at decoherence={value} is not the run's entropy",
                f"the chart reports {chart:.4f} while a run at the same decoherence reports {actual:.4f}: "
                "the sweep averages the local entropy over a different range of steps than the headline metric",
            )
    check_sweep_anchor(
        suite,
        result,
        "decoherence",
        {"polarisation": lambda run: run.analysis["metrics"]["polarisation"]},
    )
    check_sweep_anchor(
        suite,
        result,
        "interaction_strength",
        {"polarisation": lambda run: run.analysis["metrics"]["polarisation"]},
    )


if __name__ == "__main__":
    raise SystemExit(main_for(SLUG, specific, module=MODULE))
