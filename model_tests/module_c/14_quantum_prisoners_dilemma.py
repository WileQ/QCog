"""Quantum prisoners dilemma - module C, model 14 of 23.

The Eisert-Wilkert-Lewenstein scheme: two players share an entangled state, play unitary
strategies, and measure.  The checks confirm the headline comparison - mutual cooperation with
certainty against the classical Nash payoff - read the payoff table, and record what the
entanglement parameter actually does to those numbers.

Run it on its own:

    python module_c/14_quantum_prisoners_dilemma.py
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

SLUG = "quantum-prisoners-dilemma"
MODULE = "C"
TITLE = "Quantum prisoners dilemma"


def specific(suite: Suite, result) -> None:
    """The checks that only make sense for this model."""
    model = suite.model
    moves = result.results["moves"]
    suite.check("the strategy profile is recorded", "moves" in result.results, str(sorted(result.results)))
    joint = result.analysis["joint_moves"]
    suite.close("the default profile cooperates with certainty", 1.0, float(joint["C-C"]))
    suite.check("every other outcome is impossible", all(float(joint[key]) < 1e-12 for key in joint if key != "C-C"), str(joint))

    rows = {row["metric"]: row for row in result.classical_comparison}
    suite.close("mutual cooperation is certain where the classical equilibrium has none", 1.0, float(rows["probability of the mutually cooperative outcome"]["model"]), tol=1e-9)
    suite.close("the classical baseline for cooperation is exactly zero", 0.0, float(rows["probability of the mutually cooperative outcome"]["baseline"]), tol=1e-9)
    suite.check("the quantum payoff beats the classical equilibrium value", float(rows["expected payoff for the first player"]["model"]) > float(rows["expected payoff for the first player"]["baseline"]), str(rows["expected payoff for the first player"]))

    table = result.analysis["payoff_table"]
    row = table[(table["first"] == "Q") & (table["second"] == "Q")].iloc[0]
    suite.close("the payoff table gives Q-vs-Q the cooperation payoff", 3.0, float(row["quantum_first"]), tol=1e-9)
    suite.check("the payoff table flags Q-vs-Q as cooperative", float(row["cooperation"]) == 1.0, str(float(row["cooperation"])))
    suite.check("the model reports a best response to Q", "best_response" in result.analysis, str(result.analysis.get("best_response")))
    suite.check(
        "the contradiction with the drawn circuit is visible",
        float(result.analysis["joint_moves"]["C-C"]) == 1.0,
        "the circuit draws J, the strategies and J dagger; see the note below",
    )

    other = suite.run(sweeps=False, first_strategy="defect (D)")
    suite.check(
        "changing a player's strategy changes the outcome",
        abs(float(other.analysis["joint_moves"]["C-C"]) - 1.0) > 1e-6,
        f"first_strategy='defect (D)' gives {float(other.analysis['joint_moves']['C-C']):.3f}",
    )
    sweep = result.analysis["sweeps"]["entanglement"].table
    flat = all(
        max(float(row[column]) for row in sweep.to_dict("records")) - min(float(row[column]) for row in sweep.to_dict("records")) < 1e-9
        for column in ("cooperation", "quantum payoff", "quantum advantage")
    )
    suite.check("the entanglement sweep is reported", len(sweep) >= 5, f"{len(sweep)} rows")
    suite.note(
        "the entanglement parameter does not move any number this model reports",
        "the sweep is constant in every metric and a run at entanglement=0 is identical to one at pi/2: "
        "the 'quantum' strategy is Q = iZ, a diagonal unitary, so the entangling gates J and J-dagger "
        "cancel for every strategy pair and gamma never enters.  The EWL scheme needs Q = U(pi/2, 0). "
        f"Flat sweep: {flat}.",
    )
    check_sweep_anchor(
        suite,
        result,
        "entanglement",
        {"cooperation": lambda run: run.analysis["joint_moves"]["C-C"]},
    )


if __name__ == "__main__":
    raise SystemExit(main_for(SLUG, specific, module=MODULE))
