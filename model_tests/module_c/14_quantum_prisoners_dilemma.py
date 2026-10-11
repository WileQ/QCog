"""Quantum prisoners dilemma - module C, model 14 of 23.

The Eisert-Wilkens-Lewenstein scheme: two players share an entangled state, play unitary
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
    row = table[(table["first"] == "C") & (table["second"] == "C")].iloc[0]
    suite.close("the payoff table gives mutual cooperation the cooperation payoff", 3.0, float(row["quantum_first"]), tol=1e-9)
    suite.check("the payoff table flags mutual cooperation as cooperative", float(row["cooperation"]) == 1.0, str(float(row["cooperation"])))
    suite.check(
        "the table shows what defecting on a cooperator pays",
        abs(float(table[(table["first"] == "D") & (table["second"] == "C")].iloc[0]["quantum_first"])) < 1e-9,
        "at this entanglement the temptation is gone",
    )

    # the entanglement is what makes cooperation stable: the temptation falls, and defecting
    # stops being the better reply
    sweep = result.analysis["sweeps"]["entanglement"].table
    suite.check(
        "the entanglement sweep reports the temptation",
        "temptation to defect" in sweep.columns,
        ", ".join(map(str, sweep.columns)),
    )
    classical_end = float(sweep["temptation to defect"].iloc[0])
    entangled_end = float(sweep["temptation to defect"].iloc[-1])
    suite.close("with no entanglement the temptation is the classical 5", 5.0, classical_end, tol=1e-6)
    suite.check("with maximal entanglement it is gone", entangled_end < 1e-4, f"{entangled_end:.6f}")
    suite.check(
        "so cooperation becomes the better reply: the gain from cooperating turns positive",
        float(sweep["gain from cooperating"].iloc[0]) < 0 < float(sweep["gain from cooperating"].iloc[-1]),
        f"{float(sweep['gain from cooperating'].iloc[0]):+.3f} at no entanglement, "
        f"{float(sweep['gain from cooperating'].iloc[-1]):+.3f} at maximal",
    )
    best = result.analysis["best_response"]
    suite.check(
        "no deviation from mutual cooperation pays at maximal entanglement",
        float(best["payoff"]) <= 3.0 + 1e-6,
        f"best reply to cooperation is worth {float(best['payoff']):.3f}, cooperation itself 3.000",
    )
    no_entanglement = suite.run(sweeps=False, entanglement=0.0)
    suite.check(
        "without entanglement the classical dilemma is back",
        float(no_entanglement.analysis["payoff_table"][(no_entanglement.analysis["payoff_table"]["first"] == "D") & (no_entanglement.analysis["payoff_table"]["second"] == "C")].iloc[0]["quantum_first"]) > 4.9,
        "defecting on a cooperator pays the classical 5",
    )

    other = suite.run(sweeps=False, first_strategy="defect (D)")
    suite.check(
        "changing a player's strategy changes the outcome",
        abs(float(other.analysis["joint_moves"]["C-C"]) - 1.0) > 1e-6,
        f"first_strategy='defect (D)' gives {float(other.analysis['joint_moves']['C-C']):.3f}",
    )
    check_sweep_anchor(
        suite,
        result,
        "entanglement",
        {
            "cooperation": lambda run: run.analysis["joint_moves"]["C-C"],
            "temptation to defect": lambda run: run.analysis["payoff_table"][
                (run.analysis["payoff_table"]["first"] == "D")
                & (run.analysis["payoff_table"]["second"] == "C")
            ].iloc[0]["quantum_first"],
        },
    )


if __name__ == "__main__":
    raise SystemExit(main_for(SLUG, specific, module=MODULE))
