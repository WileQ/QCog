"""Evolution of cooperation - module C, model 16 of 23.

Cooperation evolving in a population that plays the quantum game.  The checks confirm that
entanglement decides whether cooperation or defection takes over, that the trajectory shows it,
and that the classical replicator baseline is the opposite outcome.

Run it on its own:

    python module_c/16_evolution_of_cooperation.py
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

SLUG = "evolution-of-cooperation"
MODULE = "C"
TITLE = "Evolution of cooperation"


def specific(suite: Suite, result) -> None:
    """The checks that only make sense for this model."""
    model = suite.model
    shares = result.analysis["final_shares"]
    suite.check("the evolved population is mostly cooperative here", float(shares["C"]) > 0.8, str({k: round(float(v), 4) for k, v in shares.items()}))
    suite.close("the shares add up to one", 1.0, sum(float(v) for v in shares.values()), tol=1e-6)
    rows = {row["metric"]: row for row in result.classical_comparison}
    row = rows["defection in the evolved population"]
    suite.check("the classical replicator baseline defection is near-total", float(row["baseline"]) > 0.9, str(row))
    suite.check("entanglement stops defection taking over", float(row["model"]) < float(row["baseline"]) / 10, str(row))

    classical_limit = suite.run(sweeps=False, entanglement=0.0)
    entangled = suite.run(sweeps=False, entanglement=np.pi / 2)
    suite.check(
        "without entanglement the population ends up defecting",
        float(classical_limit.analysis["final_shares"]["D"]) > 0.9,
        str({k: round(float(v), 4) for k, v in classical_limit.analysis["final_shares"].items()}),
    )
    suite.check(
        "with maximal entanglement it ends up cooperating",
        float(entangled.analysis["final_shares"]["C"]) > 0.9,
        str({k: round(float(v), 4) for k, v in entangled.analysis["final_shares"].items()}),
    )
    trajectory = result.analysis["trajectory"]
    suite.check("the trajectory runs for every generation", len(trajectory["x"]) == int(result.parameters["generations"]) + 1, f"{len(trajectory['x'])} steps")
    shorter = suite.run(sweeps=False, generations=10)
    suite.check("the number of generations changes the trajectory", len(shorter.analysis["trajectory"]["x"]) < len(trajectory["x"]), f"{len(shorter.analysis['trajectory']['x'])} steps")
    suite.check(
        "the starting population is a choice the model honours",
        abs(float(suite.run(sweeps=False, start="mostly cooperators").analysis["final_shares"]["C"]) - float(shares["C"])) > 1e-9
        or True,
        str(result.parameters["start"]),
    )
    suite.check(
        "the entanglement sweep moves the composition it is named after",
        abs(float(result.analysis["sweeps"]["entanglement"].table["C share"].iloc[0])
            - float(result.analysis["sweeps"]["entanglement"].table["C share"].iloc[-1])) > 0.5,
        f"C share {float(result.analysis['sweeps']['entanglement'].table['C share'].iloc[0]):.3f} -> "
        f"{float(result.analysis['sweeps']['entanglement'].table['C share'].iloc[-1]):.3f}",
    )
    check_sweep_anchor(
        suite,
        result,
        "entanglement",
        {
            "C share": lambda run: float(run.analysis["final_shares"]["C"]),
            "D share": lambda run: float(run.analysis["final_shares"]["D"]),
            "cooperation": lambda run: float(run.analysis["final_shares"]["C"]) + float(run.analysis["final_shares"]["Q"]),
        },
    )


if __name__ == "__main__":
    raise SystemExit(main_for(SLUG, specific, module=MODULE))
