"""Information cascades - module D, model 22 of 23.

Agents deciding in sequence, seeing what the others did.  The checks read the cascade onset,
compare it with the rational Bayesian posterior, and confirm that the order of the signal and the
group is what starts a cascade.

Run it on its own:

    python module_d/22_information_cascades.py
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

SLUG = "information-cascades"
MODULE = "D"
TITLE = "Information cascades"


def specific(suite: Suite, result) -> None:
    """The checks that only make sense for this model."""
    model = suite.model
    cascade = result.analysis["cascade"]
    suite.check("a cascade starts in the default run", cascade["onset"] is not None, str(cascade))
    suite.check("every agent decides", int(cascade["agents"]) == len(result.analysis["decisions"]), f"{len(result.analysis['decisions'])} decisions")
    suite.check("the cascade starts early here", int(cascade["onset"]) <= 4, f"onset {cascade['onset']}")
    suite.check("the adoption share is a probability", 0.0 <= cascade["adoptions"] / cascade["agents"] <= 1.0, str(cascade))

    group_first = suite.run(sweeps=False, order="group first")
    suite.check(
        "asking the group first is what starts a cascade",
        cascade["onset"] is not None and group_first.analysis["cascade"]["onset"] is None,
        f"group first gives onset {group_first.analysis['cascade']['onset']}",
    )
    no_pressure = suite.run(sweeps=False, social_pressure=0.0)
    suite.check(
        "without social pressure there is no cascade",
        no_pressure.analysis["cascade"]["onset"] is None,
        str(no_pressure.analysis["cascade"]),
    )
    rows = {row["metric"]: row for row in result.classical_comparison}
    row = rows["adoption share"]
    suite.check("the Bayesian posterior is the classical comparison", 0.0 < float(row["baseline"]) <= 1.0, str(row))
    suite.check(
        "the comparison explains what the classical account misses",
        "Bayes" in str(result.analysis["classical"].get("name", "")),
        str(result.analysis["classical"].get("name")),
    )
    check_sweep_anchor(
        suite,
        result,
        "social pressure",
        {
            "adoptions": lambda run: run.analysis["cascade"]["adoptions"],
            "adoption share": lambda run: run.analysis["cascade"]["adoptions"] / run.analysis["cascade"]["agents"],
        },
    )


if __name__ == "__main__":
    raise SystemExit(main_for(SLUG, specific, module=MODULE))
