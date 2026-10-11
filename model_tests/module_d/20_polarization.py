"""Polarization - module D, model 20 of 23.

Two communities that barely talk.  The checks read the community gap, compare it with the
bounded-confidence baseline, and confirm that cross-talk closes the gap.

Run it on its own:

    python module_d/20_polarization.py
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

SLUG = "polarization"
MODULE = "D"
TITLE = "Polarization"


def specific(suite: Suite, result) -> None:
    """The checks that only make sense for this model."""
    model = suite.model
    communities = result.analysis["communities"]
    suite.check("the two communities end apart", abs(float(communities["within community 2"]) - float(communities["within community 1"])) > 0.2, str({k: round(float(v), 4) for k, v in communities.items()}))
    suite.close(
        "the gap is the distance between the community means",
        abs(float(communities["within community 2"]) - float(communities["within community 1"])),
        float(communities["gap"]),
        tol=1e-6,
    )
    metrics = result.analysis["metrics"]
    suite.check("the model reports clusters rather than consensus", int(metrics["clusters"]) >= 2, str(metrics))
    rows = {row["metric"]: row for row in result.classical_comparison}
    suite.check(
        "the bounded-confidence baseline polarises further than the model",
        float(rows["gap between the communities"]["model"]) < float(rows["gap between the communities"]["baseline"]),
        str(rows["gap between the communities"]),
    )
    suite.check(
        "the bounded-confidence gap is reported in the community summary",
        abs(float(communities["gap (bounded confidence)"]) - float(rows["gap between the communities"]["baseline"])) < 1e-6,
        f"{float(communities['gap (bounded confidence)']):.4f}",
    )
    talkative = suite.run(sweeps=False, cross_talk=1.0)
    suite.check(
        "more cross-talk closes the gap",
        abs(float(talkative.analysis["communities"]["within community 1"]) - float(talkative.analysis["communities"]["within community 2"]))
        < abs(float(communities["within community 1"]) - float(communities["within community 2"])),
        f"gap {float(talkative.analysis['communities']['gap']):.4f} against {float(communities['gap']):.4f}",
    )
    equidistant = suite.run(sweeps=False, community_gap=0.2)
    suite.check("a smaller initial gap gives a smaller final gap", float(equidistant.analysis["communities"]["gap"]) < float(communities["gap"]), f"{float(equidistant.analysis['communities']['gap']):.4f}")
    check_sweep_anchor(
        suite,
        result,
        "cross talk",
        {
            "community gap": lambda run: run.analysis["communities"]["gap"],
            "polarisation": lambda run: run.analysis["metrics"]["polarisation"],
        },
    )


if __name__ == "__main__":
    raise SystemExit(main_for(SLUG, specific, module=MODULE))
