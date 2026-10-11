"""Auctions - module C, model 17 of 23.

Bidding: the value estimate and the bid are different questions, so their order matters, and
thinking about rivals rotates the belief.  The checks read both shifts, the bidding table, and the
order effect the classical risk-neutral bidder cannot produce.

Run it on its own:

    python module_c/17_auctions.py
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

SLUG = "auctions"
MODULE = "C"
TITLE = "Auctions"


def specific(suite: Suite, result) -> None:
    """The checks that only make sense for this model."""
    model = suite.model
    suite.check("the order shift is small but non-zero", abs(float(result.analysis["order_shift"])) > 1e-6, f"{float(result.analysis['order_shift']):+.4f}")
    suite.check("rival information shifts the bid", float(result.analysis["rival_shift"]) > 0.05, f"{float(result.analysis['rival_shift']):+.4f}")
    suite.check(
        "the order shift is the difference between the two orders",
        abs(float(result.analysis["order_shift"])) < 0.5,
        "within the scale of a probability",
    )
    joint = result.analysis["joint_moves"]
    suite.check("the six bidding outcomes are reported", len(joint) == 6, str(sorted(joint)))
    suite.close("the bidding distribution sums to one", 1.0, sum(float(v) for v in joint.values()))
    table = result.analysis["bidding_table"]
    suite.check("the bidding table lists every valuation", len(table) >= 3, str([row["valuation"] for row in table]))
    suite.check(
        "the bidding table is conditional probabilities",
        all(0.0 <= float(row["p(bid high | valuation)"]) <= 1.0 for row in table),
        str(table[0]),
    )
    suite.check(
        "a confident bidder bids high more often",
        float(suite.run(sweeps=False, valuation_confidence=0.9).analysis["joint_moves"]["high/bid high"])
        > float(joint["high/bid high"]),
        f"confidence 0.9 closes the gap",
    )
    suite.check(
        "the comparison row names the order invariance a classical bidder has",
        any("order" in str(row.get("metric", "")) for row in result.classical_comparison),
        str([row.get("metric") for row in result.classical_comparison]),
    )


if __name__ == "__main__":
    raise SystemExit(main_for(SLUG, specific, module=MODULE))
