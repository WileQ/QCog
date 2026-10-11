"""Contradictory beliefs - module B, model 13 of 23.

What someone says and what they do: two questions that cannot be answered from one joint
distribution.  The checks read the contextuality certificate, the order effect behind it and the
QQ equality that survives it.

Run it on its own:

    python module_b/13_contradictory_beliefs.py
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

SLUG = "contradictory-beliefs"
MODULE = "B"
TITLE = "Contradictory beliefs"


def specific(suite: Suite, result) -> None:
    """The checks that only make sense for this model."""
    model = suite.model
    certificate = result.analysis["contextuality_certificate"]
    suite.check(
        "the two questions cannot live in one joint distribution",
        float(certificate["max_ltp_violation"]) > 0.1,
        f"largest violation {float(certificate['max_ltp_violation']):.4f}",
    )
    suite.close(
        "the violation and the distance between the orders agree",
        float(certificate["total_variation_between_orders"]),
        float(certificate["max_ltp_violation"]),
        tol=1e-9,
    )
    report = result.analysis["order_effect"]
    suite.close(
        "the certificate is the order effect the model reports",
        float(report.max_marginal_shift()),
        float(certificate["max_ltp_violation"]),
        tol=1e-9,
    )
    suite.check("the self-report and the behaviour are incompatible", float(certificate["incompatibility"]) > 0.5, f"{float(certificate['incompatibility']):.4f}")
    qq = result.analysis["qq_equality"]
    suite.close("QQ still holds", float(qq.agreement_first_order), float(qq.agreement_second_order), tol=1e-9)
    suite.check(
        "the certificate appears in the comparison rows",
        any("contextuality" in str(row.get("metric", "")) for row in result.classical_comparison),
        str([row.get("metric") for row in result.classical_comparison]),
    )
    unsharp = suite.run(sweeps=False, sharpness=0.0)
    suite.check(
        "unsharp questions remove the contradiction",
        float(unsharp.analysis["contextuality_certificate"]["max_ltp_violation"]) < 1e-6,
        f"{float(unsharp.analysis['contextuality_certificate']['max_ltp_violation']):.2e}",
    )
    check_sweep_anchor(
        suite,
        result,
        "behaviour_angle",
        {
            "order effect": lambda run: run.analysis["order_effect"].max_marginal_shift(),
            "interference": lambda run: run.analysis["interference"].max_abs_interference,
        },
    )


if __name__ == "__main__":
    raise SystemExit(main_for(SLUG, specific, module=MODULE))
