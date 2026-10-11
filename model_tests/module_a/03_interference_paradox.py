"""Interference paradox - module A, model 3 of 23.

Deciding while the alternatives are still unresolved: the sure-thing principle is
violated because the two paths interfere.  The checks read the violation off the analysis, confirm
the classical part and the interference part add up, and pin the which-path visibility parameter to
the two extremes it interpolates between.

Run it on its own:

    python module_a/03_interference_paradox.py
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

SLUG = "interference-paradox"
MODULE = "A"
TITLE = "Interference paradox"


def specific(suite: Suite, result) -> None:
    """The checks that only make sense for this model."""
    model = suite.model
    report = result.analysis["interference"]
    suite.check("the interference decomposition covers two outcomes", len(report.totals) == 2, str(list(report.totals)))
    suite.check(
        "classical plus interference equals the total for every outcome",
        bool(report.check()),
        "; ".join(f"{o}: {float(report.direct[o]):.3f} {float(report.interference[o]):+.3f}" for o in report.totals),
    )
    suite.check(
        "the interference term is not zero",
        float(report.max_abs_interference) > 0.1,
        f"largest {float(report.max_abs_interference):.4f}",
    )

    combined = result.results["combined"]
    distinguished = result.results["distinguished"]
    paired = combined.marginal("decision")
    target = max(paired, key=paired.get)
    violation = float(result.analysis["sure_thing"].violation)
    suite.check("the sure-thing principle is violated", abs(violation) > 0.1, f"violation {violation:+.4f}")
    suite.close(
        "the violation is exactly the interference contribution at that outcome",
        float(report.interference[target]),
        violation,
        tol=1e-9,
    )
    suite.close(
        "distinguishing the paths shifts the decision probability by that much",
        float(combined.marginal("decision")[target]) - float(distinguished.marginal("decision")[target]),
        violation,
        tol=1e-9,
    )

    # visibility interpolates between the coherent case (0) and the classical one (1)
    suite.close(
        "at zero visibility the partial case is the coherent case",
        float(combined.marginal("decision")[target]),
        float(result.results["partial"].marginal("decision")[target]),
        tol=1e-9,
    )
    resolved = suite.run(sweeps=False, visibility=1.0)
    suite.close(
        "at full visibility the partial case is the distinguished case",
        float(distinguished.marginal("decision")[target]),
        float(resolved.results["partial"].marginal("decision")[target]),
        tol=1e-9,
    )
    middle = suite.run(sweeps=False, visibility=0.5)
    suite.check(
        "the decision probability moves monotonically with the visibility",
        float(distinguished.marginal("decision")[target])
        - 1e-9
        <= float(middle.results["partial"].marginal("decision")[target])
        <= float(combined.marginal("decision")[target]) + 1e-9,
        f"0.0: {float(combined.marginal('decision')[target]):.3f}, 0.5: "
        f"{float(middle.results['partial'].marginal('decision')[target]):.3f}, 1.0: "
        f"{float(distinguished.marginal('decision')[target]):.3f}",
    )
    check_sweep_anchor(
        suite,
        result,
        "visibility",
        {
            "target probability": lambda run: max(run.results["partial"].marginal("decision").values()),
            "p(yes)": lambda run: run.results["partial"].marginal("decision")["yes"],
        },
    )
    check_sweep_anchor(
        suite,
        result,
        "relative phase",
        {"interference": lambda run: run.analysis["interference"].max_abs_interference},
    )
    for name in ("probabilities: combined", "probabilities: distinguished", "probabilities: partial"):
        suite.check(f"the {name!r} figure is drawn", name in model.figures(result))


if __name__ == "__main__":
    raise SystemExit(main_for(SLUG, specific, module=MODULE))
