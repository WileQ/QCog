"""Disjunction effect - module B, model 7 of 23.

Deciding when the opponent's move is unknown.  The checks compare the coherent representation
with the branch mixture that the sure-thing principle demands, and confirm the model's own honesty
claim: the bipartite (world x action) representation reproduces the classical value exactly.

Run it on its own:

    python module_b/07_disjunction_effect.py
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

SLUG = "disjunction-effect"
MODULE = "B"
TITLE = "Disjunction effect"


def specific(suite: Suite, result) -> None:
    """The checks that only make sense for this model."""
    model = suite.model
    values = result.analysis["values"]
    suite.check(
        "an unresolved alternative changes the decision",
        float(values["holistic unknown"]) > float(values["branch mixture"]) + 0.1,
        f"{float(values['holistic unknown']):.3f} against {float(values['branch mixture']):.3f}",
    )
    suite.close(
        "the bipartite representation reproduces the sure-thing value",
        float(values["branch mixture"]),
        float(values["entangled unknown"]),
        tol=1e-9,
    )
    suite.close(
        "knowing the opponent's move does not change the bipartite prediction",
        float(values["entangled unknown"]),
        float(values["entangled after knowing"]),
        tol=1e-9,
    )
    suite.check(
        "the model quantifies how exactly the bipartite representation matches",
        float(result.analysis["representation_guarantee"]) < 1e-9,
        f"{float(result.analysis['representation_guarantee']):.2e}",
    )
    suite.check(
        "the comparison row says a classical representation cannot show the effect",
        any(
            "bipartite" in str(row.get("note", "")) or "sure-thing" in str(row.get("note", ""))
            for row in result.classical_comparison
        ),
        str([row.get("metric") for row in result.classical_comparison]),
    )
    phased = suite.run(sweeps=False, relative_phase=0.0)
    suite.check(
        "the phase controls whether the effect is constructive or destructive",
        abs(float(phased.analysis["values"]["holistic unknown"]) - float(values["holistic unknown"])) > 1e-6,
        f"phase 0 gives {float(phased.analysis['values']['holistic unknown']):.3f}",
    )
    check_sweep_anchor(
        suite,
        result,
        "relative phase",
        {
            "unresolved": lambda run: run.analysis["values"]["holistic unknown"],
            "sure-thing prediction": lambda run: run.analysis["values"]["branch mixture"],
        },
    )
    for name in ("probabilities: unresolved", "probabilities: resolved_mixture", "probabilities: entangled_unknown"):
        suite.check(f"the {name!r} figure is drawn", name in model.figures(result))


if __name__ == "__main__":
    raise SystemExit(main_for(SLUG, specific, module=MODULE))
