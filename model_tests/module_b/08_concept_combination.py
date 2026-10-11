"""Concept combination - module B, model 8 of 23.

'Pet fish' is neither a typical pet nor a typical fish.  The checks confirm the emergence over
the parts that the weighted-average baseline forbids, that the parts' average is what the model
says it is, and that the combined state is genuinely entangled.

Run it on its own:

    python module_b/08_concept_combination.py
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

SLUG = "concept-combination"
MODULE = "B"
TITLE = "Concept combination"


def specific(suite: Suite, result) -> None:
    """The checks that only make sense for this model."""
    model = suite.model
    typicality = result.analysis["typicality"]
    parts = (float(typicality["pet"]), float(typicality["fish"]))
    suite.check(
        "the combination is more typical than either part",
        float(typicality["combined"]) > max(parts),
        f"combined {float(typicality['combined']):.3f} against parts {parts[0]:.3f} and {parts[1]:.3f}",
    )
    suite.close("the parts' baseline is their average", sum(parts) / 2, float(typicality["average"]))
    suite.close(
        "emergence is the combination minus the better part",
        float(typicality["combined"]) - max(parts),
        float(typicality["emergence"]),
    )
    suite.check(
        "the weighted-average baseline cannot reach the combination",
        float(typicality["average"]) < float(typicality["combined"]),
        f"{float(typicality['average']):.3f} < {float(typicality['combined']):.3f}",
    )
    entanglement = result.analysis["entanglement"]
    suite.check(
        "the combined concept is not a product of its parts",
        not bool(entanglement.is_product) and float(entanglement.concurrence) > 0.0,
        f"concurrence {float(entanglement.concurrence):.4f}, entropy {float(entanglement.entropy_of_entanglement):.4f}",
    )
    other = suite.run(sweeps=False, feature="care")
    suite.check(
        "asking about a different feature gives a different typicality",
        abs(float(other.analysis["typicality"]["combined"]) - float(typicality["combined"])) > 1e-6,
        f"fins {float(typicality['combined']):.3f} against care {float(other.analysis['typicality']['combined']):.3f}",
    )
    check_sweep_anchor(
        suite,
        result,
        "combination strength",
        {
            "combination": lambda run: run.analysis["typicality"]["combined"],
            "average of parts": lambda run: run.analysis["typicality"]["average"],
            "concurrence": lambda run: run.analysis["entanglement"].concurrence,
        },
    )
    for name in ("probabilities: pet_alone", "probabilities: fish_alone", "probabilities: combined"):
        suite.check(f"the {name!r} figure is drawn", name in model.figures(result))


if __name__ == "__main__":
    raise SystemExit(main_for(SLUG, specific, module=MODULE))
