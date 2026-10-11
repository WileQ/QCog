"""Superposition beliefs - module A, model 1 of 23.

A belief that answers one question as a fair coin can still be unresolved rather than
settled-but-unknown.  The checks separate the three states the model puts side by side - coherent,
partly decohered, and a classical mixture - read the second question that is the only thing able to
tell them apart, and pin down the point of the model: asking the first question destroys exactly
that coherence.

Run it on its own:

    python module_a/01_superposition_beliefs.py
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

SLUG = "superposition-beliefs"
MODULE = "A"
TITLE = "Superposition beliefs"


def specific(suite: Suite, result) -> None:
    """The checks that only make sense for this model."""
    model = suite.model
    states = result.analysis["states"]
    suite.check(
        "the three states are coherent, decohered and ignorant",
        set(states) == {"coherent", "decohered", "ignorance"},
        str(sorted(states)),
    )
    lengths = {name: float(np.linalg.norm(state.bloch_vector())) for name, state in states.items()}
    suite.close("the coherent belief is a pure state (Bloch length 1)", 1.0, lengths["coherent"])
    suite.check("the ignorance state has no coherence at all", lengths["ignorance"] < 1e-9, f"{lengths['ignorance']:.2e}")
    suite.check(
        "the decohered belief sits strictly between the two",
        1e-9 < lengths["decohered"] < 1 - 1e-9,
        f"{lengths['decohered']:.4f}",
    )
    separation = result.analysis["distinguishability"]
    suite.close("Q2 asked of the coherent state is certain", 1.0, separation["coherent"])
    suite.close("Q2 asked of the mixture is a fair coin", 0.5, separation["ignorance"])
    suite.check(
        "partial decoherence moves the state continuously towards the mixture",
        0.5 < float(separation["decohered"]) < 1.0,
        f"p(Q2=yes) = {float(separation['decohered']):.4f}",
    )
    suite.close(
        "the distinguishability index is the gap between the two extremes",
        float(separation["coherent"]) - float(separation["ignorance"]),
        float(result.analysis["distinguishability_index"]),
    )

    item = result.results["sequential_check"]
    first, second = item.order[0], item.order[1]
    suite.close(
        "the first question is a fair coin, so it cannot tell the states apart",
        0.5,
        max(item.marginal(first).values()),
    )
    suite.check(
        "asking the first question first destroys the coherence: the sequential design is uniform",
        all(abs(float(value) - 0.25) < 1e-9 for value in item.joint.values()),
        str({str(key): round(float(value), 4) for key, value in item.joint.items()}),
    )
    suite.check(
        "the separation is only visible when the second question is asked of the state directly",
        float(separation["coherent"]) > 0.9 and max(item.marginal(second).values()) < 0.6,
        f"asked directly {float(separation['coherent']):.3f}, after the first question "
        f"{max(item.marginal(second).values()):.3f}",
    )

    full = suite.run(sweeps=False, decoherence=1.0)
    suite.close(
        "full decoherence turns the belief into the classical mixture",
        0.0,
        float(np.linalg.norm(full.analysis["states"]["decohered"].bloch_vector())),
        tol=1e-9,
    )
    tuned = suite.run(sweeps=False, probe_angle=0.3)
    suite.check(
        "the probe angle changes what the second question sees",
        abs(float(tuned.analysis["distinguishability"]["decohered"]) - float(separation["decohered"])) > 1e-6,
        f"probe_angle=0.3 gives {float(tuned.analysis['distinguishability']['decohered']):.4f}",
    )
    for name in ("coherent", "decohered", "ignorance"):
        suite.check(f"the {name} state is drawn on its own Bloch sphere", f"state: {name}" in model.figures(result))
    check_sweep_anchor(
        suite,
        result,
        "decoherence",
        {"p(Q2=yes)": lambda run: run.analysis["distinguishability"]["decohered"]},
    )


if __name__ == "__main__":
    raise SystemExit(main_for(SLUG, specific, module=MODULE))
