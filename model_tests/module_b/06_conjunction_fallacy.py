"""Conjunction fallacy - module B, model 6 of 23.

The Linda problem.  The checks measure the conjunction excess over the classical bound, confirm
that the bound is min(p(A), p(B)) as the model claims, confirm that the same model built from
independent attributes obeys the bound exactly, and read the phase that produces the effect.

Run it on its own:

    python module_b/06_conjunction_fallacy.py
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

SLUG = "conjunction-fallacy"
MODULE = "B"
TITLE = "Conjunction fallacy"


def specific(suite: Suite, result) -> None:
    """The checks that only make sense for this model."""
    model = suite.model
    conjunction = result.analysis["conjunction"]
    suite.check(
        "the conjunction is judged more probable than its least likely part",
        float(conjunction["excess"]) > 0.0,
        f"p(both) {float(conjunction['p_conjunction']):.3f} against the bound {float(conjunction['classical_bound']):.3f}",
    )
    suite.check("the model reports the bound as violated", bool(conjunction["violated"]))
    suite.close(
        "the excess is the conjunction minus the bound",
        float(conjunction["p_conjunction"]) - float(conjunction["classical_bound"]),
        float(conjunction["excess"]),
        tol=1e-5,          # the analysis reports these numbers rounded to six decimals
    )
    independent = result.analysis["independent_check"]
    suite.check(
        "with independent attributes the same model obeys the bound",
        float(independent["excess"]) <= 1e-9,
        f"excess {float(independent['excess']):+.4f}",
    )
    suite.check(
        "the bound really is the smaller of the two marginals",
        float(conjunction["classical_bound"])
        <= min(float(independent["p(feminist)"]), float(independent["p(bank teller)"])) + 1e-5,
        f"bound {float(conjunction['classical_bound']):.3f} against "
        f"{float(independent['p(feminist)']):.3f} and {float(independent['p(bank teller)']):.3f}",
    )

    neutral = suite.run(sweeps=False, relative_phase=0.0)
    suite.check(
        "without the phase the fallacy disappears",
        float(neutral.analysis["conjunction"]["excess"]) <= 1e-9,
        f"excess {float(neutral.analysis['conjunction']['excess']):+.4f}",
    )
    strongest = suite.run(sweeps=False, relative_phase=np.pi)
    suite.check(
        "the largest excess appears at the phase the model declares",
        abs(float(strongest.analysis["conjunction"]["excess"]) - float(conjunction["excess"])) < 1e-6,
        f"pi gives {float(strongest.analysis['conjunction']['excess']):+.4f}, the default {float(conjunction['excess']):+.4f}",
    )
    suite.check(
        "the amplitude dial moves the conjunction",
        abs(
            float(suite.run(sweeps=False, feminist_amplitude=0.6).analysis["conjunction"]["p_conjunction"])
            - float(conjunction["p_conjunction"])
        )
        > 1e-6,
        "feminist_amplitude=0.6",
    )
    check_sweep_anchor(
        suite,
        result,
        "relative phase",
        {
            "p(conjunction)": lambda run: run.analysis["conjunction"]["p_conjunction"],
            "classical bound": lambda run: run.analysis["conjunction"]["classical_bound"],
            "excess": lambda run: run.analysis["conjunction"]["excess"],
            "p(bank teller)": lambda run: run.analysis["independent_check"]["p(bank teller)"],
        },
    )
    for name in ("probabilities: feminist_then_teller", "probabilities: teller_only"):
        suite.check(f"the {name!r} figure is drawn", name in model.figures(result))


if __name__ == "__main__":
    raise SystemExit(main_for(SLUG, specific, module=MODULE))
