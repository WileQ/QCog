"""Question order - module A, model 2 of 23.

The flagship model: two incompatible questions in both orders.  The checks pin down the
order effect, the law-of-total-probability violation it implies, the parameter-free QQ equality,
the classical limit reached by making the questions unsharp, and the fact that the decoherence
sweep and the run agree (decoherence really does move the order effect here).

Run it on its own:

    python module_a/02_question_order.py
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

SLUG = "question-order"
MODULE = "A"
TITLE = "Question order"


def specific(suite: Suite, result) -> None:
    """The checks that only make sense for this model."""
    model = suite.model
    report = result.analysis["order_effect"]
    shift = float(report.max_marginal_shift())
    suite.check("asking the questions in two orders changes the first answer", shift > 0.1, f"largest shift {shift:.4f}")
    suite.close("the total variation between the orders is the shift it reports", shift, float(report.total_variation))
    suite.check("the questions are genuinely incompatible", float(report.incompatibility) > 0.5, f"{float(report.incompatibility):.4f}")

    qq = result.analysis["qq_equality"]
    suite.check("the QQ equality applies to two dichotomous questions", bool(qq.applicable))
    suite.close("QQ holds: agreement is the same in both orders", float(qq.agreement_first_order), float(qq.agreement_second_order), tol=1e-9)
    suite.close("the QQ difference is zero", 0.0, float(qq.difference), tol=1e-12)

    violations = result.analysis["ltp"]
    suite.check("the law of total probability is listed for each question", len(violations) >= 1, f"{len(violations)} rows")
    suite.check(
        "the largest violation is the order effect the model reports",
        abs(max(abs(float(row.direct - row.indirect)) for row in violations) - shift) < 1e-9,
        f"{max(abs(float(row.direct - row.indirect)) for row in violations):.4f}",
    )

    unsharp = suite.run(sweeps=False, sharpness=0.0)
    suite.check(
        "unsharp questions restore the classical limit (no order effect)",
        float(unsharp.analysis["order_effect"].max_marginal_shift()) < 1e-6,
        f"{float(unsharp.analysis['order_effect'].max_marginal_shift()):.2e}",
    )
    sharp = suite.run(sweeps=False, sharpness=1.0)
    suite.check(
        "sharp questions give the largest order effect",
        float(sharp.analysis["order_effect"].max_marginal_shift())
        >= float(unsharp.analysis["order_effect"].max_marginal_shift()),
    )

    # the documented asymmetry: the phase moves the interference, not the first marginals
    turned = suite.run(sweeps=False, state_angle=0.6, state_phase=1.2)
    plain = suite.run(sweeps=False, state_angle=0.6, state_phase=0.0)
    suite.check(
        "the belief phase changes the interference term",
        abs(
            float(turned.analysis["interference"].max_abs_interference)
            - float(plain.analysis["interference"].max_abs_interference)
        )
        > 1e-6,
        f"{float(plain.analysis['interference'].max_abs_interference):.4f} -> "
        f"{float(turned.analysis['interference'].max_abs_interference):.4f}",
    )
    suite.close(
        "the belief phase leaves the order effect alone, as documented",
        float(plain.analysis["order_effect"].max_marginal_shift()),
        float(turned.analysis["order_effect"].max_marginal_shift()),
        tol=1e-9,
    )

    # decoherence is applied between the questions, so at asymmetric angles it moves the effect
    asymmetric = suite.run(sweeps=False, angle_a=0.4, angle_b=1.0, decoherence=0.6)
    suite.check(
        "decoherence between the questions changes the order effect at asymmetric angles",
        abs(
            float(asymmetric.analysis["order_effect"].max_marginal_shift())
            - float(suite.run(sweeps=False, angle_a=0.4, angle_b=1.0, decoherence=0.0).analysis["order_effect"].max_marginal_shift())
        )
        > 1e-3,
        f"{float(asymmetric.analysis['order_effect'].max_marginal_shift()):.4f}",
    )
    sweep_table = result.analysis["sweeps"]["decoherence"].table
    suite.note(
        "at the declared default angles the decoherence sweep is flat",
        "the noiseless and the fully depolarised marginal coincide there (both 0.500), so the chart is "
        "correct but says nothing; at asymmetric angles it rises from "
        f"{float(sweep_table['order effect'].iloc[0]):.3f} to {float(sweep_table['order effect'].iloc[-1]):.3f}",
    )
    for metric, resolver in (
        ("order effect", lambda run: run.analysis["order_effect"].max_marginal_shift()),
        ("interference", lambda run: run.analysis["interference"].max_abs_interference),
    ):
        check_sweep_anchor(suite, result, "sharpness", {metric: resolver})
    check_sweep_anchor(
        suite,
        result,
        "decoherence",
        {"order effect": lambda run: run.analysis["order_effect"].max_marginal_shift()},
    )
    for name in ("order effect", "interference", "probabilities: A_then_B", "probabilities: B_then_A"):
        suite.check(f"the {name!r} figure is drawn", name in model.figures(result))


if __name__ == "__main__":
    raise SystemExit(main_for(SLUG, specific, module=MODULE))
