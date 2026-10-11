"""Semantic ambiguity - module B, model 10 of 23.

The context fixes which sense of an ambiguous word is meant - but only before the sense has
been resolved.  The checks read the three sense distributions, the ordering asymmetry a
prior-updating account cannot produce, and the incompatibility that drives it.

Run it on its own:

    python module_b/10_semantic_ambiguity.py
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

SLUG = "semantic-ambiguity"
MODULE = "B"
TITLE = "Semantic ambiguity"


def specific(suite: Suite, result) -> None:
    """The checks that only make sense for this model."""
    model = suite.model
    distributions = result.analysis["sense_distributions"]
    suite.check(
        "the model reports the sense distribution in each context",
        {"neutral", "context", "opposite"} <= set(distributions),
        str(sorted(distributions)),
    )
    neutral = distributions["neutral"]
    suite.check(
        "without context one sense dominates",
        max(neutral.values()) > 0.6,
        str({key: round(float(value), 3) for key, value in neutral.items()}),
    )
    broad = suite.run(sweeps=False, context_scope="broad")
    winner = max(distributions["neutral"], key=lambda key: distributions["neutral"][key])
    suite.check(
        "the context pushes the dominant sense down",
        float(distributions["context"][winner]) < float(distributions["neutral"][winner]),
        f"{float(distributions['neutral'][winner]):.3f} -> {float(distributions['context'][winner]):.3f}",
    )
    suite.check(
        "the opposite context brings the dominant sense back towards its no-context value",
        float(distributions["context"][winner])
        < float(distributions["opposite"][winner])
        <= float(distributions["neutral"][winner]) + 1e-9,
        f"no context {float(distributions['neutral'][winner]):.3f}, sentence {float(distributions['context'][winner]):.3f}, "
        f"opposite {float(distributions['opposite'][winner]):.3f}",
    )
    suite.check("the sense question is incompatible with the context", float(result.analysis["incompatibility"]) > 0.1, f"{float(result.analysis['incompatibility']):.4f}")

    resolved = result.results["sense resolved first"]
    suite.check(
        "the resolved-sense design asks the sense question twice",
        len(resolved.order) == 2,
        str(resolved.order),
    )
    suite.check(
        "asking the sense question again always reproduces the same sense",
        all(str(key[0]) == str(key[1]) for key in resolved.joint),
        str({str(key): round(float(value), 4) for key, value in resolved.joint.items()}),
    )
    suite.check(
        "the repeated question repeats the first distribution exactly",
        all(
            abs(float(resolved.marginal(resolved.order[0])[sense]) - float(resolved.marginal(resolved.order[1])[sense])) < 1e-9
            for sense in resolved.outcomes[resolved.order[0]]
        ),
        str({key: round(float(value), 4) for key, value in resolved.marginal(resolved.order[0]).items()}),
    )
    suite.check(
        "the resolved-sense figure is drawn separately",
        "probabilities: sense resolved first" in model.figures(result),
    )
    narrow_sense = max(distributions["context"], key=lambda key: distributions["context"][key])
    broad_sense = max(broad.analysis["sense_distributions"]["context"], key=lambda key: broad.analysis["sense_distributions"]["context"][key])
    suite.check(
        "the context scope chooses which sense the context selects",
        narrow_sense != broad_sense,
        f"the narrow context selects {narrow_sense!r}, the broad one {broad_sense!r}",
    )


if __name__ == "__main__":
    raise SystemExit(main_for(SLUG, specific, module=MODULE))
