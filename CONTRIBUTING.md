# Contributing to Q-Cog

Thanks for considering it. This project is deliberately built so that contributions are easy to make *well*.

## Setting up

```bash
python -m pip install -e ".[app,dev]"
python -m pytest -q            # 121 tests, about one minute
qcog evidence                  # what the library checks about itself
```

## Adding a model

1. Pick the module (`models/module_a.py` ... `module_d.py`).
2. Subclass `Model`, or `OrderBasedModel` from `models/shared.py` if the design is "two questions in both
   orders" - that base class gives you both-orders simulation, the interference decomposition, the QQ equality,
   the law-of-total-probability test, the classical comparison and the standard sweeps for free.
3. Declare the metadata: `slug`, `title`, `module`, `one_liner`, `cognitive_question`, `quantum_mechanism`,
   `classical_baseline`, `references`, `caveats`, `keywords`, `difficulty`.
4. Declare the parameters *once*, as `ParamSpec` objects. The CLI, the explorer, the fitting layer and the
   generated documentation are all derived from them.
5. Register the class in `registry.py` and in `models/__init__.py`.
6. Add a test asserting the model's central claim, and a row in the test that runs every model.

**Required, not optional:** at least one reference, and at least one caveat that says what the model does not
establish. The test `test_catalogue_is_complete_and_well_formed` enforces this.

## The two things every model must produce

1. **A classical comparison.** `comparison_rows(result)` returns rows of `metric`, `model`, `baseline` and
   `note`. The default implementation already covers the analysis dictionaries in the catalogue (order effects,
   interference, conjunction bounds, entanglements, game payoffs, network statistics, Condorcet, cascades,
   plurality ...), so most models need nothing; override it when a sharper pair of numbers exists.
   If your baseline has knobs of its own, declare them in `classical_option_names` and honour them in
   `comparison_rows(result, options)` (recompute the rows rather than relabelling them); anything you cannot honour
   is reported as ignored, which is better than silently dropping it.

2. **A circuit view, or a reason.** `circuit_designs(result)` returns named Qiskit circuits. The library scans
   the model's own `build` output for a state and the questions that act on it, draws both orders of a
   two-question design, uses the entangling circuit for a game, falls back to the sharp limit of an unsharp
   question, and reports in `circuit_notes(result)` why nothing was drawn (a mixed state, an unsharp question
   with no sharp counterpart). Add `extra_circuit_designs(result)` for a mechanism the scan cannot see.

Tests in `tests/test_circuits_and_comparison.py` enforce both, along with every parameter of every model being
changeable from Python, the circuit image options (`style`, `scale`, `fold`, `format`, `circuit_image(...)`),
and the requirement that series styles stay distinguishable.

## Before you finish

```bash
python -m pytest -q                     # the suite, including the notebook and the docs checks
python scripts/check_figures.py         # every visualization renders, with its description
python scripts/check_docs.py            # the documentation matches the code
python scripts/generate_docs.py         # regenerate docs/models.generated.md
```

`check_figures.py` is the one to run when you add or change a chart: it builds every figure function on synthetic
input *and* every model's figure set, writes them out if you pass `--out`, and exits non-zero if something is
broken or has lost its accessible description.

## House rules

* **Every claim must be checked by code.** If a model asserts a property, add it to `selftest.py` or to the test
  suite. If it cannot be verified, write it as a caveat instead.
* **Report the classical baseline.** A model that does not say what a classical account predicts is incomplete.
* **No figure without a description.** Use `qcog.viz` (which attaches the description to the figure) rather
  than calling matplotlib directly.
* **Never name a model parameter `noise`.** That keyword belongs to the execution layer; use `decoherence`.
* **Keep the accessibility practices** described in `docs/accessibility.md`: colour-blind-safe palette, no
  information by colour alone, data tables for charts, semantic HTML in reports.
* **Honesty over impressiveness.** If a classical model fits as well, say so - the information criteria in
  `qcog.core.fitting` exist for that.
* Numbers in prose must come from the code. `qcog.explain` and the findings strings are generated from computed
  values precisely so that they cannot drift.

## Style

* Python 3.10+, `from __future__ import annotations`, type hints on public functions.
* Docstrings explain *why*: what the object means scientifically, and which convention is being used.
* Keep dependencies out: the core library needs nothing beyond the declared requirements.
* Tests should be deterministic: always pass a `seed` when using a sampled backend.

## Reporting a problem

Open an issue with the model slug, the parameters, the backend, and the output of `qcog evidence`. If a
self-check fails, that is the most useful bug report possible.