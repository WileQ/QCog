# Contributing to Q-Cog

Thanks for considering it. This project is deliberately built so that contributions are easy to make *well*.

## Setting up

```bash
python -m pip install -e ".[app,dev]"
python -m pytest -q            # 83 tests, about one minute
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

## House rules

* **Every claim must be checked by code.** If a model asserts a property, add it to `selftest.py` or to the test
  suite. If it cannot be verified, write it as a caveat instead.
* **Report the classical baseline.** A model that does not say what a classical account predicts is incomplete.
* **No figure without a description.** Use `qcog.viz` (which attaches the description to the figure) rather
  than calling matplotlib directly.
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