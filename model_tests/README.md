# Model tests: one file per model, every figure looked at

This directory is a test suite for the twenty-three models in the catalogue.  It lives **outside the
package** on purpose: everything here is about *using* `qcog`, so nothing in it is imported by the
library.  `tests/` (the package's own suite) checks the library; `model_tests/` checks each model's
behaviour, its numbers, its circuits, its classical comparison and its pictures.

## Run it

```bash
python model_tests/run_all.py                     # all 23 models: figures, reports, summary
python model_tests/run_all.py --fast              # fewer shots, no Aer comparison (about a minute)
python model_tests/run_all.py --models question-order auctions
python model_tests/run_all.py --no-save           # print only

python model_tests/module_a/02_question_order.py  # one model, with all of its checks
```

Everything it writes goes to `model_tests/out/`: one PNG per figure inside `<slug>/`, one
`<slug>/report.html` per model, and `SUMMARY.md` plus an `index.html` across all of them.  The exit
code is non-zero as soon as one check fails, so it can be used in a pipeline.

## What every model is checked for

`harness.py` holds the battery that is the same for all twenty-three models:

| Area | What is checked |
| --- | --- |
| Catalogue entry | slug, module letter, title, one-liner, cognitive question, mechanism, classical baseline, keywords, references (with a year), caveats, difficulty, the `describe()` block |
| Parameters | every parameter documented and labelled, bounds ordered and containing the default, unknown names refused, wrong types refused, unknown choices refused, `None` meaning the default, values outside the bounds clamped to them, and **every parameter changing the result** |
| The numbers | every joint distribution normalised, every probability in [0, 1], every marginal summing to one, the backend recorded, findings and summary present |
| Classical comparison | rows with a model value and a baseline value, a zero baseline carrying its explanation, a verdict, unknown `classical_options` reported rather than dropped, `classical=False` switching the chart off instead of leaving an empty one, and the comparison chart drawing exactly those rows |
| Execution | determinism, the sampled backend reproducible for a seed and within sampling error of the analytic one, a shot-based backend that really samples (or says why it cannot), the compiled Aer backends agreeing with the analytic result or declining with a reason, unknown backends refused, and the `sweeps` / `circuits` / `classical` / `circuit_style` flags honoured |
| Circuits | every design builds and is stable, qubits/gates/depth/measurements are sane, no `reset`, drawn in PNG, SVG and PDF, two orders of the same design using the same resources, and - where a design matches a sequential result - **the drawn circuit reproducing the model's own probabilities** |
| Figures | every figure is a graph rather than a panel of text, carries a description that names its numbers, has labels and no `nan`, separates its series by more than colour, is structurally consistent with the data it came from (bar heights, stacked segments, sweep lines, heatmap cells, Bloch vector, circuit gates), and survives the trip to a PNG that is not blank |

Each `module_*/NN_*.py` file then adds the checks that only make sense for its own model: the
order effect and the QQ equality for `question-order`, the conjunction excess for
`conjunction-fallacy`, Condorcet's theorem for `collective-judgment`, the cascade onset for
`information-cascades`, and so on.  Reading one file tells you what that model claims and how the
claim is tested.

## Reading the report

`out/<slug>/report.html` is meant to be opened rather than parsed: it shows every figure with its
description, the model-versus-classical table, the parameters, the sweeps, the circuit statistics,
the notes and the full list of checks.  If a figure is wrong, the check that would have caught it is
listed underneath it.

## Notes, and what they mean

A **note** is deliberately not a failure: it is a finding for a human.  The ones this suite records
at the time of writing:

* `question-order` - the decoherence sweep is flat at the declared default angles (the noiseless and
  the fully depolarised marginal coincide there, both 0.500).  The chart is correct and shows
  nothing; at asymmetric angles the same sweep rises from 0.289 to 0.461.
* `question-order` - `state_phase` moves nothing at the defaults, because at `state_angle = 0` it is a
  global phase.  Away from the defaults it changes the interference term while leaving the order
  effect alone, exactly as documented.
* `quantum-prisoners-dilemma`, `coordination-games` - the `entanglement` / `shared_context` parameter
  does not move any number these models report.  The "quantum" strategy is `Q = iZ`, a diagonal
  unitary, so the entangling gates `J` and `J†` cancel for every strategy pair; the EWL scheme needs
  `Q = U(pi/2, 0)`.  The drawn circuit shows the entangling gates, so the picture promises an effect
  the numbers do not have (see the note in each report).
* `negotiation-trust` - the shared-context sweep is computed with the suspicion dial at zero, so the
  chart is not the run; it is correct for its own design.
* `quantum-opinion-dynamics` - the sweep's `mean local entropy` averages over a different range of
  steps than the headline metric of the same name.
* figures whose distribution is a single outcome (`quantum-prisoners-dilemma`'s `C + C = 1.000`, the
  `combined` and `partial` conditions of `interference-paradox`): correct, but the chart says little.
* models that cannot run on a compiled backend because their state is mixed or their questions are
  POVMs (`decision-tree`, `belief-revision`, `sequential-categorization`, ...): the library refuses
  with a reason instead of guessing, and the report says so.
* `quantum-opinion-dynamics` and `polarization` evaluate a density matrix over the whole population,
  so a shot-based backend cannot apply to them; they record `exact` and explain why rather than
  reporting a backend that never ran.

## Adding a model

Copy the smallest file that resembles the new model, change `SLUG`, `MODULE`, `TITLE` and the body
of `specific()`, and run it.  The shared battery takes care of everything that is not
model-specific.
