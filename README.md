# Q-Cog

**A Qiskit-native library for simulating, explaining and visualising quantum-probability models of human
cognition, judgement, decision making and collective behaviour.**

Q-Cog is a Python package *and* an interactive platform. The library provides the scientific and computational
foundation; the platform makes it usable by researchers, students and anyone curious about how people decide.
Twenty-three documented models are included, each of which

* uses **Qiskit** for every quantum calculation (statevector, density matrix, Aer circuits),
* reports the **classical baseline** for the same situation,
* states its own **caveats** - what it does *not* claim,
* and produces **accessible** figures, each carrying a plain-language description.

> This effort is supported by Qollab & IonQ.

---

## Contents

1. [What it does](#what-it-does)
2. [Install](#install)
3. [Quick start](#quick-start)
4. [The model catalogue](#the-model-catalogue)
5. [Command line](#command-line)
6. [Interactive explorer](#interactive-explorer)
7. [Reports you can publish](#reports-you-can-publish)
8. [Why the numbers can be trusted](#why-the-numbers-can-be-trusted)
9. [Accessibility](#accessibility)
10. [Classical baselines are not decoration](#classical-baselines-are-not-decoration)
11. [Repository layout](#repository-layout)
12. [Documentation](#documentation)
13. [Extending it](#extending-it)
14. [Scope, honesty and limitations](#scope-honesty-and-limitations)
15. [Licence and attribution](#licence-and-attribution)

---

## What it does

Most quantum-computing libraries build circuits and show outputs. Most quantum-cognition work targets a single
model, experiment or dataset. Q-Cog provides **one workflow for all of them**:

| Step | What you do | What Q-Cog gives you |
| --- | --- | --- |
| Choose a phenomenon | Decision paradox, question order, concept combination, group consensus ... | a registry of 23 models in four thematic modules |
| Configure the model | states, questions, measurement order, parameters | one parameter schema that drives the CLI, the UI, the fitting layer and the docs |
| Execute with Qiskit | exact, sampled, or compiled circuits | four interchangeable backends, cross-checked against each other |
| Visualise | state, interference, networks, trajectories | accessible figures with mandatory text descriptions |
| Explain | classical baseline, predictions, limitations | rule-based text generated from the numbers, plus an explicit caveat list |

The defining feature is that **every prediction is paired with what a classical account would say**, and every
model ships with the checks that decide between them (the law of total probability, the QQ equality, the
classical conjunction bound, Condorcet's theorem, the classical Nash equilibrium, ...).

## Install

```bash
git clone <this repository> qcog && cd qcog
python -m pip install -e .                 # library only
python -m pip install -e ".[app,dev]"      # + interactive explorer + tests
```

Requires Python 3.10+ and `qiskit>=2.0`, `qiskit-aer`, `numpy`, `scipy`, `pandas`, `matplotlib`.
Everything runs offline on the free local simulator; no account, token or network access is needed.

Without installing, put `src` on the path (all examples do this automatically) and use the module form of the
command line:

```bash
PYTHONPATH=src python -m qcog list        # every command works this way
```

```python
import sys; sys.path.insert(0, "src")     # in scripts and notebooks
```

## Quick start

```python
import sys; sys.path.insert(0, "src")

from qcog import basis_state, question, run_sequence, explain_order_effect, order_effect, qq_equality

belief = basis_state(0, 2)                       # a definite belief: |0>
question_a = question(0.955, 0.0, "A")           # "how satisfied are you?"
question_b = question(3.14159 / 2, 0.0, "B")     # "will you vote?"

forward = run_sequence(belief, [question_a, question_b])
backward = run_sequence(belief, [question_b, question_a])

report = order_effect(forward, backward, "A", "B")
print(report.summary())
print(explain_order_effect(report))
print(qq_equality(forward, backward, "A", "B").summary())
```

```
Order effect of 0.2888 in p(A = most likely outcome): the A-then-B and B-then-A orders disagree
(TV distance 0.2888, question incompatibility 1.6337).
p(A = yes) is 0.789 when A is asked first and 0.500 when B is asked first, so asking the other
question first makes this answer lower by 0.289. ... A classical model with one joint distribution
over the two answers would force this difference to be zero.
QQ equality holds: same-answer agreement is 0.9082 in one order and 0.9082 in the other ...
```

Or run a complete model, and get a report:

```python
from qcog import get_model, write_model_report

model = get_model("question-order")
result = model.run()                       # exact
result = model.run(backend="aer", shots=20000, seed=7)     # on a compiled circuit
result = model.run(decoherence=0.15)       # with noise
print(result.summary)
for finding in result.findings: print("-", finding)

write_model_report(model, result, "out/")  # accessible HTML + PNGs
```

## The model catalogue

Twenty-three models in four modules (module letters follow the project's original proposal).

### Module A - Individual cognition and decision making

| Model | Question it explores |
| --- | --- |
| `superposition-beliefs` | Two beliefs with identical answers to one question, separated by the next |
| `question-order` | Why the answer depends on which question was asked first (QQ equality, interference) |
| `interference-paradox` | Distinguishing alternatives destroys the interference that decided them |
| `decision-tree` | Does the order of information change the final decision? |
| `belief-revision` | Evidence accumulation, weak evidence, and framing |

### Module B - Logic, concepts and memory

| Model | Question it explores |
| --- | --- |
| `conjunction-fallacy` | The Linda problem: conjunction judged more probable than its conjunct |
| `disjunction-effect` | Cooperation under an unresolved opponent, and what representation it requires |
| `concept-combination` | Emergent and destroyed features in combined concepts ("pet fish") |
| `associative-memory` | Competing retrieval cues interfere with each other |
| `semantic-ambiguity` | Context fixes a word's sense - but only before the sense is resolved |
| `sequential-categorization` | Repeated labels stabilise a judgement (a Zeno effect in categorisation) |
| `conceptual-entanglement` | Joint structure without interference, quantified by concurrence |
| `contradictory-beliefs` | Self-report and behaviour that no single joint distribution can host |

### Module C - Strategic decisions and game theory

| Model | Question it explores |
| --- | --- |
| `quantum-prisoners-dilemma` | The Eisert-Wilkert-Lewenstein scheme: does entanglement change the game? |
| `negotiation-trust` | Trust as correlation; agreement as a function of shared context |
| `evolution-of-cooperation` | Replicator dynamics on quantum-induced payoffs: where does cooperation survive? |
| `auctions` | Value and bid are different questions: order, rival information, winner's curse |
| `coordination-games` | Conventions without communication |

### Module D - Groups, social networks and collective behaviour

| Model | Question it explores |
| --- | --- |
| `quantum-opinion-dynamics` | Opinions as coherent exchanges along a network |
| `polarization` | Information bubbles, cross-talk, and what decoherence does to them |
| `collective-judgment` | The wisdom of crowds and how correlation destroys it |
| `information-cascades` | Private signal or the group first? Cascades and order effects |
| `deliberation-voting` | Does deliberation change the winner? Plurality and Condorcet comparisons |

Every model also declares: the cognitive question, the quantum mechanism, the classical baseline, references,
keywords, a difficulty rating, its parameters (with bounds and which are fittable), and caveats.

## Command line

```bash
qcog list                                  # every model (or: PYTHONPATH=src python -m qcog list)
qcog info question-order                   # full documentation for one model
qcog run question-order                    # run it, print findings and tables
qcog run question-order --backend aer --shots 20000 --seed 7
qcog run conjunction-fallacy --param teller_angle=1.6 --format json
qcog sweep question-order                  # the model's own parameter sweeps
qcog fit question-order data.json          # fit to observed frequencies (JSON or CSV)
qcog report --out out/qcog_report.html --markdown
qcog evidence                              # what the library checks about itself
qcog backends                              # the available execution backends
```

## Interactive explorer

```bash
streamlit run app/streamlit_app.py
```

The explorer is generated from the model metadata: choose a model, adjust its parameters, pick a backend, and
read the findings, the classical comparison, the explanation of the order effect or the interference term, the
figures (each with its text description), and the data tables. Accessibility options - dark background, larger
type, high contrast, text-only mode - are one checkbox away, and a per-run HTML report can be downloaded.

## Reports you can publish

```python
from qcog import write_catalogue_report, write_markdown_report

write_catalogue_report("out/qcog_report.html")   # all 23 models + the self-check table
write_markdown_report("out/qcog_report.md")
```

The HTML report is self-contained: no JavaScript, no network, no external assets. Every figure has alt text, a
caption, an expandable long description and the data table it was drawn from; the page has a skip link,
semantic landmarks, a heading hierarchy and visible focus outlines. It is designed to be published as-is on a
project page.

## Why the numbers can be trusted

`qcog evidence` runs the library's checks and prints them. The same table appears in the generated report, so a
failure is visible rather than silent. The checks include:

* the interference decomposition reproduces the total probability exactly;
* the **QQ equality** - a parameter-free prediction - holds to machine precision;
* commuting questions give a **zero** order effect; uninformative (zero-sharpness) questions also do;
* the **compiled prefix circuits** reproduce the analytic engine to ~1e-16, and shot-based circuit execution
  agrees within five standard errors;
* decoherence of the belief removes the order effect exactly, while decoherence *between* questions removes the
  conditional structure (a distinction the checks make explicit);
* the depolarising channel is trace-preserving in every dimension and has the correct limits;
* maximal entanglement makes mutual cooperation payoff-dominant in the quantum prisoner's dilemma, the
  unentangled game reproduces the classical payoff matrix exactly, and entanglement measures are calibrated
  (Bell concurrence 1, product concurrence 0);
* the conjunction bound is satisfied exactly for independent attributes and violated for incompatible ones;
* perfectly correlated judges lose the Condorcet advantage;
* all 23 models run and return normalised distributions.

Two execution paths are implemented from *one* definition - projectors - and compared: the analytic engine
(Qiskit `Statevector`/`DensityMatrix`) and compiled circuits on `qiskit-aer`. Agreement between them is what
makes the compilation, the collapse semantics and the frame bookkeeping trustworthy. The test suite
(83 tests) additionally covers states, measurements, channels, backends, analysis, baselines, fitting, the CLI,
the reports and every model's central claims.

Writing those checks paid for itself: it found a depolarising channel that was not trace-preserving in the
standard sense, a player-ordering error that transposed the game-theory payoff table, and two circuit-compilation
bugs (measuring qubits a question does not involve, and assuming every answer is rank-1).

## Accessibility

Industry best practices are followed by construction, not by convention:

* **Figures** carry a plain-language description on the figure object itself, so it is impossible to export a
  chart without one; the description names the axes, the series and the values that matter.
* **Data tables** accompany every chart, so the numbers are reachable without seeing the image.
* **Colour**: the colour-blind-safe Okabe-Ito palette; no information is carried by colour alone (bars are
  labelled, deltas are printed).
* **Keyboard and screen readers**: semantic HTML with a skip link, headings in order, table captions and
  `scope` attributes, visible `:focus-visible` outlines, and Streamlit's labelled native controls.
* **Reading comfort**: a dark-background option, a larger-type option, a high-contrast option and a text-only
  mode that removes images entirely.
* **Plain language**: explanations are rule-based sentences generated from the numbers, not decorative prose.

Run `qcog report` and open the HTML: the accessibility features are inspectable in a text editor.

## Classical baselines are not decoration

Every model answers the question "what would a classical account of the same situation say?", and where a
classical model *is* adequate, Q-Cog says so. Two examples of that policy in action:

* `conceptual-entanglement` reports concurrence and an excess over the independence prediction, and notes that
  the joint probability never exceeds the marginals - the effect is correlation, *not* interference, and the
  library refuses to conflate the two.
* `question-order` reports the QQ difference with its standard error on shot-based runs, so a difference that is
  within noise is not presented as a result.

The `disjunction-effect` model goes further and *proves* something about representations: any bipartite
"world x action" state - however entangled - satisfies the sure-thing principle exactly, so the disjunction
effect requires a representation in which the unresolved alternatives contribute coherent amplitudes to the
same decision state. That check is part of the self-test table.

## Repository layout

```
src/qcog/
  core/          states, measurements (projective and POVM), channels, engine, results,
                 classical baselines, metrics, fitting, parameter schemas
  backends/      exact, sampled, Aer (compiled circuits), exact-compiled, optional IonQ
  circuits/      projector -> Qiskit circuit compilation (whole interview and per-branch)
  analysis/      interference decomposition, order effects, QQ equality, entanglement, sweeps
  experiments/   reusable designs: two questions in both orders, quantum games, populations
  models/        module_a.py ... module_d.py (23 models) + the model base class
  viz/           accessible figures and their text descriptions
  explain.py     numbers -> sentences (rule based, testable)
  report.py      self-contained accessible HTML / Markdown reports
  selftest.py    the evidence table
  registry.py    the catalogue
  data/          illustrative datasets with provenance notes
app/             Streamlit explorer
examples/        five runnable scripts, from five minutes to the full catalogue
tests/           83 tests
docs/            project page, methodology, accessibility, demo script, submission notes
```

## Documentation

* [`docs/project_page.md`](docs/project_page.md) - the project page: what it does, what problem it explores,
  how to run it, how to replicate or extend it.
* [`docs/methodology.md`](docs/methodology.md) - the mathematics, the conventions, and the statistical tests.
* [`docs/accessibility.md`](docs/accessibility.md) - the accessibility practices and how to verify them.
* [`docs/limitations.md`](docs/limitations.md) - what the library does not claim, model by model.
* [`docs/demo_video_script.md`](docs/demo_video_script.md) - a 2-3 minute demo script.
* [`docs/qollab_runbook.md`](docs/qollab_runbook.md) - running and publishing on Qollab, including IonQ.
* [`docs/submission_checklist.md`](docs/submission_checklist.md) - requirement-by-requirement status.
* [`docs/models.generated.md`](docs/models.generated.md) - the catalogue generated from the code
  (`python scripts/generate_docs.py`).

## Extending it

A new model is two things: a class and a registry entry.

```python
from qcog.models.base import Model
from qcog.core.params import ParamSpec
from qcog.models.common import measurement_for, qubit
from qcog.experiments.two_question import TwoQuestionExperiment

class MyQuestionPair(Model):
    slug = "my-question-pair"
    title = "..."
    module = "A"
    one_liner = "..."
    # cognitive_question, quantum_mechanism, classical_baseline, references, caveats, keywords

    params = (ParamSpec("angle", "float", 1.0, low=0.0, high=3.14, fittable=True),)

    def make_experiment(self, angle=1.0):
        return TwoQuestionExperiment(qubit(0.0), measurement_for(angle, "A"), measurement_for(1.57, "B"))

    def build(self, **parameters):
        return {"experiment": self.make_experiment(**parameters), "parameters": parameters}

    def simulate(self, context, engine):
        return context["experiment"].run_both_orders(engine)

    def analyse(self, context, results):
        bundle = context["experiment"].analyse(results)
        return {"order_effect": bundle.report}, [bundle.report.summary()], bundle.report.summary()
```

Subclassing the shared `OrderBasedModel` (module B's models do) removes even that boilerplate: you declare the
parameters and the two questions, and inherit both-orders simulation, the interference decomposition, the QQ
equality, the law-of-total-probability test, the classical comparison, the sweeps and the standard figures.

## Scope, honesty and limitations

* A quantum-probability model says that **the data are described by a quantum probability space**. It does not
  say that the brain is a quantum computer, and Q-Cog never implies it.
* Every model's `caveats` list is rendered in the CLI, the UI and the report; the library is built so that
  over-claiming is harder than not over-claiming.
* `classical-baseline` results are computed, not asserted, and where a classical account fits equally well the
  information criteria say so.
* Datasets shipped in `src/qcog/data/` are **illustrative and synthetic**, generated by Q-Cog itself and
  labelled as such; they exist to exercise the fitting workflow. See
  [`src/qcog/data/README.md`](src/qcog/data/README.md).
* The population models simulate at most ten agents densely; larger networks need the classical baselines or a
  tensor-network implementation. The library says so where it matters.
* Game-theoretic conventions (entangling-gate sign, strategy phases) differ between papers. Q-Cog documents its
  own convention, verifies the properties it can check, and prints the best-response scan instead of asserting an
  equilibrium claim that depends on which strategy space is assumed.

## Licence and attribution

MIT - see [`LICENSE`](LICENSE). Code, documentation and figures are released under the same licence, and every
external dependency is MIT/BSD/Apache licensed. Attribution and the disclosure of pre-existing material are in
[`ATTRIBUTION.md`](ATTRIBUTION.md). Suggested citation: [`CITATION.cff`](CITATION.cff).

**This effort is supported by Qollab & IonQ.**