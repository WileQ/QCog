# Q-Cog - a quantum cognition simulation and visualization library

> **This effort is supported by Qollab & IonQ.**

## What the submission does

Q-Cog is a reusable Python library, plus an interactive platform built on top of it, for modelling,
simulating, fitting and explaining **quantum-probability models of human judgement, decision making and
collective behaviour**. It is built on **Qiskit** and runs on a free local simulator; nothing in it requires
hardware, a token or network access.

It connects four things that are usually separate:

1. **Cognitive phenomena** - what happens in reasoning, judgement and decision making.
2. **Mathematical models** - quantum states, projectors, interference, entanglement, sequential measurement.
3. **Qiskit simulations** - runnable circuits, statevector references, shot-based sampling, noise.
4. **Interactive explanations** - why the model predicts this particular result, and how it compares with a
   classical account.

It contains **23 documented models in four modules**:

* **A - Individual cognition:** superposition of beliefs, question-order effects, the interference/disjunction
  paradox, quantum decision trees, belief revision and framing.
* **B - Logic, concepts and memory:** the conjunction fallacy, the disjunction effect, concept combination,
  associative memory and cue interference, semantic ambiguity, sequential categorisation, conceptual
  entanglement, contradictory beliefs.
* **C - Strategic decisions:** the quantum prisoner's dilemma (EWL), negotiation and trust, the evolution of
  cooperation, auctions, coordination games.
* **D - Groups and networks:** quantum opinion dynamics, polarisation, collective judgement (Condorcet),
  information cascades, deliberation and voting.

Every model can be driven and reconfigured from Python, and every run reports:

* what it simulates, as a joint and conditional response distribution;
* the **classical baseline** for the same situation, computed rather than asserted, as a quantity-by-quantity
  comparison (a table in the CLI, the explorer and the report, and a bar chart in every figure set);
* an **interference decomposition** wherever it applies (classical term plus interference term);
* the order-effect measures, including the parameter-free **QQ equality** for two dichotomous questions;
* a drawing of the **quantum circuit** wherever the model has one (the image itself is adjustable: style, scale,
  folding, format) - both orders of a two-question design, the
  entangling circuit for the games, the sharp limit of an unsharp question, the GHZ preparation for the
  correlated crowd - each carrying a plain-language description, and a written explanation when a drawing is
  impossible (a mixed state, for example);
* the **caveats**: what the model does *not* establish.

## What problem it explores

Quantum cognition asks a specific question: *can human judgements be described by a probability theory in which
questions do not commute?* If they can, then several things that look like errors become structural
consequences instead:

* the answer to a question depends on which question came before it (question-order effects);
* the probability of a conjunction can exceed the probability of its conjunct (the conjunction fallacy);
* a decision changes when an unresolved alternative is resolved (the disjunction effect);
* a group is wiser than its members only to the extent that the members are independent.

Q-Cog does not argue for this programme; it makes it **testable and comparable**. Each model states the
classical identity that its quantum counterpart can violate, computes the size of the violation, and provides
the statistical tools to ask whether an observed violation is real or noise. The library's explicit policy is
that where a classical account fits equally well, it says so.

## How to run it

### Requirements

Python 3.10+, `qiskit>=2.0`, `qiskit-aer`, `numpy`, `scipy`, `pandas`, `matplotlib`. Optional: `streamlit`
(interactive explorer), `qiskit-ionq` (IonQ submission path).

### Install and run

```bash
python -m pip install -e ".[app,dev]"      # or run from the checkout: the notebook adds src/ itself

qcog list                       # the catalogue
qcog run question-order         # run the flagship model and read its findings
qcog evidence                   # the self-check table (about a minute)
qcog report --out out/qcog_report.html --markdown
streamlit run app/streamlit_app.py
python examples/01_first_steps.py
python showcase/run_showcase.py               # a scripted tour with captured output and figures

qcog circuit quantum-prisoners-dilemma        # draw the model's circuit
qcog run question-order --fast                # headline numbers without the sweeps
qcog run collective-judgment --classical-option accuracy=0.8   # adjust the classical baseline
```

### Driving a model from Python

Every parameter of every model can be set from Python, and every parameter's bounds and meaning are declared
once and reused by the CLI, the explorer, the fitting layer and the documentation. `qcog.core.params.ParamSpec`
describes them; `model.defaults()` prints them; `model.run(**parameters)` applies them. The test suite runs each
model with every parameter at both of its bounds (or at every choice) and checks that the results stay
normalised, so "adjustable in Python" is a tested property rather than a claim. Two further checkers keep the
artefacts honest: `python scripts/check_figures.py` renders every visualization and verifies that each drawing
matches the numbers behind it, and `python scripts/check_docs.py` verifies that the documentation still matches
the code.

### Guided tours

* **`notebooks/quickstart.ipynb`** - a short, simple notebook built around one real survey: *"Are you satisfied
  with the economy?"* and *"Will you vote for the current administration?"*, asked in both orders. It explains
  the state and the questions, reads the order effect, checks the law of total probability and the parameter-free
  QQ equality, draws the circuits with the survey questions on the gates, samples them on a simulator and saves a
  report. Its cells are executed by the test suite, so it always runs.
* **`examples/06_economy_and_vote.py`** - the same survey as a script, writing the circuit images out.
* **`showcase/`** - the same ideas as a script, with its captured output, every figure and an HTML report per
  model.
* **`app/streamlit_app.py`** - the interactive explorer, generated from each model's parameter schema, with
  dark, larger-type, high-contrast and text-only modes.

### The shortest complete example

```python
import sys; sys.path.insert(0, "src")
from qcog import basis_state, question, run_sequence, order_effect, qq_equality, explain_order_effect

belief = basis_state(0, 2)
a = question(0.955, 0.0, "A")
b = question(3.14159 / 2, 0.0, "B")

forward = run_sequence(belief, [a, b])
backward = run_sequence(belief, [b, a])

report = order_effect(forward, backward, "A", "B")
print(report.summary())
print(explain_order_effect(report))
print(qq_equality(forward, backward, "A", "B").summary())
```

## How to replicate it

Everything needed to reproduce every number in this project is in the repository, and the checks are
executable:

* `python -m pytest -q` runs 121 tests covering states, questions, channels, backends, analysis, classical
  baselines, fitting, the CLI, the reports, the figures and every model's central claim.
* `qcog evidence` prints the evidence table: the interference decomposition, the QQ equality, the classical
  limits, the circuit-versus-analytic parity, the game-theory checks, the entanglement calibration, and a run of
  all 23 models.
* `qcog report` embeds that same table in the generated HTML, so a failure is visible in the published artefact.
* All randomness is seeded; the analytic backends are exact and deterministic.

The two execution paths come from **one** definition. Every question is a family of projectors; the analytic
engine applies the Born rule and the Lüders update to them directly, and the circuit compiler turns the same
projectors into basis-change unitaries. Agreement between the two (about 1e-16 for the exact compilations,
within five standard errors for shot-based runs) is what makes the compilation trustworthy.

## How to extend it

* **Add a model**: subclass `Model` (or `OrderBasedModel` for two-question designs), declare metadata and
  parameters, register it. The CLI, the explorer, the report, the docs and the tests pick it up automatically.
  The generated documentation (`scripts/generate_docs.py`) proves the claim that the docs cannot drift from the
  code.
* **Bring your own data**: `qcog fit <model> data.json` fits a model's parameters to observed frequencies and
  compares it with a classical baseline through information criteria. The data format is documented in
  `src/qcog/data/README.md`.
* **Use the pieces directly**: the core (`qcog.core`), the analysis (`qcog.analysis`), the experiment designs
  (`qcog.experiments`), the backends (`qcog.backends`) and the figures (`qcog.viz`) are all usable on their own.

## Accessibility

Industry best practices for UI accessibility are followed by construction:

* every figure carries a plain-language description on the figure object, and the report embeds it as alt text
  plus a long description, with the underlying data table next to it;
* the colour-blind-safe Okabe-Ito palette, with no information carried by colour alone;
* semantic HTML with a skip link, ordered headings, table captions and `scope` attributes, visible focus
  outlines; Streamlit's labelled native controls in the explorer;
* dark-background, larger-type, high-contrast and text-only modes;
* explanations written in plain language and generated from the computed numbers.

Details and verification instructions: [`accessibility.md`](accessibility.md).

## Documentation map

| File | Contents |
| --- | --- |
| [`../README.md`](../README.md) | overview, install, quick start, catalogue, CLI |
| [`project_page.md`](project_page.md) | this page |
| [`methodology.md`](methodology.md) | the mathematics, conventions and statistical tests |
| [`accessibility.md`](accessibility.md) | accessibility practices and how to check them |
| [`limitations.md`](limitations.md) | what the models do not claim |
| [`demo_video_script.md`](demo_video_script.md) | the 2-3 minute demo script |
| [`qollab_runbook.md`](qollab_runbook.md) | running and publishing on Qollab, including IonQ |
| [`submission_checklist.md`](submission_checklist.md) | requirement-by-requirement status |
| [`models.generated.md`](models.generated.md) | the catalogue generated from the code |
| [`../ATTRIBUTION.md`](../ATTRIBUTION.md) | attribution and disclosure |

## Licence

MIT. Code, documentation and figures are released under the same licence.