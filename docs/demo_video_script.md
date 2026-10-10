# Demo video script (2 to 3 minutes)

Target length: **2 min 40 s**. Speak calmly; the screen does the work. Have a terminal and a browser ready, and
pre-generate the report (`qcog report`) so the last section loads instantly.

---

### 0:00 - 0:15 | The problem, in one breath

**Screen:** the README title block, then the terminal.

> "Quantum cognition asks a specific question: can human judgements be described by a probability theory in
> which questions do not commute? If they can, then things that look like errors become structural. Q-Cog makes
> that programme testable, explainable and comparable with classical models - in one library, built on Qiskit."

### 0:15 - 0:45 | The flagship result

**Screen:** run `python examples/01_first_steps.py` (or paste the four-line example) and show the output.

```python
from qcog import basis_state, question, run_sequence, order_effect, qq_equality
belief = basis_state(0, 2)
a, b = question(0.955, 0.0, "A"), question(3.14159/2, 0.0, "B")
print(order_effect(run_sequence(belief, [a, b]), run_sequence(belief, [b, a]), "A", "B").summary())
print(qq_equality(run_sequence(belief, [a, b]), run_sequence(belief, [b, a]), "A", "B").summary())
```

> "Two questions. Asked in one order, p(A) is 0.79; asked in the other, 0.50. A classical model - one
> distribution over the two answers - forces those to be equal. And the model makes a prediction with no free
> parameters: the probability of answering both questions the same way is identical in the two orders. It holds
> to machine precision."

### 0:45 - 1:15 | Where the probability comes from

**Screen:** show the interference decomposition output, then the figure in `out/qcog_report.html` for
`question-order` (the stacked bar chart) and its text description below it.

> "The library does not stop at 'it's interference'. It splits the probability: this much is the classical
> law-of-total-probability part, and this much - minus 0.236 here - is interference. It can be negative, which
> is exactly why the two orders disagree."

### 1:15 - 1:45 | It runs on a real circuit

**Screen:** `qcog run question-order --backend aer --shots 20000 --seed 7`, then `qcog evidence` scrolling to the
parity rows.

> "The same model compiles to a Qiskit circuit and is sampled on the Aer simulator. `qcog evidence` prints the
> checks: the compiled prefix circuits reproduce the analytic engine to one part in ten to the sixteenth, the
> shot-based run agrees within five standard errors, commuting questions give exactly zero order effect, and
> full decoherence removes the effect exactly. Twenty-three models, all run, all normalised."

### 1:45 - 2:15 | Twenty-three models, one workflow

**Screen:** `qcog list`, then the explorer (`streamlit run app/streamlit_app.py`) changing a parameter and
showing a model re-run. If short on time, show the catalogue report instead.

> "Twenty-three models in four modules: individual judgement, concepts and memory, strategic decisions, and
> groups. Each one reports the classical baseline for the same situation, its own caveats, and a plain-language
> explanation of the numbers. Here is the conjunction fallacy: the model predicts a conjunction more probable
> than its conjunct - and the same model with independent attributes satisfies the classical bound exactly, so
> you can see what the effect really depends on."

### 2:15 - 2:40 | The output you can publish

**Screen:** open `out/qcog_report.html`, scroll through one model, expand a "Long description of this chart"
element, show a data table. Then end on the attribution line.

> "Everything exports as a self-contained accessible HTML report: no JavaScript, every chart with a text
> description and the data table behind it, every model with its caveats. MIT licensed, runs offline on a free
> simulator. This effort is supported by Qollab and IonQ."

---

## Checklist before recording

* [ ] Terminal font size increased; dark background for contrast.
* [ ] `out/qcog_report.html` already generated (`qcog report`), so the browser section is instant.
* [ ] `qcog evidence` output cached in a file or scrolled to the parity rows.
* [ ] Total runtime under three minutes; if running long, cut section 1:15-1:45 first, then section 1:45-2:15.
* [ ] Video uploaded somewhere public that opens without signing in, as the submission rules require.