# Showcase

A runnable demonstration of Q-Cog that lives **outside the package**: everything here is about *using* the
library, so nothing in this directory is imported by `qcog` itself.

## Run it

```bash
python demos/showcase/run_showcase.py                     # the four models chosen below, exact backend
python demos/showcase/run_showcase.py --all               # every model in the catalogue
python demos/showcase/run_showcase.py --models question-order auctions      # any subset
python demos/showcase/run_showcase.py --param correlation=0 --no-figures      # override parameters
python demos/showcase/run_showcase.py --reports           # also write an accessible HTML report per model
```

No installation is needed: the script adds `src/` to the path itself, so it runs from a plain checkout. For a
cell-by-cell version of the same tour, open [`demos/quickstart.ipynb`](../demos/quickstart.ipynb).

## What it prints

For each model: the cognitive question, the quantum mechanism and the classical baseline; every parameter with
its bounds (all adjustable from Python); the summary; the findings; the **model-versus-classical comparison**;
whether a **quantum circuit** was drawn (and what it contains); and the model's own caveats. At the end it
re-runs each model with a parameter moved, to show how a simulator behaves when you turn a dial.

`output.txt` in this directory is exactly what the default invocation prints, captured verbatim.

## What it saves

| Path | Contents |
| --- | --- |
| `figures/` | every figure of every selected model: results, the classical-comparison chart, circuit drawings, sweeps |
| `reports/` | a self-contained accessible HTML page per model (figures embedded, alt text, data tables, caveats) |
| `output.txt` | the captured console output of the default run |

## The models it runs by default

One per module, chosen because between them they exercise every feature of the library:

| Model | Module | Why it is here | What it shows |
| --- | --- | --- | --- |
| `superposition-beliefs` | A | the coherence-versus-ignorance contrast | two states with identical answers to one question, separated by the next |
| `semantic-ambiguity` | B | a three-level question | context fixes a word's sense, but only before the sense is resolved |
| `auctions` | C | an order effect in an economic setting | value and bid are different questions, so their order changes the bid |
| `collective-judgment` | D | a group mechanism | the crowd's advantage is exactly Condorcet's, minus correlation |

## Reading the output

* **`AGAINST THE CLASSICAL BASELINE`** is the point of the library. Each row is the same quantity computed by the
  model and by the classical account of the same situation. Where the classical value is exactly `0.000` a
  classical model is *forced* to be silent there, so the model's number is the whole content of that comparison.
* **`QUANTUM CIRCUIT`** lists the drawings saved into `figures/`. Both orders of a two-question design are
  drawn, game models get their entangling circuit, and where a drawing is impossible a `note:` says why in
  words (a mixed state, an unsharp question with no sharp counterpart).
* **`WHAT THIS DOES NOT CLAIM`** is part of the output on purpose: a model of human judgement should state its
  limits as loudly as its results.

## Reproducing the captured output

```bash
python demos/showcase/run_showcase.py --reports > demos/showcase/output.txt
```

The run is deterministic (the analytic backends are exact; the sampled backends are seeded), so the numbers in
`output.txt` can be regenerated exactly. The figure files depend on the matplotlib version in the same way any
plot does.