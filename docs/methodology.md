# Methodology

This document is the technical companion to the code: what is modelled, which conventions are used, and which
identities are tested. Every equation below corresponds to a function in the library.

## 1. The representation

* A **belief state** is a density matrix `rho` on a `d`-dimensional complex Hilbert space; pure states are kept
  as amplitude vectors as well, because interference and phase are only defined for pure states
  (`qcog.core.states.CognitiveState`).
* A **question** is a measurement.
  * `ProjectiveMeasurement` - a complete orthogonal family of projectors, `sum_i P_i = I`. Sharp questions.
  * `POVM` - a complete family of effects `E_i >= 0`, `sum_i E_i = I`. Unsharp questions; in particular
    `E_yes = (I + lambda n.sigma)/2`, where `lambda = 1` reproduces a sharp question and `lambda = 0` is the
    uninformative question. This dial is how the library shows how much quantum structure a phenomenon needs.
* An answer: the Born rule, `p(i) = Tr(P_i rho)` (or `Tr(E_i rho)`), and the **Lüders update**,
  `rho -> P_i rho P_i / p(i)` (or `M_i rho M_i^† / p(i)` with `M_i = sqrt(E_i)`).

Because a model in dimension `d` is embedded into `2^ceil(log2 d)` qubits with zero amplitudes, the same
objects are simultaneously a mathematical model and a circuit.

## 2. Sequential measurement, order effects and interference

For two questions `A` (asked first) and `B`:

    p(b) = sum_a p(a) p(b|a)  +  sum_{a != a'} <psi|P_a P_b P_a'|psi>
           \__ classical __/     \________ interference ________/

`qcog.analysis.interference_decomposition` computes both terms exactly (for POVMs the same split holds with
`M_a` and `E_b`) and `check()` verifies that they reproduce `p(b)`. This is the library's central explanatory
object: "the order matters because of interference" becomes "the interference term is -0.236 and supplies 47%
of this probability".

Sequential measurement is *not* commutative in general. The library quantifies incompatibility as
`sum_ij || [P_i, Q_j] ||_F` (`ProjectiveMeasurement.incompatibility`), which is zero exactly when the questions
share an eigenbasis - the classical case.

### The QQ equality

For two dichotomous questions `A`, `B` asked in both orders, a quantum model predicts

    p(A=yes, B=yes) + p(A=no, B=no)   [order A then B]
      = p(B=yes, A=yes) + p(B=no, A=no)   [order B then A]

that is, the probability of answering both questions *the same way* does not depend on the order
(`qcog.analysis.qq_equality`). The identity involves no free parameters, which is why the library treats it as
the sharpest available test; shot-based runs report the difference with its standard error and a z-score, so a
difference within noise is reported as such. It is defined only for two questions that share the same two
answer labels, and the function says so when they do not (`QQReport.applicable`).

### The law of total probability

`qcog.core.classical.law_of_total_probability` compares the directly measured marginal of a question with the
prediction obtained from the reversed order. Any classical model forces the difference to zero; the number
reported is therefore a contextuality certificate.

## 3. Classical baselines

Each baseline is computed, never asserted:

| Baseline | Where | What it enforces |
| --- | --- | --- |
| Single joint distribution (`markov_joint`, `LTPReport`) | order and framing models | one marginal per question, order-invariant |
| Conjunction bound `p(A and B) <= min(p(A), p(B))` | conjunction fallacy, concept combination | Kolmogorov additivity |
| Independence `p(A and B) = p(A) p(B)` | conceptual entanglement, collective judgement | product structure |
| DeGroot averaging, Hegelsmann-Krause | opinion dynamics, polarisation | consensus without superposition |
| Condorcet jury theorem | collective judgement | independent voters |
| Bayesian social learning, Bayesian cascades | information cascades, belief revision | commutative updating |
| Classical prisoner's dilemma, mixed-strategy equilibria | module C | Nash / mixed equilibria |
| Commuting quantum questions | all order models | quantum formalism is not sufficient for order effects |

The last row matters: `qcog.core.measurement.measurements_commute` and the zero-sharpness limit show that
quantum formalism *alone* guarantees nothing. Order effects come from incompatibility, and the models say so.

## 4. Execution backends

| Backend | What it does | When to use it |
| --- | --- | --- |
| `exact` | Born rule + Lüders update on the projectors, via Qiskit `Statevector`/`DensityMatrix` | reference numbers, fitting, sweeps |
| `sampled` | the exact distribution resampled with a multinomial draw | finite participant pools, statistical tests |
| `aer` | the compiled circuit on `AerSimulator`, one shot per complete interview | circuit-level evidence, noise, hardware-like behaviour |
| `aer_exact` | exact probabilities read from the statevectors of the compiled *prefix* circuits | validating the whole compilation chain without shot noise |
| `ionq` (optional) | the same circuit through `qiskit-ionq` | submitting to IonQ's simulator or hardware |

Noise enters in two distinct ways, and the library keeps them apart:

* **Kraus channels** (`qcog.core.instruments`): exact and deterministic. `NoiseSpec(..., apply_to="before_first")`
  decoheres the belief *before* the interview (which removes the order effect exactly), while
  `apply_to="each_question"` decoheres *between* questions (which leaves the first marginal untouched but
  destroys the conditional structure). Both behaviours are checked in `qcog evidence`.
* **Aer noise models** (`qcog.backends.aer_noise_model`): device-like errors applied inside the simulator.

## 5. Circuit compilation

Two compilations, from the same projectors:

* `build_sequence_circuit` - one circuit per interview: `StatePreparation`, then for each question the basis
  change `U†` on *that question's qubits only*, a measurement, and (for all but the last question) the inverse
  rotation `U`, which restores the physical frame so that later questions see the state the analytic engine
  sees. Measuring only the question's own qubits matters: measuring an uninvolved qubit would collapse
  amplitudes the question does not care about, which silently destroys coherence.
* `build_prefix_circuit` - one small circuit per answer prefix, using only state preparation, a unitary and a
  measurement: no mid-circuit measurement, no `reset`. The projection is computed on the whole block the answer
  corresponds to, which is necessary because an answer with a degenerate outcome is a subspace, not a basis
  state. This is the compilation to use on devices with limited mid-circuit measurement support, and it is the
  one used by the `aer_exact` backend to validate the analytic engine to machine precision.

Qubits are little-endian (Qiskit convention): basis index `k` has qubit `q` in state `(k >> q) & 1`. The
`Operator`-free parts of the implementation (projector arithmetic, decompositions) use NumPy directly, with
Qiskit used wherever a quantum-information object is the natural one.

Two cases the compiler handles explicitly, because both are easy to get wrong:

* **A question that lives in a smaller space.** A three-outcome question (auctions, semantic ambiguity, voting)
  acts on three basis states of a two-qubit register. Its basis change is extended block-diagonally with an
  identity, the padding state carries no population, and its index is folded into the last outcome so decoding
  can never fail on a numerical artefact.
* **A question that concerns only some qubits.** A question about one subsystem must not measure the others -
  measuring an uninvolved qubit would collapse amplitudes the question does not care about, which silently
  destroys coherence. The library detects trivial qubits structurally (an operator acts trivially on a qubit if
  it is diagonal and independent in that qubit's two tensor axes) and compiles the question onto its own qubits
  only.

## 5b. Drawing the circuit

`Model.circuit_designs(result)` walks the model's own `build` output, matches each state with the questions of
the same Hilbert-space dimension, and compiles them; `circuit_figure` draws the result through Qiskit's
matplotlib drawer and attaches the plain-language description, so the drawing is available to a screen reader
as well. What a reader can expect:

* two drawings for a two-question design - both orders, because the model is about the difference between them;
* the entangling circuit for the game models (`J`, the two strategy unitaries, `J†`, the measurement), which is
  the whole quantum mechanism;
* one network interaction step for the population models, since the population itself is a density matrix;
* the GHZ preparation for the correlated-crowd model;
* the sharp limit of a question that the model itself asks unsharply, with a note saying so;
* otherwise a note in words - a mixed state, or an unsharp question with no sharp counterpart.

## 5c. The classical comparison as a first-class output

Every run carries `result.classical`: the baseline's name and description, a list of rows (`metric`, `model`,
`baseline`, `note`) and a verdict sentence. The rows come from the model's own numbers, so they cannot drift from
the simulation, and they are rendered as a table in the CLI, the explorer and the report, and as a grouped bar
chart in every figure set. Rows whose classical value is exactly zero are the strongest form: a classical
account is *forced* to predict zero there, so the model's bar is the whole content of that comparison
(an order effect, an interference contribution, a conjunction excess, a sure-thing violation, a framing effect,
a Condorcet-cycle certificate).

## 5d. Every visualization, and how it is kept working

The library can draw eleven kinds of chart, and each one is exercised on synthetic input by
`qcog.viz.standalone_checks()`, checked against its own data by `qcog.viz.structure_checks()` - which reads the
drawing back and compares it with the numbers that produced it - and, for every model, rendered by
`qcog.viz.catalogue_checks()`. `python scripts/check_figures.py` runs all three passes. A chart cannot quietly
stop working, show the wrong data, or lose its text description without failing a check.

| Function | What it shows | Where it is used |
| --- | --- | --- |
| `probability_bars` | joint outcome probabilities of one simulation | every model |
| `marginal_bars` | the marginal answer distribution of each question | wide outcome spaces |
| `order_effect_figure` | the first question's marginal in both orders | order and framing models |
| `interference_figure` | direct versus interference contributions | interference models |
| `sweep_figure` | every metric of a parameter sweep | all models with sweeps |
| `line_figure` | trajectories (opinions, beliefs, populations) | modules A and D |
| `histogram_figure` | count distributions (votes, decisions) | cascades, collective judgement |
| `bloch_sphere` | a qubit belief state and its length | module A |
| `heatmap_figure` | any tidy grid: one value per (row label, column label) pair | games and sweeps |
| `payoff_table_figure` | a game's payoff to the first player - the quantum account beside the classical one | module C |
| `classical_comparison_figure` | model versus classical baseline, quantity by quantity | every model |
| `circuit_figure` | the compiled Qiskit circuit, with an adjustable style | every model with a circuit |

Two conventions make the charts readable rather than merely present. First, series differ by *colour and* by
linestyle, marker or hatch (eight colours, then four line styles, then markers), so a figure survives greyscale
printing and colour-vision deficiency, and a two-series comparison is never distinguished by hue alone. Second,
every figure carries a plain-language description on the figure object, so the same text serves the explorer, the
HTML report and a screen reader.

## 5e. Adjusting the classical comparison

The comparison is a first-class output and as adjustable as any parameter:

* `classical=False` skips it (and the report and figures say so, rather than showing an empty chart);
* `rows` and `max_rows` work for every model - they select and cap the comparison rows;
* models whose baseline has real parameters declare them in `classical_option_names` and honour them in
  `comparison_rows`: `accuracy` (Condorcet's jury theorem), `epsilon` and `kind` (bounded-confidence opinion
  dynamics), `signal_accuracy` (Bayesian cascades). The rows are *recomputed*, not relabelled, so a figure always
  shows the baseline that was actually used;
* options a model cannot honour are recorded in `result.classical["options_ignored"]` and named in the verdict,
  so nothing is dropped silently.

The point of the knob is a counterfactual a reader will actually ask: *would a classical group have done better,
judge for judge?* `classical_options={"accuracy": 0.8}` answers it - Condorcet's value moves from 0.874 to 0.967
at seven judges, while the model's own majority accuracy stays where its state puts it.

## 6. Statistical layer

`qcog.core.metrics`: total variation, KL and Jensen-Shannon divergence, Hellinger and Bhattacharyya distances,
Pearson chi-square and G tests, multinomial log-likelihood, AIC and BIC, a two-sided z test for proportions, and
the same-answer agreement index used by the QQ equality.

`qcog.core.fitting`: a least-squares / maximum-likelihood fit over the parameters a model marks `fittable`,
with bounds taken from the parameter schema. Passing counts switches on the multinomial likelihood, the
chi-square-style weighting and AIC/BIC; `compare_fits` returns a verdict that distinguishes "the quantum model
is favoured" from "the two accounts are statistically indistinguishable on these data".

## 7. Game theory conventions

The quantum games use the Eisert-Wilkens-Lewenstein scheme: `J(gamma) = cos(gamma/2) I + i sin(gamma/2) X x X`
applied to `|00>`, then the players' unitaries `U(theta, phi) = [[e^{i phi} cos(theta/2), sin(theta/2)],
[-sin(theta/2), e^{-i phi} cos(theta/2)]]`, then `J†`, then a computational-basis measurement. Player 1 owns the
most significant qubit, so the basis index of the move pair `(own, opponent)` is `opponent_bit + 2 * own_bit`.

In this convention it is the *entangler* that removes the dilemma, and the quantum games verify that rather
than assert it: with maximal entanglement defecting on a cooperator pays `0` instead of the classical `5`,
mutual cooperation `(3, 3)` is the best reply to itself, and at zero entanglement the classical payoff matrix -
the temptation included - is reproduced exactly. EWL's "quantum" strategy `Q = U(pi/2, 0)` is in the strategy
table, but it is not what makes cooperation stable here (two `Q` players meet in a mixed outcome), so the models
print the payoff table and the best-response scan for the strategies actually in play. Sign and phase
conventions differ between papers, so this paragraph is the statement of the one this library uses. See
[`limitations.md`](limitations.md).

## 8. Population models

For module D, agents are qubits on a graph; an interaction is a **partial swap** `exp(i theta SWAP)` on an edge,
which is a coherent exchange of amplitude rather than an average. Decoherence between rounds is per-agent
depolarising noise. Because the simulation is dense, populations are limited to ten agents - a documented limit,
with the classical baselines (DeGroot, Hegselmann-Krause) available for larger networks. Vote-count
distributions are read off the diagonal by Hamming weight, which is exact for a count observable; correlations
show up as over-dispersion relative to the independent binomial.

## 9. What is verified, and how

`qcog evidence` prints the table; the same table is embedded in the generated report. The checks are listed in
the README. Two remarks on method:

* the checks are *executable* rather than textual, so a regression fails the test suite (136 tests) and the
  evidence table;
* the checks deliberately include the classical limits (zero entanglement, zero sharpness, commuting questions,
  full decoherence), because a model that only "works" away from its own limits is usually wrong.