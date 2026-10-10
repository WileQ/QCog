# Model catalogue (generated)

> **Generated file - do not edit by hand.** Run `python scripts/generate_docs.py`.
>
> This document is produced from the model classes, so the parameters, defaults, bounds, caveats and references
> below always match the code.


**23 models in 4 modules.**

- Module A (Individual cognition and decision making): `superposition-beliefs`, `question-order`, `interference-paradox`, `decision-tree`, `belief-revision`
- Module B (Logic, concepts and memory): `conjunction-fallacy`, `disjunction-effect`, `concept-combination`, `associative-memory`, `semantic-ambiguity`, `sequential-categorization`, `conceptual-entanglement`, `contradictory-beliefs`
- Module C (Strategic decisions and game theory): `quantum-prisoners-dilemma`, `negotiation-trust`, `evolution-of-cooperation`, `auctions`, `coordination-games`
- Module D (Groups, social networks and collective behaviour): `quantum-opinion-dynamics`, `polarization`, `collective-judgment`, `information-cascades`, `deliberation-voting`

## Module A - Individual cognition and decision making

### `superposition-beliefs` - Superposition of beliefs versus classical uncertainty

*A coherent belief and an equally uncertain but already settled belief give identical answers to one question and opposite answers to the next.*

- **Cognitive question.** When someone reports 'fifty-fifty', is the attitude genuinely unresolved, or settled but unknown? Response probabilities alone cannot tell the difference.
- **Quantum mechanism.** Coherent superposition versus statistical mixture: coherence is invisible to the question whose answer distribution it defined, and visible to any question incompatible with it.
- **Classical baseline.** A classical model identifies a 50/50 belief with the distribution (1/2, 1/2) and therefore predicts that no other question can distinguish the two situations.
- **Difficulty.** introductory
- **Keywords.** superposition, mixture, coherence, state, probability

| parameter | kind | default | bounds | fittable | meaning |
| --- | --- | --- | --- | --- | --- |
| `belief_angle` | float | 1.5707963267948966 | 0.0 to 3.141592653589793 | no | Bloch angle of the coherent belief: pi/2 is the balanced superposition. |
| `probe_angle` | float | 1.5707963267948966 | 0.0 to 3.141592653589793 | yes | Angle of the second question; pi/2 makes it maximally incompatible with the first. |
| `decoherence` | float | 0.15 | 0.0 to 1.0 | no | Depolarising probability applied to the coherent state (0 = pristine, 1 = fully mixed). |

**What it does not claim.**

- This is a model of response probabilities, not a claim about what the brain does.
- A quantum model only means the data are described by a quantum probability space; classical models may fit equally well, and the classical baseline is reported for that reason.
- Coherence is a property of the model state; nothing here claims coherence has been measured in a person.
- With a single qubit, asking one sharp question erases exactly the coherence that a different sharp question would have revealed - which is why the model compares two separate questions rather than one question asked twice.

**References.**

- Busemeyer & Bruza (2012), Quantum Models of Cognition and Decision, ch. 1-2.
- Pothos & Busemeyer (2013), Can quantum probability provide a new direction for cognitive modeling? BBS 36, 255-274.

```bash
qcog run superposition-beliefs
qcog info superposition-beliefs
```

### `question-order` - Question-order effects and incompatible questions

*Asking two questions in the two possible orders changes the first answer; the model predicts by how much, decomposes the change into interference, and satisfies a parameter-free QQ equality.*

- **Cognitive question.** Why does the answer to 'how satisfied are you with your life?' depend on whether 'how satisfied are you with your marriage?' was asked first?
- **Quantum mechanism.** Sequential projective (or unsharp) measurements of non-commuting questions: answering the first question projects the state, so the second question is put to a different belief state.
- **Classical baseline.** A single joint distribution over responses (a classical sample space) forces the marginal of a question to be identical in both orders; any nonzero order effect rejects that family of models.
- **Difficulty.** intermediate
- **Keywords.** question order, non-commuting, interference, QQ equality, contextuality

| parameter | kind | default | bounds | fittable | meaning |
| --- | --- | --- | --- | --- | --- |
| `state_angle` | float | 0.0 | 0.0 to 3.141592653589793 | yes | Bloch angle of the initial belief state (0 = a definite answer to the first question). |
| `state_phase` | float | 0.0 | -3.141592653589793 to 3.141592653589793 | no | Relative phase of the belief state: it changes the interference term but not the first marginals. |
| `angle_a` | float | 0.955 | 0.0 to 3.141592653589793 | yes | Angle of the first question; 0 would make it identical to the belief basis. |
| `angle_b` | float | 1.5707963267948966 | 0.0 to 3.141592653589793 | yes | Angle of the second question; a large difference from A creates the effect. |
| `sharpness` | float | 1.0 | 0.0 to 1.0 | no | 1 = projective question, 0 = an uninformative question (the classical limit). |
| `decoherence` | float | 0.0 | 0.0 to 1.0 | no | Depolarising probability applied after each question, modelling forgetting or device noise. |

**What it does not claim.**

- This is a model of response probabilities, not a claim about what the brain does.
- A quantum model only means the data are described by a quantum probability space; classical models may fit equally well, and the classical baseline is reported for that reason.
- The QQ equality requires two dichotomous questions; for more outcomes the model checks the corresponding multi-term identity numerically instead.

**References.**

- Wang & Busemeyer (2013), A quantum question order model supported by empirical tests of an a priori and precise prediction, Topics in Cognitive Science 5, 689-710.
- Wang, Solloway, Shiffrin & Busemeyer (2014), Context effects produced by question orders reveal quantum nature of human judgments, PNAS 111, 9431-9436.
- Moore (2002), Measuring new types of question-order effects, Public Opinion Quarterly 66, 80-91.

```bash
qcog run question-order
qcog info question-order
```

### `interference-paradox` - Interference between unresolved alternatives (disjunction effect)

*Recording which of two alternatives applies removes the interference term that decided the outcome, so making a question 'more informative' changes the decision.*

- **Cognitive question.** Why do people who would accept a gamble conditional on winning and conditional on losing decline it while the outcome is still unknown (Tversky & Shafir's disjunction effect)?
- **Quantum mechanism.** Interference between the amplitudes of unresolved alternatives; a recorded intermediate question acts as a non-selective measurement that destroys the off-diagonal terms.
- **Classical baseline.** The law of total probability / sure-thing principle requires p(decide yes) = p(win) p(yes|win) + p(lose) p(yes|lose) exactly.
- **Difficulty.** intermediate
- **Keywords.** interference, disjunction effect, sure-thing principle, which path

| parameter | kind | default | bounds | fittable | meaning |
| --- | --- | --- | --- | --- | --- |
| `amplitude_angle` | float | 0.7853981633974483 | 0.0 to 1.5707963267948966 | no | How the belief splits between the two alternatives (pi/4 = equal amplitudes). |
| `relative_phase` | float | 0.0 | -3.141592653589793 to 3.141592653589793 | no | Phase between the two alternatives: 0 and pi give maximal interference, pi/2 removes it. |
| `visibility` | float | 0.0 | 0.0 to 1.0 | no | 0 = alternatives combined, 1 = fully distinguished. |
| `decision_angle` | float | 1.5707963267948966 | 0.0 to 3.141592653589793 | no | Angle of the final yes/no question relative to the alternatives. |

**What it does not claim.**

- This is a model of response probabilities, not a claim about what the brain does.
- A quantum model only means the data are described by a quantum probability space; classical models may fit equally well, and the classical baseline is reported for that reason.
- A violation of the sure-thing principle shows that respondents did not integrate both branches into one probability space; it does not by itself identify interference.

**References.**

- Tversky & Shafir (1992), The disjunction effect in choice under uncertainty, Psychological Science 3, 305-309.
- Busemeyer & Bruza (2012), Quantum Models of Cognition and Decision, ch. 9.
- Kuperman & Perepelitsa (2016), A quantum version of the disjunction effect, Frontiers in Physics 4, 25.

```bash
qcog run interference-paradox
qcog info interference-paradox
```

### `decision-tree` - Quantum decision tree: does the order of information matter?

*Three cues, two arrival orders, one final decision: the model reports how much the decision depends on the order in which evidence arrived.*

- **Cognitive question.** Two people receive the same three pieces of evidence in different orders. Do they reach the same decision, and if not, which piece of evidence should have come first?
- **Quantum mechanism.** A sequence of measurements: each cue projects the belief state, so later cues act on a state that the earlier cues have already changed.
- **Classical baseline.** Classical belief updating (Bayes' rule) is commutative: the posterior does not depend on the order in which the evidence arrived.
- **Difficulty.** advanced
- **Keywords.** decision tree, information order, sequential measurement, Bayes

| parameter | kind | default | bounds | fittable | meaning |
| --- | --- | --- | --- | --- | --- |
| `state_angle` | float | 0.35 | 0.0 to 3.141592653589793 | no | Bloch angle of the prior belief about the decision. |
| `angle_cue_1` | float | 0.4 | 0.0 to 3.141592653589793 | no | Angle of the first cue. |
| `angle_cue_2` | float | 1.5707963267948966 | 0.0 to 3.141592653589793 | no | Angle of the second cue. |
| `angle_cue_3` | float | 2.4 | 0.0 to 3.141592653589793 | no | Angle of the third cue. |
| `sharpness` | float | 0.9 | 0.0 to 1.0 | no | How decisive each cue is (1 = fully decisive, 0 = uninformative). |

**What it does not claim.**

- This is a model of response probabilities, not a claim about what the brain does.
- A quantum model only means the data are described by a quantum probability space; classical models may fit equally well, and the classical baseline is reported for that reason.
- Three cues are simulated; this demonstrates order dependence rather than calibrating a specific task.

**References.**

- Busemeyer, Pothos, Franco & Trueblood (2011), A quantum theoretical explanation for probability judgment errors, Psychological Review 118, 193-218.
- Trueblood & Busemeyer (2011), A quantum probability account of order effects in inference, Cognitive Science 35, 1518-1552.

```bash
qcog run decision-tree
qcog info decision-tree
```

### `belief-revision` - Ambiguity, evidence accumulation and framing

*Repeated weak evidence moves a belief along a trajectory, and identically informative evidence given under a different framing lands somewhere else.*

- **Cognitive question.** Does the way a belief is framed change what the same evidence does to it, and does evidence accumulate the way Bayes' rule says it should?
- **Quantum mechanism.** Unsharp (POVM) measurements for weak evidence, repeated state updates for accumulation, and a unitary framing transformation that does not commute with the evidence.
- **Classical baseline.** Bayesian updating: the posterior depends on the likelihood ratio and the prior, not on how the question was framed.
- **Difficulty.** advanced
- **Keywords.** belief revision, updating, POVM, framing, ambiguity

| parameter | kind | default | bounds | fittable | meaning |
| --- | --- | --- | --- | --- | --- |
| `prior_angle` | float | 1.5707963267948966 | 0.0 to 3.141592653589793 | no | Bloch angle of the prior judgement. |
| `evidence_angle` | float | 0.6 | 0.0 to 3.141592653589793 | no | Which proposition the evidence bears on. |
| `evidence_strength` | float | 0.8 | 0.05 to 1.0 | no | Sharpness of the evidence measurement: 0.2 is weak evidence, 1 is conclusive. |
| `framing_angle` | float | 1.1 | -3.141592653589793 to 3.141592653589793 | no | How much the framing of the question rotates the belief state before the evidence arrives. |
| `repetitions` | int | 8 | 1 to 40 | no | How many times the same weak evidence is presented. |

**What it does not claim.**

- This is a model of response probabilities, not a claim about what the brain does.
- A quantum model only means the data are described by a quantum probability space; classical models may fit equally well, and the classical baseline is reported for that reason.
- The evidence operator models 'how much this observation moves the judgement', not a statistical likelihood.

**References.**

- Busemeyer & Bruza (2012), Quantum Models of Cognition and Decision, ch. 10.
- Trueblood & Busemeyer (2012), A quantum probability model of causal reasoning, Frontiers in Psychology 3, 138.

```bash
qcog run belief-revision
qcog info belief-revision
```

## Module B - Logic, concepts and memory

### `conjunction-fallacy` - Conjunction fallacy (the Linda problem)

*A relative phase in the belief lowers p(bank teller) while leaving the conjunction untouched, so 'bank teller and feminist' is judged more probable than 'bank teller'.*

- **Cognitive question.** Why do people rate 'Linda is a bank teller and a feminist' as more probable than 'Linda is a bank teller', when no probability space allows it?
- **Quantum mechanism.** Destructive interference between the components of the belief state in the direct bank-teller question; the conjunction is evaluated after the feminist question has collapsed the state, so it keeps the higher conditional probability.
- **Classical baseline.** The conjunction rule p(A and B) <= min(p(A), p(B)), which holds in every classical probability space and is also exact for commuting (independent-attribute) quantum questions.
- **Difficulty.** intermediate
- **Keywords.** conjunction fallacy, interference, Linda, probability judgment, incompatible questions

| parameter | kind | default | bounds | fittable | meaning |
| --- | --- | --- | --- | --- | --- |
| `feminist_amplitude` | float | 1.2 | 0.0 to 3.141592653589793 | no | Angle of the belief state towards the feminist answer (pi/2 = certain). |
| `teller_angle` | float | 2.0 | 0.0 to 3.141592653589793 | yes | Angle of the bank-teller question relative to the feminist one; pi/2 makes them maximally incompatible. |
| `relative_phase` | float | 3.141592653589793 | -3.141592653589793 to 3.141592653589793 | no | Phase of the belief state: pi gives maximal destructive interference in the direct bank-teller question. |

**What it does not claim.**

- This is a model of response probabilities, not a claim about what the brain does.
- A quantum model only means the data are described by a quantum probability space; classical models may fit equally well, and the classical baseline is reported for that reason.
- The model reproduces the fallacy with a phase; it does not claim that beliefs literally have phases.
- The independent-attribute variant of the same model satisfies the classical bound exactly, so the fallacy is a property of incompatible questions, not of uncertainty as such.

**References.**

- Tversky & Kahneman (1983), Extensional versus intuitive reasoning, Psychological Review 90, 293-315.
- Busemeyer, Pothos, Franco & Trueblood (2011), A quantum theoretical explanation for probability judgment errors, Psychological Review 118, 193-218.
- Pothos & Busemeyer (2013), Can quantum probability provide a new direction for cognitive modeling? BBS 36, 255-274.

```bash
qcog run conjunction-fallacy
qcog info conjunction-fallacy
```

### `disjunction-effect` - Disjunction effect and the representation of unresolved alternatives

*Cooperation under an unknown opponent differs from the average of the two known cases - and the model proves that no 'world x action' state, entangled or not, can reproduce that.*

- **Cognitive question.** Why is the decision different only when the alternatives are unresolved, if the decision is the same in each resolved branch?
- **Quantum mechanism.** Coherent amplitudes for the two unresolved branches adding to the same decision state, versus the incoherent branch mixture produced by any tensor-product representation.
- **Classical baseline.** The sure-thing principle: p(act) must equal the branch-weighted average of the conditional act probabilities.  It is also the exact prediction of any bipartite (world x action) state.
- **Difficulty.** advanced
- **Keywords.** disjunction effect, sure-thing principle, cooperation, representation, entanglement

| parameter | kind | default | bounds | fittable | meaning |
| --- | --- | --- | --- | --- | --- |
| `cooperate_if_opponent_cooperates` | float | 0.8 | 0.0 to 1.0 | yes | Conditional cooperation rate in the resolved branch. |
| `cooperate_if_opponent_defects` | float | 0.6 | 0.0 to 1.0 | yes | Conditional cooperation rate in the other resolved branch. |
| `opponent_cooperation` | float | 0.5 | 0.05 to 0.95 | no | Weight of the cooperating branch. |
| `relative_phase` | float | 3.141592653589793 | -3.141592653589793 to 3.141592653589793 | no | The coherence of the defecting branch relative to the cooperating one: pi makes the defect amplitudes cancel when the opponent is unknown, which raises cooperation. |

**What it does not claim.**

- This is a model of response probabilities, not a claim about what the brain does.
- A quantum model only means the data are described by a quantum probability space; classical models may fit equally well, and the classical baseline is reported for that reason.
- The model demonstrates that the effect requires a non-reducible representation; it does not show that people use one.

**References.**

- Shafir & Tversky (1992), Thinking through uncertainty: nonconsequential reasoning and choice, Cognitive Psychology 24, 449-474.
- Busemeyer & Bruza (2012), Quantum Models of Cognition and Decision, ch. 9.

```bash
qcog run disjunction-effect
qcog info disjunction-effect
```

### `concept-combination` - Concept combination: emergent and destroyed features

*Combining two concepts with a unitary operation makes some features more typical than they are in either part, and others less - a joint structure the weighted-average baseline cannot produce.*

- **Cognitive question.** Why is a 'pet fish' neither a typical pet nor a typical fish, and why do combined concepts have features that neither part has?
- **Quantum mechanism.** A combination operation on the tensor product of two concepts; typicality is the projector of a feature question, and non-separability creates joint structure.
- **Classical baseline.** Feature typicality of the combination as a weighted average of the typicalities of the parts (the standard extensional baseline).
- **Difficulty.** intermediate
- **Keywords.** concept combination, emergence, typicality, SCOP, non-separability

| parameter | kind | default | bounds | fittable | meaning |
| --- | --- | --- | --- | --- | --- |
| `pet_typicality` | float | 0.8 | 0.05 to 0.95 | no | Probability that a typical pet feature applies to the first concept. |
| `fish_typicality` | float | 0.25 | 0.05 to 0.95 | no | How typical the feature is for the second concept alone. |
| `combination_strength` | float | 1.2 | 0.0 to 3.141592653589793 | no | Angle of the combination operation: 0 leaves the two concepts independent. |
| `feature` | choice (fins, care, colourful) | 'fins' | - | no | Which feature typicality to report on. |

**What it does not claim.**

- This is a model of response probabilities, not a claim about what the brain does.
- A quantum model only means the data are described by a quantum probability space; classical models may fit equally well, and the classical baseline is reported for that reason.
- Typicality ratings are modelled as probabilities of a feature question; the mapping from rating scales to probabilities is a modelling choice.

**References.**

- Aerts & Gabora (2005), A theory of concepts and their combinations, Kybernetes 34, 167-191.
- Hampton (1987), Inheritance of attributes in natural concept conjunctions, Memory & Cognition 15, 55-71.
- Blutner, Pothos & Bruza (2013), A quantum probability perspective on borderline vagueness, Topics in Cognitive Science 5, 711-736.

```bash
qcog run concept-combination
qcog info concept-combination
```

### `associative-memory` - Associative memory and competing retrieval cues

*Similar retrieval cues interfere, so presenting both cues moves recall away from the sum (or the average) of their separate effects.*

- **Cognitive question.** Why does adding a second, similar retrieval cue sometimes reduce rather than increase the chance of recall?
- **Quantum mechanism.** Cues as measurements with different bases: two similar cues are close in Hilbert space, so the paths they open to the same trace interfere.
- **Classical baseline.** Independent-cue models: the probability of recall under both cues follows from the single-cue probabilities by the law of total probability.
- **Difficulty.** intermediate
- **Keywords.** memory, recall, cue interference, similarity, retrieval

| parameter | kind | default | bounds | fittable | meaning |
| --- | --- | --- | --- | --- | --- |
| `trace_angle` | float | 0.6 | 0.0 to 3.141592653589793 | no | Bloch angle of the stored trace: the strength of the encoded memory. |
| `cue_angle` | float | 0.7 | 0.0 to 3.141592653589793 | yes | Angle of the retrieval cue relative to the trace. |
| `recall_angle` | float | 1.5707963267948966 | 0.0 to 3.141592653589793 | yes | Angle of the recall question; it is the question that can be interfered with. |
| `mode` | choice (single cue, competing cues) | 'competing cues' | - | no | Which framing to report. |

**What it does not claim.**

- This is a model of response probabilities, not a claim about what the brain does.
- A quantum model only means the data are described by a quantum probability space; classical models may fit equally well, and the classical baseline is reported for that reason.
- The model describes recall probabilities, not the neural mechanism of retrieval.

**References.**

- Khrennikov (2006), Quantum-like brain: interference of minds, BioSystems 84, 225-241.
- Busemeyer & Bruza (2012), Quantum Models of Cognition and Decision, ch. 11 (memory).

```bash
qcog run associative-memory
qcog info associative-memory
```

### `semantic-ambiguity` - Semantic ambiguity: context fixes a word's sense

*Sentence context rotates the meaning of an ambiguous word; once the sense has been resolved, the context can no longer change it.*

- **Cognitive question.** How does the surrounding sentence select one sense of an ambiguous word, and why does asking about the sense first lock the interpretation in?
- **Quantum mechanism.** Senses as basis states of a three-level system, context as a unitary transformation acting before the sense question is asked; the order matters because the sense question is a measurement.
- **Classical baseline.** Context as prior re-weighting: the sense distribution is updated by the context, and the same posterior is available before or after any other judgement.
- **Difficulty.** intermediate
- **Keywords.** semantic ambiguity, context, sense selection, order, language

| parameter | kind | default | bounds | fittable | meaning |
| --- | --- | --- | --- | --- | --- |
| `sense_1_weight` | float | 0.7 | 0.05 to 0.95 | no | Initial amplitude weight of the dominant sense. |
| `context_strength` | float | 0.9 | 0.0 to 1.5707963267948966 | no | How strongly the sentence context rotates the sense space. |
| `context_scope` | choice (narrow, broad) | 'narrow' | - | no | Which pair of senses the context mixes. |

**What it does not claim.**

- This is a model of response probabilities, not a claim about what the brain does.
- A quantum model only means the data are described by a quantum probability space; classical models may fit equally well, and the classical baseline is reported for that reason.
- Three senses and one rotation angle are a schematic; real lexical disambiguation needs a corpus.

**References.**

- Kintsch (2001), Predication, Cognitive Science 25, 173-202.
- Blutner, Pothos & Bruza (2013), A quantum probability perspective on borderline vagueness, Topics in Cognitive Science 5, 711-736.

```bash
qcog run semantic-ambiguity
qcog info semantic-ambiguity
```

### `sequential-categorization` - Sequential categorisation: why repeated labels stabilise a judgement

*Asking the same categorisation question repeatedly makes the answer more stable, not noisier; the model shows the trajectory and the classical baseline that predicts flatness.*

- **Cognitive question.** Does presenting several category labels in a row change how the category is judged, and in which direction?
- **Quantum mechanism.** Repeated projective or unsharp measurements of the same question: the state is driven towards the eigenstate of the answer, the quantum Zeno effect.
- **Classical baseline.** Independence: each judgement is drawn from the same distribution, so the response trajectory is flat in expectation.
- **Difficulty.** intermediate
- **Keywords.** categorisation, repeated measurement, Zeno effect, stabilisation

| parameter | kind | default | bounds | fittable | meaning |
| --- | --- | --- | --- | --- | --- |
| `state_angle` | float | 0.5 | 0.0 to 3.141592653589793 | no | Bloch angle of the initial category belief. |
| `label_angle` | float | 1.5707963267948966 | 0.0 to 3.141592653589793 | no | Angle of the category question; pi/2 makes it maximally incompatible with the belief. |
| `num_labels` | int | 6 | 1 to 12 | no | How many category labels are presented in sequence. The exact engine enumerates one path per label sequence, so the count is capped at twelve; longer sequences can use the sampled backend. |
| `selection_strength` | float | 0.9 | 0.05 to 1.0 | no | Sharpness of each categorisation question (1 = fully decisive, 0 = uninformative). |

**What it does not claim.**

- This is a model of response probabilities, not a claim about what the brain does.
- A quantum model only means the data are described by a quantum probability space; classical models may fit equally well, and the classical baseline is reported for that reason.
- The Zeno effect here is a property of repeated measurement; it does not require the labels to carry new information.
- The exact engine enumerates one path per label sequence (2 per label), so the parameter is capped at twelve labels; beyond that the sampled backend gives the same distribution with shot noise.

**References.**

- Busemeyer, Wang & Pothos (2015), Quantum models of cognition and decision, in The Oxford Handbook of Computational and Mathematical Psychology.
- Kvam, Pleskac, Yu & Busemeyer (2015), Interference effects of choice on confidence, PNAS 112, 10645-10650.

```bash
qcog run sequential-categorization
qcog info sequential-categorization
```

### `conceptual-entanglement` - Conceptual entanglement: joint structure without interference

*Two features can be perfectly correlated while each has the same marginal probability as in a product state: the whole carries structure the parts do not.*

- **Cognitive question.** How can a combined concept have joint structure - features that always co-occur - when neither part determines it?
- **Quantum mechanism.** Nonseparable (entangled) states: correlations in the joint distribution that no product state reproduces, quantified by concurrence and the entropy of entanglement.
- **Classical baseline.** Statistical independence of the two features, p(A and B) = p(A) p(B), which is the best a product state can do.
- **Difficulty.** intermediate
- **Keywords.** entanglement, concepts, correlation, joint typicality, non-separability

| parameter | kind | default | bounds | fittable | meaning |
| --- | --- | --- | --- | --- | --- |
| `combination_angle` | float | 0.7853981633974483 | 0.0 to 1.5707963267948966 | no | pi/4 gives the maximally entangled combination; 0 or pi/2 is a product state. |
| `decoherence` | float | 0.0 | 0.0 to 1.0 | no | Depolarising probability applied to both features (0 = coherent combination). |

**What it does not claim.**

- This is a model of response probabilities, not a claim about what the brain does.
- A quantum model only means the data are described by a quantum probability space; classical models may fit equally well, and the classical baseline is reported for that reason.
- Entanglement here is a property of the modelled concept combination; it is not a claim that concepts are physically entangled.

**References.**

- Aerts & Gabora (2005), A theory of concepts and their combinations, Kybernetes 34, 167-191.
- Bruza, Kitto, Nelson & McEvoy (2009), Extracting spooky-activation-at-a-distance from considerations of entanglement, QI 2009.

```bash
qcog run conceptual-entanglement
qcog info conceptual-entanglement
```

### `contradictory-beliefs` - Contradictory beliefs and contextual measurements

*A self-report and a behavioural question cannot be answered from one joint distribution: the model measures how far apart the two orders are.*

- **Cognitive question.** How can someone sincerely report one attitude and behave differently, without either report being a lie?
- **Quantum mechanism.** Incompatible (non-commuting) questions whose contexts cannot be jointly realised; a contextuality certificate replaces the assumption of a single joint distribution.
- **Classical baseline.** Any classical account requires a joint distribution over self-report and behaviour, so the marginal of the report cannot depend on the order in which the two were elicited.
- **Difficulty.** advanced
- **Keywords.** contradictory beliefs, contextuality, incompatible questions, self-report

| parameter | kind | default | bounds | fittable | meaning |
| --- | --- | --- | --- | --- | --- |
| `state_angle` | float | 0.0 | 0.0 to 3.141592653589793 | no | Where the state sits relative to the self-report question. |
| `report_angle` | float | 0.83 | 0.0 to 3.141592653589793 | yes | Angle of the verbal self-report question. |
| `behaviour_angle` | float | 1.5707963267948966 | 0.0 to 3.141592653589793 | yes | Angle of the behavioural question; the gap to the report drives the contradiction. |
| `sharpness` | float | 1.0 | 0.0 to 1.0 | no | How decisive each question is. |

**What it does not claim.**

- This is a model of response probabilities, not a claim about what the brain does.
- A quantum model only means the data are described by a quantum probability space; classical models may fit equally well, and the classical baseline is reported for that reason.
- A contextuality certificate for the model is not a psychological diagnosis, and the model has no opinion about which of the two questions is the 'real' attitude.

**References.**

- Bruza, Wang & Busemeyer (2015), Contextuality and probability in human cognition and decision, Phil. Trans. R. Soc. A 373.
- Wang, Solloway, Shiffrin & Busemeyer (2014), PNAS 111, 9431-9436.

```bash
qcog run contradictory-beliefs
qcog info contradictory-beliefs
```

## Module C - Strategic decisions and game theory

### `quantum-prisoners-dilemma` - Quantum prisoner's dilemma (EWL scheme)

*Entangling the players before they choose turns mutual cooperation into the equilibrium: (3,3) instead of the classical (1,1), and the advantage fades smoothly with decoherence.*

- **Cognitive question.** Can a shared, correlated context make cooperation rational between partners who are each tempted to defect?
- **Quantum mechanism.** The Eisert-Wilkert-Lewenstein scheme: an entangling gate before the strategy unitaries and its inverse after, so the players can use non-classical strategies that the entanglement makes coherent.
- **Classical baseline.** The classical prisoner's dilemma: defection strictly dominates, mutual defection is the unique Nash equilibrium, and both players receive the punishment payoff.
- **Difficulty.** intermediate
- **Keywords.** prisoner's dilemma, quantum game theory, entanglement, Nash equilibrium, cooperation

| parameter | kind | default | bounds | fittable | meaning |
| --- | --- | --- | --- | --- | --- |
| `entanglement` | float | 1.5707963267948966 | 0.0 to 1.5707963267948966 | no | Angle of the entangling gate: 0 is the classical game, pi/2 is maximal entanglement. |
| `first_strategy` | choice (cooperate (C), defect (D), quantum (Q)) | 'quantum (Q)' | - | no | Strategy used by the row player. |
| `second_strategy` | choice (cooperate (C), defect (D), quantum (Q)) | 'quantum (Q)' | - | no | Strategy used by the column player. |
| `decoherence` | float | 0.0 | 0.0 to 1.0 | no | Depolarising probability applied after the strategy unitaries. |

**What it does not claim.**

- This is a model of response probabilities, not a claim about what the brain does.
- A quantum model only means the data are described by a quantum probability space; classical models may fit equally well, and the classical baseline is reported for that reason.
- This is a game-theoretic model: 'players' need not be people, and the entanglement is part of the game's rules rather than a claim about negotiation between humans.
- Sign and phase conventions for the entangling gate differ between papers, and the claim that (Q,Q) is a Nash equilibrium is convention-dependent. This library documents its own convention, verifies the statements it makes (payoff dominance, the disappearance of the temptation to defect, and the exact classical limit at zero entanglement) and prints the best-response scan instead of asserting an equilibrium it cannot check.

**References.**

- Eisert, Wilkert & Lewenstein (1999), Quantum games and quantum strategies, Physical Review Letters 83, 3077.
- Benjamin & Hayden (2001), Multiplayer quantum games, Physical Review A 64, 030301.

```bash
qcog run quantum-prisoners-dilemma
qcog info quantum-prisoners-dilemma
```

### `negotiation-trust` - Negotiation, trust and shared context

*Model trust as correlation between the two parties: agreement becomes reachable exactly to the extent that the shared context survives decoherence.*

- **Cognitive question.** Why do two parties reach a mutually profitable agreement when each would be worse off if the other backed out?
- **Quantum mechanism.** Entanglement as shared context: correlated outcomes let both parties choose the risky, profitable strategy without communicating, and decoherence models suspicion or miscommunication.
- **Classical baseline.** The mixed-strategy equilibrium of the stag-hunt game, in which agreement is reached less often than the payoff-dominant outcome.
- **Difficulty.** intermediate
- **Keywords.** negotiation, trust, coordination, entanglement, stag hunt

| parameter | kind | default | bounds | fittable | meaning |
| --- | --- | --- | --- | --- | --- |
| `shared_context` | float | 1.5707963267948966 | 0.0 to 1.5707963267948966 | no | Entanglement available to the two parties before they choose. |
| `first_stance` | float | 0.0 | 0.0 to 3.141592653589793 | no | How close the first party is to the cooperative strategy. |
| `second_stance` | float | 0.0 | 0.0 to 3.141592653589793 | no | How close the second party is to the cooperative strategy. |
| `suspicion` | float | 0.05 | 0.0 to 1.0 | no | Decay of the shared context between committing and choosing. |

**What it does not claim.**

- This is a model of response probabilities, not a claim about what the brain does.
- A quantum model only means the data are described by a quantum probability space; classical models may fit equally well, and the classical baseline is reported for that reason.
- 'Trust' here is exactly the correlation between the two parties' strategic states; the model does not claim that trust is entanglement.

**References.**

- Eisert, Wilkert & Lewenstein (1999), Quantum games and quantum strategies, PRL 83, 3077.
- Skyrms (2004), The Stag Hunt and the Evolution of Social Structure.

```bash
qcog run negotiation-trust
qcog info negotiation-trust
```

### `evolution-of-cooperation` - Evolution of cooperation under quantum strategies

*Run a replicator dynamic on the payoff matrix induced by the quantum game: the population flips from mutual defection to mutual cooperation as the entanglement available in each interaction grows.*

- **Cognitive question.** If strategies are copied according to success, does cooperation survive in a population, and does a quantum strategy dominate a classical one?
- **Quantum mechanism.** Coherent strategy space in repeated interactions: the payoff matrix used by the evolutionary dynamics is computed from the quantum game, so entanglement changes which strategy is fittest.
- **Classical baseline.** The replicator dynamic on the classical prisoner's dilemma payoff matrix, where defection takes over from any interior starting point.
- **Difficulty.** advanced
- **Keywords.** evolution of cooperation, replicator dynamics, quantum strategies, population

| parameter | kind | default | bounds | fittable | meaning |
| --- | --- | --- | --- | --- | --- |
| `entanglement` | float | 1.5707963267948966 | 0.0 to 1.5707963267948966 | no | Entanglement available in each interaction. |
| `generations` | int | 40 | 5 to 200 | no | Number of replicator steps. |
| `selection_strength` | float | 1.0 | 0.0 to 3.0 | no | How strongly payoffs translate into reproduction (0 = neutral drift). |
| `mutation` | float | 0.02 | 0.0 to 0.5 | no | Probability per generation of switching to a random strategy. |
| `start` | choice (mostly defectors, uniform, mostly cooperators) | 'mostly defectors' | - | no | Initial population mix. |

**What it does not claim.**

- This is a model of response probabilities, not a claim about what the brain does.
- A quantum model only means the data are described by a quantum probability space; classical models may fit equally well, and the classical baseline is reported for that reason.
- The evolutionary dynamics is classical; only the payoff matrix comes from the quantum game. This hybrid is deliberate and is the standard way to ask an evolutionary question about a quantum game.

**References.**

- Eisert, Wilkert & Lewenstein (1999), Quantum games and quantum strategies, PRL 83, 3077.
- Taylor & Jonker (1978), Evolutionary stable strategies and game dynamics, Mathematical Biosciences 40, 145-156.

```bash
qcog run evolution-of-cooperation
qcog info evolution-of-cooperation
```

### `auctions` - Auctions, competitive bidding and the order of value and bid

*Asking for a valuation and asking for a bid are incompatible questions, so the order changes the bid - and rival information rotates the belief before either is asked.*

- **Cognitive question.** Why are bids correlated with the order in which a bidder thinks about the item's value and about the competition?
- **Quantum mechanism.** A three-level valuation question and a dichotomous bidding question that do not commute; a unitary rotation represents the information about rivals.
- **Classical baseline.** Risk-neutral bidding: the bid is a deterministic function of the (estimated) value, so no presentation order can change it.
- **Difficulty.** intermediate
- **Keywords.** auctions, bidding, winner's curse, order effects, incompatible questions

| parameter | kind | default | bounds | fittable | meaning |
| --- | --- | --- | --- | --- | --- |
| `valuation_confidence` | float | 0.55 | 0.05 to 0.95 | no | How strongly the belief leans to the high-value answers. |
| `rival_pressure` | float | 0.6 | 0.0 to 1.5707963267948966 | no | Rotation of the belief caused by thinking about competing bidders. |
| `bid_aggressiveness` | float | 0.8 | 0.1 to 2.0943951023931953 | no | Angle of the bidding question; small values mean an eager bidder. |

**What it does not claim.**

- This is a model of response probabilities, not a claim about what the brain does.
- A quantum model only means the data are described by a quantum probability space; classical models may fit equally well, and the classical baseline is reported for that reason.
- The valuation question is a three-outcome model of a continuous estimate; the third outcome stands for the high-value region.

**References.**

- Busemeyer & Bruza (2012), Quantum Models of Cognition and Decision, ch. 12 (economic applications).
- Thaler (1988), Anomalies: the winner's curse, Journal of Economic Perspectives 2, 191-202.

```bash
qcog run auctions
qcog info auctions
```

### `coordination-games` - Coordination games: conventions without communication

*A shared context lets two players pick the same convention without communicating; as the context decays, coordination falls back to the classical coin-flip.*

- **Cognitive question.** How do two agents who cannot talk still end up at the same convention?
- **Quantum mechanism.** Correlated strategy choice through entanglement: both players' moves are perfectly correlated in the appropriate strategic basis, with no signalling.
- **Classical baseline.** The symmetric mixed-strategy equilibrium of the coordination game, which coordinates with probability 1/2.
- **Difficulty.** intermediate
- **Keywords.** coordination, convention, correlated equilibrium, entanglement, no communication

| parameter | kind | default | bounds | fittable | meaning |
| --- | --- | --- | --- | --- | --- |
| `shared_context` | float | 1.5707963267948966 | 0.0 to 1.5707963267948966 | no | Strength of the correlation available to both players. |
| `first_bias` | float | 0.0 | 0.0 to 0.7853981633974483 | no | Small bias of the first player towards one convention. |
| `second_bias` | float | 0.0 | 0.0 to 0.7853981633974483 | no | Small bias of the second player towards the same convention. |
| `decoherence` | float | 0.0 | 0.0 to 1.0 | no | Loss of the shared context before the moves are made. |

**What it does not claim.**

- This is a model of response probabilities, not a claim about what the brain does.
- A quantum model only means the data are described by a quantum probability space; classical models may fit equally well, and the classical baseline is reported for that reason.
- The correlation is a feature of the modelled situation (a shared context), not a claim that people share quantum states.

**References.**

- Eisert, Wilkert & Lewenstein (1999), Quantum games and quantum strategies, PRL 83, 3077.
- La Mura (2005), Correlated equilibria of classical strategic games with quantum signals, International Journal of Quantum Information 3, 183-188.

```bash
qcog run coordination-games
qcog info coordination-games
```

## Module D - Groups, social networks and collective behaviour

### `quantum-opinion-dynamics` - Quantum opinion dynamics on a network

*Agents exchange amplitude coherently along a network: opinions converge, agendas become undecided, and the group develops correlations that an averaging model cannot represent.*

- **Cognitive question.** How do opinions spread through a group when each encounter is an exchange rather than an average?
- **Quantum mechanism.** Partial-swap interaction on pairs of qubits: a coherent exchange that creates superposition, plus decoherence as forgetting.
- **Classical baseline.** DeGroot averaging on the same graph: opinions converge to a weighted average and the final spread is the consensus value.
- **Difficulty.** advanced
- **Keywords.** opinion dynamics, partial swap, consensus, network, coherence

| parameter | kind | default | bounds | fittable | meaning |
| --- | --- | --- | --- | --- | --- |
| `num_agents` | int | 5 | 2 to 8 | no | Population size. The simulation is dense (2^n amplitudes), so 8 is the practical interactive limit; the classical baselines cover larger networks. |
| `topology` | choice (ring, line, complete, star, two-communities, random) | 'ring' | - | no | Who talks to whom. |
| `initial_spread` | float | 1.0 | 0.0 to 1.4 | no | How far apart the agents' initial opinions are. |
| `interaction_strength` | float | 0.7853981633974483 | 0.0 to 1.5707963267948966 | no | Angle of the partial swap: how much of each opinion is exchanged per encounter. |
| `decoherence` | float | 0.03 | 0.0 to 0.5 | no | Per-agent depolarising probability per round (forgetting). |
| `rounds` | int | 8 | 1 to 20 | no | Number of interaction rounds. |

**What it does not claim.**

- This is a model of response probabilities, not a claim about what the brain does.
- A quantum model only means the data are described by a quantum probability space; classical models may fit equally well, and the classical baseline is reported for that reason.
- The dense simulation limits the population to ten agents; larger networks need the classical baselines or a tensor-network implementation.

**References.**

- DeGroot (1974), Reaching a consensus, Journal of the American Statistical Association 69, 118-121.
- Khrennikov (2006), Quantum-like brain: interference of minds, BioSystems 84, 225-241.

```bash
qcog run quantum-opinion-dynamics
qcog info quantum-opinion-dynamics
```

### `polarization` - Polarisation and information bubbles

*Two communities with mostly internal interaction can converge inside and diverge between - or melt into undecided superpositions, depending on how coherent the exchange is.*

- **Cognitive question.** Why do groups with the same information drift apart, and what changes when the exchange between them is coherent rather than an average?
- **Quantum mechanism.** Partial-swap exchange restricted by network structure, with decoherence as the mechanism that converts coherent uncertainty into a stable position.
- **Classical baseline.** Hegselmann-Krause bounded confidence on the same graph, which preserves clusters, and DeGroot averaging, which erases them.
- **Difficulty.** advanced
- **Keywords.** polarisation, echo chamber, bounded confidence, networks, consensus

| parameter | kind | default | bounds | fittable | meaning |
| --- | --- | --- | --- | --- | --- |
| `num_agents` | int | 6 | 4 to 8 | no | Even population split into two communities. The dense simulation supports up to eight agents interactively. |
| `community_gap` | float | 1.2 | 0.2 to 1.5 | no | How far apart the two communities start. |
| `interaction_strength` | float | 0.7853981633974483 | 0.0 to 1.5707963267948966 | no | Partial-swap strength inside each community. |
| `cross_talk` | float | 0.15 | 0.0 to 1.0 | no | Weight of the single bridge between the communities. |
| `decoherence` | float | 0.0 | 0.0 to 0.5 | no | Forgetting per round; higher values make positions settle. |
| `rounds` | int | 10 | 1 to 20 | no | Number of interaction rounds. |

**What it does not claim.**

- This is a model of response probabilities, not a claim about what the brain does.
- A quantum model only means the data are described by a quantum probability space; classical models may fit equally well, and the classical baseline is reported for that reason.
- The model says nothing about which side is right and treats the initial positions as given.

**References.**

- Hegselmann & Krause (2002), Opinion dynamics and bounded confidence, JASSS 5(3).
- DeGroot (1974), Reaching a consensus, JASA 69, 118-121.

```bash
qcog run polarization
qcog info polarization
```

### `collective-judgment` - Collective judgement: the wisdom of crowds and correlation

*A group is only wiser than its members to the extent that its members are independent; the model quantifies how much correlation destroys the advantage.*

- **Cognitive question.** How much better is a majority judgement than an individual judgement, and what does correlation between the judges do to that advantage?
- **Quantum mechanism.** A GHZ-style correlated state mixed with a product state; a single parameter moves the group from independent to perfectly correlated.
- **Classical baseline.** Condorcet's jury theorem: independent voters with accuracy p give a majority accuracy that rises towards 1 with the group size.
- **Difficulty.** intermediate
- **Keywords.** collective judgement, Condorcet, wisdom of crowds, correlation, aggregation

| parameter | kind | default | bounds | fittable | meaning |
| --- | --- | --- | --- | --- | --- |
| `num_agents` | int | 7 | 3 to 9 | no | Odd group size (3 to 9). |
| `individual_accuracy` | float | 0.7 | 0.5 to 0.95 | no | Probability that a single judge is right. |
| `correlation` | float | 0.5 | 0.0 to 1.0 | no | 0 = independent judges, 1 = all judges share one state. |

**What it does not claim.**

- This is a model of response probabilities, not a claim about what the brain does.
- A quantum model only means the data are described by a quantum probability space; classical models may fit equally well, and the classical baseline is reported for that reason.
- Correlation is modelled as a GHZ-style state; real groups correlate for many reasons, and the model does not distinguish between them.
- A majority means strictly more than half, so an evenly split even-sized group has no majority; the classical baseline uses the same convention.

**References.**

- Condorcet (1785), Essai sur l'application de l'analyse a la probabilite des decisions rendues a la pluralite des voix.
- Ladha (1992), The Condorcet jury theorem, dependent voters and correlated votes, Public Choice 74, 331-342.

```bash
qcog run collective-judgment
qcog info collective-judgment
```

### `information-cascades` - Information cascades and the order of private and social information

*In a sequential decision the group's history can drown out a private signal; the model shows how strongly the decision depends on which of the two is consulted first.*

- **Cognitive question.** Why does an agent who has private evidence follow the herd, and why does it matter whether the group or the evidence is consulted first?
- **Quantum mechanism.** Two incompatible questions per agent - the private signal and the social context - asked in a specified order, with the social context encoded as a rotation that grows with the observed majority.
- **Classical baseline.** Rational Bayesian social learning: an agent uses Bayes' rule on the observed history and its private signal, which makes the decision independent of the order in which they are consulted.
- **Difficulty.** advanced
- **Keywords.** information cascade, herd behaviour, social learning, order effects, Bayes

| parameter | kind | default | bounds | fittable | meaning |
| --- | --- | --- | --- | --- | --- |
| `num_agents` | int | 6 | 3 to 12 | no | How long the decision sequence is. |
| `signal_accuracy` | float | 0.65 | 0.5 to 0.95 | no | How informative a private signal is on its own. |
| `social_pressure` | float | 0.9 | 0.0 to 1.5707963267948966 | no | How strongly the observed majority rotates an agent's belief. |
| `order` | choice (signal first, group first) | 'signal first' | - | no | Which question the agent puts to itself first. |
| `phase` | float | 0.4 | -3.141592653589793 to 3.141592653589793 | no | Phase between the signal's components; it sets how much interference the signal carries. |

**What it does not claim.**

- This is a model of response probabilities, not a claim about what the brain does.
- A quantum model only means the data are described by a quantum probability space; classical models may fit equally well, and the classical baseline is reported for that reason.
- The social context is modelled as a rotation of the belief, not as Bayesian evidence; the model is a description of order-sensitive decision making, not a claim that people are irrational.

**References.**

- Bikhchandani, Hirshleifer & Welch (1992), A theory of fads, fashion, custom and cultural change as informational cascades, Journal of Political Economy 100, 992-1026.
- Busemeyer & Bruza (2012), Quantum Models of Cognition and Decision, ch. 12.

```bash
qcog run information-cascades
qcog info information-cascades
```

### `deliberation-voting` - Deliberation, plurality and Condorcet cycles

*Deliberation is a unitary change of the belief state: it can change the plurality winner outright, while the pairwise comparisons are reported alongside as the classical consistency check.*

- **Cognitive question.** Does discussing an issue before voting change which option wins, and can a group prefer A to B, B to C and C to A at the same time?
- **Quantum mechanism.** A three-level preference state, a unitary deliberation operator, and a three-outcome measurement for the vote; pairwise preferences are further questions on the same state.
- **Classical baseline.** Classical preference aggregation: a fixed preference ordering per voter, so pairwise comparisons are transitive for each voter and the aggregate is only cyclic if the voters disagree (Condorcet).
- **Difficulty.** intermediate
- **Keywords.** voting, deliberation, Condorcet, cycle, plurality

| parameter | kind | default | bounds | fittable | meaning |
| --- | --- | --- | --- | --- | --- |
| `first_option_weight` | float | 0.55 | 0.05 to 0.9 | no | Initial preference weight of option A. |
| `second_option_weight` | float | 0.3 | 0.02 to 0.9 | no | Initial preference weight of option B (the rest goes to C). |
| `deliberation_strength` | float | 0.8 | 0.0 to 3.141592653589793 | no | Angle of the deliberation operator, which mixes the top two options. |
| `phase` | float | 0.0 | -3.141592653589793 to 3.141592653589793 | no | Relative phase between the options, which is what makes the pairwise comparisons order-dependent. |

**What it does not claim.**

- This is a model of response probabilities, not a claim about what the brain does.
- A quantum model only means the data are described by a quantum probability space; classical models may fit equally well, and the classical baseline is reported for that reason.
- The three options are modelled as basis states of one voter's belief; a real electorate needs an aggregation model on top of this.
- With this library's pairwise questions (coherent comparisons of two options) the majority relation comes out transitive; a Condorcet cycle needs context-dependent pairwise questions or aggregation across voters, and the model reports whether a cycle occurred rather than assuming one.

**References.**

- Condorcet (1785), Essai sur l'application de l'analyse a la probabilite des decisions rendues a la pluralite des voix.
- Busemeyer & Bruza (2012), Quantum Models of Cognition and Decision, ch. 12 (voting).

```bash
qcog run deliberation-voting
qcog info deliberation-voting
```
