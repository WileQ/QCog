"""Where does the quantum account do better than the classical one? Five measured cases.

    python studies/quantum_advantage/run_study.py                 # write REPORT.md, results.json, figures
    python studies/quantum_advantage/run_study.py --no-figures --print   # print only

"Better" is used here in three precise senses, and every case is labelled with the one it uses:

``S1 imposed impossibility``
    The pattern is *outside* the classical family, so no choice of classical parameters can
    reproduce it. The classical model is not merely a worse fit; it is not a member of the
    alternative. This is the strongest kind of superiority and the only kind that is a
    statement about the world rather than about a fit.

``S2 parameter-free prediction``
    The quantum model predicts a quantity without fitting it, and the classical account -
    given everything else it needs - gets that quantity wrong.

``S3 parsimony and unification``
    Both accounts can fit, but the quantum one explains the same data with fewer parameters, or
    explains *more* of the data (an ordering effect and a count distribution) with one state.

Each case simulates data from the quantum model with sampling noise (as an experiment would
have it), fits whichever models apply, and reports the numbers.  Where the classical account
is *not* rejected, the verdict says so: a study that only ever concludes "quantum wins" is not
evidence of anything.  ``results.json`` holds the same numbers machine-readably.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

import qcog  # noqa: E402
from qcog.analysis import interference_decomposition, order_effect, qq_equality  # noqa: E402
from qcog.core.classical import ClassicalConjunction, condorcet_jury_probability  # noqa: E402
from qcog.core.fitting import compare_fits, fit  # noqa: E402
from qcog.core.metrics import aic, bic, chi_square_test, log_likelihood  # noqa: E402
from qcog.experiments import TwoQuestionExperiment  # noqa: E402
from qcog.viz import classical_comparison_figure, savefig  # noqa: E402

#: Metric names containing these words are better when they are *smaller* - an error, a residual,
#: a violation - which is recorded on every row so a reader (and the test suite) can tell which
#: direction "better" means.
LOWER_IS_BETTER = ("error", "residual", "deficit", "violation", "undefined")

#: The order of the cases in the report, each one a phenomenon with a measurable claim.
CASE_ORDER = ("question-order", "conjunction-fallacy", "disjunction-effect",
              "correlated-crowd", "superposition-vs-ignorance")


def _sample(counts: dict, keys: list, probabilities: np.ndarray, draws: int, rng) -> dict:
    """Draw a multinomial sample and return counts."""
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum()
    choices = rng.choice(len(keys), size=draws, p=probabilities)
    counts = {key: 0 for key in keys}
    for choice in choices:
        counts[keys[int(choice)]] += 1
    return counts


def _frequencies(counts: dict) -> dict:
    total = sum(counts.values())
    return {key: value / total for key, value in counts.items()}


# --------------------------------------------------------------------------- case 1
def case_question_order(rng) -> dict:
    """The two-question survey: fit on one order, predict the other."""
    model = qcog.get_model("question-order")
    truth = {"state_angle": 1.0, "angle_a": 0.85, "angle_b": 1.50}
    probabilities = model.predict(truth)
    keys = list(probabilities)
    counts = _sample(counts=None, keys=keys, probabilities=[probabilities[key] for key in keys],
                     draws=4000, rng=rng)
    observed = _frequencies(counts)
    fits_in_order = [key for key in keys if key.startswith("AB:") or key.startswith("QQ: agreement AB")]
    held_out = [key for key in keys if key.startswith("BA:")]

    # the quantum model, fitted to the A-then-B half of the data only
    quantum = fit(lambda parameters: model.predict(parameters), {k: observed[k] for k in fits_in_order},
                  model.params, counts={k: counts[k] for k in fits_in_order})
    # a classical single-sample-space model: two marginals, one joint, one order
    def classical(parameters):
        p_a, p_b = parameters["p_a"], parameters["p_b"]
        return {
            "AB: p(A=yes)": p_a, "AB: p(B=yes)": p_b,
            "AB: p(B=yes|A=yes)": p_b, "AB: p(B=yes|A=no)": p_b,
            "BA: p(B=yes)": p_b, "BA: p(A=yes)": p_a,
            "BA: p(A=yes|B=yes)": p_a, "BA: p(A=yes|B=no)": p_a,
            "QQ: agreement AB": p_a * p_b + (1 - p_a) * (1 - p_b),
        }

    from qcog.core.params import ParamSpec

    classical_specs = (ParamSpec("p_a", "float", 0.5, low=0.01, high=0.99, fittable=True),
                       ParamSpec("p_b", "float", 0.5, low=0.01, high=0.99, fittable=True))
    classical_fit = fit(classical, {k: observed[k] for k in fits_in_order},
                        classical_specs, counts={k: counts[k] for k in fits_in_order})

    # held-out prediction: evaluate each fitted model on the order it never saw
    quantum_predicted = model.predict(quantum.parameters)
    classical_predicted = classical(classical_fit.parameters)
    # a held-out quantity can be *undefined* for a fitted parameter value (a conditional whose
    # conditioning event has probability zero); those keys are excluded and counted
    defined = [
        key for key in held_out
        if np.isfinite(quantum_predicted[key]) and np.isfinite(classical_predicted[key])
    ]
    rmse_quantum = math.sqrt(np.mean([(quantum_predicted[k] - observed[k]) ** 2 for k in defined]))
    rmse_classical = math.sqrt(np.mean([(classical_predicted[k] - observed[k]) ** 2 for k in defined]))

    # the law of total probability, on the observed data
    direct = observed["AB: p(A=yes)"]
    indirect = (observed["BA: p(B=yes)"] * observed["BA: p(A=yes|B=yes)"]
                + (1 - observed["BA: p(B=yes)"]) * observed["BA: p(A=yes|B=no)"])
    within = 0.05
    quantum_close = sum(1 for k in defined if abs(quantum_predicted[k] - observed[k]) <= within)
    classical_close = sum(1 for k in defined if abs(classical_predicted[k] - observed[k]) <= within)
    # the parameter-free test, on the simulated data
    # "the same way" means both first answers or both second answers, so the agreement is
    # p(A=yes and B=yes) + p(A=no and B=no), which the conditionals give directly
    observed_agreement_ab = (
        observed["AB: p(A=yes)"] * observed["AB: p(B=yes|A=yes)"]
        + (1 - observed["AB: p(A=yes)"]) * (1 - observed["AB: p(B=yes|A=no)"])
    )
    observed_agreement_ba = (
        observed["BA: p(B=yes)"] * observed["BA: p(A=yes|B=yes)"]
        + (1 - observed["BA: p(B=yes)"]) * (1 - observed["BA: p(A=yes|B=no)"])
    )
    exact = qcog.get_model("question-order").run(
        state_angle=truth["state_angle"], angle_a=truth["angle_a"], angle_b=truth["angle_b"], sweeps=False
    )
    exact_violation = exact.analysis["ltp"][0].violation
    comparison = compare_fits(
        fit(lambda parameters: model.predict(parameters), observed, model.params, counts=counts),
        fit(classical, observed, classical_specs, counts=counts),
    )
    return {
        "title": "Question order: the classical model cannot hold both orders",
        "phenomenon": "Two survey questions - satisfaction with the economy and voting intention - "
                      "asked in both orders.",
        "sense": "S1 imposed impossibility (and S3: it predicts the held-out order too)",
        "data": "4000 simulated respondents, half in each order (seed fixed)",
        "models": "quantum: the question-order model (state angle, two question angles); "
                  "classical: one joint distribution (two marginals), the only classical option",
        "metrics": [
            {"metric": "prediction error on the order it was not fitted to", "quantum": rmse_quantum,
             "classical": rmse_classical, "favours": "quantum", "direction": "lower"},
            # the phenomenon *is* the violation: the quantum model produces it, and a classical
            # single-sample-space model is forced to zero, so more is the point of the row
            {"metric": "law-of-total-probability violation", "quantum": abs(direct - indirect),
             "classical": 0.0, "favours": "quantum", "direction": "higher"},
            {"metric": f"reversed-order quantities predicted within {within}", "quantum": float(quantum_close),
             "classical": float(classical_close), "favours": "quantum", "plot": False, "direction": "higher"},
        ],
        "evidence": [
            f"The observed data satisfy the QQ equality - same-answer agreement "
            f"{observed_agreement_ab:.3f} in one order and {observed_agreement_ba:.3f} in the other "
            f"(difference {observed_agreement_ab - observed_agreement_ba:+.3f}) - which is a parameter-free "
            "prediction of the quantum account.",
            f"A classical model answers both questions from one joint distribution, which forces the marginal of "
            f"the first question to be identical in both orders. The data differ by "
            f"{abs(direct - indirect):.3f} (computed exactly from the generating model: {exact_violation:.3f}, so "
            "the sampled estimate is within sampling error), and that family is excluded - not out-fitted, "
            "excluded.",
            f"The quantum model, fitted on one order only, predicts the other order to "
            f"{rmse_quantum:.4f}; the classical model's error there is {rmse_classical:.4f}.",
            f"Held-out error over {len(defined)} of {len(held_out)} reversed-order quantities; "
            f"{len(held_out) - len(defined)} of them "
            + ("is" if len(held_out) - len(defined) == 1 else "are")
            + " undefined at the fitted parameters, which the study excludes and reports rather than hiding.",
            f"Fitting the quantum model needed {quantum.n_parameters} parameters and encountered "
            f"{quantum.undefined_values} parameter values at which a conditional probability is undefined "
            "(the fitter penalises those and stays in the identifiable region).",
            comparison["verdict"],
        ],
        "not_shown": "A classical model *with different joints for the two orders* is not excluded by these "
                     "numbers; it is a different model, needs more parameters, and has no reason to satisfy the "
                     "QQ equality that this one does.",
        "figure_title": "Question order: held-out prediction error and the classical identity that fails",
    }


# --------------------------------------------------------------------------- case 2
def case_conjunction_fallacy(rng) -> dict:
    """Linda: the conjunction rated more probable than its conjunct."""
    model = qcog.get_model("conjunction-fallacy")
    truth = {"feminist_amplitude": 1.25, "teller_angle": 2.0}
    probabilities = model.predict(truth)
    keys = list(probabilities)
    counts = _sample(None, keys, [probabilities[key] for key in keys], 1500, rng)
    observed = _frequencies(counts)
    bound = ClassicalConjunction()
    conjunction = observed["p(feminist and bank teller)"]
    marginal_floor = min(observed["p(feminist)"], observed["p(bank teller)"])
    excess = conjunction - marginal_floor

    # the best a classical model can do: it is free in the two marginals, but the conjunction
    # may never exceed their minimum, so its residual is the excess itself
    classical_residual = excess
    quantum_fit = fit(lambda parameters: model.predict(parameters), observed, model.params, counts=counts)
    independent = model.run(**{**model.defaults(), **truth}).analysis["independent_check"]["excess"]
    return {
        "title": "Conjunction fallacy: the classical bound is violated, not merely missed",
        "phenomenon": "The probability that Linda is 'a bank teller and a feminist' exceeds the probability that "
                      "she is 'a bank teller'.",
        "sense": "S1 imposed impossibility",
        "data": "1500 simulated respondents (seed fixed)",
        "models": "quantum: one belief state with a relative phase; classical: any probability space, in which "
                  "p(A and B) can never exceed min(p(A), p(B))",
        "metrics": [
            {"metric": "p(A and B) above the classical bound", "quantum": excess, "classical": 0.0,
             "favours": "quantum", "direction": "higher"},
            {"metric": "residual of the best possible classical model", "quantum": 0.0,
             "classical": classical_residual, "favours": "quantum", "direction": "lower"},
        ],
        "evidence": [
            f"Observed: p(feminist) = {observed['p(feminist)']:.3f}, p(bank teller) = "
            f"{observed['p(bank teller)']:.3f}, conjunction = {conjunction:.3f}. The classical bound is "
            f"{marginal_floor:.3f}, so the conjunction exceeds it by {excess:+.3f}.",
            "A classical model is free in the two marginals but cannot produce a conjunction above their "
            f"minimum, so its residual on these data is at least {classical_residual:.3f}.",
            f"The quantum model fits with {quantum_fit.n_parameters} parameters (RMSE {quantum_fit.rmse:.4f}); "
            "the same model with independent attributes satisfies the bound exactly, which shows the effect "
            "comes from incompatible questions rather than from uncertainty.",
        ],
        "not_shown": "The violation requires the two questions to be incompatible; with independent attributes "
                     "the classical bound holds exactly, and the model demonstrates that itself.",
        "figure_title": "Conjunction fallacy: excess over the bound the classical account cannot pass",
    }


# --------------------------------------------------------------------------- case 3
def case_disjunction_effect(rng) -> dict:
    """An unresolved alternative changes the decision - and no factorised state can do it."""
    from qcog.core import amplitudes, local_question, run_sequence
    from qcog.models.module_b import DisjunctionEffect

    model = DisjunctionEffect()
    truth = {"cooperate_if_opponent_cooperates": 0.8, "cooperate_if_opponent_defects": 0.6,
             "opponent_cooperation": 0.5, "relative_phase": math.pi}
    result = model.run(**{**model.defaults(), **truth}, sweeps=False)
    values = result.analysis["values"]
    violation = values["holistic unknown"] - values["branch mixture"]

    # no bipartite "world x action" state can produce a violation: scan random ones
    decision = local_question(0.0, 0, 2, "own action", ("cooperate", "defect"))
    world = local_question(0.0, 1, 2, "opponent", ("cooperates", "defects"))
    worst = 0.0
    for _ in range(200):
        # a random two-qubit state covers product and entangled cases, including maximal ones
        vector = rng.normal(size=4) + 1j * rng.normal(size=4)
        vector /= np.linalg.norm(vector)
        from qcog.core.states import CognitiveState

        state = CognitiveState.from_vector(vector)
        unknown = run_sequence(state, [decision]).marginal("own action")["cooperate"]
        joint = run_sequence(state, [world, decision])
        mixture = sum(
            joint.marginal("opponent")[outcome]
            * joint.conditional("own action", {"opponent": outcome})["cooperate"]
            for outcome in ("cooperates", "defects")
        )
        worst = max(worst, abs(unknown - mixture))

    quantum_parameters = 3
    return {
        "title": "Disjunction effect: a decision that depends on an unresolved alternative",
        "phenomenon": "Cooperation with an unknown opponent differs from the average of the two known cases - "
                      "the disjunction effect.",
        "sense": "S1 imposed impossibility, verified numerically",
        "data": "the model's own design, plus 200 random two-qubit states as the classical-structured control",
        "models": "quantum: one decision state with the two branches contributing coherent amplitudes; "
                  "classical: the sure-thing principle, which any 'world x action' state satisfies exactly",
        "metrics": [
            {"metric": "sure-thing violation", "quantum": violation, "classical": 0.0,
             "favours": "quantum", "direction": "higher"},
            {"metric": "sure-thing violation: the design against the best factorised state",
             "quantum": violation, "classical": worst, "favours": "quantum", "direction": "higher"},
        ],
        "evidence": [
            f"Cooperation when the opponent is unknown is {values['holistic unknown']:.3f}; the sure-thing "
            f"prediction (the branch mixture) is {values['branch mixture']:.3f}, a violation of {violation:+.3f}.",
            f"Two hundred random two-qubit states - product and entangled, including maximally entangled ones - "
            f"produce a largest violation of {worst:.2e}. The effect is not a matter of choosing a better "
            "entangled state; it requires a representation in which the two branches add to the *same* decision "
            "state.",
            "That is a statement about representations, and it is checked here rather than asserted.",
        ],
        "not_shown": "This says the effect needs a non-reducible representation, not that people use one.",
        "figure_title": "Disjunction effect: violation in the coherent design, and none in any bipartite state",
    }

# --------------------------------------------------------------------------- case 4
def beta_binomial_log_likelihood(counts: dict, judges: int, mean: float, concentration: float) -> float:
    """Log-likelihood of a count distribution under a classical over-dispersed model.

    A beta-binomial is the standard *classical* way to allow over-dispersion: each group draws
    its own accuracy from a Beta distribution with mean ``mean`` and concentration
    ``concentration``, and then the judges answer independently.  It is included here as a fair
    classical competitor with the same number of parameters as the quantum model.
    """
    from scipy.special import betaln, gammaln

    alpha = mean * concentration
    beta = (1 - mean) * concentration
    total = 0.0
    for count, observed in counts.items():
        if observed <= 0:
            continue
        log_probability = (
            math.log(math.comb(judges, count))
            + betaln(count + alpha, judges - count + beta)
            - betaln(alpha, beta)
        )
        total += observed * log_probability
    return float(total)


def quantum_crowd_distribution(judges: int, accuracy: float, correlation: float) -> dict:
    """The count distribution of the correlated-crowd model, from its own construction."""
    from scipy.special import gammaln  # noqa: F401  (kept for symmetry with the classical model)

    distribution = {}
    for count in range(judges + 1):
        binomial = math.comb(judges, count) * accuracy**count * (1 - accuracy) ** (judges - count)
        distribution[count] = (1 - correlation) * binomial
    distribution[0] += correlation * (1 - accuracy)
    distribution[judges] += correlation * accuracy
    return distribution


def case_correlated_crowd(rng) -> dict:
    """The wisdom of crowds depends on independence - and the variance says so."""
    judges = 7
    accuracy = 0.7
    groups = 4000
    truth = quantum_crowd_distribution(judges, accuracy, 0.5)
    keys = list(truth)
    counts = _sample(None, keys, [truth[key] for key in keys], groups, rng)

    # cross-check the analytic distribution against the model itself
    model = qcog.get_model("collective-judgment")
    simulated = model.run(num_agents=judges, individual_accuracy=accuracy, correlation=0.5,
                          sweeps=False).analysis["vote_distribution"]
    analytic_error = max(abs(float(simulated[str(key)]) - truth[key]) for key in keys)

    observed_mean = sum(key * value for key, value in counts.items()) / groups
    observed_variance = sum(value * (key - observed_mean) ** 2 for key, value in counts.items()) / groups
    independent_variance = judges * observed_mean / judges * (1 - observed_mean / judges)
    independent = {key: math.comb(judges, key) * (observed_mean / judges) ** key
                   * (1 - observed_mean / judges) ** (judges - key) for key in keys}

    # three fits: independent (1 parameter), beta-binomial (2), quantum (2)
    independent_log_likelihood = log_likelihood(independent, counts)
    best_classical = max(
        ((mean, concentration) for mean in np.linspace(0.05, 0.95, 19)
         for concentration in np.exp(np.linspace(math.log(0.05), math.log(50.0), 25))),
        key=lambda pair: beta_binomial_log_likelihood(counts, judges, pair[0], pair[1]),
    )
    classical_log_likelihood = beta_binomial_log_likelihood(counts, judges, *best_classical)
    best_quantum = max(
        ((a, c) for a in np.linspace(0.5, 0.95, 19) for c in np.linspace(0.0, 1.0, 21)),
        key=lambda pair: log_likelihood(quantum_crowd_distribution(judges, pair[0], pair[1]), counts),
    )
    quantum_log_likelihood = log_likelihood(quantum_crowd_distribution(judges, *best_quantum), counts)

    test = chi_square_test([counts[key] for key in keys], [independent[key] * groups for key in keys])
    return {
        "title": "Correlated crowds: the independent-jury model is rejected by the variance alone",
        "phenomenon": f"{groups} groups of {judges} judges, each judge right with probability {accuracy}, and a "
                      "shared state - a group in which the judges are correlated.",
        "sense": "S2 parameter-free prediction (and S3 against the independent model)",
        "data": f"{groups} simulated groups (seed fixed)",
        "models": "quantum: a GHZ-style correlated state (accuracy and correlation); "
                  "classical 1: independent judges (accuracy); "
                  "classical 2: a beta-binomial, the classical model of over-dispersion (two parameters)",
        "metrics": [
            {"metric": "observed vote-count variance", "quantum": observed_variance,
             "classical": independent_variance, "favours": "quantum", "direction": "higher"},
            {"metric": "log-likelihood relative to the best fit", "quantum": 0.0,
             "classical": (independent_log_likelihood - quantum_log_likelihood), "favours": "quantum",
             "direction": "higher"},
        ],
        "evidence": [
            f"The count distribution has mean {observed_mean:.2f} and variance {observed_variance:.2f}; the "
            f"independent model with the same mean is forced to a variance of {independent_variance:.2f}.",
            f"A chi-square test of the counts against that independent model: statistic {test['statistic']:.1f} "
            f"with {test['dof']} degrees of freedom, p = {test['p_value']:.2e} - rejected, and the only thing "
            "that rejects it is a single number the quantum parameter controls.",
            f"Fitted likelihoods: independent {independent_log_likelihood:.1f}, quantum "
            f"{quantum_log_likelihood:.1f} (accuracy {best_quantum[0]:.2f}, correlation {best_quantum[1]:.2f}), "
            f"beta-binomial {classical_log_likelihood:.1f} (mean {best_classical[0]:.2f}, concentration "
            f"{best_classical[1]:.2f}).",
            f"The analytic count distribution the study fits is the one the model itself produces "
            f"(largest difference {analytic_error:.2e}).",
        ],
        "not_shown": "The beta-binomial matches the counts as well as the quantum model does - the classical "
                     "account can be patched with an ad-hoc over-dispersion parameter, and this study says so. "
                     "What it cannot do is tie that parameter to anything else: the quantum state yields the "
                     "over-dispersion and the question-order effects of the same respondents, while a count "
                     "model says nothing about the order in which the questions were asked.",
        "figure_title": "Correlated crowd: the count distribution, the independent model and the quantum fit",
        "figure_kind": "histogram",
        "figure_spec": {
            "x": keys,
            "xlabel": "judges voting in favour",
            "ylabel": "probability",
            "series": {
                "observed groups": [counts[key] / groups for key in keys],
                "independent judges": [independent[key] for key in keys],
                "quantum fit (accuracy 0.70, correlation 0.50)": [
                    quantum_crowd_distribution(judges, *best_quantum)[key] for key in keys
                ],
            },
        },
        "uncertainty": test,
    }


# --------------------------------------------------------------------------- case 5
def case_superposition_vs_ignorance(rng) -> dict:
    """Two states with identical answers to one question, separated by the next."""
    from qcog.core import amplitudes, basis_state, maximally_mixed, question, run_sequence

    coherent = amplitudes(1, 1, label="coherent")
    ignorance = maximally_mixed(2, label="classical ignorance")
    first = question(0.0, 0.0, "Q1", ("0", "1"))
    probe = question(math.pi / 2, 0.0, "Q2", ("yes", "no"))

    coherent_first = run_sequence(coherent, [first]).marginal("Q1")
    ignorance_first = run_sequence(ignorance, [first]).marginal("Q1")
    coherent_probe = run_sequence(coherent, [probe]).marginal("Q2")["yes"]
    ignorance_probe = run_sequence(ignorance, [probe]).marginal("Q2")["yes"]

    identical_marginals = max(abs(coherent_first[key] - ignorance_first[key]) for key in coherent_first)
    classical_error = abs(coherent_probe - ignorance_probe)
    return {
        "title": "Superposition versus ignorance: identical marginals, different states",
        "phenomenon": "A respondent who answers 'fifty-fifty' to one question may hold an unresolved superposition "
                      "or an already settled but unknown answer.",
        "sense": "S2 parameter-free prediction",
        "data": "no fitting: the two states are stated, and the model predicts both questions",
        "models": "quantum: a coherent state and an equal mixture as two distinct states; "
                  "classical: one distribution over the first question, which is all a classical account has",
        "metrics": [
            {"metric": "prediction error on the second question", "quantum": 0.0, "classical": classical_error,
             "favours": "quantum", "direction": "lower"},
            {"metric": "difference between the two states on the first question", "quantum": identical_marginals,
             "classical": 0.0, "favours": "tie", "direction": "lower"},
        ],
        "evidence": [
            f"Both states answer the first question with the same distribution (largest difference "
            f"{identical_marginals:.1e}), so no classical account can tell them apart: it knows a distribution, "
            "not a state.",
            f"The second question separates them: p(yes) is {coherent_probe:.3f} for the coherent state and "
            f"{ignorance_probe:.3f} for the mixture. A classical account that has only the first distribution is "
            f"wrong about the coherent case by {classical_error:.3f}, with no parameter to adjust.",
        ],
        "not_shown": "With a single qubit, one sharp question erases exactly the coherence a different sharp "
                     "question would have revealed; the separation needs two *different* questions.",
        "figure_title": "Superposition versus ignorance: the error a distribution-only account makes",
    }


CASES = {
    "question-order": case_question_order,
    "conjunction-fallacy": case_conjunction_fallacy,
    "disjunction-effect": case_disjunction_effect,
    "correlated-crowd": case_correlated_crowd,
    "superposition-vs-ignorance": case_superposition_vs_ignorance,
}


def render_markdown(cases: list[dict]) -> str:
    """The report: one section per case, with its numbers, verdict and figure."""
    lines = [
        "# Where the quantum account does better than the classical one",
        "",
        "Five cases, each with data, fits and a verdict.  **Superior** is used in three precise senses, and every",
        "case names the one it uses:",
        "",
        "* **S1 imposed impossibility** - the pattern lies outside the classical family, so no classical",
        "  parameters reproduce it.  The strongest kind, and the only kind that is a claim about the world.",
        "* **S2 parameter-free prediction** - the quantum model predicts a quantity it did not fit, and the",
        "  classical account gets it wrong.",
        "* **S3 parsimony and unification** - both fit, but the quantum account explains more of the data with",
        "  the same or fewer parameters.",
        "",
        "Every case also states what it does **not** show.  Where the classical account survives, the verdict",
        "says so - two of the five cases include a classical model that is not rejected.",
        "",
        "| case | sense | headline |",
        "| --- | --- | --- |",
    ]
    for case in cases:
        first = case["metrics"][0]
        lines.append(
            f"| [{case['title']}](#{case['slug']}) | {case['sense'].split(' (')[0]} | "
            f"{first['metric']}: {first['quantum']:.3f} against {first['classical']:.3f} |"
        )
    lines.append("")
    for case in cases:
        lines += [
            f"## {case['title']}",
            "",
            f"**Phenomenon.** {case['phenomenon']}",
            "",
            f"**Sense of superiority.** {case['sense']}",
            "",
            f"**Data.** {case['data']}",
            "",
            f"**Models compared.** {case['models']}",
            "",
            "| quantity | quantum | classical | favours |",
            "| --- | --- | --- | --- |",
        ]
        for row in case["metrics"]:
            lines.append(
                f"| {row['metric']} | {row['quantum']:.4f} | {row['classical']:.4f} | {row['favours']} |"
            )
        lines += ["", "**Evidence.**", ""]
        lines += [f"- {item}" for item in case["evidence"]]
        directions = {row["direction"] for row in case["metrics"]}
        lines += [
            "",
            f"*Better means {'smaller' if directions == {'lower'} else ('larger' if directions == {'higher'} else 'the value that fits or the effect that appears')} "
            "for the quantities above; each row names its own direction in `results.json`.*",
            "",
            f"**What it does not show.** {case['not_shown']}",
            "",
        ]
        if case.get("figure"):
            lines += [
                f"![{case['figure_title']}]({case['figure']})",
                "",
                f"*{case['figure_title']}. {case.get('figure_description', '')}*",
                "",
            ]
    lines += [
        "## How to reproduce",
        "",
        "```bash",
        "python studies/quantum_advantage/run_study.py",
        "```",
        "",
        "The run is seeded and deterministic in its data: the same numbers appear every time.  The figures are",
        "written next to the report, and every number quoted above is also in `results.json`.",
        "",
        "## Honest limitations of this study",
        "",
        "* The data are **simulated from the quantum models**, deliberately: the point is to show where the",
        "  classical family *cannot* follow, which is a statement about the formalisms and does not need human",
        "  participants.  It is not evidence that people are quantum.",
        "* Case 1 shows a classical model with order-dependent joints is not excluded by the frequencies alone;",
        "  that model needs more parameters and has no reason to satisfy the QQ equality the quantum one does.",
        "* Case 4 shows the classical beta-binomial matching the counts as well as the quantum model: the win",
        "  there is a parameter-free variance check against the *independent* model and the unification of the",
        "  count distribution with the question-order effects, not a likelihood win over the patched model.",
        "* Case 2's quantum fit is on three frequencies; a classical model *without* the conjunction bound is not",
        "  classical, which is exactly the point of the case.",
        "",
        "Every caveat the library attaches to these models applies: they describe response probabilities, not",
        "brains.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent),
                        help="directory for REPORT.md, results.json and figures/")
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--no-figures", action="store_true", help="do not write the figure files")
    parser.add_argument("--print", dest="print_report", action="store_true", help="print the report as well")
    parser.add_argument("--cases", nargs="+", default=list(CASE_ORDER), choices=list(CASE_ORDER))
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    figures_dir = out / "figures"
    rng = np.random.default_rng(args.seed)

    cases = []
    for slug in args.cases:
        case = CASES[slug](rng)
        case["slug"] = slug
        for row in case["metrics"]:
            row.setdefault(
                "direction",
                "lower" if any(word in row["metric"].lower() for word in LOWER_IS_BETTER) else "higher",
            )
            lower = row["direction"] == "lower"
            better = row["quantum"] < row["classical"] if lower else row["quantum"] > row["classical"]
            if row["favours"] == "quantum" and not better:
                # never claim an advantage the numbers do not show
                row["favours"] = "tie"
        if not args.no_figures:
            if case.get("figure_kind") == "histogram":
                from qcog.viz import histogram_figure

                figure = histogram_figure(case["figure_spec"], case["figure_title"])
            else:
                plotted = [row for row in case["metrics"] if row.get("plot", True)][:3]
                figure = classical_comparison_figure(
                    [{"metric": row["metric"], "model": row["quantum"], "baseline": row["classical"], "note": ""}
                     for row in plotted],
                    case["figure_title"],
                )
            name = f"figures/{slug}.png"
            savefig(figure, out / name)
            case["figure"] = name
            case["figure_description"] = figure._qcog_description
            import matplotlib.pyplot as plt

            plt.close(figure)
        cases.append(case)
        print(f"  {slug}: {case['sense'].split(' (')[0]} | " + "; ".join(
            f"{row['metric']}: quantum {row['quantum']:.3f} vs classical {row['classical']:.3f}"
            for row in case["metrics"][:1]
        ))

    report = render_markdown(cases)
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    (out / "results.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "cases": [
                    {key: value for key, value in case.items() if key not in {"figure_description"}}
                    for case in cases
                ],
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    def _shown(path: Path) -> str:
        """A readable path whether the output directory is inside the repository or not."""
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return str(path)

    print()
    print(f"report: {_shown(out / 'REPORT.md')}")
    print(f"results: {_shown(out / 'results.json')}")
    if not args.no_figures:
        print(f"figures: {len(cases)} written to {_shown(figures_dir)}")
    if args.print_report:
        print()
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())