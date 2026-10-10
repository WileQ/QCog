"""A real-world survey, end to end, with the circuits drawn.

    python demos/examples/06_economy_and_vote.py            # print everything, save the images

The survey has two questions:

    Question A: "Are you satisfied with the economy?"
    Question B: "Will you vote for the current administration?"

Half the respondents are asked A first, half are asked B first.  In a classical world that
cannot matter: both answers come from one joint distribution, so the marginal of A is the
same either way.  In this model the two questions are *incompatible* - answering one changes
the state the next one is put to - so the order does matter, by an amount the model
predicts.  The script prints the numbers, then draws the circuits that produce them, with
the questions' real names on the gates.

The same model ships as ``qcog run question-order``; here we build the design by hand so the
questions can carry their survey names, and so the circuits can be shown next to the numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import qcog  # noqa: E402
from qcog.analysis import interference_decomposition, order_effect, qq_equality  # noqa: E402
from qcog.circuits import build_prefix_circuit, build_sequence_circuit, circuit_text_alternative  # noqa: E402
from qcog.core import Engine, amplitudes, question, run_sequence  # noqa: E402
from qcog.core.classical import law_of_total_probability  # noqa: E402
from qcog.experiments import TwoQuestionExperiment  # noqa: E402
from qcog.viz import circuit_figure, savefig  # noqa: E402

# --------------------------------------------------------------------------- the survey
# The respondent's disposition towards the administration, as an angle from "against":
# |0> is firmly against, |1> is firmly in favour, and this respondent leans in favour
# without being decided (a superposition, not a hidden answer).
RESPONDENT = 1.0  # radians
# The two questions, as directions in that same space.  Their angle *difference* decides
# how incompatible they are (here they are far apart, which is what makes the order matter).
ECONOMY_ANGLE = 0.85
VOTE_ANGLE = 1.50

import numpy as np  # noqa: E402

belief = amplitudes(np.cos(RESPONDENT / 2), np.sin(RESPONDENT / 2), label="respondent belief")
economy = question(ECONOMY_ANGLE, 0.0, "satisfied with the economy?", ("satisfied", "dissatisfied"),
                   "Question A of the survey")
vote = question(VOTE_ANGLE, 0.0, "vote for the administration?", ("would vote", "would not vote"),
                "Question B of the survey")

design = TwoQuestionExperiment(belief, economy, vote)
engine = Engine(backend="exact")

print("=" * 92)
print("Survey: A = 'Are you satisfied with the economy?'   B = 'Will you vote for the current administration?'")
print("=" * 92)
print(f"Respondent's belief angle      : {RESPONDENT:.2f} rad (leaning in favour, not decided)")
print(f"Question incompatibility       : {design.incompatibility:.3f}"
      f"   (0 would mean the two questions are classically compatible)")
print()

# ------------------------------------------------------------------- both orders
results = design.run_both_orders(engine)
forward, backward = results["A_then_B"], results["B_then_A"]
report = order_effect(forward, backward, economy.name, vote.name)

print("WHAT THE TWO QUESTION ORDERS PREDICT")
print(f"  asked A first : p(satisfied with the economy) = {forward.marginal(economy.name)['satisfied']:.3f}")
print(f"  asked B first : p(satisfied with the economy) = {backward.marginal(economy.name)['satisfied']:.3f}")
print(f"  shift         : {report.max_marginal_shift():+.3f}"
      f"   (total variation between the two orders {report.total_variation:.3f})")
print()
print("  The first answer also moves the second:")
for outcome in ("satisfied", "dissatisfied"):
    conditional = forward.conditional_table(economy.name, vote.name)[outcome]
    print(f"    among respondents who said '{outcome}': "
          f"p({vote.name} = would vote) = {conditional['would vote']:.3f}")
print()

# ------------------------------------------------- the classical account, and the tests
classical = law_of_total_probability(
    economy.name, "satisfied",
    forward.marginal(economy.name)["satisfied"],
    backward.marginal(vote.name),
    backward.conditional_table(vote.name, economy.name),
    conditional_on=vote.name,
)
qq = qq_equality(forward, backward, economy.name, vote.name)
interference = interference_decomposition(belief, economy, vote)

print("WHAT A CLASSICAL ACCOUNT IS FORCED TO PREDICT")
print("  one joint distribution over the two answers, so the marginal of A cannot depend on the order:")
print(f"    measured directly          : {classical.direct:.3f}")
print(f"    computed from the other order: {classical.indirect:.3f}")
print(f"    violation                   : {classical.violation:+.3f}   (a classical model: exactly 0)")
print()
print("WHERE THE PROBABILITY COMES FROM")
print("  " + interference.summary())
print()
print("THE PARAMETER-FREE TEST")
print("  " + qq.summary())
print()

# ------------------------------------------------------- the same numbers, with noise
sampled = run_sequence(belief, [economy, vote], backend="aer", shots=8000, seed=7)
sampled_backward = run_sequence(belief, [vote, economy], backend="aer", shots=8000, seed=7)
sampled_qq = qq_equality(sampled, sampled_backward, economy.name, vote.name)
sampled_report = order_effect(sampled, sampled_backward, economy.name, vote.name)
print("ON A SIMULATOR, WITH 8000 RESPONDENTS PER ORDER")
print(f"  order effect {sampled_report.max_marginal_shift():+.3f} (exact: {report.max_marginal_shift():+.3f})")
print(f"  " + sampled_qq.summary())
print()

# --------------------------------------------------------------- the circuits
out = ROOT / "out" / "economy_vote"
out.mkdir(parents=True, exist_ok=True)
circuits = {
    "A then B (the whole interview)": build_sequence_circuit(belief, [economy, vote]),
    "B then A (the whole interview)": build_sequence_circuit(belief, [vote, economy]),
    "branch: A = satisfied": build_prefix_circuit(belief, [economy, vote], ("satisfied",)),
    "branch: A = dissatisfied": build_prefix_circuit(belief, [economy, vote], ("dissatisfied",)),
}
print("THE CIRCUITS")
for name, circuit in circuits.items():
    print(f"  {name}")
    print(f"    {circuit_text_alternative(circuit)}")
    path = out / (name.split(" (")[0].replace(" ", "_").replace(":", "").replace("=", "") + ".png")
    savefig(circuit_figure(circuit, title=name), path)
    print(f"    image: {path.relative_to(ROOT)}")
print()
print("READING THE DRAWINGS")
print("  * the first box prepares the respondent's belief - a superposition, not a hidden answer;")
print("  * the next box rotates the state into the basis of 'satisfied with the economy?' and the")
print("    measurement records the answer; for all but the last question the rotation is undone,")
print("    so the second question is put to the *collapsed* state the first answer left behind -")
print("    that collapse is the whole reason the order matters;")
print("  * the branch circuits need no mid-circuit measurement at all: they prepare the state that")
print("    follows from a given first answer and then ask the second question, which is the form to")
print("    submit to hardware with limited mid-circuit measurement support.")
print()

# ------------------------------------------------- and the same thing, packaged
packaged = qcog.get_model("question-order").run(
    state_angle=RESPONDENT, angle_a=ECONOMY_ANGLE, angle_b=VOTE_ANGLE, sweeps=False
)
print("THE SAME MODEL, PACKAGED")
print(f"  {packaged.summary}")
print(f"  (qcog run question-order --param state_angle={RESPONDENT} --param angle_a={ECONOMY_ANGLE}"
      f" --param angle_b={VOTE_ANGLE})")
print()
print("WHAT THIS DOES NOT CLAIM")
for caveat in packaged.caveats:
    print(f"  - {caveat}")