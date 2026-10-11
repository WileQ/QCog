"""Q-Cog in five minutes: a belief state, two questions, two orders.

Run it with ``python demos/examples/01_first_steps.py``. Nothing here needs a network connection or
an account.

A respondent has an attitude you cannot see. You ask them one question, then a second, related one -
and then you ask the same two questions in the opposite order to somebody else. A classical account
says the order cannot matter: both answers come out of one joint distribution, so the marginal of the
first question is fixed. This script builds that situation and prints, in order: the two question
orders, the size of the order effect and why it happens, where the probability of the second answer
comes from (a classical part and an interference part), the parameter-free QQ equality, what a
classical model would have had to say about the same data, and finally the same design run on a
compiled Qiskit circuit with shot noise.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qcog import (  # noqa: E402
    basis_state,
    explain_order_effect,
    interference_decomposition,
    order_effect,
    question,
    qq_equality,
    run_sequence,
)

# 1. A person holds a definite belief about a proposition: here, the state |0>.
belief = basis_state(0, 2, label="belief")

# 2. Two questions. A is "moderately" incompatible with the belief basis, B is the
#    question that maximally reorients it.
question_a = question(0.955, 0.0, "A", description="a satisfaction question")
question_b = question(3.14159 / 2, 0.0, "B", description="a voting-intention question")

# 3. Ask them in both orders.
forward = run_sequence(belief, [question_a, question_b])
backward = run_sequence(belief, [question_b, question_a])

report = order_effect(forward, backward, "A", "B")
print("The order effect")
print(report.summary())
print(explain_order_effect(report))
print()

# 4. Where does p(B) come from?  Split it into a classical part and interference.
interference = interference_decomposition(belief, question_a, question_b)
print("Where the probability comes from")
print(interference.summary())
print("decomposition is exact:", interference.check())
print()

# 5. The parameter-free prediction: same-answer agreement is order independent.
qq = qq_equality(forward, backward, "A", "B")
print("The parameter-free prediction (QQ equality)")
print(qq.summary())
print()

# 6. What a classical model would have said.
print("Classical requirement: the marginal of A is the same in both orders.")
for outcome in ("yes", "no"):
    first = forward.marginal("A")[outcome]
    second = backward.marginal("A")[outcome]
    print(f"  p(A = {outcome}): {first:.3f} when A is asked first, {second:.3f} when it is asked second")

# 7. The same model on a real circuit (Qiskit Aer, with shot noise).
shots = run_sequence(belief, [question_a, question_b], backend="aer", shots=20000, seed=7)
print()
print("Compiled circuit run:", shots.summary())
print("Circuit:", shots.metadata["circuit"])