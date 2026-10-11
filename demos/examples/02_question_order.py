"""The flagship model, run three ways: analytically, on a circuit, and with noise.

``python demos/examples/02_question_order.py``

The same two-question design as the first example, run the way a study would run it: once
analytically, once by compiling the design into a Qiskit circuit and reading the result, and once
with depolarising noise applied between the two questions. The three summaries should tell the same
story, and the compiled circuit should reproduce the analytic numbers - that agreement is what makes
the circuit drawings on the model pages worth trusting. The last block writes an HTML report of the
run and prints where it went.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qcog import get_model, write_model_report  # noqa: E402

model = get_model("question-order")

# Exact evaluation of the analytic model.
exact = model.run()
print("=" * 78)
print(exact.title)
print(exact.summary)
for finding in exact.findings:
    print(" -", finding)

# The same model, compiled to a Qiskit circuit and sampled.
sampled = model.run(backend="aer", shots=20000, seed=11)
print()
print("Shot-based circuit run:", sampled.summary)
for finding in sampled.findings[:2]:
    print(" -", finding)

# Decoherence: how much noise can the effect tolerate?
noisy = model.run(decoherence=0.15)
print()
print("With 15% depolarising noise after each question:", noisy.summary)

# An accessible HTML report, with figures and data tables.
path = write_model_report(model, exact, "out/example_question_order")
print()
print("Report written to", path)