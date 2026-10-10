"""Fitting a model to observed response frequencies, and comparing with a baseline.

``python demos/examples/04_fit_to_data.py``

The data file used here is *illustrative synthetic data* shaped after the
qualitative pattern reported in the literature; see ``data/README.md``.  Its purpose
is to demonstrate the fitting and model-comparison workflow, not to make an
empirical claim.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qcog import get_model  # noqa: E402
from qcog.core.fitting import compare_fits, fit  # noqa: E402
from qcog.core.params import ParamSpec  # noqa: E402

data_path = Path(__file__).resolve().parents[2] / "src" / "qcog" / "data" / "benchmarks.json"
bundle = json.loads(data_path.read_text(encoding="utf-8"))
dataset = bundle["question_order_illustrative"]

observed = dataset["frequencies"]
counts = dataset["counts"]
print(f"Dataset: {dataset['name']}")
print(f"Provenance: {dataset['provenance']}")
print(f"Responses: {len(observed)}")
print()

model = get_model("question-order")

# The quantum model, with its own two fittable angles.
quantum_fit = fit(lambda parameters: model.predict(parameters), observed, model.params, counts=counts)
print("Quantum model:", quantum_fit.summary())
print(quantum_fit.table().to_string(index=False))
print()

# A classical baseline: the two marginals are fitted, but the joint is forced to be
# a product, which is what a single classical probability space requires.
classical_specs = (
    ParamSpec("p_a", "float", 0.5, low=0.01, high=0.99, fittable=True),
    ParamSpec("p_b", "float", 0.5, low=0.01, high=0.99, fittable=True),
)


def classical_predict(parameters):
    p_a, p_b = parameters["p_a"], parameters["p_b"]
    return {
        "AB: p(A=yes)": p_a,
        "AB: p(B=yes)": p_b,
        "AB: p(B=yes|A=yes)": p_b,
        "AB: p(B=yes|A=no)": p_b,
        "BA: p(B=yes)": p_b,
        "BA: p(A=yes)": p_a,
        "BA: p(A=yes|B=yes)": p_a,
        "BA: p(A=yes|B=no)": p_a,
        "QQ: agreement AB": p_a * p_b + (1 - p_a) * (1 - p_b),
        "QQ: agreement BA": p_a * p_b + (1 - p_a) * (1 - p_b),
    }


classical_fit = fit(classical_predict, observed, classical_specs, counts=counts)
print("Classical baseline:", classical_fit.summary())
print()

comparison = compare_fits(quantum_fit, classical_fit)
print("Comparison:", comparison["verdict"])
print("AIC (quantum):", round(quantum_fit.aic, 2), " AIC (classical):", round(classical_fit.aic, 2))
print()
print("Read the verdict carefully: an information criterion is not proof. The model's")
print("own caveats say so, and the QQ equality - which involves no fitted parameter -")
print("is the sharper test of the two accounts.")