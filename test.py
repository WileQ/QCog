import sys

# Ensure src is on the path for the local library
sys.path.insert(0, "src")

from qcog import get_model


def run_cognitive_order_comparison():
  print("=" * 60)
  print(" Q-COG: Question Order Model & Classical Baseline Comparison")
  print("=" * 60)

  # Load the built-in question-order model from the catalogue
  model = get_model("question-order")

  # Run the simulation (exact backend)
  result = model.run()

  print("\n[1] Key Findings from the Simulation:")
  for finding in result.findings:
    print(f"  • {finding}")

  print("\n[2] Quantitative Model vs. Classical Baseline Comparison:")
  print(
      "  (Note: A classical model assumes a single pre-existing joint"
      " distribution,"
  )
  print(
      "   which forces order differences and interference terms to exactly"
      " zero.)"
  )
  print("-" * 60)
  print(f"{'Metric':<35} | {'Quantum Model':<15} | {'Classical Baseline'}")
  print("-" * 60)

  for row in result.classical_comparison:
    metric = row.get("metric", "Unknown Metric")
    model_val = row.get("model", 0.0)
    baseline_val = row.get("baseline", 0.0)
    print(f"{metric:<35} | {model_val:<15.4f} | {baseline_val:.4f}")

  print("-" * 60)
  print(
      "Conclusion: The quantum model captures the shift in human judgment"
      " sequence,"
  )
  print(
      "whereas the classical baseline yields a flat zero difference because it"
      " lacks interference."
  )


if __name__ == "__main__":
  run_cognitive_order_comparison()