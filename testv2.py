import sys
sys.path.insert(0, "src")

from qcog import get_model

print("=== RUNNING Q-COG MODEL TESTS ===\n")

# ==========================================
# Module B: Logic, Concepts and Memory
# ==========================================
# Real-World Example: The "Linda Problem" (Conjunction Fallacy)
# People often judge a detailed conjunction of events (e.g., "bank teller and active in feminist movement") 
# to be more probable than a single broader event ("bank teller") because of cognitive representation.
model_b = get_model("conjunction-fallacy")
result_b = model_b.run(sweeps=False)  # sweeps=False gives headline numbers quickly

print("1. Module B Test: Conjunction Fallacy")
print(f"Summary: {result_b.summary}")
print(f"Model vs Classical Baseline: {result_b.classical_comparison[0]['model']:.3f} vs {result_b.classical_comparison[0]['baseline']:.3f}\n")


# ==========================================
# Module C: Strategic Decisions & Game Theory
# ==========================================
# Real-World Example: Competing Businesses (Quantum Prisoner's Dilemma)
# Two rival companies choosing whether to cooperate on industry standards or undercut each other. 
# Quantum representation introduces entanglement to model non-classical payoff strategies.
model_c = get_model("quantum-prisoners-dilemma")
result_c = model_c.run(sweeps=False)

print("2. Module C Test: Quantum Prisoner's Dilemma")
print(f"Summary: {result_c.summary}")
print(f"Model vs Classical Baseline: {result_c.classical_comparison[0]['model']:.3f} vs {result_c.classical_comparison[0]['baseline']:.3f}\n")


# ==========================================
# Module D: Groups & Collective Behaviour
# ==========================================
# Real-World Example: Wisdom of Crowds (Collective Judgment)
# A crowd attempting to estimate a value or vote on a decision. 
# The model explores how information correlation among group members can destroy the "wisdom of crowds" advantage.
model_d = get_model("collective-judgment")
result_d = model_d.run(sweeps=False)

print("3. Module D Test: Collective Judgment")
print(f"Summary: {result_d.summary}")
print(f"Model vs Classical Baseline: {result_d.classical_comparison[0]['model']:.3f} vs {result_d.classical_comparison[0]['baseline']:.3f}")



import sys
import os
sys.path.insert(0, "src")

from qcog import get_model

# Create output folder
os.makedirs("out", exist_ok=True)

model = get_model("quantum-prisoners-dilemma")
result = model.run(sweeps=False)

# 1. Save all circuit designs as images
model.circuit_images(result, directory="out")
print("Saved circuit images to the 'out/' directory!")

# 2. Save model figures (comparison charts, etc.)
figures = model.figures(result)
for i, fig in enumerate(figures):
    fig.savefig(f"out/figure_{i}.png", bbox_inches="tight")
print("Saved model figures to the 'out/' directory as well!")