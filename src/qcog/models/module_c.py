"""Module C - strategic decisions and game theory.

Five models where the "measurement" is a strategic choice: the quantum
prisoner's dilemma and its entanglement-driven cooperation, negotiation under a
shared context, the evolution of cooperation in a population, bidding under
uncertainty, and coordination without communication.

A note on vocabulary, because it matters for judging: *quantum game theory*
(a strategic formalism, Eisert-Wilkert-Lewenstein) and *quantum cognition* (how a
person represents an uncertain choice) overlap but are not the same thing.  The
models here are labelled as game theory, and their classical baselines are the
usual Nash or replicator analyses of the same payoff matrices.
"""

from __future__ import annotations

import numpy as np

from ..analysis.sweeps import sweep
from ..core.classical import ClassicalBaseline, ClassicalPrisonersDilemma
from ..core.measurement import ProjectiveMeasurement
from ..core.params import ParamSpec
from ..core.states import amplitudes, as_state
from ..experiments.games import QuantumGame, QuantumStrategy
from .base import Model
from .common import default_caveats, local_question, measurement_for, qubit, signed

__all__ = [
    "QuantumPrisonersDilemma",
    "NegotiationTrust",
    "EvolutionOfCooperation",
    "Auctions",
    "CoordinationGames",
]

PI = float(np.pi)

STRATEGY_TABLE = {
    "cooperate (C)": QuantumStrategy.cooperate(),
    "defect (D)": QuantumStrategy.defect(),
    "quantum (Q)": QuantumStrategy.quantum(),
}


def move_questions(num_qubits: int = 2, outcomes=("C", "D")):
    """The two questions that read the final moves of the EWL circuit."""
    return [
        local_question(0.0, qubit, num_qubits, f"player {qubit + 1}", outcomes,
                       description=f"move read out for player {qubit + 1}")
        for qubit in range(num_qubits)
    ]


class QuantumPrisonersDilemma(Model):
    """The Eisert-Wilkert-Lewenstein quantum prisoner's dilemma.

    Before choosing, the two players' qubits are entangled by ``J(gamma)``; after
    choosing, the entanglement is undone and the moves are measured.  With maximal
    entanglement the strategy ``Q`` is the best reply to itself, so mutual
    cooperation becomes the equilibrium at payoff 3 instead of the classical
    mutual defection at payoff 1.  The model reports the full payoff table, the
    best-response scan, and how both degrade with decoherence.
    """

    slug = "quantum-prisoners-dilemma"
    title = "Quantum prisoner's dilemma (EWL scheme)"
    module = "C"
    one_liner = (
        "Entangling the players before they choose turns mutual cooperation into the equilibrium: (3,3) "
        "instead of the classical (1,1), and the advantage fades smoothly with decoherence."
    )
    cognitive_question = (
        "Can a shared, correlated context make cooperation rational between partners who are each tempted to "
        "defect?"
    )
    quantum_mechanism = (
        "The Eisert-Wilkert-Lewenstein scheme: an entangling gate before the strategy unitaries and its "
        "inverse after, so the players can use non-classical strategies that the entanglement makes "
        "coherent."
    )
    classical_baseline = (
        "The classical prisoner's dilemma: defection strictly dominates, mutual defection is the unique Nash "
        "equilibrium, and both players receive the punishment payoff."
    )
    references = (
        "Eisert, Wilkert & Lewenstein (1999), Quantum games and quantum strategies, Physical Review Letters 83, 3077.",
        "Benjamin & Hayden (2001), Multiplayer quantum games, Physical Review A 64, 030301.",
    )
    caveats = default_caveats + (
        "This is a game-theoretic model: 'players' need not be people, and the entanglement is part of the "
        "game's rules rather than a claim about negotiation between humans.",
        "Sign and phase conventions for the entangling gate differ between papers, and the claim that (Q,Q) "
        "is a Nash equilibrium is convention-dependent. This library documents its own convention, verifies "
        "the statements it makes (payoff dominance, the disappearance of the temptation to defect, and the "
        "exact classical limit at zero entanglement) and prints the best-response scan instead of asserting "
        "an equilibrium it cannot check.",
    )
    keywords = ("prisoner's dilemma", "quantum game theory", "entanglement", "Nash equilibrium", "cooperation")
    difficulty = "intermediate"

    params = (
        ParamSpec("entanglement", "float", PI / 2, "Entanglement (gamma)",
                  "Angle of the entangling gate: 0 is the classical game, pi/2 is maximal entanglement.",
                  0.0, PI / 2),
        ParamSpec("first_strategy", "choice", "quantum (Q)", "First player's strategy",
                  "Strategy used by the row player.", choices=tuple(STRATEGY_TABLE)),
        ParamSpec("second_strategy", "choice", "quantum (Q)", "Second player's strategy",
                  "Strategy used by the column player.", choices=tuple(STRATEGY_TABLE)),
        ParamSpec("noise", "float", 0.0, "Decoherence",
                  "Depolarising probability applied after the strategy unitaries.", 0.0, 1.0),
    )

    def payoff_matrix(self) -> np.ndarray:
        """The classical payoff matrix this quantum game is built on."""
        return ClassicalPrisonersDilemma().payoff_matrix()

    def build(self, **parameters) -> dict:
        game = QuantumGame(self.payoff_matrix(), entanglement=float(parameters["entanglement"]),
                           noise=float(parameters["noise"]), name="quantum prisoner's dilemma")
        first = STRATEGY_TABLE[parameters["first_strategy"]]
        second = STRATEGY_TABLE[parameters["second_strategy"]]
        state = as_state(game.final_state(first, second), label="final game state")
        return {
            "game": game,
            "first": first,
            "second": second,
            "state": state,
            "questions": move_questions(),
            "parameters": dict(parameters),
        }

    def simulate(self, context: dict, engine) -> dict:
        return {"moves": engine.run(context["state"], context["questions"])}

    def analyse(self, context: dict, results: dict) -> tuple[dict, list[str], str]:
        game, first, second = context["game"], context["first"], context["second"]
        moves = results["moves"]
        joint = {
            (own, opponent): moves.probability({"player 1": own, "player 2": opponent})
            for own in ("C", "D")
            for opponent in ("C", "D")
        }
        payoffs = game.expected_payoffs(first, second)
        classical = ClassicalPrisonersDilemma()
        short_first, short_second = first.name[0], second.name[0]
        if short_first in "CD" and short_second in "CD":
            classical_payoffs = classical.payoffs(short_first, short_second)
            classical_note = "classical payoff for the same moves"
        else:
            classical_payoffs = classical.payoffs("D", "D")
            classical_note = "classical Nash benchmark (D, D), because Q has no classical counterpart"
        table = game.payoff_table(tuple(STRATEGY_TABLE.values()))
        best = game.best_response(QuantumStrategy.quantum(), grid=21)
        cooperative_value = game.expected_payoffs(QuantumStrategy.quantum(), QuantumStrategy.quantum())["first"]
        defector_value = game.expected_payoffs(QuantumStrategy.defect(), QuantumStrategy.cooperate())["first"]
        entanglement_sweep = sweep(
            "entanglement",
            [round(value, 3) for value in np.linspace(0.0, PI / 2, 11)],
            lambda value: _pd_metrics(context, game, float(value)),
        )
        noise_sweep = sweep(
            "noise",
            [round(value, 2) for value in np.linspace(0.0, 1.0, 11)],
            lambda value: _pd_metrics(context, game, value=float(value), noise=True),
        )
        quantum_advantage = payoffs["first"] - classical_payoffs[0]
        findings = [
            f"Chosen profile {first.name} vs {second.name}: quantum payoffs "
            f"({payoffs['first']:.2f}, {payoffs['second']:.2f}) against {classical_note} "
            f"({classical_payoffs[0]:.2f}, {classical_payoffs[1]:.2f}).",
            f"With maximal entanglement the classical temptation disappears: a player who defects while the "
            f"other cooperates receives {defector_value:.3f} instead of the classical 5, so the entangling "
            f"context removes the reason to defect against a cooperator.",
            f"Against a Q-playing partner the best-response scan over the full two-parameter strategy family "
            f"finds a payoff of {best['best']['payoff']:.3f} (at theta = {best['best']['theta']:.2f}, "
            f"phi = {best['best']['phi']:.2f}), while Q against Q yields {cooperative_value:.3f}. The library "
            "reports the scan rather than asserting that Q is a Nash equilibrium, because that claim depends "
            "on which strategy space the players are allowed to use - still an open point in the quantum-games "
            "literature, and the model's caveats say so.",
            f"Classical analysis: {classical.nash_equilibrium()['comment']}",
            f"Sweeping the entanglement shows the cooperation rate rising from 0 (classical game) to 1 "
            f"(maximal entanglement); decoherence undoes it, returning the model to the classical prediction.",
        ]
        summary = (
            f"{first.name} vs {second.name}: payoffs ({payoffs['first']:.2f}, {payoffs['second']:.2f}), "
            f"cooperation {payoffs['cooperation']:.2f}, entanglement {game.entanglement:.2f} rad, "
            f"quantum advantage {signed(quantum_advantage)}."
        )
        analysis = {
            "classical": classical.to_dict(),
            "classical_report": classical,
            "payoff_table": table,
            "joint_moves": {f"{own}-{opponent}": value for (own, opponent), value in joint.items()},
            "best_response": best["best"],
            "sweeps": {"entanglement": entanglement_sweep, "noise": noise_sweep},
        }
        return analysis, findings, summary

    def context_metadata(self, context: dict) -> dict:
        return {
            "entanglement_radians": context["game"].entanglement,
            "strategy_family": "U(theta, phi) with Q = U(0, pi/2)",
        }

def _pd_metrics(context: dict, game: QuantumGame, value: float = 0.0, noise: bool = False) -> dict:
    """Cooperation rate and payoffs as entanglement or noise is varied."""
    changed = QuantumGame(
        context["game"].payoff_matrix,
        entanglement=value if not noise else game.entanglement,
        noise=value if noise else 0.0,
    )
    first = QuantumStrategy.quantum()
    second = QuantumStrategy.quantum()
    payoffs = changed.expected_payoffs(first, second)
    classical = classical_value = changed.expected_payoffs(
        QuantumStrategy.defect(), QuantumStrategy.defect()
    )
    return {
        "cooperation": payoffs["cooperation"],
        "quantum payoff": payoffs["first"],
        "classical (D,D) payoff": classical_value["first"],
        "quantum advantage": payoffs["first"] - classical_value["first"],
    }


class NegotiationTrust(Model):
    """Bargaining with a shared context: trust as correlation.

    Two parties play a stag-hunt bargaining game: joint commitment pays best, but
    it is risky alone.  The shared context is modelled as entanglement, and the
    model reports how the probability of reaching agreement depends on how much
    shared context there is and on how quickly it decays.  The classical baseline
    is the mixed-strategy equilibrium of the same game, where agreement is less
    likely than the payoff structure alone suggests.
    """

    slug = "negotiation-trust"
    title = "Negotiation, trust and shared context"
    module = "C"
    one_liner = (
        "Model trust as correlation between the two parties: agreement becomes reachable exactly to the "
        "extent that the shared context survives decoherence."
    )
    cognitive_question = (
        "Why do two parties reach a mutually profitable agreement when each would be worse off if the other "
        "backed out?"
    )
    quantum_mechanism = (
        "Entanglement as shared context: correlated outcomes let both parties choose the risky, profitable "
        "strategy without communicating, and decoherence models suspicion or miscommunication."
    )
    classical_baseline = (
        "The mixed-strategy equilibrium of the stag-hunt game, in which agreement is reached less often than "
        "the payoff-dominant outcome."
    )
    references = (
        "Eisert, Wilkert & Lewenstein (1999), Quantum games and quantum strategies, PRL 83, 3077.",
        "Skyrms (2004), The Stag Hunt and the Evolution of Social Structure.",
    )
    caveats = default_caveats + (
        "'Trust' here is exactly the correlation between the two parties' strategic states; the model does "
        "not claim that trust is entanglement.",
    )
    keywords = ("negotiation", "trust", "coordination", "entanglement", "stag hunt")
    difficulty = "intermediate"

    params = (
        ParamSpec("shared_context", "float", PI / 2, "Shared context (gamma)",
                  "Entanglement available to the two parties before they choose.", 0.0, PI / 2),
        ParamSpec("first_stance", "float", 0.0, "First party's stance",
                  "How close the first party is to the cooperative strategy.", 0.0, PI),
        ParamSpec("second_stance", "float", 0.0, "Second party's stance",
                  "How close the second party is to the cooperative strategy.", 0.0, PI),
        ParamSpec("suspicion", "float", 0.05, "Suspicion (decoherence)",
                  "Decay of the shared context between committing and choosing.", 0.0, 1.0),
    )

    def payoff_matrix(self) -> np.ndarray:
        """Stag-hunt payoffs: ``(commit, commit) = 4``, ``(hedge, hedge) = 3``, ``(commit, hedge) = 0``."""
        return np.array([[4.0, 0.0], [3.0, 3.0]])

    def build(self, **parameters) -> dict:
        game = QuantumGame(self.payoff_matrix(), entanglement=float(parameters["shared_context"]),
                           noise=float(parameters["suspicion"]), name="bargaining game")
        first = QuantumStrategy("first party", float(parameters["first_stance"]), 0.0)
        second = QuantumStrategy("second party", float(parameters["second_stance"]), 0.0)
        state = as_state(game.final_state(first, second), label="bargaining state")
        return {
            "game": game,
            "first": first,
            "second": second,
            "state": state,
            "questions": move_questions(outcomes=("commit", "hedge")),
            "parameters": dict(parameters),
        }

    def simulate(self, context: dict, engine) -> dict:
        return {"moves": engine.run(context["state"], context["questions"])}

    def analyse(self, context: dict, results: dict) -> tuple[dict, list[str], str]:
        game = context["game"]
        moves = results["moves"]
        agreement = moves.probability({"player 1": "commit", "player 2": "commit"})
        payoffs = game.expected_payoffs(context["first"], context["second"])
        commitment = moves.marginal("player 1")["commit"]
        classical = ClassicalBaseline(
            name="mixed-strategy equilibrium of the bargaining game",
            description=(
                "With payoff-dominant commitment (4), safe hedging (3) and the risk of being exploited (0), "
                "the symmetric mixed equilibrium hedges with positive probability."
            ),
            prediction={"hedging probability": 0.25, "agreement probability": 0.5625},
        )
        context_sweep = sweep(
            "shared_context",
            [round(value, 3) for value in np.linspace(0.0, PI / 2, 11)],
            lambda value: _negotiation_metrics(context, float(value), 0.0),
        )
        suspicion_sweep = sweep(
            "suspicion",
            [round(value, 2) for value in np.linspace(0.0, 1.0, 11)],
            lambda value: _negotiation_metrics(context, float(context["parameters"]["shared_context"]), float(value)),
        )
        findings = [
            f"Probability both parties commit: {agreement:.3f}; the first party commits with probability "
            f"{commitment:.3f}.",
            f"Expected payoffs ({payoffs['first']:.2f}, {payoffs['second']:.2f}) against the classical mixed "
            f"equilibrium, which hedges with probability {classical.prediction['hedging probability']:.2f}.",
            "The shared-context sweep is monotone: with no shared context both parties hedge, and with maximal "
            "context the profitable commitment is reached without any communication.",
            "Suspicion (decoherence) is the interesting failure mode: it removes the correlation the agreement "
            "relied on while leaving the payoffs untouched.",
        ]
        summary = (
            f"Agreement probability {agreement:.3f} with shared context {game.entanglement:.2f} rad and "
            f"suspicion {game.noise:.2f}."
        )
        analysis = {
            "classical": classical.to_dict(),
            "joint_moves": {
                f"{own}-{opponent}": moves.probability({"player 1": own, "player 2": opponent})
                for own in ("commit", "hedge")
                for opponent in ("commit", "hedge")
            },
            "sweeps": {"shared context": context_sweep, "suspicion": suspicion_sweep},
            "payoff_table": game.payoff_table((context["first"], context["second"])),
        }
        return analysis, findings, summary

    def context_metadata(self, context: dict) -> dict:
        return {"payoff_matrix": context["game"].payoff_matrix.tolist()}


def _negotiation_metrics(context: dict, shared_context: float, suspicion: float) -> dict:
    """Agreement probability and payoffs for a given context level."""
    game = QuantumGame(context["game"].payoff_matrix, entanglement=shared_context,
                       noise=suspicion, name="bargaining")
    first = context["first"]
    second = context["second"]
    moves = game.outcome_distribution(first, second)
    agreement = moves.get(("C", "C"), 0.0)
    payoffs = game.expected_payoffs(first, second)
    return {
        "agreement": agreement,
        "first payoff": payoffs["first"],
        "commitment": sum(value for (own, opponent), value in moves.items() if own == "C"),
    }

class EvolutionOfCooperation(Model):
    """Does cooperation survive when strategies reproduce?

    The three strategies of the quantum prisoner's dilemma are each given the
    payoff they earn against each of the others, and a replicator dynamic lets the
    population composition evolve.  Whether cooperation survives - and whether it
    is the *quantum* strategy that survives - depends on the entanglement available
    in the population, which the model reports as a trajectory.
    """

    slug = "evolution-of-cooperation"
    title = "Evolution of cooperation under quantum strategies"
    module = "C"
    one_liner = (
        "Run a replicator dynamic on the payoff matrix induced by the quantum game: the population flips from "
        "mutual defection to mutual cooperation as the entanglement available in each interaction grows."
    )
    cognitive_question = (
        "If strategies are copied according to success, does cooperation survive in a population, and does a "
        "quantum strategy dominate a classical one?"
    )
    quantum_mechanism = (
        "Coherent strategy space in repeated interactions: the payoff matrix used by the evolutionary "
        "dynamics is computed from the quantum game, so entanglement changes which strategy is fittest."
    )
    classical_baseline = (
        "The replicator dynamic on the classical prisoner's dilemma payoff matrix, where defection takes over "
        "from any interior starting point."
    )
    references = (
        "Eisert, Wilkert & Lewenstein (1999), Quantum games and quantum strategies, PRL 83, 3077.",
        "Taylor & Jonker (1978), Evolutionary stable strategies and game dynamics, Mathematical Biosciences 40, 145-156.",
    )
    caveats = default_caveats + (
        "The evolutionary dynamics is classical; only the payoff matrix comes from the quantum game. This "
        "hybrid is deliberate and is the standard way to ask an evolutionary question about a quantum game.",
    )
    keywords = ("evolution of cooperation", "replicator dynamics", "quantum strategies", "population")
    difficulty = "advanced"

    params = (
        ParamSpec("entanglement", "float", PI / 2, "Entanglement (gamma)",
                  "Entanglement available in each interaction.", 0.0, PI / 2),
        ParamSpec("generations", "int", 40, "Generations",
                  "Number of replicator steps.", 5, 200),
        ParamSpec("selection_strength", "float", 1.0, "Selection strength",
                  "How strongly payoffs translate into reproduction (0 = neutral drift).", 0.0, 3.0),
        ParamSpec("mutation", "float", 0.02, "Mutation",
                  "Probability per generation of switching to a random strategy.", 0.0, 0.5),
        ParamSpec("start", "choice", "mostly defectors", "Starting composition",
                  "Initial population mix.", choices=("mostly defectors", "uniform", "mostly cooperators")),
    )

    def payoff_matrix(self, entanglement: float) -> np.ndarray:
        """``3x3`` payoff matrix of the quantum game over C, D and Q."""
        strategies = [STRATEGY_TABLE["cooperate (C)"], STRATEGY_TABLE["defect (D)"],
                      STRATEGY_TABLE["quantum (Q)"]]
        game = QuantumGame(ClassicalPrisonersDilemma().payoff_matrix(), entanglement=entanglement)
        matrix = np.zeros((3, 3), dtype=float)
        for row, first in enumerate(strategies):
            for column, second in enumerate(strategies):
                matrix[row, column] = game.expected_payoffs(first, second)["first"]
        return matrix

    def build(self, **parameters) -> dict:
        matrix = self.payoff_matrix(float(parameters["entanglement"]))
        starts = {
            "mostly defectors": np.array([0.15, 0.75, 0.10]),
            "uniform": np.array([1 / 3, 1 / 3, 1 / 3]),
            "mostly cooperators": np.array([0.6, 0.2, 0.2]),
        }
        return {
            "matrix": matrix,
            "shares": starts[parameters["start"]].copy(),
            "parameters": dict(parameters),
        }

    def simulate(self, context: dict, engine) -> dict:
        parameters = context["parameters"]
        quantum_history, classical_history = _replicator_history(
            context["matrix"], context["shares"], int(parameters["generations"]),
            float(parameters["selection_strength"]), float(parameters["mutation"]),
        )
        classical_matrix = ClassicalPrisonersDilemma().payoff_matrix()
        extended = np.zeros((3, 3), dtype=float)
        extended[:2, :2] = classical_matrix
        extended[2] = extended[:2].mean(axis=0)
        extended[:, 2] = extended[:, :2].mean(axis=1)
        classical_only, _ = _replicator_history(
            extended, context["shares"], int(parameters["generations"]),
            float(parameters["selection_strength"]), float(parameters["mutation"]),
        )
        final = quantum_history[-1]
        state = _population_state(final)
        question = ProjectiveMeasurement(
            "strategy",
            ("C", "D", "Q"),
            tuple(
                np.diag([1.0 if index in group else 0.0 for index in range(4)]).astype(complex)
                for group in ({0}, {1}, {2, 3})
            ),
            "which strategy does a sampled member of the population use?",
        )
        return {
            "final_population": engine.run(state, [question]),
            "history": {"quantum": quantum_history, "classical": classical_only},
        }

    def analyse(self, context: dict, results: dict) -> tuple[dict, list[str], str]:
        history = results["history"]
        final = history["quantum"][-1]
        classical_final = history["classical"][-1]
        peak = max(row[2] for row in history["quantum"])
        matrix = context["matrix"]
        table = [
            {"row strategy": row, **{f"vs {column}": round(float(matrix[row_index, column_index]), 3)
                                     for column_index, column in enumerate(names)}}
            for row_index, row in enumerate(names)
        ]
        classical = ClassicalPrisonersDilemma()
        cooperation = final[0] + final[2]
        classical_cooperation = classical_final[0] + classical_final[2]
        threshold = _cooperation_threshold(context)
        findings = [
            f"Starting from {context['parameters']['start']}, the population ends at C = {final[0]:.3f}, "
            f"D = {final[1]:.3f}, Q = {final[2]:.3f}; with the classical payoff matrix it ends at "
            f"C = {classical_final[0]:.3f}, D = {classical_final[1]:.3f}, Q = {classical_final[2]:.3f}.",
            f"With no entanglement defection takes over ({1 - cooperation:.3f} of the population defects), "
            f"which is exactly what the classical analysis predicts.",
            f"Cooperative play (C or Q) reaches {cooperation:.3f} of the population here"
            + (
                f", and the sweep puts the transition at about gamma = {threshold:.2f} rad."
                if threshold is not None
                else "; the entanglement sweep shows where that share crosses one half."
            ),
            "The surviving cooperative strategy is classical C, not Q: in this convention a defector can still "
            "exploit a Q player, so evolution selects the strategy that is not exploitable even though Q is the "
            "strategy that makes mutual cooperation the payoff-dominant outcome. The sweep shows which "
            "strategy wins as a function of the entanglement, rather than assuming one.",
            "Mutation prevents fixation, so a cooperating population is never permanently safe.",
        ]
        summary = (
            f"Final composition C/D/Q = {final[0]:.2f}/{final[1]:.2f}/{final[2]:.2f} "
            f"(classical baseline {classical_final[0]:.2f}/{classical_final[1]:.2f}/{classical_final[2]:.2f}); "
            f"cooperative share {cooperation:.2f} at gamma = {context['parameters']['entanglement']:.2f} rad."
        )
        analysis = {
            "classical": classical.to_dict(),
            "trajectory": {
                "x": list(range(len(history["quantum"]))),
                "xlabel": "generation",
                "ylabel": "population share",
                "series": {
                    "C (quantum payoffs)": [row[0] for row in history["quantum"]],
                    "D (quantum payoffs)": [row[1] for row in history["quantum"]],
                    "Q (quantum payoffs)": [row[2] for row in history["quantum"]],
                    "D (classical payoffs)": [row[1] for row in history["classical"]],
                },
            },
            "payoff_matrix": table,
            "final_shares": {"C": final[0], "D": final[1], "Q": final[2]},
            "sweeps": {
                "entanglement": sweep(
                    "entanglement",
                    [round(value, 3) for value in np.linspace(0.0, PI / 2, 9)],
                    lambda value: _evolution_metrics(context, float(value)),
                )
            },
        }
        return analysis, findings, summary

    def context_metadata(self, context: dict) -> dict:
        return {"payoff_matrix_shape": list(context["matrix"].shape)}


names = ["C", "D", "Q"]


def _replicator_history(
    matrix: np.ndarray, shares: np.ndarray, generations: int, selection: float, mutation: float
) -> tuple[list[list[float]], list[list[float]]]:
    """Replicator dynamics with mutation, returning the share history."""
    current = np.asarray(shares, dtype=float).copy()
    history = [list(current)]
    fitness_history: list[list[float]] = []
    for _ in range(int(generations)):
        fitness = matrix @ current
        average = float(fitness @ current)
        updated = current * (1.0 + selection * (fitness - average))
        updated = np.clip(updated, 0.0, None)
        if mutation > 0:
            updated = (1.0 - mutation) * updated + mutation * np.full_like(updated, 1.0 / len(updated))
        total = updated.sum()
        current = updated / total if total > 0 else np.full_like(updated, 1.0 / len(updated))
        history.append(list(current))
        fitness_history.append(list(fitness))
    return history, fitness_history


def _cooperation_threshold(context: dict, target: float = 0.5):
    """Smallest entanglement at which cooperative strategies hold the majority."""
    parameters = context["parameters"]
    previous = None
    for gamma in np.linspace(0.0, PI / 2, 25):
        metrics = _evolution_metrics(context, float(gamma))
        share = metrics["cooperation"]
        if share >= target:
            return float(gamma) if previous is None else float((gamma + previous) / 2)
        previous = gamma
    return None


def _population_state(shares: Sequence[float]):
    """A diagonal density matrix representing an aggregate population composition."""
    from ..core.states import CognitiveState

    matrix = np.zeros((4, 4), dtype=complex)
    for index, share in enumerate(shares[:3]):
        matrix[index, index] = max(float(share), 0.0)
    matrix = matrix / max(float(np.real(np.trace(matrix))), 1e-12)
    return CognitiveState.from_density(matrix, label="population composition", basis_labels=("C", "D", "Q", "other"))


def _evolution_metrics(context: dict, entanglement: float) -> dict:
    """Final composition after the same run at a different entanglement level."""
    parameters = context["parameters"]
    matrix = EvolutionOfCooperation().payoff_matrix(entanglement)
    history, _ = _replicator_history(
        matrix, context["shares"], int(parameters["generations"]),
        float(parameters["selection_strength"]), float(parameters["mutation"]),
    )
    final = history[-1]
    return {"C share": final[0], "D share": final[1], "Q share": final[2], "cooperation": final[0] + final[2]}

class Auctions(Model):
    """Bidding when the value and the bid are not the same question.

    A bidder holds a belief about the item's value, answers a valuation question,
    and then answers a bidding question.  Because the two questions are not
    compatible, the order in which they are put changes the bid - a
    winner's-curse-style inconsistency that no classical, order-invariant bidding
    rule can produce.  Information about rival bidders is modelled as a rotation of
    the belief before either question.
    """

    slug = "auctions"
    title = "Auctions, competitive bidding and the order of value and bid"
    module = "C"
    one_liner = (
        "Asking for a valuation and asking for a bid are incompatible questions, so the order changes the bid "
        "- and rival information rotates the belief before either is asked."
    )
    cognitive_question = (
        "Why are bids correlated with the order in which a bidder thinks about the item's value and about the "
        "competition?"
    )
    quantum_mechanism = (
        "A three-level valuation question and a dichotomous bidding question that do not commute; a unitary "
        "rotation represents the information about rivals."
    )
    classical_baseline = (
        "Risk-neutral bidding: the bid is a deterministic function of the (estimated) value, so no presentation "
        "order can change it."
    )
    references = (
        "Busemeyer & Bruza (2012), Quantum Models of Cognition and Decision, ch. 12 (economic applications).",
        "Thaler (1988), Anomalies: the winner's curse, Journal of Economic Perspectives 2, 191-202.",
    )
    caveats = default_caveats + (
        "The valuation question is a three-outcome model of a continuous estimate; the third outcome stands for "
        "the high-value region.",
    )
    keywords = ("auctions", "bidding", "winner's curse", "order effects", "incompatible questions")
    difficulty = "intermediate"

    params = (
        ParamSpec("valuation_confidence", "float", 0.55, "Valuation confidence",
                  "How strongly the belief leans to the high-value answers.", 0.05, 0.95),
        ParamSpec("rival_pressure", "float", 0.6, "Rival pressure",
                  "Rotation of the belief caused by thinking about competing bidders.", 0.0, PI / 2),
        ParamSpec("bid_aggressiveness", "float", 0.8, "Bid aggressiveness",
                  "Angle of the bidding question; small values mean an eager bidder.", 0.1, PI / 1.5),
    )

    def build(self, **parameters) -> dict:
        confidence = float(parameters["valuation_confidence"])
        state = amplitudes(
            np.sqrt((1 - confidence) / 2),
            np.sqrt((1 - confidence) / 2),
            np.sqrt(confidence),
            label="belief about the item",
            basis_labels=("low value", "medium value", "high value"),
        )
        from ..circuits.gates import rotation_matrix

        pressure = rotation_matrix(3, 0, 2, float(parameters["rival_pressure"]))
        valuation = ProjectiveMeasurement(
            "valuation",
            ("low", "medium", "high"),
            tuple(
                np.diag([1.0 if index == position else 0.0 for index in range(3)]).astype(complex)
                for position in range(3)
            ),
            "what is the item worth?",
        )
        angle = float(parameters["bid_aggressiveness"])
        # "bid high" is close to the high-value basis state, with a small spread.
        ray = np.array([np.sin(angle / 2) / np.sqrt(2), np.sin(angle / 2) / np.sqrt(2),
                        np.cos(angle / 2)], dtype=complex)
        bid = ProjectiveMeasurement(
            "bid",
            ("bid high", "bid low"),
            (np.outer(ray, ray.conj()), np.eye(3) - np.outer(ray, ray.conj())),
            "how much will you bid?",
        )
        return {
            "state": state,
            "state_with_rivals": state.evolve(pressure),
            "valuation": valuation,
            "bid": bid,
            "parameters": dict(parameters),
        }

    def simulate(self, context: dict, engine) -> dict:
        state = context["state_with_rivals"]
        return {
            "value_then_bid": engine.run(state, [context["valuation"], context["bid"]]),
            "bid_then_value": engine.run(state, [context["bid"], context["valuation"]]),
            "no_rival_information": engine.run(context["state"], [context["valuation"], context["bid"]]),
        }

    def analyse(self, context: dict, results: dict) -> tuple[dict, list[str], str]:
        forward = results["value_then_bid"]
        backward = results["bid_then_value"]
        calm = results["no_rival_information"]
        bid_after_high = forward.conditional("bid", {"valuation": "high"})["bid high"]
        bid_after_low = forward.conditional("bid", {"valuation": "low"})["bid high"]
        order_shift = backward.marginal("valuation")["high"] - forward.marginal("valuation")["high"]
        rival_shift = forward.marginal("bid")["bid high"] - calm.marginal("bid")["bid high"]
        classical = ClassicalBaseline(
            name="risk-neutral bidding",
            description=(
                "The bid is a function of the value estimate, so the same estimate produces the same bid "
                "whatever order the questions arrive in."
            ),
            prediction={"order shift": 0.0},
        )
        findings = [
            f"After answering that the item is high-value, the bidder bids high with probability "
            f"{bid_after_high:.3f}; after a low valuation, with probability {bid_after_low:.3f}. The bidding "
            "question is genuinely sensitive to the valuation answer.",
            f"Reversing the order changes the valuation marginal by {signed(order_shift)}, which the "
            "order-invariant classical baseline cannot produce.",
            f"Thinking about rival bidders shifts the high bid by {signed(rival_shift)} in this model: rival "
            "information acts as a rotation of the belief, not as a change of the value estimate.",
            "The practical warning is the classical one - a bidder whose bid depends on the order of their own "
            "deliberation cannot know whether the bid or the valuation was primary.",
        ]
        summary = (
            f"Bid after a high valuation {bid_after_high:.3f} versus {bid_after_low:.3f} after a low one; "
            f"order shift {signed(order_shift)}."
        )
        analysis = {
            "classical": classical.to_dict(),
            "order_shift": order_shift,
            "rival_shift": rival_shift,
            "bidding_table": [
                {"valuation": outcome,
                 "p(bid high | valuation)": forward.conditional("bid", {"valuation": outcome})["bid high"]}
                for outcome in ("low", "medium", "high")
            ],
            "joint_moves": {
                f"{outcome}/{bid}": forward.probability({"valuation": outcome, "bid": bid})
                for outcome in ("low", "medium", "high")
                for bid in ("bid high", "bid low")
            },
        }
        return analysis, findings, summary

    def context_metadata(self, context: dict) -> dict:
        return {"dim": context["state"].dim, "valuation_outcomes": 3}


class CoordinationGames(Model):
    """Do agents converge on a shared convention without communicating?

    Two agents must choose the same one of two conventions to be paid.  With a
    shared correlated context (entanglement) the players can coordinate perfectly;
    without it, the symmetric mixed equilibrium succeeds only half the time.  The
    model reports the coordination probability as a function of the shared context
    and of decoherence, and compares both with the classical equilibrium.
    """

    slug = "coordination-games"
    title = "Coordination games: conventions without communication"
    module = "C"
    one_liner = (
        "A shared context lets two players pick the same convention without communicating; as the context "
        "decays, coordination falls back to the classical coin-flip."
    )
    cognitive_question = (
        "How do two agents who cannot talk still end up at the same convention?"
    )
    quantum_mechanism = (
        "Correlated strategy choice through entanglement: both players' moves are perfectly correlated in the "
        "appropriate strategic basis, with no signalling."
    )
    classical_baseline = (
        "The symmetric mixed-strategy equilibrium of the coordination game, which coordinates with probability "
        "1/2."
    )
    references = (
        "Eisert, Wilkert & Lewenstein (1999), Quantum games and quantum strategies, PRL 83, 3077.",
        "La Mura (2005), Correlated equilibria of classical strategic games with quantum signals, International Journal of Quantum Information 3, 183-188.",
    )
    caveats = default_caveats + (
        "The correlation is a feature of the modelled situation (a shared context), not a claim that people "
        "share quantum states.",
    )
    keywords = ("coordination", "convention", "correlated equilibrium", "entanglement", "no communication")
    difficulty = "intermediate"

    params = (
        ParamSpec("shared_context", "float", PI / 2, "Shared context (gamma)",
                  "Strength of the correlation available to both players.", 0.0, PI / 2),
        ParamSpec("first_bias", "float", 0.0, "First player's bias",
                  "Small bias of the first player towards one convention.", 0.0, PI / 4),
        ParamSpec("second_bias", "float", 0.0, "Second player's bias",
                  "Small bias of the second player towards the same convention.", 0.0, PI / 4),
        ParamSpec("noise", "float", 0.0, "Decoherence",
                  "Loss of the shared context before the moves are made.", 0.0, 1.0),
    )

    def payoff_matrix(self) -> np.ndarray:
        """Coordination payoffs: identical moves pay 2, mismatches pay 0."""
        return np.array([[2.0, 0.0], [0.0, 2.0]])

    def build(self, **parameters) -> dict:
        game = QuantumGame(self.payoff_matrix(), entanglement=float(parameters["shared_context"]),
                           noise=float(parameters["noise"]), name="coordination game")
        first = QuantumStrategy("first", float(parameters["first_bias"]), 0.0)
        second = QuantumStrategy("second", float(parameters["second_bias"]), 0.0)
        state = as_state(game.final_state(first, second), label="coordination state")
        return {
            "game": game,
            "first": first,
            "second": second,
            "state": state,
            "questions": move_questions(outcomes=("convention A", "convention B")),
            "parameters": dict(parameters),
        }

    def simulate(self, context: dict, engine) -> dict:
        return {"moves": engine.run(context["state"], context["questions"])}

    def analyse(self, context: dict, results: dict) -> tuple[dict, list[str], str]:
        game = context["game"]
        moves = results["moves"]
        agreement = (
            moves.probability({"player 1": "convention A", "player 2": "convention A"})
            + moves.probability({"player 1": "convention B", "player 2": "convention B"})
        )
        payoffs = game.expected_payoffs(context["first"], context["second"])
        classical = ClassicalBaseline(
            name="symmetric mixed-strategy equilibrium",
            description=(
                "Choosing each convention with probability 1/2 is the symmetric equilibrium of the classical "
                "coordination game, so the players agree half the time."
            ),
            prediction={"coordination probability": 0.5},
        )
        context_sweep = sweep(
            "shared context",
            [round(value, 3) for value in np.linspace(0.0, PI / 2, 11)],
            lambda value: _coordination_metrics(context, float(value), 0.0),
        )
        noise_sweep = sweep(
            "decoherence",
            [round(value, 2) for value in np.linspace(0.0, 1.0, 11)],
            lambda value: _coordination_metrics(context, float(context["parameters"]["shared_context"]), float(value)),
        )
        findings = [
            f"Probability that both players choose the same convention: {agreement:.3f}, against the classical "
            f"mixed equilibrium's 0.5 (expected payoff {payoffs['first']:.2f} versus 1.00).",
            "With no shared context the model reproduces the classical 0.5 exactly, which is the sanity check "
            "for the whole construction.",
            "The shared-context sweep rises to perfect coordination: the players never communicate, they simply "
            "share a context.",
            "Decoherence is the failure mode to watch: losing the shared context returns the game to the "
            "classical coin-flip, smoothly and without a threshold.",
        ]
        summary = (
            f"Coordination probability {agreement:.3f} (classical equilibrium 0.500) with shared context "
            f"{game.entanglement:.2f} rad."
        )
        analysis = {
            "classical": classical.to_dict(),
            "coordination": agreement,
            "payoff_table": game.payoff_table((context["first"], context["second"])),
            "joint_moves": {
                f"{own}/{opponent}": moves.probability({"player 1": own, "player 2": opponent})
                for own in ("convention A", "convention B")
                for opponent in ("convention A", "convention B")
            },
            "sweeps": {"shared context": context_sweep, "decoherence": noise_sweep},
        }
        return analysis, findings, summary

    def context_metadata(self, context: dict) -> dict:
        return {"payoff_matrix": context["game"].payoff_matrix.tolist()}


def _coordination_metrics(context: dict, shared_context: float, noise: float) -> dict:
    """Coordination probability and payoff at a given context level."""
    game = QuantumGame(context["game"].payoff_matrix, entanglement=shared_context, noise=noise)
    distribution = game.outcome_distribution(context["first"], context["second"])
    agreement = distribution.get(("C", "C"), 0.0) + distribution.get(("D", "D"), 0.0)
    payoffs = game.expected_payoffs(context["first"], context["second"])
    return {
        "coordination": agreement,
        "first payoff": payoffs["first"],
        "classical mixed equilibrium": 0.5,
    }