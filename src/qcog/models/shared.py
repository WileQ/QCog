"""Reusable model skeletons.

Several of the twenty models differ only in what they call their two questions.
Rather than copy the same simulation and analysis nine times, they subclass
:class:`OrderBasedModel`, which implements the two-orders-in-one-design logic and
lets each model add its own narrative and extra analysis.  The same idea is used
for the population models in Module D through :class:`PopulationModel`.
"""

from __future__ import annotations

import numpy as np

from ..analysis.order import order_effect
from ..analysis.sweeps import sweep
from ..core.params import ParamSpec
from ..experiments.two_question import TwoQuestionExperiment
from .base import Model
from .common import classical_sentence, default_caveats, order_effect_findings, signed

__all__ = ["OrderBasedModel", "PopulationModel"]


class OrderBasedModel(Model):
    """A model whose design is "two questions, asked in both orders".

    Subclasses implement :meth:`make_experiment` (state plus two questions) and
    optionally :meth:`extra_findings`.  Everything else - running both orders,
    computing the order effect, the interference decomposition, the QQ equality,
    the law-of-total-probability violations and the classical comparison - is
    inherited, which is also what makes the comparisons across models consistent.
    """

    sweeps: tuple[ParamSpec, ...] = ()

    def make_experiment(self, **parameters) -> TwoQuestionExperiment:
        """Build the design: a belief state plus two questions."""
        raise NotImplementedError

    def build(self, **parameters) -> dict:
        experiment = self.make_experiment(**parameters)
        return {"experiment": experiment, "parameters": dict(parameters)}

    def simulate(self, context: dict, engine) -> dict:
        return context["experiment"].run_both_orders(engine)

    def analyse(self, context: dict, results: dict) -> tuple[dict, list[str], str]:
        bundle = context["experiment"].analyse(results)
        findings = self.extra_findings(context, results, bundle) + order_effect_findings(bundle)
        findings.append(classical_sentence(bundle))
        summary = self.extra_summary(context, results, bundle)
        analysis = {
            "order_effect": bundle.report,
            "qq_equality": bundle.qq,
            "interference": bundle.interference,
            "ltp": bundle.ltp,
            "classical": bundle.classical,
            "bundle": bundle,
        }
        extra = self.extra_analysis(context, results, bundle)
        analysis.update(extra)
        if self.sweeps:
            analysis["sweeps"] = self.sweep_analysis(context)
        return analysis, findings, summary

    # ------------------------------------------------------------- subclass hooks
    def extra_findings(self, context: dict, results: dict, bundle) -> list[str]:
        """Model-specific findings, shown before the shared order-effect findings."""
        return []

    def extra_analysis(self, context: dict, results: dict, bundle) -> dict:
        """Additional analysis objects (tables, sweeps, figures data)."""
        return {}

    def extra_summary(self, context: dict, results: dict, bundle) -> str:
        """One-line summary; defaults to the order-effect sentence."""
        return bundle.report.summary()

    def sweep_analysis(self, context: dict) -> dict[str, object]:
        """Sweeps declared through :attr:`sweeps`.

        Each fittable parameter gets a sweep of the order effect over its range,
        which is usually the most informative single picture of a model.
        """
        tables: dict[str, object] = {}
        for spec in self.sweeps:
            low = spec.low if spec.low is not None else 0.0
            high = spec.high if spec.high is not None else 1.0
            values = [round(value, 3) for value in np.linspace(low, high, 11)]
            tables[spec.name] = sweep(
                spec.name,
                values,
                lambda value, name=spec.name: self.sweep_metrics(context, name, float(value)),
            )
        return tables

    def sweep_metrics(self, context: dict, name: str, value: float) -> dict:
        """Order effect and interference when one parameter is set to ``value``."""
        parameters = dict(context["parameters"])
        parameters[name] = value
        experiment = self.make_experiment(**parameters)
        from ..core.engine import Engine

        results = experiment.run_both_orders(Engine(backend="exact"))
        report = order_effect(
            results["A_then_B"],
            results["B_then_A"],
            experiment.question_a.name,
            experiment.question_b.name,
        )
        metrics = {"order effect": report.max_marginal_shift()}
        if experiment.state.is_pure:
            metrics["interference"] = float(experiment.interference().max_abs_interference)
        return metrics


class PopulationModel(Model):
    """Base class for models that simulate several agents interacting.

    Subclasses build a population (see :mod:`qcog.experiments.population`), run it
    for a number of steps, and analyse the resulting trajectories.  The shared
    part here is the reporting convention for trajectories and the classical
    comparison, which is used by all four Module D models.
    """

    def trajectory_analysis(self, history: list[list[float]], label: str = "opinion") -> dict:
        """Package a per-agent history into the generic line-plot format."""
        steps = len(history)
        series = {
            f"agent {index + 1}": [float(history[step][index]) for step in range(steps)]
            for index in range(len(history[0])) if steps
        }
        return {
            "x": list(range(steps)),
            "xlabel": "interaction step",
            "ylabel": label,
            "series": series,
        }

    def classical_trajectory(self, initial: list[float], adjacency, steps: int, kind: str = "de-groot") -> dict:
        """Classical baseline trajectory (DeGroot averaging or bounded confidence)."""
        from ..core.classical import bounded_confidence_step, de_groot_step

        opinions = np.asarray(initial, dtype=float)
        series = {f"agent {index + 1}": [float(opinions[index])] for index in range(len(initial))}
        for _ in range(int(steps)):
            if kind == "de-groot":
                opinions = de_groot_step(opinions, np.asarray(adjacency, dtype=float))
            else:
                opinions = bounded_confidence_step(opinions, 0.3, np.asarray(adjacency, dtype=float))
            for index in range(len(initial)):
                series[f"agent {index + 1}"].append(float(opinions[index]))
        return {
            "x": list(range(int(steps) + 1)),
            "xlabel": "interaction step",
            "ylabel": "opinion (classical baseline)",
            "series": series,
        }

    @staticmethod
    def dispersion(opinions: list[float]) -> dict:
        """Consensus and polarisation summary of one opinion configuration."""
        values = np.asarray(opinions, dtype=float)
        return {
            "mean": float(values.mean()),
            "polarisation (sd)": float(values.std()),
            "consensus": float(abs(values.mean())),
            "spread (max-min)": float(values.max() - values.min()),
            "num_clusters_0.1": int(_cluster_count(values, 0.1)),
        }


def _cluster_count(values: np.ndarray, tolerance: float) -> int:
    """Number of opinion clusters separated by more than ``tolerance``."""
    ordered = np.sort(values)
    clusters = 1
    for left, right in zip(ordered[:-1], ordered[1:]):
        if right - left > tolerance:
            clusters += 1
    return clusters