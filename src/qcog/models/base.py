"""The model interface: metadata, parameters, simulation, analysis, figures.

A model in Q-Cog is a small, self-describing object.  It declares

* what cognitive question it explores and which quantum mechanism it uses,
* its parameters (once, as :class:`~qcog.core.params.ParamSpec` objects),
* how to simulate it, and
* which limitations a reader must keep in mind.

Everything else - the interactive explorer, the command line, the report
generator, the documentation and the tests - is driven from that declaration.
That is what makes twenty models maintainable instead of twenty scripts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping, Sequence

import numpy as np

from ..core.engine import Engine
from ..core.params import ParamSpec, coerce_params, parameter_table
from ..core.results import SequentialResult

__all__ = ["Model", "ModelResult", "MODULES", "make_engine"]

#: The four thematic modules of the library, in the order the proposal lists them.
MODULES = {
    "A": "Individual cognition and decision making",
    "B": "Logic, concepts and memory",
    "C": "Strategic decisions and game theory",
    "D": "Groups, social networks and collective behaviour",
}

#: Keyword arguments that configure execution rather than the model itself.
RESERVED = ("backend", "shots", "seed", "noise", "engine")


@dataclass
class ModelResult:
    """Everything one run of a model produces."""

    slug: str
    title: str
    module: str
    parameters: dict
    results: dict[str, Any] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    findings: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    backend: str = "exact"
    shots: int | None = None
    seed: int | None = None
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ tables
    def tables(self) -> dict[str, Any]:
        """Named :class:`pandas.DataFrame` views of the result, for display and export."""
        frames: dict[str, Any] = {}
        for name, value in self.results.items():
            if isinstance(value, SequentialResult):
                frames[f"joint_{name}"] = value.joint_frame()
                frames[f"marginals_{name}"] = value.marginal_frame()
        return frames

    def sequential_results(self) -> dict[str, SequentialResult]:
        """Only the simulation results, without the analysis objects."""
        return {name: value for name, value in self.results.items() if isinstance(value, SequentialResult)}

    def report(self) -> dict:
        """The analysis section, with any report objects converted to dictionaries."""
        out = {}
        for name, value in self.analysis.items():
            if hasattr(value, "to_dict"):
                out[name] = value.to_dict()
            elif isinstance(value, Mapping):
                out[name] = {
                    key: (item.to_dict() if hasattr(item, "to_dict") else item) for key, item in value.items()
                }
            else:
                out[name] = value
        return out

    def to_dict(self) -> dict:
        return {
            "slug": self.slug,
            "title": self.title,
            "module": self.module,
            "parameters": {key: (value if isinstance(value, (int, float, str, bool)) else str(value))
                           for key, value in self.parameters.items()},
            "summary": self.summary,
            "findings": list(self.findings),
            "caveats": list(self.caveats),
            "references": list(self.references),
            "backend": self.backend,
            "shots": self.shots,
            "seed": self.seed,
            "results": {name: value.to_dict() for name, value in self.sequential_results().items()},
            "analysis": self.report(),
            "metadata": {key: value for key, value in self.metadata.items()
                         if isinstance(value, (int, float, str, bool, type(None), list, dict))},
        }

    def headline(self) -> str:
        """One-sentence summary used in reports and the UI."""
        return self.summary or self.title


def make_engine(
    backend: str = "exact",
    shots: int | None = None,
    seed: int | None = None,
    noise=None,
    **options,
) -> Engine:
    """Build an :class:`~qcog.core.engine.Engine` from plain arguments."""
    if backend in {"sampled", "aer"} and not shots:
        shots = 4096
    return Engine(backend=backend, shots=shots, seed=seed, noise=noise, options=options)


class Model(ABC):
    """Base class for every Q-Cog model.

    Subclasses declare metadata as class attributes, parameters as the ``params``
    tuple, and implement :meth:`simulate` and :meth:`analyse`.  :meth:`run` adds
    parameter handling, execution and the analytical side-effects, so a subclass
    never has to repeat that boilerplate.
    """

    slug: ClassVar[str] = ""
    title: ClassVar[str] = ""
    module: ClassVar[str] = "A"
    one_liner: ClassVar[str] = ""
    cognitive_question: ClassVar[str] = ""
    quantum_mechanism: ClassVar[str] = ""
    classical_baseline: ClassVar[str] = ""
    references: ClassVar[tuple[str, ...]] = ()
    caveats: ClassVar[tuple[str, ...]] = ()
    keywords: ClassVar[tuple[str, ...]] = ()
    difficulty: ClassVar[str] = "intermediate"
    params: ClassVar[tuple[ParamSpec, ...]] = ()

    # ----------------------------------------------------------------- metadata
    @classmethod
    def module_title(cls) -> str:
        """Human-readable name of the module this model belongs to."""
        return MODULES.get(cls.module, "Unfiled")

    @classmethod
    def defaults(cls) -> dict:
        """Default value of every parameter."""
        return coerce_params(cls.params, {})

    @classmethod
    def parameter_table(cls) -> list[dict]:
        """Rows describing the model's parameters."""
        return parameter_table(cls.params)

    @classmethod
    def describe(cls) -> str:
        """Multi-line description used by ``qcog info`` and the documentation."""
        lines = [
            f"{cls.title} ({cls.slug}, module {cls.module})",
            f"  one-liner  : {cls.one_liner}",
            f"  question   : {cls.cognitive_question}",
            f"  mechanism  : {cls.quantum_mechanism}",
            f"  baseline   : {cls.classical_baseline}",
        ]
        if cls.params:
            lines.append("  parameters :")
            for spec in cls.params:
                bounds = ""
                if spec.low is not None or spec.high is not None:
                    bounds = f" in [{spec.low}, {spec.high}]"
                lines.append(
                    f"    - {spec.name} ({spec.kind}{bounds}, default {spec.default!r}): {spec.description}"
                )
        if cls.caveats:
            lines.append("  caveats    :")
            lines.extend(f"    - {caveat}" for caveat in cls.caveats)
        if cls.references:
            lines.append("  references :")
            lines.extend(f"    - {reference}" for reference in cls.references)
        return "\n".join(lines)

    @classmethod
    def to_dict(cls) -> dict:
        return {
            "slug": cls.slug,
            "title": cls.title,
            "module": cls.module,
            "module_title": cls.module_title(),
            "one_liner": cls.one_liner,
            "cognitive_question": cls.cognitive_question,
            "quantum_mechanism": cls.quantum_mechanism,
            "classical_baseline": cls.classical_baseline,
            "difficulty": cls.difficulty,
            "keywords": list(cls.keywords),
            "references": list(cls.references),
            "caveats": list(cls.caveats),
            "parameters": cls.parameter_table(),
            "defaults": cls.defaults(),
        }

    # ----------------------------------------------------------------- execution
    def run(self, **kwargs) -> ModelResult:
        """Simulate the model and analyse the outcome.

        Reserved keyword arguments (``backend``, ``shots``, ``seed``, ``noise``,
        ``engine``) configure execution; everything else is a model parameter and
        is validated against :attr:`params`.
        """
        engine_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key in RESERVED}
        parameters = coerce_params(self.params, kwargs)
        engine = engine_kwargs.pop("engine", None) or make_engine(**engine_kwargs)
        context = self.build(**parameters)
        results = self.simulate(context, engine)
        analysis, findings, summary = self.analyse(context, results)
        durations = [
            value.duration_seconds
            for value in results.values()
            if isinstance(value, SequentialResult)
        ]
        metadata = {
            "engine_backend": engine.backend,
            "engine_shots": engine.shots,
            "engine_seed": engine.seed,
            "runtime_seconds": float(sum(durations)),
            "difficulty": self.difficulty,
            "module_title": self.module_title(),
        }
        metadata.update(self.context_metadata(context))
        return ModelResult(
            slug=self.slug,
            title=self.title,
            module=self.module,
            parameters=parameters,
            results=results,
            analysis=analysis,
            summary=summary,
            findings=findings,
            caveats=list(self.caveats),
            references=list(self.references),
            backend=engine.backend,
            shots=engine.shots,
            seed=engine.seed,
            metadata=metadata,
        )

    # ------------------------------------------------------------ subclass hooks
    @abstractmethod
    def build(self, **parameters) -> dict:
        """Construct the model objects (state, questions, network...) from parameters."""

    @abstractmethod
    def simulate(self, context: dict, engine: Engine) -> dict[str, SequentialResult]:
        """Run the simulations the model is built around."""

    @abstractmethod
    def analyse(self, context: dict, results: dict) -> tuple[dict, list[str], str]:
        """Return ``(analysis, findings, summary)`` for the simulated results."""

    def context_metadata(self, context: dict) -> dict:
        """Extra run metadata worth recording (measured quantities, circuit sizes...)."""
        return {}

    def predict(self, parameters: Mapping[str, Any] | None = None) -> dict[str, float]:
        """The model's predicted probabilities, used for fitting.

        Keys are stable response labels such as ``"AB:yes:yes"``.  Models that are
        not meant to be fitted raise :class:`NotImplementedError`, which the
        fitting layer reports as "this model has no closed-form predictor".
        """
        raise NotImplementedError(f"{self.slug} does not provide closed-form predictions")

    def figures(self, result: ModelResult) -> dict:
        """Named :class:`matplotlib.figure.Figure` objects for this result."""
        from ..viz.figures import default_figures

        return default_figures(self, result)

    def text_alternatives(self, result: ModelResult) -> dict[str, str]:
        """Plain-language descriptions of every figure, keyed by figure name."""
        from ..viz.figures import text_alternatives

        return text_alternatives(self.figures(result))

    # ------------------------------------------------------------------ helpers
    def _engine(self, backend: str = "exact", shots: int | None = None, seed: int | None = None, noise=None) -> Engine:
        return make_engine(backend=backend, shots=shots, seed=seed, noise=noise)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} slug={self.slug!r}>"


def combine_findings(*groups: Sequence[str]) -> list[str]:
    """Flatten groups of finding strings, dropping empties."""
    out: list[str] = []
    for group in groups:
        out.extend(text for text in group if text)
    return out


def format_probability(value: float) -> str:
    """Format a probability for human-readable output."""
    return f"{value:.3f}"


def largest_outcome(distribution: Mapping[str, float]) -> tuple[str, float]:
    """Most likely outcome of a distribution."""
    return max(distribution.items(), key=lambda item: item[1])


def numeric(value) -> float:
    """Coerce numpy scalars to plain floats for reporting."""
    return float(np.asarray(value).ravel()[0])