"""The execution facade used by models, the CLI and the interactive explorer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .instruments import Channel, NoiseModel, NoiseSpec
from .measurement import Measurement
from .results import SequentialResult

__all__ = ["Engine", "run_sequence"]


@dataclass
class Engine:
    """Configure *how* an interview is executed, independently of the model.

    Parameters
    ----------
    backend:
        ``"exact"`` (analytic, default), ``"sampled"``, ``"aer"``,
        ``"aer_statevector"``, ``"aer_density_matrix"`` or ``"ionq"``.
    shots:
        Number of shots; ignored by the analytic backend unless explicitly set.
    seed:
        Seed for reproducible sampling.
    noise:
        A :class:`~qcog.core.instruments.NoiseSpec` or
        :class:`~qcog.core.instruments.NoiseModel` describing decoherence.
    between_steps:
        Optional mapping ``step index -> channels`` applied *before* that
        question.  This is how a model asks a question "without recording the
        answer", the mechanism behind question-order effects.
    """

    backend: str = "exact"
    shots: int | None = None
    seed: int | None = None
    noise: NoiseSpec | NoiseModel | None = None
    between_steps: Mapping[int, Sequence[Channel]] | None = None
    options: dict = field(default_factory=dict)

    def resolve_backend(self):
        """Instantiate the configured backend object."""
        from ..backends import get_backend

        return get_backend(self.backend, **self.options)

    def resolve_noise(self) -> NoiseModel:
        """The :class:`NoiseModel` implied by :attr:`noise`.

        ``noise`` here is the *execution* keyword (a :class:`NoiseSpec` or
        :class:`NoiseModel`), which is why a model's own decoherence parameter is never
        called ``noise``.  A wrong type is reported clearly rather than failing later.
        """
        if isinstance(self.noise, NoiseModel):
            return self.noise
        if self.noise is None:
            return NoiseModel()
        if isinstance(self.noise, NoiseSpec):
            return NoiseModel.from_spec(self.noise)
        raise TypeError(
            "the 'noise' keyword belongs to the execution layer and expects a NoiseSpec or a NoiseModel; "
            f"got {type(self.noise).__name__}. To set a model's own decoherence, use that model's parameter "
            "(for example decoherence=...)."
        )

    def run(self, state, measurements: Sequence[Measurement], **overrides) -> SequentialResult:
        """Execute one interview and return its joint response distribution."""
        backend = self.resolve_backend()
        noise = self.resolve_noise()
        kwargs = {
            "noise": noise,
            "between_steps": self.between_steps,
            "seed": self.seed,
        }
        if self.shots is not None:
            kwargs["shots"] = self.shots
        kwargs.update(overrides)
        return backend.run(state, measurements, **{k: v for k, v in kwargs.items() if v is not None})


def run_sequence(
    state,
    measurements: Sequence[Measurement],
    *,
    backend: str = "exact",
    shots: int | None = None,
    seed: int | None = None,
    noise: NoiseSpec | NoiseModel | None = None,
    between_steps: Mapping[int, Sequence[Channel]] | None = None,
    **options,
) -> SequentialResult:
    """Convenience wrapper around :class:`Engine` for one-off simulations."""
    engine = Engine(
        backend=backend,
        shots=shots,
        seed=seed,
        noise=noise,
        between_steps=between_steps,
        options=options,
    )
    return engine.run(state, measurements)