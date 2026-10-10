"""Analytic backends: exact evaluation and shot sampling of the same model."""

from __future__ import annotations

import time
from typing import Mapping, Sequence

import numpy as np

from ..core.instruments import Channel, NoiseModel, apply_channels
from ..core.measurement import Measurement, POVM, ProjectiveMeasurement
from ..core.results import SequentialResult, StepRecord
from ..core.states import CognitiveState, as_state

__all__ = ["ExactBackend", "SampledBackend"]


class ExactBackend:
    """Evaluate the model analytically with Qiskit ``Statevector``/``DensityMatrix``.

    This backend is the reference implementation: it applies the Born rule and
    the Lueders projection postulate directly to the projectors, without any
    circuit compilation.  Every other backend is validated against it.
    """

    name = "exact"
    description = "analytic evaluation of the Born rule and the Lueders update (Qiskit statevector/density matrix)"

    def __init__(self, **options) -> None:
        self.options = options

    # ------------------------------------------------------------------ public
    def run(
        self,
        state,
        measurements: Sequence[Measurement],
        *,
        noise: NoiseModel | None = None,
        between_steps: Mapping[int, Sequence[Channel]] | None = None,
        shots: int | None = None,
        seed: int | None = None,
        **metadata,
    ) -> SequentialResult:
        """Run the sequence and return the joint distribution."""
        confidence = as_state(state)
        questions = list(measurements)
        if not questions:
            raise ValueError("at least one question is required")
        names = [question.name for question in questions]
        if len(set(names)) != len(names):
            raise ValueError(f"question names must be unique within one interview, got {names}")
        noise = noise or NoiseModel()
        started = time.perf_counter()
        joint: dict[tuple[str, ...], float] = {}
        steps: list[StepRecord] = []
        states: dict[tuple[str, ...], np.ndarray] = {}

        def walk(step: int, current: CognitiveState, prefix: tuple[str, ...], probability: float) -> None:
            if step == len(questions):
                joint[prefix] = joint.get(prefix, 0.0) + probability
                return
            if between_steps and step in between_steps:
                current = apply_channels(current, between_steps[step])
            if noise.apply_to == "before_first" and step == 0:
                current = noise.apply(current, step=0, num_steps=len(questions))
            question = questions[step]
            distribution = question.probabilities(current)
            outcomes = tuple(question.outcomes)
            steps.append(
                StepRecord(
                    step=step,
                    measurement=question.name,
                    outcomes=outcomes,
                    prefix=prefix,
                    conditional={outcome: float(p) for outcome, p in zip(outcomes, distribution)},
                )
            )
            for outcome, branch_probability in zip(outcomes, distribution):
                if branch_probability <= 1e-15:
                    continue
                try:
                    post = question.post_state(current, outcome)
                except ValueError:
                    # The branch probability is numerically zero: there is no state
                    # to continue with, so the branch contributes nothing to the joint
                    # distribution.  Fitting explores such corners of parameter space,
                    # so this must not raise.
                    continue
                if noise.apply_to == "each_question":
                    post = noise.apply(post, step=step, num_steps=len(questions))
                states[prefix + (outcome,)] = post.density.copy()
                walk(step + 1, post, prefix + (outcome,), probability * float(branch_probability))

        walk(0, confidence, (), 1.0)
        duration = time.perf_counter() - started
        result = SequentialResult(
            order=tuple(names),
            outcomes={question.name: tuple(question.outcomes) for question in questions},
            joint=joint,
            steps=steps,
            states=states,
            backend=self.name,
            shots=None,
            seed=None,
            duration_seconds=duration,
            noise=noise.to_dict() if not noise.is_trivial else None,
            metadata={**metadata, "measurements": {question.name: question for question in questions}},
        )
        if shots:
            return SampledBackend(**self.options).resample(result, shots, seed=seed)
        return result


class SampledBackend(ExactBackend):
    """Draw finite-shot samples from the analytic distribution.

    Useful to show how an experiment behaves with the finite number of
    participants (or shots) a real study, or a real device, would have: the
    order effect stays visible, the QQ equality becomes approximate, and the
    statistical tests in :mod:`qcog.analysis` start to matter.
    """

    name = "sampled"
    description = "analytic distribution with multinomial shot sampling"

    def run(self, state, measurements, *, noise=None, between_steps=None, shots: int = 4096, seed: int | None = None, **metadata) -> SequentialResult:
        if shots <= 0:
            raise ValueError("shots must be a positive integer")
        exact = ExactBackend(**self.options).run(
            state, measurements, noise=noise, between_steps=between_steps, **metadata
        )
        return self.resample(exact, shots, seed=seed)

    def resample(self, result: SequentialResult, shots: int, seed: int | None = None) -> SequentialResult:
        """Convert an analytic result into a finite-shot sample of itself."""
        if shots <= 0:
            raise ValueError("shots must be a positive integer")
        keys = list(result.joint)
        probabilities = np.array([result.joint[key] for key in keys], dtype=float)
        total = probabilities.sum()
        if total <= 0:
            raise ValueError("cannot sample from an empty distribution")
        probabilities = probabilities / total
        rng = np.random.default_rng(seed)
        draws = rng.choice(len(keys), size=shots, p=probabilities)
        counts: dict[tuple[str, ...], int] = {}
        for draw in draws:
            key = keys[int(draw)]
            counts[key] = counts.get(key, 0) + 1
        sampled = {key: counts.get(key, 0) / shots for key in keys}
        return SequentialResult(
            order=result.order,
            outcomes=result.outcomes,
            joint=sampled,
            steps=result.steps,
            states=result.states,
            backend="sampled",
            shots=shots,
            seed=seed,
            counts=counts,
            duration_seconds=result.duration_seconds,
            noise=result.noise,
            metadata={**result.metadata, "sampled_from": result.backend},
        )