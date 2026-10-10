"""Fit model parameters to observed response frequencies.

The point of fitting is not to make a model look good; it is to make the
comparison between a quantum model and a classical baseline *fair*.  Both are
fitted to the same data with the same method, and the comparison uses information
criteria that penalise extra parameters.  A model that needs three parameters to
match what the baseline gets with one is not obviously the better explanation, and
Q-Cog is built to say so.

Data are given as a mapping from a response key to a *count* or a *probability*.
The model is a callable ``parameters -> {key: probability}``, which every Q-Cog
model provides through its ``predict`` method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

from .metrics import aic as _aic
from .metrics import bic as _bic
from .metrics import log_likelihood as _log_likelihood
from .params import ParamSpec

__all__ = ["FitResult", "fit", "fittable_parameters", "compare_fits", "R2_LIKE"]


#: A "pseudo R^2" for probability models: 1 - SSE / total sum of squares.
R2_LIKE = "r_squared"


@dataclass
class FitResult:
    """Outcome of a least-squares (or maximum-likelihood) fit."""

    parameters: dict[str, float]
    keys: list[str]
    observed: dict[str, float]
    predicted: dict[str, float]
    cost: float
    success: bool
    message: str
    iterations: int
    n_parameters: int
    weights: list[float] = field(default_factory=list)
    counts: dict[str, float] | None = None
    backend: str = "least_squares"

    @property
    def residuals(self) -> np.ndarray:
        """``predicted - observed`` for every response key."""
        return np.array([self.predicted[key] - self.observed[key] for key in self.keys], dtype=float)

    @property
    def r_squared(self) -> float:
        """Coefficient of determination of the predicted probabilities."""
        observed = np.array([self.observed[key] for key in self.keys], dtype=float)
        predicted = np.array([self.predicted[key] for key in self.keys], dtype=float)
        total = float(np.sum((observed - observed.mean()) ** 2))
        if total <= 0:
            return 1.0 if float(np.sum((observed - predicted) ** 2)) <= 0 else 0.0
        return float(1.0 - np.sum((observed - predicted) ** 2) / total)

    @property
    def rmse(self) -> float:
        """Root-mean-square difference between predictions and observations."""
        return float(np.sqrt(np.mean(self.residuals ** 2)))

    @property
    def n_observations(self) -> int:
        return len(self.keys)

    @property
    def log_likelihood(self) -> float | None:
        """Multinomial log-likelihood (needs counts, i.e. a maximum-likelihood fit)."""
        if self.counts is None:
            return None
        return _log_likelihood(self.predicted, self.counts)

    @property
    def aic(self) -> float | None:
        value = self.log_likelihood
        return None if value is None else _aic(value, self.n_parameters)

    @property
    def bic(self) -> float | None:
        value = self.log_likelihood
        if value is None or self.counts is None:
            return None
        return _bic(value, self.n_parameters, float(sum(self.counts.values())))

    def table(self):
        """Per-key comparison as a :class:`pandas.DataFrame`."""
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "response": key,
                    "observed": self.observed[key],
                    "predicted": self.predicted[key],
                    "residual": self.predicted[key] - self.observed[key],
                }
                for key in self.keys
            ]
        )

    def to_dict(self) -> dict:
        return {
            "parameters": {key: round(float(value), 8) for key, value in self.parameters.items()},
            "cost": round(float(self.cost), 10),
            "r_squared": round(self.r_squared, 6),
            "rmse": round(self.rmse, 6),
            "log_likelihood": None if self.log_likelihood is None else round(self.log_likelihood, 6),
            "aic": None if self.aic is None else round(self.aic, 6),
            "bic": None if self.bic is None else round(self.bic, 6),
            "n_parameters": self.n_parameters,
            "n_observations": self.n_observations,
            "success": bool(self.success),
            "message": self.message,
            "backend": self.backend,
            "observed": {key: round(value, 8) for key, value in self.observed.items()},
            "predicted": {key: round(value, 8) for key, value in self.predicted.items()},
        }

    def summary(self) -> str:
        parameters = ", ".join(f"{key} = {value:.4f}" for key, value in self.parameters.items())
        quality = f"R2 = {self.r_squared:.4f}, RMSE = {self.rmse:.4f}"
        if self.aic is not None and self.bic is not None:
            quality += f", AIC = {self.aic:.2f}, BIC = {self.bic:.2f}"
        return f"fitted {parameters} | {quality}"


def fittable_parameters(specs: Sequence[ParamSpec]) -> list[ParamSpec]:
    """The numeric parameters that :func:`fit` is allowed to vary."""
    return [spec for spec in specs if spec.fittable and spec.kind in {"float", "int"}]


def fit(
    predict: Callable[[dict], Mapping[str, float]],
    observed: Mapping[str, float],
    specs: Sequence[ParamSpec],
    *,
    counts: Mapping[str, float] | None = None,
    max_iterations: int = 2000,
    tolerance: float = 1e-12,
    weights: Mapping[str, float] | None = None,
    start: Mapping[str, float] | None = None,
) -> FitResult:
    """Fit the fittable parameters of a model to observed frequencies.

    Parameters
    ----------
    predict:
        ``parameters -> {response key: probability}``.  Keys not present are
        treated as zero, so a model may return only the responses it predicts.
    observed:
        Observed probability for each response key.  Pass ``counts`` as well when
        the data are raw counts: that switches on the multinomial likelihood,
        the chi-square style weighting and AIC/BIC.
    specs:
        The model's parameter specifications; only ``fittable=True`` numeric
        parameters are varied, the rest stay at their declared defaults.
    """
    variable = fittable_parameters(specs)
    keys = list(observed)
    defaults = {spec.name: spec.coerce(spec.default) for spec in specs}
    if start:
        defaults.update({key: specs_lookup(specs, key).coerce(value) for key, value in start.items()})
    target = np.array([float(observed[key]) for key in keys], dtype=float)
    if counts is not None:
        total = float(sum(float(counts.get(key, 0.0)) for key in keys)) or 1.0
        sigma = np.array(
            [np.sqrt(max(float(counts.get(key, 0.0)), 1.0)) / total for key in keys], dtype=float
        )
    elif weights is not None:
        sigma = np.array([1.0 / max(float(weights.get(key, 1.0)), 1e-9) for key in keys], dtype=float)
    else:
        sigma = np.ones(len(keys), dtype=float)

    def residuals(vector: np.ndarray) -> np.ndarray:
        parameters = dict(defaults)
        for spec, value in zip(variable, vector):
            parameters[spec.name] = spec.coerce(float(value))
        predicted = predict(parameters)
        difference = np.array([float(predicted.get(key, 0.0)) - target[index] for index, key in enumerate(keys)])
        return difference / sigma

    if variable:
        initial = np.array([float(spec.coerce(defaults[spec.name])) for spec in variable], dtype=float)
        bounds = (
            [(-np.inf if spec.low is None else float(spec.low)) for spec in variable],
            [(np.inf if spec.high is None else float(spec.high)) for spec in variable],
        )
        solution = least_squares(
            residuals, initial, bounds=bounds, max_nfev=max_iterations, xtol=tolerance, ftol=tolerance
        )
        fitted = dict(defaults)
        for spec, value in zip(variable, solution.x):
            fitted[spec.name] = spec.coerce(float(value))
        cost = float(solution.cost)
        success = bool(solution.success)
        message = str(solution.message)
        iterations = int(solution.nfev)
    else:
        fitted = dict(defaults)
        cost = float(np.sum(residuals(np.array([])) ** 2))
        success = True
        message = "no fittable parameters: the model was evaluated at its defaults"
        iterations = 1

    predicted_final = {key: float(predict(fitted).get(key, 0.0)) for key in keys}
    return FitResult(
        parameters={spec.name: float(fitted[spec.name]) for spec in variable} if variable else {},
        keys=keys,
        observed={key: float(observed[key]) for key in keys},
        predicted=predicted_final,
        cost=cost,
        success=success,
        message=message,
        iterations=iterations,
        n_parameters=len(variable) or len(specs),
        weights=[float(value) for value in sigma],
        counts={key: float(value) for key, value in counts.items()} if counts is not None else None,
        backend="least_squares",
    )


def specs_lookup(specs: Sequence[ParamSpec], name: str) -> ParamSpec:
    """Find a parameter specification by name."""
    for spec in specs:
        if spec.name == name:
            return spec
    raise KeyError(f"no parameter named {name!r}")


def compare_fits(quantum: FitResult, classical: FitResult, alpha: float = 2.0) -> dict:
    """Compare two fits, preferring the one with the lower information criterion.

    A difference in AIC below ``alpha`` is reported as "indistinguishable" rather
    than as a win, because that is what the number supports.
    """
    verdict = "no information criterion available (fit used probabilities rather than counts)"
    delta_aic = delta_bic = None
    if quantum.aic is not None and classical.aic is not None:
        delta_aic = float(classical.aic - quantum.aic)
        delta_bic = (
            float(classical.bic - quantum.bic)
            if quantum.bic is not None and classical.bic is not None
            else None
        )
        if abs(delta_aic) < alpha:
            verdict = "the two accounts are statistically indistinguishable on these data"
        elif delta_aic > 0:
            verdict = "the quantum model is favoured (lower AIC)"
        else:
            verdict = "the classical baseline is favoured (lower AIC)"
    return {
        "delta_aic_quantum_minus_classical": None if delta_aic is None else round(-delta_aic, 6),
        "delta_bic_quantum_minus_classical": None if delta_bic is None else round(-delta_bic, 6),
        "verdict": verdict,
        "quantum": quantum.to_dict(),
        "classical": classical.to_dict(),
    }