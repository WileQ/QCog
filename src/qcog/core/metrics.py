"""Distance measures, information criteria and simple statistical tests.

These are the numbers Q-Cog uses to compare a quantum model with a classical
baseline, and to say how confident a comparison is.  They are intentionally
standard: a reader should be able to check any of them by hand or with SciPy.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy import stats

__all__ = [
    "total_variation",
    "kl_divergence",
    "jensen_shannon",
    "hellinger",
    "bhattacharyya",
    "chi_square_test",
    "g_test",
    "log_likelihood",
    "aic",
    "bic",
    "proportion_z_test",
    "agreement_index",
    "clamp_probability",
    "normalise",
]


def _align(first: Mapping[str, float], second: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
    keys = sorted(set(first) | set(second))
    left = np.array([first.get(key, 0.0) for key in keys], dtype=float)
    right = np.array([second.get(key, 0.0) for key in keys], dtype=float)
    if left.sum() > 0:
        left = left / left.sum()
    if right.sum() > 0:
        right = right / right.sum()
    return left, right


def total_variation(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    """Total variation distance ``0.5 * sum |p - q|`` in ``[0, 1]``."""
    left, right = _align(first, second)
    return float(0.5 * np.abs(left - right).sum())


def kl_divergence(first: Mapping[str, float], second: Mapping[str, float], epsilon: float = 1e-12) -> float:
    """Kullback-Leibler divergence ``D(p || q)`` in bits."""
    left, right = _align(first, second)
    left = np.clip(left, epsilon, None)
    right = np.clip(right, epsilon, None)
    return float(np.sum(left * np.log2(left / right)))


def jensen_shannon(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    """Jensen-Shannon divergence in bits (symmetric, bounded by 1)."""
    left, right = _align(first, second)
    middle = 0.5 * (left + right)
    return float(0.5 * kl_divergence(dict(enumerate(left)), dict(enumerate(middle)))
                 + 0.5 * kl_divergence(dict(enumerate(right)), dict(enumerate(middle))))


def hellinger(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    """Hellinger distance in ``[0, 1]``."""
    left, right = _align(first, second)
    return float(np.sqrt(0.5 * np.sum((np.sqrt(left) - np.sqrt(right)) ** 2)))


def bhattacharyya(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    """Bhattacharyya coefficient in ``[0, 1]``."""
    left, right = _align(first, second)
    return float(np.sum(np.sqrt(left * right)))


def chi_square_test(observed: Sequence[float], expected: Sequence[float]) -> dict:
    """Pearson chi-square goodness of fit.

    Returns the statistic, degrees of freedom and p-value.  Used to ask whether a
    fitted quantum model is *rejected* by a set of observed frequencies, which is
    the question a judge or a reviewer will actually ask.
    """
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if observed.shape != expected.shape:
        raise ValueError("observed and expected must have the same shape")
    mask = expected > 0
    statistic = float(np.sum((observed[mask] - expected[mask]) ** 2 / expected[mask]))
    dof = max(int(mask.sum()) - 1, 1)
    return {
        "statistic": statistic,
        "dof": dof,
        "p_value": float(stats.chi2.sf(statistic, dof)),
    }


def g_test(observed: Sequence[float], expected: Sequence[float]) -> dict:
    """Likelihood-ratio (G) test, the information-theoretic sibling of chi-square."""
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    mask = (observed > 0) & (expected > 0)
    statistic = 2.0 * float(np.sum(observed[mask] * np.log(observed[mask] / expected[mask])))
    dof = max(int(mask.sum()) - 1, 1)
    return {"statistic": statistic, "dof": dof, "p_value": float(stats.chi2.sf(statistic, dof))}


def log_likelihood(probabilities: Mapping[str, float], counts: Mapping[str, float], epsilon: float = 1e-12) -> float:
    """Multinomial log-likelihood of ``counts`` under ``probabilities``."""
    total = 0.0
    for key, count in counts.items():
        probability = max(float(probabilities.get(key, 0.0)), epsilon)
        total += float(count) * np.log(probability)
    return float(total)


def aic(log_likelihood_value: float, num_parameters: int) -> float:
    """Akaike information criterion (lower is better)."""
    return float(2 * num_parameters - 2 * log_likelihood_value)


def bic(log_likelihood_value: float, num_parameters: int, num_observations: float) -> float:
    """Bayesian information criterion (lower is better)."""
    if num_observations <= 0:
        raise ValueError("num_observations must be positive")
    return float(num_parameters * np.log(num_observations) - 2 * log_likelihood_value)


def proportion_z_test(estimate: float, reference: float, samples: int) -> dict:
    """Two-sided z test for a proportion against a reference value."""
    if samples <= 0:
        raise ValueError("samples must be positive")
    standard_error = float(np.sqrt(max(reference * (1.0 - reference), 1e-12) / samples))
    z_value = (estimate - reference) / standard_error
    return {
        "z": float(z_value),
        "p_value": float(2 * stats.norm.sf(abs(z_value))),
        "standard_error": standard_error,
    }


def agreement_index(distribution: Mapping[tuple, float]) -> float:
    """Probability of answering both questions the same way.

    Used for the QQ equality: a classical model may produce any value, a quantum
    model with compatible questions predicts exactly ``1``, and a quantum model
    with incompatible questions predicts the same value in both orders.
    """
    total = 0.0
    for key, value in distribution.items():
        if len(key) == 2 and key[0] == key[1]:
            total += float(value)
    return float(total)


def clamp_probability(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clip a value into ``[low, high]`` and return it as a float."""
    return float(min(max(float(value), low), high))


def normalise(distribution: Mapping[str, float]) -> dict[str, float]:
    """Return the distribution rescaled to sum to one (raises if empty)."""
    total = float(sum(distribution.values()))
    if total <= 0:
        raise ValueError("cannot normalise an all-zero distribution")
    return {key: float(value) / total for key, value in distribution.items()}