"""Pure log-space nested-sampling quadrature."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp

from .exceptions import NumericalInvariantError


@dataclass(frozen=True, slots=True)
class QuadratureSummary:
    """Final evidence, posterior weights, information, and uncertainty."""

    logz: float
    information: float
    logzerr: float
    log_contributions: NDArray[np.float64]
    log_posterior_weights: NDArray[np.float64]


def update_log_weighted_mean(
    log_total: float,
    mean: float,
    log_contribution: float,
    value: float,
) -> tuple[float, float]:
    """Add one positive log-space contribution to a weighted mean.

    Parameters
    ----------
    log_total
        Logarithm of the current total weight, or ``-inf`` when empty.
    mean
        Current weighted mean. Ignored when ``log_total == -inf``.
    log_contribution
        Logarithm of the new non-negative weight.
    value
        Value associated with the new weight.

    Returns
    -------
    tuple
        Updated ``(log_total, weighted_mean)``.
    """
    if np.isneginf(log_contribution):
        return log_total, mean
    if np.isneginf(log_total):
        return log_contribution, value
    new_log_total = float(np.logaddexp(log_total, log_contribution))
    old_fraction = float(np.exp(log_total - new_log_total))
    new_fraction = float(np.exp(log_contribution - new_log_total))
    return new_log_total, old_fraction * mean + new_fraction * value


def estimate_information(
    *,
    logz_dead: float,
    dead_log_psi_mean: float,
    logz_live: float,
    live_log_psi: ArrayLike,
    logz_total: float,
) -> float:
    """Estimate current information from dead and remaining live mass.

    This is the same discrete information calculation used at finalization,
    expressed incrementally for progress reporting.

    Returns
    -------
    float
        Non-negative information estimate in nats. Tiny negative round-off is
        clipped to zero.
    """
    live_psi = np.asarray(live_log_psi, dtype=float)
    if live_psi.ndim != 1 or len(live_psi) < 2:
        raise ValueError("live_log_psi must be one-dimensional with length >= 2")
    total_mean = 0.0
    if np.isfinite(logz_dead):
        total_mean += float(np.exp(logz_dead - logz_total)) * dead_log_psi_mean
    if np.isfinite(logz_live):
        positive = np.isfinite(live_psi)
        live_normalizer = float(logsumexp(live_psi))
        live_mean = float(
            np.sum(np.exp(live_psi[positive] - live_normalizer) * live_psi[positive])
        )
        total_mean += float(np.exp(logz_live - logz_total)) * live_mean
    return max(0.0, total_mean - logz_total)


def logdiffexp(log_a: float, log_b: float) -> float:
    """Return ``log(exp(log_a) - exp(log_b))`` for ``log_a > log_b``.

    Raises
    ------
    ValueError
        If the interval is not strictly positive.
    """
    if not log_a > log_b:
        raise ValueError("log_a must be greater than log_b")
    return float(log_a + np.log1p(-np.exp(log_b - log_a)))


def dead_log_contribution(
    iteration: int,
    n_live: int,
    dead_log_psi: float,
) -> tuple[float, float, float]:
    """Return ``(log_x, log_delta_x, log_weight)`` for one dead point."""
    if iteration < 1:
        raise ValueError("iteration must be >= 1")
    if n_live < 2:
        raise ValueError("n_live must be >= 2")
    log_x_prev = -(iteration - 1) / n_live
    log_x = -iteration / n_live
    log_delta_x = logdiffexp(log_x_prev, log_x)
    return log_x, log_delta_x, log_delta_x + dead_log_psi


def live_log_contributions(
    log_x: float,
    live_log_psi: ArrayLike,
) -> NDArray[np.float64]:
    """Return individual final-live log evidence contributions."""
    values = np.asarray(live_log_psi, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("live_log_psi must be one-dimensional with length >= 2")
    return np.asarray(log_x - np.log(len(values)) + values, dtype=np.float64)


def estimated_live_logz(log_x: float, live_log_psi: ArrayLike) -> float:
    """Return the mean-live estimate of remaining log evidence."""
    contributions = live_log_contributions(log_x, live_log_psi)
    return float(logsumexp(contributions))


def finalize_quadrature(
    dead_log_weights: ArrayLike,
    dead_log_psi: ArrayLike,
    log_x: float,
    live_log_psi: ArrayLike,
    n_live: int,
) -> QuadratureSummary:
    """Combine dead and live contributions and calculate ``Z``, ``H``, error.

    All inputs and outputs are in natural-log units except ``information``,
    which is in nats and is non-negative.
    """
    dead_weights = np.asarray(dead_log_weights, dtype=float)
    dead_psi = np.asarray(dead_log_psi, dtype=float)
    live_psi = np.asarray(live_log_psi, dtype=float)
    if dead_weights.ndim != 1 or dead_psi.shape != dead_weights.shape:
        raise ValueError("dead arrays must be one-dimensional with equal shape")
    if live_psi.shape != (n_live,):
        raise ValueError(f"live_log_psi must have shape ({n_live},)")

    live_weights = live_log_contributions(log_x, live_psi)
    contributions = np.concatenate((dead_weights, live_weights))
    all_log_psi = np.concatenate((dead_psi, live_psi))
    # Normalize after removing the common log-scale explicitly.  Calling
    # ``logsumexp(contributions)`` and then subtracting that potentially very
    # large-magnitude result loses precision in high-dimensional problems.
    # Keeping the subtraction in the shifted frame makes the posterior
    # weights normalize to working precision even when log(Z) is far from
    # zero.
    normalization_offset = float(np.max(contributions))
    if not np.isfinite(normalization_offset):
        raise NumericalInvariantError("final evidence is not finite")
    shifted_contributions = contributions - normalization_offset
    log_normalizer = float(logsumexp(shifted_contributions))
    logz = normalization_offset + log_normalizer
    if not np.isfinite(logz):
        raise NumericalInvariantError("final evidence is not finite")
    log_posterior_weights = shifted_contributions - log_normalizer
    posterior_weights = np.exp(log_posterior_weights)
    if not np.isclose(logsumexp(log_posterior_weights), 0.0, rtol=0.0, atol=1.0e-12):
        raise NumericalInvariantError("posterior weights do not sum to one")
    positive = posterior_weights > 0.0
    information = float(
        np.sum(posterior_weights[positive] * (all_log_psi[positive] - logz))
    )
    if information < -1.0e-10:
        raise NumericalInvariantError(
            f"materially negative information estimate: {information}"
        )
    information = max(0.0, information)
    return QuadratureSummary(
        logz=logz,
        information=information,
        logzerr=float(np.sqrt(information / n_live)),
        log_contributions=contributions,
        log_posterior_weights=log_posterior_weights,
    )
