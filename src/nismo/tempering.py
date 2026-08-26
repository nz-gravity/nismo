"""Power-tempered importance normalization and finite-pool construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from .exceptions import InvalidProposalOutput
from .proposals import Proposal


@dataclass(frozen=True, slots=True)
class BetaTemperingDiagnostics:
    """Diagnostics for the direct-Monte-Carlo power-tempering stage."""

    beta: float
    log_z_beta: float
    log_z_beta_error: float
    n_mc_samples: int
    mc_effective_sample_size: float
    pool_size: int
    normalization_exact: bool

    def __post_init__(self) -> None:
        if not np.isfinite(self.beta) or not 0.0 < self.beta <= 1.0:
            raise ValueError("beta must be finite and in (0, 1]")
        if not np.isfinite(self.log_z_beta):
            raise ValueError("log_z_beta must be finite")
        if not np.isfinite(self.log_z_beta_error) or self.log_z_beta_error < 0.0:
            raise ValueError("log_z_beta_error must be finite and non-negative")
        if self.n_mc_samples < 0 or self.pool_size < 1:
            raise ValueError("tempering sample counts are invalid")
        if (
            not np.isfinite(self.mc_effective_sample_size)
            or self.mc_effective_sample_size < 0.0
        ):
            raise ValueError("tempering effective sample size is invalid")


def exact_beta_diagnostics(*, pool_size: int) -> BetaTemperingDiagnostics:
    """Return the no-Monte-Carlo diagnostics for the exact ``beta=1`` path."""
    return BetaTemperingDiagnostics(
        beta=1.0,
        log_z_beta=0.0,
        log_z_beta_error=0.0,
        n_mc_samples=0,
        mc_effective_sample_size=0.0,
        pool_size=pool_size,
        normalization_exact=True,
    )


def _validate_candidate_batch(
    points: NDArray[np.float64],
    *,
    n: int,
    ndim: int,
) -> NDArray[np.float64]:
    candidates = np.asarray(points, dtype=float)
    if candidates.shape != (n, ndim):
        raise InvalidProposalOutput(
            f"proposal sample must have shape {(n, ndim)}, got {candidates.shape}"
        )
    if not np.all(np.isfinite(candidates)):
        raise InvalidProposalOutput("proposal sample contains NaN or infinity")
    return candidates


def sample_power_tempered_pool(
    proposal: Proposal,
    *,
    beta: float,
    pool_size: int,
    n_mc_samples: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], BetaTemperingDiagnostics]:
    r"""Estimate ``z_beta`` and construct a finite sample from ``q**beta``.

    For ``0 < beta < 1``, direct Monte Carlo under the normalized density
    ``q`` gives

    ``z_beta = E_q[q(theta)**(beta - 1)]``.

    The same independent candidate batch is converted into a unique finite
    pool by weighted sampling without replacement.  This is a finite-candidate
    sampling-importance-resampling approximation to the normalized
    ``q(theta)**beta / z_beta`` density.  The returned ESS and log-normalizer
    error make the quality of that approximation auditable.

    ``beta=1`` is an exact fast path and consumes precisely one ordinary
    proposal draw, preserving the pre-tempering sampler behavior.
    """
    if not np.isfinite(beta) or not 0.0 < beta <= 1.0:
        raise ValueError("beta must be finite and in (0, 1]")
    if pool_size < 1:
        raise ValueError("pool_size must be positive")
    if beta == 1.0:
        points = _validate_candidate_batch(
            proposal.sample(pool_size, rng),
            n=pool_size,
            ndim=proposal.ndim,
        )
        return np.array(points, copy=True), exact_beta_diagnostics(pool_size=pool_size)
    if n_mc_samples < pool_size:
        raise ValueError("n_mc_samples must be at least pool_size")

    candidates = _validate_candidate_batch(
        proposal.sample(n_mc_samples, rng),
        n=n_mc_samples,
        ndim=proposal.ndim,
    )
    log_q = np.asarray(proposal.log_prob(candidates), dtype=float)
    if log_q.shape != (n_mc_samples,):
        raise InvalidProposalOutput(
            "proposal log_prob must return one value per beta Monte Carlo sample"
        )
    if not np.all(np.isfinite(log_q)):
        raise InvalidProposalOutput(
            "proposal log_prob must be finite at its own beta Monte Carlo samples"
        )

    log_weights = (beta - 1.0) * log_q
    log_weight_sum = float(logsumexp(log_weights))
    log_z_beta = log_weight_sum - np.log(n_mc_samples)
    log_weight_square_sum = float(logsumexp(2.0 * log_weights))
    log_ess = 2.0 * log_weight_sum - log_weight_square_sum
    mc_ess = min(float(n_mc_samples), float(np.exp(log_ess)))

    log_relative_second_moment = (
        log_weight_square_sum - np.log(n_mc_samples) - 2.0 * log_z_beta
    )
    if log_relative_second_moment > np.log(np.finfo(float).max):
        raise InvalidProposalOutput(
            "beta Monte Carlo normalization has non-finite variance; "
            "increase beta or use a better reference distribution"
        )
    relative_variance = max(0.0, float(np.expm1(log_relative_second_moment)))
    log_z_beta_error = float(np.sqrt(relative_variance / n_mc_samples))
    if not np.isfinite(log_z_beta_error):
        raise InvalidProposalOutput(
            "beta Monte Carlo normalization has non-finite uncertainty"
        )

    probabilities = np.exp(log_weights - log_weight_sum)
    positive = int(np.count_nonzero(probabilities > 0.0))
    if positive < pool_size:
        raise InvalidProposalOutput(
            "beta Monte Carlo weights contain too few positive candidates for "
            f"a unique pool of size {pool_size}; increase beta_mc_samples"
        )
    selected = np.asarray(
        rng.choice(
            n_mc_samples,
            size=pool_size,
            replace=False,
            p=probabilities,
        ),
        dtype=np.int64,
    )
    pool = np.array(candidates[selected], copy=True)
    diagnostics = BetaTemperingDiagnostics(
        beta=beta,
        log_z_beta=log_z_beta,
        log_z_beta_error=log_z_beta_error,
        n_mc_samples=n_mc_samples,
        mc_effective_sample_size=mc_ess,
        pool_size=pool_size,
        normalization_exact=False,
    )
    return pool, diagnostics
