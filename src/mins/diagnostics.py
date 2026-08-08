"""Pure diagnostic summaries for stored MINS results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .results import MINSResult


@dataclass(frozen=True, slots=True)
class RunDiagnostics:
    """Small run-health summary derived without mutating the result."""

    posterior_ess: float
    relative_posterior_ess: float
    proposal_acceptance_fraction: float
    maximum_proposals_per_replacement: int
    thresholds_monotone: bool
    conservative_log_remaining: float
    final_remaining_fraction: float
    final_remaining_dlogz: float
    final_live_ess: float
    final_live_mean_rse: float
    final_live_logz_error: float
    final_logz_stability: float
    final_stopping_streak: int
    queue_efficiency: float
    compute_efficiency: float


def posterior_ess(result: MINSResult) -> float:
    """Return Kish effective sample size of normalized quadrature weights."""
    weights = result.posterior_weights
    return float(1.0 / np.sum(weights**2))


def summarize(result: MINSResult) -> RunDiagnostics:
    """Calculate minimal evidence-run health diagnostics."""
    ess = posterior_ess(result)
    n_weighted = result.niter + result.nlive
    acceptance = result.niter / result.n_proposals if result.n_proposals else 0.0
    maximum_proposals = int(np.max(result.history.proposals)) if result.niter else 0
    log_x = -result.niter / result.nlive
    conservative = log_x + float(np.max(result.final_live_log_psi0))
    if result.niter:
        final_remaining_fraction = float(result.history.remaining_fraction[-1])
        final_remaining_dlogz = float(result.history.remaining_dlogz[-1])
        final_live_ess = float(result.history.live_ess[-1])
        final_live_mean_rse = float(result.history.live_mean_rse[-1])
        final_live_logz_error = float(result.history.live_logz_error[-1])
        final_logz_stability = float(result.history.logz_stability[-1])
        final_stopping_streak = int(result.history.stopping_streak[-1])
    else:
        final_remaining_fraction = float("nan")
        final_remaining_dlogz = float("nan")
        final_live_ess = float("nan")
        final_live_mean_rse = float("nan")
        final_live_logz_error = float("nan")
        final_logz_stability = float("nan")
        final_stopping_streak = 0
    return RunDiagnostics(
        posterior_ess=ess,
        relative_posterior_ess=ess / n_weighted,
        proposal_acceptance_fraction=acceptance,
        maximum_proposals_per_replacement=maximum_proposals,
        thresholds_monotone=bool(np.all(np.diff(result.dead_log_psi0) >= 0.0)),
        conservative_log_remaining=conservative,
        final_remaining_fraction=final_remaining_fraction,
        final_remaining_dlogz=final_remaining_dlogz,
        final_live_ess=final_live_ess,
        final_live_mean_rse=final_live_mean_rse,
        final_live_logz_error=final_live_logz_error,
        final_logz_stability=final_logz_stability,
        final_stopping_streak=final_stopping_streak,
        queue_efficiency=result.queue_diagnostics.queue_efficiency,
        compute_efficiency=result.queue_diagnostics.compute_efficiency,
    )
