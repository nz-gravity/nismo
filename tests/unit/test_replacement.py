from __future__ import annotations

import numpy as np
import pytest

from nismo import (
    ConfigurationError,
    EvaluationCounts,
    ParallelSettings,
    QueueDiagnostics,
    ReplacementResult,
    ReplacementSnapshot,
)
from nismo.constrained import ConstrainedAttempt, ConstrainedDraw, EvaluatedPoint
from nismo.replacement import ReplacementQueue

pytestmark = pytest.mark.unit


def _result(job_id: int, *, log_psi0: float, revision: int = 0) -> ReplacementResult:
    point = EvaluatedPoint(
        theta=np.array([float(job_id)]),
        log_likelihood=0.0,
        log_prior=0.0,
        log_q0=0.0,
        log_psi0=log_psi0,
        tie_breaker=0.5,
    )
    attempt = ConstrainedAttempt(
        draw=ConstrainedDraw(point=point, n_proposed=3, n_valid=1),
        reason=None,
        n_proposed=3,
        n_valid=1,
    )
    return ReplacementResult(
        job_id=job_id,
        attempt=attempt,
        threshold_at_creation=0.0,
        threshold_tie_breaker_at_creation=0.0,
        proposal_revision=revision,
        counts=EvaluationCounts(likelihood_calls=3, prior_calls=3),
    )


def test_parallel_settings_resolve_defaults_and_validate_positive_integers() -> None:
    assert ParallelSettings().queue_size == 1
    assert ParallelSettings(n_workers=4).queue_size == 4
    with pytest.raises(ConfigurationError, match="n_workers"):
        ParallelSettings(n_workers=0)
    with pytest.raises(ConfigurationError, match="queue_size"):
        ParallelSettings(queue_size=0)


def test_replacement_snapshot_owns_read_only_copies() -> None:
    theta = np.arange(6.0).reshape(3, 2)
    values = np.arange(3.0)
    snapshot = ReplacementSnapshot(
        threshold=0.0,
        threshold_tie_breaker=0.25,
        worst=0,
        live_theta=theta,
        live_log_likelihood=values,
        live_log_prior=values,
        live_log_q0=values,
        live_log_psi0=values,
        live_tie_breakers=values,
        proposal_revision=2,
    )
    theta[:] = -1.0
    values[:] = -1.0
    assert np.all(snapshot.live_theta >= 0.0)
    assert np.all(snapshot.live_log_psi0 >= 0.0)
    assert not snapshot.live_theta.flags.writeable
    assert not snapshot.live_log_psi0.flags.writeable


def test_replacement_queue_is_fifo_and_revalidates_threshold_and_revision() -> None:
    queue = ReplacementQueue()
    queue.extend([_result(3, log_psi0=1.0), _result(4, log_psi0=3.0)])
    assert queue.popleft().job_id == 3
    current = queue.popleft()
    valid, reason = queue.is_current_and_valid(
        current,
        threshold=2.0,
        threshold_tie_breaker=0.0,
        proposal_revision=0,
        tie_policy="strict",
    )
    assert valid
    assert reason is None

    stale = _result(5, log_psi0=1.0)
    assert queue.is_current_and_valid(
        stale,
        threshold=2.0,
        threshold_tie_breaker=0.0,
        proposal_revision=0,
        tie_policy="strict",
    ) == (False, "stale")
    old_revision = _result(6, log_psi0=4.0, revision=1)
    assert queue.is_current_and_valid(
        old_revision,
        threshold=2.0,
        threshold_tie_breaker=0.0,
        proposal_revision=2,
        tie_policy="strict",
    ) == (False, "proposal_revision")


def test_queue_diagnostics_report_exact_efficiencies() -> None:
    diagnostics = QueueDiagnostics(
        queue_jobs_submitted=5,
        queue_jobs_completed=5,
        queue_candidates_consumed=3,
        queue_candidates_stale=1,
        queue_candidates_invalidated=1,
        queue_refills=2,
        prefetch_likelihood_calls=50,
        used_prefetch_likelihood_calls=30,
        wasted_prefetch_likelihood_calls=20,
    )
    assert diagnostics.queue_efficiency == pytest.approx(0.6)
    assert diagnostics.compute_efficiency == pytest.approx(0.6)
