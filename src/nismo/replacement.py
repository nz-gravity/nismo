"""Pure replacement construction and deterministic FIFO prefetch queues.

Workers in this module receive frozen sampler snapshots and return complete
replacement attempts.  They never own or mutate nested-sampling quadrature,
live-point, stopping, or history state.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, fields
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

from .config import NISMOConfig, TiePolicy
from .constrained import (
    BatchEvaluator,
    ConstrainedAttempt,
    draw_constrained,
    passes_constraint,
)
from .mcmc import (
    RWalkSampler,
    SRWalkSampler,
    draw_ensemble_rwalk_constrained,
    draw_rwalk_constrained,
    draw_srwalk_constrained,
)
from .model import Model
from .proposals import Proposal


def _readonly_float(values: NDArray[np.float64]) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class ReplacementSnapshot:
    """Immutable live-set and proposal-epoch state supplied to one job."""

    threshold: float
    threshold_tie_breaker: float
    worst: int
    live_theta: NDArray[np.float64]
    live_log_likelihood: NDArray[np.float64]
    live_log_prior: NDArray[np.float64]
    live_log_q0: NDArray[np.float64]
    live_log_psi0: NDArray[np.float64]
    live_tie_breakers: NDArray[np.float64]
    proposal_revision: int
    srwalk_factor: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        theta = _readonly_float(self.live_theta)
        if theta.ndim != 2:
            raise ValueError("live_theta must be a two-dimensional array")
        n_live = len(theta)
        if not 0 <= self.worst < n_live:
            raise ValueError("worst live-point index is out of bounds")
        object.__setattr__(self, "live_theta", theta)
        for name in (
            "live_log_likelihood",
            "live_log_prior",
            "live_log_q0",
            "live_log_psi0",
            "live_tie_breakers",
        ):
            array = _readonly_float(getattr(self, name))
            if array.shape != (n_live,):
                raise ValueError(f"{name} must have shape ({n_live},)")
            object.__setattr__(self, name, array)
        if self.proposal_revision < 0:
            raise ValueError("proposal_revision must be non-negative")
        if self.srwalk_factor is not None:
            factor = _readonly_float(self.srwalk_factor)
            ndim = theta.shape[1]
            if factor.shape != (ndim, ndim):
                raise ValueError("srwalk_factor must have shape (ndim, ndim)")
            object.__setattr__(self, "srwalk_factor", factor)


@dataclass(frozen=True, slots=True)
class EvaluationCounts:
    """Worker-local evaluation counts returned to the coordinator."""

    likelihood_calls: int = 0
    prior_calls: int = 0
    outside_prior: int = 0
    zero_likelihood: int = 0
    likelihood_seconds: float = 0.0
    prior_seconds: float = 0.0
    q0_seconds: float = 0.0

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name.endswith("_seconds"):
                if not np.isfinite(value) or value < 0.0:
                    raise ValueError(
                        "evaluation timings must be finite and non-negative"
                    )
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("evaluation counts must be non-negative integers")

    @classmethod
    def from_evaluator(cls, evaluator: BatchEvaluator) -> EvaluationCounts:
        return cls(
            likelihood_calls=evaluator.n_likelihood_calls,
            prior_calls=evaluator.n_prior_calls,
            outside_prior=evaluator.outside_prior,
            zero_likelihood=evaluator.zero_likelihood,
            likelihood_seconds=evaluator.likelihood_seconds,
            prior_seconds=evaluator.prior_seconds,
            q0_seconds=evaluator.q0_seconds,
        )


@dataclass(frozen=True, slots=True)
class ReplacementJob:
    """Pickle-safe description of one complete replacement construction."""

    job_id: int
    snapshot: ReplacementSnapshot
    config: NISMOConfig
    model: Model
    importance_morph: Proposal
    proposal_morph: Proposal
    seed_entropy: tuple[int, ...]
    max_likelihood_calls: int | None
    deadline: float | None
    rwalk_scale: float | None = None
    srwalk_scale: float | None = None


@dataclass(frozen=True, slots=True)
class ReplacementResult:
    """Complete worker result, including all accounting and epoch metadata."""

    job_id: int
    attempt: ConstrainedAttempt
    threshold_at_creation: float
    threshold_tie_breaker_at_creation: float
    proposal_revision: int
    counts: EvaluationCounts
    proposal_scale: float | None = None


@dataclass(frozen=True, slots=True)
class QueueDiagnostics:
    """Immutable audit record for replacement-prefetch work."""

    queue_jobs_submitted: int = 0
    queue_jobs_completed: int = 0
    queue_candidates_consumed: int = 0
    queue_candidates_stale: int = 0
    queue_candidates_invalidated: int = 0
    queue_refills: int = 0
    prefetch_likelihood_calls: int = 0
    used_prefetch_likelihood_calls: int = 0
    wasted_prefetch_likelihood_calls: int = 0

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("queue diagnostics must be non-negative integers")
        if self.queue_jobs_completed > self.queue_jobs_submitted:
            raise ValueError("completed queue jobs cannot exceed submitted jobs")
        if self.used_prefetch_likelihood_calls > self.prefetch_likelihood_calls:
            raise ValueError("used prefetch calls cannot exceed all prefetch calls")
        classified = (
            self.queue_candidates_consumed
            + self.queue_candidates_stale
            + self.queue_candidates_invalidated
        )
        if classified > self.queue_jobs_completed:
            raise ValueError("classified candidates cannot exceed completed jobs")
        if (
            self.wasted_prefetch_likelihood_calls
            != self.prefetch_likelihood_calls - self.used_prefetch_likelihood_calls
        ):
            raise ValueError("wasted prefetch calls must equal total minus used")

    @property
    def queue_efficiency(self) -> float:
        if not self.queue_jobs_completed:
            return 0.0
        return self.queue_candidates_consumed / self.queue_jobs_completed

    @property
    def compute_efficiency(self) -> float:
        if not self.prefetch_likelihood_calls:
            return 0.0
        return self.used_prefetch_likelihood_calls / self.prefetch_likelihood_calls


class QueueAccounting:
    """Mutable coordinator-only accumulator frozen into ``QueueDiagnostics``."""

    def __init__(self) -> None:
        self.queue_jobs_submitted = 0
        self.queue_jobs_completed = 0
        self.queue_candidates_consumed = 0
        self.queue_candidates_stale = 0
        self.queue_candidates_invalidated = 0
        self.queue_refills = 0
        self.prefetch_likelihood_calls = 0
        self.used_prefetch_likelihood_calls = 0

    def freeze(self) -> QueueDiagnostics:
        return QueueDiagnostics(
            queue_jobs_submitted=self.queue_jobs_submitted,
            queue_jobs_completed=self.queue_jobs_completed,
            queue_candidates_consumed=self.queue_candidates_consumed,
            queue_candidates_stale=self.queue_candidates_stale,
            queue_candidates_invalidated=self.queue_candidates_invalidated,
            queue_refills=self.queue_refills,
            prefetch_likelihood_calls=self.prefetch_likelihood_calls,
            used_prefetch_likelihood_calls=self.used_prefetch_likelihood_calls,
            wasted_prefetch_likelihood_calls=(
                self.prefetch_likelihood_calls - self.used_prefetch_likelihood_calls
            ),
        )


class ReplacementQueue:
    """Deterministic FIFO queue with coordinator-side candidate validation."""

    def __init__(self) -> None:
        self._results: deque[ReplacementResult] = deque()

    def __len__(self) -> int:
        return len(self._results)

    def extend(self, results: list[ReplacementResult]) -> None:
        if self._results and results and results[0].job_id <= self._results[-1].job_id:
            raise ValueError("replacement job IDs must increase across refills")
        if any(right.job_id <= left.job_id for left, right in pairwise(results)):
            raise ValueError("replacement results must use increasing job IDs")
        self._results.extend(results)

    def popleft(self) -> ReplacementResult:
        return self._results.popleft()

    def clear(self) -> tuple[ReplacementResult, ...]:
        discarded = tuple(self._results)
        self._results.clear()
        return discarded

    @staticmethod
    def is_current_and_valid(
        result: ReplacementResult,
        *,
        threshold: float,
        threshold_tie_breaker: float,
        proposal_revision: int,
        tie_policy: TiePolicy,
    ) -> tuple[bool, str | None]:
        if result.proposal_revision != proposal_revision:
            return False, "proposal_revision"
        draw = result.attempt.draw
        if draw is None:
            return False, "failed"
        if not passes_constraint(
            draw.point.log_psi0,
            draw.point.tie_breaker,
            threshold=threshold,
            threshold_tie_breaker=threshold_tie_breaker,
            tie_policy=tie_policy,
        ):
            return False, "stale"
        return True, None


def prepare_replacement_snapshot(
    *,
    threshold: float,
    threshold_tie_breaker: float,
    worst: int,
    live_theta: NDArray[np.float64],
    live_log_likelihood: NDArray[np.float64],
    live_log_prior: NDArray[np.float64],
    live_log_q0: NDArray[np.float64],
    live_log_psi0: NDArray[np.float64],
    live_tie_breakers: NDArray[np.float64],
    proposal_revision: int,
    srwalk_factor: NDArray[np.float64] | None = None,
) -> ReplacementSnapshot:
    """Copy authoritative coordinator state into a read-only worker snapshot."""
    return ReplacementSnapshot(
        threshold=threshold,
        threshold_tie_breaker=threshold_tie_breaker,
        worst=worst,
        live_theta=live_theta,
        live_log_likelihood=live_log_likelihood,
        live_log_prior=live_log_prior,
        live_log_q0=live_log_q0,
        live_log_psi0=live_log_psi0,
        live_tie_breakers=live_tie_breakers,
        proposal_revision=proposal_revision,
        srwalk_factor=srwalk_factor,
    )


def build_replacement(job: ReplacementJob) -> ReplacementResult:
    """Construct one complete replacement without mutating coordinator state."""
    rng = np.random.default_rng(np.random.SeedSequence(job.seed_entropy))
    evaluator = BatchEvaluator(
        job.model,
        job.importance_morph,
        profile=(
            job.config.proposal_scheme == "s-rwalk"
            and job.config.srwalk_settings.profile
        ),
    )
    snapshot = job.snapshot
    config = job.config
    proposal_scale: float | None = None

    if config.proposal_scheme in ("fixed_morph", "adaptive_morph"):
        attempt = draw_constrained(
            evaluator=evaluator,
            proposal_morph=job.proposal_morph,
            threshold=snapshot.threshold,
            threshold_tie_breaker=snapshot.threshold_tie_breaker,
            tie_policy=config.tie_policy,
            rng=rng,
            batch_size=config.proposal_batch_size,
            max_proposals=config.max_proposals_per_replacement,
            max_likelihood_calls=job.max_likelihood_calls,
            deadline=job.deadline,
        )
    elif config.proposal_scheme == "rwalk":
        rwalk_controller = RWalkSampler(
            settings=config.rwalk_settings,
            ndim=job.model.ndim,
        )
        if job.rwalk_scale is not None:
            rwalk_controller.scale = job.rwalk_scale
        proposal_scale = rwalk_controller.scale
        attempt = draw_rwalk_constrained(
            evaluator=evaluator,
            live_theta=snapshot.live_theta,
            live_log_likelihood=snapshot.live_log_likelihood,
            live_log_prior=snapshot.live_log_prior,
            live_log_q0=snapshot.live_log_q0,
            live_log_psi0=snapshot.live_log_psi0,
            live_tie_breakers=snapshot.live_tie_breakers,
            worst=snapshot.worst,
            threshold=snapshot.threshold,
            threshold_tie_breaker=snapshot.threshold_tie_breaker,
            tie_policy=config.tie_policy,
            sampler=rwalk_controller,
            rng=rng,
            max_proposals=config.max_proposals_per_replacement,
            max_likelihood_calls=job.max_likelihood_calls,
            deadline=job.deadline,
        )
    elif config.proposal_scheme == "s-rwalk":
        srwalk_controller = SRWalkSampler(
            settings=config.srwalk_settings,
            ndim=job.model.ndim,
        )
        if job.srwalk_scale is not None:
            srwalk_controller.scale = job.srwalk_scale
        proposal_scale = srwalk_controller.scale
        attempt = draw_srwalk_constrained(
            evaluator=evaluator,
            live_theta=snapshot.live_theta,
            live_log_likelihood=snapshot.live_log_likelihood,
            live_log_prior=snapshot.live_log_prior,
            live_log_q0=snapshot.live_log_q0,
            live_log_psi0=snapshot.live_log_psi0,
            live_tie_breakers=snapshot.live_tie_breakers,
            worst=snapshot.worst,
            threshold=snapshot.threshold,
            threshold_tie_breaker=snapshot.threshold_tie_breaker,
            tie_policy=config.tie_policy,
            sampler=srwalk_controller,
            rng=rng,
            max_proposals=config.max_proposals_per_replacement,
            max_likelihood_calls=job.max_likelihood_calls,
            deadline=job.deadline,
            proposal_factor=snapshot.srwalk_factor,
        )
    elif config.proposal_scheme == "en-rwalk":
        attempt = draw_ensemble_rwalk_constrained(
            evaluator=evaluator,
            live_theta=snapshot.live_theta,
            live_log_likelihood=snapshot.live_log_likelihood,
            live_log_prior=snapshot.live_log_prior,
            live_log_q0=snapshot.live_log_q0,
            live_log_psi0=snapshot.live_log_psi0,
            live_tie_breakers=snapshot.live_tie_breakers,
            worst=snapshot.worst,
            threshold=snapshot.threshold,
            threshold_tie_breaker=snapshot.threshold_tie_breaker,
            tie_policy=config.tie_policy,
            settings=config.ensemble_rwalk_settings,
            rng=rng,
            max_proposals=config.max_proposals_per_replacement,
            max_likelihood_calls=job.max_likelihood_calls,
            deadline=job.deadline,
        )
    else:  # pragma: no cover - NISMOConfig validates proposal schemes
        raise RuntimeError(f"unsupported proposal scheme: {config.proposal_scheme!r}")

    return ReplacementResult(
        job_id=job.job_id,
        attempt=attempt,
        threshold_at_creation=snapshot.threshold,
        threshold_tie_breaker_at_creation=snapshot.threshold_tie_breaker,
        proposal_revision=snapshot.proposal_revision,
        counts=EvaluationCounts.from_evaluator(evaluator),
        proposal_scale=proposal_scale,
    )
