"""Pure replacement construction and deterministic FIFO prefetch queues.

Workers in this module receive frozen sampler snapshots and return complete
replacement attempts.  They never own or mutate nested-sampling quadrature,
live-point, stopping, or history state.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from collections import deque
from dataclasses import dataclass, fields
from itertools import pairwise
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .batch import evolve_srwalk_batch
from .config import NISMOConfig, TiePolicy
from .constrained import (
    BatchEvaluator,
    ConstrainedAttempt,
    EvaluatedPoint,
    draw_constrained,
    passes_constraint,
)
from .mcmc import (
    draw_ensemble_rwalk_constrained,
    evolve_srwalk_constrained,
)
from .model import Model
from .proposals import Proposal


def _readonly_float(values: NDArray[np.float64]) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class ReplacementWorkerContext:
    """Heavy immutable objects cached once in every replacement worker."""

    config: NISMOConfig
    model: Model
    importance_morph: Proposal
    log_z_beta: float = 0.0


_WORKER_CONTEXT: ReplacementWorkerContext | None = None
_THREAD_LIMITER: object | None = None
_THREAD_COUNTS: tuple[int, ...] = ()


def initialize_replacement_worker(
    context: ReplacementWorkerContext, ready: Any = None
) -> None:
    """Install the run context in a spawned worker process.

    This mirrors Dynesty's pool initializer: model and fixed-importance
    objects cross the process boundary once, while map jobs contain only the
    state that changes from one proposed replacement to the next.
    """
    global _WORKER_CONTEXT, _THREAD_LIMITER, _THREAD_COUNTS
    _WORKER_CONTEXT = context
    if (
        context.config.parallel.worker_threads is not None
        and mp.current_process().name != "MainProcess"
    ):
        from threadpoolctl import threadpool_limits

        _THREAD_LIMITER = threadpool_limits(context.config.parallel.worker_threads)
    from threadpoolctl import threadpool_info

    _THREAD_COUNTS = tuple(int(item["num_threads"]) for item in threadpool_info())
    if ready is not None:
        ready.put(_THREAD_COUNTS)


def _replacement_worker_context() -> ReplacementWorkerContext:
    context = _WORKER_CONTEXT
    if context is None:
        raise RuntimeError("replacement worker context was not initialized")
    return context


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
    """Generic replacement state not already cached in the worker."""

    job_id: int
    snapshot: ReplacementSnapshot
    proposal_morph: Proposal
    rseed: np.random.SeedSequence | np.random.Generator
    max_likelihood_calls: int | None
    deadline: float | None
    log_z_beta: float | None = None
    created_iteration: int = 0


@dataclass(frozen=True, slots=True)
class SRWalkTask:
    """Dynesty-style frozen start and axes for one complete ``s-rwalk``."""

    job_id: int
    starting: EvaluatedPoint
    threshold: float
    threshold_tie_breaker: float
    proposal_revision: int
    proposal_factor: NDArray[np.float64]
    scale: float
    n_steps: int
    rseed: np.random.SeedSequence | np.random.Generator
    max_likelihood_calls: int | None
    deadline: float | None
    log_z_beta: float | None = None
    created_iteration: int = 0
    tuning_revision: int = 0

    def __post_init__(self) -> None:
        if self.proposal_revision < 0:
            raise ValueError("proposal_revision must be non-negative")
        if self.n_steps < 1:
            raise ValueError("s-rwalk task n_steps must be positive")
        factor = _readonly_float(self.proposal_factor)
        ndim = len(self.starting.theta)
        if factor.shape != (ndim, ndim):
            raise ValueError("s-rwalk task factor has the wrong shape")
        object.__setattr__(self, "proposal_factor", factor)


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
    created_iteration: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0
    execution_seconds: float = 0.0
    worker_pid: int = 0
    worker_threads: tuple[int, ...] = ()
    normalized_squared_displacement: float | None = None
    reference_revision: int = 0
    tuning_revision: int = 0


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
    queue_candidates_failed: int = 0
    queue_candidates_unused: int = 0
    total_candidate_age: int = 0
    max_candidate_age: int = 0

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
        self.queue_candidates_failed = 0
        self.queue_candidates_unused = 0
        self.total_candidate_age = 0
        self.max_candidate_age = 0

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
            queue_candidates_failed=self.queue_candidates_failed,
            queue_candidates_unused=self.queue_candidates_unused,
            total_candidate_age=self.total_candidate_age,
            max_candidate_age=self.max_candidate_age,
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
        reference_revision: int = 0,
    ) -> tuple[bool, str | None]:
        # The reference is fixed for an evidence run. Tuning revisions only
        # change future symmetric proposals and never invalidate old endpoints.
        if result.reference_revision != reference_revision:
            return False, "reference_revision"
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
    started = time.monotonic()
    context = _replacement_worker_context()
    rng = np.random.default_rng(job.rseed)
    evaluator = BatchEvaluator(
        context.model,
        context.importance_morph,
        beta=context.config.beta,
        log_z_beta=context.log_z_beta if job.log_z_beta is None else job.log_z_beta,
        profile=(
            context.config.proposal_scheme in ("s-rwalk", "mor-rwalk")
            and context.config.srwalk_settings.profile
        ),
    )
    snapshot = job.snapshot
    config = context.config
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
        created_iteration=job.created_iteration,
        started_at=started,
        finished_at=time.monotonic(),
        execution_seconds=time.monotonic() - started,
        worker_pid=os.getpid(),
    )


def build_srwalk_replacement(task: SRWalkTask) -> ReplacementResult:
    """Evolve one coordinator-prepared ``s-rwalk`` start in a worker."""
    started = time.monotonic()
    context = _replacement_worker_context()
    config = context.config
    rng = np.random.default_rng(task.rseed)
    evaluator = BatchEvaluator(
        context.model,
        context.importance_morph,
        beta=config.beta,
        log_z_beta=context.log_z_beta if task.log_z_beta is None else task.log_z_beta,
        profile=config.srwalk_settings.profile,
    )
    attempt = evolve_srwalk_constrained(
        evaluator=evaluator,
        starting=task.starting,
        threshold=task.threshold,
        threshold_tie_breaker=task.threshold_tie_breaker,
        tie_policy=config.tie_policy,
        proposal_factor=task.proposal_factor,
        scale=task.scale,
        n_steps=task.n_steps,
        zero_move_policy=config.srwalk_settings.zero_move_policy,
        max_proposals=config.max_proposals_per_replacement,
        max_likelihood_calls=task.max_likelihood_calls,
        deadline=task.deadline,
        rng=rng,
    )
    return ReplacementResult(
        job_id=task.job_id,
        attempt=attempt,
        threshold_at_creation=task.threshold,
        threshold_tie_breaker_at_creation=task.threshold_tie_breaker,
        proposal_revision=task.proposal_revision,
        counts=EvaluationCounts.from_evaluator(evaluator),
        proposal_scale=task.scale,
        created_iteration=task.created_iteration,
        started_at=started,
        finished_at=time.monotonic(),
        execution_seconds=time.monotonic() - started,
        worker_pid=os.getpid(),
        worker_threads=_worker_thread_counts()
        if config.srwalk_settings.profile
        else (),
        normalized_squared_displacement=_normalized_displacement(task, attempt),
        tuning_revision=task.tuning_revision,
    )


def _worker_thread_counts() -> tuple[int, ...]:
    return _THREAD_COUNTS


def _normalized_displacement(task: SRWalkTask, attempt: ConstrainedAttempt) -> float:
    if (
        attempt.draw is None
        or not _replacement_worker_context().config.srwalk_settings.profile
    ):
        return 0.0
    delta = attempt.draw.point.theta - task.starting.theta
    whitened = np.linalg.solve(task.proposal_factor, delta)
    return float(np.sum(whitened**2))


def build_srwalk_batch(tasks: tuple[SRWalkTask, ...]) -> list[ReplacementResult]:
    """Execute a small batch of complete independent chains in one worker."""
    started = time.monotonic()
    context = _replacement_worker_context()
    config = context.config
    first = tasks[0]
    if any(
        (
            task.threshold,
            task.threshold_tie_breaker,
            task.n_steps,
            task.scale,
            task.log_z_beta,
            task.deadline,
        )
        != (
            first.threshold,
            first.threshold_tie_breaker,
            first.n_steps,
            first.scale,
            first.log_z_beta,
            first.deadline,
        )
        or not np.array_equal(task.proposal_factor, first.proposal_factor)
        for task in tasks
    ):
        raise ValueError("batched tasks must share frozen threshold and tuning")
    evaluator = BatchEvaluator(
        context.model,
        context.importance_morph,
        beta=config.beta,
        log_z_beta=context.log_z_beta if first.log_z_beta is None else first.log_z_beta,
        profile=config.srwalk_settings.profile,
    )
    budgets = [task.max_likelihood_calls for task in tasks]
    if any(value is not None and value < first.n_steps for value in budgets):
        raise ValueError("batched tasks require full-chain call reservations")
    budget = (
        None
        if any(value is None for value in budgets)
        else sum(value for value in budgets if value is not None)
    )
    walked = evolve_srwalk_batch(
        evaluator=evaluator,
        starts=[task.starting for task in tasks],
        threshold=first.threshold,
        threshold_tie_breaker=first.threshold_tie_breaker,
        tie_policy=config.tie_policy,
        proposal_factor=first.proposal_factor,
        scale=first.scale,
        n_steps=first.n_steps,
        rngs=[np.random.default_rng(task.rseed) for task in tasks],
        max_proposals=config.max_proposals_per_replacement,
        max_likelihood_calls=budget,
        deadline=first.deadline,
        zero_move_policy=config.srwalk_settings.zero_move_policy,
    )
    finished = time.monotonic()
    threads = _worker_thread_counts() if config.srwalk_settings.profile else ()
    results = []
    for i, (task, attempt) in enumerate(zip(tasks, walked.attempts, strict=True)):
        results.append(
            ReplacementResult(
                job_id=task.job_id,
                attempt=attempt,
                threshold_at_creation=task.threshold,
                threshold_tie_breaker_at_creation=task.threshold_tie_breaker,
                proposal_revision=task.proposal_revision,
                proposal_scale=task.scale,
                counts=EvaluationCounts(
                    likelihood_calls=walked.likelihood_calls[i],
                    prior_calls=attempt.n_proposed,
                    outside_prior=walked.outside_prior[i],
                    zero_likelihood=walked.zero_likelihood[i],
                    likelihood_seconds=evaluator.likelihood_seconds / len(tasks),
                    prior_seconds=evaluator.prior_seconds / len(tasks),
                    q0_seconds=evaluator.q0_seconds / len(tasks),
                ),
                created_iteration=task.created_iteration,
                started_at=started,
                finished_at=finished,
                execution_seconds=(finished - started) / len(tasks),
                worker_pid=os.getpid(),
                worker_threads=threads,
                normalized_squared_displacement=_normalized_displacement(task, attempt),
                tuning_revision=task.tuning_revision,
            )
        )
    return results
