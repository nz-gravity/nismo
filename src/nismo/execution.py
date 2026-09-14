"""Owned evaluation pool and bounded, submission-ordered task retrieval.

No live points or quadrature live here. Reservations are in parameter points;
results are accounted by the coordinator even when shutdown leaves them unused.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray

from .constrained import BatchEvaluator, EvaluatedBatch
from .replacement import (
    EvaluationCounts,
    ReplacementResult,
    ReplacementWorkerContext,
    _replacement_worker_context,
    _worker_thread_counts,
    initialize_replacement_worker,
)

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True, slots=True)
class TaskTiming:
    job_id: int
    created_iteration: int
    retrieved_iteration: int
    started_at: float
    finished_at: float
    execution_seconds: float
    worker_pid: int
    worker_threads: tuple[int, ...]
    normalized_squared_displacement: float | None
    tuning_revision: int


@dataclass(frozen=True, slots=True)
class ExecutionDiagnostics:
    """Elapsed phases and separate summed worker execution (never additive)."""

    phase_seconds: tuple[tuple[str, float], ...] = ()
    worker_execution_seconds: float = 0.0
    peak_outstanding_chains: int = 0
    task_timings: tuple[TaskTiming, ...] = ()
    worker_thread_counts: tuple[tuple[int, ...], ...] = ()


@dataclass
class ExecutionService:
    """One process pool for initialization, beta densities and complete walks."""

    pool: Any = None
    phases: dict[str, float] = field(default_factory=dict)
    context: ReplacementWorkerContext | None = None
    worker_seconds: float = 0.0
    peak_outstanding: int = 0
    timings: list[TaskTiming] = field(default_factory=list)
    thread_counts: list[tuple[int, ...]] = field(default_factory=list)

    def start(self, context: ReplacementWorkerContext) -> None:
        started = time.monotonic()
        self.context = context
        parallel = context.config.parallel
        initialize_replacement_worker(context)
        if parallel.backend != "vectorized" and parallel.n_workers > 1:
            spawn_context = mp.get_context("spawn")
            ready = spawn_context.Queue()
            try:
                self.pool = spawn_context.Pool(
                    processes=parallel.n_workers,
                    initializer=initialize_replacement_worker,
                    initargs=(context, ready),
                )
                for _ in range(parallel.n_workers):
                    self.thread_counts.append(ready.get(timeout=30.0))
            finally:
                ready.close()
                ready.join_thread()
        else:
            self.thread_counts.append(_worker_thread_counts())
        self.add_time("pool_startup", time.monotonic() - started)

    def add_time(self, phase: str, seconds: float) -> None:
        self.phases[phase] = self.phases.get(phase, 0.0) + seconds

    def close(self, *, terminate: bool = False) -> None:
        if self.pool is None:
            return
        started = time.monotonic()
        try:
            if terminate:
                self.pool.terminate()
            else:
                self.pool.close()
            self.pool.join()
        finally:
            self.pool = None
            self.add_time("pool_shutdown", time.monotonic() - started)

    def __enter__(self) -> ExecutionService:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close(terminate=exc_type is not None)

    def map_chunks(
        self, function: Callable[[T], R], chunks: Iterator[T], *, processes: bool
    ) -> Iterator[R]:
        """Bound serialized initialization data to at most one chunk per worker."""
        if not processes or self.pool is None:
            yield from map(function, chunks)
            return
        pending: deque[Any] = deque()
        workers = self.context.config.parallel.n_workers if self.context else 1
        for _ in range(workers):
            try:
                pending.append(self.pool.apply_async(function, (next(chunks),)))
            except StopIteration:
                break
        while pending:
            yield pending.popleft().get()
            with suppress(StopIteration):
                pending.append(self.pool.apply_async(function, (next(chunks),)))

    def density(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.context is None:
            raise RuntimeError("evaluation service has not started")
        p = self.context.config.parallel
        chunks = (
            points[i : i + p.evaluation_chunk_size]
            for i in range(0, len(points), p.evaluation_chunk_size)
        )
        return np.concatenate(
            list(
                self.map_chunks(
                    evaluate_density_chunk,
                    chunks,
                    processes=p.backend == "process" and p.initialization != "serial",
                )
            )
        )

    def evaluate(
        self,
        evaluator: BatchEvaluator,
        points: NDArray[np.float64],
        cached_log_q0: NDArray[np.float64] | None = None,
    ) -> EvaluatedBatch:
        if self.context is None:
            raise RuntimeError("evaluation service has not started")
        p = self.context.config.parallel
        processes = p.initialization == "process" or (
            p.initialization == "auto"
            and p.backend == "process"
            and getattr(self.context.model, "vectorized", True) is False
        )
        if p.backend == "compatibility" and p.initialization == "auto":
            return evaluator.evaluate(points, cached_log_q0=cached_log_q0)
        chunks = (
            (
                points[i : i + p.evaluation_chunk_size],
                evaluator.log_z_beta,
                None
                if cached_log_q0 is None
                else cached_log_q0[i : i + p.evaluation_chunk_size],
            )
            for i in range(0, len(points), p.evaluation_chunk_size)
        )
        outputs = list(
            self.map_chunks(evaluate_initial_chunk, chunks, processes=processes)
        )
        for _, counts in outputs:
            evaluator.n_likelihood_calls += counts.likelihood_calls
            evaluator.n_prior_calls += counts.prior_calls
            evaluator.outside_prior += counts.outside_prior
            evaluator.zero_likelihood += counts.zero_likelihood
            evaluator.prior_seconds += counts.prior_seconds
            evaluator.likelihood_seconds += counts.likelihood_seconds
            evaluator.q0_seconds += counts.q0_seconds
        return EvaluatedBatch(
            *(
                np.concatenate([getattr(batch, name) for batch, _ in outputs])
                for name in (
                    "theta",
                    "log_likelihood",
                    "log_prior",
                    "log_q0",
                    "log_psi0",
                )
            )
        )

    def record(self, result: ReplacementResult, niter: int) -> None:
        self.worker_seconds += result.execution_seconds
        if self.context is not None and self.context.config.srwalk_settings.profile:
            self.timings.append(
                TaskTiming(
                    result.job_id,
                    result.created_iteration,
                    niter,
                    result.started_at,
                    result.finished_at,
                    result.execution_seconds,
                    result.worker_pid,
                    result.worker_threads,
                    result.normalized_squared_displacement,
                    result.tuning_revision,
                )
            )

    def freeze(self) -> ExecutionDiagnostics:
        return ExecutionDiagnostics(
            tuple(sorted(self.phases.items())),
            self.worker_seconds,
            self.peak_outstanding,
            tuple(self.timings),
            tuple(self.thread_counts),
        )


def evaluate_density_chunk(points: NDArray[np.float64]) -> NDArray[np.float64]:
    context = _replacement_worker_context()
    return np.asarray(context.importance_morph.log_prob(points), dtype=float)


def evaluate_initial_chunk(
    job: tuple[NDArray[np.float64], float, NDArray[np.float64] | None],
) -> tuple[EvaluatedBatch, EvaluationCounts]:
    context = _replacement_worker_context()
    points, log_z_beta, cached = job
    evaluator = BatchEvaluator(
        context.model,
        context.importance_morph,
        beta=context.config.beta,
        log_z_beta=log_z_beta,
        profile=context.config.srwalk_settings.profile,
    )
    batch = evaluator.evaluate(points, cached_log_q0=cached)
    return batch, EvaluationCounts.from_evaluator(evaluator)


class OrderedTasks:
    """Bounded running plus buffered chains, retired strictly by job ID.

    Submission happens only at deterministic coordinator retirement events;
    neither completion order nor readiness chooses the next candidate.
    """

    def __init__(self, service: ExecutionService, capacity: int) -> None:
        self.service = service
        self.capacity = capacity
        self.pending: deque[tuple[Any, int, int]] = deque()
        self.buffer: deque[ReplacementResult] = deque()
        self.outstanding = 0
        self.reserved_calls = 0

    def __len__(self) -> int:
        return self.outstanding

    def submit(
        self, function: Callable[[Any], Any], job: Any, *, n_chains: int, calls: int
    ) -> None:
        if self.outstanding + n_chains > self.capacity:
            raise RuntimeError("speculative chain capacity exceeded")
        if self.service.pool is None:
            value = function(job)
            handle = [value] if isinstance(value, ReplacementResult) else value
        else:
            handle = self.service.pool.apply_async(function, (job,))
        self.pending.append((handle, n_chains, calls))
        self.outstanding += n_chains
        self.reserved_calls += calls
        self.service.peak_outstanding = max(
            self.service.peak_outstanding, self.outstanding
        )

    def receive(self) -> list[ReplacementResult]:
        """Retrieve the oldest task; caller accounts every returned chain now."""
        started = time.monotonic()
        handle, n_chains, calls = self.pending.popleft()
        if self.service.pool is None:
            results = handle
        else:
            value = handle.get()
            results = [value] if isinstance(value, ReplacementResult) else value
        self.service.add_time("ordered_wait", time.monotonic() - started)
        if len(results) != n_chains:
            raise RuntimeError("worker returned an incomplete task result")
        self.reserved_calls -= calls
        self.buffer.extend(results)
        return list(results)

    def pop(self) -> ReplacementResult:
        self.outstanding -= 1
        return self.buffer.popleft()
