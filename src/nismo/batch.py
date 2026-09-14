"""Independent constrained chains, vectorized across walkers at each MH step."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import TiePolicy
from .constrained import (
    BatchEvaluator,
    ConstrainedAttempt,
    ConstrainedDraw,
    EvaluatedPoint,
    passes_constraint,
)


@dataclass(frozen=True, slots=True)
class BatchWalkResult:
    """Complete/failed attempts and exact per-chain point counts."""

    attempts: tuple[ConstrainedAttempt, ...]
    likelihood_calls: tuple[int, ...]
    outside_prior: tuple[int, ...]
    zero_likelihood: tuple[int, ...]


def evolve_srwalk_batch(
    *,
    evaluator: BatchEvaluator,
    starts: Sequence[EvaluatedPoint],
    threshold: float,
    threshold_tie_breaker: float,
    tie_policy: TiePolicy,
    proposal_factor: NDArray[np.float64],
    scale: float,
    n_steps: int,
    rngs: Sequence[np.random.Generator],
    max_proposals: int,
    max_likelihood_calls: int | None = None,
    deadline: float | None = None,
    zero_move_policy: str = "allow",
) -> BatchWalkResult:
    """Evolve fixed-length, frozen-geometry symmetric walks targeting q0**beta.

    Every chain owns its RNG, including rejected trials. No future transition
    is evaluated before its current state is known. Reserve all potential
    likelihood calls before starting; interrupted chains have no usable draw.
    One-chain and multi-chain layouts consume identical per-chain randomness.
    """
    count = len(starts)
    if count < 1 or len(rngs) != count or len({id(rng) for rng in rngs}) != count:
        raise ValueError("provide one distinct RNG for each nonempty chain")
    if n_steps < 1 or zero_move_policy not in ("allow", "stop"):
        raise ValueError("invalid batch walk settings")
    factor = np.asarray(proposal_factor, dtype=float)
    if factor.shape != (evaluator.ndim, evaluator.ndim) or not np.all(
        np.isfinite(factor)
    ):
        raise ValueError("batch factor must be finite with shape (ndim, ndim)")
    theta = np.array([p.theta for p in starts], dtype=float)
    if theta.shape != (count, evaluator.ndim) or not np.all(np.isfinite(theta)):
        raise ValueError("batch starts have invalid coordinates")
    current = np.array(
        [
            [p.log_likelihood, p.log_prior, p.log_q0, p.log_psi0, p.tie_breaker]
            for p in starts
        ]
    )
    if not np.all(np.isfinite(current[:, 2])):
        raise ValueError("batch starts require finite log_q0")
    if not np.all(
        passes_constraint(
            current[:, 3],
            current[:, 4],
            threshold=threshold,
            threshold_tie_breaker=threshold_tie_breaker,
            tie_policy=tie_policy,
        )
    ):
        raise ValueError("batch starts must satisfy the frozen constraint")
    valid_count = np.zeros(count, dtype=np.int64)
    accepted = np.zeros(count, dtype=np.int64)
    likelihood = np.zeros(count, dtype=np.int64)
    outside = np.zeros(count, dtype=np.int64)
    zero = np.zeros(count, dtype=np.int64)
    completed = 0
    reason = None
    proposal_seconds = 0.0
    if n_steps > max_proposals:
        reason = "max_proposals_per_replacement"
    elif (
        max_likelihood_calls is not None
        and evaluator.n_likelihood_calls + count * n_steps > max_likelihood_calls
    ):
        reason = "max_likelihood_calls"
    elif deadline is not None and time.monotonic() >= deadline:
        reason = "max_wall_time"
    if reason is None:
        start = time.perf_counter()
        # Same matrix operation and stream order as the scalar reference.
        increments = np.array(
            [
                scale * (rng.standard_normal(size=(n_steps, evaluator.ndim)) @ factor.T)
                for rng in rngs
            ]
        )
        proposal_seconds = time.perf_counter() - start
        for step in range(n_steps):
            if deadline is not None and time.monotonic() >= deadline:
                reason = "max_wall_time"
                break
            candidates = evaluator.evaluate(
                theta + increments[:, step], max_likelihood_calls=max_likelihood_calls
            )
            inside = np.isfinite(candidates.log_prior)
            likelihood += inside
            outside += ~inside
            zero += inside & np.isneginf(candidates.log_likelihood)
            ties = np.array([rng.random() for rng in rngs])
            valid = np.asarray(
                passes_constraint(
                    candidates.log_psi0,
                    ties,
                    threshold=threshold,
                    threshold_tie_breaker=threshold_tie_breaker,
                    tie_policy=tie_policy,
                )
            )
            valid_count += valid
            # Each chain draws its own MH uniform only when the constraint
            # passes, exactly as in the scalar reference. Masks cannot couple RNGs.
            move = np.zeros(count, dtype=bool)
            log_alpha = np.minimum(
                0.0, evaluator.beta * (candidates.log_q0[valid] - current[valid, 2])
            )
            uniforms = np.array([rngs[i].random() for i in np.flatnonzero(valid)])
            move[valid] = np.log(uniforms) < log_alpha
            theta[move] = candidates.theta[move]
            for column, values in enumerate(
                (
                    candidates.log_likelihood,
                    candidates.log_prior,
                    candidates.log_q0,
                    candidates.log_psi0,
                    ties,
                )
            ):
                current[move, column] = values[move]
            accepted += move
            completed += 1
        if deadline is not None and time.monotonic() >= deadline:
            reason = "max_wall_time"
    attempts = []
    for i in range(count):
        failure = reason
        if failure is None and accepted[i] == 0 and zero_move_policy == "stop":
            failure = "srwalk_stalled"
        point = EvaluatedPoint(
            np.array(theta[i], copy=True), *(float(value) for value in current[i])
        )
        draw = (
            None if failure else ConstrainedDraw(point, completed, int(valid_count[i]))
        )
        attempts.append(
            ConstrainedAttempt(
                draw,
                failure,
                completed,
                int(valid_count[i]),
                int(accepted[i]),
                int(accepted[i] > 0),
                completed,
                srwalk_proposal_seconds=proposal_seconds / count,
                srwalk_squared_displacement=float(
                    np.sum((theta[i] - starts[i].theta) ** 2)
                ),
            )
        )
    return BatchWalkResult(
        tuple(attempts),
        tuple(map(int, likelihood)),
        tuple(map(int, outside)),
        tuple(map(int, zero)),
    )
