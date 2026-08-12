"""Evaluated points and unbiased constrained rejection sampling."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import EnsembleMoveName, TiePolicy
from .exceptions import (
    InvalidModelOutput,
    InvalidProposalOutput,
    ProposalSupportError,
)
from .model import Model, validate_points
from .proposals import Proposal


@dataclass(frozen=True, slots=True)
class EvaluatedPoint:
    """One candidate with quantities tied to the fixed importance Morph."""

    theta: NDArray[np.float64]
    log_likelihood: float
    log_prior: float
    log_q0: float
    log_psi0: float
    tie_breaker: float


@dataclass(frozen=True, slots=True)
class EvaluatedBatch:
    """A validated batch evaluated against the fixed importance Morph."""

    theta: NDArray[np.float64]
    log_likelihood: NDArray[np.float64]
    log_prior: NDArray[np.float64]
    log_q0: NDArray[np.float64]
    log_psi0: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ConstrainedDraw:
    """Successful independent constrained-proposal draw."""

    point: EvaluatedPoint
    n_proposed: int
    n_valid: int


@dataclass(frozen=True, slots=True)
class EnsembleMoveStats:
    """Counts for one ensemble move during one replacement.

    ``n_moved`` counts distinct walkers moved by this move. A walker can be
    counted for multiple moves if it accepts more than one move type.
    """

    name: EnsembleMoveName
    n_proposed: int
    n_valid: int
    n_accepted: int
    n_moved: int

    def __post_init__(self) -> None:
        if self.name not in ("de", "stretch", "gaussian"):
            raise ValueError(f"unsupported ensemble move name: {self.name!r}")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (
                self.n_proposed,
                self.n_valid,
                self.n_accepted,
                self.n_moved,
            )
        ):
            raise ValueError("ensemble move counts must be non-negative integers")
        if self.n_valid > self.n_proposed:
            raise ValueError("ensemble valid count cannot exceed proposed count")
        if self.n_accepted > self.n_valid:
            raise ValueError("ensemble accepted count cannot exceed valid count")
        if self.n_moved > self.n_accepted:
            raise ValueError("ensemble moved count cannot exceed accepted count")


@dataclass(frozen=True, slots=True)
class ConstrainedAttempt:
    """A successful draw or a typed resource-limit failure."""

    draw: ConstrainedDraw | None
    reason: str | None
    n_proposed: int
    n_valid: int
    n_accepted: int = 0
    n_moved: int = 0
    n_completed: int = 0
    ensemble_move_stats: tuple[EnsembleMoveStats, ...] = ()
    srwalk_factorization_seconds: float = 0.0
    srwalk_proposal_seconds: float = 0.0
    srwalk_squared_displacement: float = 0.0


class LikelihoodBudgetExhausted(RuntimeError):
    """Internal signal that a finite-prior point needs one unavailable call."""


def passes_constraint(
    log_psi0: NDArray[np.float64] | float,
    tie_breaker: NDArray[np.float64] | float,
    *,
    threshold: float,
    threshold_tie_breaker: float,
    tie_policy: TiePolicy,
) -> NDArray[np.bool_] | bool:
    """Apply the strict or lexicographic randomized-plateau ordering.

    Under ``randomized_plateau``, the tie breaker is part of the augmented
    state and equal-pseudo-likelihood points pass only when their tie breaker
    exceeds the discarded point's tie breaker.
    """
    values = np.asarray(log_psi0, dtype=float)
    ties = np.asarray(tie_breaker, dtype=float)
    if tie_policy == "strict":
        result = values > threshold
    elif tie_policy == "randomized_plateau":
        result = (values > threshold) | (
            (values == threshold) & (ties > threshold_tie_breaker)
        )
    else:  # pragma: no cover - public configuration validates this
        raise ValueError(f"unsupported tie_policy: {tie_policy!r}")
    if result.ndim == 0:
        return bool(result)
    return result


class BatchEvaluator:
    """Validate and count model/fixed-importance evaluations.

    ``n_likelihood_calls`` counts evaluated parameter points rather than Python
    function invocations, including vectorized batches.
    """

    def __init__(
        self,
        model: Model,
        importance_morph: Proposal,
        *,
        profile: bool = False,
    ) -> None:
        if model.ndim != importance_morph.ndim:
            raise ValueError(
                f"model ndim {model.ndim} does not match importance Morph ndim "
                f"{importance_morph.ndim}"
            )
        self.model = model
        self.importance_morph = importance_morph
        self.ndim = model.ndim
        self.n_likelihood_calls = 0
        self.n_prior_calls = 0
        self.outside_prior = 0
        self.zero_likelihood = 0
        self.profile = bool(profile)
        self.prior_seconds = 0.0
        self.likelihood_seconds = 0.0
        self.q0_seconds = 0.0

    def _log_prior(
        self,
        points: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        start = time.perf_counter() if self.profile else 0.0
        trusted = getattr(self.model, "_log_prior_validated", None)
        if callable(trusted):
            values = trusted(points)
        else:
            values = np.asarray(self.model.log_prior(points), dtype=float)
        if self.profile:
            self.prior_seconds += time.perf_counter() - start
        return np.asarray(values, dtype=float)

    def _log_likelihood(
        self,
        points: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        start = time.perf_counter() if self.profile else 0.0
        trusted = getattr(self.model, "_log_likelihood_validated", None)
        if callable(trusted):
            values = trusted(points)
        else:
            values = np.asarray(self.model.log_likelihood(points), dtype=float)
        if self.profile:
            self.likelihood_seconds += time.perf_counter() - start
        return np.asarray(values, dtype=float)

    def _log_q0(
        self,
        points: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        start = time.perf_counter() if self.profile else 0.0
        values = np.asarray(self.importance_morph.log_prob(points), dtype=float)
        if self.profile:
            self.q0_seconds += time.perf_counter() - start
        return values

    @staticmethod
    def _validate_values(
        name: str,
        values: NDArray[np.float64],
        *,
        expected: int,
        error_type: type[InvalidModelOutput] | type[InvalidProposalOutput],
    ) -> None:
        if values.shape != (expected,):
            raise error_type(
                f"{name} must return shape ({expected},), got {values.shape}"
            )
        if np.any(np.isnan(values)):
            raise error_type(f"{name} returned NaN")
        if np.any(np.isposinf(values)):
            raise error_type(f"{name} returned +infinity")

    def evaluate(self, theta: NDArray[np.float64]) -> EvaluatedBatch:
        """Evaluate and validate an ``(n, ndim)`` batch.

        ``-inf`` likelihood or prior values are valid zero-density values.
        A finite target numerator paired with ``log_q0 == -inf`` is a fatal
        support error.
        """
        points = validate_points(theta, self.ndim)
        n_points = len(points)
        log_prior = self._log_prior(points)
        self.n_prior_calls += n_points
        self._validate_values(
            "log_prior",
            log_prior,
            expected=n_points,
            error_type=InvalidModelOutput,
        )

        inside_prior = np.isfinite(log_prior)
        likelihood_indices = np.flatnonzero(inside_prior)
        log_likelihood = np.full(n_points, -np.inf, dtype=float)
        if len(likelihood_indices):
            evaluated_likelihood = self._log_likelihood(points[likelihood_indices])
            self.n_likelihood_calls += len(likelihood_indices)
            self._validate_values(
                "log_likelihood",
                evaluated_likelihood,
                expected=len(likelihood_indices),
                error_type=InvalidModelOutput,
            )
            log_likelihood[likelihood_indices] = evaluated_likelihood

        numerator = log_likelihood + log_prior
        finite_numerator = np.isfinite(numerator)
        q0_indices = np.flatnonzero(finite_numerator)
        log_q0 = np.full(n_points, -np.inf, dtype=float)
        if len(q0_indices):
            evaluated_q0 = self._log_q0(points[q0_indices])
            self._validate_values(
                "log_q0",
                evaluated_q0,
                expected=len(q0_indices),
                error_type=InvalidProposalOutput,
            )
            log_q0[q0_indices] = evaluated_q0
            support_failure = np.isneginf(evaluated_q0)
            if np.any(support_failure):
                first = int(q0_indices[np.flatnonzero(support_failure)[0]])
                raise ProposalSupportError(
                    "proposal support failure: finite log_likelihood + log_prior "
                    f"with log_q0 == -inf at batch row {first}"
                )

        log_psi0 = np.full(n_points, -np.inf, dtype=float)
        valid = finite_numerator & np.isfinite(log_q0)
        log_psi0[valid] = numerator[valid] - log_q0[valid]
        if np.any(np.isnan(log_psi0)) or np.any(np.isposinf(log_psi0)):
            raise InvalidModelOutput("log_psi0 is NaN or +infinity")

        self.outside_prior += int(np.count_nonzero(np.isneginf(log_prior)))
        self.zero_likelihood += int(
            np.count_nonzero(np.isneginf(log_likelihood[likelihood_indices]))
        )
        return EvaluatedBatch(
            theta=points,
            log_likelihood=log_likelihood,
            log_prior=log_prior,
            log_q0=log_q0,
            log_psi0=log_psi0,
        )

    def evaluate_one(
        self,
        theta: NDArray[np.float64],
        *,
        max_likelihood_calls: int | None = None,
    ) -> tuple[NDArray[np.float64], float, float, float, float]:
        """Evaluate one point, rejecting outside-prior points before likelihood."""
        points = validate_points(theta, self.ndim)
        if len(points) != 1:
            raise InvalidModelOutput("evaluate_one requires exactly one point")

        log_prior_values = self._log_prior(points)
        self.n_prior_calls += 1
        self._validate_values(
            "log_prior",
            log_prior_values,
            expected=1,
            error_type=InvalidModelOutput,
        )
        log_prior = float(log_prior_values[0])
        if np.isneginf(log_prior):
            self.outside_prior += 1
            return points[0], -np.inf, log_prior, -np.inf, -np.inf

        if (
            max_likelihood_calls is not None
            and self.n_likelihood_calls >= max_likelihood_calls
        ):
            raise LikelihoodBudgetExhausted
        log_likelihood_values = self._log_likelihood(points)
        self.n_likelihood_calls += 1
        self._validate_values(
            "log_likelihood",
            log_likelihood_values,
            expected=1,
            error_type=InvalidModelOutput,
        )
        log_likelihood = float(log_likelihood_values[0])
        if np.isneginf(log_likelihood):
            self.zero_likelihood += 1
            return points[0], log_likelihood, log_prior, -np.inf, -np.inf

        log_q0_values = self._log_q0(points)
        self._validate_values(
            "log_q0",
            log_q0_values,
            expected=1,
            error_type=InvalidProposalOutput,
        )
        log_q0 = float(log_q0_values[0])

        numerator = log_likelihood + log_prior
        if np.isfinite(numerator) and np.isneginf(log_q0):
            raise ProposalSupportError(
                "proposal support failure: finite log_likelihood + log_prior "
                "with log_q0 == -inf at batch row 0"
            )

        if np.isfinite(numerator) and np.isfinite(log_q0):
            log_psi0 = numerator - log_q0
        else:
            log_psi0 = -np.inf
        if np.isnan(log_psi0) or np.isposinf(log_psi0):
            raise InvalidModelOutput("log_psi0 is NaN or +infinity")

        return points[0], log_likelihood, log_prior, log_q0, log_psi0


def validate_proposal_sample(
    theta: NDArray[np.float64],
    *,
    n: int,
    ndim: int,
) -> NDArray[np.float64]:
    """Validate points returned by a proposal without reclassifying failures."""
    points = np.asarray(theta, dtype=float)
    if points.shape != (n, ndim):
        raise InvalidProposalOutput(
            f"proposal sample must have shape {(n, ndim)}, got {points.shape}"
        )
    if not np.all(np.isfinite(points)):
        raise InvalidProposalOutput("proposal sample contains NaN or infinity")
    return points


def draw_constrained(
    *,
    evaluator: BatchEvaluator,
    proposal_morph: Proposal,
    threshold: float,
    threshold_tie_breaker: float,
    tie_policy: TiePolicy,
    rng: np.random.Generator,
    batch_size: int,
    max_proposals: int,
    max_likelihood_calls: int | None,
    deadline: float | None,
) -> ConstrainedAttempt:
    """Draw from the active proposal under a fixed-``q0`` constraint.

    Independent proposals are scanned in generation order; the first valid
    proposal is accepted. No maximum-of-batch selection is performed.

    Returns
    -------
    ConstrainedAttempt
        A draw, or a failure reason with exact evaluated proposal counts.
    """
    n_proposed = 0
    n_valid = 0
    while n_proposed < max_proposals:
        if deadline is not None and time.monotonic() >= deadline:
            return ConstrainedAttempt(None, "max_wall_time", n_proposed, n_valid)
        remaining_global = max_proposals
        if max_likelihood_calls is not None:
            remaining_global = max_likelihood_calls - evaluator.n_likelihood_calls
            if remaining_global <= 0:
                return ConstrainedAttempt(
                    None, "max_likelihood_calls", n_proposed, n_valid
                )
        current_size = min(
            batch_size,
            max_proposals - n_proposed,
            remaining_global,
        )
        points = validate_proposal_sample(
            proposal_morph.sample(current_size, rng),
            n=current_size,
            ndim=evaluator.ndim,
        )
        batch = evaluator.evaluate(points)
        tie_breakers = rng.random(current_size)
        valid = np.asarray(
            passes_constraint(
                batch.log_psi0,
                tie_breakers,
                threshold=threshold,
                threshold_tie_breaker=threshold_tie_breaker,
                tie_policy=tie_policy,
            ),
            dtype=bool,
        )
        valid_indices = np.flatnonzero(valid)
        n_proposed += current_size
        n_valid += len(valid_indices)
        if len(valid_indices):
            index = int(valid_indices[0])
            point = EvaluatedPoint(
                theta=np.array(batch.theta[index], copy=True),
                log_likelihood=float(batch.log_likelihood[index]),
                log_prior=float(batch.log_prior[index]),
                log_q0=float(batch.log_q0[index]),
                log_psi0=float(batch.log_psi0[index]),
                tie_breaker=float(tie_breakers[index]),
            )
            return ConstrainedAttempt(
                ConstrainedDraw(point, n_proposed, n_valid),
                None,
                n_proposed,
                n_valid,
            )
    return ConstrainedAttempt(
        None,
        "constrained_sampling_exhausted",
        n_proposed,
        n_valid,
    )
