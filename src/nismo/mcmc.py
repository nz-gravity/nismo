"""Constrained Metropolis replacement kernels targeting the fixed density.

The standard random-walk controller, uniform-ball proposal, and single
bounding-ellipsoid construction are adapted from dynesty 3.1.0. NISMO applies
an additional fixed-``q0`` Metropolis correction because it operates directly
in parameter space rather than through dynesty's unit-prior transform. The
separate ``s-rwalk`` kernel uses a frozen regularized survivor covariance and
Gaussian proposals. The split ``en-rwalk`` kernel selects fixed-weight DE,
stretch, or frozen-covariance Gaussian half-ensemble moves.
"""

from __future__ import annotations

import math
import time
import warnings
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as scipy_linalg

from .config import (
    EnsembleMoveName,
    EnsembleMoveWeights,
    EnsembleRWalkSettings,
    RWalkSettings,
    SRWalkSettings,
    TiePolicy,
)
from .constrained import (
    BatchEvaluator,
    ConstrainedAttempt,
    ConstrainedDraw,
    EnsembleMoveStats,
    EvaluatedBatch,
    EvaluatedPoint,
    LikelihoodBudgetExhausted,
    passes_constraint,
)
from .exceptions import ConfigurationError, NumericalInvariantError

RWALK_CITATIONS = [
    ("Skilling (2006)", "projecteuclid.org/euclid.ba/1340370944"),
]


class RWalkSampler:
    """Stateful Dynesty-style controller for NISMO's standard random walk."""

    def __init__(self, *, settings: RWalkSettings, ndim: int) -> None:
        walks = ndim + 20 if settings.walks is None else settings.walks
        self.walks = max(2, walks)
        self.facc = min(1.0, max(1.0 / self.walks, settings.facc))
        self.ncdim = ndim if settings.ncdim is None else settings.ncdim
        if self.ncdim != ndim:
            raise ConfigurationError(
                "rwalk ncdim must equal the model dimension for NISMO proposals"
            )
        self.scale = 1.0
        self.rwalk_history = {"n_accept": 0, "n_reject": 0}
        self._axes: NDArray[np.float64] | None = None
        self._calls_since_bound_update = 0

    def tune(
        self,
        tuning_info: dict[str, float | int],
        update: bool = True,
    ) -> None:
        """Update proposal scale using dynesty's acceptance-rate recursion."""
        self.scale = float(tuning_info["scale"])
        self.rwalk_history["n_accept"] += int(tuning_info["accept"])
        self.rwalk_history["n_reject"] += int(tuning_info["reject"])
        if not update:
            return
        accept = self.rwalk_history["n_accept"]
        reject = self.rwalk_history["n_reject"]
        acceptance_fraction = accept / (accept + reject)
        self.scale *= math.exp(
            (acceptance_fraction - self.facc) / self.ncdim / self.facc
        )
        self.rwalk_history["n_accept"] = 0
        self.rwalk_history["n_reject"] = 0

    @property
    def update_bound_interval_ratio(self) -> int:
        """Return Dynesty's bound-update interval in calls per live point."""
        return self.walks

    def axes_for(
        self,
        live_theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return cached axes, rebuilding at Dynesty's standard cadence."""
        update_calls = self.update_bound_interval_ratio * len(live_theta)
        if self._axes is None or self._calls_since_bound_update >= update_calls:
            self._axes = bounding_ellipsoid_axes(live_theta)
            self._calls_since_bound_update = 0
        return self._axes

    def record_completed_walk(self, *, accept: int, scale: float) -> None:
        """Record one complete chain and tune before the next replacement."""
        self.record_completed_epoch(((accept, scale),))

    def record_completed_epoch(
        self,
        walks: Iterable[tuple[int, float]],
    ) -> None:
        """Aggregate walks constructed under one scale, then tune once."""
        completed = tuple(walks)
        for index, (accept, scale) in enumerate(completed):
            self._calls_since_bound_update += self.walks
            self.tune(
                {
                    "accept": accept,
                    "reject": self.walks - accept,
                    "scale": scale,
                },
                update=index == len(completed) - 1,
            )

    @property
    def citations(self) -> list[tuple[str, str]]:
        return list(RWALK_CITATIONS)


class SRWalkSampler:
    """Adaptive controller for the Gaussian-covariance ``s-rwalk`` kernel."""

    def __init__(
        self,
        *,
        settings: SRWalkSettings,
        ndim: int,
        step_limit: int | None = None,
    ) -> None:
        self.base_n_steps = settings.n_steps
        self.n_steps = settings.n_steps
        self.facc = min(1.0, max(1.0 / self.n_steps, settings.facc))
        self.ndim = ndim
        self.scale = (
            2.38 / math.sqrt(ndim) if settings.scale is None else settings.scale
        )
        self.covariance_shrinkage = settings.covariance_shrinkage
        self.covariance_jitter = settings.covariance_jitter
        self.dynamic_steps = settings.dynamic_steps
        self.max_steps = min(
            settings.max_steps,
            settings.max_steps if step_limit is None else max(self.n_steps, step_limit),
        )
        self.target_zero_move_probability = settings.target_zero_move_probability
        self.max_step_growth = settings.max_step_growth
        self.zero_accept_scale_factor = settings.zero_accept_scale_factor
        self.zero_move_policy = settings.zero_move_policy
        self._acceptance_history: deque[tuple[int, int]] = deque(
            maxlen=settings.acceptance_window
        )
        self.srwalk_history = {"n_accept": 0, "n_reject": 0}

    @property
    def estimated_acceptance(self) -> float:
        """Return the proposal acceptance observed in the recent walk window."""
        proposed = sum(n_proposed for _, n_proposed in self._acceptance_history)
        if not proposed:
            return self.facc
        return sum(n_accept for n_accept, _ in self._acceptance_history) / proposed

    @property
    def predicted_zero_move_probability(self) -> float:
        """Approximate the chance that the next fixed-length walk never moves."""
        acceptance = self.estimated_acceptance
        if acceptance <= 0.0:
            return 1.0
        if acceptance >= 1.0:
            return 0.0
        return float(math.exp(self.n_steps * math.log1p(-acceptance)))

    def _update_n_steps(self, *, epoch_accept: int) -> None:
        if not self.dynamic_steps:
            self.n_steps = self.base_n_steps
            return
        if epoch_accept == 0:
            desired = math.ceil(self.n_steps * self.max_step_growth)
        else:
            acceptance = self.estimated_acceptance
            if acceptance <= 0.0:
                desired = self.max_steps
            elif acceptance >= 1.0:
                desired = self.base_n_steps
            else:
                desired = math.ceil(
                    math.log(self.target_zero_move_probability)
                    / math.log1p(-acceptance)
                )
        growth_limit = max(
            self.n_steps,
            math.ceil(self.n_steps * self.max_step_growth),
        )
        self.n_steps = min(
            self.max_steps,
            growth_limit,
            max(self.base_n_steps, desired),
        )

    def tune(
        self,
        tuning_info: dict[str, float | int],
        update: bool = True,
    ) -> None:
        """Update proposal scale using the current ``rwalk`` recursion."""
        self.scale = float(tuning_info["scale"])
        self.srwalk_history["n_accept"] += int(tuning_info["accept"])
        self.srwalk_history["n_reject"] += int(tuning_info["reject"])
        if not update:
            return
        accept = self.srwalk_history["n_accept"]
        reject = self.srwalk_history["n_reject"]
        acceptance_fraction = accept / (accept + reject)
        self.scale *= math.exp(
            (acceptance_fraction - self.facc) / self.ndim / self.facc
        )
        self.srwalk_history["n_accept"] = 0
        self.srwalk_history["n_reject"] = 0

    def record_completed_walk(
        self,
        *,
        accept: int,
        scale: float,
        n_steps: int | None = None,
    ) -> None:
        """Tune the scale after one complete fixed-geometry chain."""
        completed_steps = self.n_steps if n_steps is None else n_steps
        self.record_completed_epoch(((accept, scale, completed_steps),))

    def record_completed_epoch(
        self,
        walks: Iterable[tuple[int, float] | tuple[int, float, int]],
    ) -> None:
        """Tune once after a frozen serial walk or parallel queue epoch."""
        completed = tuple(walks)
        normalized: list[tuple[int, float, int]] = []
        for walk in completed:
            if len(walk) == 2:
                accept, scale = walk
                walk_steps = self.n_steps
            else:
                accept, scale, walk_steps = walk
            normalized.append((accept, scale, walk_steps))
        for index, (accept, scale, walk_steps) in enumerate(normalized):
            self.tune(
                {
                    "accept": accept,
                    "reject": walk_steps - accept,
                    "scale": scale,
                },
                update=index == len(normalized) - 1,
            )
            self._acceptance_history.append((accept, walk_steps))
        if not normalized:
            return
        epoch_accept = sum(accept for accept, _, _ in normalized)
        if epoch_accept == 0:
            self.scale *= self.zero_accept_scale_factor
        self._update_n_steps(epoch_accept=epoch_accept)


def _improve_covariance(
    covariance: NDArray[np.float64],
    *,
    ntries: int = 100,
    max_condition_number: float = 1.0e12,
) -> tuple[
    bool,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Condition an ellipsoid covariance using dynesty's repair strategy."""
    ndim = covariance.shape[0]
    repaired = np.array(covariance, dtype=float, copy=True)
    coefficient_minimum = 1.0e-10
    eigenvalue_multiplier = 10.0
    failed = 0
    trial = 0
    eigenvalues = np.empty(ndim)
    eigenvectors = np.eye(ndim)
    axes = np.eye(ndim)

    for trial in range(ntries):
        failed = 0
        try:
            eigenvalues, eigenvectors = scipy_linalg.eigh(
                repaired,
                check_finite=False,
            )
            maximum = float(eigenvalues.max())
            minimum = float(eigenvalues.min())
            if np.all(np.isfinite(eigenvalues)):
                if maximum <= 0.0:
                    failed = 2
                elif minimum < maximum / max_condition_number:
                    failed = 1
                else:
                    axes = eigenvectors * np.sqrt(eigenvalues)
                    break
            else:
                failed = 2
        except scipy_linalg.LinAlgError:
            failed = 2

        if failed == 1:
            fixed = np.maximum(
                eigenvalues,
                eigenvalue_multiplier * maximum / max_condition_number,
            )
            repaired = (eigenvectors * fixed) @ eigenvectors.T
        elif failed == 2:
            coefficient = coefficient_minimum * (1.0 / coefficient_minimum) ** (
                trial / (ntries - 1)
            )
            repaired = (1.0 - coefficient) * repaired + coefficient * np.eye(ndim)

    if failed > 0:
        warnings.warn(
            "Failed to guarantee nonsingular rwalk ellipsoid axes; "
            "defaulting to a unit sphere.",
            RuntimeWarning,
            stacklevel=2,
        )
        repaired = np.eye(ndim)
        precision = np.eye(ndim)
        axes = np.eye(ndim)
    else:
        precision = (eigenvectors * (1.0 / eigenvalues)) @ eigenvectors.T
    return trial == 0, repaired, precision, np.asarray(axes, dtype=float)


def bounding_ellipsoid_axes(
    points: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return axes of Dynesty's single ellipsoid bounding all live points."""
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise NumericalInvariantError(
            "rwalk bounding ellipsoid requires at least two live points"
        )
    if not np.all(np.isfinite(values)):
        raise NumericalInvariantError(
            "rwalk bounding ellipsoid cannot contain nonfinite live points"
        )

    center = np.mean(values, axis=0)
    covariance = np.atleast_2d(np.cov(values, rowvar=False))
    delta = values - center
    one_minus_a_bit = 1.0 - 1.0e-3
    axes = np.eye(values.shape[1])

    for iteration in range(2):
        good_matrix, covariance, precision, axes = _improve_covariance(covariance)
        maximum_distance = float(
            np.einsum("ij,jk,ik->i", delta, precision, delta).max()
        )
        if iteration == 0 and maximum_distance > one_minus_a_bit:
            multiplier = maximum_distance / one_minus_a_bit
            covariance *= multiplier
            precision /= multiplier
            axes *= np.sqrt(multiplier)
        if iteration == 1 and maximum_distance >= 1.0:
            raise NumericalInvariantError(
                "failed to construct an ellipsoid containing all live points"
            )
        if good_matrix:
            break
    if not np.all(np.isfinite(axes)):
        raise NumericalInvariantError("rwalk ellipsoid axes are not finite")
    return np.asarray(axes, dtype=float)


def _random_unit_ball(
    ndim: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Draw uniformly inside an n-dimensional unit ball as in dynesty."""
    direction = rng.standard_normal(size=ndim)
    radius = rng.random() ** (1.0 / ndim)
    norm = scipy_linalg.norm(direction, check_finite=False)
    return np.asarray(direction * (radius / norm), dtype=float)


def covariance_factor(
    points: NDArray[np.float64],
    *,
    shrinkage: float,
    jitter: float,
) -> NDArray[np.float64]:
    """Return a deterministic factor of a regularized live-set covariance.

    The empirical geometry is frozen by each caller for a complete replacement.
    Eigenvalues are floored at ``jitter`` so this remains usable for one point,
    one dimension, and rank-deficient live sets.
    """
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise NumericalInvariantError(
            "proposal covariance requires a nonempty (n_points, ndim) array"
        )
    if not np.all(np.isfinite(values)):
        raise NumericalInvariantError(
            "proposal covariance cannot be formed from nonfinite live points"
        )
    centered = values - np.mean(values, axis=0)
    denominator = max(values.shape[0] - 1, 1)
    with np.errstate(over="ignore", invalid="ignore"):
        covariance = (centered.T @ centered) / denominator
    return _factor_empirical_covariance(
        covariance,
        shrinkage=shrinkage,
        jitter=jitter,
    )


def _factor_empirical_covariance(
    covariance: NDArray[np.float64],
    *,
    shrinkage: float,
    jitter: float,
) -> NDArray[np.float64]:
    """Regularize and factor an empirical covariance with Cholesky first."""
    values = np.asarray(covariance, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or not len(values):
        raise NumericalInvariantError("proposal covariance must be a nonempty square")
    with np.errstate(over="ignore", invalid="ignore"):
        regularized = (1.0 - shrinkage) * values
        diagonal_indices = np.diag_indices_from(regularized)
        regularized[diagonal_indices] = np.diag(values) + jitter
    if not np.all(np.isfinite(regularized)):
        raise NumericalInvariantError(
            "a finite regularized live-set covariance could not be formed"
        )
    regularized = 0.5 * (regularized + regularized.T)
    try:
        factor = np.linalg.cholesky(regularized)
    except np.linalg.LinAlgError:
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(regularized)
        except np.linalg.LinAlgError as eigen_error:
            raise NumericalInvariantError(
                "regularized live-set covariance factorization failed"
            ) from eigen_error
        eigenvalues = np.maximum(eigenvalues, jitter)
        factor = eigenvectors * np.sqrt(eigenvalues)[np.newaxis, :]
    if not np.all(np.isfinite(factor)):
        raise NumericalInvariantError(
            "a finite regularized live-set covariance factor could not be formed"
        )
    return np.asarray(factor, dtype=float)


def _mean_and_scatter(
    points: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise NumericalInvariantError(
            "rolling s-rwalk geometry requires at least two live points"
        )
    if not np.all(np.isfinite(values)):
        raise NumericalInvariantError(
            "rolling s-rwalk geometry cannot contain nonfinite live points"
        )
    mean = np.mean(values, axis=0)
    centered = values - mean
    scatter = centered.T @ centered
    return np.asarray(mean, dtype=float), np.asarray(scatter, dtype=float)


class SRWalkGeometry:
    """Rolling live-set covariance and cached factor for ``s-rwalk``.

    The full live-set mean and scatter are updated with one remove/add pair per
    committed replacement. A survivor covariance is obtained by downdating the
    discarded point in :math:`O(d^2)`. The factor may be reused for several
    replacements because a frozen symmetric Gaussian proposal remains valid.
    """

    def __init__(
        self,
        live_theta: NDArray[np.float64],
        *,
        settings: SRWalkSettings,
    ) -> None:
        self.n_live = len(live_theta)
        self.ndim = live_theta.shape[1]
        self.shrinkage = settings.covariance_shrinkage
        self.jitter = settings.covariance_jitter
        self.update_interval = settings.covariance_update_interval
        self.rebuild_interval = (
            self.n_live
            if settings.covariance_rebuild_interval is None
            else settings.covariance_rebuild_interval
        )
        self.profile = settings.profile
        self.mean, self.scatter = _mean_and_scatter(live_theta)
        self._factor: NDArray[np.float64] | None = None
        self._updates_since_factor = self.update_interval
        self._updates_since_rebuild = 0
        self.n_updates = 0
        self.n_rebuilds = 0
        self.n_factorizations = 0
        self.update_seconds = 0.0
        self.rebuild_seconds = 0.0
        self.factorization_seconds = 0.0

    def factor_for_worst(
        self,
        worst_theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return the current cached factor, refreshing at its configured cadence."""
        if (
            self._factor is not None
            and self._updates_since_factor < self.update_interval
        ):
            return self._factor
        start = time.perf_counter() if self.profile else 0.0
        survivor_count = self.n_live - 1
        survivor_mean = (
            self.n_live * self.mean - np.asarray(worst_theta, dtype=float)
        ) / survivor_count
        survivor_scatter = self.scatter - np.outer(
            np.asarray(worst_theta, dtype=float) - self.mean,
            np.asarray(worst_theta, dtype=float) - survivor_mean,
        )
        survivor_scatter = 0.5 * (survivor_scatter + survivor_scatter.T)
        covariance = survivor_scatter / max(survivor_count - 1, 1)
        factor = _factor_empirical_covariance(
            covariance,
            shrinkage=self.shrinkage,
            jitter=self.jitter,
        )
        factor.setflags(write=False)
        self._factor = factor
        self._updates_since_factor = 0
        self.n_factorizations += 1
        if self.profile:
            self.factorization_seconds += time.perf_counter() - start
        return factor

    def commit_replacement(
        self,
        *,
        outgoing: NDArray[np.float64],
        incoming: NDArray[np.float64],
        live_theta: NDArray[np.float64],
    ) -> None:
        """Update geometry after exactly one authoritative live-set commit."""
        start = time.perf_counter() if self.profile else 0.0
        outgoing_values = np.asarray(outgoing, dtype=float)
        incoming_values = np.asarray(incoming, dtype=float)
        survivor_count = self.n_live - 1
        survivor_mean = (self.n_live * self.mean - outgoing_values) / survivor_count
        survivor_scatter = self.scatter - np.outer(
            outgoing_values - self.mean,
            outgoing_values - survivor_mean,
        )
        new_mean = survivor_mean + (incoming_values - survivor_mean) / self.n_live
        new_scatter = survivor_scatter + np.outer(
            incoming_values - survivor_mean,
            incoming_values - new_mean,
        )
        self.mean = np.asarray(new_mean, dtype=float)
        self.scatter = np.asarray(0.5 * (new_scatter + new_scatter.T), dtype=float)
        self.n_updates += 1
        self._updates_since_factor += 1
        self._updates_since_rebuild += 1
        if self.profile:
            self.update_seconds += time.perf_counter() - start

        if self._updates_since_rebuild >= self.rebuild_interval:
            rebuild_start = time.perf_counter() if self.profile else 0.0
            self.mean, self.scatter = _mean_and_scatter(live_theta)
            self._updates_since_rebuild = 0
            self.n_rebuilds += 1
            if self.profile:
                self.rebuild_seconds += time.perf_counter() - rebuild_start


def log_q0_acceptance_ratio(
    *,
    current_log_q0: float,
    proposed_log_q0: float,
) -> float:
    """Return the symmetric-proposal MH log ratio for constrained ``q0``."""
    return log_metropolis_acceptance_ratio(
        current_log_q0=current_log_q0,
        proposed_log_q0=proposed_log_q0,
        log_hastings_ratio=0.0,
    )


def log_metropolis_acceptance_ratio(
    *,
    current_log_q0: float,
    proposed_log_q0: float,
    log_hastings_ratio: float = 0.0,
) -> float:
    """Return the general fixed-``q0`` Metropolis--Hastings log ratio."""
    if not np.isfinite(current_log_q0):
        raise NumericalInvariantError(
            "an eligible MCMC starting state must have finite log_q0"
        )
    if np.isnan(proposed_log_q0) or np.isposinf(proposed_log_q0):
        raise NumericalInvariantError("an MCMC proposal has invalid log_q0")
    if np.isnan(log_hastings_ratio) or np.isposinf(log_hastings_ratio):
        raise NumericalInvariantError("an MCMC proposal has invalid Hastings ratio")
    if np.isneginf(proposed_log_q0) or np.isneginf(log_hastings_ratio):
        return -np.inf
    return min(
        0.0,
        proposed_log_q0 - current_log_q0 + log_hastings_ratio,
    )


def accepts_metropolis(
    *,
    current_log_q0: float,
    proposed_log_q0: float,
    log_hastings_ratio: float,
    rng: np.random.Generator,
) -> bool:
    """Draw a fixed-``q0`` Metropolis--Hastings decision in log space."""
    log_alpha = log_metropolis_acceptance_ratio(
        current_log_q0=current_log_q0,
        proposed_log_q0=proposed_log_q0,
        log_hastings_ratio=log_hastings_ratio,
    )
    return bool(np.log(rng.random()) < log_alpha)


def accepts_log_q0_metropolis(
    *,
    current_log_q0: float,
    proposed_log_q0: float,
    rng: np.random.Generator,
) -> bool:
    """Draw the exact fixed-``q0`` Metropolis decision in log space."""
    return accepts_metropolis(
        current_log_q0=current_log_q0,
        proposed_log_q0=proposed_log_q0,
        log_hastings_ratio=0.0,
        rng=rng,
    )


def eligible_survivor_indices(
    *,
    live_log_psi0: NDArray[np.float64],
    live_tie_breakers: NDArray[np.float64],
    worst: int,
    threshold: float,
    threshold_tie_breaker: float,
    tie_policy: TiePolicy,
) -> NDArray[np.int64]:
    """Return surviving live indices inside the active augmented constraint."""
    valid = np.asarray(
        passes_constraint(
            live_log_psi0,
            live_tie_breakers,
            threshold=threshold,
            threshold_tie_breaker=threshold_tie_breaker,
            tie_policy=tie_policy,
        ),
        dtype=bool,
    )
    if valid.ndim != 1 or live_tie_breakers.shape != live_log_psi0.shape:
        raise NumericalInvariantError("live constraint arrays must be one-dimensional")
    if worst < 0 or worst >= len(valid):
        raise NumericalInvariantError("worst live-point index is out of range")
    valid[worst] = False
    return np.asarray(np.flatnonzero(valid), dtype=np.int64)


def _validate_live_arrays(
    *,
    evaluator: BatchEvaluator,
    live_theta: NDArray[np.float64],
    live_log_likelihood: NDArray[np.float64],
    live_log_prior: NDArray[np.float64],
    live_log_q0: NDArray[np.float64],
    live_log_psi0: NDArray[np.float64],
    live_tie_breakers: NDArray[np.float64],
) -> None:
    if live_theta.ndim != 2 or live_theta.shape[1] != evaluator.ndim:
        raise NumericalInvariantError(
            f"live_theta must have shape (n_live, {evaluator.ndim})"
        )
    expected = (len(live_theta),)
    for name, values in (
        ("live_log_likelihood", live_log_likelihood),
        ("live_log_prior", live_log_prior),
        ("live_log_q0", live_log_q0),
        ("live_log_psi0", live_log_psi0),
        ("live_tie_breakers", live_tie_breakers),
    ):
        if values.shape != expected:
            raise NumericalInvariantError(f"{name} must have shape {expected}")


def _point_from_live(
    index: int,
    *,
    live_theta: NDArray[np.float64],
    live_log_likelihood: NDArray[np.float64],
    live_log_prior: NDArray[np.float64],
    live_log_q0: NDArray[np.float64],
    live_log_psi0: NDArray[np.float64],
    live_tie_breakers: NDArray[np.float64],
) -> EvaluatedPoint:
    return EvaluatedPoint(
        theta=np.array(live_theta[index], copy=True),
        log_likelihood=float(live_log_likelihood[index]),
        log_prior=float(live_log_prior[index]),
        log_q0=float(live_log_q0[index]),
        log_psi0=float(live_log_psi0[index]),
        tie_breaker=float(live_tie_breakers[index]),
    )


def _point_from_batch(
    batch: EvaluatedBatch,
    index: int,
    tie_breaker: float,
) -> EvaluatedPoint:
    return EvaluatedPoint(
        theta=np.array(batch.theta[index], copy=True),
        log_likelihood=float(batch.log_likelihood[index]),
        log_prior=float(batch.log_prior[index]),
        log_q0=float(batch.log_q0[index]),
        log_psi0=float(batch.log_psi0[index]),
        tie_breaker=tie_breaker,
    )


def _preflight_failure(
    *,
    evaluator: BatchEvaluator,
    required_proposals: int,
    max_proposals: int,
    max_likelihood_calls: int | None,
    deadline: float | None,
) -> str | None:
    if required_proposals > max_proposals:
        return "max_proposals_per_replacement"
    if (
        max_likelihood_calls is not None
        and evaluator.n_likelihood_calls + required_proposals > max_likelihood_calls
    ):
        return "max_likelihood_calls"
    if deadline is not None and time.monotonic() >= deadline:
        return "max_wall_time"
    return None


def draw_rwalk_constrained(
    *,
    evaluator: BatchEvaluator,
    live_theta: NDArray[np.float64],
    live_log_likelihood: NDArray[np.float64],
    live_log_prior: NDArray[np.float64],
    live_log_q0: NDArray[np.float64],
    live_log_psi0: NDArray[np.float64],
    live_tie_breakers: NDArray[np.float64],
    worst: int,
    threshold: float,
    threshold_tie_breaker: float,
    tie_policy: TiePolicy,
    sampler: RWalkSampler,
    rng: np.random.Generator,
    max_proposals: int,
    max_likelihood_calls: int | None,
    deadline: float | None,
) -> ConstrainedAttempt:
    """Evolve a survivor with MH invariant under constrained fixed ``q0``.

    The discarded point supplies the threshold but is never a chain state.
    Exactly ``walks`` symmetric ellipsoidal-ball candidates are attempted.
    """
    _validate_live_arrays(
        evaluator=evaluator,
        live_theta=live_theta,
        live_log_likelihood=live_log_likelihood,
        live_log_prior=live_log_prior,
        live_log_q0=live_log_q0,
        live_log_psi0=live_log_psi0,
        live_tie_breakers=live_tie_breakers,
    )
    failure = _preflight_failure(
        evaluator=evaluator,
        required_proposals=sampler.walks,
        max_proposals=max_proposals,
        max_likelihood_calls=max_likelihood_calls,
        deadline=deadline,
    )
    if failure is not None:
        return ConstrainedAttempt(None, failure, 0, 0)
    eligible = eligible_survivor_indices(
        live_log_psi0=live_log_psi0,
        live_tie_breakers=live_tie_breakers,
        worst=worst,
        threshold=threshold,
        threshold_tie_breaker=threshold_tie_breaker,
        tie_policy=tie_policy,
    )
    if len(eligible) == 0:
        return ConstrainedAttempt(None, "insufficient_eligible_survivors", 0, 0)
    axes = sampler.axes_for(live_theta)
    start_index = int(rng.choice(eligible))
    current = _point_from_live(
        start_index,
        live_theta=live_theta,
        live_log_likelihood=live_log_likelihood,
        live_log_prior=live_log_prior,
        live_log_q0=live_log_q0,
        live_log_psi0=live_log_psi0,
        live_tie_breakers=live_tie_breakers,
    )
    scale = sampler.scale
    n_proposed = 0
    n_valid = 0
    n_accepted = 0
    moved = False
    for _ in range(sampler.walks):
        if deadline is not None and time.monotonic() >= deadline:
            return ConstrainedAttempt(
                None,
                "max_wall_time",
                n_proposed,
                n_valid,
                n_accepted,
                int(moved),
                n_proposed,
            )
        proposal = current.theta + scale * (
            axes @ _random_unit_ball(evaluator.ndim, rng)
        )
        (
            proposal_theta,
            proposal_log_likelihood,
            proposal_log_prior,
            proposal_log_q0,
            proposal_log_psi0,
        ) = evaluator.evaluate_one(proposal)
        n_proposed += 1
        proposed_tie = float(rng.random())
        valid = bool(
            passes_constraint(
                proposal_log_psi0,
                proposed_tie,
                threshold=threshold,
                threshold_tie_breaker=threshold_tie_breaker,
                tie_policy=tie_policy,
            )
        )
        if not valid:
            continue
        n_valid += 1
        if accepts_log_q0_metropolis(
            current_log_q0=current.log_q0,
            proposed_log_q0=proposal_log_q0,
            rng=rng,
        ):
            current = EvaluatedPoint(
                theta=proposal_theta,
                log_likelihood=proposal_log_likelihood,
                log_prior=proposal_log_prior,
                log_q0=proposal_log_q0,
                log_psi0=proposal_log_psi0,
                tie_breaker=proposed_tie,
            )
            n_accepted += 1
            moved = True
    draw = ConstrainedDraw(current, n_proposed, n_valid)
    sampler.record_completed_walk(accept=n_accepted, scale=scale)
    return ConstrainedAttempt(
        draw,
        None,
        n_proposed,
        n_valid,
        n_accepted,
        int(moved),
        sampler.walks,
    )


def draw_srwalk_constrained(
    *,
    evaluator: BatchEvaluator,
    live_theta: NDArray[np.float64],
    live_log_likelihood: NDArray[np.float64],
    live_log_prior: NDArray[np.float64],
    live_log_q0: NDArray[np.float64],
    live_log_psi0: NDArray[np.float64],
    live_tie_breakers: NDArray[np.float64],
    worst: int,
    threshold: float,
    threshold_tie_breaker: float,
    tie_policy: TiePolicy,
    sampler: SRWalkSampler,
    rng: np.random.Generator,
    max_proposals: int,
    max_likelihood_calls: int | None,
    deadline: float | None,
    proposal_factor: NDArray[np.float64] | None = None,
) -> ConstrainedAttempt:
    """Evolve a survivor with Gaussian MH targeting constrained fixed ``q0``.

    The discarded point defines the constraint but is never a chain state. The
    regularized survivor covariance and adaptive scale are frozen for all
    ``n_steps`` transitions in this replacement. Each valid symmetric proposal
    is accepted with the fixed-importance ratio ``q0(proposed) / q0(current)``.
    """
    _validate_live_arrays(
        evaluator=evaluator,
        live_theta=live_theta,
        live_log_likelihood=live_log_likelihood,
        live_log_prior=live_log_prior,
        live_log_q0=live_log_q0,
        live_log_psi0=live_log_psi0,
        live_tie_breakers=live_tie_breakers,
    )
    walk_steps = sampler.n_steps
    failure = _preflight_failure(
        evaluator=evaluator,
        required_proposals=walk_steps,
        max_proposals=max_proposals,
        # Prior-first evaluation makes this an upper bound rather than an
        # exact likelihood-call requirement. The loop enforces the hard limit.
        max_likelihood_calls=None,
        deadline=deadline,
    )
    if failure is not None:
        return ConstrainedAttempt(None, failure, 0, 0)
    eligible = eligible_survivor_indices(
        live_log_psi0=live_log_psi0,
        live_tie_breakers=live_tie_breakers,
        worst=worst,
        threshold=threshold,
        threshold_tie_breaker=threshold_tie_breaker,
        tie_policy=tie_policy,
    )
    if len(eligible) == 0:
        return ConstrainedAttempt(None, "insufficient_eligible_survivors", 0, 0)
    factorization_start = time.perf_counter() if evaluator.profile else 0.0
    if proposal_factor is None:
        survivors = np.concatenate((live_theta[:worst], live_theta[worst + 1 :]))
        factor = covariance_factor(
            survivors,
            shrinkage=sampler.covariance_shrinkage,
            jitter=sampler.covariance_jitter,
        )
    else:
        factor = np.asarray(proposal_factor, dtype=float)
        if factor.shape != (evaluator.ndim, evaluator.ndim):
            raise NumericalInvariantError("prepared s-rwalk factor has the wrong shape")
        if not np.all(np.isfinite(factor)):
            raise NumericalInvariantError(
                "prepared s-rwalk factor contains NaN or infinity"
            )
    factorization_seconds = (
        time.perf_counter() - factorization_start
        if evaluator.profile and proposal_factor is None
        else 0.0
    )
    start_index = int(rng.choice(eligible))
    starting = _point_from_live(
        start_index,
        live_theta=live_theta,
        live_log_likelihood=live_log_likelihood,
        live_log_prior=live_log_prior,
        live_log_q0=live_log_q0,
        live_log_psi0=live_log_psi0,
        live_tie_breakers=live_tie_breakers,
    )
    scale = sampler.scale
    proposal_start = time.perf_counter() if evaluator.profile else 0.0
    standard_normals = rng.standard_normal(size=(walk_steps, evaluator.ndim))
    increments = scale * (standard_normals @ factor.T)
    proposal_seconds = (
        time.perf_counter() - proposal_start if evaluator.profile else 0.0
    )
    current_theta = starting.theta
    current_log_likelihood = starting.log_likelihood
    current_log_prior = starting.log_prior
    current_log_q0 = starting.log_q0
    current_log_psi0 = starting.log_psi0
    current_tie_breaker = starting.tie_breaker
    n_proposed = 0
    n_valid = 0
    n_accepted = 0
    moved = False
    n_completed = 0
    for increment in increments:
        if deadline is not None and time.monotonic() >= deadline:
            return ConstrainedAttempt(
                None,
                "max_wall_time",
                n_proposed,
                n_valid,
                n_accepted,
                int(moved),
                n_completed,
                srwalk_factorization_seconds=factorization_seconds,
                srwalk_proposal_seconds=proposal_seconds,
                srwalk_squared_displacement=float(
                    np.sum((current_theta - starting.theta) ** 2)
                ),
            )
        proposal = current_theta + increment
        n_proposed += 1
        try:
            (
                proposal_theta,
                proposal_log_likelihood,
                proposal_log_prior,
                proposal_log_q0,
                proposal_log_psi0,
            ) = evaluator.evaluate_one(
                proposal,
                max_likelihood_calls=max_likelihood_calls,
            )
        except LikelihoodBudgetExhausted:
            return ConstrainedAttempt(
                None,
                "max_likelihood_calls",
                n_proposed,
                n_valid,
                n_accepted,
                int(moved),
                n_completed,
                srwalk_factorization_seconds=factorization_seconds,
                srwalk_proposal_seconds=proposal_seconds,
                srwalk_squared_displacement=float(
                    np.sum((current_theta - starting.theta) ** 2)
                ),
            )
        proposed_tie = float(rng.random())
        valid = bool(
            passes_constraint(
                proposal_log_psi0,
                proposed_tie,
                threshold=threshold,
                threshold_tie_breaker=threshold_tie_breaker,
                tie_policy=tie_policy,
            )
        )
        if not valid:
            n_completed += 1
            continue
        n_valid += 1
        if accepts_log_q0_metropolis(
            current_log_q0=current_log_q0,
            proposed_log_q0=proposal_log_q0,
            rng=rng,
        ):
            current_theta = proposal_theta
            current_log_likelihood = proposal_log_likelihood
            current_log_prior = proposal_log_prior
            current_log_q0 = proposal_log_q0
            current_log_psi0 = proposal_log_psi0
            current_tie_breaker = proposed_tie
            n_accepted += 1
            moved = True
        n_completed += 1
    current = EvaluatedPoint(
        theta=np.array(current_theta, copy=True),
        log_likelihood=current_log_likelihood,
        log_prior=current_log_prior,
        log_q0=current_log_q0,
        log_psi0=current_log_psi0,
        tie_breaker=current_tie_breaker,
    )
    draw = ConstrainedDraw(current, n_proposed, n_valid)
    sampler.record_completed_walk(
        accept=n_accepted,
        scale=scale,
        n_steps=walk_steps,
    )
    if not moved and sampler.zero_move_policy == "stop":
        return ConstrainedAttempt(
            None,
            "srwalk_stalled",
            n_proposed,
            n_valid,
            n_accepted,
            0,
            walk_steps,
            srwalk_factorization_seconds=factorization_seconds,
            srwalk_proposal_seconds=proposal_seconds,
            srwalk_squared_displacement=0.0,
        )
    return ConstrainedAttempt(
        draw,
        None,
        n_proposed,
        n_valid,
        n_accepted,
        int(moved),
        walk_steps,
        srwalk_factorization_seconds=factorization_seconds,
        srwalk_proposal_seconds=proposal_seconds,
        srwalk_squared_displacement=float(
            np.sum((current_theta - starting.theta) ** 2)
        ),
    )


_ENSEMBLE_MOVE_NAMES: tuple[EnsembleMoveName, ...] = (
    "de",
    "stretch",
    "gaussian",
)


@dataclass(frozen=True, slots=True)
class _EnsembleProposalBatch:
    """One vectorized half-ensemble proposal and its Hastings corrections."""

    theta: NDArray[np.float64]
    log_hastings_ratio: NDArray[np.float64]
    move_name: EnsembleMoveName


def _select_ensemble_move(
    *,
    move_weights: EnsembleMoveWeights,
    rng: np.random.Generator,
) -> EnsembleMoveName:
    """Select one move, avoiding an RNG draw for a pure configuration."""
    names, probabilities = move_weights.active_names_and_probabilities
    if len(names) == 1:
        return names[0]
    selected = int(rng.choice(len(names), p=probabilities))
    return names[selected]


def _propose_de_move(
    *,
    ensemble_theta: NDArray[np.float64],
    active: NDArray[np.int64],
    complement: NDArray[np.int64],
    settings: EnsembleRWalkSettings,
    factor: NDArray[np.float64],
    rng: np.random.Generator,
) -> _EnsembleProposalBatch:
    """Propose the legacy ordered-pair differential-evolution move."""
    ndim = ensemble_theta.shape[1]
    gamma = 2.38 / np.sqrt(2.0 * ndim) if settings.gamma is None else settings.gamma
    proposals = np.empty((len(active), ndim), dtype=float)
    for row, walker in enumerate(active):
        references = np.asarray(
            rng.choice(complement, size=2, replace=False),
            dtype=np.int64,
        )
        difference = ensemble_theta[references[0]] - ensemble_theta[references[1]]
        jitter = settings.jitter_scale * (factor @ rng.normal(size=ndim))
        proposals[row] = ensemble_theta[walker] + gamma * difference + jitter
    return _EnsembleProposalBatch(
        theta=proposals,
        log_hastings_ratio=np.zeros(len(active), dtype=float),
        move_name="de",
    )


def _draw_stretch_factors(
    *,
    n_active: int,
    stretch_scale: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Draw Goodman--Weare stretch factors with density proportional to z^-1/2."""
    uniform = np.asarray(rng.random(n_active), dtype=float)
    return np.asarray(
        ((stretch_scale - 1.0) * uniform + 1.0) ** 2 / stretch_scale,
        dtype=float,
    )


def _propose_stretch_move(
    *,
    ensemble_theta: NDArray[np.float64],
    active: NDArray[np.int64],
    complement: NDArray[np.int64],
    stretch_scale: float,
    rng: np.random.Generator,
) -> _EnsembleProposalBatch:
    """Propose a split-ensemble stretch move with its Jacobian correction."""
    reference_rows = np.asarray(
        rng.choice(complement, size=len(active), replace=True),
        dtype=np.int64,
    )
    stretch = _draw_stretch_factors(
        n_active=len(active),
        stretch_scale=stretch_scale,
        rng=rng,
    )
    reference_theta = ensemble_theta[reference_rows]
    proposals = reference_theta + stretch[:, np.newaxis] * (
        ensemble_theta[active] - reference_theta
    )
    ndim = ensemble_theta.shape[1]
    return _EnsembleProposalBatch(
        theta=np.asarray(proposals, dtype=float),
        log_hastings_ratio=np.asarray((ndim - 1.0) * np.log(stretch), dtype=float),
        move_name="stretch",
    )


def _propose_gaussian_move(
    *,
    ensemble_theta: NDArray[np.float64],
    active: NDArray[np.int64],
    gaussian_scale: float,
    factor: NDArray[np.float64],
    rng: np.random.Generator,
) -> _EnsembleProposalBatch:
    """Propose symmetric Gaussian steps using frozen survivor geometry."""
    ndim = ensemble_theta.shape[1]
    proposals = np.empty((len(active), ndim), dtype=float)
    for row, walker in enumerate(active):
        proposals[row] = ensemble_theta[walker] + gaussian_scale * (
            factor @ rng.standard_normal(ndim)
        )
    return _EnsembleProposalBatch(
        theta=proposals,
        log_hastings_ratio=np.zeros(len(active), dtype=float),
        move_name="gaussian",
    )


def _ensemble_move_stats(
    *,
    proposed: NDArray[np.int64],
    valid: NDArray[np.int64],
    accepted: NDArray[np.int64],
    moved: NDArray[np.bool_],
) -> tuple[EnsembleMoveStats, ...]:
    """Freeze internal per-move counters in canonical order."""
    return tuple(
        EnsembleMoveStats(
            name=name,
            n_proposed=int(proposed[index]),
            n_valid=int(valid[index]),
            n_accepted=int(accepted[index]),
            n_moved=int(np.count_nonzero(moved[index])),
        )
        for index, name in enumerate(_ENSEMBLE_MOVE_NAMES)
    )


def draw_ensemble_rwalk_constrained(
    *,
    evaluator: BatchEvaluator,
    live_theta: NDArray[np.float64],
    live_log_likelihood: NDArray[np.float64],
    live_log_prior: NDArray[np.float64],
    live_log_q0: NDArray[np.float64],
    live_log_psi0: NDArray[np.float64],
    live_tie_breakers: NDArray[np.float64],
    worst: int,
    threshold: float,
    threshold_tie_breaker: float,
    tie_policy: TiePolicy,
    settings: EnsembleRWalkSettings,
    rng: np.random.Generator,
    max_proposals: int,
    max_likelihood_calls: int | None,
    deadline: float | None,
) -> ConstrainedAttempt:
    """Run a split-ensemble MH mixture targeting constrained fixed ``q0``."""
    _validate_live_arrays(
        evaluator=evaluator,
        live_theta=live_theta,
        live_log_likelihood=live_log_likelihood,
        live_log_prior=live_log_prior,
        live_log_q0=live_log_q0,
        live_log_psi0=live_log_psi0,
        live_tie_breakers=live_tie_breakers,
    )
    required_proposals = settings.n_walkers * settings.n_sweeps
    failure = _preflight_failure(
        evaluator=evaluator,
        required_proposals=required_proposals,
        max_proposals=max_proposals,
        max_likelihood_calls=max_likelihood_calls,
        deadline=deadline,
    )
    if failure is not None:
        return ConstrainedAttempt(None, failure, 0, 0)
    eligible = eligible_survivor_indices(
        live_log_psi0=live_log_psi0,
        live_tie_breakers=live_tie_breakers,
        worst=worst,
        threshold=threshold,
        threshold_tie_breaker=threshold_tie_breaker,
        tie_policy=tie_policy,
    )
    if len(eligible) < settings.n_walkers:
        return ConstrainedAttempt(None, "insufficient_eligible_walkers", 0, 0)
    initial_indices = np.asarray(
        rng.choice(eligible, size=settings.n_walkers, replace=False),
        dtype=np.int64,
    )
    ensemble_theta = np.array(live_theta[initial_indices], copy=True)
    ensemble_log_likelihood = np.array(
        live_log_likelihood[initial_indices],
        copy=True,
    )
    ensemble_log_prior = np.array(live_log_prior[initial_indices], copy=True)
    ensemble_log_q0 = np.array(live_log_q0[initial_indices], copy=True)
    ensemble_log_psi0 = np.array(live_log_psi0[initial_indices], copy=True)
    ensemble_ties = np.array(live_tie_breakers[initial_indices], copy=True)
    factor_cache: NDArray[np.float64] | None = None

    def frozen_factor() -> NDArray[np.float64]:
        nonlocal factor_cache
        if factor_cache is None:
            survivors = np.delete(live_theta, worst, axis=0)
            factor_cache = covariance_factor(
                survivors,
                shrinkage=settings.covariance_shrinkage,
                jitter=settings.covariance_jitter,
            )
        return factor_cache

    gaussian_scale = (
        2.38 / np.sqrt(evaluator.ndim)
        if settings.gaussian_scale is None
        else settings.gaussian_scale
    )
    n_proposed = 0
    n_valid = 0
    n_accepted = 0
    completed_sweeps = 0
    moved = np.zeros(settings.n_walkers, dtype=bool)
    move_proposed = np.zeros(len(_ENSEMBLE_MOVE_NAMES), dtype=np.int64)
    move_valid = np.zeros(len(_ENSEMBLE_MOVE_NAMES), dtype=np.int64)
    move_accepted = np.zeros(len(_ENSEMBLE_MOVE_NAMES), dtype=np.int64)
    moved_by_move = np.zeros(
        (len(_ENSEMBLE_MOVE_NAMES), settings.n_walkers),
        dtype=bool,
    )

    for _ in range(settings.n_sweeps):
        permutation = np.asarray(rng.permutation(settings.n_walkers), dtype=np.int64)
        midpoint = settings.n_walkers // 2
        halves = (
            (permutation[:midpoint], permutation[midpoint:]),
            (permutation[midpoint:], permutation[:midpoint]),
        )
        for active, complement in halves:
            if deadline is not None and time.monotonic() >= deadline:
                return ConstrainedAttempt(
                    None,
                    "max_wall_time",
                    n_proposed,
                    n_valid,
                    n_accepted,
                    int(np.count_nonzero(moved)),
                    completed_sweeps,
                    _ensemble_move_stats(
                        proposed=move_proposed,
                        valid=move_valid,
                        accepted=move_accepted,
                        moved=moved_by_move,
                    ),
                )
            move_name = _select_ensemble_move(
                move_weights=settings.move_weights,
                rng=rng,
            )
            if move_name == "de":
                proposal = _propose_de_move(
                    ensemble_theta=ensemble_theta,
                    active=active,
                    complement=complement,
                    settings=settings,
                    factor=frozen_factor(),
                    rng=rng,
                )
            elif move_name == "stretch":
                proposal = _propose_stretch_move(
                    ensemble_theta=ensemble_theta,
                    active=active,
                    complement=complement,
                    stretch_scale=settings.stretch_scale,
                    rng=rng,
                )
            else:
                proposal = _propose_gaussian_move(
                    ensemble_theta=ensemble_theta,
                    active=active,
                    gaussian_scale=gaussian_scale,
                    factor=frozen_factor(),
                    rng=rng,
                )
            move_index = _ENSEMBLE_MOVE_NAMES.index(proposal.move_name)
            batch = evaluator.evaluate(proposal.theta)
            n_proposed += len(active)
            move_proposed[move_index] += len(active)
            proposed_ties = np.asarray(rng.random(len(active)), dtype=float)
            valid = np.asarray(
                passes_constraint(
                    batch.log_psi0,
                    proposed_ties,
                    threshold=threshold,
                    threshold_tie_breaker=threshold_tie_breaker,
                    tie_policy=tie_policy,
                ),
                dtype=bool,
            )
            valid_count = int(np.count_nonzero(valid))
            n_valid += valid_count
            move_valid[move_index] += valid_count
            for row, walker in enumerate(active):
                if not valid[row]:
                    continue
                if not accepts_metropolis(
                    current_log_q0=float(ensemble_log_q0[walker]),
                    proposed_log_q0=float(batch.log_q0[row]),
                    log_hastings_ratio=float(proposal.log_hastings_ratio[row]),
                    rng=rng,
                ):
                    continue
                ensemble_theta[walker] = batch.theta[row]
                ensemble_log_likelihood[walker] = batch.log_likelihood[row]
                ensemble_log_prior[walker] = batch.log_prior[row]
                ensemble_log_q0[walker] = batch.log_q0[row]
                ensemble_log_psi0[walker] = batch.log_psi0[row]
                ensemble_ties[walker] = proposed_ties[row]
                n_accepted += 1
                move_accepted[move_index] += 1
                moved[walker] = True
                moved_by_move[move_index, walker] = True
        completed_sweeps += 1

    replacement_index = int(rng.integers(settings.n_walkers))
    replacement = EvaluatedPoint(
        theta=np.array(ensemble_theta[replacement_index], copy=True),
        log_likelihood=float(ensemble_log_likelihood[replacement_index]),
        log_prior=float(ensemble_log_prior[replacement_index]),
        log_q0=float(ensemble_log_q0[replacement_index]),
        log_psi0=float(ensemble_log_psi0[replacement_index]),
        tie_breaker=float(ensemble_ties[replacement_index]),
    )
    draw = ConstrainedDraw(replacement, n_proposed, n_valid)
    return ConstrainedAttempt(
        draw,
        None,
        n_proposed,
        n_valid,
        n_accepted,
        int(np.count_nonzero(moved)),
        completed_sweeps,
        _ensemble_move_stats(
            proposed=move_proposed,
            valid=move_valid,
            accepted=move_accepted,
            moved=moved_by_move,
        ),
    )
