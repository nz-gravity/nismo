"""Validated stopping policies and pure stopping diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp

from .exceptions import ConfigurationError, NumericalInvariantError

StoppingCriterionName = Literal[
    "remaining_fraction",
    "remaining_dlogz",
    "live_logz_error",
    "logz_stability",
    "live_ess",
    "logzerr",
]
StoppingMode = Literal["all", "any"]

SUPPORTED_STOPPING_CRITERIA = frozenset(
    {
        "remaining_fraction",
        "remaining_dlogz",
        "live_logz_error",
        "logz_stability",
        "live_ess",
        "logzerr",
    }
)
LESS_THAN_OR_EQUAL = frozenset(
    {
        "remaining_fraction",
        "remaining_dlogz",
        "live_logz_error",
        "logz_stability",
        "logzerr",
    }
)
GREATER_THAN_OR_EQUAL = frozenset({"live_ess"})
SCIENTIFIC_TERMINATION_REASONS = frozenset(
    {
        "remaining_evidence",
        "stopping_criteria",
    }
)


def _validated_tolerance(name: str, tolerance: object) -> float:
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real):
        raise ConfigurationError(f"{name} tolerance must be a real number")
    value = float(tolerance)
    if not np.isfinite(value):
        raise ConfigurationError(f"{name} tolerance must be finite")
    if name == "remaining_fraction":
        if not 0.0 < value < 1.0:
            raise ConfigurationError(
                "remaining_fraction tolerance must satisfy 0 < tolerance < 1"
            )
    elif name == "live_ess":
        if value < 1.0:
            raise ConfigurationError("live_ess tolerance must be >= 1")
    elif value <= 0.0:
        raise ConfigurationError(f"{name} tolerance must be > 0")
    return value


def _validated_logzerr(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise NumericalInvariantError("logzerr must be finite and non-negative")
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise NumericalInvariantError("logzerr must be finite and non-negative")
    return number


@dataclass(frozen=True, slots=True)
class StoppingCriterionConfig:
    """One scientifically defined stopping metric and its threshold."""

    name: StoppingCriterionName
    tolerance: float

    def __post_init__(self) -> None:
        if self.name not in SUPPORTED_STOPPING_CRITERIA:
            raise ConfigurationError(f"unsupported stopping criterion: {self.name!r}")
        object.__setattr__(
            self,
            "tolerance",
            _validated_tolerance(self.name, self.tolerance),
        )


@dataclass(frozen=True, slots=True)
class StoppingPolicy:
    """Immutable combination and persistence rules for scientific stopping."""

    criteria: tuple[StoppingCriterionConfig, ...]
    mode: StoppingMode = "all"
    consecutive: int = 1
    min_iterations: int = 0
    stability_window: int = 10

    def __post_init__(self) -> None:
        criteria = tuple(self.criteria)
        if not criteria:
            raise ConfigurationError("stopping policy requires at least one criterion")
        if not all(
            isinstance(criterion, StoppingCriterionConfig) for criterion in criteria
        ):
            raise ConfigurationError(
                "stopping criteria must be StoppingCriterionConfig instances"
            )
        names = [criterion.name for criterion in criteria]
        if len(set(names)) != len(names):
            raise ConfigurationError("stopping criterion names must be unique")
        if self.mode not in ("all", "any"):
            raise ConfigurationError(f"unsupported stopping mode: {self.mode!r}")
        for name, value, minimum in (
            ("consecutive", self.consecutive, 1),
            ("min_iterations", self.min_iterations, 0),
            ("stability_window", self.stability_window, 2),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                comparator = ">=" if minimum != 1 else "a positive integer"
                requirement = (
                    f"an integer {comparator} {minimum}"
                    if comparator == ">="
                    else comparator
                )
                raise ConfigurationError(f"{name} must be {requirement}")
        object.__setattr__(self, "criteria", criteria)


@dataclass(frozen=True, slots=True)
class StoppingMetrics:
    """Stopping diagnostic values for one completed replacement."""

    remaining_fraction: float
    remaining_dlogz: float
    live_ess: float
    live_mean_rse: float
    live_logz_error: float
    logz_stability: float
    logzerr: float


@dataclass(frozen=True, slots=True)
class CriterionEvaluation:
    """Value, threshold, and result for one enabled criterion."""

    name: StoppingCriterionName
    value: float
    tolerance: float
    met: bool


@dataclass(frozen=True, slots=True)
class StoppingDecision:
    """Combined policy result and updated consecutive-pass state."""

    evaluations: tuple[CriterionEvaluation, ...]
    combined_met: bool
    streak: int
    should_stop: bool


def validate_stopping_policy_for_n_live(
    policy: StoppingPolicy,
    n_live: int,
) -> None:
    """Validate criteria whose bounds depend on the sampler live count."""
    for criterion in policy.criteria:
        if criterion.name == "live_ess" and criterion.tolerance > n_live:
            raise ConfigurationError(
                "live_ess tolerance must be <= n_live "
                f"({criterion.tolerance:g} > {n_live})"
            )


def calculate_remaining_dlogz(
    *,
    logz_dead: float,
    logz_live: float,
) -> float:
    """Return the estimated log-evidence increment from the live remainder.

    Computes ``log(Z_dead + Z_live) - log(Z_dead)``, equivalently
    ``log(1 + Z_live / Z_dead)``, in natural-log units.
    """
    if np.isnan(logz_dead) or np.isnan(logz_live):
        raise NumericalInvariantError(
            "dead and live log-evidence values must not be NaN"
        )
    if np.isposinf(logz_dead) or np.isposinf(logz_live):
        raise NumericalInvariantError(
            "dead and live log-evidence values must not be positive infinity"
        )
    if np.isneginf(logz_dead):
        return float("inf")
    if np.isneginf(logz_live):
        return 0.0

    log_live_to_dead = logz_live - logz_dead
    return float(np.logaddexp(0.0, log_live_to_dead))


def calculate_stopping_metrics(
    *,
    live_log_psi: ArrayLike,
    logz_dead: float,
    logz_live: float,
    logz_total: float,
    logz_history: ArrayLike,
    logzerr: float,
    stability_window: int,
) -> StoppingMetrics:
    """Calculate finite-live-set and evidence diagnostics in log space.

    ``live_logz_error`` is the first-order delta-method estimate
    ``remaining_fraction * live_mean_rse``. It measures Monte Carlo uncertainty
    transmitted from representing the remaining integral by a finite live-set
    mean. It is not a complete uncertainty estimate: shrinkage, fitted-proposal
    error, missing support, undiscovered modes, and invalid constrained draws
    are outside its scope.
    """
    values = np.asarray(live_log_psi, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("live_log_psi must be one-dimensional with length >= 2")
    if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        raise NumericalInvariantError("live_log_psi contains NaN or positive infinity")
    if (
        isinstance(stability_window, bool)
        or not isinstance(stability_window, int)
        or stability_window < 2
    ):
        raise ValueError("stability_window must be an integer >= 2")
    checked_logzerr = _validated_logzerr(logzerr)
    remaining_dlogz = calculate_remaining_dlogz(
        logz_dead=logz_dead,
        logz_live=logz_live,
    )

    n_live = len(values)
    all_zero = bool(np.all(np.isneginf(values)))
    if all_zero:
        if np.isnan(logz_total) or np.isposinf(logz_total):
            raise NumericalInvariantError(
                "all-zero live contributions require finite or zero total evidence"
            )
        if not np.isneginf(logz_live):
            raise NumericalInvariantError(
                "all-zero live contributions require logz_live == -inf"
            )
        remaining_fraction = 0.0
        live_ess = float(n_live)
        live_mean_rse = 0.0
        live_logz_error = 0.0
    else:
        if not np.isfinite(logz_live) or not np.isfinite(logz_total):
            raise NumericalInvariantError(
                "nonzero live contributions require finite live and total logz"
            )
        with np.errstate(over="ignore"):
            log_fraction = float(logz_live - logz_total)
        if log_fraction > 1.0e-12:
            raise NumericalInvariantError("live evidence exceeds total evidence")
        remaining_fraction = float(np.exp(min(log_fraction, 0.0)))

        finite = np.isfinite(values)
        with np.errstate(over="ignore"):
            centered = values - float(np.max(values[finite]))
        log_sum = float(logsumexp(centered))
        log_sum_squared = float(logsumexp(2.0 * centered))
        live_ess = float(np.exp(2.0 * log_sum - log_sum_squared))
        live_ess = float(np.clip(live_ess, 1.0, n_live))
        if np.all(finite) and np.all(values == values[0]):
            live_ess = float(n_live)
        variance_ratio = max(
            (n_live / live_ess - 1.0) / (n_live - 1),
            0.0,
        )
        live_mean_rse = float(np.sqrt(variance_ratio))
        live_logz_error = remaining_fraction * live_mean_rse

    history = np.asarray(logz_history, dtype=float)
    if history.ndim != 1:
        raise ValueError("logz_history must be one-dimensional")
    if np.any(~np.isfinite(history)):
        raise NumericalInvariantError("logz_history must contain only finite values")
    if len(history) < stability_window:
        stability = float("nan")
    else:
        window = history[-stability_window:]
        with np.errstate(over="ignore"):
            stability = float(np.max(window) - np.min(window))
        if not np.isfinite(stability):
            raise NumericalInvariantError(
                "logz stability range is not representable as a finite value"
            )

    return StoppingMetrics(
        remaining_fraction=remaining_fraction,
        remaining_dlogz=remaining_dlogz,
        live_ess=live_ess,
        live_mean_rse=live_mean_rse,
        live_logz_error=live_logz_error,
        logz_stability=stability,
        logzerr=checked_logzerr,
    )


def evaluate_stopping_policy(
    *,
    metrics: StoppingMetrics,
    policy: StoppingPolicy,
    niter: int,
    previous_streak: int,
) -> StoppingDecision:
    """Evaluate one policy and return its updated run-local streak."""
    if (
        isinstance(niter, bool)
        or not isinstance(niter, int)
        or niter < 0
        or isinstance(previous_streak, bool)
        or not isinstance(previous_streak, int)
        or previous_streak < 0
    ):
        raise ValueError("niter and previous_streak must be non-negative integers")

    evaluations: list[CriterionEvaluation] = []
    for criterion in policy.criteria:
        value = float(getattr(metrics, criterion.name))
        if not np.isfinite(value):
            if (criterion.name == "remaining_dlogz" and np.isposinf(value)) or (
                criterion.name == "logz_stability" and np.isnan(value)
            ):
                met = False
            else:
                raise NumericalInvariantError(
                    f"{criterion.name} stopping metric must be finite"
                )
        elif criterion.name in LESS_THAN_OR_EQUAL:
            met = value <= criterion.tolerance
        elif criterion.name in GREATER_THAN_OR_EQUAL:
            met = value >= criterion.tolerance
        else:  # pragma: no cover - policy validation makes this unreachable
            raise NumericalInvariantError(
                f"no comparison direction for criterion {criterion.name!r}"
            )
        evaluations.append(
            CriterionEvaluation(
                name=criterion.name,
                value=value,
                tolerance=criterion.tolerance,
                met=met,
            )
        )

    if policy.mode == "all":
        criteria_met = all(evaluation.met for evaluation in evaluations)
    else:
        criteria_met = any(evaluation.met for evaluation in evaluations)
    combined_met = niter >= policy.min_iterations and criteria_met
    streak = previous_streak + 1 if combined_met else 0
    return StoppingDecision(
        evaluations=tuple(evaluations),
        combined_met=combined_met,
        streak=streak,
        should_stop=streak >= policy.consecutive,
    )
