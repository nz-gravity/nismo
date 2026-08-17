"""Validated immutable sampler configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Literal

import numpy as np

from .exceptions import ConfigurationError
from .stopping import (
    StoppingCriterionConfig,
    StoppingPolicy,
    validate_stopping_policy_for_n_live,
)

TiePolicy = Literal["strict", "randomized_plateau"]
EnsembleMoveName = Literal["de", "stretch", "gaussian"]
ProposalScheme = Literal[
    "fixed_morph",
    "adaptive_morph",
    "mor-rwalk",
    "s-rwalk",
    "en-rwalk",
]


@dataclass(frozen=True, slots=True)
class ParallelSettings:
    """Replacement-prefetch worker and queue settings.

    ``queue_size`` defaults to ``n_workers``.  The all-serial compatibility
    path is therefore represented by the default ``(1, 1)`` settings.  A
    queue larger than one requires a process pool, so it also requires
    ``n_workers > 1``.
    """

    n_workers: int = 1
    queue_size: int | None = None

    def __post_init__(self) -> None:
        workers = _positive_integer(self.n_workers, name="parallel n_workers")
        queue_size = (
            workers
            if self.queue_size is None
            else _positive_integer(
                self.queue_size,
                name="parallel queue_size",
            )
        )
        object.__setattr__(self, "n_workers", workers)
        object.__setattr__(self, "queue_size", queue_size)
        if workers == 1 and queue_size > 1:
            raise ConfigurationError("parallel queue_size > 1 requires n_workers > 1")


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ConfigurationError(f"{name} must be a positive integer")
    return int(value)


def _positive_finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigurationError(f"{name} must be positive and finite")
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ConfigurationError(f"{name} must be positive and finite")
    return number


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigurationError(f"{name} must be finite")
    number = float(value)
    if not np.isfinite(number):
        raise ConfigurationError(f"{name} must be finite")
    return number


def _shrinkage(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigurationError(f"{name} must be finite and in [0, 1]")
    number = float(value)
    if not np.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ConfigurationError(f"{name} must be finite and in [0, 1]")
    return number


@dataclass(frozen=True, slots=True)
class EnsembleMoveWeights:
    """Relative weights for the ``en-rwalk`` proposal mixture."""

    de: float = 0.60
    stretch: float = 0.25
    gaussian: float = 0.15

    def __post_init__(self) -> None:
        values = []
        for name in ("de", "stretch", "gaussian"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ConfigurationError(
                    f"ensemble move weight {name} must be finite and non-negative"
                )
            number = float(value)
            if not np.isfinite(number) or number < 0.0:
                raise ConfigurationError(
                    f"ensemble move weight {name} must be finite and non-negative"
                )
            object.__setattr__(self, name, number)
            values.append(number)
        if not any(value > 0.0 for value in values):
            raise ConfigurationError(
                "at least one ensemble move weight must be strictly positive"
            )

    @property
    def active_names_and_probabilities(
        self,
    ) -> tuple[tuple[EnsembleMoveName, ...], tuple[float, ...]]:
        """Return active moves and normalized probabilities in canonical order."""
        weighted_names: tuple[tuple[EnsembleMoveName, float], ...] = (
            ("de", self.de),
            ("stretch", self.stretch),
            ("gaussian", self.gaussian),
        )
        active = tuple((name, weight) for name, weight in weighted_names if weight > 0)
        maximum = max(weight for _, weight in active)
        scaled = tuple((name, weight / maximum) for name, weight in active)
        total = sum(weight for _, weight in scaled)
        return (
            tuple(name for name, _ in scaled),
            tuple(weight / total for _, weight in scaled),
        )


@dataclass(frozen=True, slots=True)
class RWalkSettings:
    """Dynesty-style settings for the constrained ``q0`` random walk."""

    walks: int | None = None
    facc: float = 0.5
    ncdim: int | None = None

    def __post_init__(self) -> None:
        if self.walks is not None:
            object.__setattr__(
                self,
                "walks",
                max(2, _positive_integer(self.walks, name="rwalk walks")),
            )
        object.__setattr__(self, "facc", _finite(self.facc, name="rwalk facc"))
        if self.ncdim is not None:
            object.__setattr__(
                self,
                "ncdim",
                _positive_integer(self.ncdim, name="rwalk ncdim"),
            )


@dataclass(frozen=True, slots=True)
class SRWalkSettings:
    """Settings for the Gaussian-covariance constrained ``q0`` random walk."""

    n_steps: int = 75
    scale: float | None = None
    facc: float = 0.5
    dynamic_steps: bool = True
    max_steps: int = 5_000
    target_zero_move_probability: float = 1.0e-3
    acceptance_window: int = 20
    max_step_growth: float = 2.0
    zero_accept_scale_factor: float = 0.5
    zero_move_policy: Literal["allow", "stop"] = "allow"
    covariance_shrinkage: float = 0.25
    covariance_jitter: float = 1.0e-10
    covariance_update_interval: int = 1
    covariance_rebuild_interval: int | None = None
    profile: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "n_steps",
            _positive_integer(self.n_steps, name="s-rwalk n_steps"),
        )
        if self.scale is not None:
            object.__setattr__(
                self,
                "scale",
                _positive_finite(self.scale, name="s-rwalk scale"),
            )
        object.__setattr__(self, "facc", _finite(self.facc, name="s-rwalk facc"))
        if not isinstance(self.dynamic_steps, bool):
            raise ConfigurationError("s-rwalk dynamic_steps must be a boolean")
        max_steps = _positive_integer(self.max_steps, name="s-rwalk max_steps")
        if max_steps < self.n_steps:
            raise ConfigurationError("s-rwalk max_steps must be >= n_steps")
        object.__setattr__(self, "max_steps", max_steps)
        target = _positive_finite(
            self.target_zero_move_probability,
            name="s-rwalk target_zero_move_probability",
        )
        if target >= 1.0:
            raise ConfigurationError(
                "s-rwalk target_zero_move_probability must be less than one"
            )
        object.__setattr__(self, "target_zero_move_probability", target)
        object.__setattr__(
            self,
            "acceptance_window",
            _positive_integer(
                self.acceptance_window,
                name="s-rwalk acceptance_window",
            ),
        )
        growth = _positive_finite(
            self.max_step_growth,
            name="s-rwalk max_step_growth",
        )
        if growth < 1.0:
            raise ConfigurationError("s-rwalk max_step_growth must be >= 1")
        object.__setattr__(self, "max_step_growth", growth)
        zero_scale = _positive_finite(
            self.zero_accept_scale_factor,
            name="s-rwalk zero_accept_scale_factor",
        )
        if zero_scale > 1.0:
            raise ConfigurationError("s-rwalk zero_accept_scale_factor must be <= 1")
        object.__setattr__(self, "zero_accept_scale_factor", zero_scale)
        if self.zero_move_policy not in ("allow", "stop"):
            raise ConfigurationError(
                "s-rwalk zero_move_policy must be 'allow' or 'stop'"
            )
        object.__setattr__(
            self,
            "covariance_shrinkage",
            _shrinkage(
                self.covariance_shrinkage,
                name="s-rwalk covariance_shrinkage",
            ),
        )
        object.__setattr__(
            self,
            "covariance_jitter",
            _positive_finite(
                self.covariance_jitter,
                name="s-rwalk covariance_jitter",
            ),
        )
        object.__setattr__(
            self,
            "covariance_update_interval",
            _positive_integer(
                self.covariance_update_interval,
                name="s-rwalk covariance_update_interval",
            ),
        )
        if self.covariance_rebuild_interval is not None:
            object.__setattr__(
                self,
                "covariance_rebuild_interval",
                _positive_integer(
                    self.covariance_rebuild_interval,
                    name="s-rwalk covariance_rebuild_interval",
                ),
            )
        if not isinstance(self.profile, bool):
            raise ConfigurationError("s-rwalk profile must be a boolean")


@dataclass(frozen=True, slots=True)
class MORWalkSettings:
    """Settings for the Morph-pool then ``s-rwalk`` hybrid scheme."""

    n_proposals: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "n_proposals",
            _positive_integer(
                self.n_proposals,
                name="mor-rwalk n_proposals",
            ),
        )


@dataclass(frozen=True, slots=True)
class EnsembleRWalkSettings:
    """Settings for split-ensemble constrained Metropolis--Hastings moves."""

    n_walkers: int = 8
    n_sweeps: int = 4
    gamma: float | None = None
    jitter_scale: float = 1.0e-6
    covariance_shrinkage: float = 0.1
    covariance_jitter: float = 1.0e-10
    move_weights: EnsembleMoveWeights = field(default_factory=EnsembleMoveWeights)
    stretch_scale: float = 1.5
    gaussian_scale: float | None = None

    def __post_init__(self) -> None:
        n_walkers = _positive_integer(
            self.n_walkers,
            name="ensemble rwalk n_walkers",
        )
        if n_walkers < 4 or n_walkers % 2:
            raise ConfigurationError(
                "ensemble rwalk n_walkers must be an even integer >= 4"
            )
        object.__setattr__(self, "n_walkers", n_walkers)
        object.__setattr__(
            self,
            "n_sweeps",
            _positive_integer(self.n_sweeps, name="ensemble rwalk n_sweeps"),
        )
        if self.gamma is not None:
            object.__setattr__(
                self,
                "gamma",
                _positive_finite(self.gamma, name="ensemble rwalk gamma"),
            )
        if not isinstance(self.move_weights, EnsembleMoveWeights):
            raise ConfigurationError(
                "ensemble rwalk move_weights must be an EnsembleMoveWeights"
            )
        stretch_scale = _positive_finite(
            self.stretch_scale,
            name="ensemble rwalk stretch_scale",
        )
        if stretch_scale <= 1.0:
            raise ConfigurationError(
                "ensemble rwalk stretch_scale must be finite and greater than one"
            )
        object.__setattr__(self, "stretch_scale", stretch_scale)
        if self.gaussian_scale is not None:
            object.__setattr__(
                self,
                "gaussian_scale",
                _positive_finite(
                    self.gaussian_scale,
                    name="ensemble rwalk gaussian_scale",
                ),
            )
        object.__setattr__(
            self,
            "jitter_scale",
            _positive_finite(
                self.jitter_scale,
                name="ensemble rwalk jitter_scale",
            ),
        )
        object.__setattr__(
            self,
            "covariance_shrinkage",
            _shrinkage(
                self.covariance_shrinkage,
                name="ensemble rwalk covariance_shrinkage",
            ),
        )
        object.__setattr__(
            self,
            "covariance_jitter",
            _positive_finite(
                self.covariance_jitter,
                name="ensemble rwalk covariance_jitter",
            ),
        )


def _validate_dlogz(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigurationError("dlogz must be a positive finite number")
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ConfigurationError("dlogz must be a positive finite number")
    return number


@dataclass(frozen=True, slots=True)
class NISMOConfig:
    """Numerical and resource settings for one NISMO run.

    Parameters
    ----------
    n_live
        Number of live points. Must be at least two.
    dlogz
        Optional maximum estimated log-evidence increment from the current
        mean-live remaining-evidence estimate. Must be positive and finite.
    stopping
        Resolved immutable stopping policy. Supplying this together with
        ``dlogz`` is invalid.
    proposal_batch_size
        Number of independent Morph draws evaluated per rejection batch.
    proposal_scheme
        Fixed/adaptive Morph rejection or one of the constrained fixed-``q0``
        Metropolis replacement kernels.
    proposal_update_interval
        Completed-iteration interval between adaptive Morph refit attempts.
    srwalk_settings
        Gaussian-covariance random-walk Metropolis settings.
    mor_rwalk_settings
        Initial Morph-pool size for the ``mor-rwalk`` hybrid scheme.
    ensemble_rwalk_settings
        Split-ensemble Metropolis--Hastings mixture settings.
    parallel
        Complete-replacement worker count and deterministic prefetch depth.
    max_iterations
        Maximum number of completed dead-point replacements.
    max_proposals_per_replacement
        Maximum proposal points evaluated for one replacement.
    max_likelihood_calls
        Optional run-wide likelihood-evaluation limit.
    max_wall_time
        Optional run-wide wall-time limit in seconds.
    tie_policy
        Strict likelihood ordering or lexicographic randomized plateau ordering.
    """

    n_live: int
    dlogz: float | None = None
    stopping: StoppingPolicy | None = None
    proposal_batch_size: int = 64
    proposal_scheme: ProposalScheme = "fixed_morph"
    proposal_update_interval: int = 25
    rwalk_settings: RWalkSettings = field(
        default_factory=RWalkSettings,
        init=False,
        repr=False,
        compare=False,
    )
    srwalk_settings: SRWalkSettings = field(default_factory=SRWalkSettings)
    mor_rwalk_settings: MORWalkSettings | None = None
    ensemble_rwalk_settings: EnsembleRWalkSettings = field(
        default_factory=EnsembleRWalkSettings
    )
    parallel: ParallelSettings = field(default_factory=ParallelSettings)
    max_iterations: int = 10_000
    max_proposals_per_replacement: int = 100_000
    max_likelihood_calls: int | None = None
    max_wall_time: float | None = None
    tie_policy: TiePolicy = "strict"

    def __post_init__(self) -> None:
        """Validate all settings eagerly."""
        if (
            isinstance(self.n_live, bool)
            or not isinstance(self.n_live, int)
            or self.n_live < 2
        ):
            raise ConfigurationError("n_live must be an integer >= 2")
        if self.dlogz is not None and self.stopping is not None:
            raise ConfigurationError("dlogz and stopping cannot both be supplied")
        dlogz = self.dlogz
        stopping = self.stopping
        if dlogz is None and stopping is None:
            dlogz = 1.0e-3
        if dlogz is not None:
            dlogz = _validate_dlogz(dlogz)
            stopping = StoppingPolicy(
                criteria=(
                    StoppingCriterionConfig(
                        name="remaining_dlogz",
                        tolerance=dlogz,
                    ),
                )
            )
        elif not isinstance(stopping, StoppingPolicy):
            raise ConfigurationError("stopping must be a StoppingPolicy")
        if stopping is None:  # pragma: no cover - resolution above is exhaustive
            raise ConfigurationError("a stopping policy could not be resolved")
        validate_stopping_policy_for_n_live(stopping, self.n_live)
        object.__setattr__(self, "dlogz", dlogz)
        object.__setattr__(self, "stopping", stopping)
        for name in (
            "proposal_batch_size",
            "proposal_update_interval",
            "max_iterations",
            "max_proposals_per_replacement",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigurationError(f"{name} must be a positive integer")
        if self.proposal_scheme not in (
            "fixed_morph",
            "adaptive_morph",
            "mor-rwalk",
            "s-rwalk",
            "en-rwalk",
        ):
            raise ConfigurationError(
                f"unsupported proposal_scheme: {self.proposal_scheme!r}"
            )
        if not isinstance(self.rwalk_settings, RWalkSettings):
            raise ConfigurationError("rwalk_settings must be an RWalkSettings")
        if not isinstance(self.srwalk_settings, SRWalkSettings):
            raise ConfigurationError("srwalk_settings must be an SRWalkSettings")
        if self.mor_rwalk_settings is not None and not isinstance(
            self.mor_rwalk_settings,
            MORWalkSettings,
        ):
            raise ConfigurationError(
                "mor_rwalk_settings must be a MORWalkSettings or None"
            )
        if self.proposal_scheme == "mor-rwalk":
            if self.mor_rwalk_settings is None:
                raise ConfigurationError("mor-rwalk requires mor_rwalk_settings")
            if self.mor_rwalk_settings.n_proposals < self.n_live:
                raise ConfigurationError("mor-rwalk n_proposals must be >= n_live")
        if not isinstance(
            self.ensemble_rwalk_settings,
            EnsembleRWalkSettings,
        ):
            raise ConfigurationError(
                "ensemble_rwalk_settings must be an EnsembleRWalkSettings"
            )
        if not isinstance(self.parallel, ParallelSettings):
            raise ConfigurationError("parallel must be a ParallelSettings")
        if (
            self.proposal_scheme == "en-rwalk"
            and self.ensemble_rwalk_settings.n_walkers > self.n_live - 1
        ):
            raise ConfigurationError("ensemble rwalk n_walkers must be <= n_live - 1")
        if self.max_likelihood_calls is not None and (
            isinstance(self.max_likelihood_calls, bool)
            or not isinstance(self.max_likelihood_calls, int)
            or self.max_likelihood_calls < self.n_live
        ):
            raise ConfigurationError(
                "max_likelihood_calls must be an integer >= n_live"
            )
        if (
            self.proposal_scheme == "mor-rwalk"
            and self.max_likelihood_calls is not None
            and self.mor_rwalk_settings is not None
            and self.max_likelihood_calls < self.mor_rwalk_settings.n_proposals
        ):
            raise ConfigurationError(
                "max_likelihood_calls must be >= mor-rwalk n_proposals"
            )
        if self.max_wall_time is not None and (
            isinstance(self.max_wall_time, bool)
            or not isinstance(self.max_wall_time, (int, float))
            or self.max_wall_time <= 0.0
        ):
            raise ConfigurationError("max_wall_time must be positive")
        if self.tie_policy not in ("strict", "randomized_plateau"):
            raise ConfigurationError(f"unsupported tie_policy: {self.tie_policy!r}")
