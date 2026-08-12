"""Model protocol and validated callable adapter."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .exceptions import InvalidModelOutput


@runtime_checkable
class Model(Protocol):
    """A normalized-prior Bayesian model evaluated on batches."""

    ndim: int
    parameter_names: Sequence[str]

    def log_likelihood(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate log likelihood for points shaped ``(n, ndim)``."""

    def log_prior(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate normalized log prior for points shaped ``(n, ndim)``."""


LogDensityCallable = Callable[[NDArray[np.float64]], ArrayLike]
ScalarLikelihoodMap = Callable[
    [LogDensityCallable, NDArray[np.float64]],
    Iterable[ArrayLike],
]


@dataclass(frozen=True, slots=True)
class CallableModel:
    """Adapt explicit scalar or vectorized callables to :class:`Model`.

    Parameters
    ----------
    ndim
        Parameter dimension.
    parameter_names
        Unique parameter names in column order.
    log_likelihood_fn, log_prior_fn
        Callables returning log densities. Vectorized callables receive an
        ``(n, ndim)`` array. Scalar callables receive one ``(ndim,)`` row.
    vectorized
        Whether both callables accept batches.
    scalar_likelihood_map
        Optional ordered mapper used only when ``vectorized=False`` for
        likelihood evaluations.  It receives the scalar likelihood callable
        and an ``(n, ndim)`` array, and must yield one result per input row in
        the same order.  A persistent process-pool ``map`` method can be
        supplied for expensive, picklable likelihoods; priors remain local.

    Notes
    -----
    ``log_prior_fn`` must include every normalization constant.
    """

    ndim: int
    parameter_names: tuple[str, ...]
    log_likelihood_fn: LogDensityCallable
    log_prior_fn: LogDensityCallable
    vectorized: bool = True
    scalar_likelihood_map: ScalarLikelihoodMap | None = None

    def __post_init__(self) -> None:
        if isinstance(self.ndim, bool) or self.ndim < 1:
            raise ValueError("ndim must be a positive integer")
        if len(self.parameter_names) != self.ndim:
            raise ValueError("parameter_names length must equal ndim")
        if len(set(self.parameter_names)) != self.ndim:
            raise ValueError("parameter_names must be unique")

    def _evaluate(
        self,
        function: LogDensityCallable,
        theta: NDArray[np.float64],
        name: str,
        scalar_map: ScalarLikelihoodMap | None = None,
    ) -> NDArray[np.float64]:
        points = validate_points(theta, self.ndim)
        return self._evaluate_validated(function, points, name, scalar_map)

    def _evaluate_validated(
        self,
        function: LogDensityCallable,
        points: NDArray[np.float64],
        name: str,
        scalar_map: ScalarLikelihoodMap | None = None,
    ) -> NDArray[np.float64]:
        """Evaluate an already validated ``(n, ndim)`` array.

        This internal path lets :class:`nismo.constrained.BatchEvaluator`
        validate proposal coordinates once instead of repeating the same
        finite-shape scan for the prior and likelihood.
        """
        if self.vectorized:
            values = np.asarray(function(points), dtype=float)
        elif scalar_map is None:
            values = np.asarray([function(row) for row in points], dtype=float)
        else:
            values = np.asarray(list(scalar_map(function, points)), dtype=float)
        if values.shape == () and len(points) == 1:
            values = values.reshape(1)
        if values.shape != (len(points),):
            raise InvalidModelOutput(
                f"{name} must return shape ({len(points)},), got {values.shape}"
            )
        return values

    def _log_likelihood_validated(
        self,
        points: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return self._evaluate_validated(
            self.log_likelihood_fn,
            points,
            "log_likelihood",
            self.scalar_likelihood_map,
        )

    def _log_prior_validated(
        self,
        points: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return self._evaluate_validated(self.log_prior_fn, points, "log_prior")

    def log_likelihood(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return log-likelihood values with shape ``(n,)``."""
        return self._evaluate(
            self.log_likelihood_fn,
            theta,
            "log_likelihood",
            self.scalar_likelihood_map,
        )

    def log_prior(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return normalized log-prior values with shape ``(n,)``."""
        return self._evaluate(self.log_prior_fn, theta, "log_prior")


def validate_points(theta: ArrayLike, ndim: int) -> NDArray[np.float64]:
    """Return a finite-shape floating point array of model points.

    Parameters
    ----------
    theta
        Points with shape ``(n, ndim)`` or one point with shape ``(ndim,)``.
    ndim
        Expected final dimension.

    Returns
    -------
    numpy.ndarray
        A two-dimensional array with shape ``(n, ndim)``.
    """
    points = np.asarray(theta, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.ndim != 2 or points.shape[1] != ndim:
        raise InvalidModelOutput(
            f"theta must have shape (n, {ndim}), got {points.shape}"
        )
    if not np.all(np.isfinite(points)):
        raise InvalidModelOutput("theta contains NaN or infinity")
    return points
