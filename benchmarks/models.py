"""Analytic and regression targets kept outside the sampler implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

from nismo import CallableModel


@dataclass(frozen=True, slots=True)
class GaussianBenchmark:
    """Normalized Gaussian prior and likelihood with analytic evidence."""

    prior_scale: float = 2.0
    likelihood_scale: float = 1.0

    @property
    def logz(self) -> float:
        return float(
            -0.5
            * np.log(2.0 * np.pi * (self.prior_scale**2 + self.likelihood_scale**2))
        )

    @property
    def posterior_variance(self) -> float:
        return float(1.0 / (1.0 / self.prior_scale**2 + 1.0 / self.likelihood_scale**2))

    def model(self) -> CallableModel:
        normal_constant = -0.5 * np.log(2.0 * np.pi)
        prior_scale = self.prior_scale
        likelihood_scale = self.likelihood_scale
        return CallableModel(
            ndim=1,
            parameter_names=("x",),
            log_likelihood_fn=lambda theta: (
                normal_constant
                - np.log(likelihood_scale)
                - 0.5 * (theta[:, 0] / likelihood_scale) ** 2
            ),
            log_prior_fn=lambda theta: (
                normal_constant
                - np.log(prior_scale)
                - 0.5 * (theta[:, 0] / prior_scale) ** 2
            ),
        )

    def posterior_samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(
            scale=np.sqrt(self.posterior_variance),
            size=(n, 1),
        )


@dataclass(frozen=True, slots=True)
class PeakPlateauBenchmark:
    """Piecewise-constant target under a normalized uniform prior."""

    inner_half_width: float = 0.25
    inner_likelihood: float = 2.0
    outer_likelihood: float = 0.5

    @property
    def evidence(self) -> float:
        inner_prior_mass = self.inner_half_width
        return (
            inner_prior_mass * self.inner_likelihood
            + (1.0 - inner_prior_mass) * self.outer_likelihood
        )

    @property
    def logz(self) -> float:
        return float(np.log(self.evidence))

    def model(self) -> CallableModel:
        half_width = self.inner_half_width
        inner = np.log(self.inner_likelihood)
        outer = np.log(self.outer_likelihood)

        def log_prior(theta: np.ndarray) -> np.ndarray:
            inside = np.abs(theta[:, 0]) <= 1.0
            return np.where(inside, -np.log(2.0), -np.inf)

        def log_likelihood(theta: np.ndarray) -> np.ndarray:
            return np.where(np.abs(theta[:, 0]) <= half_width, inner, outer)

        return CallableModel(
            ndim=1,
            parameter_names=("x",),
            log_likelihood_fn=log_likelihood,
            log_prior_fn=log_prior,
        )


class UniformBoxProposal:
    """Normalized one-dimensional uniform density on ``[-1, 1]``."""

    ndim = 1

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(-1.0, 1.0, size=(n, 1))

    def log_prob(self, theta: np.ndarray) -> np.ndarray:
        inside = np.abs(theta[:, 0]) <= 1.0
        return np.where(inside, -np.log(2.0), -np.inf)


@dataclass(frozen=True, slots=True)
class GaussianShellBenchmark:
    """Two-dimensional Gaussian prior with a radial shell likelihood."""

    radius: float = 2.0
    shell_scale: float = 0.2

    @property
    def logz(self) -> float:
        radius = self.radius
        shell_scale = self.shell_scale

        def radial_integrand(value: float) -> float:
            return value * np.exp(
                -0.5 * value**2 - 0.5 * ((value - radius) / shell_scale) ** 2
            )

        evidence, _ = quad(radial_integrand, 0.0, np.inf, epsabs=1e-12)
        return float(np.log(evidence))

    def model(self) -> CallableModel:
        radius = self.radius
        shell_scale = self.shell_scale

        def log_likelihood(theta: np.ndarray) -> np.ndarray:
            radial = np.linalg.norm(theta, axis=1)
            return -0.5 * ((radial - radius) / shell_scale) ** 2

        def log_prior(theta: np.ndarray) -> np.ndarray:
            return -np.log(2.0 * np.pi) - 0.5 * np.sum(theta**2, axis=1)

        return CallableModel(
            ndim=2,
            parameter_names=("x", "y"),
            log_likelihood_fn=log_likelihood,
            log_prior_fn=log_prior,
        )

    def approximate_posterior_samples(
        self,
        n: int,
        rng: np.random.Generator,
        *,
        candidate_count: int = 20_000,
    ) -> np.ndarray:
        """Create deterministic-seed weighted-resampling training data."""
        candidates = rng.normal(size=(candidate_count, 2))
        radius = np.linalg.norm(candidates, axis=1)
        log_weights = -0.5 * ((radius - self.radius) / self.shell_scale) ** 2
        weights = np.exp(log_weights - np.max(log_weights))
        weights /= np.sum(weights)
        indices = rng.choice(candidate_count, size=n, replace=True, p=weights)
        return candidates[indices]
