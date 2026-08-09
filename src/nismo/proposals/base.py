"""Normalized proposal protocol used by the sampler."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class Proposal(Protocol):
    """A normalized distribution supporting density evaluation and sampling."""

    ndim: int

    def sample(
        self,
        n: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Draw ``n`` independent points with shape ``(n, ndim)``."""

    def log_prob(
        self,
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return normalized log density for points shaped ``(n, ndim)``."""


@runtime_checkable
class RefittableProposal(Proposal, Protocol):
    """Normalized proposal supporting a non-mutating fit to new samples."""

    def refit(
        self,
        training_theta: NDArray[np.float64],
    ) -> Proposal:
        """Return a new fitted proposal without changing this instance."""
