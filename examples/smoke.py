"""Phase 1 deterministic wiring smoke test."""

from __future__ import annotations

import numpy as np

from nismo import CallableModel


def main() -> None:
    """Evaluate a tiny normalized model with an explicit generator."""
    rng = np.random.default_rng(20260725)
    model = CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=lambda theta: -0.5 * theta[:, 0] ** 2,
        log_prior_fn=lambda theta: np.full(len(theta), -np.log(2.0), dtype=float),
    )
    theta = rng.uniform(-1.0, 1.0, size=(4, 1))
    assert model.log_likelihood(theta).shape == (4,)
    assert model.log_prior(theta).shape == (4,)


if __name__ == "__main__":
    main()
