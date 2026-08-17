from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from nismo import (
    EnsembleRWalkSettings,
    NISMOSampler,
    SRWalkSettings,
)

pytestmark = pytest.mark.statistical


class StandardNormalProposal:
    ndim = 1

    def sample(
        self,
        n: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        return rng.normal(size=(n, 1))

    def log_prob(
        self,
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return -0.5 * theta[:, 0] ** 2 - 0.5 * np.log(2.0 * np.pi)


class ShiftedGaussianModel:
    ndim = 1
    parameter_names = ("x",)
    exact_logz = -0.25 - 0.5 * np.log(2.0)

    def log_likelihood(
        self,
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return -0.5 * (theta[:, 0] - 1.0) ** 2

    def log_prior(
        self,
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return StandardNormalProposal().log_prob(theta)


@pytest.mark.parametrize(
    ("proposal_scheme", "settings"),
    [
        ("s-rwalk", {"srwalk_settings": SRWalkSettings(n_steps=12)}),
        (
            "en-rwalk",
            {
                "ensemble_rwalk_settings": EnsembleRWalkSettings(
                    n_walkers=6,
                    n_sweeps=2,
                )
            },
        ),
    ],
)
def test_parallel_and_singleton_queues_have_equivalent_gaussian_aggregates(
    proposal_scheme: str,
    settings: dict[str, object],
) -> None:
    means: list[float] = []
    for n_workers, queue_size in ((1, 1), (2, 3)):
        estimates = []
        for seed in (311, 312, 313):
            result = NISMOSampler(
                model=ShiftedGaussianModel(),
                importance_morph=StandardNormalProposal(),
                proposal_scheme=proposal_scheme,  # type: ignore[arg-type]
                n_live=30,
                rng=seed,
                n_workers=n_workers,
                queue_size=queue_size,
                **settings,  # type: ignore[arg-type]
            ).run(
                dlogz=0.12,
                max_iterations=400,
                max_proposals_per_replacement=5_000,
            )
            assert result.success
            assert np.all(np.diff(result.dead_log_psi0) >= 0.0)
            assert np.sum(result.posterior_weights) == pytest.approx(1.0)
            estimates.append(result.logz)
        mean = float(np.mean(estimates))
        means.append(mean)
        assert mean == pytest.approx(ShiftedGaussianModel.exact_logz, abs=0.12)
    assert means[0] == pytest.approx(means[1], abs=0.10)
