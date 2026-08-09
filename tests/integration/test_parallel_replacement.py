from __future__ import annotations

import time

import numpy as np
import pytest
from numpy.typing import NDArray
from tests.helpers import StandardNormalProposal

from nismo import (
    EnsembleRWalkSettings,
    NISMOSampler,
    ParallelSettings,
    RWalkSettings,
    SRWalkSettings,
)

pytestmark = pytest.mark.integration


class ConstantNormalModel:
    ndim = 1
    parameter_names = ("x",)

    def log_likelihood(
        self,
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return np.full(len(theta), np.log(2.5))

    def log_prior(
        self,
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return StandardNormalProposal().log_prob(theta)


class DelayedConstantNormalModel(ConstantNormalModel):
    def log_likelihood(
        self,
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        time.sleep(0.002 * len(theta))
        return super().log_likelihood(theta)


@pytest.mark.parametrize(
    ("proposal_scheme", "settings"),
    [
        ("rwalk", {"rwalk_settings": RWalkSettings(walks=4)}),
        ("s-rwalk", {"srwalk_settings": SRWalkSettings(n_steps=4)}),
        (
            "en-rwalk",
            {
                "ensemble_rwalk_settings": EnsembleRWalkSettings(
                    n_walkers=4,
                    n_sweeps=1,
                )
            },
        ),
    ],
)
def test_parallel_replacement_jobs_are_reproducible_and_serially_consumed(
    proposal_scheme: str,
    settings: dict[str, object],
) -> None:
    results = [
        NISMOSampler(
            model=ConstantNormalModel(),
            importance_morph=StandardNormalProposal(),
            proposal_scheme=proposal_scheme,  # type: ignore[arg-type]
            n_live=10,
            rng=88,
            tie_policy="randomized_plateau",
            parallel=ParallelSettings(n_workers=2, queue_size=4),
            **settings,  # type: ignore[arg-type]
        ).run(
            dlogz=0.5,
            max_iterations=100,
            max_proposals_per_replacement=20,
        )
        for _ in range(2)
    ]
    first, second = results
    assert first.success
    assert first.niter == 10
    assert len(first.dead_points) == first.niter
    assert np.all(np.diff(first.dead_log_psi0) >= 0.0)
    np.testing.assert_array_equal(first.dead_points, second.dead_points)
    np.testing.assert_array_equal(
        first.dead_tie_breakers,
        second.dead_tie_breakers,
    )
    assert first.queue_diagnostics == second.queue_diagnostics
    diagnostics = first.queue_diagnostics
    assert diagnostics.queue_jobs_submitted == diagnostics.queue_jobs_completed
    assert diagnostics.queue_candidates_consumed == first.niter
    assert first.n_likelihood_calls == (
        first.nlive + diagnostics.prefetch_likelihood_calls
    )
    assert diagnostics.wasted_prefetch_likelihood_calls == (
        diagnostics.prefetch_likelihood_calls
        - diagnostics.used_prefetch_likelihood_calls
    )


def test_serial_queue_discards_stale_candidates_in_fifo_order() -> None:
    result = NISMOSampler(
        model=ConstantNormalModel(),
        importance_morph=StandardNormalProposal(),
        proposal_scheme="rwalk",
        rwalk_settings=RWalkSettings(walks=4),
        n_live=10,
        rng=88,
        tie_policy="randomized_plateau",
        parallel=ParallelSettings(n_workers=1, queue_size=4),
    ).run(
        dlogz=0.5,
        max_iterations=100,
        max_proposals_per_replacement=20,
    )
    diagnostics = result.queue_diagnostics
    assert diagnostics.queue_candidates_stale > 0
    assert diagnostics.queue_candidates_consumed == result.niter
    assert diagnostics.queue_efficiency < 1.0
    assert np.all(np.diff(result.dead_tie_breakers) >= 0.0)


def test_explicit_singleton_queue_preserves_default_serial_stream() -> None:
    kwargs = {
        "model": ConstantNormalModel(),
        "importance_morph": StandardNormalProposal(),
        "proposal_scheme": "rwalk",
        "rwalk_settings": RWalkSettings(walks=4),
        "n_live": 10,
        "rng": 144,
        "tie_policy": "randomized_plateau",
    }
    default = NISMOSampler(**kwargs).run(  # type: ignore[arg-type]
        dlogz=0.5,
        max_iterations=100,
        max_proposals_per_replacement=20,
    )
    explicit = NISMOSampler(
        **kwargs,  # type: ignore[arg-type]
        parallel=ParallelSettings(n_workers=1, queue_size=1),
    ).run(
        dlogz=0.5,
        max_iterations=100,
        max_proposals_per_replacement=20,
    )
    assert default.logz == explicit.logz
    assert default.rng_state_final == explicit.rng_state_final
    assert default.n_likelihood_calls == explicit.n_likelihood_calls
    np.testing.assert_array_equal(default.dead_points, explicit.dead_points)
    np.testing.assert_array_equal(
        default.dead_tie_breakers,
        explicit.dead_tie_breakers,
    )


def test_parallel_call_reservations_never_overshoot_hard_limit() -> None:
    result = NISMOSampler(
        model=ConstantNormalModel(),
        importance_morph=StandardNormalProposal(),
        proposal_scheme="rwalk",
        rwalk_settings=RWalkSettings(walks=4),
        n_live=10,
        rng=89,
        tie_policy="randomized_plateau",
        parallel=ParallelSettings(n_workers=2, queue_size=4),
    ).run(
        dlogz=0.01,
        max_iterations=100,
        max_likelihood_calls=23,
        max_proposals_per_replacement=20,
    )
    assert result.termination_reason == "max_likelihood_calls"
    assert result.n_likelihood_calls == 22
    assert result.n_likelihood_calls <= 23
    assert result.niter == 3


def test_parallel_deadline_discards_prefetch_without_committing_late_result() -> None:
    result = NISMOSampler(
        model=DelayedConstantNormalModel(),
        importance_morph=StandardNormalProposal(),
        proposal_scheme="rwalk",
        rwalk_settings=RWalkSettings(walks=4),
        n_live=10,
        rng=90,
        tie_policy="randomized_plateau",
        parallel=ParallelSettings(n_workers=2, queue_size=2),
    ).run(
        dlogz=0.01,
        max_iterations=100,
        max_wall_time=0.03,
        max_proposals_per_replacement=20,
    )
    assert result.termination_reason == "max_wall_time"
    assert result.niter == 0
