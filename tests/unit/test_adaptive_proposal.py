from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
from numpy.typing import NDArray

from nismo import CallableModel, NISMOSampler
from nismo.adaptive import AdaptiveMorphController

pytestmark = pytest.mark.unit


@dataclass
class RefitState:
    inputs: list[NDArray[np.float64]] = field(default_factory=list)
    fail_calls: set[int] = field(default_factory=set)


class TrackingNormalProposal:
    ndim = 1

    def __init__(
        self,
        *,
        location: float = 0.0,
        scale: float = 1.0,
        state: RefitState | None = None,
    ) -> None:
        self.location = location
        self.scale = scale
        self.state = RefitState() if state is None else state
        self.metadata = (location, scale)

    def sample(
        self,
        n: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        return rng.normal(self.location, self.scale, size=(n, 1))

    def log_prob(
        self,
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        points = np.asarray(theta)
        standardized = (points[:, 0] - self.location) / self.scale
        return -0.5 * standardized**2 - np.log(self.scale) - 0.5 * np.log(2.0 * np.pi)

    def refit(
        self,
        training_theta: NDArray[np.float64],
    ) -> TrackingNormalProposal:
        copied = np.array(training_theta, copy=True)
        self.state.inputs.append(copied)
        call = len(self.state.inputs)
        if call in self.state.fail_calls:
            raise np.linalg.LinAlgError("singular live-point covariance")
        return TrackingNormalProposal(
            location=float(np.mean(copied[:, 0])),
            scale=max(float(np.std(copied[:, 0])), 0.2),
            state=self.state,
        )


def test_controller_refits_on_boundaries_and_retains_previous_on_failure() -> None:
    state = RefitState(fail_calls={2})
    importance = TrackingNormalProposal(location=4.0, scale=2.0, state=state)
    controller = AdaptiveMorphController(
        importance_morph=importance,
        update_interval=25,
    )
    live = np.arange(6.0).reshape(-1, 1)

    assert controller.update_if_due(iteration=24, live_theta=live) is importance
    first = controller.update_if_due(iteration=25, live_theta=live)
    assert first is not importance
    assert controller.revision == 1
    assert len(state.inputs) == 1
    np.testing.assert_array_equal(state.inputs[0], live)
    live[:] = -99.0
    assert not np.any(state.inputs[0] == -99.0)

    retained = controller.update_if_due(iteration=50, live_theta=live)
    assert retained is first
    assert controller.revision == 1
    assert controller.update_failures == 1
    assert controller.records[1].iteration == 50
    assert not controller.records[1].success
    assert controller.records[1].error_type == "LinAlgError"

    third = controller.update_if_due(iteration=75, live_theta=live)
    assert third is not retained
    assert controller.revision == 2
    assert [record.iteration for record in controller.records] == [25, 50, 75]


def test_adaptive_sampler_keeps_q0_fixed_and_refits_from_live_rows_only() -> None:
    state = RefitState()
    importance = TrackingNormalProposal(state=state)
    original_parameters = (importance.location, importance.scale)
    model = CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=lambda x: -0.5 * (x[:, 0] - 1.5) ** 2,
        log_prior_fn=importance.log_prob,
    )
    result = NISMOSampler(
        model=model,
        importance_morph=importance,
        proposal_scheme="adaptive_morph",
        proposal_update_interval=5,
        n_live=10,
        rng=17,
        proposal_batch_size=8,
    ).run(
        dlogz=1.0e-8,
        max_iterations=12,
        max_proposals_per_replacement=10_000,
    )

    assert result.termination_reason == "max_iterations"
    assert [record.iteration for record in result.proposal_updates] == [5, 10]
    assert all(record.success for record in result.proposal_updates)
    assert [values.shape for values in state.inputs] == [(10, 1), (10, 1)]
    assert (importance.location, importance.scale) == original_parameters
    points = result.all_points
    expected_log_q0 = importance.log_prob(points)
    expected_log_psi0 = (
        model.log_likelihood(points) + model.log_prior(points) - expected_log_q0
    )
    np.testing.assert_allclose(
        np.concatenate((result.dead_log_q0, result.final_live_log_q0)),
        expected_log_q0,
    )
    np.testing.assert_allclose(result.all_log_psi0, expected_log_psi0)
    assert result.history.proposal_revision.tolist() == [
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
    ]


def test_adaptive_sampler_continues_after_refit_failure() -> None:
    state = RefitState(fail_calls={1})
    importance = TrackingNormalProposal(state=state)
    model = CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=lambda x: -0.5 * (x[:, 0] - 0.5) ** 2,
        log_prior_fn=importance.log_prob,
    )
    result = NISMOSampler(
        model=model,
        importance_morph=importance,
        proposal_scheme="adaptive_morph",
        proposal_update_interval=3,
        n_live=8,
        rng=29,
    ).run(dlogz=1.0e-8, max_iterations=4)

    assert result.niter == 4
    assert len(result.proposal_updates) == 1
    assert not result.proposal_updates[0].success
    assert result.proposal_updates[0].active_revision == 0
    assert result.history.proposal_update_failures.tolist() == [0, 0, 0, 1]


def test_run_stopping_at_update_boundary_does_not_refit() -> None:
    state = RefitState()
    importance = TrackingNormalProposal(state=state)
    model = CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=lambda x: -0.5 * x[:, 0] ** 2,
        log_prior_fn=importance.log_prob,
    )
    result = NISMOSampler(
        model=model,
        importance_morph=importance,
        proposal_scheme="adaptive_morph",
        proposal_update_interval=3,
        n_live=8,
        rng=31,
    ).run(dlogz=1.0e-8, max_iterations=3)

    assert result.niter == 3
    assert result.proposal_updates == ()
    assert state.inputs == []


def test_serial_prefetch_epochs_end_at_adaptive_refit_boundaries() -> None:
    state = RefitState()
    importance = TrackingNormalProposal(state=state)
    model = CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=lambda x: -0.5 * (x[:, 0] - 1.5) ** 2,
        log_prior_fn=importance.log_prob,
    )
    result = NISMOSampler(
        model=model,
        importance_morph=importance,
        proposal_scheme="adaptive_morph",
        proposal_update_interval=5,
        n_live=10,
        rng=17,
        proposal_batch_size=8,
        n_workers=1,
        queue_size=4,
    ).run(
        dlogz=1.0e-8,
        max_iterations=12,
        max_proposals_per_replacement=10_000,
    )

    assert [record.iteration for record in result.proposal_updates] == [5, 10]
    assert result.history.proposal_revision.tolist() == [
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
    ]
    assert result.queue_diagnostics.queue_candidates_invalidated == 0
