from __future__ import annotations

import time

import numpy as np
import pytest
from tests.helpers import StandardNormalProposal, UniformProposal
from tests.integration.test_parallel_replacement import (
    ConstantNormalModel,
    ConstantUniformModel,
)

from nismo.batch import evolve_srwalk_batch
from nismo.constrained import BatchEvaluator, EvaluatedPoint
from nismo.exceptions import InvalidModelOutput, ProposalSupportError
from nismo.mcmc import evolve_srwalk_constrained

pytestmark = pytest.mark.unit


def starts(evaluator, coordinates):
    batch = evaluator.evaluate(np.array(coordinates).reshape(-1, 1))
    return [
        EvaluatedPoint(
            batch.theta[i],
            batch.log_likelihood[i],
            batch.log_prior[i],
            batch.log_q0[i],
            batch.log_psi0[i],
            0.9,
        )
        for i in range(len(batch.theta))
    ]


@pytest.mark.parametrize("beta", [1.0, 0.8, 0.5])
@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("scale", [0.3, 1e12])
def test_batch_matches_scalar_streams_counts_and_self_transitions(beta, uniform, scale):
    model = ConstantUniformModel() if uniform else ConstantNormalModel()
    proposal = UniformProposal() if uniform else StandardNormalProposal()
    evaluator = BatchEvaluator(model, proposal, beta=beta, log_z_beta=0.2)
    initial = starts(evaluator, [0.1, 0.2, 0.3, 0.4, 0.5])
    settings = dict(
        threshold=-2.0,
        threshold_tie_breaker=0.2,
        tie_policy="randomized_plateau",
        proposal_factor=np.eye(1),
        scale=scale,
        n_steps=15,
        max_proposals=15,
        max_likelihood_calls=None,
        deadline=None,
        zero_move_policy="allow",
    )
    seeds = [10, 11, 12, 13, 14]
    result = evolve_srwalk_batch(
        evaluator=evaluator,
        starts=initial,
        rngs=[np.random.default_rng(seed) for seed in seeds],
        **settings,
    )
    for i, seed in enumerate(seeds):
        reference = BatchEvaluator(model, proposal, beta=beta, log_z_beta=0.2)
        scalar = evolve_srwalk_constrained(
            evaluator=reference,
            starting=initial[i],
            rng=np.random.default_rng(seed),
            **settings,
        )
        actual = result.attempts[i]
        assert actual.draw is not None and scalar.draw is not None
        np.testing.assert_array_equal(actual.draw.point.theta, scalar.draw.point.theta)
        assert actual.draw.point.tie_breaker == scalar.draw.point.tie_breaker
        assert (
            actual.n_proposed,
            actual.n_valid,
            actual.n_accepted,
            actual.n_completed,
        ) == (scalar.n_proposed, scalar.n_valid, scalar.n_accepted, scalar.n_completed)
        assert result.likelihood_calls[i] == reference.n_likelihood_calls
        assert result.outside_prior[i] == reference.outside_prior


def test_plateau_ties_and_batch_layout_do_not_change_streams():
    evaluator = BatchEvaluator(ConstantUniformModel(), UniformProposal())
    initial = starts(evaluator, [0.2, 0.3, 0.4])
    kwargs = dict(
        threshold=np.log(2.5),
        threshold_tie_breaker=0.7,
        tie_policy="randomized_plateau",
        proposal_factor=np.eye(1),
        scale=0.5,
        n_steps=40,
        max_proposals=40,
    )
    batch = evolve_srwalk_batch(
        evaluator=evaluator,
        starts=initial,
        rngs=[np.random.default_rng(i) for i in range(3)],
        **kwargs,
    )
    for i in range(3):
        single = evolve_srwalk_batch(
            evaluator=evaluator,
            starts=initial[i : i + 1],
            rngs=[np.random.default_rng(i)],
            **kwargs,
        )
        np.testing.assert_array_equal(
            batch.attempts[i].draw.point.theta, single.attempts[0].draw.point.theta
        )
        assert batch.attempts[i].draw.point.tie_breaker > 0.7


@pytest.mark.parametrize("limit", ["calls", "proposals", "deadline"])
def test_batch_reserves_complete_chains_before_evaluation(limit):
    evaluator = BatchEvaluator(ConstantUniformModel(), UniformProposal())
    initial = starts(evaluator, [0.2, 0.3])
    before = evaluator.n_likelihood_calls
    result = evolve_srwalk_batch(
        evaluator=evaluator,
        starts=initial,
        rngs=[np.random.default_rng(1), np.random.default_rng(2)],
        threshold=0,
        threshold_tie_breaker=0,
        tie_policy="strict",
        proposal_factor=np.eye(1),
        scale=0.3,
        n_steps=10,
        max_proposals=9 if limit == "proposals" else 10,
        max_likelihood_calls=before + 19 if limit == "calls" else None,
        deadline=time.monotonic() - 1 if limit == "deadline" else None,
    )
    assert evaluator.n_likelihood_calls == before
    assert all(a.draw is None and a.n_completed == 0 for a in result.attempts)


def test_interruption_discards_partially_evolved_chains():
    class SlowModel(ConstantUniformModel):
        delay = False

        def log_likelihood(self, theta):
            if self.delay:
                time.sleep(0.03)
            return super().log_likelihood(theta)

    model = SlowModel()
    evaluator = BatchEvaluator(model, UniformProposal())
    initial = starts(evaluator, [0.5, 0.5])
    model.delay = True
    result = evolve_srwalk_batch(
        evaluator=evaluator,
        starts=initial,
        rngs=[np.random.default_rng(1), np.random.default_rng(2)],
        threshold=0,
        threshold_tie_breaker=0,
        tie_policy="strict",
        proposal_factor=np.eye(1),
        scale=0.001,
        n_steps=10,
        max_proposals=10,
        deadline=time.monotonic() + 0.02,
    )
    assert all(a.draw is None and a.n_completed == 1 for a in result.attempts)
    assert sum(result.likelihood_calls) == 2


def test_batch_preserves_invalid_output_and_support_errors():
    class BrokenProposal(StandardNormalProposal):
        broken = False

        def log_prob(self, theta):
            return (
                np.full(len(theta), -np.inf) if self.broken else super().log_prob(theta)
            )

    proposal = BrokenProposal()
    evaluator = BatchEvaluator(ConstantNormalModel(), proposal)
    initial = starts(evaluator, [0.2, 0.3])
    proposal.broken = True
    kwargs = dict(
        evaluator=evaluator,
        starts=initial,
        rngs=[np.random.default_rng(1), np.random.default_rng(2)],
        threshold=-10,
        threshold_tie_breaker=0,
        tie_policy="strict",
        proposal_factor=np.eye(1),
        scale=0.01,
        n_steps=2,
        max_proposals=2,
    )
    with pytest.raises(ProposalSupportError):
        evolve_srwalk_batch(**kwargs)
    proposal.broken = False
    evaluator.model.log_likelihood = lambda theta: np.full(len(theta), np.nan)
    with pytest.raises(InvalidModelOutput):
        evolve_srwalk_batch(**kwargs)
