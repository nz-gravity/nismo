from __future__ import annotations

import numpy as np
import pytest
from tests.helpers import UniformProposal

from nismo import (
    CallableModel,
    InvalidModelOutput,
    InvalidProposalOutput,
    ProposalSupportError,
)
from nismo.constrained import (
    BatchEvaluator,
    LikelihoodBudgetExhausted,
    draw_constrained,
)

pytestmark = pytest.mark.unit


def test_first_valid_constrained_draw_has_uniform_conditional_distribution() -> None:
    proposal = UniformProposal()
    model = CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=lambda x: np.log(x[:, 0]),
        log_prior_fn=lambda x: np.zeros(len(x)),
    )
    evaluator = BatchEvaluator(model, proposal)
    rng = np.random.default_rng(20260725)
    draws = []
    for _ in range(3_000):
        attempt = draw_constrained(
            evaluator=evaluator,
            proposal_morph=proposal,
            threshold=np.log(0.5),
            threshold_tie_breaker=0.0,
            tie_policy="strict",
            rng=rng,
            batch_size=16,
            max_proposals=128,
            max_likelihood_calls=None,
            deadline=None,
        )
        assert attempt.draw is not None
        draws.append(attempt.draw.point.theta[0])
    assert np.mean(draws) == pytest.approx(0.75, abs=0.01)
    assert min(draws) > 0.5
    assert max(draws) < 1.0


def test_constrained_draw_obeys_proposal_limit() -> None:
    proposal = UniformProposal()
    model = CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=lambda x: np.zeros(len(x)),
        log_prior_fn=lambda x: np.zeros(len(x)),
    )
    attempt = draw_constrained(
        evaluator=BatchEvaluator(model, proposal),
        proposal_morph=proposal,
        threshold=1.0,
        threshold_tie_breaker=0.0,
        tie_policy="strict",
        rng=np.random.default_rng(1),
        batch_size=8,
        max_proposals=19,
        max_likelihood_calls=None,
        deadline=None,
    )
    assert attempt.draw is None
    assert attempt.reason == "constrained_sampling_exhausted"
    assert attempt.n_proposed == 19


def test_evaluator_rejects_nan() -> None:
    proposal = UniformProposal()
    model = CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=lambda x: np.full(len(x), np.nan),
        log_prior_fn=lambda x: np.zeros(len(x)),
    )
    with pytest.raises(InvalidModelOutput, match="NaN"):
        BatchEvaluator(model, proposal).evaluate(np.array([[0.5]]))


def test_evaluator_rejects_positive_infinity() -> None:
    proposal = UniformProposal()
    model = CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=lambda x: np.full(len(x), np.inf),
        log_prior_fn=lambda x: np.zeros(len(x)),
    )
    with pytest.raises(InvalidModelOutput, match="infinity"):
        BatchEvaluator(model, proposal).evaluate(np.array([[0.5]]))


def test_evaluator_detects_proposal_support_failure() -> None:
    class EmptyProposal(UniformProposal):
        def log_prob(self, theta: np.ndarray) -> np.ndarray:
            return np.full(len(theta), -np.inf)

    proposal = EmptyProposal()
    model = CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=lambda x: np.zeros(len(x)),
        log_prior_fn=lambda x: np.zeros(len(x)),
    )
    with pytest.raises(ProposalSupportError, match="support failure"):
        BatchEvaluator(model, proposal).evaluate(np.array([[0.5]]))


def test_evaluator_short_circuits_prior_and_zero_likelihood_rows() -> None:
    class CountingProposal:
        ndim = 1

        def __init__(self) -> None:
            self.rows = 0

        def log_prob(self, theta: np.ndarray) -> np.ndarray:
            self.rows += len(theta)
            return np.zeros(len(theta))

    likelihood_rows = 0

    def log_prior(theta: np.ndarray) -> np.ndarray:
        return np.where(theta[:, 0] < 0.0, -np.inf, 0.0)

    def log_likelihood(theta: np.ndarray) -> np.ndarray:
        nonlocal likelihood_rows
        likelihood_rows += len(theta)
        return np.where(theta[:, 0] == 0.0, -np.inf, -(theta[:, 0] ** 2))

    proposal = CountingProposal()
    model = CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=log_likelihood,
        log_prior_fn=log_prior,
    )
    evaluator = BatchEvaluator(model, proposal)
    batch = evaluator.evaluate(np.array([[-1.0], [0.0], [1.0]]))

    assert evaluator.n_prior_calls == 3
    assert evaluator.n_likelihood_calls == 2
    assert likelihood_rows == 2
    assert proposal.rows == 1
    assert evaluator.outside_prior == 1
    assert evaluator.zero_likelihood == 1
    np.testing.assert_array_equal(batch.log_psi0, [-np.inf, -np.inf, -1.0])


def test_evaluate_one_checks_prior_before_likelihood_budget() -> None:
    proposal = UniformProposal()
    model = CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=lambda theta: np.zeros(len(theta)),
        log_prior_fn=proposal.log_prob,
    )
    evaluator = BatchEvaluator(model, proposal)

    outside = evaluator.evaluate_one(
        np.array([-1.0]),
        max_likelihood_calls=0,
    )
    assert np.isneginf(outside[1])
    assert evaluator.n_prior_calls == 1
    assert evaluator.n_likelihood_calls == 0
    with pytest.raises(LikelihoodBudgetExhausted):
        evaluator.evaluate_one(
            np.array([0.5]),
            max_likelihood_calls=0,
        )
    assert evaluator.n_prior_calls == 2
    assert evaluator.n_likelihood_calls == 0


def test_constrained_draw_rejects_bad_proposal_sample_shape() -> None:
    class BadShapeProposal(UniformProposal):
        def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
            return np.zeros((n, 2))

    proposal = BadShapeProposal()
    model = CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=lambda x: np.zeros(len(x)),
        log_prior_fn=lambda x: np.zeros(len(x)),
    )
    with pytest.raises(InvalidProposalOutput, match="sample must have shape"):
        draw_constrained(
            evaluator=BatchEvaluator(model, proposal),
            proposal_morph=proposal,
            threshold=-1.0,
            threshold_tie_breaker=0.0,
            tie_policy="strict",
            rng=np.random.default_rng(1),
            batch_size=4,
            max_proposals=8,
            max_likelihood_calls=None,
            deadline=None,
        )
