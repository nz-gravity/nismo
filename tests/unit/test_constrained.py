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
from nismo.constrained import BatchEvaluator, draw_constrained

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
