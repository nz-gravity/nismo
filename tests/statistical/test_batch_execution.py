from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import truncnorm
from tests.helpers import StandardNormalProposal
from tests.integration.test_parallel_replacement import ConstantNormalModel
from tests.unit.test_batch import starts

from nismo import NISMOSampler, ParallelSettings, SRWalkSettings
from nismo.batch import evolve_srwalk_batch
from nismo.constrained import BatchEvaluator

pytestmark = pytest.mark.statistical


class HalfNormalModel(ConstantNormalModel):
    def log_prior(self, theta):
        return np.where(theta[:, 0] > 0.2, super().log_prior(theta), -np.inf)


@pytest.mark.parametrize("beta", [1.0, 0.8, 0.5])
def test_vectorized_kernel_targets_truncated_tempered_reference(beta):
    proposal = StandardNormalProposal()
    evaluator = BatchEvaluator(HalfNormalModel(), proposal, beta=beta)
    initial = starts(evaluator, np.full(2000, 0.5))
    walked = evolve_srwalk_batch(
        evaluator=evaluator,
        starts=initial,
        threshold=-np.inf,
        threshold_tie_breaker=0,
        tie_policy="strict",
        proposal_factor=np.eye(1),
        scale=1.0,
        n_steps=100,
        max_proposals=100,
        rngs=[
            np.random.default_rng(seed)
            for seed in np.random.SeedSequence(87).spawn(2000)
        ],
    )
    values = np.array([attempt.draw.point.theta[0] for attempt in walked.attempts])
    target = truncnorm(0.2 * np.sqrt(beta), np.inf, scale=1 / np.sqrt(beta))
    assert np.mean(values) == pytest.approx(target.mean(), abs=0.07)
    assert np.var(values) == pytest.approx(target.var(), abs=0.12)


class GaussianModel(ConstantNormalModel):
    def log_likelihood(self, theta):
        return -0.5 * theta[:, 0] ** 2


@pytest.mark.parametrize("beta", [1.0, 0.8])
@pytest.mark.parametrize("backend", ["vectorized", "process"])
def test_complete_repeated_gaussian_runs_recover_evidence_and_moments(beta, backend):
    logz = []
    means = []
    variances = []
    errors = []
    for seed in range(6):
        result = NISMOSampler(
            model=GaussianModel(),
            importance_morph=StandardNormalProposal(),
            proposal_scheme="s-rwalk",
            beta=beta,
            beta_mc_samples=20000,
            n_live=80,
            rng=seed + 104,
            srwalk_settings=SRWalkSettings(n_steps=20, dynamic_steps=False),
            parallel=ParallelSettings(
                backend=backend,
                queue_size=4,
                chains_per_task=2,
                scheduler="rolling" if backend == "process" else "epoch",
                n_workers=2 if backend == "process" else 1,
                adaptation_interval=4,
            ),
        ).run(dlogz=0.1, max_iterations=1000)
        assert result.success
        weights = result.posterior_weights
        values = result.all_points[:, 0]
        logz.append(result.logz)
        errors.append(result.logzerr)
        means.append(weights @ values)
        variances.append(weights @ values**2)
    truth = -0.5 * np.log(2)
    assert np.mean(logz) == pytest.approx(truth, abs=0.07)
    assert np.mean(means) == pytest.approx(0, abs=0.07)
    assert np.mean(variances) == pytest.approx(0.5, abs=0.08)
    assert sum(abs(z - truth) < 3 * e for z, e in zip(logz, errors, strict=True)) >= 4
