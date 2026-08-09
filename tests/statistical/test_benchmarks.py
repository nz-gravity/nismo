from __future__ import annotations

import numpy as np
import pytest
from benchmarks.models import (
    GaussianBenchmark,
    GaussianShellBenchmark,
    PeakPlateauBenchmark,
    UniformBoxProposal,
)
from scipy.special import logsumexp

from nismo import MorphProposal, NISMOSampler

pytestmark = pytest.mark.statistical


@pytest.fixture(scope="module")
def gaussian_problem() -> tuple[GaussianBenchmark, MorphProposal]:
    benchmark = GaussianBenchmark()
    training = benchmark.posterior_samples(300, np.random.default_rng(50))
    proposal = MorphProposal.fit(
        training,
        groups=[],
        param_names=("x",),
        kde_bw="silverman",
    )
    return benchmark, proposal


def test_analytic_gaussian_repeated_seed_aggregate(
    gaussian_problem: tuple[GaussianBenchmark, MorphProposal],
) -> None:
    benchmark, proposal = gaussian_problem
    logz_values = []
    logz_errors = []
    second_moments = []
    for seed in (11, 12, 13):
        result = NISMOSampler(
            model=benchmark.model(),
            importance_morph=proposal,
            n_live=25,
            rng=seed,
            proposal_batch_size=16,
        ).run(
            dlogz=0.08,
            max_iterations=400,
            max_proposals_per_replacement=5_000,
        )
        assert result.success
        logz_values.append(result.logz)
        logz_errors.append(result.logzerr)
        second_moments.append(
            np.sum(result.posterior_weights * result.all_points[:, 0] ** 2)
        )

    bias = np.mean(logz_values) - benchmark.logz
    empirical_spread = np.std(logz_values, ddof=1)
    assert abs(bias) < 0.04
    assert empirical_spread < 3.0 * np.mean(logz_errors)
    assert np.mean(second_moments) == pytest.approx(
        benchmark.posterior_variance, abs=0.3
    )


def test_direct_importance_cross_check(
    gaussian_problem: tuple[GaussianBenchmark, MorphProposal],
) -> None:
    benchmark, proposal = gaussian_problem
    rng = np.random.default_rng(44)
    theta = proposal.sample(2_000, rng)
    model = benchmark.model()
    log_weights = (
        model.log_likelihood(theta) + model.log_prior(theta) - proposal.log_prob(theta)
    )
    direct_logz = float(logsumexp(log_weights) - np.log(len(theta)))
    nested = NISMOSampler(
        model=model,
        importance_morph=proposal,
        n_live=25,
        rng=44,
        proposal_batch_size=16,
    ).run(
        dlogz=0.08,
        max_iterations=400,
        max_proposals_per_replacement=5_000,
    )
    assert nested.success
    assert direct_logz == pytest.approx(benchmark.logz, abs=0.05)
    assert nested.logz == pytest.approx(direct_logz, abs=0.06)


def test_adaptive_morph_keeps_fixed_q0_fields_and_records_refits(
    gaussian_problem: tuple[GaussianBenchmark, MorphProposal],
) -> None:
    benchmark, importance_morph = gaussian_problem
    model = benchmark.model()
    result = NISMOSampler(
        model=model,
        importance_morph=importance_morph,
        proposal_scheme="adaptive_morph",
        proposal_update_interval=25,
        n_live=25,
        rng=45,
        proposal_batch_size=16,
    ).run(
        dlogz=0.08,
        max_iterations=400,
        max_proposals_per_replacement=5_000,
    )

    assert result.success
    assert result.proposal_updates
    assert all(record.success for record in result.proposal_updates)
    assert all(record.n_training == result.nlive for record in result.proposal_updates)
    assert np.isfinite(result.logz)
    assert np.all(np.diff(result.dead_log_psi0) >= 0.0)
    expected_log_q0 = importance_morph.log_prob(result.all_points)
    np.testing.assert_allclose(
        np.concatenate((result.dead_log_q0, result.final_live_log_q0)),
        expected_log_q0,
    )


def test_peak_plateau_regression_uses_explicit_tie_policy() -> None:
    benchmark = PeakPlateauBenchmark()
    result = NISMOSampler(
        model=benchmark.model(),
        importance_morph=UniformBoxProposal(),
        n_live=40,
        rng=2026,
        proposal_batch_size=32,
        tie_policy="randomized_plateau",
    ).run(
        dlogz=0.05,
        max_iterations=500,
        max_proposals_per_replacement=20_000,
    )
    assert result.success
    assert result.logz == pytest.approx(benchmark.logz, abs=0.12)
    assert np.all(np.diff(result.dead_log_psi0) >= 0.0)
    assert "randomized_plateau" in " ".join(result.warnings)


@pytest.mark.slow
def test_grouped_morph_gaussian_shell_regression() -> None:
    benchmark = GaussianShellBenchmark()
    training = benchmark.approximate_posterior_samples(400, np.random.default_rng(71))
    proposal = MorphProposal.fit(
        training,
        groups=[[[0, 1], 1.0]],
        param_names=("x", "y"),
        kde_bw="silverman",
    )
    result = NISMOSampler(
        model=benchmark.model(),
        importance_morph=proposal,
        n_live=25,
        rng=72,
        proposal_batch_size=16,
    ).run(
        dlogz=0.1,
        max_iterations=400,
        max_proposals_per_replacement=10_000,
    )
    assert result.success
    assert np.all(np.isfinite(result.final_live_log_q0))
    assert result.logz == pytest.approx(benchmark.logz, abs=0.3)
    assert np.all(np.diff(result.dead_log_psi0) >= 0.0)
