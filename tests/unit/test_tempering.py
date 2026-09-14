from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp
from tests.helpers import StandardNormalProposal

from nismo.tempering import sample_power_tempered_pool

pytestmark = pytest.mark.unit


def test_beta_one_is_exact_and_preserves_direct_proposal_sampling() -> None:
    proposal = StandardNormalProposal()
    pool, diagnostics = sample_power_tempered_pool(
        proposal,
        beta=1.0,
        pool_size=20,
        n_mc_samples=1,
        rng=np.random.default_rng(91),
    )
    expected = proposal.sample(20, np.random.default_rng(91))

    np.testing.assert_array_equal(pool, expected)
    assert diagnostics.normalization_exact
    assert diagnostics.log_z_beta == 0.0
    assert diagnostics.log_z_beta_error == 0.0
    assert diagnostics.n_mc_samples == 0


def test_direct_mc_normalizer_and_pool_match_power_tempered_normal() -> None:
    proposal = StandardNormalProposal()
    beta = 0.8
    pool, diagnostics = sample_power_tempered_pool(
        proposal,
        beta=beta,
        pool_size=2_000,
        n_mc_samples=80_000,
        rng=np.random.default_rng(20260826),
    )
    expected_log_z = 0.5 * ((1.0 - beta) * np.log(2.0 * np.pi) - np.log(beta))

    assert not diagnostics.normalization_exact
    assert diagnostics.log_z_beta == pytest.approx(
        expected_log_z,
        abs=5.0 * diagnostics.log_z_beta_error,
    )
    assert diagnostics.mc_effective_sample_size > 20_000
    assert np.mean(pool[:, 0]) == pytest.approx(0.0, abs=0.07)
    assert np.var(pool[:, 0]) == pytest.approx(1.0 / beta, abs=0.12)


def test_beta_normalizer_uses_direct_q_monte_carlo_identity() -> None:
    proposal = StandardNormalProposal()
    rng = np.random.default_rng(4)
    candidates = proposal.sample(200, rng)
    log_q = proposal.log_prob(candidates)
    beta = 0.9
    expected = float(logsumexp((beta - 1.0) * log_q) - np.log(len(log_q)))

    class ReplayProposal(StandardNormalProposal):
        def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
            assert n == len(candidates)
            return np.array(candidates, copy=True)

    _, diagnostics = sample_power_tempered_pool(
        ReplayProposal(),
        beta=beta,
        pool_size=20,
        n_mc_samples=len(candidates),
        rng=np.random.default_rng(8),
    )
    assert diagnostics.log_z_beta == pytest.approx(expected)


def test_equal_candidate_and_pool_sizes_do_not_cancel_beta():
    proposal = StandardNormalProposal()
    pool, diagnostics = sample_power_tempered_pool(
        proposal,
        beta=0.5,
        pool_size=30_000,
        n_mc_samples=30_000,
        rng=np.random.default_rng(144),
    )
    assert np.var(pool[:, 0]) == pytest.approx(2.0, abs=0.25)
    assert diagnostics.sampling_method == "multinomial_sir"
    assert 0 < diagnostics.unique_pool_size < diagnostics.pool_size


def test_chunked_density_and_cached_selection_match_direct_normalization():
    from nismo.tempering import prepare_power_tempered_pool

    proposal = StandardNormalProposal()

    def chunked(points):
        return np.concatenate(
            [proposal.log_prob(points[i : i + 7]) for i in range(0, len(points), 7)]
        )

    a, qa, da = prepare_power_tempered_pool(
        proposal,
        beta=0.8,
        pool_size=100,
        n_mc_samples=301,
        rng=np.random.default_rng(13),
        density_evaluator=chunked,
    )
    b, qb, db = prepare_power_tempered_pool(
        proposal,
        beta=0.8,
        pool_size=100,
        n_mc_samples=301,
        rng=np.random.default_rng(13),
    )
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(qa, qb)
    np.testing.assert_array_equal(qa, proposal.log_prob(a))
    assert da == db
