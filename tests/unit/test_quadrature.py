from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp

from nismo.quadrature import (
    dead_log_contribution,
    estimate_information,
    estimated_live_logz,
    finalize_quadrature,
    live_log_contributions,
    logdiffexp,
    update_log_weighted_mean,
)

pytestmark = pytest.mark.unit


def test_logdiffexp_matches_safe_direct_arithmetic() -> None:
    for log_a, log_b in [(0.0, -0.1), (-3.0, -7.0), (12.0, 11.0)]:
        expected = np.log(np.exp(log_a) - np.exp(log_b))
        assert logdiffexp(log_a, log_b) == pytest.approx(expected)


def test_logdiffexp_rejects_nonpositive_interval() -> None:
    with pytest.raises(ValueError, match="greater"):
        logdiffexp(-1.0, -1.0)


def test_dead_and_live_contributions_cover_unit_volume() -> None:
    nlive = 10
    niter = 17
    log_c = np.log(3.25)
    dead = [dead_log_contribution(i, nlive, log_c)[2] for i in range(1, niter + 1)]
    live = live_log_contributions(-niter / nlive, np.full(nlive, log_c))
    assert logsumexp(np.concatenate((dead, live))) == pytest.approx(log_c, abs=1e-14)


def test_finalize_constant_integrand_identity_and_weights() -> None:
    nlive = 20
    niter = 41
    log_c = np.log(7.0)
    dead = np.array(
        [dead_log_contribution(i, nlive, log_c)[2] for i in range(1, niter + 1)]
    )
    summary = finalize_quadrature(
        dead,
        np.full(niter, log_c),
        -niter / nlive,
        np.full(nlive, log_c),
        nlive,
    )
    assert summary.logz == pytest.approx(log_c, abs=1e-13)
    assert summary.information == pytest.approx(0.0, abs=1e-13)
    assert summary.logzerr == pytest.approx(0.0, abs=1e-13)
    assert np.sum(np.exp(summary.log_posterior_weights)) == pytest.approx(1.0)


def test_finalize_normalizes_weights_with_large_log_offset() -> None:
    """High-dimensional log scales must not degrade weight normalization."""
    nlive = 100
    niter = 1_000
    log_c = -1.0e12
    dead = np.array(
        [dead_log_contribution(i, nlive, log_c)[2] for i in range(1, niter + 1)]
    )
    summary = finalize_quadrature(
        dead,
        np.full(niter, log_c),
        -niter / nlive,
        np.full(nlive, log_c),
        nlive,
    )

    assert summary.logz == log_c
    assert logsumexp(summary.log_posterior_weights) == pytest.approx(0.0, abs=1e-15)
    assert np.sum(np.exp(summary.log_posterior_weights)) == pytest.approx(
        1.0, abs=1e-15
    )
    assert summary.information == pytest.approx(0.0, abs=1e-13)


def test_incremental_progress_information_matches_final_quadrature() -> None:
    nlive = 8
    dead_log_psi = np.linspace(-2.0, 0.4, 12)
    live_log_psi = np.linspace(0.5, 1.2, nlive)
    dead_log_weights = []
    logz_dead = -np.inf
    dead_mean = 0.0
    for iteration, log_psi in enumerate(dead_log_psi, start=1):
        log_weight = dead_log_contribution(iteration, nlive, log_psi)[2]
        dead_log_weights.append(log_weight)
        logz_dead, dead_mean = update_log_weighted_mean(
            logz_dead,
            dead_mean,
            log_weight,
            log_psi,
        )

    log_x = -len(dead_log_psi) / nlive
    logz_live = estimated_live_logz(log_x, live_log_psi)
    logz_total = float(np.logaddexp(logz_dead, logz_live))
    progress_information = estimate_information(
        logz_dead=logz_dead,
        dead_log_psi_mean=dead_mean,
        logz_live=logz_live,
        live_log_psi=live_log_psi,
        logz_total=logz_total,
    )
    final = finalize_quadrature(
        dead_log_weights,
        dead_log_psi,
        log_x,
        live_log_psi,
        nlive,
    )
    assert progress_information == pytest.approx(final.information, abs=1e-13)
    assert logz_total == pytest.approx(final.logz, abs=1e-13)
