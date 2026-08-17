from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.special import ndtr, ndtri
from tests.helpers import StandardNormalProposal

from nismo import (
    CallableModel,
    EnsembleMoveWeights,
    EnsembleRWalkSettings,
    NISMOSampler,
    SRWalkSettings,
)
from nismo.config import RWalkSettings
from nismo.constrained import BatchEvaluator, ConstrainedAttempt
from nismo.mcmc import (
    RWalkSampler,
    SRWalkSampler,
    draw_ensemble_rwalk_constrained,
    draw_rwalk_constrained,
    draw_srwalk_constrained,
)

pytestmark = pytest.mark.statistical


Kernel = Callable[..., ConstrainedAttempt]


def _normal_log_density(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return -0.5 * values**2 - 0.5 * np.log(2.0 * np.pi)


@pytest.mark.parametrize(
    ("kernel", "settings", "required_proposals"),
    [
        (draw_rwalk_constrained, RWalkSettings(walks=20), 20),
        (draw_srwalk_constrained, SRWalkSettings(n_steps=20), 20),
        (
            draw_ensemble_rwalk_constrained,
            EnsembleRWalkSettings(
                n_walkers=8,
                n_sweeps=3,
                gamma=0.7,
                jitter_scale=0.05,
                move_weights=EnsembleMoveWeights(de=1, stretch=0, gaussian=0),
            ),
            24,
        ),
        (
            draw_ensemble_rwalk_constrained,
            EnsembleRWalkSettings(
                n_walkers=8,
                n_sweeps=3,
                move_weights=EnsembleMoveWeights(de=0, stretch=1, gaussian=0),
            ),
            24,
        ),
        (
            draw_ensemble_rwalk_constrained,
            EnsembleRWalkSettings(
                n_walkers=8,
                n_sweeps=3,
                move_weights=EnsembleMoveWeights(de=0, stretch=0, gaussian=1),
            ),
            24,
        ),
        (
            draw_ensemble_rwalk_constrained,
            EnsembleRWalkSettings(
                n_walkers=8,
                n_sweeps=3,
                move_weights=EnsembleMoveWeights(
                    de=0.60,
                    stretch=0.25,
                    gaussian=0.15,
                ),
            ),
            24,
        ),
    ],
)
def test_truncated_normal_stationarity_requires_q0_mh_ratio(
    kernel: Kernel,
    settings: RWalkSettings | SRWalkSettings | EnsembleRWalkSettings,
    required_proposals: int,
) -> None:
    proposal = StandardNormalProposal()
    model = CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=lambda theta: theta[:, 0],
        log_prior_fn=proposal.log_prob,
    )
    threshold = 0.5
    threshold_cdf = float(ndtr(threshold))
    rng = np.random.default_rng(20260801)
    output = []
    for _ in range(600):
        survivors = ndtri(threshold_cdf + (1.0 - threshold_cdf) * rng.random(16))
        live_values = np.concatenate(([threshold], survivors))
        live_theta = live_values[:, np.newaxis]
        log_q0 = _normal_log_density(live_values)
        if isinstance(settings, RWalkSettings):
            kernel_setting = {"sampler": RWalkSampler(settings=settings, ndim=1)}
        elif isinstance(settings, SRWalkSettings):
            kernel_setting = {"sampler": SRWalkSampler(settings=settings, ndim=1)}
        else:
            kernel_setting = {"settings": settings}
        attempt = kernel(
            evaluator=BatchEvaluator(model, proposal),
            live_theta=live_theta,
            live_log_likelihood=live_values,
            live_log_prior=log_q0,
            live_log_q0=log_q0,
            live_log_psi0=live_values,
            live_tie_breakers=np.zeros(len(live_values)),
            worst=0,
            threshold=threshold,
            threshold_tie_breaker=0.0,
            tie_policy="strict",
            **kernel_setting,
            rng=rng,
            max_proposals=required_proposals,
            max_likelihood_calls=None,
            deadline=None,
        )
        assert attempt.draw is not None
        output.append(attempt.draw.point.theta[0])

    values = np.sort(np.asarray(output))
    tail_probability = 1.0 - threshold_cdf
    density_at_threshold = np.exp(-0.5 * threshold**2) / np.sqrt(2.0 * np.pi)
    expected_mean = density_at_threshold / tail_probability
    expected_variance = 1.0 + threshold * expected_mean - expected_mean * expected_mean
    expected_quantiles = ndtri(
        threshold_cdf
        + tail_probability
        * np.array(
            [0.1, 0.5, 0.9],
        )
    )
    conditional_cdf = (ndtr(values) - threshold_cdf) / tail_probability
    empirical_upper = np.arange(1, len(values) + 1) / len(values)
    empirical_lower = np.arange(len(values)) / len(values)
    ks_distance = max(
        float(np.max(np.abs(empirical_upper - conditional_cdf))),
        float(np.max(np.abs(empirical_lower - conditional_cdf))),
    )

    assert np.mean(values) == pytest.approx(expected_mean, abs=0.06)
    assert np.var(values) == pytest.approx(expected_variance, abs=0.07)
    np.testing.assert_allclose(
        np.quantile(values, [0.1, 0.5, 0.9]),
        expected_quantiles,
        atol=0.09,
        rtol=0.0,
    )
    assert ks_distance < 0.07


class _CorrelatedNormalProposal:
    ndim = 2
    covariance = np.array([[1.0, 0.75], [0.75, 1.5]])
    precision = np.linalg.inv(covariance)
    log_normalization = float(
        -0.5 * (2 * np.log(2.0 * np.pi) + np.linalg.slogdet(covariance)[1])
    )

    def sample(
        self,
        n: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        return np.asarray(
            rng.multivariate_normal(np.zeros(2), self.covariance, size=n),
            dtype=float,
        )

    def log_prob(
        self,
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return self.log_normalization - 0.5 * np.einsum(
            "ni,ij,nj->n",
            theta,
            self.precision,
            theta,
        )


def _draw_truncated_correlated(
    proposal: _CorrelatedNormalProposal,
    *,
    n: int,
    direction: NDArray[np.float64],
    threshold: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    accepted: list[NDArray[np.float64]] = []
    count = 0
    while count < n:
        candidates = proposal.sample(max(2 * (n - count), 32), rng)
        valid = candidates @ direction > threshold
        batch = candidates[valid]
        accepted.append(batch)
        count += len(batch)
    return np.concatenate(accepted, axis=0)[:n]


@pytest.mark.parametrize(
    ("kernel", "settings", "required_proposals"),
    [
        (draw_rwalk_constrained, RWalkSettings(walks=16), 16),
        (draw_srwalk_constrained, SRWalkSettings(n_steps=16), 16),
        (
            draw_ensemble_rwalk_constrained,
            EnsembleRWalkSettings(
                n_walkers=8,
                n_sweeps=2,
                gamma=0.7,
                jitter_scale=0.05,
                move_weights=EnsembleMoveWeights(de=1, stretch=0, gaussian=0),
            ),
            16,
        ),
        (
            draw_ensemble_rwalk_constrained,
            EnsembleRWalkSettings(
                n_walkers=8,
                n_sweeps=2,
                move_weights=EnsembleMoveWeights(de=0, stretch=1, gaussian=0),
            ),
            16,
        ),
        (
            draw_ensemble_rwalk_constrained,
            EnsembleRWalkSettings(
                n_walkers=8,
                n_sweeps=2,
                move_weights=EnsembleMoveWeights(de=0, stretch=0, gaussian=1),
            ),
            16,
        ),
        (
            draw_ensemble_rwalk_constrained,
            EnsembleRWalkSettings(
                n_walkers=8,
                n_sweeps=2,
                move_weights=EnsembleMoveWeights(
                    de=0.60,
                    stretch=0.25,
                    gaussian=0.15,
                ),
            ),
            16,
        ),
    ],
)
def test_correlated_gaussian_non_axis_aligned_constraint(
    kernel: Kernel,
    settings: RWalkSettings | SRWalkSettings | EnsembleRWalkSettings,
    required_proposals: int,
) -> None:
    proposal = _CorrelatedNormalProposal()
    direction = np.array([1.0, -0.4])
    threshold = 0.2
    model = CallableModel(
        ndim=2,
        parameter_names=("x", "y"),
        log_likelihood_fn=lambda theta: theta @ direction,
        log_prior_fn=proposal.log_prob,
    )
    rng = np.random.default_rng(20260802)
    reference = _draw_truncated_correlated(
        proposal,
        n=150_000,
        direction=direction,
        threshold=threshold,
        rng=rng,
    )
    output = []
    boundary = direction * threshold / float(direction @ direction)
    for _ in range(400):
        survivors = _draw_truncated_correlated(
            proposal,
            n=16,
            direction=direction,
            threshold=threshold,
            rng=rng,
        )
        live_theta = np.vstack((boundary, survivors))
        live_log_likelihood = live_theta @ direction
        live_log_q0 = proposal.log_prob(live_theta)
        if isinstance(settings, RWalkSettings):
            kernel_setting = {"sampler": RWalkSampler(settings=settings, ndim=2)}
        elif isinstance(settings, SRWalkSettings):
            kernel_setting = {"sampler": SRWalkSampler(settings=settings, ndim=2)}
        else:
            kernel_setting = {"settings": settings}
        attempt = kernel(
            evaluator=BatchEvaluator(model, proposal),
            live_theta=live_theta,
            live_log_likelihood=live_log_likelihood,
            live_log_prior=live_log_q0,
            live_log_q0=live_log_q0,
            live_log_psi0=live_log_likelihood,
            live_tie_breakers=np.zeros(len(live_theta)),
            worst=0,
            threshold=threshold,
            threshold_tie_breaker=0.0,
            tie_policy="strict",
            **kernel_setting,
            rng=rng,
            max_proposals=required_proposals,
            max_likelihood_calls=None,
            deadline=None,
        )
        assert attempt.draw is not None
        output.append(attempt.draw.point.theta)
    values = np.asarray(output)
    np.testing.assert_allclose(
        np.mean(values, axis=0),
        np.mean(reference, axis=0),
        atol=0.13,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.cov(values, rowvar=False),
        np.cov(reference, rowvar=False),
        atol=0.24,
        rtol=0.0,
    )


def test_end_to_end_gaussian_evidence_agrees_across_replacement_schemes() -> None:
    proposal = StandardNormalProposal()
    model = CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=lambda theta: -0.5 * (theta[:, 0] - 1.0) ** 2,
        log_prior_fn=proposal.log_prob,
    )
    exact_logz = -0.25 - 0.5 * np.log(2.0)
    schemes = (
        ("fixed_morph", {}),
        ("s-rwalk", {"srwalk_settings": SRWalkSettings(n_steps=16)}),
        (
            "en-rwalk",
            {
                "ensemble_rwalk_settings": EnsembleRWalkSettings(
                    n_walkers=8,
                    n_sweeps=2,
                    gamma=0.7,
                    jitter_scale=0.05,
                    move_weights=EnsembleMoveWeights(de=1, stretch=0, gaussian=0),
                )
            },
        ),
        (
            "en-rwalk",
            {
                "ensemble_rwalk_settings": EnsembleRWalkSettings(
                    n_walkers=8,
                    n_sweeps=2,
                    move_weights=EnsembleMoveWeights(
                        de=0.60,
                        stretch=0.25,
                        gaussian=0.15,
                    ),
                )
            },
        ),
    )
    scheme_means = []
    for proposal_scheme, settings in schemes:
        logz = []
        for seed in (31, 32, 33):
            result = NISMOSampler(
                model=model,
                importance_morph=proposal,
                proposal_scheme=proposal_scheme,  # type: ignore[arg-type]
                n_live=50,
                rng=seed,
                proposal_batch_size=16,
                **settings,  # type: ignore[arg-type]
            ).run(
                dlogz=0.1,
                max_iterations=400,
                max_proposals_per_replacement=5_000,
            )
            assert result.success
            assert np.all(np.diff(result.dead_log_psi0) >= 0.0)
            assert np.sum(result.posterior_weights) == pytest.approx(1.0)
            logz.append(result.logz)
        mean_logz = float(np.mean(logz))
        scheme_means.append(mean_logz)
        assert mean_logz == pytest.approx(exact_logz, abs=0.09)
    assert max(scheme_means) - min(scheme_means) < 0.05
