from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from tests.helpers import StandardNormalProposal

from nismo import (
    CallableModel,
    ConfigurationError,
    EnsembleMoveWeights,
    EnsembleRWalkSettings,
    NISMOConfig,
    NumericalInvariantError,
    RWalkSettings,
    SRWalkSettings,
)
from nismo.constrained import BatchEvaluator, passes_constraint
from nismo.mcmc import (
    RWalkSampler,
    SRWalkGeometry,
    SRWalkSampler,
    _propose_gaussian_move,
    _propose_stretch_move,
    _random_unit_ball,
    _select_ensemble_move,
    accepts_log_q0_metropolis,
    accepts_metropolis,
    bounding_ellipsoid_axes,
    covariance_factor,
    draw_ensemble_rwalk_constrained,
    draw_rwalk_constrained,
    draw_srwalk_constrained,
    eligible_survivor_indices,
    log_metropolis_acceptance_ratio,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "proposal_scheme",
    ["fixed_morph", "adaptive_morph", "rwalk", "s-rwalk", "en-rwalk"],
)
def test_all_proposal_schemes_are_configured(proposal_scheme: str) -> None:
    settings = EnsembleRWalkSettings(n_walkers=4)
    config = NISMOConfig(
        n_live=5,
        proposal_scheme=proposal_scheme,  # type: ignore[arg-type]
        ensemble_rwalk_settings=settings,
    )
    assert config.proposal_scheme == proposal_scheme


@pytest.mark.parametrize(
    "kwargs",
    [
        {"walks": 0},
        {"walks": True},
        {"facc": True},
        {"facc": np.inf},
        {"facc": np.nan},
        {"ncdim": 0},
        {"ncdim": True},
    ],
)
def test_rwalk_settings_reject_invalid_values(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError):
        RWalkSettings(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_steps": 4},
        {"scale": 1.0},
        {"covariance_shrinkage": 0.1},
        {"covariance_jitter": 1.0e-10},
    ],
)
def test_removed_rwalk_settings_are_not_accepted(kwargs: dict[str, Any]) -> None:
    with pytest.raises(TypeError):
        RWalkSettings(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_steps": 0},
        {"n_steps": True},
        {"scale": 0.0},
        {"scale": np.inf},
        {"scale": True},
        {"facc": True},
        {"facc": np.nan},
        {"covariance_shrinkage": -0.1},
        {"covariance_shrinkage": 1.1},
        {"covariance_jitter": 0.0},
        {"covariance_jitter": np.inf},
        {"covariance_update_interval": 0},
        {"covariance_update_interval": True},
        {"covariance_rebuild_interval": 0},
        {"covariance_rebuild_interval": True},
        {"profile": 1},
    ],
)
def test_srwalk_settings_reject_invalid_values(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError):
        SRWalkSettings(**kwargs)


def test_srwalk_sampler_uses_gaussian_default_and_rwalk_adaptation() -> None:
    sampler = SRWalkSampler(settings=SRWalkSettings(n_steps=4), ndim=2)
    assert sampler.n_steps == 4
    assert sampler.facc == 0.5
    assert sampler.scale == pytest.approx(2.38 / np.sqrt(2.0))

    initial_scale = sampler.scale
    sampler.record_completed_walk(accept=4, scale=initial_scale)
    assert sampler.scale == pytest.approx(initial_scale * np.exp(0.5))
    sampler.record_completed_walk(accept=0, scale=sampler.scale)
    assert sampler.scale == pytest.approx(initial_scale)

    explicit = SRWalkSampler(
        settings=SRWalkSettings(n_steps=5, scale=0.25, facc=0.2),
        ndim=1,
    )
    assert explicit.scale == 0.25
    assert explicit.facc == 0.2


def test_rwalk_sampler_resolves_dynesty_defaults_and_clamps_controls() -> None:
    default = RWalkSampler(settings=RWalkSettings(), ndim=4)
    assert default.walks == 24
    assert default.facc == 0.5
    assert default.ncdim == 4
    assert default.scale == 1.0
    assert default.update_bound_interval_ratio == 24
    assert default.citations == [
        ("Skilling (2006)", "projecteuclid.org/euclid.ba/1340370944")
    ]

    low = RWalkSampler(settings=RWalkSettings(walks=1, facc=-2.0), ndim=2)
    assert low.walks == 2
    assert low.facc == 0.5
    high = RWalkSampler(settings=RWalkSettings(walks=8, facc=3.0), ndim=2)
    assert high.facc == 1.0

    with pytest.raises(ConfigurationError, match="model dimension"):
        RWalkSampler(settings=RWalkSettings(ncdim=1), ndim=2)


def test_rwalk_tuning_accumulates_updates_and_resets_history() -> None:
    sampler = RWalkSampler(
        settings=RWalkSettings(walks=4, facc=0.5),
        ndim=2,
    )
    sampler.tune({"scale": 1.0, "accept": 4, "reject": 0}, update=False)
    assert sampler.scale == 1.0
    assert sampler.rwalk_history == {"n_accept": 4, "n_reject": 0}
    sampler.tune({"scale": 1.0, "accept": 0, "reject": 4})
    assert sampler.scale == pytest.approx(1.0)
    assert sampler.rwalk_history == {"n_accept": 0, "n_reject": 0}

    sampler.tune({"scale": 1.0, "accept": 4, "reject": 0})
    assert sampler.scale == pytest.approx(np.exp(0.5))
    sampler.tune({"scale": 1.0, "accept": 0, "reject": 4})
    assert sampler.scale == pytest.approx(np.exp(-0.5))


def test_rwalk_bound_cache_refreshes_after_walks_times_nlive_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def counted_axes(points: NDArray[np.float64]) -> NDArray[np.float64]:
        nonlocal calls
        calls += 1
        return np.ones((points.shape[1], points.shape[1]))

    monkeypatch.setattr("nismo.mcmc.bounding_ellipsoid_axes", counted_axes)
    sampler = RWalkSampler(settings=RWalkSettings(walks=2), ndim=1)
    live = np.array([[0.0], [1.0], [2.0]])
    sampler.axes_for(live)
    for _ in range(3):
        sampler.record_completed_walk(accept=1, scale=sampler.scale)
        sampler.axes_for(live)
    assert calls == 2


def test_random_ball_draws_stay_inside_the_unit_ball() -> None:
    rng = np.random.default_rng(917)
    draws = np.array([_random_unit_ball(3, rng) for _ in range(2_000)])
    radii = np.linalg.norm(draws, axis=1)
    assert np.all(radii <= 1.0)
    assert np.mean(radii) == pytest.approx(0.75, abs=0.02)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_walkers": 2},
        {"n_walkers": 5},
        {"n_walkers": True},
        {"n_sweeps": 0},
        {"gamma": 0.0},
        {"gamma": np.nan},
        {"jitter_scale": 0.0},
        {"jitter_scale": np.inf},
        {"covariance_shrinkage": -0.1},
        {"covariance_shrinkage": 1.1},
        {"covariance_jitter": 0.0},
    ],
)
def test_ensemble_settings_reject_invalid_values(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError):
        EnsembleRWalkSettings(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"de": -1.0},
        {"stretch": np.nan},
        {"gaussian": np.inf},
        {"de": True},
        {"de": 0.0, "stretch": 0.0, "gaussian": 0.0},
    ],
)
def test_ensemble_move_weights_reject_invalid_values(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError):
        EnsembleMoveWeights(**kwargs)


def test_ensemble_move_weights_normalize_active_moves_in_canonical_order() -> None:
    weights = EnsembleMoveWeights(de=6, stretch=3, gaussian=1)
    names, probabilities = weights.active_names_and_probabilities
    assert names == ("de", "stretch", "gaussian")
    assert probabilities == pytest.approx((0.6, 0.3, 0.1))
    huge = np.finfo(float).max
    assert EnsembleMoveWeights(
        de=huge,
        stretch=huge,
        gaussian=huge,
    ).active_names_and_probabilities[1] == pytest.approx((1 / 3, 1 / 3, 1 / 3))
    assert EnsembleRWalkSettings().move_weights == EnsembleMoveWeights(
        de=0.60,
        stretch=0.25,
        gaussian=0.15,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stretch_scale": 1.0},
        {"stretch_scale": 0.5},
        {"stretch_scale": np.inf},
        {"stretch_scale": np.nan},
        {"gaussian_scale": 0.0},
        {"gaussian_scale": -1.0},
        {"gaussian_scale": np.inf},
        {"gaussian_scale": np.nan},
        {"move_weights": {"de": 1.0}},
    ],
)
def test_ensemble_settings_reject_invalid_mixture_values(
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises(ConfigurationError):
        EnsembleRWalkSettings(**kwargs)


def test_ensemble_size_is_checked_against_live_survivors() -> None:
    with pytest.raises(ConfigurationError, match="n_live"):
        NISMOConfig(
            n_live=8,
            proposal_scheme="en-rwalk",
            ensemble_rwalk_settings=EnsembleRWalkSettings(n_walkers=8),
        )


def test_constraint_helper_implements_strict_and_plateau_ordering() -> None:
    values = np.array([0.9, 1.0, 1.0, 1.1])
    ties = np.array([0.9, 0.4, 0.6, 0.1])
    strict = passes_constraint(
        values,
        ties,
        threshold=1.0,
        threshold_tie_breaker=0.5,
        tie_policy="strict",
    )
    randomized = passes_constraint(
        values,
        ties,
        threshold=1.0,
        threshold_tie_breaker=0.5,
        tie_policy="randomized_plateau",
    )
    np.testing.assert_array_equal(strict, [False, False, False, True])
    np.testing.assert_array_equal(randomized, [False, False, True, True])
    assert not passes_constraint(
        1.0,
        0.9,
        threshold=1.0,
        threshold_tie_breaker=0.5,
        tie_policy="strict",
    )


def test_covariance_factor_handles_one_point_and_rank_deficiency() -> None:
    one_dimensional = covariance_factor(
        np.array([[2.0]]),
        shrinkage=0.1,
        jitter=1.0e-8,
    )
    assert one_dimensional.shape == (1, 1)
    assert one_dimensional[0, 0] == pytest.approx(1.0e-4)

    rank_deficient = covariance_factor(
        np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]),
        shrinkage=0.0,
        jitter=1.0e-10,
    )
    assert rank_deficient.shape == (3, 3)
    assert np.all(np.isfinite(rank_deficient))
    assert np.linalg.matrix_rank(rank_deficient) == 3


def test_rolling_srwalk_geometry_matches_full_survivor_covariance() -> None:
    rng = np.random.default_rng(20260812)
    live = rng.normal(size=(32, 6))
    settings = SRWalkSettings(
        covariance_shrinkage=0.2,
        covariance_jitter=1.0e-9,
        covariance_update_interval=1,
        covariance_rebuild_interval=10_000,
    )
    geometry = SRWalkGeometry(live, settings=settings)

    for iteration in range(2_000):
        worst = iteration % len(live)
        rolling_factor = geometry.factor_for_worst(live[worst])
        survivors = np.concatenate((live[:worst], live[worst + 1 :]))
        direct_factor = covariance_factor(
            survivors,
            shrinkage=settings.covariance_shrinkage,
            jitter=settings.covariance_jitter,
        )
        np.testing.assert_allclose(
            rolling_factor @ rolling_factor.T,
            direct_factor @ direct_factor.T,
            rtol=2.0e-10,
            atol=2.0e-10,
        )

        outgoing = np.array(live[worst], copy=True)
        incoming = rng.normal(size=live.shape[1])
        live[worst] = incoming
        geometry.commit_replacement(
            outgoing=outgoing,
            incoming=incoming,
            live_theta=live,
        )

    np.testing.assert_allclose(geometry.mean, np.mean(live, axis=0), atol=2.0e-13)
    centered = live - np.mean(live, axis=0)
    np.testing.assert_allclose(
        geometry.scatter,
        centered.T @ centered,
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_srwalk_geometry_caches_factor_and_periodically_rebuilds() -> None:
    rng = np.random.default_rng(47)
    live = rng.normal(size=(8, 3))
    geometry = SRWalkGeometry(
        live,
        settings=SRWalkSettings(
            covariance_update_interval=3,
            covariance_rebuild_interval=4,
            profile=True,
        ),
    )
    first = geometry.factor_for_worst(live[0])
    assert geometry.factor_for_worst(live[1]) is first

    for index in range(4):
        worst = index
        outgoing = np.array(live[worst], copy=True)
        incoming = rng.normal(size=3)
        live[worst] = incoming
        geometry.commit_replacement(
            outgoing=outgoing,
            incoming=incoming,
            live_theta=live,
        )
        geometry.factor_for_worst(live[(worst + 1) % len(live)])

    assert geometry.n_updates == 4
    assert geometry.n_rebuilds == 1
    assert geometry.n_factorizations == 2
    assert geometry.update_seconds >= 0.0
    assert geometry.rebuild_seconds >= 0.0
    assert geometry.factorization_seconds >= 0.0


def test_dynesty_ellipsoid_contains_rank_deficient_live_points() -> None:
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]],
    )
    axes = bounding_ellipsoid_axes(points)
    assert axes.shape == (3, 3)
    assert np.all(np.isfinite(axes))
    assert np.linalg.matrix_rank(axes) == 3
    inverse = np.linalg.inv(axes)
    transformed = (points - np.mean(points, axis=0)) @ inverse.T
    assert np.all(np.linalg.norm(transformed, axis=1) < 1.0)


def test_fixed_q0_ratio_can_reject_a_point_above_the_constraint() -> None:
    rejected_rng = np.random.default_rng(1)
    assert not accepts_log_q0_metropolis(
        current_log_q0=0.0,
        proposed_log_q0=-1.0,
        rng=rejected_rng,
    )
    accepted_rng = np.random.default_rng(1)
    assert accepts_log_q0_metropolis(
        current_log_q0=-1.0,
        proposed_log_q0=0.0,
        rng=accepted_rng,
    )


def test_general_metropolis_ratio_validates_and_applies_hastings_term() -> None:
    assert log_metropolis_acceptance_ratio(
        current_log_q0=0.0,
        proposed_log_q0=-1.0,
        log_hastings_ratio=0.25,
    ) == pytest.approx(-0.75)
    assert (
        log_metropolis_acceptance_ratio(
            current_log_q0=0.0,
            proposed_log_q0=-np.inf,
        )
        == -np.inf
    )
    assert (
        log_metropolis_acceptance_ratio(
            current_log_q0=0.0,
            proposed_log_q0=0.0,
            log_hastings_ratio=-np.inf,
        )
        == -np.inf
    )
    for invalid in (np.nan, np.inf, -np.inf):
        with pytest.raises(NumericalInvariantError):
            log_metropolis_acceptance_ratio(
                current_log_q0=invalid,
                proposed_log_q0=0.0,
            )
    for invalid in (np.nan, np.inf):
        with pytest.raises(NumericalInvariantError):
            log_metropolis_acceptance_ratio(
                current_log_q0=0.0,
                proposed_log_q0=invalid,
            )
        with pytest.raises(NumericalInvariantError):
            log_metropolis_acceptance_ratio(
                current_log_q0=0.0,
                proposed_log_q0=0.0,
                log_hastings_ratio=invalid,
            )


def test_stretch_hastings_term_can_change_the_acceptance_decision() -> None:
    symmetric = accepts_log_q0_metropolis(
        current_log_q0=0.0,
        proposed_log_q0=-1.0,
        rng=np.random.default_rng(1),
    )
    corrected = accepts_metropolis(
        current_log_q0=0.0,
        proposed_log_q0=-1.0,
        log_hastings_ratio=2.0,
        rng=np.random.default_rng(1),
    )
    assert not symmetric
    assert corrected


def test_pure_ensemble_move_selection_consumes_no_rng_draw() -> None:
    class NoChoice:
        def choice(self, *args: Any, **kwargs: Any) -> int:
            raise AssertionError("pure move selection must not consume RNG")

    configurations = (
        (EnsembleMoveWeights(de=1, stretch=0, gaussian=0), "de"),
        (EnsembleMoveWeights(de=0, stretch=1, gaussian=0), "stretch"),
        (EnsembleMoveWeights(de=0, stretch=0, gaussian=1), "gaussian"),
    )
    for weights, expected in configurations:
        assert (
            _select_ensemble_move(
                move_weights=weights,
                rng=NoChoice(),  # type: ignore[arg-type]
            )
            == expected
        )


def test_weighted_ensemble_move_selection_is_reproducible_and_normalized() -> None:
    weights = EnsembleMoveWeights(de=6, stretch=3, gaussian=1)

    def selected(seed: int) -> list[str]:
        rng = np.random.default_rng(seed)
        return [
            _select_ensemble_move(move_weights=weights, rng=rng) for _ in range(20_000)
        ]

    first = selected(112)
    assert first == selected(112)
    frequencies = np.array(
        [first.count(name) for name in ("de", "stretch", "gaussian")]
    )
    np.testing.assert_allclose(frequencies / len(first), [0.6, 0.3, 0.1], atol=0.01)
    zero_weight_draws = [
        _select_ensemble_move(
            move_weights=EnsembleMoveWeights(de=2, stretch=1, gaussian=0),
            rng=np.random.default_rng(seed),
        )
        for seed in range(1_000)
    ]
    assert "gaussian" not in zero_weight_draws


def test_stretch_move_uses_exact_factor_and_hastings_formula() -> None:
    ensemble = np.array(
        [[0.0, 1.0, 2.0], [2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]
    )
    active = np.array([0, 1], dtype=np.int64)
    complement = np.array([2, 3], dtype=np.int64)
    scale = 2.5
    expected_rng = np.random.default_rng(401)
    references = expected_rng.choice(complement, size=2, replace=True)
    uniform = expected_rng.random(2)
    stretch = ((scale - 1.0) * uniform + 1.0) ** 2 / scale
    expected = ensemble[references] + stretch[:, np.newaxis] * (
        ensemble[active] - ensemble[references]
    )

    proposal = _propose_stretch_move(
        ensemble_theta=ensemble,
        active=active,
        complement=complement,
        stretch_scale=scale,
        rng=np.random.default_rng(401),
    )
    np.testing.assert_allclose(proposal.theta, expected)
    np.testing.assert_allclose(proposal.log_hastings_ratio, 2.0 * np.log(stretch))
    assert np.all(stretch >= 1.0 / scale)
    assert np.all(stretch <= scale)


def test_gaussian_move_respects_scale_and_has_zero_hastings_ratio() -> None:
    ensemble = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
    active = np.array([0, 1], dtype=np.int64)
    factor = np.diag([2.0, 3.0])
    expected_rng = np.random.default_rng(99)
    expected = np.empty((2, 2))
    for row, walker in enumerate(active):
        expected[row] = ensemble[walker] + 0.25 * (
            factor @ expected_rng.standard_normal(2)
        )
    proposal = _propose_gaussian_move(
        ensemble_theta=ensemble,
        active=active,
        gaussian_scale=0.25,
        factor=factor,
        rng=np.random.default_rng(99),
    )
    np.testing.assert_allclose(proposal.theta, expected)
    np.testing.assert_array_equal(proposal.log_hastings_ratio, np.zeros(2))


def _all_rejected_problem() -> tuple[
    BatchEvaluator,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    proposal = StandardNormalProposal()
    model = CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=lambda theta: np.full(len(theta), -np.inf),
        log_prior_fn=proposal.log_prob,
    )
    theta = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    log_q0 = proposal.log_prob(theta)
    return (
        BatchEvaluator(model, proposal),
        theta,
        np.arange(6.0),
        np.array(log_q0, copy=True),
        log_q0,
        np.arange(6.0),
        np.linspace(0.1, 0.6, 6),
    )


def test_rwalk_starts_uniformly_from_an_eligible_survivor_and_can_stay_put(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        evaluator,
        theta,
        log_likelihood,
        log_prior,
        log_q0,
        log_psi0,
        ties,
    ) = _all_rejected_problem()
    seed = 20260731
    expected_rng = np.random.default_rng(seed)
    eligible = eligible_survivor_indices(
        live_log_psi0=log_psi0,
        live_tie_breakers=ties,
        worst=0,
        threshold=0.0,
        threshold_tie_breaker=ties[0],
        tie_policy="strict",
    )
    expected_index = int(expected_rng.choice(eligible))
    bound_calls = 0

    def counted_bound(
        points: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        nonlocal bound_calls
        bound_calls += 1
        assert points.shape == (6, 1)
        return np.ones((1, 1))

    monkeypatch.setattr("nismo.mcmc.bounding_ellipsoid_axes", counted_bound)
    sampler = RWalkSampler(settings=RWalkSettings(walks=7), ndim=1)
    attempt = draw_rwalk_constrained(
        evaluator=evaluator,
        live_theta=theta,
        live_log_likelihood=log_likelihood,
        live_log_prior=log_prior,
        live_log_q0=log_q0,
        live_log_psi0=log_psi0,
        live_tie_breakers=ties,
        worst=0,
        threshold=0.0,
        threshold_tie_breaker=ties[0],
        tie_policy="strict",
        sampler=sampler,
        rng=np.random.default_rng(seed),
        max_proposals=7,
        max_likelihood_calls=None,
        deadline=None,
    )
    assert attempt.draw is not None
    assert bound_calls == 1
    np.testing.assert_array_equal(attempt.draw.point.theta, theta[expected_index])
    assert attempt.draw.point.log_likelihood == log_likelihood[expected_index]
    assert attempt.draw.point.tie_breaker == ties[expected_index]
    assert expected_index != 0
    assert attempt.n_proposed == 7
    assert evaluator.n_likelihood_calls == 7
    assert attempt.n_valid == 0
    assert attempt.n_accepted == 0
    assert attempt.n_moved == 0
    assert attempt.n_completed == 7


def test_srwalk_uses_frozen_survivor_covariance_and_can_stay_put(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        evaluator,
        theta,
        log_likelihood,
        log_prior,
        log_q0,
        log_psi0,
        ties,
    ) = _all_rejected_problem()
    seed = 20260804
    expected_rng = np.random.default_rng(seed)
    eligible = eligible_survivor_indices(
        live_log_psi0=log_psi0,
        live_tie_breakers=ties,
        worst=0,
        threshold=0.0,
        threshold_tie_breaker=ties[0],
        tie_policy="strict",
    )
    expected_index = int(expected_rng.choice(eligible))
    covariance_calls = 0

    def counted_covariance(
        points: NDArray[np.float64],
        *,
        shrinkage: float,
        jitter: float,
    ) -> NDArray[np.float64]:
        nonlocal covariance_calls
        covariance_calls += 1
        np.testing.assert_array_equal(points, theta[1:])
        assert shrinkage == 0.2
        assert jitter == 1.0e-8
        return np.ones((1, 1))

    monkeypatch.setattr("nismo.mcmc.covariance_factor", counted_covariance)
    sampler = SRWalkSampler(
        settings=SRWalkSettings(
            n_steps=7,
            scale=1.0,
            covariance_shrinkage=0.2,
            covariance_jitter=1.0e-8,
        ),
        ndim=1,
    )
    attempt = draw_srwalk_constrained(
        evaluator=evaluator,
        live_theta=theta,
        live_log_likelihood=log_likelihood,
        live_log_prior=log_prior,
        live_log_q0=log_q0,
        live_log_psi0=log_psi0,
        live_tie_breakers=ties,
        worst=0,
        threshold=0.0,
        threshold_tie_breaker=ties[0],
        tie_policy="strict",
        sampler=sampler,
        rng=np.random.default_rng(seed),
        max_proposals=7,
        max_likelihood_calls=None,
        deadline=None,
    )
    assert attempt.draw is not None
    assert covariance_calls == 1
    np.testing.assert_array_equal(attempt.draw.point.theta, theta[expected_index])
    assert attempt.draw.point.log_likelihood == log_likelihood[expected_index]
    assert attempt.draw.point.tie_breaker == ties[expected_index]
    assert expected_index != 0
    assert attempt.n_proposed == 7
    assert evaluator.n_likelihood_calls == 7
    assert attempt.n_valid == 0
    assert attempt.n_accepted == 0
    assert attempt.n_moved == 0
    assert attempt.n_completed == 7
    assert sampler.scale == pytest.approx(np.exp(-1.0))


def test_srwalk_batches_gaussian_increment_linear_algebra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator, theta, log_likelihood, log_prior, log_q0, log_psi0, ties = (
        _all_rejected_problem()
    )

    class CountingGenerator:
        def __init__(self) -> None:
            self.normal_calls = 0

        def choice(self, values: NDArray[np.int64]) -> int:
            return int(values[0])

        def standard_normal(self, *, size: tuple[int, int]) -> NDArray[np.float64]:
            self.normal_calls += 1
            assert size == (7, 1)
            return np.ones(size)

        def random(self) -> float:
            return 0.5

    monkeypatch.setattr(
        "nismo.mcmc.covariance_factor",
        lambda *args, **kwargs: np.ones((1, 1)),
    )
    rng = CountingGenerator()
    attempt = draw_srwalk_constrained(
        evaluator=evaluator,
        live_theta=theta,
        live_log_likelihood=log_likelihood,
        live_log_prior=log_prior,
        live_log_q0=log_q0,
        live_log_psi0=log_psi0,
        live_tie_breakers=ties,
        worst=0,
        threshold=0.0,
        threshold_tie_breaker=ties[0],
        tie_policy="strict",
        sampler=SRWalkSampler(settings=SRWalkSettings(n_steps=7), ndim=1),
        rng=rng,  # type: ignore[arg-type]
        max_proposals=7,
        max_likelihood_calls=None,
        deadline=None,
    )
    assert attempt.draw is not None
    assert rng.normal_calls == 1
    assert attempt.n_completed == 7


def test_srwalk_uses_prepared_snapshot_factor_without_recomputing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator, theta, log_likelihood, log_prior, log_q0, log_psi0, ties = (
        _all_rejected_problem()
    )

    def fail_covariance(*args: Any, **kwargs: Any) -> NDArray[np.float64]:
        raise AssertionError("worker recomputed prepared covariance")

    monkeypatch.setattr("nismo.mcmc.covariance_factor", fail_covariance)
    attempt = draw_srwalk_constrained(
        evaluator=evaluator,
        live_theta=theta,
        live_log_likelihood=log_likelihood,
        live_log_prior=log_prior,
        live_log_q0=log_q0,
        live_log_psi0=log_psi0,
        live_tie_breakers=ties,
        worst=0,
        threshold=0.0,
        threshold_tie_breaker=ties[0],
        tie_policy="strict",
        sampler=SRWalkSampler(settings=SRWalkSettings(n_steps=4), ndim=1),
        rng=np.random.default_rng(199),
        max_proposals=4,
        max_likelihood_calls=None,
        deadline=None,
        proposal_factor=np.ones((1, 1)),
    )
    assert attempt.draw is not None
    assert attempt.n_completed == 4


def test_srwalk_prior_rejections_do_not_consume_likelihood_budget() -> None:
    class UniformBox:
        ndim = 1

        def log_prob(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.where(
                (theta[:, 0] >= 0.0) & (theta[:, 0] <= 1.0),
                0.0,
                -np.inf,
            )

    class OutsideGenerator:
        def choice(self, values: NDArray[np.int64]) -> int:
            return int(values[0])

        def standard_normal(self, *, size: tuple[int, int]) -> NDArray[np.float64]:
            return np.ones(size)

        def random(self) -> float:
            return 0.5

    proposal = UniformBox()
    model = CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=lambda theta: theta[:, 0],
        log_prior_fn=proposal.log_prob,
    )
    theta = np.array([[0.1], [0.2], [0.3]])
    log_likelihood = theta[:, 0]
    evaluator = BatchEvaluator(model, proposal)
    attempt = draw_srwalk_constrained(
        evaluator=evaluator,
        live_theta=theta,
        live_log_likelihood=log_likelihood,
        live_log_prior=np.zeros(3),
        live_log_q0=np.zeros(3),
        live_log_psi0=log_likelihood,
        live_tie_breakers=np.zeros(3),
        worst=0,
        threshold=0.1,
        threshold_tie_breaker=0.0,
        tie_policy="strict",
        sampler=SRWalkSampler(
            settings=SRWalkSettings(n_steps=3, scale=1.0),
            ndim=1,
        ),
        rng=OutsideGenerator(),  # type: ignore[arg-type]
        max_proposals=3,
        max_likelihood_calls=0,
        deadline=None,
        proposal_factor=np.array([[10.0]]),
    )
    assert attempt.draw is not None
    assert attempt.n_proposed == 3
    assert attempt.n_completed == 3
    assert evaluator.n_prior_calls == 3
    assert evaluator.n_likelihood_calls == 0


class _ScriptedGenerator:
    def __init__(
        self,
        *,
        start: int,
        normal_values: list[float],
        random_values: list[float],
    ):
        self.start = start
        self.normal_values = iter(normal_values)
        self.random_values = iter(random_values)

    def choice(
        self,
        values: NDArray[np.int64],
        size: int | None = None,
        replace: bool = True,
    ) -> int:
        assert size is None
        assert replace
        assert self.start in values
        return self.start

    def standard_normal(self, *, size: int) -> NDArray[np.float64]:
        return np.full(size, next(self.normal_values))

    def random(self, size: int | None = None) -> float:
        assert size is None
        return next(self.random_values)


def test_rejected_mh_proposal_retains_every_cached_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LinearLogQ:
        ndim = 1

        def log_prob(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
            return -2.0 * theta[:, 0]

    proposal = LinearLogQ()
    model = CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=lambda theta: 4.0 + theta[:, 0],
        log_prior_fn=proposal.log_prob,
    )
    theta = np.array([[-2.0], [0.0], [2.0]])
    log_q0 = proposal.log_prob(theta)
    log_likelihood = 4.0 + theta[:, 0]
    log_psi0 = np.array(log_likelihood, copy=True)
    ties = np.array([0.1, 0.2, 0.3])
    monkeypatch.setattr(
        "nismo.mcmc.bounding_ellipsoid_axes",
        lambda *args: np.ones((1, 1)),
    )
    sampler = RWalkSampler(settings=RWalkSettings(walks=2), ndim=1)
    attempt = draw_rwalk_constrained(
        evaluator=BatchEvaluator(model, proposal),
        live_theta=theta,
        live_log_likelihood=log_likelihood,
        live_log_prior=log_q0,
        live_log_q0=log_q0,
        live_log_psi0=log_psi0,
        live_tie_breakers=ties,
        worst=0,
        threshold=1.0,
        threshold_tie_breaker=ties[0],
        tie_policy="strict",
        sampler=sampler,
        rng=_ScriptedGenerator(  # type: ignore[arg-type]
            start=1,
            normal_values=[1.0, 1.0],
            random_values=[1.0, 0.9, 0.9, 1.0, 0.9, 0.9],
        ),
        max_proposals=2,
        max_likelihood_calls=None,
        deadline=None,
    )
    assert attempt.draw is not None
    point = attempt.draw.point
    np.testing.assert_array_equal(point.theta, theta[1])
    assert point.log_likelihood == log_likelihood[1]
    assert point.log_prior == log_q0[1]
    assert point.log_q0 == log_q0[1]
    assert point.log_psi0 == log_psi0[1]
    assert point.tie_breaker == ties[1]
    assert attempt.n_valid == 2
    assert attempt.n_accepted == 0


def test_accepted_mh_proposal_updates_every_cached_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LinearLogQ:
        ndim = 1

        def log_prob(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
            return -2.0 * theta[:, 0]

    proposal = LinearLogQ()
    model = CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=lambda theta: 4.0 + theta[:, 0],
        log_prior_fn=proposal.log_prob,
    )
    theta = np.array([[-2.0], [1.0], [2.0]])
    log_q0 = proposal.log_prob(theta)
    log_likelihood = 4.0 + theta[:, 0]
    log_psi0 = np.array(log_likelihood, copy=True)
    ties = np.array([0.1, 0.2, 0.3])
    monkeypatch.setattr(
        "nismo.mcmc.bounding_ellipsoid_axes",
        lambda *args: np.full((1, 1), 10.0),
    )
    sampler = RWalkSampler(settings=RWalkSettings(walks=2), ndim=1)
    attempt = draw_rwalk_constrained(
        evaluator=BatchEvaluator(model, proposal),
        live_theta=theta,
        live_log_likelihood=log_likelihood,
        live_log_prior=log_q0,
        live_log_q0=log_q0,
        live_log_psi0=log_psi0,
        live_tie_breakers=ties,
        worst=0,
        threshold=1.0,
        threshold_tie_breaker=ties[0],
        tie_policy="strict",
        sampler=sampler,
        rng=_ScriptedGenerator(  # type: ignore[arg-type]
            start=1,
            normal_values=[-1.0, -10.0],
            random_values=[0.1, 0.8, 0.999, 1.0, 0.5],
        ),
        max_proposals=2,
        max_likelihood_calls=None,
        deadline=None,
    )
    assert attempt.draw is not None
    point = attempt.draw.point
    np.testing.assert_array_equal(point.theta, [0.0])
    assert point.log_likelihood == 4.0
    assert point.log_prior == 0.0
    assert point.log_q0 == 0.0
    assert point.log_psi0 == 4.0
    assert point.tie_breaker == 0.8
    assert attempt.n_valid == 1
    assert attempt.n_accepted == 1
    assert attempt.n_moved == 1


def test_ensemble_walk_has_exact_counts_and_reproducible_output() -> None:
    first = _all_rejected_problem()
    second = _all_rejected_problem()
    settings = EnsembleRWalkSettings(
        n_walkers=4,
        n_sweeps=3,
        move_weights=EnsembleMoveWeights(de=1, stretch=0, gaussian=0),
    )
    attempts = []
    for problem in (first, second):
        evaluator, theta, log_likelihood, log_prior, log_q0, log_psi0, ties = problem
        attempts.append(
            draw_ensemble_rwalk_constrained(
                evaluator=evaluator,
                live_theta=theta,
                live_log_likelihood=log_likelihood,
                live_log_prior=log_prior,
                live_log_q0=log_q0,
                live_log_psi0=log_psi0,
                live_tie_breakers=ties,
                worst=0,
                threshold=0.0,
                threshold_tie_breaker=ties[0],
                tie_policy="strict",
                settings=settings,
                rng=np.random.default_rng(73),
                max_proposals=12,
                max_likelihood_calls=None,
                deadline=None,
            )
        )
        assert evaluator.n_likelihood_calls == 12
    assert attempts[0].draw is not None
    assert attempts[1].draw is not None
    np.testing.assert_array_equal(
        attempts[0].draw.point.theta,
        attempts[1].draw.point.theta,
    )
    assert attempts[0].n_proposed == 12
    assert attempts[0].n_completed == 3
    assert attempts[0].n_valid == 0
    assert attempts[0].n_accepted == 0
    assert attempts[0].n_moved == 0
    assert attempts[0].draw.point.theta[0] == 3.0
    assert tuple(stat.name for stat in attempts[0].ensemble_move_stats) == (
        "de",
        "stretch",
        "gaussian",
    )
    assert sum(stat.n_proposed for stat in attempts[0].ensemble_move_stats) == 12


@pytest.mark.parametrize(
    ("weights", "expected_covariance_calls"),
    [
        (EnsembleMoveWeights(de=1, stretch=0, gaussian=0), 1),
        (EnsembleMoveWeights(de=0, stretch=1, gaussian=0), 0),
        (EnsembleMoveWeights(de=0, stretch=0, gaussian=1), 1),
    ],
)
def test_ensemble_covariance_is_lazy_frozen_and_excludes_discarded_point(
    monkeypatch: pytest.MonkeyPatch,
    weights: EnsembleMoveWeights,
    expected_covariance_calls: int,
) -> None:
    evaluator, theta, log_likelihood, log_prior, log_q0, log_psi0, ties = (
        _all_rejected_problem()
    )
    covariance_calls = 0

    def counted_covariance(
        points: NDArray[np.float64],
        *,
        shrinkage: float,
        jitter: float,
    ) -> NDArray[np.float64]:
        nonlocal covariance_calls
        covariance_calls += 1
        np.testing.assert_array_equal(points, theta[1:])
        assert shrinkage == 0.2
        assert jitter == 1.0e-8
        return np.ones((1, 1))

    monkeypatch.setattr("nismo.mcmc.covariance_factor", counted_covariance)
    attempt = draw_ensemble_rwalk_constrained(
        evaluator=evaluator,
        live_theta=theta,
        live_log_likelihood=log_likelihood,
        live_log_prior=log_prior,
        live_log_q0=log_q0,
        live_log_psi0=log_psi0,
        live_tie_breakers=ties,
        worst=0,
        threshold=0.0,
        threshold_tie_breaker=ties[0],
        tie_policy="strict",
        settings=EnsembleRWalkSettings(
            n_walkers=4,
            n_sweeps=3,
            move_weights=weights,
            covariance_shrinkage=0.2,
            covariance_jitter=1.0e-8,
        ),
        rng=np.random.default_rng(709),
        max_proposals=12,
        max_likelihood_calls=None,
        deadline=None,
    )
    assert attempt.draw is not None
    assert covariance_calls == expected_covariance_calls
    assert attempt.n_proposed == 12
    assert evaluator.n_likelihood_calls == 12
    assert sum(stat.n_proposed for stat in attempt.ensemble_move_stats) == 12
    assert sum(stat.n_valid for stat in attempt.ensemble_move_stats) == 0
    assert sum(stat.n_accepted for stat in attempt.ensemble_move_stats) == 0


@pytest.mark.parametrize(
    ("configured_scale", "expected_scale"),
    [(None, 2.38), (0.125, 0.125)],
)
def test_ensemble_gaussian_scale_is_resolved_once_and_frozen(
    monkeypatch: pytest.MonkeyPatch,
    configured_scale: float | None,
    expected_scale: float,
) -> None:
    evaluator, theta, log_likelihood, log_prior, log_q0, log_psi0, ties = (
        _all_rejected_problem()
    )
    observed_scales: list[float] = []

    def captured_gaussian(
        *,
        ensemble_theta: NDArray[np.float64],
        active: NDArray[np.int64],
        gaussian_scale: float,
        factor: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> Any:
        observed_scales.append(gaussian_scale)
        return _propose_gaussian_move(
            ensemble_theta=ensemble_theta,
            active=active,
            gaussian_scale=gaussian_scale,
            factor=factor,
            rng=rng,
        )

    monkeypatch.setattr("nismo.mcmc._propose_gaussian_move", captured_gaussian)
    monkeypatch.setattr(
        "nismo.mcmc.covariance_factor",
        lambda *args, **kwargs: np.ones((1, 1)),
    )
    attempt = draw_ensemble_rwalk_constrained(
        evaluator=evaluator,
        live_theta=theta,
        live_log_likelihood=log_likelihood,
        live_log_prior=log_prior,
        live_log_q0=log_q0,
        live_log_psi0=log_psi0,
        live_tie_breakers=ties,
        worst=0,
        threshold=0.0,
        threshold_tie_breaker=ties[0],
        tie_policy="strict",
        settings=EnsembleRWalkSettings(
            n_walkers=4,
            n_sweeps=2,
            move_weights=EnsembleMoveWeights(de=0, stretch=0, gaussian=1),
            gaussian_scale=configured_scale,
        ),
        rng=np.random.default_rng(219),
        max_proposals=8,
        max_likelihood_calls=None,
        deadline=None,
    )
    assert attempt.draw is not None
    assert observed_scales == pytest.approx([expected_scale] * 4)


def test_ensemble_selects_one_move_per_half_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator, theta, log_likelihood, log_prior, log_q0, log_psi0, ties = (
        _all_rejected_problem()
    )
    selections = 0

    def counted_selection(**kwargs: Any) -> str:
        nonlocal selections
        selections += 1
        return "stretch"

    monkeypatch.setattr("nismo.mcmc._select_ensemble_move", counted_selection)
    attempt = draw_ensemble_rwalk_constrained(
        evaluator=evaluator,
        live_theta=theta,
        live_log_likelihood=log_likelihood,
        live_log_prior=log_prior,
        live_log_q0=log_q0,
        live_log_psi0=log_psi0,
        live_tie_breakers=ties,
        worst=0,
        threshold=0.0,
        threshold_tie_breaker=ties[0],
        tie_policy="strict",
        settings=EnsembleRWalkSettings(n_walkers=4, n_sweeps=4),
        rng=np.random.default_rng(761),
        max_proposals=16,
        max_likelihood_calls=None,
        deadline=None,
    )
    assert attempt.draw is not None
    assert selections == 8
    assert attempt.ensemble_move_stats[1].n_proposed == 16


def test_gaussian_ensemble_acceptance_updates_all_cached_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FlatProposal:
        ndim = 1

        def log_prob(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.zeros(len(theta))

    proposal = FlatProposal()
    model = CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=lambda theta: 10.0 + theta[:, 0],
        log_prior_fn=proposal.log_prob,
    )
    theta = np.arange(6.0)[:, np.newaxis]
    log_likelihood = 10.0 + theta[:, 0]
    zeros = np.zeros(6)
    ties = np.linspace(0.1, 0.6, 6)
    monkeypatch.setattr(
        "nismo.mcmc.covariance_factor",
        lambda *args, **kwargs: np.ones((1, 1)),
    )
    attempt = draw_ensemble_rwalk_constrained(
        evaluator=BatchEvaluator(model, proposal),
        live_theta=theta,
        live_log_likelihood=log_likelihood,
        live_log_prior=zeros,
        live_log_q0=zeros,
        live_log_psi0=log_likelihood,
        live_tie_breakers=ties,
        worst=0,
        threshold=-100.0,
        threshold_tie_breaker=ties[0],
        tie_policy="strict",
        settings=EnsembleRWalkSettings(
            n_walkers=4,
            n_sweeps=1,
            move_weights=EnsembleMoveWeights(de=0, stretch=0, gaussian=1),
            gaussian_scale=0.01,
        ),
        rng=np.random.default_rng(909),
        max_proposals=4,
        max_likelihood_calls=None,
        deadline=None,
    )
    assert attempt.draw is not None
    point = attempt.draw.point
    assert point.log_likelihood == pytest.approx(10.0 + point.theta[0])
    assert point.log_prior == 0.0
    assert point.log_q0 == 0.0
    assert point.log_psi0 == pytest.approx(10.0 + point.theta[0])
    assert attempt.n_valid == 4
    assert attempt.n_accepted == 4
    assert attempt.n_moved == 4


def test_gaussian_ensemble_rejection_retains_all_cached_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator, theta, log_likelihood, log_prior, log_q0, log_psi0, ties = (
        _all_rejected_problem()
    )
    monkeypatch.setattr(
        "nismo.mcmc.covariance_factor",
        lambda *args, **kwargs: np.ones((1, 1)),
    )
    attempt = draw_ensemble_rwalk_constrained(
        evaluator=evaluator,
        live_theta=theta,
        live_log_likelihood=log_likelihood,
        live_log_prior=log_prior,
        live_log_q0=log_q0,
        live_log_psi0=log_psi0,
        live_tie_breakers=ties,
        worst=0,
        threshold=0.0,
        threshold_tie_breaker=ties[0],
        tie_policy="strict",
        settings=EnsembleRWalkSettings(
            n_walkers=4,
            n_sweeps=2,
            move_weights=EnsembleMoveWeights(de=0, stretch=0, gaussian=1),
        ),
        rng=np.random.default_rng(512),
        max_proposals=8,
        max_likelihood_calls=None,
        deadline=None,
    )
    assert attempt.draw is not None
    point = attempt.draw.point
    matching = np.flatnonzero(theta[:, 0] == point.theta[0])
    assert len(matching) == 1
    index = int(matching[0])
    assert point.log_likelihood == log_likelihood[index]
    assert point.log_prior == log_prior[index]
    assert point.log_q0 == log_q0[index]
    assert point.log_psi0 == log_psi0[index]
    assert point.tie_breaker == ties[index]
    assert attempt.n_valid == 0
    assert attempt.n_accepted == 0
    assert attempt.n_moved == 0


def test_mcmc_preflight_does_not_start_a_shortened_evolution() -> None:
    evaluator, theta, log_likelihood, log_prior, log_q0, log_psi0, ties = (
        _all_rejected_problem()
    )
    attempt = draw_rwalk_constrained(
        evaluator=evaluator,
        live_theta=theta,
        live_log_likelihood=log_likelihood,
        live_log_prior=log_prior,
        live_log_q0=log_q0,
        live_log_psi0=log_psi0,
        live_tie_breakers=ties,
        worst=0,
        threshold=0.0,
        threshold_tie_breaker=ties[0],
        tie_policy="strict",
        sampler=RWalkSampler(settings=RWalkSettings(walks=5), ndim=1),
        rng=np.random.default_rng(9),
        max_proposals=4,
        max_likelihood_calls=None,
        deadline=None,
    )
    assert attempt.draw is None
    assert attempt.reason == "max_proposals_per_replacement"
    assert attempt.n_proposed == 0
    assert evaluator.n_likelihood_calls == 0


def test_srwalk_preflight_does_not_start_a_shortened_evolution() -> None:
    evaluator, theta, log_likelihood, log_prior, log_q0, log_psi0, ties = (
        _all_rejected_problem()
    )
    attempt = draw_srwalk_constrained(
        evaluator=evaluator,
        live_theta=theta,
        live_log_likelihood=log_likelihood,
        live_log_prior=log_prior,
        live_log_q0=log_q0,
        live_log_psi0=log_psi0,
        live_tie_breakers=ties,
        worst=0,
        threshold=0.0,
        threshold_tie_breaker=ties[0],
        tie_policy="strict",
        sampler=SRWalkSampler(settings=SRWalkSettings(n_steps=5), ndim=1),
        rng=np.random.default_rng(9),
        max_proposals=4,
        max_likelihood_calls=None,
        deadline=None,
    )
    assert attempt.draw is None
    assert attempt.reason == "max_proposals_per_replacement"
    assert attempt.n_proposed == 0
    assert evaluator.n_likelihood_calls == 0
