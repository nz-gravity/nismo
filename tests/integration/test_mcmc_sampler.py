from __future__ import annotations

import numpy as np
import pytest
from tests.helpers import StandardNormalProposal

from nismo import (
    CallableModel,
    EnsembleMoveWeights,
    EnsembleRWalkSettings,
    MORWalkSettings,
    NISMOSampler,
    RWalkSettings,
    SRWalkSettings,
)

pytestmark = pytest.mark.integration


def _constant_problem() -> tuple[CallableModel, StandardNormalProposal]:
    proposal = StandardNormalProposal()
    model = CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=lambda theta: np.full(len(theta), np.log(2.5)),
        log_prior_fn=proposal.log_prob,
    )
    return model, proposal


@pytest.mark.parametrize(
    ("proposal_scheme", "settings"),
    [
        ("rwalk", RWalkSettings(walks=4)),
        ("s-rwalk", SRWalkSettings(n_steps=4)),
        (
            "en-rwalk",
            EnsembleRWalkSettings(n_walkers=4, n_sweeps=1),
        ),
    ],
)
def test_mcmc_samplers_are_reproducible_and_report_separate_diagnostics(
    proposal_scheme: str,
    settings: RWalkSettings | SRWalkSettings | EnsembleRWalkSettings,
) -> None:
    model, proposal = _constant_problem()
    if isinstance(settings, RWalkSettings):
        kwargs = {"rwalk_settings": settings}
    elif isinstance(settings, SRWalkSettings):
        kwargs = {"srwalk_settings": settings}
    else:
        kwargs = {"ensemble_rwalk_settings": settings}
    results = [
        NISMOSampler(
            model=model,
            importance_morph=proposal,
            proposal_scheme=proposal_scheme,  # type: ignore[arg-type]
            n_live=10,
            rng=88,
            tie_policy="randomized_plateau",
            **kwargs,  # type: ignore[arg-type]
        ).run(
            dlogz=0.5,
            max_iterations=100,
            max_proposals_per_replacement=20,
        )
        for _ in range(2)
    ]
    first, second = results
    assert first.success
    assert first.termination_reason == "remaining_evidence"
    assert first.logz == pytest.approx(np.log(2.5), abs=1.0e-12)
    assert first.niter == 10
    assert first.n_proposals == 40
    assert first.n_likelihood_calls == 50
    assert np.all(np.diff(first.dead_log_psi0) >= 0.0)
    assert np.sum(first.posterior_weights) == pytest.approx(1.0)
    assert np.all(np.isfinite(first.history.mh_acceptance_fraction))
    assert np.all(np.isfinite(first.history.constraint_pass_fraction))
    assert np.all(first.history.mcmc_accepted >= 0)
    assert np.all(first.history.mcmc_moved >= 0)
    np.testing.assert_array_equal(first.dead_points, second.dead_points)
    np.testing.assert_array_equal(
        first.dead_tie_breakers,
        second.dead_tie_breakers,
    )
    np.testing.assert_array_equal(
        first.history.mh_acceptance_fraction,
        second.history.mh_acceptance_fraction,
    )
    if proposal_scheme == "en-rwalk":
        move_history = first.ensemble_move_history
        assert move_history is not None
        assert move_history.names == ("de", "stretch", "gaussian")
        assert move_history.proposed.shape == (first.niter, 3)
        assert not move_history.proposed.flags.writeable
        assert not move_history.valid.flags.writeable
        assert not move_history.accepted.flags.writeable
        assert not move_history.moved.flags.writeable
        np.testing.assert_array_equal(
            np.sum(move_history.proposed, axis=1),
            first.history.proposals,
        )
        np.testing.assert_array_equal(
            np.sum(move_history.accepted, axis=1),
            first.history.mcmc_accepted,
        )
    else:
        assert first.ensemble_move_history is None


def test_mixed_ensemble_moves_report_exact_result_level_counts() -> None:
    model, proposal = _constant_problem()
    result = NISMOSampler(
        model=model,
        importance_morph=proposal,
        proposal_scheme="en-rwalk",
        ensemble_rwalk_settings=EnsembleRWalkSettings(
            n_walkers=4,
            n_sweeps=3,
            move_weights=EnsembleMoveWeights(de=0.6, stretch=0.25, gaussian=0.15),
        ),
        n_live=10,
        rng=101,
        tie_policy="randomized_plateau",
    ).run(
        dlogz=0.5,
        max_iterations=100,
        max_proposals_per_replacement=12,
    )
    assert result.success
    assert result.n_proposals == result.niter * 12
    assert result.n_likelihood_calls == result.nlive + result.n_proposals
    move_history = result.ensemble_move_history
    assert move_history is not None
    np.testing.assert_array_equal(
        np.sum(move_history.proposed, axis=1),
        result.history.proposals,
    )
    np.testing.assert_array_equal(
        np.sum(move_history.accepted, axis=1),
        result.history.mcmc_accepted,
    )
    expected_valid = np.rint(
        result.history.constraint_pass_fraction * result.history.proposals
    ).astype(np.int64)
    np.testing.assert_array_equal(np.sum(move_history.valid, axis=1), expected_valid)


def test_non_mcmc_history_keeps_nested_efficiency_semantics() -> None:
    model, proposal = _constant_problem()
    result = NISMOSampler(
        model=model,
        importance_morph=proposal,
        n_live=10,
        rng=88,
        tie_policy="randomized_plateau",
    ).run(
        dlogz=0.5,
        max_iterations=100,
    )
    assert np.all(np.isnan(result.history.mh_acceptance_fraction))
    assert np.all(np.isnan(result.history.constraint_pass_fraction))
    assert np.all(result.history.mcmc_accepted == 0)
    assert np.all(result.history.mcmc_moved == 0)
    assert np.all(result.history.mcmc_completed == 0)
    assert result.ensemble_move_history is None
    np.testing.assert_allclose(
        result.history.acceptance_fraction,
        np.arange(1, result.niter + 1) / np.cumsum(result.history.proposals),
    )


def test_mor_rwalk_uses_one_morph_batch_then_switches_to_srwalk() -> None:
    model, proposal = _constant_problem()
    results = [
        NISMOSampler(
            model=model,
            importance_morph=proposal,
            proposal_scheme="mor-rwalk",
            mor_rwalk_settings=MORWalkSettings(n_proposals=11),
            srwalk_settings=SRWalkSettings(n_steps=4),
            n_live=10,
            rng=88,
            tie_policy="randomized_plateau",
        ).run(
            dlogz=0.5,
            max_iterations=100,
            max_proposals_per_replacement=20,
        )
        for _ in range(2)
    ]
    first, second = results

    assert first.success
    assert first.niter == 10
    assert first.n_likelihood_calls == 47
    assert first.n_proposals == 37
    assert np.isnan(first.history.mh_acceptance_fraction[0])
    assert first.history.mcmc_completed[0] == 0
    assert np.all(np.isfinite(first.history.mh_acceptance_fraction[1:]))
    assert np.all(first.history.mcmc_completed[1:] == 4)
    np.testing.assert_array_equal(first.dead_points, second.dead_points)
    np.testing.assert_array_equal(
        first.history.mcmc_completed,
        second.history.mcmc_completed,
    )


def test_mcmc_likelihood_limit_never_returns_a_shortened_chain() -> None:
    model, proposal = _constant_problem()
    result = NISMOSampler(
        model=model,
        importance_morph=proposal,
        proposal_scheme="rwalk",
        rwalk_settings=RWalkSettings(walks=4),
        n_live=10,
        rng=89,
        tie_policy="randomized_plateau",
    ).run(
        dlogz=0.1,
        max_iterations=100,
        max_likelihood_calls=13,
        max_proposals_per_replacement=20,
    )
    assert not result.success
    assert result.termination_reason == "max_likelihood_calls"
    assert result.niter == 0
    assert result.n_likelihood_calls == 10
    assert result.n_proposals == 0


def test_rwalk_exposes_only_the_skilling_method_citation() -> None:
    model, proposal = _constant_problem()
    sampler = NISMOSampler(
        model=model,
        importance_morph=proposal,
        proposal_scheme="rwalk",
        n_live=10,
        rng=90,
    )
    assert sampler.citations == [
        ("Skilling (2006)", "projecteuclid.org/euclid.ba/1340370944")
    ]


def test_rwalk_default_uses_ndim_plus_twenty_walks() -> None:
    model, proposal = _constant_problem()
    result = NISMOSampler(
        model=model,
        importance_morph=proposal,
        proposal_scheme="rwalk",
        n_live=10,
        rng=91,
        tie_policy="randomized_plateau",
    ).run(
        dlogz=0.1,
        max_iterations=1,
        max_proposals_per_replacement=21,
    )
    assert result.niter == 1
    assert result.n_proposals == 21
    assert result.history.mcmc_completed[0] == 21


def test_srwalk_profiling_reports_rolling_geometry_and_component_timings() -> None:
    model, proposal = _constant_problem()
    result = NISMOSampler(
        model=model,
        importance_morph=proposal,
        proposal_scheme="s-rwalk",
        srwalk_settings=SRWalkSettings(
            n_steps=4,
            covariance_update_interval=3,
            covariance_rebuild_interval=5,
            profile=True,
        ),
        n_live=10,
        rng=190,
        tie_policy="randomized_plateau",
    ).run(
        dlogz=0.5,
        max_iterations=100,
        max_proposals_per_replacement=20,
    )
    diagnostics = result.srwalk_diagnostics
    assert diagnostics is not None
    assert diagnostics.geometry_updates == result.niter
    assert diagnostics.geometry_rebuilds == result.niter // 5
    assert diagnostics.factor_refreshes == 4
    assert diagnostics.completed_walks == result.niter
    assert diagnostics.completed_candidates == result.niter
    assert diagnostics.stale_candidates == 0
    assert diagnostics.factorization_seconds >= 0.0
    assert diagnostics.proposal_linear_algebra_seconds >= 0.0
    assert diagnostics.prior_seconds >= 0.0
    assert diagnostics.likelihood_seconds >= 0.0
    assert diagnostics.q0_seconds >= 0.0
    assert diagnostics.mean_squared_displacement >= 0.0
    assert diagnostics.stale_candidate_fraction == 0.0


def test_srwalk_diagnostics_are_opt_in() -> None:
    model, proposal = _constant_problem()
    result = NISMOSampler(
        model=model,
        importance_morph=proposal,
        proposal_scheme="s-rwalk",
        srwalk_settings=SRWalkSettings(n_steps=4),
        n_live=10,
        rng=191,
        tie_policy="randomized_plateau",
    ).run(
        dlogz=0.5,
        max_iterations=100,
        max_proposals_per_replacement=20,
    )
    assert result.srwalk_diagnostics is None
