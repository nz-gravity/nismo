from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from tests.helpers import StandardNormalProposal

from nismo import (
    CallableModel,
    ConfigurationError,
    NISMOSampler,
    StoppingCriterionConfig,
    StoppingPolicy,
)
from nismo.diagnostics import summarize

pytestmark = pytest.mark.integration


def _constant_problem() -> tuple[CallableModel, StandardNormalProposal]:
    proposal = StandardNormalProposal()
    model = CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=lambda x: np.full(len(x), np.log(2.5)),
        log_prior_fn=proposal.log_prob,
    )
    return model, proposal


def test_constant_integrand_end_to_end_with_randomized_plateau() -> None:
    model, proposal = _constant_problem()
    result = NISMOSampler(
        model=model,
        importance_morph=proposal,
        n_live=20,
        rng=8,
        tie_policy="randomized_plateau",
    ).run(
        dlogz=0.2,
        max_iterations=200,
        max_proposals_per_replacement=10_000,
    )
    assert result.success
    assert result.termination_reason == "remaining_evidence"
    assert result.config.stopping is not None
    assert result.config.stopping.criteria[0].name == "remaining_dlogz"
    assert result.history.remaining_dlogz[-1] <= 0.2
    expected_remaining_dlogz = np.logaddexp(
        0.0,
        result.history.logz_live[-1] - result.history.logz_dead[-1],
    )
    assert result.history.remaining_dlogz[-1] == pytest.approx(expected_remaining_dlogz)
    assert result.logz == pytest.approx(np.log(2.5), abs=1e-12)
    assert result.information == pytest.approx(0.0, abs=1e-12)
    assert result.niter == 35
    assert np.sum(result.posterior_weights) == pytest.approx(1.0)
    assert not result.dead_points.flags.writeable
    assert summarize(result).thresholds_monotone


def test_hybrid_stopping_policy_succeeds_and_records_complete_state() -> None:
    model, proposal = _constant_problem()
    snapshots: list[dict[str, float | int]] = []
    policy = StoppingPolicy(
        criteria=(
            StoppingCriterionConfig("live_logz_error", 5.0e-3),
            StoppingCriterionConfig("remaining_fraction", 0.5),
            StoppingCriterionConfig("logz_stability", 5.0e-3),
        ),
        mode="all",
        consecutive=3,
        min_iterations=10,
        stability_window=10,
    )
    result = NISMOSampler(
        model=model,
        importance_morph=proposal,
        n_live=20,
        rng=81,
        tie_policy="randomized_plateau",
    ).run(
        stopping=policy,
        max_iterations=200,
        progress=lambda info: snapshots.append(dict(info)),
    )
    assert result.success
    assert result.termination_reason == "stopping_criteria"
    assert result.logz == pytest.approx(np.log(2.5), abs=1.0e-12)
    assert result.config.dlogz is None
    assert result.config.stopping == policy
    assert result.history.live_logz_error[-1] <= 5.0e-3
    assert result.history.remaining_fraction[-1] <= 0.5
    assert result.history.logz_stability[-1] <= 5.0e-3
    assert result.history.stopping_streak[-1] >= policy.consecutive
    expected_flags = {
        "criterion_live_logz_error_met",
        "criterion_remaining_fraction_met",
        "criterion_logz_stability_met",
    }
    assert expected_flags <= snapshots[-1].keys()
    assert all(snapshots[-1][key] == 1 for key in expected_flags)
    assert "criterion_live_ess_met" not in snapshots[-1]
    for name in (
        "remaining_dlogz",
        "live_ess",
        "live_mean_rse",
        "live_logz_error",
        "logz_stability",
        "stopping_streak",
    ):
        values = getattr(result.history, name)
        assert values.shape == (result.niter,)
        assert not values.flags.writeable
    diagnostics = summarize(result)
    assert diagnostics.final_remaining_dlogz == pytest.approx(
        result.history.remaining_dlogz[-1]
    )
    assert diagnostics.final_live_logz_error == pytest.approx(
        result.history.live_logz_error[-1]
    )
    assert diagnostics.final_stopping_streak == result.history.stopping_streak[-1]


def test_all_waits_for_every_criterion_while_any_stops_on_first() -> None:
    model, proposal = _constant_problem()
    criteria = (
        StoppingCriterionConfig("live_logz_error", 1.0e-12),
        StoppingCriterionConfig("remaining_fraction", 0.5),
    )
    all_result = NISMOSampler(
        model=model,
        importance_morph=proposal,
        n_live=20,
        rng=82,
        tie_policy="randomized_plateau",
    ).run(
        stopping=StoppingPolicy(criteria=criteria, mode="all"),
        max_iterations=100,
    )
    any_result = NISMOSampler(
        model=model,
        importance_morph=proposal,
        n_live=20,
        rng=82,
        tie_policy="randomized_plateau",
    ).run(
        stopping=StoppingPolicy(criteria=criteria, mode="any"),
        max_iterations=100,
    )
    assert all_result.success
    assert any_result.success
    assert all_result.niter == 14
    assert any_result.niter == 1


def test_hard_limit_remains_failure_after_only_one_all_criterion_passes() -> None:
    model, proposal = _constant_problem()
    result = NISMOSampler(
        model=model,
        importance_morph=proposal,
        n_live=20,
        rng=83,
        tie_policy="randomized_plateau",
    ).run(
        stopping=StoppingPolicy(
            criteria=(
                StoppingCriterionConfig("live_logz_error", 1.0e-12),
                StoppingCriterionConfig("logz_stability", 1.0e-12),
                StoppingCriterionConfig("remaining_dlogz", 1.0e-8),
            ),
            mode="all",
            stability_window=2,
        ),
        max_iterations=3,
    )
    assert not result.success
    assert result.termination_reason == "max_iterations"
    assert result.history.live_logz_error[-1] == 0.0
    assert result.history.logz_stability[-1] == pytest.approx(0.0, abs=1.0e-12)
    expected = -np.log1p(-np.exp(-result.niter / result.nlive))
    assert result.history.remaining_dlogz[-1] == pytest.approx(expected)
    assert result.history.remaining_dlogz[-1] > 1.0e-8
    assert result.history.stopping_streak[-1] == 0


def test_run_rejects_simultaneous_legacy_and_policy_arguments() -> None:
    model, proposal = _constant_problem()
    with pytest.raises(ConfigurationError, match="dlogz and stopping"):
        NISMOSampler(model=model, importance_morph=proposal, n_live=10, rng=84).run(
            dlogz=0.1,
            stopping=StoppingPolicy(
                criteria=(StoppingCriterionConfig("logzerr", 0.1),)
            ),
        )


def test_same_seed_reproduces_scientific_result() -> None:
    model, proposal = _constant_problem()
    first = NISMOSampler(
        model=model,
        importance_morph=proposal,
        n_live=12,
        rng=123,
        tie_policy="randomized_plateau",
    ).run(dlogz=0.25, max_iterations=100)
    second = NISMOSampler(
        model=model,
        importance_morph=proposal,
        n_live=12,
        rng=123,
        tie_policy="randomized_plateau",
    ).run(dlogz=0.25, max_iterations=100)
    assert first.logz == second.logz
    np.testing.assert_array_equal(first.dead_points, second.dead_points)
    np.testing.assert_array_equal(first.dead_tie_breakers, second.dead_tie_breakers)
    np.testing.assert_array_equal(
        first.log_posterior_weights, second.log_posterior_weights
    )


def test_strict_plateau_returns_partial_failed_result() -> None:
    model, proposal = _constant_problem()
    result = NISMOSampler(
        model=model,
        importance_morph=proposal,
        n_live=10,
        rng=4,
        tie_policy="strict",
        proposal_batch_size=4,
    ).run(
        dlogz=0.1,
        max_iterations=100,
        max_proposals_per_replacement=12,
    )
    assert not result.success
    assert result.termination_reason == "plateau_stall"
    assert result.niter == 0
    assert result.n_proposals == 12
    assert result.logz == pytest.approx(np.log(2.5), abs=1e-12)
    diagnostics = summarize(result)
    assert np.isnan(diagnostics.final_remaining_fraction)
    assert np.isnan(diagnostics.final_remaining_dlogz)
    assert np.isnan(diagnostics.final_live_ess)
    assert result.history.remaining_dlogz.shape == (0,)
    assert not result.history.remaining_dlogz.flags.writeable
    assert diagnostics.final_stopping_streak == 0


def test_iteration_limit_is_not_scientific_success() -> None:
    proposal = StandardNormalProposal()
    model = CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=lambda x: -0.5 * (x[:, 0] - 1.0) ** 2,
        log_prior_fn=proposal.log_prob,
    )
    result = NISMOSampler(model=model, importance_morph=proposal, n_live=15, rng=2).run(
        dlogz=1e-8,
        max_iterations=3,
        max_proposals_per_replacement=1_000,
    )
    assert not result.success
    assert result.termination_reason == "max_iterations"
    assert result.niter == 3
    assert result.history.iteration.tolist() == [1, 2, 3]
    assert result.warnings == ()


def test_likelihood_call_limit_is_not_scientific_success() -> None:
    proposal = StandardNormalProposal()
    model = CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=lambda x: -0.5 * (x[:, 0] - 0.5) ** 2,
        log_prior_fn=proposal.log_prob,
    )
    result = NISMOSampler(
        model=model,
        importance_morph=proposal,
        n_live=10,
        rng=19,
        proposal_batch_size=4,
    ).run(
        dlogz=1e-8,
        max_iterations=100,
        max_likelihood_calls=14,
        max_proposals_per_replacement=100,
    )
    assert not result.success
    assert result.termination_reason == "max_likelihood_calls"
    assert result.n_likelihood_calls == 14


def test_wall_time_limit_returns_initialized_partial_result() -> None:
    model, proposal = _constant_problem()
    result = NISMOSampler(
        model=model,
        importance_morph=proposal,
        n_live=10,
        rng=7,
        tie_policy="randomized_plateau",
    ).run(
        dlogz=0.1,
        max_iterations=100,
        max_wall_time=1e-12,
    )
    assert not result.success
    assert result.termination_reason == "max_wall_time"
    assert result.niter == 0
    assert result.logz == pytest.approx(np.log(2.5), abs=1e-12)


def test_progress_callback_receives_standard_nested_sampling_information() -> None:
    model, proposal = _constant_problem()
    snapshots: list[dict[str, float | int]] = []
    result = NISMOSampler(
        model=model,
        importance_morph=proposal,
        n_live=10,
        rng=31,
        tie_policy="randomized_plateau",
    ).run(
        dlogz=0.3,
        max_iterations=100,
        progress=lambda info: snapshots.append(dict(info)),
    )
    required = {
        "iteration",
        "nlive",
        "likelihood_calls",
        "proposals",
        "efficiency_percent",
        "logz",
        "logzerr",
        "information",
        "remaining_fraction",
        "remaining_dlogz",
        "live_ess",
        "live_mean_rse",
        "live_logz_error",
        "logz_stability",
        "stopping_streak",
        "stopping_consecutive",
        "criterion_remaining_dlogz_met",
        "stopping_tolerance",
        "threshold",
        "elapsed_seconds",
        "proposal_revision",
        "proposal_update_attempts",
        "proposal_update_failures",
    }
    assert len(snapshots) == result.niter
    assert required <= snapshots[-1].keys()
    assert snapshots[-1]["logz"] == pytest.approx(result.logz)
    assert snapshots[-1]["logzerr"] == pytest.approx(result.logzerr, abs=1e-8)
    assert snapshots[-1]["information"] == pytest.approx(result.information, abs=1e-12)
    assert snapshots[-1]["remaining_dlogz"] <= result.config.dlogz
    assert result.history.logzerr[-1] == pytest.approx(result.logzerr, abs=1e-8)
    assert result.history.information[-1] == pytest.approx(
        result.information, abs=1e-12
    )


def test_progress_true_renders_standard_terminal_fields(capsys: Any) -> None:
    model, proposal = _constant_problem()
    result = NISMOSampler(
        model=model,
        importance_morph=proposal,
        n_live=8,
        rng=32,
        tie_policy="randomized_plateau",
    ).run(
        dlogz=0.4,
        max_iterations=50,
        progress=True,
    )
    captured = capsys.readouterr()
    assert result.success
    assert "logZ" in captured.err
    assert "logZerr" in captured.err
    assert "ncall" in captured.err
    assert "eff" in captured.err
    assert "dlogZrem" in captured.err
    assert " liveErr=" not in captured.err
    assert " ESSlive=" not in captured.err
    assert " rem=" not in captured.err
    assert "prop=0" not in captured.err
    assert "%|" not in captured.err
    assert "/50" not in captured.err


def test_terminal_progress_shows_only_enabled_criterion_metrics(
    capsys: Any,
) -> None:
    model, proposal = _constant_problem()
    result = NISMOSampler(
        model=model,
        importance_morph=proposal,
        n_live=8,
        rng=321,
        tie_policy="randomized_plateau",
    ).run(
        stopping=StoppingPolicy(
            criteria=(
                StoppingCriterionConfig("remaining_fraction", 0.99),
                StoppingCriterionConfig("live_logz_error", 0.1),
                StoppingCriterionConfig("live_ess", 1.0),
                StoppingCriterionConfig("logz_stability", 0.1),
            ),
            stability_window=2,
        ),
        max_iterations=50,
        progress=True,
    )
    captured = capsys.readouterr()

    assert result.success
    assert "liveErr=" in captured.err
    assert "ESSlive=" in captured.err
    assert "stable=" in captured.err
    assert "dlogZrem=" not in captured.err
    assert " rem=" not in captured.err


def test_invalid_progress_option_is_rejected() -> None:
    model, proposal = _constant_problem()
    with pytest.raises(TypeError, match="progress"):
        NISMOSampler(model=model, importance_morph=proposal, n_live=8, rng=33).run(
            max_iterations=1,
            progress="yes",  # type: ignore[arg-type]
        )


def test_result_equal_weight_resampling_is_reproducible_and_configurable() -> None:
    model, proposal = _constant_problem()
    result = NISMOSampler(
        model=model,
        importance_morph=proposal,
        n_live=10,
        rng=34,
        tie_policy="randomized_plateau",
    ).run(
        dlogz=0.3,
        max_iterations=100,
    )
    default_first = result.resample_equal(rng=91)
    default_second = result.resample_equal(rng=91)
    custom = result.resample_equal(
        rng=np.random.default_rng(92),
        n_samples=2_000,
    )

    np.testing.assert_array_equal(default_first, default_second)
    assert default_first.shape == result.all_points.shape
    assert custom.shape == (2_000, result.all_points.shape[1])
    assert custom.flags.owndata
    weighted_mean = np.average(
        result.all_points,
        axis=0,
        weights=result.posterior_weights,
    )
    assert np.mean(custom, axis=0) == pytest.approx(weighted_mean, abs=0.1)


@pytest.mark.parametrize("n_samples", [0, -1, 1.5, True])
def test_result_equal_weight_resampling_validates_sample_count(
    n_samples: Any,
) -> None:
    model, proposal = _constant_problem()
    result = NISMOSampler(
        model=model,
        importance_morph=proposal,
        n_live=8,
        rng=35,
        tie_policy="randomized_plateau",
    ).run(dlogz=0.4, max_iterations=50)
    with pytest.raises(ValueError, match="n_samples"):
        result.resample_equal(rng=1, n_samples=n_samples)


def test_result_equal_weight_resampling_requires_explicit_rng() -> None:
    model, proposal = _constant_problem()
    result = NISMOSampler(
        model=model,
        importance_morph=proposal,
        n_live=8,
        rng=36,
        tie_policy="randomized_plateau",
    ).run(dlogz=0.4, max_iterations=50)
    with pytest.raises(TypeError, match="rng"):
        result.resample_equal(rng="seed")  # type: ignore[arg-type]
