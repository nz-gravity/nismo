from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from nismo import (
    ConfigurationError,
    NISMOConfig,
    NumericalInvariantError,
    StoppingCriterionConfig,
    StoppingPolicy,
)
from nismo.stopping import (
    StoppingMetrics,
    calculate_remaining_dlogz,
    calculate_stopping_metrics,
    evaluate_stopping_policy,
)

pytestmark = pytest.mark.unit


def _metrics(
    *,
    remaining_fraction: float = 0.1,
    remaining_dlogz: float = 0.1,
    live_ess: float = 8.0,
    live_logz_error: float = 0.01,
    logz_stability: float = 0.02,
    logzerr: float = 0.03,
) -> StoppingMetrics:
    return StoppingMetrics(
        remaining_fraction=remaining_fraction,
        remaining_dlogz=remaining_dlogz,
        live_ess=live_ess,
        live_mean_rse=0.1,
        live_logz_error=live_logz_error,
        logz_stability=logz_stability,
        logzerr=logzerr,
    )


def _calculate(
    live_log_psi: list[float],
    *,
    remaining_fraction: float = 0.25,
    history: list[float] | None = None,
    stability_window: int = 3,
) -> StoppingMetrics:
    logz_live = (
        -np.inf if np.all(np.isneginf(live_log_psi)) else np.log(remaining_fraction)
    )
    return calculate_stopping_metrics(
        live_log_psi=live_log_psi,
        logz_dead=(
            -np.inf
            if np.all(np.isneginf(live_log_psi))
            else float(np.log1p(-remaining_fraction))
        ),
        logz_live=float(logz_live),
        logz_total=0.0,
        logz_history=[0.0] if history is None else history,
        logzerr=0.2,
        stability_window=stability_window,
    )


@pytest.mark.parametrize("offset", [0.0, 1.0e200, -1.0e200])
def test_equal_finite_live_values_have_full_ess_and_zero_error(offset: float) -> None:
    metrics = _calculate([offset] * 5)
    assert metrics.live_ess == 5.0
    assert metrics.live_mean_rse == 0.0
    assert metrics.live_logz_error == 0.0


def test_one_dominant_live_value_has_unit_ess_and_scaled_error() -> None:
    first = _calculate([0.0, -1_000.0, -1_000.0, -1_000.0], remaining_fraction=0.4)
    second = _calculate(
        [0.0, -1_000.0, -1_000.0, -1_000.0],
        remaining_fraction=0.2,
    )
    assert first.live_ess == pytest.approx(1.0)
    assert first.live_mean_rse > 0.0
    assert first.live_logz_error == pytest.approx(
        first.remaining_fraction * first.live_mean_rse
    )
    assert second.live_logz_error == pytest.approx(first.live_logz_error / 2.0)


def test_all_zero_live_contributions_have_defined_zero_uncertainty() -> None:
    metrics = calculate_stopping_metrics(
        live_log_psi=[-np.inf] * 6,
        logz_dead=-np.inf,
        logz_live=-np.inf,
        logz_total=-np.inf,
        logz_history=[],
        logzerr=0.0,
        stability_window=3,
    )
    assert metrics.remaining_fraction == 0.0
    assert np.isposinf(metrics.remaining_dlogz)
    assert metrics.live_ess == 6.0
    assert metrics.live_mean_rse == 0.0
    assert metrics.live_logz_error == 0.0
    assert not np.isnan(metrics.live_ess)


def test_ess_is_stable_under_large_common_log_offsets() -> None:
    reference = _calculate([0.0, -2.0, -4.0, -6.0])
    positive = _calculate([1.0e200, 1.0e200 - 2.0, 1.0e200 - 4.0, 1.0e200 - 6.0])
    negative = _calculate([-1.0e200, -1.0e200 - 2.0, -1.0e200 - 4.0, -1.0e200 - 6.0])
    assert np.isfinite(positive.live_ess)
    assert np.isfinite(negative.live_ess)
    assert 1.0 <= reference.live_ess <= 4.0
    assert 1.0 <= positive.live_ess <= 4.0
    assert 1.0 <= negative.live_ess <= 4.0


def test_remaining_fraction_matches_log_evidence_ratio() -> None:
    logz_live = -4.0
    logz_total = -3.25
    metrics = calculate_stopping_metrics(
        live_log_psi=[-1.0, -2.0, -3.0],
        logz_dead=logz_total + np.log1p(-np.exp(logz_live - logz_total)),
        logz_live=logz_live,
        logz_total=logz_total,
        logz_history=[logz_total],
        logzerr=0.1,
        stability_window=2,
    )
    assert metrics.remaining_fraction == pytest.approx(np.exp(logz_live - logz_total))
    assert metrics.remaining_dlogz == pytest.approx(
        -np.log1p(-metrics.remaining_fraction)
    )


def test_remaining_dlogz_matches_log_evidence_increment() -> None:
    value = calculate_remaining_dlogz(
        logz_dead=np.log(9.0),
        logz_live=np.log(1.0),
    )

    expected = np.log(10.0) - np.log(9.0)
    assert np.isclose(value, expected)


@pytest.mark.parametrize("remaining_fraction", [1e-8, 1e-3, 0.01, 0.1, 0.5, 0.9])
def test_remaining_dlogz_matches_remaining_fraction_identity(
    remaining_fraction: float,
) -> None:
    value = calculate_remaining_dlogz(
        logz_dead=np.log(1.0 - remaining_fraction),
        logz_live=np.log(remaining_fraction),
    )

    assert value == pytest.approx(-np.log1p(-remaining_fraction))


def test_remaining_dlogz_is_zero_for_zero_live_evidence() -> None:
    value = calculate_remaining_dlogz(logz_dead=0.0, logz_live=-np.inf)

    assert value == 0.0


@pytest.mark.parametrize("logz_live", [0.0, -np.inf])
def test_remaining_dlogz_is_infinite_without_dead_evidence(
    logz_live: float,
) -> None:
    value = calculate_remaining_dlogz(logz_dead=-np.inf, logz_live=logz_live)

    assert np.isposinf(value)


@pytest.mark.parametrize(
    ("logz_dead", "logz_live"),
    [
        (np.nan, 0.0),
        (0.0, np.nan),
        (np.inf, 0.0),
        (0.0, np.inf),
    ],
)
def test_remaining_dlogz_rejects_invalid_values(
    logz_dead: float,
    logz_live: float,
) -> None:
    with pytest.raises(NumericalInvariantError):
        calculate_remaining_dlogz(logz_dead=logz_dead, logz_live=logz_live)


def test_stability_requires_the_exact_full_window() -> None:
    unavailable = _calculate(
        [0.0, 0.0],
        history=[1.0, 1.1],
        stability_window=3,
    )
    available = _calculate(
        [0.0, 0.0],
        history=[100.0, 1.0, 1.4, 1.2],
        stability_window=3,
    )
    assert np.isnan(unavailable.logz_stability)
    assert available.logz_stability == pytest.approx(0.4)


@pytest.mark.parametrize(
    ("name", "tolerance", "metrics"),
    [
        ("remaining_fraction", 0.1, _metrics(remaining_fraction=0.1)),
        ("remaining_dlogz", 0.1, _metrics(remaining_dlogz=0.1)),
        ("live_logz_error", 0.1, _metrics(live_logz_error=0.1)),
        ("logz_stability", 0.1, _metrics(logz_stability=0.1)),
        ("logzerr", 0.1, _metrics(logzerr=0.1)),
        ("live_ess", 1.0, _metrics(live_ess=1.0)),
    ],
)
def test_each_criterion_passes_at_equality(
    name: str,
    tolerance: float,
    metrics: StoppingMetrics,
) -> None:
    decision = evaluate_stopping_policy(
        metrics=metrics,
        policy=StoppingPolicy(
            criteria=(
                StoppingCriterionConfig(name=name, tolerance=tolerance),  # type: ignore[arg-type]
            )
        ),
        niter=1,
        previous_streak=0,
    )
    assert decision.evaluations[0].met
    assert decision.should_stop


def test_live_ess_uses_greater_than_direction() -> None:
    policy = StoppingPolicy(criteria=(StoppingCriterionConfig("live_ess", 5.0),))
    below = evaluate_stopping_policy(
        metrics=_metrics(live_ess=4.9),
        policy=policy,
        niter=1,
        previous_streak=0,
    )
    above = evaluate_stopping_policy(
        metrics=_metrics(live_ess=5.1),
        policy=policy,
        niter=1,
        previous_streak=0,
    )
    assert not below.combined_met
    assert above.combined_met


def test_all_and_any_modes_combine_criterion_results() -> None:
    criteria = (
        StoppingCriterionConfig("remaining_fraction", 0.2),
        StoppingCriterionConfig("live_logz_error", 0.001),
    )
    metrics = _metrics(remaining_fraction=0.1, live_logz_error=0.01)
    all_decision = evaluate_stopping_policy(
        metrics=metrics,
        policy=StoppingPolicy(criteria=criteria, mode="all"),
        niter=1,
        previous_streak=0,
    )
    any_decision = evaluate_stopping_policy(
        metrics=metrics,
        policy=StoppingPolicy(criteria=criteria, mode="any"),
        niter=1,
        previous_streak=0,
    )
    assert not all_decision.combined_met
    assert any_decision.combined_met


def test_minimum_iterations_and_consecutive_streak_are_applied_exactly() -> None:
    policy = StoppingPolicy(
        criteria=(StoppingCriterionConfig("remaining_fraction", 0.2),),
        consecutive=2,
        min_iterations=3,
    )
    metrics = _metrics(remaining_fraction=0.1)
    too_early = evaluate_stopping_policy(
        metrics=metrics,
        policy=policy,
        niter=2,
        previous_streak=4,
    )
    first = evaluate_stopping_policy(
        metrics=metrics,
        policy=policy,
        niter=3,
        previous_streak=too_early.streak,
    )
    second = evaluate_stopping_policy(
        metrics=metrics,
        policy=policy,
        niter=4,
        previous_streak=first.streak,
    )
    assert too_early.streak == 0
    assert first.streak == 1
    assert not first.should_stop
    assert second.streak == 2
    assert second.should_stop


def test_streak_resets_immediately_after_combined_failure() -> None:
    policy = StoppingPolicy(
        criteria=(StoppingCriterionConfig("remaining_fraction", 0.2),),
        consecutive=3,
    )
    decision = evaluate_stopping_policy(
        metrics=_metrics(remaining_fraction=0.3),
        policy=policy,
        niter=5,
        previous_streak=2,
    )
    assert not decision.combined_met
    assert decision.streak == 0
    assert not decision.should_stop


def test_unavailable_stability_is_unmet() -> None:
    decision = evaluate_stopping_policy(
        metrics=_metrics(logz_stability=np.nan),
        policy=StoppingPolicy(
            criteria=(StoppingCriterionConfig("logz_stability", 0.1),)
        ),
        niter=100,
        previous_streak=0,
    )
    assert not decision.evaluations[0].met


def test_infinite_remaining_dlogz_is_unmet() -> None:
    decision = evaluate_stopping_policy(
        metrics=_metrics(remaining_dlogz=np.inf),
        policy=StoppingPolicy(
            criteria=(StoppingCriterionConfig("remaining_dlogz", 0.1),)
        ),
        niter=1,
        previous_streak=0,
    )

    assert not decision.evaluations[0].met
    assert not decision.should_stop


@pytest.mark.parametrize("value", [np.nan, -np.inf])
def test_invalid_nonpositive_infinite_remaining_dlogz_metric_raises(
    value: float,
) -> None:
    with pytest.raises(NumericalInvariantError, match="must be finite"):
        evaluate_stopping_policy(
            metrics=_metrics(remaining_dlogz=value),
            policy=StoppingPolicy(
                criteria=(StoppingCriterionConfig("remaining_dlogz", 0.1),)
            ),
            niter=1,
            previous_streak=0,
        )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: StoppingPolicy(criteria=()), "at least one"),
        (
            lambda: StoppingPolicy(
                criteria=(
                    StoppingCriterionConfig("logzerr", 0.1),
                    StoppingCriterionConfig("logzerr", 0.2),
                )
            ),
            "unique",
        ),
        (
            lambda: StoppingCriterionConfig("unknown", 0.1),  # type: ignore[arg-type]
            "unsupported",
        ),
        (
            lambda: StoppingPolicy(
                criteria=(StoppingCriterionConfig("logzerr", 0.1),),
                mode="neither",  # type: ignore[arg-type]
            ),
            "mode",
        ),
        (
            lambda: StoppingPolicy(
                criteria=(StoppingCriterionConfig("logzerr", 0.1),),
                consecutive=0,
            ),
            "consecutive",
        ),
        (
            lambda: StoppingPolicy(
                criteria=(StoppingCriterionConfig("logzerr", 0.1),),
                consecutive=True,
            ),
            "consecutive",
        ),
        (
            lambda: StoppingPolicy(
                criteria=(StoppingCriterionConfig("logzerr", 0.1),),
                min_iterations=-1,
            ),
            "min_iterations",
        ),
        (
            lambda: StoppingPolicy(
                criteria=(StoppingCriterionConfig("logzerr", 0.1),),
                stability_window=1,
            ),
            "stability_window",
        ),
    ],
)
def test_policy_validation_rejects_invalid_configuration(
    factory: Any,
    message: str,
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        factory()


@pytest.mark.parametrize(
    ("name", "tolerance"),
    [
        ("remaining_fraction", 0.0),
        ("remaining_fraction", 1.0),
        ("remaining_fraction", np.nan),
        ("remaining_dlogz", 0.0),
        ("remaining_dlogz", -1.0),
        ("remaining_dlogz", np.nan),
        ("remaining_dlogz", np.inf),
        ("remaining_dlogz", True),
        ("live_logz_error", 0.0),
        ("live_logz_error", np.inf),
        ("logz_stability", np.nan),
        ("live_ess", 0.9),
        ("logzerr", -1.0),
        ("logzerr", np.inf),
        ("logzerr", True),
    ],
)
def test_criterion_validation_rejects_invalid_tolerances(
    name: str,
    tolerance: Any,
) -> None:
    with pytest.raises(ConfigurationError, match="tolerance"):
        StoppingCriterionConfig(name=name, tolerance=tolerance)  # type: ignore[arg-type]


def test_config_rejects_live_ess_target_above_n_live() -> None:
    policy = StoppingPolicy(criteria=(StoppingCriterionConfig("live_ess", 11.0),))
    with pytest.raises(ConfigurationError, match=r"live_ess.*n_live"):
        NISMOConfig(n_live=10, stopping=policy)


def test_config_resolves_legacy_and_rejects_ambiguous_stopping() -> None:
    default = NISMOConfig(n_live=10)
    explicit = NISMOConfig(n_live=10, dlogz=2.0)
    policy = StoppingPolicy(criteria=(StoppingCriterionConfig("logzerr", 0.1),))
    assert default.dlogz == 1.0e-3
    assert default.stopping is not None
    assert default.stopping.criteria[0].name == "remaining_dlogz"
    assert default.stopping.criteria[0].tolerance == 1.0e-3
    assert explicit.stopping is not None
    assert explicit.stopping.criteria[0].name == "remaining_dlogz"
    assert explicit.stopping.criteria[0].tolerance == 2.0
    remaining_fraction = StoppingPolicy(
        criteria=(StoppingCriterionConfig("remaining_fraction", 0.02),)
    )
    assert NISMOConfig(n_live=10, stopping=remaining_fraction).stopping == (
        remaining_fraction
    )
    with pytest.raises(ConfigurationError, match="dlogz and stopping"):
        NISMOConfig(n_live=10, dlogz=0.1, stopping=policy)
    with pytest.raises(ConfigurationError, match="dlogz"):
        NISMOConfig(n_live=10, dlogz=True)


@pytest.mark.parametrize("dlogz", [0.0, -0.1, np.nan, np.inf, True])
def test_config_rejects_invalid_dlogz(dlogz: Any) -> None:
    with pytest.raises(ConfigurationError, match="positive finite"):
        NISMOConfig(n_live=10, dlogz=dlogz)


def test_inconsistent_nonfinite_metric_state_raises_typed_error() -> None:
    with pytest.raises(NumericalInvariantError, match="all-zero"):
        calculate_stopping_metrics(
            live_log_psi=[-np.inf, -np.inf],
            logz_dead=0.0,
            logz_live=0.0,
            logz_total=0.0,
            logz_history=[0.0],
            logzerr=0.1,
            stability_window=2,
        )
