"""Small dependency-free checks for the LIGO NISMO runner helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "analysis"
    / "LIGO"
    / "fast_pp"
    / "nismo_computation.py"
)
SPEC = importlib.util.spec_from_file_location("ligo_nismo_computation", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


class _Column:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def to_numpy(self, dtype: type[float] = float) -> np.ndarray:
        return np.asarray(self._values, dtype=dtype)


class _Posterior:
    def __init__(self, columns: dict[str, list[float]]) -> None:
        self._columns = columns

    def __contains__(self, name: str) -> bool:
        return name in self._columns

    def __getitem__(self, names: str | list[str]) -> _Column | _Posterior:
        if isinstance(names, str):
            return _Column(self._columns[names])
        return _Posterior({name: self._columns[name] for name in names})

    def to_numpy(self, dtype: type[float] = float) -> np.ndarray:
        return np.asarray(list(self._columns.values()), dtype=dtype).T


def test_posterior_parameter_names_and_training_samples_use_all_rows() -> None:
    result = SimpleNamespace(
        search_parameter_keys=["x", "y"],
        posterior=_Posterior({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]}),
    )

    names = RUNNER.posterior_parameter_names(result)

    assert names == ("x", "y")
    np.testing.assert_allclose(
        RUNNER.training_samples(result, names),
        np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]),
    )


def test_posterior_parameter_names_rejects_missing_coordinate() -> None:
    result = SimpleNamespace(
        search_parameter_keys=["x", "missing"],
        posterior=_Posterior({"x": [1.0, 2.0]}),
    )

    with pytest.raises(ValueError, match="missing"):
        RUNNER.posterior_parameter_names(result)


def test_default_max_iterations_scales_with_live_count() -> None:
    assert RUNNER.default_max_iterations(100) == 10_000
    assert RUNNER.default_max_iterations(500) == 12_500
    assert RUNNER.default_max_iterations(2_000) == 50_000


def test_ligo_runner_is_fixed_to_srwalk() -> None:
    assert RUNNER.NISMO_PROPOSAL_SCHEME == "s-rwalk"


def test_ligo_runner_cli_selects_replica_seed() -> None:
    defaults = RUNNER.parse_args(["48"])
    replica = RUNNER.parse_args(["48", "--nismo-seed", "47"])

    assert defaults.lvk_seed == 48
    assert defaults.nismo_seed == RUNNER.NISMO_DEFAULT_SEED
    assert replica.nismo_seed == 47


def test_dynesty_result_path_keeps_low_live_runs_isolated(tmp_path: Path) -> None:
    assert RUNNER.dynesty_result_path(tmp_path, 48, 2_000) == (
        tmp_path / "seed_48" / "dynesty_result.json"
    )
    assert RUNNER.dynesty_result_path(tmp_path, 48, 100) == (
        tmp_path / "seed_48" / "dynesty_nlive100" / "dynesty_nlive100_result.json"
    )


class _NaNPrior:
    def ln_prob(self, parameters: dict[str, float]) -> float:
        return np.nan if parameters["x"] < 0.0 else -0.5 * parameters["x"] ** 2


class _CountingLikelihood:
    def __init__(self) -> None:
        self.calls = 0

    def log_likelihood(self, parameters: dict[str, float]) -> float:
        self.calls += 1
        return -(parameters["x"] ** 2)


def test_bilby_model_converts_nonfinite_prior_to_zero_support() -> None:
    likelihood = _CountingLikelihood()
    model = RUNNER.BilbyLIGOModel(
        parameter_names=("x",),
        likelihood=likelihood,
        sampled_priors=_NaNPrior(),
        fixed_values={},
    )
    theta = np.array([[-1.0], [2.0]])

    np.testing.assert_allclose(model.log_prior(theta), [-np.inf, -2.0])
    np.testing.assert_allclose(model.log_likelihood(theta), [-np.inf, -4.0])
    assert likelihood.calls == 1
