"""Small dependency-free checks for the LIGO NISMO runner helpers."""

from __future__ import annotations

import importlib.util
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
