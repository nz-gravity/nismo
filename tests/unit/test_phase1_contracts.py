from __future__ import annotations

import numpy as np
import pytest

import nismo
from nismo import CallableModel, ConfigurationError, InvalidModelOutput, NISMOConfig

pytestmark = pytest.mark.unit


def test_public_import_and_version() -> None:
    assert nismo.Model is not None
    assert nismo.Proposal is not None
    assert nismo.RefittableProposal is not None
    assert nismo.ProposalUpdateRecord is not None
    assert nismo.summarize is not None
    assert nismo.plot_run is not None


def test_callable_model_vectorized_and_scalar_agree() -> None:
    points = np.array([[-1.0], [0.5], [2.0]])
    vectorized = CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=lambda x: -(x[:, 0] ** 2),
        log_prior_fn=lambda x: np.zeros(len(x)),
    )
    scalar = CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=lambda x: -(x[0] ** 2),
        log_prior_fn=lambda x: 0.0,
        vectorized=False,
    )
    np.testing.assert_allclose(
        vectorized.log_likelihood(points), scalar.log_likelihood(points)
    )
    np.testing.assert_allclose(vectorized.log_prior(points), scalar.log_prior(points))


def test_scalar_likelihood_map_is_ordered_and_does_not_map_prior() -> None:
    points = np.array([[-1.0], [0.5], [2.0]])
    mapped_rows: list[np.ndarray] = []

    def mapper(function: object, rows: np.ndarray) -> list[float]:
        callable_function = function
        mapped_rows.extend(np.array(row, copy=True) for row in rows)
        return [callable_function(row) for row in rows]  # type: ignore[operator]

    model = CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=lambda x: -(x[0] ** 2),
        log_prior_fn=lambda x: -abs(x[0]),
        vectorized=False,
        scalar_likelihood_map=mapper,
    )

    np.testing.assert_allclose(model.log_likelihood(points), [-1.0, -0.25, -4.0])
    np.testing.assert_allclose(model.log_prior(points), [-1.0, -0.5, -2.0])
    np.testing.assert_array_equal(np.asarray(mapped_rows), points)


def test_model_rejects_invalid_shape_and_nonfinite_points() -> None:
    model = CallableModel(
        ndim=2,
        parameter_names=("x", "y"),
        log_likelihood_fn=lambda x: np.zeros(len(x)),
        log_prior_fn=lambda x: np.zeros(len(x)),
    )
    with pytest.raises(InvalidModelOutput, match="shape"):
        model.log_likelihood(np.zeros((3, 1)))
    with pytest.raises(InvalidModelOutput, match="NaN or infinity"):
        model.log_prior(np.array([[0.0, np.nan]]))


def test_model_rejects_wrong_output_shape() -> None:
    model = CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=lambda x: np.zeros((len(x), 1)),
        log_prior_fn=lambda x: np.zeros(len(x)),
    )
    with pytest.raises(InvalidModelOutput, match="must return shape"):
        model.log_likelihood(np.zeros((2, 1)))


def test_rng_reproducibility_is_explicit() -> None:
    first = np.random.default_rng(10).normal(size=8)
    second = np.random.default_rng(10).normal(size=8)
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_live": 1}, "n_live"),
        ({"n_live": 2, "dlogz": 0.0}, "dlogz"),
        ({"n_live": 2, "proposal_batch_size": 0}, "proposal_batch_size"),
        ({"n_live": 2, "proposal_update_interval": 0}, "proposal_update_interval"),
        ({"n_live": 2, "proposal_scheme": "slice"}, "proposal_scheme"),
        ({"n_live": 5, "max_likelihood_calls": 4}, "max_likelihood_calls"),
        ({"n_live": 2, "tie_policy": "jitter"}, "tie_policy"),
    ],
)
def test_config_validation(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        NISMOConfig(**kwargs)


def test_adaptive_scheme_requires_refittable_importance_morph() -> None:
    class FixedNormal:
        ndim = 1

        def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
            return rng.normal(size=(n, 1))

        def log_prob(self, theta: np.ndarray) -> np.ndarray:
            return -0.5 * theta[:, 0] ** 2 - 0.5 * np.log(2.0 * np.pi)

    importance = FixedNormal()
    model = CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=lambda x: np.zeros(len(x)),
        log_prior_fn=importance.log_prob,
    )
    with pytest.raises(TypeError, match="refit"):
        nismo.NISMOSampler(
            model=model,
            importance_morph=importance,
            proposal_scheme="adaptive_morph",
            n_live=4,
            rng=1,
        )
