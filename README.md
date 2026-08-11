# NISMO

NISMO is a Python implementation of Morphing Importance Nested Sampling for
Bayesian evidence estimation and weighted posterior inference. It uses a fixed,
normalized importance distribution to define the nested-sampling target and
supports rejection, random-walk, ensemble, and parallel replacement strategies.

> NISMO is alpha research software. Validate evidence accuracy and sampling
> efficiency with repeated runs on your target problem. The fixed importance
> distribution must cover every region where the likelihood times prior is
> nonzero; missing support can bias the result. The adaptive-Morph proposal is
> explicitly heuristic and can also bias evidence estimates.

## Installation

Install the standard user stack from PyPI:

```bash
python -m pip install "nismo[all]"
```

The base package requires only NumPy and SciPy. Optional extras are available
individually:

- `nismo[morph]` adds MorphZ proposal fitting;
- `nismo[plot]` adds Matplotlib diagnostics;
- `nismo[progress]` adds the tqdm terminal/notebook display.

Until the first PyPI release is published, install the repository version with:

```bash
python -m pip install "nismo[all] @ git+https://github.com/nz-gravity/nismo.git@main"
```

Python 3.10 through 3.12 are tested.

## Quick start

```python
import numpy as np

from nismo import CallableModel, MorphProposal, NISMOSampler

LOG_2PI = np.log(2.0 * np.pi)


def log_likelihood(theta: np.ndarray) -> np.ndarray:
    return -0.5 * (theta[:, 0] ** 2 + LOG_2PI)


def log_prior(theta: np.ndarray) -> np.ndarray:
    return -0.5 * ((theta[:, 0] / 2.0) ** 2 + LOG_2PI) - np.log(2.0)


model = CallableModel(
    ndim=1,
    parameter_names=("x",),
    log_likelihood_fn=log_likelihood,
    log_prior_fn=log_prior,
)

# Representative posterior draws used only to fit the importance density.
training_rng = np.random.default_rng(7)
posterior_samples = training_rng.normal(scale=np.sqrt(0.8), size=(2_000, 1))
importance_morph = MorphProposal.fit(
    posterior_samples,
    param_names=model.parameter_names,
    groups=[],
)

result = NISMOSampler(
    model=model,
    importance_morph=importance_morph,
    n_live=200,
    rng=42,
).run(dlogz=1e-2, progress=True)

print(result.logz, result.logzerr, result.termination_reason)
weighted_samples = result.all_points
weights = result.posterior_weights
equal_weight_samples = result.resample_equal(rng=43, n_samples=10_000)
```

Both model functions receive batches with shape `(n, ndim)` and return one
natural-log value per row. `log_prior` must be normalized and include every
normalization constant. Posterior training samples fit the importance density;
they are not reused as live points.

## Documentation

- [Documentation index](https://github.com/nz-gravity/nismo/blob/main/docs/index.md)
- [Quick-start guide](https://github.com/nz-gravity/nismo/blob/main/docs/quickstart.md)
- [Configuration reference](https://github.com/nz-gravity/nismo/blob/main/docs/configuration.md)
- [Results and diagnostics](https://github.com/nz-gravity/nismo/blob/main/docs/results.md)
- [Complete public API](https://github.com/nz-gravity/nismo/blob/main/docs/api.md)

Runnable scripts and notebooks are under the
[`examples/` directory](https://github.com/nz-gravity/nismo/tree/main/examples).

## Development

```bash
git clone https://github.com/nz-gravity/nismo.git
cd nismo
uv sync --extra dev
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest -m "not slow"
uv build
uvx twine check dist/*
```

See the [contribution guide](https://github.com/nz-gravity/nismo/blob/main/CONTRIBUTING.md)
and [release guide](https://github.com/nz-gravity/nismo/blob/main/RELEASING.md).

## License

NISMO is distributed under the
[BSD 3-Clause license](https://github.com/nz-gravity/nismo/blob/main/LICENSE).
