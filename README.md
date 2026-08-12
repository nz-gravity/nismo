# NISMO: Nested Importance Sampling Morph Optimisation

[![PyPI version](https://img.shields.io/pypi/v/nismo.svg)](https://pypi.org/project/nismo/)
[![Python versions](https://img.shields.io/pypi/pyversions/nismo.svg)](https://pypi.org/project/nismo/)
[![License](https://img.shields.io/pypi/l/nismo.svg)](https://pypi.org/project/nismo/)


NISMO is a Python sampler for Bayesian evidence estimation
and weighted posterior inference. It combines **nested importance sampling
(NIS)** with a normalized **Morph** approximation, concentrating the nested
sampling calculation in the regions that matter most to the posterior.

NISMO provides:

- log-evidence estimates and weighted posterior samples;
- three replacement schemes: `fixed_morph`, `s-rwalk`, and `en-rwalk`;
- multiprocessing with FIFO replacement prefetching;
- configurable scientific stopping criteria and hard resource limits;
- reproducible run histories, diagnostics, and plotting helpers.

## Nested importance sampling

For likelihood $L(\theta)$ and normalized prior $\pi(\theta)$, the marginal
likelihood (Bayesian evidence) is

$$
Z = p(y) = \int_{\Theta} L(\theta)\,\pi(\theta)\,d\theta.
$$

NIS introduces a normalized importance density $q_0(\theta)$ and the
transformed integrand

$$
\Psi(\theta)
= \frac{L(\theta)\,\pi(\theta)}{q_0(\theta)}.
$$

The evidence can then be written as an expectation under $q_0$, or as the
usual one-dimensional nested-sampling integral:

$$
Z
= \int_{\Theta} \Psi(\theta)\,q_0(\theta)\,d\theta
= \int_0^1 \Psi(X)\,dX,
$$

where

$$
X(\lambda)
= \int_{\Psi(\theta)>\lambda}q_0(\theta)\,d\theta
$$

is the remaining probability mass under the importance density.

### How NISMO implements NIS

1. A normalized Morph density $q_0$ is fitted to representative posterior
   samples and then fixed for the evidence calculation.
2. The initial $N_{\rm live}$ points are drawn independently from $q_0$.
3. NISMO evaluates
   `log_psi0 = log_likelihood + log_prior - log_q0` and removes the live point
   with the smallest transformed integrand.
4. Deterministic nested-volume shrinkage is used:
   $X_i=\exp(-i/N_{\rm live})$. Each dead point contributes
   $(X_{i-1}-X_i)\Psi_i$ to the evidence quadrature.
5. A replacement is drawn from $q_0$, subject to the current
   $\Psi$-constraint. The `fixed_morph`, `s-rwalk`, and `en-rwalk`
   schemes are available.
6. At termination, the remaining live-point contribution is added and all
   contributions are normalized to produce posterior weights.

The fixed importance density must have support everywhere that
$L(\theta)\pi(\theta)$ is nonzero. A missing mode cannot be recovered by the
quadrature and can bias both the evidence and posterior. As with any evidence
sampler, validate the complete configuration with repeated seeds and suitable
benchmark problems.

## Installation

Install NISMO 0.1.2 and all user-facing optional features from PyPI:

```bash
python -m pip install "nismo[all]==0.1.2"
```

The core package requires only NumPy and SciPy. Optional extras can be installed
individually:

| Extra | Provides |
|---|---|
| `nismo[morph]` | MorphZ proposal fitting |
| `nismo[plot]` | Matplotlib diagnostics |
| `nismo[progress]` | tqdm terminal and notebook progress displays |
| `nismo[all]` | All user-facing optional features |

NISMO supports Python 3.10 or newer and is tested on Python 3.10–3.12.

## API example

The example below defines a normalized one-dimensional model, fits the fixed
Morph importance density, runs NISMO, and extracts posterior samples.

```python
import numpy as np

from nismo import CallableModel, MorphProposal, NISMOSampler

LOG_2PI = np.log(2.0 * np.pi)


def log_likelihood(theta: np.ndarray) -> np.ndarray:
    """N(x=0 | theta, sigma=1), evaluated in batches."""
    return -0.5 * (theta[:, 0] ** 2 + LOG_2PI)


def log_prior(theta: np.ndarray) -> np.ndarray:
    """Normalized N(theta | 0, 2) prior."""
    return -0.5 * ((theta[:, 0] / 2.0) ** 2 + LOG_2PI) - np.log(2.0)


model = CallableModel(
    ndim=1,
    parameter_names=("x",),
    log_likelihood_fn=log_likelihood,
    log_prior_fn=log_prior,
)

# Representative posterior draws used only to fit q0.
training_rng = np.random.default_rng(7)
posterior_samples = training_rng.normal(
    scale=np.sqrt(0.8),
    size=(2_000, 1),
)

importance_morph = MorphProposal.fit(
    posterior_samples,
    param_names=model.parameter_names,
    groups=[],
)

sampler = NISMOSampler(
    model=model,
    importance_morph=importance_morph,
    proposal_scheme="fixed_morph",
    n_live=200,
    rng=42,
    n_workers=1,
)

result = sampler.run(dlogz=1e-2, progress=True)

print(f"logZ = {result.logz:.6f} +/- {result.logzerr:.6f}")
print(result.success, result.termination_reason)

# The weighted representation is the primary posterior output.
points = result.all_points
weights = result.posterior_weights
posterior_mean = np.average(points, axis=0, weights=weights)

# Equal-weight samples are available when required by downstream software.
equal_weight_samples = result.resample_equal(rng=43, n_samples=10_000)
```

Model functions receive batches with shape `(n, ndim)` and return one natural
logarithm per row. The prior must be normalized and include every normalization
constant. The samples used to fit the Morph are not reused as live points.

## Replacement schemes

Select a scheme through `NISMOSampler(..., proposal_scheme=...)`:

| Scheme | Replacement mechanism |
|---|---|
| `fixed_morph` | Independent constrained rejection draws from the fixed Morph |
| `s-rwalk` | Gaussian-covariance random walk targeting constrained $q_0$ |
| `en-rwalk` | Split-ensemble differential-evolution, stretch, and Gaussian move mixture |

Finite-length MCMC replacements must be calibrated for the dimension and
geometry of the target. See the configuration guide for move settings,
parallel workers, queue size, stopping policies, and resource limits.

Configure replacement parallelism directly in the sampler constructor, for
example with `NISMOSampler(..., n_workers=8, queue_size=8)`.

## Results and diagnostics

`NISMOSampler.run(...)` returns an immutable `NISMOResult` containing:

- `logz`, `logzerr`, `information`, `success`, and `termination_reason`;
- weighted dead and final-live points;
- likelihood, prior, importance-density, transformed-integrand, and quadrature
  arrays;
- per-iteration stopping, acceptance, cost, MCMC, and queue histories;
- reproducibility metadata and the initial and final random-number-generator
  states.

The theoretical `logzerr = sqrt(H / n_live)` estimate is not a complete error
budget: it does not include missing importance support, imperfect finite-length
MCMC mixing, or adaptive-proposal approximation error.

## Documentation

- [Quick-start guide](https://github.com/nz-gravity/nismo/blob/main/docs/quickstart.md)
- [Configuration reference](https://github.com/nz-gravity/nismo/blob/main/docs/configuration.md)
- [Results and diagnostics](https://github.com/nz-gravity/nismo/blob/main/docs/results.md)
- [Complete public API](https://github.com/nz-gravity/nismo/blob/main/docs/api.md)
- [Runnable examples](https://github.com/nz-gravity/nismo/tree/main/examples)

## Development

Clone the repository and create the development environment with
[uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/nz-gravity/nismo.git
cd nismo
uv sync --extra dev
```

Run the development checks and build the package:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest -m "not slow"
uv build
uvx twine check dist/*
```

See [CONTRIBUTING.md](https://github.com/nz-gravity/nismo/blob/main/CONTRIBUTING.md)
for the contribution workflow and
[RELEASING.md](https://github.com/nz-gravity/nismo/blob/main/RELEASING.md) for the
release process.

## Citation

If you use NISMO in research, cite the software using the repository's
[`CITATION.cff`](https://github.com/nz-gravity/nismo/blob/main/CITATION.cff) or
GitHub's **Cite this repository** menu. For version 0.1.2, a BibTeX entry is:

```bibtex
@software{nismo_2026,
  author  = {{NISMO contributors}},
  title   = {NISMO: Morphing Importance Nested Sampling},
  year    = {2026},
  version = {0.1.2},
  url     = {https://github.com/nz-gravity/nismo}
}
```

Configured replacement methods may require additional citations. Inspect them
programmatically with `sampler.citations`.

## License

NISMO is distributed under the
[BSD 3-Clause license](https://github.com/nz-gravity/nismo/blob/main/LICENSE).
