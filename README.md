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
- four replacement schemes: `fixed_morph`, `mor-rwalk`, `s-rwalk`, and
  `en-rwalk`;
- multiprocessing with FIFO replacement prefetching;
- configurable scientific stopping criteria and hard resource limits;
- reproducible run histories, diagnostics, and plotting helpers.

## Nested importance sampling

For likelihood $L(\theta)$ and normalized prior $\pi(\theta)$, the marginal
likelihood (Bayesian evidence) is

$$
Z = p(y) = \int_{\Theta} L(\theta)\,\pi(\theta)\,d\theta.
$$

NIS introduces a normalized importance density $q_0(\theta)$. NISMO can use
the power-tempered (diffused) density

$$
\widetilde g_\beta(\theta)=q_0(\theta)^\beta,
\qquad
C_\beta=\int_\Theta q_0(\theta)^\beta\,d\theta,
\qquad
g_\beta(\theta)=\frac{q_0(\theta)^\beta}{C_\beta},
$$

with $0<\beta\leq1$. The corresponding transformed integrand is

$$
\Psi_\beta(\theta)
= \frac{C_\beta L(\theta)\,\pi(\theta)}{q_0(\theta)^\beta}.
$$

The evidence can then be written as an expectation under $g_\beta$, or as the
usual one-dimensional nested-sampling integral:

$$
Z
= \int_{\Theta} \Psi_\beta(\theta)\,g_\beta(\theta)\,d\theta
= \int_0^1 \Psi_\beta(X)\,dX,
$$

where

$$
X(\lambda)
= \int_{\Psi_\beta(\theta)>\lambda}g_\beta(\theta)\,d\theta
$$

is the remaining probability mass under the importance density.

`beta=1` is the standard sampler: $C_1=1$, $g_1=q_0$, and no Monte Carlo
normalization is performed. For `0 < beta < 1`, NISMO estimates the
normalizer directly from draws $\theta_j\sim q_0$ using

$$
C_\beta
= \mathbb E_{q_0}\!\left[q_0(\theta)^{\beta-1}\right]
\approx \frac{1}{N}\sum_{j=1}^N q_0(\theta_j)^{\beta-1}.
$$

The same finite candidate batch is importance-resampled to initialize the
diffused pool. This mode is available for `mor-rwalk` and `s-rwalk`; their MH
fallback targets constrained $g_\beta$ using the log-density difference
$\beta[\log q_0(\theta')-\log q_0(\theta)]$. The endpoint `beta=0` is not
supported because its normalizer is generally infinite on unbounded parameter
spaces.

### How NISMO implements NIS

1. A normalized Morph density $q_0$ is fitted to representative posterior
   samples and then fixed for the evidence calculation.
2. The initial $N_{\rm live}$ points are drawn from $g_\beta$ (directly from
   $q_0$ when `beta=1`, or from the finite tempered pool otherwise).
3. NISMO evaluates
   `log_psi_beta = log_likelihood + log_prior - beta * log_q0 + log_C_beta`
   and removes the live point with the smallest transformed integrand.
4. Deterministic nested-volume shrinkage is used:
   $X_i=\exp(-i/N_{\rm live})$. Each dead point contributes
   $(X_{i-1}-X_i)\Psi_i$ to the evidence quadrature.
5. A replacement is drawn from the configured importance density, subject to
   the current $\Psi_\beta$-constraint.
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
    output_path="runs/example",
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

When `output_path` is set, every completed or valid partial run automatically
writes `weighted_samples.npz`, `run_history.npz`, `diagnostics.json`, and three
diagnostic figures. Install `nismo[plot]` to enable the PNG output; samples and
diagnostics are still saved when Matplotlib is unavailable. The same bundle can
be written later with `result.save("runs/example")`.

Model functions receive batches with shape `(n, ndim)` and return one natural
logarithm per row. The prior must be normalized and include every normalization
constant. The samples used to fit the Morph are not reused as live points.

## Replacement schemes

Select a scheme through `NISMOSampler(..., proposal_scheme=...)`:

| Scheme | Replacement mechanism |
|---|---|
| `fixed_morph` | Independent constrained rejection draws from the fixed Morph |
| `mor-rwalk` | One pre-evaluated Morph or power-tempered pool, followed by `s-rwalk` when the pool can no longer meet the constraint |
| `s-rwalk` | Gaussian-covariance random walk targeting constrained $g_\beta$ |
| `en-rwalk` | Split-ensemble differential-evolution, stretch, and Gaussian move mixture |

Configure the hybrid scheme with a total initial pool size. NISMO randomly
chooses the live set from this batch and retains the other points in randomized
proposal order:

```python
from nismo import MORWalkSettings, SRWalkSettings

sampler = NISMOSampler(
    ...,
    proposal_scheme="mor-rwalk",
    beta=0.7,
    beta_mc_samples=100_000,
    mor_rwalk_settings=MORWalkSettings(n_proposals=20_000),
    srwalk_settings=SRWalkSettings(n_steps=75),
)
```

The complete batch is evaluated once. The retained stream is consumed until no
remaining proposal passes the current `log_psi0` threshold, after which the
sampler switches permanently to `s-rwalk`.

Finite-length MCMC replacements must be calibrated for the dimension and
geometry of the target. See the configuration guide for move settings,
parallel workers, queue size, stopping policies, and resource limits.

Configure replacement parallelism directly in the sampler constructor, for
example with `NISMOSampler(..., n_workers=8, queue_size=8)`.

## Results and diagnostics

`NISMOSampler.run(...)` returns an immutable `NISMOResult` containing:

- `logz`, `logzerr`, `information`, `success`, and `termination_reason`;
- `beta_diagnostics`, including the estimated `log_z_beta`, Monte Carlo error,
  and effective sample size of the normalizing-constant estimate;
- weighted dead and final-live points;
- likelihood, prior, importance-density, transformed-integrand, and quadrature
  arrays;
- per-iteration stopping, acceptance, cost, MCMC, and queue histories;
- optional `s-rwalk` component timings through
  `SRWalkSettings(profile=True)` and `result.srwalk_diagnostics`;
- reproducibility metadata and the initial and final random-number-generator
  states.

For `beta=1`, `logzerr = sqrt(H / n_live)`. For `beta<1`, the reported value
combines that term and the direct-Monte-Carlo error estimate for `log_z_beta`
in quadrature. It remains an incomplete error budget: it does not include
missing importance support, finite-pool approximation, imperfect finite-length
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

If you also want the LVK analysis workflow under
`analysis/LIGO/fast_pp`, install the matching `lvk` extra:

```bash
uv sync --extra dev --extra lvk
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
