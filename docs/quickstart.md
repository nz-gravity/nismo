# Quick start

## 1. Install the user stack

```bash
python -m pip install "nismo[all]"
```

## 2. Define a normalized model

`CallableModel` accepts vectorized functions by default. Each receives an array
with shape `(n, ndim)` and returns an array with shape `(n,)`.

```python
import numpy as np

from nismo import CallableModel

LOG_2PI = np.log(2.0 * np.pi)


def log_likelihood(theta: np.ndarray) -> np.ndarray:
    """N(x=0 | theta, sigma=1), viewed as a function of theta."""
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
```

Use `-np.inf` for zero density. Model output containing `NaN` or positive
infinity is rejected. The prior must contain all normalization constants,
because an omitted constant directly shifts the evidence.

## 3. Fit the fixed importance Morph

Supply representative posterior draws with shape `(n_samples, ndim)`:

```python
from nismo import MorphProposal

training_rng = np.random.default_rng(7)
posterior_samples = training_rng.normal(scale=np.sqrt(0.8), size=(2_000, 1))

importance_morph = MorphProposal.fit(
    posterior_samples,
    param_names=model.parameter_names,
    groups=[],
    kde_bw="silverman",
)
```

`groups=[]` fits independent one-dimensional components. For correlated
problems, choose exactly one grouping input:

```python
# Let MorphZ select disjoint groups from second-order total correlations.
importance_morph = MorphProposal.fit(
    posterior_samples,
    param_names=model.parameter_names,
    morph_type="2_group",
)

# Or supply groups in memory / from a MorphZ JSON file.
# groups=[[...], ...]
# group_file="groups.json"
```

The training rows initialize neither the live set nor the nested quadrature.
They are used only to fit the normalized fixed density `q0`.

## 4. Run NISMO

```python
from nismo import NISMOSampler

sampler = NISMOSampler(
    model=model,
    importance_morph=importance_morph,
    proposal_scheme="fixed_morph",
    n_live=200,
    rng=42,
    n_workers=1,
    output_path="runs/example",
)

result = sampler.run(
    dlogz=1e-2,
    max_iterations=10_000,
    progress=True,
)

print(f"logZ = {result.logz:.6f} +/- {result.logzerr:.6f}")
print(result.success, result.termination_reason)
```

Omitting both `dlogz` and `stopping` uses `dlogz=1e-3`. A hard resource limit
returns a valid partial result with `success=False`; malformed densities and
missing fixed-importance support raise typed exceptions.

To evaluate one large Morph batch up front and fall back to the
Gaussian-covariance walk only when that pool is exhausted, use:

```python
from nismo import MORWalkSettings, SRWalkSettings

sampler = NISMOSampler(
    model=model,
    importance_morph=importance_morph,
    proposal_scheme="mor-rwalk",
    mor_rwalk_settings=MORWalkSettings(n_proposals=20_000),
    srwalk_settings=SRWalkSettings(n_steps=75),
    n_live=200,
    rng=42,
)
```

`output_path` creates the directory if necessary and automatically saves valid
complete or partial results as:

```text
runs/example/
├── diagnostics.json
├── nested_progress.png
├── run_diagnostics.png
├── run_history.npz
├── weight_health.png
└── weighted_samples.npz
```

The two NPZ files and JSON summary are always written. Install `nismo[plot]`
for the PNG files. Without Matplotlib, NISMO emits a warning after saving the
data files. Existing standard output filenames are replaced; unrelated files
in the directory are preserved.

Set `n_workers` directly on `NISMOSampler` to construct replacement candidates
in parallel. `queue_size` defaults to `n_workers` and can be set explicitly to
control the FIFO prefetch depth.

## 5. Use the posterior

The primary posterior representation is weighted:

```python
points = result.all_points
weights = result.posterior_weights

mean = np.average(points, axis=0, weights=weights)
variance = np.average((points - mean) ** 2, axis=0, weights=weights)
```

For software that requires equal weights, use explicit reproducible systematic
resampling:

```python
equal_samples = result.resample_equal(rng=43, n_samples=10_000)
```

Repeated rows are expected after resampling. Keep `all_points` and
`posterior_weights` for the highest-fidelity summaries.

You can also save a returned result explicitly or omit plots:

```python
result.save("runs/copy")
result.save("runs/data-only", plots=False)
```

## 6. Inspect run health

```python
from nismo import plot_run, summarize

diagnostics = summarize(result)
print(diagnostics.posterior_ess)
print(diagnostics.proposal_acceptance_fraction)
print(diagnostics.thresholds_monotone)

figure, axes = plot_run(result)
figure.show()
```

Plotting functions return Matplotlib objects and never call `show()` or write a
file themselves. See [results and diagnostics](results.md) before interpreting
the theoretical evidence error as an empirical accuracy guarantee.
