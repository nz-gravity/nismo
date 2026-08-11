# Public API reference

All names below are supported public imports from `nismo` unless a module is
shown explicitly. NISMO uses natural logarithms and NumPy arrays with parameter
batches arranged as `(n, ndim)`.

## Model API

### `Model`

Runtime-checkable protocol:

```python
class Model(Protocol):
    ndim: int
    parameter_names: Sequence[str]

    def log_likelihood(self, theta: ndarray) -> ndarray: ...
    def log_prior(self, theta: ndarray) -> ndarray: ...
```

Both methods return shape `(n,)`. The prior must be normalized. `-inf` denotes
zero density; `NaN` and positive infinity are invalid.

### `CallableModel`

```python
CallableModel(
    ndim,
    parameter_names,
    log_likelihood_fn,
    log_prior_fn,
    vectorized=True,
    scalar_likelihood_map=None,
)
```

With `vectorized=False`, each function receives one `(ndim,)` row. An optional
ordered `scalar_likelihood_map(function, rows)` parallelizes only likelihood
rows; prior evaluation remains local.

## Proposal API

### `Proposal`

Runtime-checkable protocol:

```python
class Proposal(Protocol):
    ndim: int

    def sample(self, n: int, rng: np.random.Generator) -> ndarray: ...
    def log_prob(self, theta: ndarray) -> ndarray: ...
```

`sample` returns finite shape `(n, ndim)`. `log_prob` returns the normalized log
density with shape `(n,)`.

### `RefittableProposal`

Extends `Proposal` with:

```python
def refit(self, training_theta: ndarray) -> RefittableProposal: ...
```

The method returns a new proposal and does not mutate the fixed importance
object.

### `MorphProposal`

Construct instances with `fit`, not the low-level initializer:

```python
MorphProposal.fit(
    posterior_samples,
    *,
    morph_type=None,
    group_file=None,
    groups=None,
    param_names=None,
    kde_bw="silverman",
    min_tc=None,
    top_k_greedy=1,
)
```

Choose at most one of `morph_type`, `group_file`, and `groups`. Automatic
grouping uses strings such as `"2_group"`; `groups=[]` selects independent
one-dimensional KDEs. `metadata` is a `MorphMetadata` record containing the
training dimension, parameter names, bandwidth, selected groups, MorphZ
version, and reproducibility note.

Public methods:

```python
proposal.sample(n, rng)
proposal.log_prob(theta)
proposal.refit(training_theta)
```

`MorphProposal.fit` requires the `morph` extra.

## Sampler API

### `NISMOSampler`

```python
NISMOSampler(
    *,
    model,
    importance_morph,
    proposal_scheme="fixed_morph",
    proposal_update_interval=25,
    n_live,
    rng,
    proposal_batch_size=64,
    tie_policy="strict",
    rwalk_settings=None,
    srwalk_settings=None,
    ensemble_rwalk_settings=None,
    parallel=None,
)
```

Alternative constructor:

```python
NISMOSampler.from_posterior_samples(
    *,
    model,
    posterior_samples,
    morph_config,
    n_live,
    rng,
    proposal_batch_size=64,
    proposal_scheme="fixed_morph",
    proposal_update_interval=25,
    tie_policy="strict",
    rwalk_settings=None,
    srwalk_settings=None,
    ensemble_rwalk_settings=None,
    parallel=None,
)
```

`morph_config` is passed to `MorphProposal.fit`. `sampler.citations` returns a
list of `(citation, url)` pairs required by the configured replacement method.

Run method:

```python
sampler.run(
    *,
    dlogz=None,
    stopping=None,
    max_iterations=10_000,
    max_proposals_per_replacement=100_000,
    max_likelihood_calls=None,
    max_wall_time=None,
    progress=False,
) -> NISMOResult
```

Omitting `dlogz` and `stopping` resolves to `dlogz=1e-3`. A callback supplied
as `progress` receives a mapping after every completed iteration.

## Immutable configuration

### `NISMOConfig`

```python
NISMOConfig(
    n_live,
    dlogz=None,
    stopping=None,
    proposal_batch_size=64,
    proposal_scheme="fixed_morph",
    proposal_update_interval=25,
    rwalk_settings=RWalkSettings(),
    srwalk_settings=SRWalkSettings(),
    ensemble_rwalk_settings=EnsembleRWalkSettings(),
    parallel=ParallelSettings(),
    max_iterations=10_000,
    max_proposals_per_replacement=100_000,
    max_likelihood_calls=None,
    max_wall_time=None,
    tie_policy="strict",
)
```

The sampler resolves and stores this complete validated record in
`result.config`.

### Proposal settings

```python
RWalkSettings(walks=None, facc=0.5, ncdim=None)

SRWalkSettings(
    n_steps=25,
    scale=None,
    facc=0.5,
    covariance_shrinkage=0.1,
    covariance_jitter=1e-10,
)

EnsembleMoveWeights(de=0.60, stretch=0.25, gaussian=0.15)

EnsembleRWalkSettings(
    n_walkers=8,
    n_sweeps=4,
    gamma=None,
    jitter_scale=1e-6,
    covariance_shrinkage=0.1,
    covariance_jitter=1e-10,
    move_weights=EnsembleMoveWeights(),
    stretch_scale=1.5,
    gaussian_scale=None,
)

ParallelSettings(n_workers=1, queue_size=None)
```

`EnsembleMoveWeights.active_names_and_probabilities` returns enabled names and
normalized probabilities. `QueueDiagnostics.queue_efficiency` and
`compute_efficiency` are derived properties.

### Stopping settings

```python
StoppingCriterionConfig(name, tolerance)

StoppingPolicy(
    criteria,
    mode="all",
    consecutive=1,
    min_iterations=0,
    stability_window=10,
)
```

Supported criterion names are `remaining_fraction`, `remaining_dlogz`,
`live_logz_error`, `logz_stability`, `live_ess`, and `logzerr`.

## Result API

### `NISMOResult`

Important scalar fields:

```text
logz, logzerr, information, success, termination_reason
niter, nlive, n_likelihood_calls, n_prior_calls, n_proposals
rng_bit_generator, rng_state_initial, rng_state_final
importance_morph_description, warnings, nonfinite_counts
```

Array fields and record fields are documented in [results and
diagnostics](results.md). Convenience API:

```python
result.all_points
result.all_log_psi0
result.posterior_weights
result.resample_equal(rng, n_samples=None)
```

### `RunHistory`

Immutable per-iteration arrays for quadrature, stopping, live-set state,
resource use, MCMC transitions, and adaptive updates. Every array has shape
`(niter,)`.

### Output records

These immutable records are returned within a result and usually should not be
constructed by application code:

- `ProposalUpdateRecord`
- `EnsembleMoveHistory`
- `EvaluationCounts`
- `QueueDiagnostics`
- `ReplacementSnapshot`
- `ReplacementResult`
- `MorphMetadata`

`ReplacementSnapshot` and `ReplacementResult` are exposed for advanced queue
instrumentation; the coordinator owns their construction and consumption.

## Diagnostics API

```python
posterior_ess(result: NISMOResult) -> float
summarize(result: NISMOResult) -> RunDiagnostics
```

`RunDiagnostics` is an immutable summary with posterior ESS, proposal and queue
efficiency, stopping values, and threshold health.

## Plotting API

All functions require the `plot` extra and return `(figure, axes)`:

```python
plot_run(result)
plot_nested_progress(result)
plot_weight_health(result)
plot_posterior_1d(
    result,
    *,
    parameter=0,
    bins=30,
    truth_x=None,
    truth_density=None,
)
```

They never call `show()` or write files.

## Exceptions

All package-specific exceptions inherit from `NISMOError`:

| Exception | Meaning |
|---|---|
| `ConfigurationError` | Invalid sampler, proposal, stopping, or resource setting |
| `InvalidModelOutput` | Invalid model input/output shape or non-finite output |
| `InvalidProposalOutput` | Invalid proposal sample or log-density output |
| `MissingOptionalDependency` | Requested optional feature is not installed |
| `ProposalSupportError` | Fixed `q0` is zero at a finite target-integrand point |
| `NumericalInvariantError` | Evidence or diagnostic invariant is violated |

## Public aliases and version

```python
ProposalScheme  # Literal proposal-scheme names
EnsembleMoveName  # Literal ensemble move names
__version__  # Installed distribution version
```
