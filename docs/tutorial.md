# Fixed-importance Morph evidence tutorial

Phase 2 estimates evidence after posterior sampling. It does not discover a
posterior from the original prior.

## 1. Supply representative posterior samples

Use an array whose columns match the model coordinates:

```python
posterior_samples = np.load("posterior_samples.npy")
```

Every material posterior mode must be represented. The non-defensive Phase 2
proposal cannot repair omitted support.

## 2. Define normalized model densities

```python
model = CallableModel(
    ndim=2,
    parameter_names=("x", "y"),
    log_likelihood_fn=log_likelihood,
    log_prior_fn=normalized_log_prior,
)
```

Both functions receive `(n, 2)` batches and return `(n,)`. Include all prior
normalization constants.

## 3. Fit the importance Morph once

```python
importance_morph = MorphProposal.fit(
    posterior_samples,
    morph_type="2_group",
    param_names=model.parameter_names,
    kde_bw="silverman",
)
```

MorphZ computes all second-order total correlations, greedily selects disjoint
groups, and fits one fixed `GroupKDE`. This object remains the importance
density `q0` for the complete run.

## 4. Run

```python
sampler = NISMOSampler(
    model=model,
    importance_morph=importance_morph,
    proposal_scheme="fixed_morph",
    n_live=500,
    rng=np.random.default_rng(42),
)
result = sampler.run(
    dlogz=1e-4,
    max_iterations=20_000,
    max_proposals_per_replacement=200_000,
    progress=True,
)
```

The terminal live display's `dlogZrem` field is the estimated change in
accumulated log evidence from adding the mean-live remainder. The scientific
stop occurs when `dlogZrem <= dlogz`. The display has no convergence
percentage because `max_iterations` is only a hard limit. Criterion-specific
fields appear only when enabled; the live-evidence fraction remains available
in history and callbacks. Hard resource-limit stops are shown separately.

Inspect termination before interpreting evidence:

```python
print(result.success, result.termination_reason)
print(result.logz, result.logzerr)
print(summarize(result))
```

`logzerr` is the theoretical \(\sqrt{H/N_{\rm live}}\) approximation and does
not include all Morph-fit uncertainty. Repeat complete runs and compare with
direct importance sampling under the same fixed Morph proposal.

Every displayed statistic remains available after the run:

```python
result.history.logz_total
result.history.logzerr
result.history.information
result.history.remaining_fraction
result.history.remaining_dlogz
result.history.acceptance_fraction
result.history.likelihood_calls
result.history.proposal_revision
```

To experiment with periodic live-set proposal fits, select:

```python
sampler = NISMOSampler(
    model=model,
    importance_morph=importance_morph,
    proposal_scheme="adaptive_morph",
    proposal_update_interval=25,
    n_live=500,
    rng=np.random.default_rng(42),
)
```

The proposal is refitted from all current live rows before replacements 26,
51, 76, and so on. The original importance Morph still computes every
`log_q0` and `log_psi0`. Threshold-only adaptive acceptance is heuristic,
however: it samples from the refitted proposal under the constraint rather
than from constrained `q0`, so the resulting `logZ` can be biased.

For a constrained Metropolis replacement that preserves fixed `q0`, choose a
Dynesty-style walk, the statistically specified Gaussian walk, or an ensemble
walk:

```python
from nismo import (
    EnsembleMoveWeights,
    EnsembleRWalkSettings,
    ParallelSettings,
    RWalkSettings,
    SRWalkSettings,
)

rwalk_sampler = NISMOSampler(
    model=model,
    importance_morph=importance_morph,
    proposal_scheme="rwalk",
    rwalk_settings=RWalkSettings(walks=50, facc=0.5),
    n_live=200,
    rng=42,
)

statistical_rwalk_sampler = NISMOSampler(
    model=model,
    importance_morph=importance_morph,
    proposal_scheme="s-rwalk",
    srwalk_settings=SRWalkSettings(n_steps=50, facc=0.5),
    n_live=200,
    rng=42,
)

ensemble_sampler = NISMOSampler(
    model=model,
    importance_morph=importance_morph,
    proposal_scheme="en-rwalk",
    ensemble_rwalk_settings=EnsembleRWalkSettings(
        n_walkers=16,
        n_sweeps=6,
        move_weights=EnsembleMoveWeights(
            de=0.60,
            stretch=0.25,
            gaussian=0.15,
        ),
    ),
    n_live=200,
    rng=42,
)
```

To prefetch whole constrained walks using persistent worker processes, pass
the same settings to any sampler above:

```python
parallel = ParallelSettings(n_workers=8, queue_size=8)
```

Nested-sampling commits remain serial and candidates are consumed in FIFO
submission order after revalidation. Start with `queue_size=n_workers`, then
inspect `result.queue_diagnostics`; a large stale fraction means the queue is
too deep for the rate at which the constraint advances. Worker callables must
be pickleable, so define likelihood and prior functions at module scope rather
than as lambdas when `n_workers > 1`.

The discarded live point defines the constraint, is not used as a chain start,
and does not enter the ensemble covariance. `EnsembleRWalkSettings()` uses the
60/25/15 mixture shown above. The weights are fixed and relative, are normalized
internally, and are not adapted from acceptance rates. Pure DE is available
with `EnsembleMoveWeights(de=1, stretch=0, gaussian=0)`. DE
uses ordered complementary differences, stretch uses the required
`(ndim - 1) * log(z)` Hastings correction, and Gaussian uses a local frozen
survivor covariance with default scale `2.38 / sqrt(model.ndim)`.

MCMC replacements remain correlated with eligible survivors. Increasing
`walks`, `n_steps`, or `n_sweeps` can improve mixing but does not create an
independence guarantee. Examine `result.history.constraint_pass_fraction`,
`result.history.mh_acceptance_fraction`, `result.history.mcmc_moved`, and, for
the ensemble scheme, `result.ensemble_move_history`. The latter contains
read-only `(niter, 3)` proposed, valid, accepted, and moved arrays ordered as
`("de", "stretch", "gaussian")`. Calibrate settings with repeated complete
runs. Do not discard unchanged outputs or select only walkers that moved;
conditioning on movement biases the replacement. See the dedicated
[MCMC replacement guide](mcmc_replacements.md).

If `walks` is omitted, NISMO uses `model.ndim + 20`. Standard `rwalk` starts
with scale 1 and tunes it toward the configured `facc` after every completed
replacement. `ncdim` is accepted for Dynesty API familiarity but must be
omitted or equal to the complete model dimension.
The `s-rwalk` geometry is a regularized covariance of the survivors, frozen
for all `n_steps` transitions. Its scale starts at `2.38 / sqrt(model.ndim)`
unless explicitly supplied and adapts toward `facc` between replacements.

## 5. Obtain equal-weight posterior samples

The native result is weighted:

```python
points = result.all_points
weights = result.posterior_weights
```

When a downstream tool requires equal weights, resample directly from the
result:

```python
equal_samples = result.resample_equal(
    rng=123,
    n_samples=10_000,
)
```

The method uses systematic resampling and returns a new `(n_samples, ndim)`
array. Duplicate rows are normal. Weighted estimates from `all_points` and
`posterior_weights` should be preferred when the consumer supports them.

## 6. Plot stored results

```python
figure, axes = plot_run(result)
figure.savefig("run.png")

from nismo import plot_nested_progress

progress_figure, progress_axes = plot_nested_progress(result)
progress_figure.savefig("nested_progress.png", dpi=150)
```

Plots consume only the result. The full executable Gaussian example is
[`examples/phase2_gaussian.py`](../examples/phase2_gaussian.py).
