# Phase 2 API

MINS uses natural logarithms throughout and expects point batches with shape
`(n, ndim)`.

## Model

`Model` defines:

```python
ndim: int
parameter_names: Sequence[str]
log_likelihood(theta) -> ndarray shape (n,)
log_prior(theta) -> ndarray shape (n,)
```

`log_prior` must be normalized and must include all constants. Values of
`-inf` represent zero density. NaN and positive infinity are rejected.

`CallableModel` adapts explicit vectorized functions. Set `vectorized=False`
only when both functions accept individual `(ndim,)` rows; the adapter does not
catch user exceptions to guess vectorization.

For an expensive scalar likelihood, pass a persistent ordered mapper as
`scalar_likelihood_map`. This is used for likelihood rows only; prior and
importance-density evaluation remain in the parent process so an `en-rwalk`
half-ensemble can still use a batched `q0` evaluation:

```python
from concurrent.futures import ProcessPoolExecutor

# Both callables must be module-level and picklable when using processes.
with ProcessPoolExecutor(max_workers=4) as pool:
    model = CallableModel(
        ndim=ndim,
        parameter_names=names,
        log_likelihood_fn=slow_scalar_log_likelihood,
        log_prior_fn=scalar_log_prior,
        vectorized=False,
        scalar_likelihood_map=pool.map,
    )
    sampler = MINSampler(
        model=model,
        importance_morph=importance_morph,
        n_live=nlive,
        rng=seed,
    )
    result = sampler.run(dlogz=0.1)
```

Use this only when each likelihood call is large relative to process-pool
overhead. A vectorized likelihood is usually faster; pool construction must be
protected by the usual `if __name__ == "__main__":` guard on spawn platforms.

## MorphProposal

```python
importance_morph = MorphProposal.fit(
    posterior_samples,
    morph_type="2_group",
    param_names=("x", "y"),
    kde_bw=0.02,
)
```

`morph_type="{k}_group"` follows MorphZ's automatic grouped workflow:
`Nth_TC.compute_and_save_tc` computes the k-order total correlations, then
`GroupKDE` greedily selects non-overlapping groups. For example, use
`morph_type="2_group"` or `morph_type="3_group"`; the literal string
`"n_group"` is not valid. Intermediate TC files are held in a temporary
directory and removed after fitting.

Alternatively, load precomputed MorphZ entries with `group_file=...`, or pass
them directly through `groups=...`; `groups=[]` selects independent
one-dimensional components. These three grouping inputs are mutually
exclusive. Training data must be finite with shape `(n_samples, ndim)`.

The selected structure is recorded in metadata:

```python
importance_morph.metadata.selected_groups
importance_morph.metadata.single_parameters
```

The adapter uses the installed `morphZ.GroupKDE` directly. It copies training
data, resolves grouping data in memory, calls `GroupKDE.resample` for draws,
and calls the same fitted object's normalized `logpdf` for `log_prob`. MorphZ
0.4.1 uses integer seeds, so the adapter derives one from the sampler's
explicit NumPy Generator per resample. The inspected MorphZ implementation
restores legacy global RNG state.

`importance_morph.refit(live_theta)` returns a new `MorphProposal` using a
deep-copied version of the original bandwidth and grouping configuration. It
does not mutate the importance object. Automatic `morph_type` grouping is
recomputed; file-backed grouping is retained in memory.

## MINSampler

```python
sampler = MINSampler(
    model=model,
    importance_morph=importance_morph,
    proposal_scheme="fixed_morph",
    proposal_update_interval=25,
    n_live=500,
    rng=np.random.default_rng(42),
    proposal_batch_size=64,
    tie_policy="strict",
    parallel=ParallelSettings(n_workers=1, queue_size=1),
)
result = sampler.run(
    dlogz=1e-4,
    max_iterations=20_000,
    max_proposals_per_replacement=200_000,
    max_likelihood_calls=None,
    max_wall_time=None,
    progress=True,
)
```

`dlogz` is the maximum estimated change in log evidence caused by adding the
current mean-live remaining-evidence estimate:
`log(Z_dead + Z_live) - log(Z_dead)`. It must be positive and finite.
`dlogz` and `stopping` are mutually exclusive. Omitting both uses
`dlogz=1e-3`. The multi-criterion API uses immutable
`StoppingCriterionConfig` and `StoppingPolicy` values:

```python
from mins import StoppingCriterionConfig, StoppingPolicy

n_live = sampler.n_live
stopping = StoppingPolicy(
    criteria=(
        StoppingCriterionConfig("remaining_dlogz", 1e-2),
        StoppingCriterionConfig("live_logz_error", 2e-3),
        StoppingCriterionConfig("logz_stability", 5e-3),
    ),
    mode="all",
    consecutive=3,
    min_iterations=n_live,
    stability_window=n_live,
)
result = sampler.run(
    stopping=stopping,
    max_iterations=20_000,
)
```

The resolved policy is retained in `result.config.stopping`. See
[Stopping criteria](stopping.md) for metric definitions, custom `"all"` and
`"any"` policies, persistence rules, and limitations.

The constructor validates dimensions and owns the supplied generator. Initial
live points always come from the fixed `importance_morph`. Every candidate is
evaluated with
`log_psi0 = log_likelihood + log_prior - log_q0`, where `log_q0` always comes
from that original object.

The proposal scheme is one of `"fixed_morph"`, `"adaptive_morph"`, `"rwalk"`,
`"s-rwalk"`, or `"en-rwalk"`. The immutable MCMC configuration objects are
public:

```python
from mins import (
    EnsembleMoveWeights,
    EnsembleRWalkSettings,
    RWalkSettings,
    SRWalkSettings,
)

rwalk = MINSampler(
    model=model,
    importance_morph=importance_morph,
    proposal_scheme="rwalk",
    rwalk_settings=RWalkSettings(
        walks=50,
        facc=0.5,
        ncdim=None,
    ),
    n_live=200,
    rng=42,
)

statistical_rwalk = MINSampler(
    model=model,
    importance_morph=importance_morph,
    proposal_scheme="s-rwalk",
    srwalk_settings=SRWalkSettings(
        n_steps=50,
        scale=None,
        facc=0.5,
        covariance_shrinkage=0.1,
        covariance_jitter=1e-10,
        move_weights=EnsembleMoveWeights(
            de=0.60,
            stretch=0.25,
            gaussian=0.15,
        ),
        stretch_scale=2.0,
        gaussian_scale=None,
    ),
    n_live=200,
    rng=42,
)

ensemble = MINSampler(
    model=model,
    importance_morph=importance_morph,
    proposal_scheme="en-rwalk",
    ensemble_rwalk_settings=EnsembleRWalkSettings(
        n_walkers=8,
        n_sweeps=6,
        gamma=None,
        jitter_scale=1e-6,
        covariance_shrinkage=0.1,
        covariance_jitter=1e-10,
    ),
    n_live=200,
    rng=42,
)
```

`MINSConfig` stores all three settings objects, and
`MINSampler.from_posterior_samples` accepts them as well. Ensemble walker
count must be even, at least four, and no greater than `n_live - 1`.

`proposal_scheme="fixed_morph"` uses the importance Morph for constrained
draws and preserves the original rejection sampler. With
`proposal_scheme="adaptive_morph"`, the sampler refits a separate proposal
Morph from all `n_live` current live rows before replacements 26, 51, 76, and
so on for the default interval. Successful fits atomically replace only the
proposal Morph. Failed fits retain the previous proposal and are retried at the
next interval.

Adaptive candidates are accepted directly when they pass the old-Morph
`log_psi0` constraint. This samples the refitted proposal under the constraint,
not constrained `q0`; no MH or density-ratio correction is applied. Adaptive
`logZ` and posterior weights are therefore heuristic and may be biased.

`proposal_scheme="rwalk"` selects an eligible survivor uniformly and performs
exactly `walks` symmetric ellipsoidal-ball Metropolis transitions. Omitting
`walks` uses `model.ndim + 20`. The initial scale is 1 and is tuned toward
`facc` after each completed replacement. A Dynesty-style single ellipsoid is
built from the complete live set and cached for approximately
`walks * n_live` calls. `ncdim` must be omitted or equal to `model.ndim`.
`proposal_scheme="s-rwalk"` instead attempts exactly `n_steps` Gaussian
Metropolis transitions using a regularized covariance computed from the
survivors and frozen for that replacement. Its initial scale defaults to
`2.38 / sqrt(model.ndim)`; an explicit `scale` replaces that initial value.
After each complete chain, the scale is tuned toward `facc` using the same
recursion as `rwalk`.

`proposal_scheme="en-rwalk"` selects distinct eligible survivors and performs
split-half ensemble Metropolis--Hastings updates. One move is selected for each
half-update using `EnsembleMoveWeights`; weights are relative, zero-weight moves
are omitted, and active weights are normalized internally. They remain fixed
rather than being adapted from acceptance rates. The default
`EnsembleRWalkSettings()` uses 60% DE, 25% stretch, and 15% Gaussian weights.
Any pure configuration consumes no move-selection random draw; use
`EnsembleMoveWeights(de=1, stretch=0, gaussian=0)` for the former DE-only
sequence.

For dimension $D$, the available moves are:

- `de`: \(\theta_i'=\theta_i+\gamma(\theta_j-\theta_k)+\sigma Lz\),
  with distinct ordered references and default
  \(\gamma=2.38/\sqrt{2D}\);
- `stretch`: \(\theta_i'=\theta_j+z(\theta_i-\theta_j)\), where
  \(g(z)\propto z^{-1/2}\) on `[1 / stretch_scale, stretch_scale]`;
- `gaussian`: \(\theta_i'=\theta_i+sLz\), with default
  \(s=2.38/\sqrt D\).

The DE and Gaussian moves are symmetric. Stretch proposals require the
mandatory Hastings correction `(D - 1) * log(z)`. The complementary half is
frozen while its active half is generated and evaluated. The regularized
survivor-covariance factor is constructed lazily, excludes the discarded point,
and is frozen for a complete replacement. A stretch-only replacement does not
construct it. The final replacement is selected uniformly from every final
walker, including unchanged walkers.

For all three MCMC modes, a candidate must first pass the fixed-`log_psi0`
constraint. Its generic log acceptance ratio is
`min(0, proposed.log_q0 - current.log_q0 + log_hastings_ratio)`; the Hastings
term is zero except for stretch moves. The target is constrained fixed `q0`, not
the posterior, prior, likelihood, or a uniform constrained density. The
discarded point supplies the threshold but is outside the strict constraint and
is never an MCMC starting state. Proposals are not clipped or redrawn at prior
boundaries.

The complete contract, proposal equations, resource behavior, and mixing
limitations are in [MCMC replacements](mcmc_replacements.md).

For a standard random-walk sampler, `sampler.citations` reports the Skilling
(2006) method citation.

`tie_policy="strict"` is correct for ordinary continuous pseudo-likelihoods.
Use `"randomized_plateau"` for targets with exact nonzero-probability ties. It
stores an independent uniform auxiliary value for every dead and live point.

Progress is silent by default. `progress=True` enables an optional
`tqdm.auto` live display suitable for terminals and notebooks. Install it
through `.[progress]`. It displays:

- iteration and `n_live`;
- likelihood calls and constrained-proposal efficiency;
- current total `logZ` and theoretical `logZerr`;
- the current stopping streak and required consecutive count.

The hard iteration limit is not rendered as a convergence percentage.
Criterion-specific values such as `dlogZrem`, `liveErr`, `ESSlive`, and
stability appear only when their criterion is enabled. Proposal revision and
failure fields appear only after adaptive proposal updates occur.
`remaining_fraction` remains in callback mappings and `RunHistory`, but is not
shown in the terminal display.

A custom callable can be passed instead of `True`. It receives a mapping after
every completed iteration with the displayed quantities plus
`remaining_dlogz`, `live_mean_rse`, `logz_stability`, dead/live evidence,
per-iteration proposal counts, live `logPsi` range, and elapsed time. It also
receives integer met flags named `criterion_<name>_met` for each enabled
criterion. Library code does not otherwise print, display, or write files.

Progress mappings and `RunHistory` also include `proposal_revision`,
`proposal_update_attempts`, and `proposal_update_failures`. MCMC iterations
add `mh_acceptance_fraction`, `constraint_pass_fraction`, `mcmc_accepted`,
`mcmc_moved`, and `mcmc_completed`. For non-MCMC modes the two fractions are
`NaN` and the three counts are zero. The existing `acceptance_fraction`
retains its cumulative nested-replacement-efficiency meaning.

## Result

`MINSResult` is frozen and its arrays are read-only. Key scalar fields are:

- `logz`, `logzerr`, `information`;
- `success`, `termination_reason`;
- `niter`, `nlive`, likelihood/prior/proposal counts;
- complete `config` and initial/final RNG state representations.

Dead and final-live arrays separately store points, `log_likelihood`,
`log_prior`, fixed `log_q0`, fixed `log_psi0`, volume/weight values, and tie
breakers.
`log_posterior_weights` covers dead then live points. Convenience properties
`all_points`, `all_log_psi0`, and `posterior_weights` use that same ordering.

`proposal_updates` contains immutable records for every scheduled adaptive
refit, including its boundary iteration, success, active revision, training
row count, proposal metadata, and any error. `importance_morph_description`
describes the fixed density used for all `log_q0` values.

For `proposal_scheme="en-rwalk"`, `ensemble_move_history` is an immutable
`EnsembleMoveHistory` with canonical `names == ("de", "stretch", "gaussian")`.
Its read-only `proposed`, `valid`, `accepted`, and `moved` arrays each have shape
`(niter, 3)`. A per-move moved count is the number of distinct walkers moved by
that move during the replacement; one walker can appear in more than one move
column. Other proposal schemes store `ensemble_move_history=None`.

For consumers that require unweighted samples:

```python
equal_samples = result.resample_equal(
    rng=123,
    n_samples=10_000,
)
```

`resample_equal` uses randomized systematic resampling and has no Dynesty
dependency. The explicit seed or NumPy Generator is required for
reproducibility. If `n_samples` is omitted, it returns `niter + nlive` rows.
Repeated points are expected. Retain the original points and weights for
highest-fidelity posterior summaries.

`RunHistory` stores every completed threshold, volume interval, cumulative
dead/live/total log evidence, information, `logzerr`, remaining fraction,
remaining log-evidence increment, live pseudo-likelihood ESS, live-mean RSE,
live log-evidence error, log-evidence stability, stopping streak, live
pseudo-likelihood range, calls, proposal counts, acceptance fraction, elapsed
time, MCMC constraint/MH counts and fractions, and proposal update counters.
Every array is read-only and has shape
`(niter,)`; early `logz_stability` entries are `NaN` until its exact window is
available.

## Replacement prefetch parallelism

`ParallelSettings(n_workers=N, queue_size=Q)` constructs up to `Q` complete
replacement attempts per proposal epoch. `N > 1` uses one persistent process
pool; `N = 1, Q > 1` exercises identical FIFO queue semantics without
multiprocessing. Only the coordinator removes a dead point, updates prior
volume and evidence, or evaluates stopping criteria. Candidates retain their
creation threshold and proposal revision and are revalidated against the
current lexicographic constraint immediately before use.

`rwalk`, `s-rwalk`, and `en-rwalk` freeze scale and geometry for each refill,
then aggregate scale tuning once the epoch has drained. Adaptive Morph refit
boundaries also end epochs. The coordinator reserves likelihood-call budgets
before submission, so completed worker work cannot overshoot
`max_likelihood_calls`.

`result.queue_diagnostics` contains submitted/completed/consumed/stale/
invalidated counts, refill count, total/used/wasted prefetch likelihood calls,
and the derived `queue_efficiency` and `compute_efficiency`. Models, proposals,
and their callables must be pickleable for `n_workers > 1`. Fine-grained
`CallableModel.scalar_likelihood_map` execution is disabled inside replacement
workers to avoid nested multiprocessing.

## Failure semantics

Scientific termination reasons are:

- `remaining_evidence` for the `dlogz` path;
- `stopping_criteria` for an explicit `StoppingPolicy`.

Both have `success=True`. Hard stops include:

- `max_iterations`;
- `max_likelihood_calls`;
- `max_wall_time`;
- `constrained_sampling_exhausted`;
- `max_proposals_per_replacement`;
- `insufficient_eligible_survivors`;
- `insufficient_eligible_walkers`;
- `plateau_stall`.

Hard stops return a valid partial result with the current final-live correction.
Malformed model/proposal output and proposal support failure raise typed
exceptions because no statistically coherent result can be formed.
Adaptive Morph refit failures are different: they are recorded in
`proposal_updates`, the last working proposal remains active, and the sampler
retries at the next configured interval.

## Diagnostics and plots

`mins.diagnostics.summarize(result)` reports posterior ESS, proposal acceptance,
maximum proposals per replacement, threshold monotonicity, the separate
conservative max-live remainder, and the final values of every stopping
diagnostic and streak, plus replacement-queue and compute efficiency.

`plot_run`, `plot_nested_progress`, `plot_weight_health`, and
`plot_posterior_1d` return Matplotlib objects. `plot_nested_progress` shows the
stored live-set pseudo-likelihood envelope and median, remaining live
`logz_live`, and discarded-threshold trajectory. Plot helpers never call
`show` or save files.
