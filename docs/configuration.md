# Configuration reference

## Sampler construction

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
    beta=1.0,
    beta_mc_samples=100_000,
    tie_policy="strict",
    srwalk_settings=None,
    mor_rwalk_settings=None,
    ensemble_rwalk_settings=None,
    n_workers=1,
    queue_size=None,
    output_path=None,
)
```

`model.ndim` must equal `importance_morph.ndim`. `rng` is a NumPy Generator or
integer seed; a supplied Generator is consumed in place. `n_live` must be at
least two, although ensemble sampling imposes a higher effective minimum.

`beta` controls power tempering of the fixed importance density:

```text
g_beta(theta) = q0(theta)**beta / C_beta
log_psi_beta = log_likelihood + log_prior - beta * log_q0 + log_C_beta
```

The default `beta=1` is the standard sampler and exactly preserves direct
`q0` initialization. Values in `0 < beta < 1` diffuse the importance density
and are supported by `mor-rwalk` and `s-rwalk`. NISMO estimates `C_beta`
directly from `beta_mc_samples` independent `q0` draws, then importance-resamples
the finite candidate batch to obtain the initial tempered pool. The candidate
count must be at least the requested Morph-pool size (`n_live` for `s-rwalk`).
The endpoint `beta=0` is rejected because the corresponding uniform-density
normalizer is generally undefined on unbounded support.

`tie_policy="strict"` is appropriate for ordinary continuous
pseudo-likelihoods. Use `"randomized_plateau"` when exact ties have nonzero
probability.

`output_path` accepts a string or path-like directory. When set, `run()` saves
weighted samples, complete iteration history, a strict JSON diagnostic summary,
and the standard run, nested-progress, and weight-health plots. The directory
is created before sampling so invalid or inaccessible paths fail early. Reusing
the same path replaces NISMO's standard output files.

## Replacement proposals

| Scheme | Replacement behavior | Status |
|---|---|---|
| `fixed_morph` | Rejection draws from fixed `q0` under the current `log_psi0` constraint | Default |
| `adaptive_morph` | Periodically refits a separate proposal to the live set | Heuristic; evidence may be biased |
| `mor-rwalk` | Initializes from one pre-evaluated Morph or power-tempered pool, consumes its randomized remainder, then switches permanently to `s-rwalk` | Initial batch must fit the likelihood budget; finite-pool and finite-walk behavior must be calibrated |
| `s-rwalk` | Gaussian-covariance MH transitions targeting constrained `q0**beta` | Finite-walk mixing must be calibrated |
| `en-rwalk` | Split-ensemble DE/stretch/Gaussian MH mixture targeting constrained `q0` | Finite-walk mixing must be calibrated |

All MCMC schemes start from eligible surviving live points, never the discarded
threshold point. Passing the constraint is not enough: the generic acceptance
ratio also contains the fixed-density ratio `q0(proposed) / q0(current)` and a
Hastings correction where required.

### Morph-pool then Gaussian random walk

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

`n_proposals` is the total one-time pool and must be at least `n_live`.
At `beta=1`, NISMO draws the pool directly from `q0`. At `beta<1`, it draws
`beta_mc_samples` candidates from `q0`, estimates
`C_beta = mean(q0**(beta - 1))`, and resamples the pool with replacement using
weights proportional to `q0**(beta - 1)`. NISMO evaluates the resulting pool
together, randomly selects `n_live` members as the initial live set, and
retains the remainder in randomized order. Each early replacement is the
first retained proposal satisfying the current constraint.
When no remaining proposal passes, NISMO discards the exhausted remainder and
uses `s-rwalk` for every subsequent replacement by default. Optional beta=1
refills are described in [parallel execution](parallel-execution.md).

The randomized order is statistically important: sorting the pool and always
choosing the lowest passing `log_psi0` would bias replacements toward the
constraint. `max_likelihood_calls`, when supplied, must be at least
`n_proposals` so that the initial vectorized batch can be completed.

### Gaussian-covariance random walk

```python
from nismo import SRWalkSettings

settings = SRWalkSettings(
    n_steps=25,
    scale=None,
    facc=0.5,
    dynamic_steps=True,
    max_steps=5_000,
    target_zero_move_probability=1e-3,
    acceptance_window=20,
    max_step_growth=2.0,
    zero_accept_scale_factor=0.5,
    zero_move_policy="allow",
    covariance_shrinkage=0.1,
    covariance_jitter=1e-10,
    covariance_update_interval=1,
    covariance_rebuild_interval=None,
    profile=False,
)
```

The MH invariant density is constrained `q0**beta`: a symmetric proposal from
`theta` to `theta_prime` has log acceptance ratio
`beta * (log_q0(theta_prime) - log_q0(theta))` after the transformed-integrand
constraint is satisfied. Thus the fallback from a diffused `mor-rwalk` pool
continues to target the same `g_beta` density.

The live-set mean and scatter are computed once and updated in `O(ndim**2)`
after each committed replacement. The survivor covariance uses a Cholesky
factor when possible and is frozen for a complete chain. A queued refill shares
one prepared factor across all its jobs.

`covariance_update_interval` controls how many committed replacements may reuse
a factor; its default of one refreshes at every serial replacement or queue
snapshot. `covariance_rebuild_interval=None` reconstructs mean and scatter from
the complete live set every `n_live` commits to limit floating-point drift.
`scale=None` starts at `2.38 / sqrt(ndim)` and then adapts toward `facc`.

By default, `s-rwalk` also adapts the length of the next complete walk from the
recent MH acceptance rate. It selects enough transitions to make the estimated
probability of zero accepted moves no larger than
`target_zero_move_probability`, subject to `n_steps`, `max_steps`, and the
per-update `max_step_growth`. A queue epoch uses one frozen length and adapts
only after every prefetched job in that epoch has completed.

If an entire walk or queue epoch accepts nothing, the next length grows by
`max_step_growth` and the proposal scale is additionally multiplied by
`zero_accept_scale_factor`. This avoids conditioning a replacement on eventual
acceptance, which would change the invariant MH distribution. Consequently,
the default `zero_move_policy="allow"` retains a rare valid self-transition.
Use `zero_move_policy="stop"` to terminate with `srwalk_stalled` instead of
inserting a duplicate survivor. Set `dynamic_steps=False` to recover a fixed
`n_steps` controller.

The actual length and movement outcome of every replacement are stored in
`result.history.mcmc_completed` and `result.history.mcmc_moved`.

Set `profile=True` to populate `result.srwalk_diagnostics` with component
timings, factor refresh counts, squared displacement, and stale-candidate
fraction. Profiling is opt-in because high-resolution timers add overhead to
cheap likelihoods.

### Ensemble random walk

```python
from nismo import EnsembleMoveWeights, EnsembleRWalkSettings

settings = EnsembleRWalkSettings(
    n_walkers=8,
    n_sweeps=4,
    gamma=None,
    jitter_scale=1e-6,
    covariance_shrinkage=0.1,
    covariance_jitter=1e-10,
    move_weights=EnsembleMoveWeights(
        de=0.60,
        stretch=0.25,
        gaussian=0.15,
    ),
    stretch_scale=1.5,
    gaussian_scale=None,
)
```

`n_walkers` must be even, at least four, and no greater than `n_live - 1`.
Move weights are relative and normalized internally; zero disables a move.
Use `EnsembleMoveWeights(de=1, stretch=0, gaussian=0)` for pure differential
evolution. The stretch move includes the `(ndim - 1) * log(z)` Hastings term.

## Replacement prefetching

```python
sampler = NISMOSampler(
    ...,
    n_workers=8,
    queue_size=8,
)
```

`queue_size=None` resolves to `n_workers`. Defaults therefore resolve to one
worker and a one-item queue, preserving serial behavior. Completed replacements
are consumed FIFO and revalidated against the current threshold. Larger queues
can waste likelihood calls when prefetched candidates become stale.

As in Dynesty, one pool job constructs one complete constrained replacement;
steps within an individual MCMC chain remain sequential. Queue refills use a
synchronous ordered `map` with `chunksize=1`. The coordinator selects each
`s-rwalk` starting survivor and freezes its covariance factor before dispatch,
while the model and fixed importance proposal are initialized once per worker.
The nested quadrature, live-point commit, stopping logic, and proposal
adaptation remain coordinator-only. Proposal adaptation occurs only after the
current FIFO queue epoch drains.

`queue_size > 1` requires `n_workers > 1`. Normally set `queue_size` equal to
`n_workers`; a larger queue is useful only when replacement runtimes are highly
variable and may increase stale speculative work.

The older `parallel=ParallelSettings(...)` form remains accepted for backward
compatibility, but it cannot be combined with `n_workers` or `queue_size`.

For `n_workers > 1`, the model, proposal, and their callables must be pickleable.
Use module-level functions and protect process-starting application code with
`if __name__ == "__main__":`. `CallableModel.scalar_likelihood_map` is disabled
inside replacement workers to prevent nested process pools.

## Run controls

```python
result = sampler.run(
    dlogz=None,
    stopping=None,
    max_iterations=10_000,
    max_proposals_per_replacement=100_000,
    max_likelihood_calls=None,
    max_wall_time=None,
    progress=False,
)
```

`progress` accepts `False`, `True`, or a callback receiving a mapping after each
completed iteration. `True` requires the `progress` extra.

`max_iterations`, `max_proposals_per_replacement`, `max_likelihood_calls`, and
`max_wall_time` are hard safeguards. They do not indicate convergence and yield
`success=False` if reached before a scientific stopping condition.

## Stopping policies

The shorthand

```python
result = sampler.run(dlogz=1e-3)
```

is equivalent to one `remaining_dlogz <= 1e-3` condition. `dlogz` and
`stopping` are mutually exclusive. A custom immutable policy is:

```python
from nismo import StoppingCriterionConfig, StoppingPolicy

policy = StoppingPolicy(
    criteria=(
        StoppingCriterionConfig("remaining_dlogz", 1e-2),
        StoppingCriterionConfig("live_logz_error", 2e-3),
        StoppingCriterionConfig("logz_stability", 5e-3),
    ),
    mode="all",
    consecutive=3,
    min_iterations=200,
    stability_window=200,
)
result = sampler.run(stopping=policy)
```

Thresholds in this example are illustrative, not universal defaults.

| Criterion | Pass condition | Meaning |
|---|---|---|
| `remaining_dlogz` | value <= tolerance | Estimated change from adding the mean-live evidence remainder |
| `remaining_fraction` | value <= tolerance | Fraction of current total evidence represented by the live remainder |
| `live_logz_error` | value <= tolerance | First-order finite-live-mean uncertainty transmitted to log evidence |
| `logz_stability` | value <= tolerance | Range of total `logz` over `stability_window` values |
| `live_ess` | value >= tolerance | Kish ESS of transformed live contributions |
| `logzerr` | value <= tolerance | Theoretical `sqrt(H / n_live)` approximation |

`mode="all"` requires every condition; `mode="any"` requires one.
`min_iterations` gates evaluation and `consecutive` requires a persistent run
of passing decisions. `logz_stability` cannot pass until its full window exists.

`live_logz_error`, stability, and theoretical `logzerr` are incomplete error
measures. They do not detect missing proposal support or guarantee that a
finite-length MCMC replacement has mixed.

## Vectorized and rolling execution

`ParallelSettings` also supports opt-in `vectorized` and `process` backends,
independent chain batch sizes, bounded ordered scheduling, initialization
chunks, and worker thread limits. See [parallel execution](parallel-execution.md)
for configuration, diagnostics, beta corrections, and validation limits.
