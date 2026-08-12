# Results and diagnostics

`NISMOSampler.run(...)` returns a frozen `NISMOResult`. Stored arrays are
read-only and contain enough information to reproduce its quadrature without
reevaluating the model.

## Evidence and termination

| Field | Meaning |
|---|---|
| `logz` | Natural logarithm of the evidence, including the final live correction |
| `logzerr` | Theoretical nested-sampling approximation `sqrt(information / nlive)` |
| `information` | Estimated nested-sampling information |
| `success` | `True` only for scientific stopping |
| `termination_reason` | Scientific stop or hard-limit reason |
| `niter`, `nlive` | Completed replacements and live-set size |
| `n_likelihood_calls`, `n_prior_calls`, `n_proposals` | Resource counts |

Scientific termination reasons are `remaining_evidence` for the `dlogz` API and
`stopping_criteria` for an explicit policy. Both set `success=True`.

Hard-stop reasons include `max_iterations`, `max_likelihood_calls`,
`max_wall_time`, `max_proposals_per_replacement`,
`constrained_sampling_exhausted`, `insufficient_eligible_survivors`,
`insufficient_eligible_walkers`, and `plateau_stall`. These return a coherent
partial quadrature with `success=False`; inspect `termination_reason` before
using its evidence estimate.

Malformed model/proposal output and missing fixed-importance support instead
raise subclasses of `NISMOError`, because a coherent result cannot be formed.

## Weighted posterior

Dead points are followed by the final live points in every combined view:

```python
points = result.all_points
log_weights = result.log_posterior_weights
weights = result.posterior_weights
log_psi0 = result.all_log_psi0
```

`weights` sums to one. The separated arrays retain likelihood, prior, fixed
importance density, pseudo-likelihood, volume/weight, and tie-breaker values:

- `dead_points`, `dead_log_likelihood`, `dead_log_prior`, `dead_log_q0`,
  `dead_log_psi0`, `dead_tie_breakers`, `dead_log_x`, `dead_log_weights`;
- `final_live_points`, `final_live_log_likelihood`, `final_live_log_prior`,
  `final_live_log_q0`, `final_live_log_psi0`, `final_live_tie_breakers`.

Equal-weight output is available when required:

```python
samples = result.resample_equal(rng=123, n_samples=10_000)
```

The default sample count is `niter + nlive`. Systematic resampling adds Monte
Carlo variation and can repeat points, so weighted summaries remain primary.

## History

Every `RunHistory` array has shape `(niter,)`. Important groups are:

- quadrature: `iteration`, `log_x`, `log_delta_x`, `logz_dead`, `logz_live`,
  `logz_total`, `information`, and `logzerr`;
- stopping: `remaining_fraction`, `remaining_dlogz`, `live_ess`,
  `live_mean_rse`, `live_logz_error`, `logz_stability`, and `stopping_streak`;
- live set: `discarded_log_psi`, `live_min_log_psi`,
  `live_median_log_psi`, and `live_max_log_psi`;
- cost: `proposals`, `likelihood_calls`, `acceptance_fraction`, and
  `elapsed_seconds`;
- MCMC: `mh_acceptance_fraction`, `constraint_pass_fraction`,
  `mcmc_accepted`, `mcmc_moved`, and `mcmc_completed`;
- adaptive proposal: `proposal_revision`, `proposal_update_attempts`, and
  `proposal_update_failures`.

`logz_stability` is `NaN` until its configured window is full. Non-MCMC modes
store `NaN` for MCMC fractions and zero for MCMC counts.

## Proposal and queue records

`result.proposal_updates` records every scheduled adaptive refit, including the
iteration, success, active revision, training size, metadata, and any captured
error. The original fixed density is described by
`result.importance_morph_description`.

For `en-rwalk`, `result.ensemble_move_history` contains proposed, valid,
accepted, and moved count arrays with shape `(niter, 3)` in
`("de", "stretch", "gaussian")` order. Other schemes store `None`.

`result.queue_diagnostics` exposes job and candidate counts plus:

```python
result.queue_diagnostics.queue_efficiency
result.queue_diagnostics.compute_efficiency
result.queue_diagnostics.wasted_prefetch_likelihood_calls
```

Queue efficiency measures consumed candidates per completed job. Compute
efficiency measures used prefetch likelihood calls per total prefetch calls.

## Summaries

```python
from nismo import posterior_ess, summarize

ess = posterior_ess(result)
diagnostics = summarize(result)
```

`RunDiagnostics` includes posterior ESS, relative ESS, proposal acceptance,
maximum proposals per replacement, threshold monotonicity, a conservative
remaining-integral diagnostic, final stopping values, and queue/compute
efficiency. These are run-health indicators, not substitutes for repeated-run
calibration.

## Plots

Install `nismo[plot]`, then import any helper from the top-level package:

```python
from nismo import (
    plot_nested_progress,
    plot_posterior_1d,
    plot_run,
    plot_weight_health,
)

figure, axes = plot_run(result)
figure.savefig("run.png", dpi=160)
```

All helpers return `(figure, axes)`, never call `show()`, and never write files.
`plot_posterior_1d` accepts `parameter`, `bins`, and optional equally shaped
`truth_x` / `truth_density` arrays.
