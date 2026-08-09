# Stopping criteria

NISMO uses a single remaining-log-evidence-increment criterion by default:

```python
result = sampler.run(dlogz=1e-3)
```

This is equivalent to a single `remaining_dlogz <= 1e-3` criterion and
retains `termination_reason="remaining_evidence"`. Omitting `dlogz` also uses
`1e-3`.

The multi-criterion API is opt-in:

```python
from nismo import StoppingCriterionConfig, StoppingPolicy

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

This recommended experimental configuration says that:

1. the mean-live remainder would change accumulated `logZ` by at most `0.01`;
2. estimated live-set uncertainty transmitted to `logZ` is at most `0.002`;
3. the range of the last `n_live` total `logZ` estimates is at most `0.005`;
4. every condition holds for three consecutive completed iterations.

These values are illustrative calibration choices, not universally valid
constants. They require repeated-run calibration for the target family,
proposal, and live count.

## Metrics

### Remaining log-evidence increment

\[
\Delta\log Z_{\rm rem}
=\log(Z_{\rm dead}+Z_{\rm live})-\log Z_{\rm dead}
=\log\left(1+\frac{Z_{\rm live}}{Z_{\rm dead}}\right).
\]

`remaining_dlogz <= tolerance` estimates the maximum change in the accumulated
dead-point log evidence caused by adding the current mean-live remainder. It is
measured in natural-log units, and its tolerance must be positive and finite;
it need not be less than one. Before any finite positive dead evidence has
been accumulated, its value is `+inf` and the criterion cannot pass.

### Remaining fraction

\[
f_{\rm live}=\frac{Z_{\rm live}}{Z_{\rm total}}.
\]

`remaining_fraction <= tolerance` measures the magnitude of the live evidence
contribution. It does not directly estimate evidence uncertainty. Its
tolerance must be strictly between zero and one.

The two metrics are related exactly by

\[
\mathtt{remaining\_dlogz}
=-\log(1-\mathtt{remaining\_fraction}),
\]

when the remaining fraction lies strictly between zero and one. They are
similar only for a small live remainder and remain independently selectable.

### Live effective sample size and mean RSE

For transformed live likelihoods
\(\Psi_j=L_j\pi_j/q_j\), NISMO calculates Kish ESS in log space:

\[
N_{\rm eff,live}
=\frac{(\sum_j\Psi_j)^2}{\sum_j\Psi_j^2}.
\]

`live_ess >= tolerance` is an optional criterion. The threshold must lie
between one and `n_live`. NISMO also stores

\[
\operatorname{RSE}_{\rm live}
=\sqrt{\max\left(
\frac{N/N_{\rm eff,live}-1}{N-1},0\right)}
\]

as `live_mean_rse`. It is a diagnostic rather than a separately selectable
criterion.

### Live log-evidence error

\[
\sigma_{\log Z,\rm live}
\approx f_{\rm live}\operatorname{RSE}_{\rm live}.
\]

`live_logz_error <= tolerance` is a first-order delta-method estimate of Monte
Carlo uncertainty transmitted from representing the remaining integral by a
finite live-set mean. It is not a complete error estimate. In particular it
does not include:

- stochastic shrinkage uncertainty;
- uncertainty from fitting Morph;
- missing proposal support or undiscovered modes;
- correlated or invalid constrained draws.

Equal finite live contributions have zero empirical live-mean uncertainty.
The recommended `remaining_dlogz` completion guard prevents that fact alone
from causing immediate termination.

### Evidence stability

`logz_stability` is the range of the most recent `stability_window` total
`logZ` values, including the current iteration. It is unavailable (`NaN`) and
cannot pass until exactly that many values exist.

`logz_stability <= tolerance` is supporting evidence only: a biased estimate
can be stable. Do not use stability alone.

### Theoretical log-evidence error

`logzerr <= tolerance` uses

\[
\mathtt{logzerr}=\sqrt{H/N_{\rm live}}.
\]

This is a theoretical nested-sampling approximation, not a complete
calibration guarantee, and is not included in the recommended hybrid policy.

## Combining and persisting conditions

`mode="all"` requires every enabled criterion to pass. For example:

```python
StoppingPolicy(
    criteria=(
        StoppingCriterionConfig("remaining_fraction", 1e-2),
        StoppingCriterionConfig("live_ess", 400.0),
    ),
    mode="all",
    consecutive=2,
)
```

`mode="any"` stops when the first enabled criterion passes:

```python
StoppingPolicy(
    criteria=(
        StoppingCriterionConfig("live_logz_error", 1e-2),
        StoppingCriterionConfig("logzerr", 1e-2),
    ),
    mode="any",
)
```

`"any"` is easier to satisfy and can terminate substantially earlier than
`"all"`. Use it only when each individual condition is independently
sufficient.

`min_iterations` gates the combined result. `consecutive` then specifies how
many completed iterations in a row must pass. The streak resets to zero
immediately on a failure. Stopping is checked only after a completed
replacement.

## Results, progress, and hard limits

`result.config.stopping` contains the fully resolved immutable policy.
`RunHistory` stores `remaining_fraction`, `remaining_dlogz`, `live_ess`,
`live_mean_rse`, `live_logz_error`, `logz_stability`, `logzerr`, and
`stopping_streak` for every completed iteration. Progress callbacks receive
the same metrics, the required consecutive count, and an integer
`criterion_<name>_met` flag for each enabled criterion.

For a run with no completed replacement, diagnostic summaries use `NaN` for
final floating-point stopping metrics and zero for the final stopping streak.

An explicit policy that passes terminates with
`termination_reason="stopping_criteria"` and `success=True`. Hard limits and
constrained-sampling exhaustion remain independent failures with
`success=False`, even if some—but not all—enabled conditions have passed.

Stopping does not alter the evidence estimator. NISMO continues to use
deterministic \(X_i=\exp(-i/N)\), the existing dead-point weights, and the same
final live-point correction.
