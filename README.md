# NISMO

NISMO is an experimental implementation of Morph-assisted nested importance
sampling for post-processing Bayesian posterior samples.

> **Development status:** pre-alpha research software. The importance Morph is
> fixed and non-defensive. Missing importance support can bias the evidence
> estimate. The optional adaptive proposal scheme is heuristic and can add
> further bias; the API is not yet stable and no PyPI release has been made.

## Installation

Install [uv](https://docs.astral.sh/uv/) and sync a development environment:

```bash
uv sync --extra dev
```

The scientific Morph adapter requires MorphZ 0.4.1.dev2 or newer. If MorphZ is
already installed from its source repository, the base package can be installed
without resolving the optional extra:

```bash
uv sync --no-extra
```

For a normal user installation with Morph and the terminal/notebook progress
bar:

```bash
uv sync --extra morph --extra progress
```

To run commands in the project environment, prefix them with `uv run`.

## Minimal API sketch

```python
import numpy as np

from nismo import (
    CallableModel,
    EnsembleMoveWeights,
    EnsembleRWalkSettings,
    NISMOSampler,
    MorphProposal,
    ParallelSettings,
    RWalkSettings,
    SRWalkSettings,
    StoppingCriterionConfig,
    StoppingPolicy,
)

model = CallableModel(
    ndim=1,
    parameter_names=("x",),
    log_likelihood_fn=lambda x: -0.5 * x[:, 0] ** 2,
    log_prior_fn=lambda x: -0.5 * x[:, 0] ** 2 - 0.5 * np.log(2.0 * np.pi),
)

posterior_samples = np.random.default_rng(7).normal(size=(2_000, 1))
importance_morph = MorphProposal.fit(
    posterior_samples,
    param_names=model.parameter_names,
    groups=[],
)
sampler = NISMOSampler(
    model=model,
    importance_morph=importance_morph,
    proposal_scheme="fixed_morph",
    n_live=100,
    rng=42,
)
result = sampler.run(
    dlogz=1e-3,
    max_iterations=5_000,
    progress=True,
)

print(result.logz, result.logzerr, result.termination_reason)

equal_samples = result.resample_equal(
    rng=43,
    n_samples=10_000,
)
```

The `dlogz` interface remains the default scientific behavior. It bounds the
estimated log-evidence increment from adding the mean-live remainder,
`log(Z_dead + Z_live) - log(Z_dead)`. A multi-criterion policy is available as
an opt-in experimental alternative:

```python
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
result = sampler.run(stopping=stopping, max_iterations=20_000)
```

These thresholds are illustrative calibration choices, not universal
constants; calibrate them with repeated runs for the target family.
`remaining_dlogz` is the estimated log-evidence correction,
`live_logz_error` estimates uncertainty transmitted from the finite live-set
mean, and `remaining_fraction` remains available separately for the live
evidence share. See the [stopping guide](docs/stopping.md) for the complete API
and limitations.

Set `proposal_scheme="adaptive_morph"` to refit a separate proposal Morph from
the complete equal-weight live set every `proposal_update_interval=25`
completed iterations. The original `importance_morph` remains fixed and still
defines `log_q0` and `log_psi0`. Adaptive candidates are accepted directly
after the `log_psi0` constraint, so this mode is a heuristic: it does not in
general draw from the constrained importance Morph and its `logZ` may be
biased.

Three constrained Metropolis replacement schemes preserve the fixed importance
target \(q_0\) without an external MCMC dependency:

```python
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

All three start from eligible surviving live points, never the discarded
threshold point. The generic constrained MH log ratio is
`min(0, proposed.log_q0 - current.log_q0 + log_hastings_ratio)`.
`en-rwalk` chooses one move per split-half update from fixed relative weights:
differential evolution, a Goodman--Weare stretch move with the
mandatory `(ndim - 1) * log(z)` Hastings correction, or a symmetric Gaussian
move using frozen survivor covariance. Weights are normalized internally and
are not adaptively tuned. `EnsembleRWalkSettings()` defaults to the 60% DE,
25% stretch, and 15% Gaussian mixture shown above. Pure DE remains available by
setting `EnsembleMoveWeights(de=1, stretch=0, gaussian=0)` explicitly.

Finite walk length preserves the constrained target but does not make the
replacement independent or guarantee adequate mixing. Calibrate walk lengths,
move weights, and live-set size with repeated runs; low MH acceptance and
unchanged ensemble walkers are mixing warnings, not reasons to condition output
on movement. `result.ensemble_move_history` exposes read-only proposed, valid,
accepted, and moved matrices in `("de", "stretch", "gaussian")` order. See
[MCMC replacements](docs/mcmc_replacements.md).

When `walks` is omitted, standard `rwalk` uses `model.ndim + 20`, matching
Dynesty's top-level default. It starts at scale 1 and adapts toward `facc`.
`s-rwalk` uses a Gaussian proposal transformed by a regularized covariance of
the current survivors. Its default scale is `2.38 / sqrt(model.ndim)` and is
adapted toward `facc` after every completed replacement.

Complete replacements can be prefetched concurrently while dead-point and
quadrature updates remain serial:

```python
parallel_sampler = NISMOSampler(
    model=model,
    importance_morph=importance_morph,
    proposal_scheme="rwalk",
    rwalk_settings=RWalkSettings(walks=50),
    n_live=200,
    rng=42,
    parallel=ParallelSettings(n_workers=8, queue_size=8),
)
```

The pool persists for the run. Results are consumed in submission-order FIFO,
not completion order, and every candidate is checked against the current
constraint before one live point is replaced. The defaults
`n_workers=1, queue_size=1` retain the original random-number stream and serial
behavior. Larger queues can waste work as the threshold advances; inspect
`result.queue_diagnostics.queue_efficiency`, `compute_efficiency`, stale and
wasted-call counts when tuning them. Multiprocess models and proposals must be
pickleable (in particular, use module-level functions rather than lambdas).
An existing `CallableModel.scalar_likelihood_map` is disabled inside workers
to prevent nested process pools.

The terminal live display reports iteration, live-point count, likelihood
calls, proposal efficiency, `logZ`, theoretical `logZerr`, and the stopping
streak without treating the hard iteration limit as a convergence percentage.
Criterion-specific metrics appear only when enabled. Proposal revision fields
appear only after adaptive proposal updates are used. The remaining-evidence
fraction remains available in callbacks and history rather than the terminal
display.

`result.all_points` and `result.posterior_weights` are the primary weighted
posterior representation. `result.resample_equal(...)` provides reproducible
equal-weight draws for tools that do not accept weights.

The normalized prior, Morph target transformation, evidence quadrature, and
stopping semantics are defined in
[the mathematical contract](docs/mathematical_contract.md). See the
[API guide](docs/api.md), [stopping guide](docs/stopping.md), and
[MCMC replacement guide](docs/mcmc_replacements.md), and
[Phase 2 tutorial](docs/tutorial.md) before using the estimator.

## Development

```bash
uv run pytest -m "not slow"
uv run ruff format --check .
uv run ruff check .
uv build
uvx twine check dist/*
```

Phase reports and exact validation commands are stored under
[`docs/phases/`](docs/phases/).

The non-CI eggbox comparison in
[`benchmarks/compare_stopping_policies.py`](benchmarks/compare_stopping_policies.py)
records repeated-seed accuracy, cost, failure-rate, and speed-up summaries for
two `remaining_dlogz` tolerances and a hybrid policy.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
