# Parallel execution and vectorized chains

This implements the near-term execution work in the 7 September 2026
parallelization review, based on main commit `ad52a65`: measurement and cheaper
preparation, vectorized independent walks, shared initialization workers,
ordered scheduling, and less coordinator work. New backends are opt-in.

Every mode retains **one death and one replacement at a time**, with
`log X_i = -i / n_live`. A batch of endpoints is a speculative replacement
queue. Every endpoint is checked against the current augmented threshold
before insertion. Reference and tuning revision metadata are separate;
only a reference change invalidates an otherwise valid old symmetric walk. The fixed importance density, beta, and normalizer remain
unchanged during the evidence run.

## Choose an execution mode

Existing calls keep compatibility behavior:

```python
sampler = NISMOSampler(
    model=model,
    importance_morph=morph,
    proposal_scheme="s-rwalk",
    n_live=400,
    rng=42,
    n_workers=4,
    queue_size=8,
)
```

For models and Morph densities that evaluate arrays efficiently:

```python
from nismo import NISMOSampler, ParallelSettings, SRWalkSettings

sampler = NISMOSampler(
    model=model,
    importance_morph=morph,
    proposal_scheme="s-rwalk",
    n_live=400,
    rng=42,
    srwalk_settings=SRWalkSettings(n_steps=75, profile=True),
    parallel=ParallelSettings(
        backend="vectorized",
        queue_size=32,
        chains_per_task=16,
        evaluation_chunk_size=1024,
    ),
)
result = sampler.run(dlogz=0.1)
```

For expensive scalar likelihoods, use persistent processes and ordered retirement:

```python
parallel = ParallelSettings(
    backend="process",
    scheduler="rolling",
    n_workers=4,
    queue_size=8,
    chains_per_task=1,
    adaptation_interval=20,
    worker_threads=1,
)
```

Pass this as `parallel=parallel` to `NISMOSampler`; do not also pass direct
`n_workers` or `queue_size` arguments. Use an `if __name__ == "__main__":` guard
when starting processes in a script. Model/proposal objects must be pickleable.
`chains_per_task > 1` also enables vectorization within each process task.
These backends support `s-rwalk` and the walk phase of `mor-rwalk`.

| Setting | Meaning |
| --- | --- |
| `backend="compatibility"` | Original scalar kernel, coordinator RNG and blocking map epochs. Default `(1,1)` preserves the original serial trajectory. |
| `backend="vectorized"` | One process, independent chains evaluated together at each MH transition. |
| `backend="process"` | Persistent process workers; scalar or batched complete chains. |
| `scheduler="epoch"` | Freeze tuning for an entire queue, collect all results, then retire in order. Default. |
| `scheduler="ordered"` | Retrieve each task in submission order while later epoch tasks can still run; tune at the epoch boundary. |
| `scheduler="rolling"` | Refill vacancies on deterministic retirement events; tune after `adaptation_interval` complete retired chains. |
| `queue_size` | Total outstanding **plus buffered chains**, independently bounded. Defaults to `n_workers`. |
| `chains_per_task` | Maximum walkers evaluated together; queue tails may be smaller. Defaults to 1. |
| `worker_threads` | Optional BLAS/OpenMP limit inside spawned workers. `None` preserves existing threading. Coordinator settings are preserved. |
| `initialization="auto"` | In process mode, parallelize explicit scalar `CallableModel`s; use local vectorized initialization for other models. Compatibility keeps its original initialization. |
| `initialization="process"` | Explicitly distribute initialization chunks for other scalar model wrappers; requires more than one worker. |
| `initialization="serial"` | Evaluate initialization locally. |
| `evaluation_chunk_size` | Maximum initial/beta-density evaluation chunk size. Coordinates are generated centrally; at most one pending chunk per worker is serialized. |
| `diagnostic_interval` | Median/display cadence. Default 1. Scientific stopping and quadrature still run every committed death. |

Changing queue capacity or adaptation settings changes a run's trajectory.
In the new modes every job, including singleton tails, gets its own seed.
Replaying fixed settings with deterministic model functions is independent of
worker completion order. A valid symmetric walk with older tuning can still
be inserted; new geometry and scale affect only subsequently submitted work.
No candidate is chosen because it finished first, and stale walks are not
continued or mined for intermediate states.

## Initialization, beta and Morph refills

One owned pool serves initial model evaluations, chunked beta-MC densities, and
replacement chains. An explicit scalar mapper on `CallableModel` is removed
inside workers to prevent nested pools. Beta normalization is supplied as an
immutable scalar to each evaluation/chain task once it is known.

The direct-MC identity remains

`C_beta = E_q0[exp((beta - 1) * log q0)]`.

Beta-MC log densities are reused for selected initial points. The former
weighted **without-replacement** pool construction is corrected to multinomial
resampling **with replacement**. Otherwise, choosing all MC candidates simply
reproduced `q0` even at beta below one.

This correction is still finite-candidate sampling-importance-resampling,
not exact independent continuous sampling from the tempered reference.
Diagnostics report MC ESS, log-normalizer error, unique selected candidates,
and the sampling method. Duplicates and shared-candidate dependence remain;
MC ESS and the estimated error do not establish adequate tail coverage.
Increasing workers does not reduce the shared normalizer uncertainty.

For beta=1, an optional Morph refill policy is available:

```python
from nismo import MORWalkSettings

settings = MORWalkSettings(
    n_proposals=30_000,
    refill=True,
    refill_min_acceptance=0.05,
    refill_max_batches=10,
)
```

After exhausting a randomized stream, the sampler commits one pilot walk.
It compares the preceding stream's useful candidates per evaluation second
with that walk's measured rate before deciding to draw a fresh stream.
The minimum acceptance, batch cap and likelihood budget also apply. Fresh
points retain random proposal order. Refills are off by default and are
restricted to beta=1. Because this optional policy uses measured costs, its
source-switching decisions can change with machine load.

## Diagnostics and budgets

`result.execution_diagnostics` is also saved in `diagnostics.json`:

- `phase_seconds` includes pool startup/shutdown, initialization, proposal and
  beta preparation, task preparation, map/ordered wait, serial evolution,
  coordinator commit/stopping, finalization and total elapsed run time. Some
  aggregate phases overlap their component phases; do not sum them all.
  Morph fit time is available when using `from_posterior_samples`; it occurs
  before `run_total`. Externally fitted proposals report zero fit time.
- `worker_execution_seconds` is summed task execution time, separate from
  elapsed wall time. `profile=True` adds job start/end times, retrieval
  iteration, PID, observed thread counts and covariance-normalized endpoint
  displacement. Batched evaluation/execution times are divided among their
  chains so their sum counts a task only once. These are timings, not CPU
  utilization estimates.
- `queue_diagnostics` includes used, stale, failed and unused work, candidate
  age in committed deaths, and all wasted likelihood calls. `invalidated`
  retains its existing meaning, including valid unused endpoints at shutdown;
  `unused` is an additional classification, not an extra disjoint total.
- When `diagnostic_interval > 1`, unsampled median history entries are `NaN`.
  All other scientific history entries and stopping decisions are still
  recorded at every death. The first and final planned/scientific-stop rows
  include a median. A resource stop between those rows can leave a final `NaN`.

Every new batched task reserves worst-case full-chain likelihood capacity
before it starts. Prior rejections release unused reservations on retrieval.
A too-small remainder may therefore be left unused. Interrupted partial
chains have no replacement endpoint. Normal termination drains outstanding
work and counts it even when unused; errors terminate/join the owned pool and
propagate. A user likelihood already executing cannot be preempted safely,
so a wall-time limit prevents further commits but shutdown can take longer.
A complete initial live set is constructed before a partial result is possible.

## Reproduce and extend the measurements

From the repository, with development/Morph dependencies installed:

```bash
PYTHONPATH=src python benchmarks/execution_scaling.py \
  --dimension 60 --n-live 100 400 --workers 1 2 4 8 \
  --backends compatibility vectorized ordered rolling \
  --queue-size 16 --chains-per-task 8 --repeat 10 \
  --output execution-gaussian.jsonl

PYTHONPATH=src python benchmarks/execution_scaling.py \
  --dimension 60 --n-live 400 --workers 1 4 --morph \
  --queue-size 16 --chains-per-task 8 --repeat 10 \
  --output execution-morph.jsonl
```

The runner fixes one fitted Morph across all backends and sampling seeds,
limits numerical libraries to one thread, checks available CPU quota, and
reports initialization, steady replacements/s, volume progression/s, complete
wall time, actual CPU time including child processes, evidence error/scatter/
coverage, posterior moments and a simple mode-weight diagnostic. Use
`--target correlated` or `--target mixture` for additional targets, and
`--delay` for controlled state-dependent scalar work. Use `--training-seed`
in separate invocations to study training variability.

A `--factory module:function` hook accepts the real shell/PTA model and fixed
proposal: `factory(args)` must return `(model, proposal, exact_logz_or_None)`.
This allows reusing the user's exact training samples/grouping and likelihood.
The short, iteration-limited measurements in the accompanying validation
record isolate execution cost; they are not evidence-accuracy release gates.

The review's broader 30D/60D shell and PTA grid, ten-or-more repeated runs,
lineage/insertion-order calibration and high-core-count scaling remain release
validation work on those workloads. True batch deletion, changing live-count
quadrature, and JAX/GPU execution remain the review's explicitly deferred
algorithmic extension. No execution mode here changes the volume estimator.
