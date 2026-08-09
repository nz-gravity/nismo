# NISMO Parallel Replacement Queue Implementation Plan

## Implementation Progress Ledger

Last updated: 2026-08-08 (production implementation and repository verification
complete; extended research-validation matrix remains available for follow-up).

| Phase | Status | Implemented evidence |
|---|---|---|
| Phase 1 | Complete | `src/nismo/replacement.py` defines immutable copied snapshots, jobs, results, and worker-local counts; the coordinator retains all commits. The default singleton path preserves the original RNG stream. |
| Phase 2 | Complete | Deterministic `ReplacementQueue` FIFO semantics, current-threshold/tie revalidation, stale rejection, revision invalidation, and serial `queue_size > 1` execution are implemented and tested. |
| Phase 3 | Complete | One persistent `ProcessPoolExecutor` per run constructs complete `rwalk`, `s-rwalk`, and `en-rwalk` attempts. Ordered `executor.map` consumption is scheduling-independent; job RNG entropy is generated deterministically by the coordinator. |
| Phase 4 | Complete | Queue refills are proposal epochs. Walk scales and live geometry are frozen per epoch, tuning is aggregated only when the epoch drains, and adaptive-Morph refit boundaries cap an epoch. |
| Phase 5 | Complete | Public `ParallelSettings(n_workers, queue_size)` defaults to `(1, 1)` with `queue_size=n_workers` when omitted. Existing fine-grained `CallableModel.scalar_likelihood_map` values are disabled inside worker jobs to prevent nested pools. |
| Phase 6 | Complete | Exact queue/call diagnostics and derived efficiencies are stored in every result. Coordinator reservations enforce the hard likelihood-call ceiling. Deadline checks stop commits after an expired refill and pool cleanup uses `cancel_futures=True`. |
| Phase 7 | In progress | Multi-seed Gaussian equivalence tests cover all three MCMC schemes. `benchmarks/parallel_replacement_queue.py` implements the requested worker/queue matrix and metrics. A focused spawn-worker expensive-likelihood run measured 1.45x speedup for `rwalk` at 2 workers/queue 2. Broader correlated, eggbox, shell, multimodal, and plateau parallel regressions remain to be run/added before declaring the research validation complete. |

### Verification recorded so far

- Pre-change baseline: `python -m pytest -q` — 243 passed.
- Final Ruff check: passed for `src`, `tests`, and the benchmark harness.
- Final mypy check: passed for all 20 source modules. A pre-existing Morph
  adapter `Literal` inference issue was resolved with a type-only annotation.
- Complete test suite: `python -m pytest -q` — 258 passed.
- Replacement unit and worker integration tests include FIFO, stale/revision
  rejection, deterministic parallel results, singleton-stream compatibility,
  hard call reservations, deadline behavior, and adaptive epoch boundaries.
- Parallel Gaussian multi-seed statistical tests: 3 passed.
- Focused `rwalk` benchmark (`delay=0.002`, `n_live=30`): singleton
  4.014 s; 2 workers/queue 2, 2.776 s (1.45x), with statistically compatible
  evidence estimates and explicit stale/wasted accounting.
- `uv` is not installed in the current environment, so verification uses the
  existing Python environment despite the repository preference for `uv`.

### Continuation notes

1. Add/run the remaining distribution-level parallel statistical regressions
   listed under Phase 7, preferably as slow/overnight tests.
2. Run the full benchmark matrix on representative expensive real likelihoods;
   retain CSV output and choose production queue sizes from measured stale-work
   tradeoffs.
3. Preserve unrelated pre-existing worktree deletions; they are not part of
   this implementation.

## Goal

Implement Dynesty-style parallelism for NISMO by **constructing several possible replacement live points ahead of time in parallel**, while keeping the actual nested-sampling evolution serial.

The objective is to use available CPU resources efficiently without changing the statistical semantics of the current sampler.

The implementation must **not** update multiple live points simultaneously.

---

## Core Statistical Invariants

These rules are mandatory throughout the implementation:

1. **Never update several NISMO live points simultaneously.**
2. **One queued candidate represents one hypothetical future replacement.**
3. **Only the main sampler/coordinator may remove or replace a live point.**
4. **Every queued candidate must be revalidated against the current `log_psi0` threshold before use.**
5. **Dead-point ordering must remain serial.**
6. **The prior-volume evolution \(X_i\) must remain serial.**
7. **Evidence, information, uncertainty, stopping criteria, and history updates must remain serial.**
8. **Parallelize complete replacement constructions, not small individual likelihood calls.**
9. **Use persistent workers and independent deterministic RNG streams.**
10. **Do not create nested multiprocessing pools for `en-rwalk`; retain its existing internal vectorization.**
11. **`n_workers=1` and `queue_size=1` must preserve the current serial behaviour.**
12. **Complete and validate one phase before proceeding to the next.**

---

# Target Architecture

The main sampler remains the authoritative owner of:

- live-point arrays;
- current worst live point;
- current `log_psi0` threshold;
- dead-point ordering;
- quadrature state;
- evidence estimates;
- stopping state;
- proposal adaptation;
- counters;
- run history;
- progress reporting.

Workers only construct possible replacement points.

Conceptually:

```text
                     current live set
                           |
                     current threshold
                           |
          +----------------+----------------+
          |                |                |
       worker 1         worker 2         worker N
          |                |                |
      build full        build full       build full
      replacement       replacement      replacement
          |                |                |
      candidate A      candidate B      candidate C
          +----------------+----------------+
                           |
                    replacement queue
                           |
                     SERIAL consume
                           |
                 revalidate candidate
                           |
                   replace ONE worst
                           |
                  update X and logZ
                           |
                    next iteration
```

---

# Phase 1 — Refactor Replacement Construction

## Goal

Separate:

1. **constructing a replacement candidate**, and
2. **committing a nested-sampling iteration**.

Do not add multiprocessing yet.

## Current concern

The current `NISMOSampler.run()` controls both candidate generation and all nested-sampling state updates.

This should be split so replacement construction can later run inside workers without allowing workers to mutate sampler state.

## Recommended new data structures

Create a worker-safe immutable snapshot:

```python
@dataclass(frozen=True)
class ReplacementSnapshot:
    threshold: float
    threshold_tie_breaker: float

    live_theta: np.ndarray
    live_log_likelihood: np.ndarray
    live_log_prior: np.ndarray
    live_log_q0: np.ndarray
    live_log_psi0: np.ndarray
    live_tie_breakers: np.ndarray

    proposal_revision: int
    seed: int
```

Create a replacement result:

```python
@dataclass(frozen=True)
class ReplacementResult:
    attempt: ConstrainedAttempt

    threshold_at_creation: float
    threshold_tie_breaker_at_creation: float

    proposal_revision: int

    likelihood_calls: int
    prior_calls: int
```

The exact fields may be adjusted to fit the existing code, but worker results must contain everything required by the coordinator without mutating shared state.

## Refactor target

Move from:

```python
attempt = _draw_replacement(...)
# immediately mutate sampler state
```

toward:

```python
snapshot = prepare_replacement_snapshot(...)
result = build_replacement(snapshot, ...)
commit_replacement(result)
```

`build_replacement(...)` must not modify global sampler state.

`commit_replacement(...)` remains serial.

## Suggested files

Primary files:

```text
src/nismo/sampler.py
src/nismo/constrained.py
src/nismo/mcmc.py
```

Recommended new module:

```text
src/nismo/replacement.py
```

Do not substantially rewrite the mathematics in:

```text
src/nismo/quadrature.py
src/nismo/stopping.py
```

These should continue to operate in the main coordinator.

## Phase 1 acceptance criteria

Before continuing:

- existing test suite passes;
- serial sampler outputs remain unchanged for fixed seeds where deterministic equality is expected;
- `n_workers=1`, `queue_size=1` behaviour is equivalent to the current sampler;
- no multiprocessing has been introduced yet.

---

# Phase 2 — Implement Replacement Queue Semantics Serially

## Goal

Implement the queue logic before introducing processes.

This separates statistical correctness from multiprocessing complexity.

## Add a replacement queue

Suggested abstraction:

```python
class ReplacementQueue: ...
```

Responsibilities:

- store replacement candidates;
- preserve deterministic ordering;
- expose the next candidate;
- reject stale candidates;
- track queue diagnostics;
- invalidate candidates when necessary.

## Candidate revalidation

A queued candidate was generated using an older threshold:

```text
threshold_at_creation
```

Before using it, it must satisfy the **current** threshold.

Use the existing constraint logic:

```python
passes_constraint(
    candidate.log_psi0,
    candidate.tie_breaker,
    threshold=current_threshold,
    threshold_tie_breaker=current_threshold_tie,
    tie_policy=tie_policy,
)
```

If it does not pass:

```text
discard candidate
try next queued candidate
```

Never weaken the current threshold to accommodate a queued candidate.

## Queue ordering

Use deterministic FIFO ordering initially.

Do not select candidates based on which worker finishes first.

The eventual parallel implementation should therefore behave independently of operating-system process scheduling as much as possible.

## Initial configuration

Add something similar to:

```python
@dataclass(frozen=True)
class ParallelSettings:
    n_workers: int = 1
    queue_size: int = 1
```

At this phase `queue_size > 1` may generate replacement candidates serially.

This is intentional.

## Phase 2 acceptance criteria

Before multiprocessing:

- `queue_size=1` behaves like the serial implementation;
- `queue_size>1` correctly discards candidates that no longer satisfy the current threshold;
- dead-point updates remain one-at-a-time;
- evidence and stopping calculations remain unchanged;
- queue ordering is deterministic;
- statistical regression tests show no bias relative to the current sampler.

---

# Phase 3 — Parallelize Whole Replacement Constructions

## Goal

Use a persistent multiprocessing pool to construct several complete replacement candidates simultaneously.

This is the main performance implementation.

## Important design decision

Do **not** parallelize tiny likelihood calls with repeated pool communication.

The repository has previously tested fine-grained multiprocessing and found it slower.

Instead parallelize:

```python
build_replacement(job)
```

where each job performs enough work to amortize process overhead.

For example:

```python
jobs = [make_replacement_job(snapshot, seed) for seed in child_seeds]

results = pool.map(build_replacement, jobs)
```

Each worker should execute an entire:

- `rwalk`,
- `s-rwalk`, or
- `en-rwalk`

replacement construction.

## Worker responsibilities

A worker should:

1. receive an immutable replacement job;
2. construct its own RNG from the assigned seed;
3. run the complete constrained replacement algorithm;
4. perform required likelihood/prior/Morph evaluations locally;
5. return one `ReplacementResult`;
6. return local evaluation counts.

Workers must not:

- mutate the parent live set;
- update log evidence;
- update stopping criteria;
- update shared proposal state;
- modify run history.

## Persistent worker pool

Create the pool once per sampler run.

Do not repeatedly create/destroy worker pools during iterations.

Support initially:

```python
n_workers >= 1
queue_size >= 1
```

Recommended default:

```python
queue_size = n_workers
```

## RNG design

Use independent deterministic RNG streams.

Recommended:

```python
seed_sequence = np.random.SeedSequence(master_seed)
child_sequences = seed_sequence.spawn(number_of_jobs)
```

Each replacement job receives its own child seed.

Do not allow workers to inherit identical RNG state through process forking.

## Counters

`BatchEvaluator` currently maintains mutable counters.

Do not attempt to share this mutable object between processes.

Each worker should return local counts, for example:

```python
@dataclass(frozen=True)
class EvaluationCounts:
    likelihood_calls: int
    prior_calls: int
    outside_prior: int
    zero_likelihood: int
```

The coordinator aggregates them.

## Initial proposal priority

Implement multiprocessing support in this order:

1. `rwalk`
2. `s-rwalk`
3. `en-rwalk`

Do not prioritize `fixed_morph` multiprocessing yet.

`fixed_morph` already benefits from batching/vectorization and may not gain enough from process-level replacement jobs to justify the overhead.

## Phase 3 acceptance criteria

- multiprocessing produces statistically equivalent results to serial execution;
- workers construct complete replacements;
- only the coordinator commits live-point changes;
- no nested multiprocessing occurs;
- worker RNG streams are independent and reproducible;
- call counters remain exact;
- resource cleanup is correct when runs finish or error.

---

# Phase 4 — Proposal Epochs and Adaptation

## Goal

Prevent queued candidates from mixing incompatible adaptive proposal states.

Treat each queue refill as a **proposal epoch**.

## Proposal epoch concept

Freeze for an epoch:

- proposal revision;
- `rwalk` scale;
- `s-rwalk` scale;
- geometry/covariance where relevant;
- current live-set snapshot used by workers.

Conceptually:

```text
freeze proposal state
        |
submit replacement jobs
        |
consume queue serially
        |
queue empty / invalidated
        |
apply tuning or proposal update
        |
start new queue epoch
```

## `rwalk` and `s-rwalk`

Do not modify tuning parameters while replacement candidates generated with the previous tuning state remain queued.

Aggregate tuning information returned from workers.

Apply tuning when the queue epoch finishes.

## `adaptive_morph`

Attach:

```python
proposal_revision
```

to each candidate.

If Morph is refitted:

```python
queue.clear()
```

Initially invalidate all queued candidates from the previous Morph revision.

This is intentionally conservative and easier to validate.

Optimization of cross-revision candidate reuse can be considered later.

## Phase 4 acceptance criteria

- no candidate generated under one proposal revision is silently used under another;
- scale adaptation occurs only at clear epoch boundaries;
- queue invalidation is tested;
- serial and parallel statistical results remain consistent.

---

# Phase 5 — Resource Scheduling and Efficiency

## Goal

Improve CPU utilisation once the basic queue is statistically correct.

## Public API

Prefer one configuration object:

```python
parallel = ParallelSettings(
    n_workers=16,
    queue_size=16,
)
```

Then:

```python
sampler = NISMOSampler(
    ...,
    parallel=parallel,
)
```

Keep serial defaults:

```python
n_workers = 1
queue_size = 1
```

## Queue size

Start with:

```python
queue_size = n_workers
```

Later benchmark:

```python
queue_size = 2 * n_workers
```

for workloads with variable likelihood runtimes.

Avoid very large queues because the threshold advances while candidates wait, increasing stale candidate waste.

## `en-rwalk`

Do not create a process pool inside worker processes.

Each worker runs one complete `en-rwalk` replacement.

Keep the existing vectorized evaluation of half-ensemble proposals inside each worker.

This gives:

```text
process-level parallelism:
    multiple replacement candidates

within each process:
    vectorized en-rwalk proposals
```

## Optional future backends

Only after multiprocessing is stable, consider:

- user-supplied executors;
- MPI;
- cluster executors.

Do not include these in the initial implementation.

---

# Phase 6 — Limits, Diagnostics, and Queue Accounting

## Goal

Make parallel work auditable.

Add queue-specific diagnostics.

Recommended counters:

```text
queue_jobs_submitted
queue_jobs_completed
queue_candidates_consumed
queue_candidates_stale
queue_candidates_invalidated
queue_refills

prefetch_likelihood_calls
used_prefetch_likelihood_calls
wasted_prefetch_likelihood_calls
```

Useful derived quantities:

```text
queue_efficiency =
    queue_candidates_consumed / queue_jobs_completed
```

and:

```text
compute_efficiency =
    used_prefetch_likelihood_calls /
    prefetch_likelihood_calls
```

## Likelihood-call limits

Parallel workers can otherwise overshoot:

```python
max_likelihood_calls
```

because several jobs may be in flight simultaneously.

Implement a reservation system.

Before submitting a replacement job, reserve the maximum allowed call budget for that job or otherwise guarantee the hard global limit cannot be exceeded.

Do not silently exceed configured resource limits.

## Wall-time limits

When the deadline is reached:

- stop submitting new work;
- cancel pending jobs where possible;
- safely collect or discard in-flight results;
- preserve valid partial sampler state;
- terminate cleanly.

---

# Phase 7 — Validation and Benchmarks

Parallelisation must be validated statistically, not only by wall time.

## Statistical tests

Compare serial and parallel runs across many seeds for:

- simple Gaussian;
- correlated Gaussian;
- eggbox;
- Gaussian shells;
- multimodal distributions;
- likelihood plateaus/ties.

Compare:

- mean `logZ`;
- variance of `logZ`;
- reported `logZ` uncertainty;
- posterior moments;
- insertion-rank diagnostics if available;
- termination behaviour;
- constraint violations.

## Performance matrix

Benchmark:

```text
n_workers:
1, 2, 4, 8, 16, 32

queue_size:
1, 2, 4, 8, 16, 32
```

for:

```text
rwalk
s-rwalk
en-rwalk
```

Record:

- wall time;
- total likelihood calls;
- calls per second;
- queue efficiency;
- stale candidate fraction;
- wasted calls;
- evidence error;
- speedup relative to serial.

## Acceptance rule

A parallel implementation should only be considered successful when it provides meaningful wall-time improvement for expensive likelihoods **without detectable statistical bias**.

---

# Implementation Order

The implementing agent should proceed in this order:

## Milestone A

Complete:

- Phase 1
- Phase 2

Then stop and run the full test suite.

Do not continue until queue semantics are validated serially.

## Milestone B

Complete:

- Phase 3

Implement multiprocessing first for `rwalk`.

Validate.

Then add:

- `s-rwalk`
- `en-rwalk`

Validate each independently.

## Milestone C

Complete:

- Phase 4
- Phase 5

Only after basic multiprocessing has shown correct statistical behaviour.

## Milestone D

Complete:

- Phase 6
- Phase 7

Add production diagnostics, limits, and benchmarking.

---

# Explicit Non-Goals

Do not implement these as part of this task:

- simultaneous replacement of multiple live points;
- batched dead-point quadrature;
- new shrinkage mathematics;
- nested multiprocessing;
- GPU parallelism;
- distributed multi-node execution;
- MPI;
- asynchronous completion-order candidate selection;
- major changes to the existing evidence formulas.

Any algorithm that removes multiple live points per nested-sampling iteration should be treated as a **separate experimental sampler**, because it changes the nested-sampling statistics.

---

# Definition of Done

The work is complete when:

1. NISMO can construct several replacement live points ahead of time using multiple workers.
2. The main sampler still replaces exactly one worst live point per nested-sampling iteration.
3. Every queued candidate is checked against the current constraint before use.
4. `n_workers=1, queue_size=1` preserves serial behaviour.
5. `rwalk`, `s-rwalk`, and `en-rwalk` support replacement-level parallelism.
6. Multiprocessing workers are persistent.
7. RNG streams are reproducible and independent.
8. Proposal adaptation is handled at clear queue-epoch boundaries.
9. Queue waste and resource use are reported.
10. Statistical tests show no detectable bias relative to serial execution.
11. Benchmarks demonstrate useful wall-time speedup for sufficiently expensive likelihoods.
12. All repository tests pass after each implementation milestone.
