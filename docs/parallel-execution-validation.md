# Parallel execution validation — 14 September 2026

The implementation starts from `ad52a65`, the reviewed main revision.
New backends remain opt-in. This record distinguishes correctness checks,
short scheduling measurements, and completed analytic runs. It does not
establish production PTA or high-dimensional shell convergence.

## Correctness checks

- The retained beta=1 serial path was run against a detached `ad52a65`
  checkout with the same 1D shifted Gaussian, seed 311, 30 live points,
  12 initial walk steps and `dlogz=0.12`. Evidence, all dead coordinates,
  tie breakers, volume trajectory, likelihood calls and final RNG state
  matched exactly.
- Fixed-stream scalar/batched kernels agree exactly for beta 1, 0.8 and 0.5,
  normal and asymmetric uniform support, ordinary and extreme proposal scales,
  ties, and zero-move chains. Splitting batches does not change each chain's
  random stream. Invalid outputs and support errors still raise.
- Batch preflight reserves full chains, including potential prior rejections.
  Deadlines do not turn partial walks into endpoints. End-to-end scheduler
  tests cover partial queue tails, wasted/unused calls, beta runs, dynamic
  tuning and hard call limits.
- Ordered/rolling runs replay with state-dependent delays and different
  process counts. Worker and callback exceptions clean up owned processes.
  Explicit worker thread limits leave coordinator thread settings unchanged.
- Truncated-tempered-normal checks cover beta 1, 0.8 and 0.5. Repeated complete
  Gaussian runs cover both new execution paths at beta 1 and 0.8, checking
  evidence, moments and broad uncertainty coverage. These small tests screen
  for regressions; they cannot establish general mixing or coverage.
- The beta regression with 30,000 MC candidates and a same-size pool now
  recovers approximately twice the variance at beta 0.5. Chunked density
  evaluation and selected-density reuse agree with direct evaluation. The
  initializer remains finite-candidate SIR with shared-candidate dependence.

`pytest -m "not slow"`, Ruff checks/formatting, strict mypy, an offline lockfile
consistency check, and source/wheel builds are the local gates. The existing
slow shell regression is outside the fast suite.

## Timing environment and scope

Python 3.12.13, NumPy 2.3.5, SciPy 1.17.0; Linux with eight CPUs of cgroup
quota and nine CPUs in the affinity mask. Numerical libraries and spawned
workers use one thread. Timing experiments were run sequentially, without
concurrent statistical test jobs. Morph fit time is recorded separately and
excluded from each run timer. Each experiment fixes its model, training sample
seed and fitted Morph across all execution modes.

### Short 60D scheduling runs

The target is a standard-normal prior with `logL = -0.05 * sum(theta**2)`,
400 live points and 75 fixed walk steps. These stop at a specified number of
iterations (`success=False`), and their final evidence/error bars are not
accuracy validation. The analytic-reference experiment uses 400 replacements
and two repeats; the real Morph experiment uses 120 replacements and one
repeat, fitting 1,000 samples with independent parameter KDEs (`groups=[]`).

| Execution | Workers | Analytic reference, mean seconds | Morph KDE, seconds |
| --- | ---: | ---: | ---: |
| Compatibility, singleton queue | 1 | 2.125 | 44.699 |
| Compatibility, queue 16 | 4 | 1.125 | 18.365 |
| Vectorized, queue 16 / 8 chains per task | 1 | 0.571 | 15.409 |
| Rolling, queue 16 / 8 chains per task | 1 | 0.626 | 15.237 |
| Rolling, queue 16 / 8 chains per task | 4 | 0.765 | 14.898 |

This supports batching density work for these cases. More processes do not
necessarily improve complete run time: startup, coordinator work and the
number of independent tasks per refill all matter. These are full short-run
timers, not just kernel timings, and are not forecasts for the user's Morph
or PTA likelihood.

### Completed 10D Gaussian runs with a real Morph

100 live points; 20 fixed walk steps; `dlogz=0.1`; 400 training samples;
three sampling seeds. Every run reached its scientific stopping criterion.
The analytic evidence is `-5 * log(1.1) = -0.476551`.

| Execution | Workers | Mean seconds | Mean logZ | Between-run scatter |
| --- | ---: | ---: | ---: | ---: |
| Compatibility, singleton queue | 1 | 4.005 | -0.460781 | 0.018844 |
| Compatibility, queue 8 | 4 | 3.784 | -0.470621 | 0.008663 |
| Vectorized, queue 8 / 4 chains per task | 1 | 1.615 | -0.470621 | 0.008663 |
| Rolling, queue 8 / 4 chains per task | 1 | 1.716 | -0.478895 | 0.007958 |
| Rolling, queue 8 / 4 chains per task | 4 | 3.333 | -0.478895 | 0.007958 |

The matched-queue compatibility and vectorized runs produce the same evidence
estimates. The rolling backend also produces the same estimates at one and
four workers; completion order does not choose endpoints. Three seeds are
insufficient to calibrate uncertainty coverage or small biases.

Raw run records, configuration and preparation timings:

- [Analytic 60D timings](benchmarks/parallel-execution-20260914/scaling-analytic.jsonl)
- [Morph 60D timings](benchmarks/parallel-execution-20260914/scaling-morph.jsonl)
- [Completed Morph 10D runs](benchmarks/parallel-execution-20260914/scaling-morph-complete.jsonl)

Use [the execution guide](parallel-execution.md) and
`benchmarks/execution_scaling.py` to reproduce these cases or supply a real
PTA/shell fixture. The 30D/60D shell live-count grid, representative grouped
Morphs, PTA likelihoods, lineage/insertion-order diagnostics and machines with
16–24 allocated CPUs remain workload-specific release gates. This PR does
not implement the review's later batch-deletion/GPU algorithm.
