# PP test

## Setup

From the repository root, create the uv environment for this LVK analysis:

```bash
uv sync --group lvk
```

If you are also modifying NISMO itself, use the development environment plus
the LVK group:

```bash
uv sync --extra dev --group lvk
```

```csv
seed, lnZ_dynesty, lnZ_dynesty_err, lnZ_mcmc, lnZ_mcmc_err, lnZ_morph_dynesty, lnZ_morph_dynesty_err, lnZ_morph_mcmc, lnZ_morph_mcmc_err
...
```

## Dynesty / MorphZ / NISMO comparison

`nismo_computation.py` reads one completed `dynesty_result.json` and runs a
fresh NISMO calculation. NISMO fits its proposal from every stored Dynesty
posterior row; it does not reuse Dynesty's sampler state or checkpoint files.
The existing MorphZ results remain the comparator by default. Add
`--recompute-morphz` only when a fresh MorphZ calculation is required.
When no explicit `--max-iterations` is supplied, the runner uses
`max(10000, 25 * n_live)` so larger live sets are not truncated by NISMO's
fixed library default.

```bash
python nismo_computation.py 48
```

The command can run from another directory when both paths are explicit:

```bash
python /path/to/nismo/analysis/LIGO/fast_pp/nismo_computation.py 48 \
  --result-path /scratch/results/seed_48/dynesty_result.json \
  --output-dir /scratch/results/seed_48
```

It writes `nismo_dynesty_comparison.json` beside the result. This includes the
Dynesty, MorphZ, and NISMO log evidences, NISMO termination state and call
counts, Morph metadata, and a reconstructed-prior/likelihood audit. The audit
must pass before NISMO starts; use `--audit-tolerance` only to accommodate a
verified, small cross-version numerical difference.

Some legacy Bilby result files store `posterior.log_likelihood` as a likelihood
ratio while their `log_evidence` and fresh likelihood evaluations use the full
normalization. The audit records that constant as `log_likelihood_offset` and
checks the residual variation; NISMO always uses the full likelihood.

For an array run on OzSTAR, submit `nismo.slurm`. The script loads
`gcc/13.3.0` and `python/3.12.3`, defaults `MORPHZ_VENV` to
`/fred/oz200/avajpeyi/projects/MORPH/nismo/.venv`, and runs array indices
`0-99` to match `injections.csv`.

By default it reads the existing Dynesty-2000 result for each seed from
`analysis/LIGO/fast_pp/outdir/seed_<index>/dynesty_result.json` and writes the
NISMO comparison JSON back into that same seed directory. Override
`RESULT_ROOT` at submit time when the OzSTAR results live elsewhere.

## Low-live-point training-posterior check

To test whether NISMO remains accurate when its Morph proposal is trained on a
smaller Dynesty posterior, keep the low-live-point Dynesty result isolated from
the production result:

```bash
SLURM_CPUS_PER_TASK=4 uv run --extra morph --extra ligo \
  python analysis/LIGO/fast_pp/pp_analysis.py \
  --index 48 --sampler dynesty --nlive 500 \
  --output-dir analysis/LIGO/fast_pp/outdir/seed_48/dynesty_nlive500 \
  --label dynesty_nlive500 --no-corner
```

If an interrupted run has its `dynesty_nlive500_resume.pickle` checkpoint,
repeat the same command with `--resume` to continue it.

Then run NISMO with the production live count, using only that new posterior as
its proposal-training input:

```bash
uv run --extra morph --extra ligo \
  python analysis/LIGO/fast_pp/nismo_computation.py 48 \
  --result-path analysis/LIGO/fast_pp/outdir/seed_48/dynesty_nlive500/dynesty_nlive500_result.json \
  --output-dir analysis/LIGO/fast_pp/outdir/seed_48/nismo_from_dynesty_nlive500 \
  --n-live 2000 --progress
```

The relevant comparison is against the stored Dynesty-2000 evidence. Agreement
is possible only if the Dynesty-500 posterior includes all material modes and
gives Morph enough support in every evidence-relevant region. Repeat the paired
experiment with independent sampler random seeds before treating one agreement
as calibration evidence.
