# PP test

## Setup

From the repository root, create the uv environment for this LVK analysis:

```bash
uv sync --extra lvk
```

If you are also modifying NISMO itself, use the development environment plus
the same LVK extra:

```bash
uv sync --extra dev --extra lvk
```

```csv
seed, lnZ_dynesty, lnZ_dynesty_err, lnZ_mcmc, lnZ_mcmc_err, lnZ_morph_dynesty, lnZ_morph_dynesty_err, lnZ_morph_mcmc, lnZ_morph_mcmc_err
...
```

## Dynesty / MorphZ / NISMO comparison

`nismo_computation.py` reads
`outdir/seed_<LVK seed>/dynesty_result.json` and runs a fresh fixed-settings
NISMO calculation. NISMO fits its proposal from every stored Dynesty posterior
row; it does not reuse Dynesty's sampler state or checkpoint files. The only
run-time choices are the LVK seed and, optionally, the NISMO replica seed.

```bash
python nismo_computation.py 48
```

To make an independent NISMO replica, use:

```bash
python nismo_computation.py 48 --nismo-seed 47
```

It writes to `outdir/seed_<LVK seed>/nismo_swalk_seed_<NISMO seed>/`. This
includes the Dynesty, MorphZ, and NISMO log evidences, NISMO termination state
and call counts, Morph metadata, and a reconstructed-prior/likelihood audit.
The audit must pass before NISMO starts.

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
NISMO comparison JSON to a separate directory for each NISMO seed:
`outdir/seed_<index>/nismo_<scheme>_seed_<NISMO seed>/`.

## Dynesty-100 followed by NISMO

The dedicated `dynesty_nlive100.slurm` and
`nismo_from_dynesty_nlive100.slurm` arrays preserve the existing Dynesty-2000
campaign. They write, respectively,
`outdir/seed_<index>/dynesty_nlive100/dynesty_nlive100_result.json` and
`outdir/seed_<index>/nismo_from_dynesty_nlive100_<scheme>_seed_<NISMO seed>/`.
The NISMO run uses the same live-point count as its Dynesty training run, so
this pair runs both samplers with 100 live points.

From the repository `analysis/` directory on OzSTAR, submit NISMO only after
the complete Dynesty array has succeeded:

```bash
dynesty_job=$(sbatch --parsable LIGO/fast_pp/dynesty_nlive100.slurm)
sbatch --dependency=afterok:${dynesty_job} \
  LIGO/fast_pp/nismo_from_dynesty_nlive100.slurm
```

For an independent NISMO replacement replica, set `NISMO_SEED` on the second
submission. The Dynesty-100 array resumes a matching incomplete checkpoint but
refuses to overwrite a completed result.

## Low-live-point training-posterior check

To test whether NISMO remains accurate when its Morph proposal is trained on a
smaller Dynesty posterior, keep the low-live-point Dynesty result isolated from
the production result:

```bash
SLURM_CPUS_PER_TASK=4 uv run --extra lvk \
  python analysis/LIGO/fast_pp/pp_analysis.py \
  --index 48 --sampler dynesty --nlive 500 \
  --output-dir analysis/LIGO/fast_pp/outdir/seed_48/dynesty_nlive500 \
  --label dynesty_nlive500 --no-corner
```

If an interrupted run has its `dynesty_nlive500_resume.pickle` checkpoint,
repeat the same command with `--resume` to continue it.

Run NISMO on that isolated posterior with `--dynesty-nlive 500`; NISMO will
also use 500 live points and will write to a setting-specific output directory.

## Rough rslice → wider Morph pilot

One script runs both stages on the existing simulated H1/L1 injection, with
all eight intrinsic parameters sampled and extrinsics fixed:

```bash
uv run --extra lvk python analysis/LIGO/fast_pp/rslice_morph_demo.py 48 \
  --output-dir analysis/LIGO/fast_pp/outdir/rslice_morph_seed48
```

Defaults: 100 live points in **both** stages; Dynesty `rslice` with three slices
and a loose `dlogz=3`; automatic pairwise Morph grouping; kernel widths twice
Silverman's rule; NISMO `fixed_morph` with `dlogz=0.1`. The posterior is used
only for fitting the normalized proposal. NISMO draws a fresh live set and
uses the original full likelihood and sampled-coordinate prior.

`--bandwidth-scale 2` doubles kernel standard deviations (quadruples kernel
covariances), using the appropriate Silverman factor for each group dimension.
It does not stretch posterior coordinates. KDE draws outside the prior have
zero target density. The mass constraints are checked to be redundant over
the sampled mass box before either stage runs.

To compare bandwidths using exactly the same rough posterior:

```bash
uv run --extra lvk python analysis/LIGO/fast_pp/rslice_morph_demo.py 48 \
  --rough-from analysis/LIGO/fast_pp/outdir/rslice_morph_seed48 \
  --bandwidth-scale 1 \
  --output-dir analysis/LIGO/fast_pp/outdir/rslice_morph_seed48_bw1
```

Use a new output directory for each run. Reuse checks the noise/PSD/prior/waveform
fingerprint, parameter order, injection index, and live count. Other controls
are `--rough-seed`, `--nismo-seed`, `--nlive`, `--rough-dlogz`, `--slices`,
`--rough-maxcall`, `--dlogz`, `--max-calls`, and `--max-seconds` (NISMO only).
The rough call limit is approximate, as Dynesty finishes its current update.

Outputs include `rough.npz` (weighted samples and resampled training draws),
`rough.json`, `proposal.json`, `settings.json`, `summary.json`, and the NISMO
weighted samples/history/diagnostics under `nismo/`. The summary records both
stages' evidence estimates, ESS, termination, call counts and elapsed times.
Pipeline cost includes the original rough run even when reused; invocation
time records the current execution. Call counts count sampler likelihood
requests, including NISMO rows rejected by the prior before waveform evaluation.

A hard limit produces saved **partial** NISMO results and exit code 2; scientific
stopping gives exit code 0. A call-limited rough posterior may still train the
proposal, with its unmet target recorded. Broadening can cover nearby tails but
cannot establish that a rough run found every mode. Compare against independent
high-live-point nested sampling before claiming accuracy or speed gains.
