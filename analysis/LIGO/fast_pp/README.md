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

Corner plots with injection truth are saved automatically under `plots/` in PNG
and PDF formats: all eight coordinates before and after NISMO, plus a side-by-side
mass/spin comparison. These use the original posterior weights, unsmoothed
histograms and shared axes. Incomplete results are labelled explicitly. Marginal
5/50/95% quantiles and truth CDFs are saved in `weighted_marginals.csv`; they are
summaries of the finite weighted samples, not a claim of calibrated coverage.
To regenerate plots from an existing run without sampling:

```bash
uv run --extra lvk python analysis/LIGO/fast_pp/rslice_morph_demo.py 48 \
  --plots-only --output-dir docs/benchmarks/lvk-rslice-morph-20260915/bw2
```

For a reused rough run whose original location has moved, provide the archived
training directory via `--rough-from`.

### Stage recovery checks

After **each** stage the script prints and saves (`stage_checks.json`):

- The sampler's stopping-target status and weighted posterior ESS.
- The 5/50/95% noise-weighted **network overlap** with the injected signal.
- The 5/50/95% noise-weighted waveform difference, expressed as residual SNR.
- The injected log likelihood minus the best stored log likelihood. A positive
  gap flags a known higher-likelihood point missed by the run, but does not
  measure the probability mass in that region.

Using the same detector responses, frequency masks and PSDs as the likelihood,
we concatenate `h_I(f) * sqrt(4 / (T_I * PSD_I(f)))` over the detectors. Overlap
is the real inner product divided by the two norms. Residual SNR is the norm of
`h - h_injected` in those coordinates, retaining amplitude differences that a
normalized overlap can hide. Time, phase and amplitude are **not** optimized:
the demo fixes its extrinsics. This follows
[Bilby's noise-weighted inner-product convention](https://bilby-dev.github.io/bilby/api/bilby.gw.utils.html).

The combined **heuristic screen** passes only if the stage's stopping target
was met, ESS is at least `--check-min-ess` (default 100), and the 5th percentile
of overlap is at least `--overlap-min` (default 0.99). These are configurable
screening choices, not calibrated convergence criteria. Overlap assesses signal
recovery; it cannot establish parameter recovery, mode coverage, evidence
accuracy, or posterior convergence. Residual SNR is descriptive, with no hard
cut applied. The screen is reporting-only: it does not change sampling,
termination criteria, or existing process exit codes.

The distribution summaries use `--check-draws` (default 128) deterministic
systematic draws from the weighted empirical posterior; duplicate waveforms
are evaluated once. These are approximate posterior summaries, not 128 new
independent samples. ESS is always computed from the original weights. Check
time and diagnostic likelihood calls are recorded separately from sampling;
pipeline time includes check time. Injection truth is used only for diagnostics.

To check saved results without rerunning inference:

```bash
uv run --extra lvk python analysis/LIGO/fast_pp/rslice_morph_demo.py 48 \
  --checks-only --output-dir docs/benchmarks/lvk-rslice-morph-20260915/bw2
```

### Start from an existing Bilby posterior

The same demo accepts `--bilby-result PATH` in place of a fresh rslice run:

```bash
uv run --extra lvk python analysis/LIGO/fast_pp/rslice_morph_demo.py 0 \
  --bilby-result analysis/LIGO/fast_pp/outdir/ozstar_seed0_20260915/raw/cheap_nlive1000_rslice_result.json \
  --nlive 100 --bandwidth-scale 2 --max-seconds 180 \
  --output-dir /tmp/seed0_from_existing_posterior
```

It compares the source priors/fixed parameters and sampled coordinate order,
then checks fresh prior and likelihood values at 32 posterior rows. Only a
zero offset or the known noise-evidence offset is accepted for stored likelihoods.
The original JSON is preserved; the source hash, original sampler/live count,
time/calls and audit are recorded in `rough.json`. All equal-weight Bilby
posterior rows train the KDE. Their weight ESS equals their count and **does
not** diagnose autocorrelation or mixing. A saved result JSON alone does not
certify its stopping reason, so the imported stopping status is `UNKNOWN`.

`--nlive` sets the new NISMO live count independently of the imported source's
live count. Add `--training-only` to import/audit/check without running NISMO;
that output can subsequently be supplied via `--rough-from`. Imported source
labels are retained in the printed checks and corner plots. Original training
cost and current import cost are reported separately; an expensive historical
training run is not a new end-to-end speedup.
