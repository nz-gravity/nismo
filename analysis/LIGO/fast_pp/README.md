# PP test



```csv
seed, lnZ_dynesty, lnZ_dynesty_err, lnZ_mcmc, lnZ_mcmc_err, lnZ_morph_dynesty, lnZ_morph_dynesty_err, lnZ_morph_mcmc, lnZ_morph_mcmc_err
...
```

## Dynesty / MorphZ / NISMO comparison

`nismo_computation.py` reads one completed `dynesty_result.json`, refits the
existing MorphZ estimate, and runs a fresh fixed-importance NISMO calculation.
NISMO fits its proposal from every stored Dynesty posterior row; it does not
reuse Dynesty's sampler state or checkpoint files.

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

For an array run, submit `nismo.slurm`. Set `MORPHZ_VENV` if the existing
Bilby/MorphZ environment is not `/morphZ_casestudy_CBC_pe/.venv`.
