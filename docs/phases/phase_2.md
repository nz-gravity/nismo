# Phase 2 report

## Objective and scope

Implement the smallest statistically coherent post-processing evidence
estimator using one fixed installed MorphZ `GroupKDE` as the normalized
pseudo-prior. Stop before defensive, adaptive, or Phase 3 work.

## Inputs received from the project owner

- Phase 1/2 DOCX implementation plan.
- Authoritative Phase 2 Markdown mathematical and implementation specification.
- Installed editable MorphZ 0.4.1 development package.

No NISMO prototype, experimental notebooks, group JSON, posterior data, or
stored benchmark summaries were present.

## Mathematical decisions and assumptions

The implementation follows

\[
\log\Psi=\log L+\log\pi-\log q,\qquad
Z=\int\Psi q\,d\theta.
\]

It uses new Morph draws for live points, first-valid independent rejection,
deterministic \(X_i=e^{-i/N}\), rectangular dead weights, and individual
final-live corrections. See `docs/mathematical_contract.md`.

## Architecture/files changed

- `proposals/morph.py`: all MorphZ calls and RNG adaptation.
- `constrained.py`: cached evaluation and unbiased constrained rejection.
- `quadrature.py`: pure log-space evidence calculations.
- `sampler.py`: serial state machine and typed termination.
- `results.py`: immutable recomputable results and histories.
- `NISMOResult.resample_equal`: dependency-free equal-weight posterior draws.
- `progress.py`: optional tqdm bar and callback snapshots.
- `diagnostics.py`, `plotting.py`: result-only health checks and figures.
- `benchmarks/`, `tests/`, `examples/`, and documentation pages.

## Commands run

```bash
MPLCONFIGDIR=/tmp/nismo-mpl PYTHONPATH=src python -m pytest \
  tests/unit tests/integration -m "not slow"
MPLCONFIGDIR=/tmp/nismo-mpl PYTHONPATH=src python -m pytest \
  tests/statistical -m "statistical and not slow"
MPLCONFIGDIR=/tmp/nismo-mpl PYTHONPATH=src python -m pytest \
  tests/statistical/test_benchmarks.py::test_grouped_morph_gaussian_shell_regression \
  -m slow -vv
python -m ruff format --check .
python -m ruff check .
python -m mypy src/nismo
python -m pytest --cov=nismo --cov-report=term-missing
python -m build --no-isolation
python -m twine check dist/*
MPLCONFIGDIR=/tmp/nismo-mpl PYTHONPATH=src python \
  examples/phase2_gaussian.py --output-dir docs/plots
```

## Test and benchmark results

| Check | Expected logz | Obtained logz / status |
|---|---:|---:|
| Full test suite | all pass | 52 passed |
| Constant integrand | `0.91629073` | exact to `1e-12` |
| Gaussian example, seed 12 | `-1.72365749` | `-1.72975180 ± 0.01589677` |
| Peak–plateau, seed 2026 | `-0.13353139` | `-0.12867532 ± 0.07629106` |
| Grouped Gaussian shell, seed 72 | `-1.97926019` | `-1.91506791 ± 0.15125830` |
| Gaussian direct importance, seed 44 | `-1.72365749` | `-1.72432916` |
| Gaussian nested cross-check, seed 44 | direct `-1.72432916` | `-1.74942546 ± 0.02662801` |

Ruff format/lint, strict mypy, wheel/sdist build, Twine metadata checks,
clean-wheel installation, outside-repository import, and an installed
Morph-backed sampler smoke run also passed. Branch-aware coverage was 86%
overall; no arbitrary pass threshold is configured. The optional tqdm
terminal/notebook bar and full custom-callback snapshot were also exercised
through integration tests and the installed wheel.

The repeated Gaussian test uses seeds 11, 12, and 13 and declares tolerances
before assertion: absolute mean log-evidence bias below `0.04`, empirical
spread below three times mean reported error, and aggregate posterior second
moment within `0.3` of `0.8`.

## Diagnostic plots

- `docs/plots/gaussian_run.png`
- `docs/plots/gaussian_weight_health.png`
- `docs/plots/gaussian_posterior.png`

## Deviations from the plan

The unavailable owner prototype could not be preserved or compared. Analytic
peak–plateau and Gaussian-shell fixtures were created and clearly identified as
new regressions. MorphZ's inspected batch `logpdf` convention in 0.4.1 is
adapted by reliable row-wise public calls; no Morph density is reimplemented.

## Known failures and limitations

- Proposal support is non-defensive; omitted posterior regions can bias `logz`.
- `logzerr = sqrt(H / n_live)` does not include all fit uncertainty.
- Deterministic shrinkage omits random-volume uncertainty.
- Row-wise MorphZ density evaluation prioritizes correctness over speed.
- MorphZ 0.4.1 accepts integer resampling seeds and temporarily uses/restores
  legacy NumPy global state internally.
- Progress `logzerr` is the current \(\sqrt{H/N_{\rm live}}\) approximation,
  not an independent convergence guarantee.
- Exact owner experiments and group files were not supplied.

## Reproduction instructions

```bash
python -m pip install -e ".[dev]"
MPLCONFIGDIR=/tmp/nismo-mpl python -m pytest
MPLCONFIGDIR=/tmp/nismo-mpl python examples/phase2_gaussian.py \
  --output-dir docs/plots
```

## Questions requiring owner decision

1. Provide the original peak–plateau and Gaussian-shell experiments/group
   files if exact regression equivalence is required.
2. Confirm the preferred MorphZ version constraint and distribution workflow.
3. Independently test support coverage and repeated-run calibration on the
   owner's posterior data before authorizing a defensive proposal.

## Recommendation: proceed, revise, or stop

Stop at Phase 2 and wait for independent owner testing. Do not begin Phase 3.
