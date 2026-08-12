# Changelog

All notable changes to this project are documented here.

## Unreleased

- Added sampler-level `output_path` persistence for weighted samples, complete
  run history, strict JSON diagnostics, and standard diagnostic plots, plus
  reusable `NISMOResult.save()` and `save_run_outputs()` APIs.
- Optimized `s-rwalk` with rolling remove/add live-set covariance statistics,
  Cholesky-first factors, configurable factor refreshes, shared queue-snapshot
  geometry, and batched Gaussian proposal linear algebra.
- Added prior-first evaluation so outside-prior and zero-likelihood proposals
  skip unnecessary likelihood or fixed-Morph density calls while preserving
  hard likelihood budgets and complete-chain semantics.
- Added opt-in `s-rwalk` component profiling, a profiling benchmark, and
  covariance-drift, stationarity, accounting, and integration regressions.

## 0.1.2 - 2026-08-11

- Updated release metadata and README version references for the PyPI release.
## 0.1.1 - 2026-08-11
## 0.1.0 - 2026-08-11

- Prepared the first PyPI release with synchronized package metadata, an OIDC
  trusted-publishing workflow, and distribution validation.
- Replaced development reports and generated artifacts in `docs/` with focused
  installation, quick-start, configuration, result, and public API guides.
- Added all user-facing diagnostics and plotting helpers to the top-level
  `nismo` namespace.

- `dlogz` now measures an estimated remaining increment in log evidence. Users
  requiring the previous live-evidence-fraction behavior should use the
  explicit `remaining_fraction` stopping criterion.
- Split the frozen importance Morph from the active constrained-sampling Morph.
- Added fixed and periodically refitted live-set Morph proposal schemes.
- Renamed stored density and pseudo-likelihood arrays to explicit `log_q0` and
  `log_psi0` forms and added proposal-update diagnostics.
- Added constrained fixed-`q0` Metropolis `"rwalk"` and split
  differential-evolution `"en-rwalk"` replacement schemes, immutable settings,
  resource accounting, and separate MCMC diagnostics.
- Adapted standard `"rwalk"` to Dynesty 3.1.0 ellipsoidal-ball proposals,
  `ndim + 20` default walks, acceptance-target scale tuning, cached
  single-ellipsoid bounds, and Skilling citation reporting.
- Added a separate statistically specified Gaussian-covariance `"s-rwalk"`
  kernel with fixed-per-replacement survivor geometry, exact fixed-`q0` MH
  correction, and acceptance-target scale adaptation.
- Extended `"en-rwalk"` with fixed weighted mixtures of the existing
  differential-evolution move, a Jacobian-corrected Goodman--Weare stretch
  move, and a frozen-covariance Gaussian move. Added `EnsembleMoveWeights`,
  general Hastings-ratio acceptance, and immutable per-move result history.
  The default weights are 60% DE, 25% stretch, and 15% Gaussian; explicit pure
  configurations, including the former DE-only sequence, remain available.

## 0.1.0.dev3 - 2026-07-25

- Added automatic MorphZ total-correlation and greedy group selection through
  `MorphProposal.fit(..., morph_type="{k}_group")`.

## 0.1.0.dev2 - 2026-07-25

- Added reproducible equal-weight posterior resampling on `NISMOResult`.

## 0.1.0.dev1 - 2026-07-25

- Added optional tqdm progress reporting with standard nested-sampling state.
- Persisted per-iteration information, theoretical log-evidence error, and
  remaining-evidence fraction.

## 0.1.0.dev0 - 2026-07-25

- Created the Phase 1 package and quality-control skeleton.
- Added the experimental Phase 2 fixed-Morph nested-importance estimator.
