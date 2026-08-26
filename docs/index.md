# NISMO documentation

This directory contains only user-facing documentation for the NISMO software
API. Design reports, implementation plans, benchmark outputs, generated plots,
and LaTeX build artifacts are intentionally kept out of the API documentation.

## Start here

1. [Quick start](quickstart.md) — install NISMO, define a model, fit a Morph,
   run the sampler, and consume weighted samples.
2. [Configuration](configuration.md) — choose a proposal, stopping policy,
   resource limits, and replacement queue.
3. [Results and diagnostics](results.md) — interpret evidence, posterior
   weights, history, termination, plots, and run-health summaries.
4. [Public API](api.md) — signatures and contracts for every supported public
   object.

## Installation choices

```bash
# Recommended user installation
python -m pip install "nismo[all]"

# Core API with a custom Proposal implementation
python -m pip install nismo

# Select only the optional features you need
python -m pip install "nismo[morph,progress]"
```

The optional extras are:

| Extra | Provides |
|---|---|
| `morph` | `MorphProposal.fit(...)` through MorphZ 0.4.1 or newer |
| `plot` | Matplotlib plotting helpers |
| `progress` | tqdm output for `run(progress=True)` |
| `all` | All three user extras |

NISMO supports Python 3.10 or newer and is tested on Python 3.10–3.12.

## Core numerical contract

NISMO uses natural logarithms and batch arrays with shape `(n, ndim)`.
For parameter point `theta`, the model supplies a likelihood `L(theta)` and a
normalized prior `pi(theta)`. A fixed normalized importance density `q0(theta)`
and `0 < beta <= 1` define

```text
g_beta(theta) = q0(theta)**beta / C_beta
log_psi_beta = log_likelihood + log_prior - beta * log_q0 + log_C_beta
```

`beta=1` is the standard sampler and has `C_beta=1` exactly. For `beta<1`,
NISMO estimates `C_beta` by direct Monte Carlo and initializes `mor-rwalk` or
`s-rwalk` from a finite importance-resampled candidate batch. Their random-walk
MH correction targets the same diffused `q0**beta` density.

The importance density must have support everywhere that `L * pi` is nonzero.
NISMO cannot diagnose a mode that is absent from both the importance fit and
the live population. Missing support can therefore bias the evidence and
posterior without producing a numerical error.

The sampler uses deterministic prior-volume shrinkage. For `beta<1`,
`result.logzerr` combines the standard theoretical nested-sampling
approximation `sqrt(H / n_live)` with the estimated Monte Carlo uncertainty in
`log_C_beta`. It remains an incomplete error budget: it does not cover an
imperfect Morph fit, missing modes, finite-pool approximation, correlated
finite-length walks, or the heuristic adaptive proposal. Calibrate the full
procedure with repeated seeds and known benchmarks appropriate to the target
problem.
