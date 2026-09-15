#!/usr/bin/env python3
"""Run NISMO using an arbitrary Bilby result as the Morph training posterior.

``nismo_computation.py`` only knows the fixed ``--dynesty-nlive`` path layout.
This driver reuses its building blocks but takes an explicit ``--result-path``,
so a cheap "rough" posterior (e.g. a few-minute rslice run) can be used as the
importance distribution.  It also exposes ``--kde-bw``: a low-ESS training
posterior makes Silverman's rule under-smooth, and an under-dispersed proposal
is the dangerous direction for importance weighting.

    python nismo_from_result.py 0 \
        --result-path outdir/seed_0/cheap_nlive1000_rslice/cheap_nlive1000_rslice_result.json \
        --output-dir  outdir/seed_0/nismo_from_cheap_rslice \
        --kde-bw 1.4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from nismo import MorphProposal, NISMOSampler, ParallelSettings

import nismo_computation as nc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("lvk_seed", type=int)
    p.add_argument("--result-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    # Default 2000 to match the already-characterised production NISMO runs
    # (bias -0.126, sd 0.229 vs Dynesty-4000), so the only thing varying here
    # is the training posterior.
    p.add_argument("--n-live", type=int, default=2000)
    p.add_argument("--dlogz", type=float, default=nc.NISMO_DLOGZ)
    p.add_argument("--morph-type", default=nc.NISMO_MORPH_TYPE)
    p.add_argument(
        "--kde-bw",
        default="silverman",
        help="'silverman' or a float multiplier/bandwidth passed to MorphProposal.fit",
    )
    p.add_argument("--nismo-seed", type=int, default=nc.NISMO_DEFAULT_SEED)
    p.add_argument("--n-workers", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from bilby.core.result import read_in_result
    from pp_setup import load_simulation

    result_path = args.result_path.resolve()
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        kde_bw: str | float = float(args.kde_bw)
    except ValueError:
        kde_bw = args.kde_bw

    training = read_in_result(filename=str(result_path))
    result_root = Path(__file__).resolve().parent / "outdir"
    likelihood, priors, _, _, _ = load_simulation(
        args.lvk_seed, output_root=result_root, plot_data=False
    )
    names = nc.posterior_parameter_names(training)
    samples = nc.training_samples(training, names)
    model = nc.build_model(
        likelihood=likelihood,
        priors=priors,
        names=names,
        fixed_values=nc.fixed_parameter_values(priors),
    )

    audit = nc.audit_posterior_contract(
        model=model,
        result=training,
        samples=samples,
        n_points=nc.POSTERIOR_AUDIT_POINTS,
    )
    print("Posterior reconstruction audit:", json.dumps(audit, indent=2))
    worst = max(
        audit["max_abs_log_prior_difference"],
        audit["max_abs_log_likelihood_residual"],
    )
    if worst > nc.POSTERIOR_AUDIT_TOLERANCE:
        raise RuntimeError(
            f"reconstructed model disagrees with the training result: {worst:.3e}"
        )

    print(
        f"Training on {len(samples):,} samples from {result_path.name}; "
        f"kde_bw={kde_bw!r}, n_live={args.n_live}"
    )
    proposal = MorphProposal.fit(
        samples,
        param_names=names,
        morph_type=args.morph_type,
        kde_bw=kde_bw,
    )
    sampler = NISMOSampler(
        model=model,
        importance_morph=proposal,
        proposal_scheme=nc.NISMO_PROPOSAL_SCHEME,
        n_live=args.n_live,
        rng=args.nismo_seed,
        parallel=ParallelSettings(n_workers=args.n_workers, queue_size=args.n_workers),
    )
    start = time.perf_counter()
    res = sampler.run(
        dlogz=args.dlogz,
        progress=True,
        max_iterations=nc.default_max_iterations(args.n_live),
    )
    runtime = time.perf_counter() - start

    payload = {
        "training_result": str(result_path),
        "n_training_samples": int(len(samples)),
        "kde_bw": kde_bw,
        "morph_type": args.morph_type,
        "n_live": args.n_live,
        "nismo_seed": args.nismo_seed,
        "posterior_audit": audit,
        "training_lnz": float(training.log_evidence),
        "training_lnz_err": float(training.log_evidence_err),
        "nismo": {
            "lnz": float(res.logz),
            "lnz_err": float(res.logzerr),
            "success": bool(res.success),
            "termination_reason": str(res.termination_reason),
            "n_likelihood_calls": int(res.n_likelihood_calls),
            "n_iterations": int(res.niter),
            "n_proposals": int(res.n_proposals),
            "runtime_seconds": runtime,
            "warnings": list(res.warnings),
        },
    }
    target = output_dir / "nismo_from_result.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["nismo"], indent=2))
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
