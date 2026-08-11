#!/usr/bin/env python3
"""Compare Dynesty, MorphZ, and NISMO evidences for one PP injection.

The Dynesty result supplies posterior draws used to fit both MorphZ and the
fixed NISMO importance Morph.  NISMO then evaluates a freshly reconstructed
Bilby likelihood; it never reads a Dynesty checkpoint or resume pickle.

Run from any directory, for example from the NISMO repository root::

    uv run python analysis/LIGO/fast_pp/nismo_computation.py 48

``--result-path`` and ``--output-dir`` are useful when the result files are
staged elsewhere on a cluster.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def posterior_parameter_names(result: Any) -> tuple[str, ...]:
    """Return the coordinates actually sampled by Bilby, in result order."""
    names = tuple(str(name) for name in result.search_parameter_keys)
    if not names:
        raise ValueError("Bilby result has no search_parameter_keys")
    missing = [name for name in names if name not in result.posterior]
    if missing:
        raise ValueError(f"posterior is missing sampled parameters: {missing}")
    return names


def training_samples(result: Any, names: Sequence[str]) -> np.ndarray:
    """Return all finite posterior samples; no thinning is applied."""
    samples = result.posterior[list(names)].to_numpy(dtype=float)
    if samples.ndim != 2 or len(samples) < 2:
        raise ValueError("NISMO requires at least two posterior samples")
    if not np.all(np.isfinite(samples)):
        raise ValueError("posterior training samples contain NaN or infinity")
    return samples


def fixed_parameter_values(priors: Any) -> dict[str, float]:
    """Extract values for Bilby priors fixed by the PP analysis."""
    return {str(name): float(priors[name].peak) for name in priors.fixed_keys}


def build_model(
    *,
    likelihood: Any,
    priors: Any,
    names: Sequence[str],
    fixed_values: dict[str, float],
) -> Any:
    """Wrap the original Bilby parameterization in NISMO's batch contract."""
    from bilby.core.prior import PriorDict

    from nismo import CallableModel

    parameter_names = tuple(names)
    # Fixed parameters define the lower-dimensional model but must not
    # contribute delta-function densities to its Lebesgue prior measure.
    sampled_priors = PriorDict(
        dictionary={name: priors[name] for name in parameter_names}
    )

    def parameters_for(theta: np.ndarray) -> dict[str, float]:
        values = dict(zip(parameter_names, theta, strict=True))
        values.update(fixed_values)
        return values

    def log_prior(theta: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                sampled_priors.ln_prob(dict(zip(parameter_names, row, strict=True)))
                for row in theta
            ],
            dtype=float,
        )

    def log_likelihood(theta: np.ndarray) -> np.ndarray:
        values = np.empty(len(theta), dtype=float)
        for index, row in enumerate(theta):
            sampled_values = dict(zip(parameter_names, row, strict=True))
            # A KDE proposal has tails beyond the physical prior.  Do not pass
            # those points to LAL: their posterior integrand is exactly zero.
            if not np.isfinite(sampled_priors.ln_prob(sampled_values)):
                values[index] = -np.inf
                continue
            params = parameters_for(row)
            # Pass parameters explicitly: this is compatible with the Bilby
            # API used to create the legacy Dynesty results and avoids its
            # deprecated mutable-parameter fallback.
            values[index] = likelihood.log_likelihood(params)
        return values

    return CallableModel(
        ndim=len(parameter_names),
        parameter_names=parameter_names,
        log_likelihood_fn=log_likelihood,
        log_prior_fn=log_prior,
        vectorized=True,
    )


def audit_posterior_contract(
    *,
    model: Any,
    result: Any,
    samples: np.ndarray,
    n_points: int,
) -> dict[str, float | int]:
    """Compare reconstructed densities against stored Bilby values.

    This deliberately samples a deterministic spread of rows: it detects an
    incompatible likelihood/prior setup without pre-paying an evaluation of
    every posterior draw before the NISMO calculation starts.
    """
    if n_points < 1:
        raise ValueError("audit_points must be positive")
    indices = np.unique(
        np.linspace(0, len(samples) - 1, min(n_points, len(samples)), dtype=int)
    )
    evaluated_prior = model.log_prior(samples[indices])
    evaluated_likelihood = model.log_likelihood(samples[indices])
    stored_prior = result.posterior["log_prior"].to_numpy(dtype=float)[indices]
    stored_likelihood = result.posterior["log_likelihood"].to_numpy(dtype=float)[
        indices
    ]
    prior_difference = evaluated_prior - stored_prior
    likelihood_difference = evaluated_likelihood - stored_likelihood
    if not np.all(np.isfinite(prior_difference)) or not np.all(
        np.isfinite(likelihood_difference)
    ):
        raise RuntimeError("reconstructed posterior audit produced non-finite values")
    likelihood_offset = float(np.median(likelihood_difference))
    likelihood_residual = likelihood_difference - likelihood_offset
    return {
        "n_points": len(indices),
        "max_abs_log_prior_difference": float(np.max(np.abs(prior_difference))),
        "log_likelihood_offset": likelihood_offset,
        "max_abs_log_likelihood_residual": float(np.max(np.abs(likelihood_residual))),
    }


def run_nismo(
    *,
    model: Any,
    samples: np.ndarray,
    names: Sequence[str],
    n_live: int,
    dlogz: float,
    seed: int,
    morph_type: str,
) -> tuple[Any, Any]:
    """Fit one fixed Morph proposal and run a fresh NISMO calculation."""
    from nismo import MorphProposal, NISMOSampler

    proposal = MorphProposal.fit(
        samples,
        param_names=names,
        morph_type=morph_type,
        kde_bw="silverman",
    )
    sampler = NISMOSampler(
        model=model,
        importance_morph=proposal,
        proposal_scheme="fixed_morph",
        n_live=n_live,
        rng=seed,
    )
    return sampler.run(dlogz=dlogz, progress=True), proposal


def result_payload(
    *,
    dynesty_result: Any,
    nismo_result: Any,
    proposal: Any,
    result_path: Path,
    names: Sequence[str],
    audit: dict[str, float | int],
    morphz: dict[str, float] | None,
    seed: int,
    n_live: int,
    dlogz: float,
) -> dict[str, Any]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dynesty_result": str(result_path.resolve()),
        "parameter_names": list(names),
        "n_training_samples": int(proposal.metadata.n_training),
        "n_live": n_live,
        "dlogz": dlogz,
        "seed": seed,
        "proposal_scheme": "fixed_morph",
        "morph_metadata": {
            "selected_groups": [
                list(group) for group in proposal.metadata.selected_groups
            ],
            "single_parameters": list(proposal.metadata.single_parameters),
            "morphz_version": proposal.metadata.morphz_version,
        },
        "posterior_audit": audit,
        "dynesty": {
            "lnz": float(dynesty_result.log_evidence),
            "lnz_err": float(dynesty_result.log_evidence_err),
        },
        "morphz": morphz,
        "nismo": {
            "lnz": float(nismo_result.logz),
            "lnz_err": float(nismo_result.logzerr),
            "success": bool(nismo_result.success),
            "termination_reason": str(nismo_result.termination_reason),
            "n_likelihood_calls": int(nismo_result.n_likelihood_calls),
            "n_iterations": int(nismo_result.niter),
            "n_proposals": int(nismo_result.n_proposals),
            "warnings": list(nismo_result.warnings),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=int, help="row in injections.csv")
    parser.add_argument("--result-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--n-live", type=int, default=500)
    parser.add_argument("--dlogz", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--morph-type", default="2_group")
    parser.add_argument("--audit-points", type=int, default=32)
    parser.add_argument(
        "--audit-tolerance",
        type=float,
        default=1.0e-6,
        help="maximum allowed absolute reconstructed log-density discrepancy",
    )
    parser.add_argument(
        "--skip-morphz",
        action="store_true",
        help="do not recompute the existing MorphZ post-processing estimate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from bilby.core.result import read_in_result
    from pp_setup import load_simulation

    default_result = (
        Path(__file__).resolve().parent
        / "outdir"
        / f"seed_{args.index}"
        / "dynesty_result.json"
    )
    result_path = (args.result_path or default_result).resolve()
    output_dir = (args.output_dir or result_path.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not result_path.is_file():
        raise FileNotFoundError(result_path)

    dynesty_result = read_in_result(filename=str(result_path))
    likelihood, priors, _, _, _ = load_simulation(args.index, output_root=output_dir)
    names = posterior_parameter_names(dynesty_result)
    samples = training_samples(dynesty_result, names)
    model = build_model(
        likelihood=likelihood,
        priors=priors,
        names=names,
        fixed_values=fixed_parameter_values(priors),
    )
    audit = audit_posterior_contract(
        model=model,
        result=dynesty_result,
        samples=samples,
        n_points=args.audit_points,
    )
    print("Posterior reconstruction audit:", json.dumps(audit, indent=2))
    max_audit_difference = max(
        audit["max_abs_log_prior_difference"],
        audit["max_abs_log_likelihood_residual"],
    )
    if max_audit_difference > args.audit_tolerance:
        raise RuntimeError(
            "reconstructed Bilby model disagrees with the Dynesty result: "
            f"maximum discrepancy {max_audit_difference:.3e} exceeds "
            f"--audit-tolerance={args.audit_tolerance:.3e}"
        )
    morphz = None
    if not args.skip_morphz:
        # Keep MorphZ as an independently reported post-processing comparator.
        # Its legacy routine also recomputes the Bilby likelihood at the
        # posterior points it uses.
        from morphz_computation import get_morphz_evidence

        morphz = get_morphz_evidence(
            dynesty_result,
            priors,
            likelihood,
            label=f"seed_{args.index}_dynesty",
            output_dir=output_dir,
        )
    nismo_result, proposal = run_nismo(
        model=model,
        samples=samples,
        names=names,
        n_live=args.n_live,
        dlogz=args.dlogz,
        seed=args.seed,
        morph_type=args.morph_type,
    )
    payload = result_payload(
        dynesty_result=dynesty_result,
        nismo_result=nismo_result,
        proposal=proposal,
        result_path=result_path,
        names=names,
        audit=audit,
        morphz=morphz,
        seed=args.seed,
        n_live=args.n_live,
        dlogz=args.dlogz,
    )
    target = output_dir / "nismo_dynesty_comparison.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["dynesty"], indent=2))
    print(json.dumps(payload["morphz"], indent=2))
    print(json.dumps(payload["nismo"], indent=2))
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
