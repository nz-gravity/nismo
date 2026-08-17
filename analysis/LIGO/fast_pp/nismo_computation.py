#!/usr/bin/env python3
"""Compare Dynesty, MorphZ, and NISMO evidences for one PP injection.

The Dynesty result supplies posterior draws used to fit both MorphZ and the
fixed NISMO importance Morph.  NISMO then evaluates a freshly reconstructed
Bilby likelihood; it never reads a Dynesty checkpoint or resume pickle.

Run from any directory, for example from the NISMO repository root::

    uv run python analysis/LIGO/fast_pp/nismo_computation.py 48

"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from nismo import MorphProposal, MORWalkSettings, NISMOSampler, ParallelSettings

NISMO_PROPOSAL_SCHEME = "mor-rwalk"
NISMO_N_LIVE = 500
NISMO_DLOGZ = 0.1
NISMO_MORPH_TYPE = "2_group"
NISMO_MOR_RWALK_N_PROPOSALS = 20_000
NISMO_DEFAULT_SEED = 20260811
POSTERIOR_AUDIT_POINTS = 32
POSTERIOR_AUDIT_TOLERANCE = 1.0e-6


parallel_settings = ParallelSettings(n_workers=4, queue_size=4)


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


def default_max_iterations(n_live: int) -> int:
    """Return a live-count-scaled hard ceiling for production analyses."""
    return max(10_000, 25 * n_live)


def dynesty_result_path(result_root: Path, lvk_seed: int, dynesty_nlive: int) -> Path:
    """Return the isolated result file for one Dynesty live-point setting."""
    seed_dir = result_root / f"seed_{lvk_seed}"
    if dynesty_nlive == 2_000:
        return seed_dir / "dynesty_result.json"
    label = f"dynesty_nlive{dynesty_nlive}"
    return seed_dir / label / f"{label}_result.json"


def load_existing_morphz(result_path: Path, index: int) -> dict[str, Any] | None:
    """Load the existing Dynesty-trained MorphZ estimate when available."""
    comparison_path = result_path.parent / f"seed_{index}_lnz_comparison.csv"
    if not comparison_path.is_file():
        return None
    with comparison_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("method") == "dynesty_morphz":
                return {
                    "lnz_mean": float(row["lnz"]),
                    "lnz_err": float(row["lnz_err"]),
                    "source": str(comparison_path.resolve()),
                }
    return None


def fixed_parameter_values(priors: Any) -> dict[str, float]:
    """Extract values for Bilby priors fixed by the PP analysis."""
    return {str(name): float(priors[name].peak) for name in priors.fixed_keys}


@dataclass
class BilbyLIGOModel:
    """Pickleable batch-model adapter for NISMO replacement workers."""

    parameter_names: tuple[str, ...]
    likelihood: Any
    sampled_priors: Any
    fixed_values: dict[str, float]

    @property
    def ndim(self) -> int:
        return len(self.parameter_names)

    def _sampled_parameters(self, theta: np.ndarray) -> dict[str, float]:
        return dict(zip(self.parameter_names, theta, strict=True))

    def _parameters(self, theta: np.ndarray) -> dict[str, float]:
        parameters = self._sampled_parameters(theta)
        parameters.update(self.fixed_values)
        return parameters

    def _log_sampled_prior(self, sampled_values: dict[str, float]) -> float:
        """Return a valid log density, treating numerical failures as no support.

        Bilby's compound priors can return NaN at invalid proposal points rather
        than ``-np.inf``.  Those points have zero posterior integrand and must
        be rejected before they reach the likelihood or NISMO's strict model
        output validation.
        """
        value = float(self.sampled_priors.ln_prob(sampled_values))
        return value if np.isfinite(value) else -np.inf

    def log_prior(self, theta: np.ndarray) -> np.ndarray:
        return np.asarray(
            [self._log_sampled_prior(self._sampled_parameters(row)) for row in theta],
            dtype=float,
        )

    def log_likelihood(self, theta: np.ndarray) -> np.ndarray:
        values = np.empty(len(theta), dtype=float)
        for index, row in enumerate(theta):
            sampled_values = self._sampled_parameters(row)
            # A KDE proposal has tails beyond the physical prior.  Do not pass
            # those points to LAL: their posterior integrand is exactly zero.
            if not np.isfinite(self._log_sampled_prior(sampled_values)):
                values[index] = -np.inf
                continue
            values[index] = self.likelihood.log_likelihood(self._parameters(row))
        return values


def build_model(
    *,
    likelihood: Any,
    priors: Any,
    names: Sequence[str],
    fixed_values: dict[str, float],
) -> Any:
    """Wrap the original Bilby parameterization in NISMO's batch contract."""
    from bilby.core.prior import PriorDict

    parameter_names = tuple(names)
    # Fixed parameters define the lower-dimensional model but must not
    # contribute delta-function densities to its Lebesgue prior measure.
    sampled_priors = PriorDict(
        dictionary={name: priors[name] for name in parameter_names}
    )

    return BilbyLIGOModel(
        parameter_names=parameter_names,
        likelihood=likelihood,
        sampled_priors=sampled_priors,
        fixed_values=fixed_values,
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
    proposal_scheme: str,
    mor_rwalk_n_proposals: int,
    max_iterations: int | None,
    progress: bool = True,
) -> tuple[Any, Any]:
    """Fit one fixed Morph proposal and run a fresh NISMO calculation."""

    proposal = MorphProposal.fit(
        samples,
        param_names=names,
        morph_type=morph_type,
        kde_bw="silverman",
    )
    sampler_kwargs: dict[str, Any] = {
        "model": model,
        "importance_morph": proposal,
        "proposal_scheme": proposal_scheme,
        "n_live": n_live,
        "rng": seed,
        "parallel": parallel_settings,
    }
    if proposal_scheme == "mor-rwalk":
        sampler_kwargs["mor_rwalk_settings"] = MORWalkSettings(
            n_proposals=mor_rwalk_n_proposals,
        )
    sampler = NISMOSampler(**sampler_kwargs)
    run_kwargs: dict[str, Any] = {"dlogz": dlogz, "progress": progress}
    if max_iterations is not None:
        run_kwargs["max_iterations"] = max_iterations
    return sampler.run(**run_kwargs), proposal


def result_payload(
    *,
    dynesty_result: Any,
    nismo_result: Any,
    proposal: Any,
    result_path: Path,
    names: Sequence[str],
    audit: dict[str, float | int],
    morphz: dict[str, Any] | None,
    proposal_scheme: str,
    dynesty_nlive: int,
    seed: int,
    n_live: int,
    dlogz: float,
    max_iterations: int | None,
    runtime_seconds: float,
) -> dict[str, Any]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dynesty_result": str(result_path.resolve()),
        "dynesty_nlive": dynesty_nlive,
        "parameter_names": list(names),
        "n_training_samples": int(proposal.metadata.n_training),
        "n_live": n_live,
        "dlogz": dlogz,
        "max_iterations": max_iterations,
        "seed": seed,
        "proposal_scheme": proposal_scheme,
        "parallel": {"n_workers": 4, "queue_size": 4},
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
            "runtime_seconds": runtime_seconds,
            "warnings": list(nismo_result.warnings),
        },
    }


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lvk_seed", type=int, help="row in injections.csv")
    parser.add_argument(
        "--dynesty-nlive",
        type=int,
        default=2_000,
        help="live points in the Dynesty result used to train NISMO",
    )
    parser.add_argument(
        "--proposal-scheme",
        choices=("mor-rwalk", "s-rwalk", "en-rwalk"),
        default=NISMO_PROPOSAL_SCHEME,
        help="NISMO replacement scheme to use for this replica",
    )
    parser.add_argument(
        "--nismo-seed",
        type=int,
        default=NISMO_DEFAULT_SEED,
        help="random seed for this NISMO replica",
    )
    parser.add_argument(
        "--mor-rwalk-n-proposals",
        type=int,
        default=NISMO_MOR_RWALK_N_PROPOSALS,
        help="initial Morph pool size when proposal_scheme is mor-rwalk",
    )
    return parser.parse_args(arguments)


def main() -> None:
    args = parse_args()
    from bilby.core.result import read_in_result
    from pp_setup import load_simulation

    result_root = Path(__file__).resolve().parent / "outdir"
    if args.dynesty_nlive <= 0:
        raise ValueError("--dynesty-nlive must be a positive integer")
    result_path = dynesty_result_path(
        result_root, args.lvk_seed, args.dynesty_nlive
    ).resolve()
    scheme_name = args.proposal_scheme.replace("-", "_")
    training_name = (
        "dynesty"
        if args.dynesty_nlive == 2_000
        else f"dynesty_nlive{args.dynesty_nlive}"
    )
    output_name = f"nismo_{scheme_name}_seed_{args.nismo_seed}"
    if args.dynesty_nlive != 2_000:
        output_name = f"nismo_from_{training_name}_{scheme_name}_seed_{args.nismo_seed}"
    output_dir = (result_root / f"seed_{args.lvk_seed}" / output_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not result_path.is_file():
        raise FileNotFoundError(result_path)

    dynesty_result = read_in_result(filename=str(result_path))
    likelihood, priors, _, _, _ = load_simulation(
        args.lvk_seed,
        output_root=result_root,
        plot_data=False,
    )
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
        n_points=POSTERIOR_AUDIT_POINTS,
    )
    print("Posterior reconstruction audit:", json.dumps(audit, indent=2))
    max_audit_difference = max(
        audit["max_abs_log_prior_difference"],
        audit["max_abs_log_likelihood_residual"],
    )
    if max_audit_difference > POSTERIOR_AUDIT_TOLERANCE:
        raise RuntimeError(
            "reconstructed Bilby model disagrees with the Dynesty result: "
            f"maximum discrepancy {max_audit_difference:.3e} exceeds "
            f"the fixed tolerance {POSTERIOR_AUDIT_TOLERANCE:.3e}"
        )
    morphz = load_existing_morphz(result_path, args.lvk_seed)
    max_iterations = default_max_iterations(NISMO_N_LIVE)
    nismo_start = time.perf_counter()
    nismo_result, proposal = run_nismo(
        model=model,
        samples=samples,
        names=names,
        n_live=NISMO_N_LIVE,
        dlogz=NISMO_DLOGZ,
        seed=args.nismo_seed,
        morph_type=NISMO_MORPH_TYPE,
        proposal_scheme=args.proposal_scheme,
        mor_rwalk_n_proposals=args.mor_rwalk_n_proposals,
        max_iterations=max_iterations,
        progress=True,
    )
    nismo_runtime_seconds = time.perf_counter() - nismo_start
    payload = result_payload(
        dynesty_result=dynesty_result,
        nismo_result=nismo_result,
        proposal=proposal,
        result_path=result_path,
        names=names,
        audit=audit,
        morphz=morphz,
        proposal_scheme=args.proposal_scheme,
        dynesty_nlive=args.dynesty_nlive,
        seed=args.nismo_seed,
        n_live=NISMO_N_LIVE,
        dlogz=NISMO_DLOGZ,
        max_iterations=max_iterations,
        runtime_seconds=nismo_runtime_seconds,
    )
    target = output_dir / "nismo_dynesty_comparison.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["dynesty"], indent=2))
    print(json.dumps(payload["morphz"], indent=2))
    print(json.dumps(payload["nismo"], indent=2))
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
