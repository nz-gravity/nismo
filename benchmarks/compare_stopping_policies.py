"""Non-CI repeated-seed comparison of NISMO stopping policies on eggbox."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import logsumexp

from nismo import (
    CallableModel,
    NISMOSampler,
    StoppingCriterionConfig,
    StoppingPolicy,
)
from nismo.diagnostics import summarize

BOX_WIDTH = 10.0 * np.pi


class EggboxProposal:
    """Normalized uniform proposal on the standard two-dimensional eggbox."""

    ndim = 2

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(0.0, BOX_WIDTH, size=(n, self.ndim))

    def log_prob(self, theta: np.ndarray) -> np.ndarray:
        inside = np.all((theta >= 0.0) & (theta <= BOX_WIDTH), axis=1)
        return np.where(inside, -2.0 * np.log(BOX_WIDTH), -np.inf)


def eggbox_log_likelihood(theta: np.ndarray) -> np.ndarray:
    """Return the conventional multimodal eggbox log likelihood."""
    modulation = np.cos(theta[:, 0] / 2.0) * np.cos(theta[:, 1] / 2.0)
    return np.asarray((2.0 + modulation) ** 5, dtype=float)


def eggbox_model() -> tuple[CallableModel, EggboxProposal]:
    proposal = EggboxProposal()
    return (
        CallableModel(
            ndim=2,
            parameter_names=("x", "y"),
            log_likelihood_fn=eggbox_log_likelihood,
            log_prior_fn=proposal.log_prob,
        ),
        proposal,
    )


def reference_logz(order: int = 512) -> float:
    """High-order tensor Gauss-Legendre reference under the uniform prior."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    coordinates = 0.5 * (nodes + 1.0) * BOX_WIDTH
    x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
    points = np.column_stack((x.ravel(), y.ravel()))
    log_weights = np.log(np.outer(weights, weights).ravel())
    return float(logsumexp(log_weights + eggbox_log_likelihood(points)) - np.log(4.0))


@dataclass(frozen=True, slots=True)
class PolicyCase:
    name: str
    dlogz: float | None = None
    stopping: StoppingPolicy | None = None


POLICIES = (
    PolicyCase(name="remaining_dlogz_1e-3", dlogz=1.0e-3),
    PolicyCase(name="remaining_dlogz_1e-2", dlogz=1.0e-2),
    PolicyCase(
        name="hybrid",
        stopping=StoppingPolicy(
            criteria=(
                StoppingCriterionConfig("live_logz_error", 5.0e-3),
                StoppingCriterionConfig("remaining_fraction", 5.0e-2),
                StoppingCriterionConfig("logz_stability", 5.0e-3),
            ),
            mode="all",
            consecutive=3,
            min_iterations=10,
            stability_window=10,
        ),
    ),
)


def run_case(
    *,
    case: PolicyCase,
    seed: int,
    n_live: int,
    max_iterations: int,
    truth: float,
) -> dict[str, Any]:
    model, proposal = eggbox_model()
    sampler = NISMOSampler(
        model=model,
        importance_morph=proposal,
        n_live=n_live,
        rng=seed,
        proposal_batch_size=512,
    )
    start = time.perf_counter()
    result = sampler.run(
        dlogz=case.dlogz,
        stopping=case.stopping,
        max_iterations=max_iterations,
        max_proposals_per_replacement=1_000_000,
    )
    wall_seconds = time.perf_counter() - start
    diagnostics = summarize(result)
    return {
        "seed": seed,
        "policy": case.name,
        "success": result.success,
        "termination_reason": result.termination_reason,
        "niter": result.niter,
        "n_likelihood_calls": result.n_likelihood_calls,
        "n_proposals": result.n_proposals,
        "wall_seconds": wall_seconds,
        "logz": result.logz,
        "logz_error_from_truth": result.logz - truth,
        "final_remaining_fraction": diagnostics.final_remaining_fraction,
        "final_remaining_dlogz": diagnostics.final_remaining_dlogz,
        "final_live_ess": diagnostics.final_live_ess,
        "final_live_logz_error": diagnostics.final_live_logz_error,
        "final_logz_stability": diagnostics.final_logz_stability,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> None:
    baseline = [row for row in rows if row["policy"] == "remaining_dlogz_1e-3"]
    baseline_runtime = float(np.median([row["wall_seconds"] for row in baseline]))
    print("\nSummary")
    for case in POLICIES:
        selected = [row for row in rows if row["policy"] == case.name]
        errors = np.asarray([row["logz_error_from_truth"] for row in selected])
        runtimes = np.asarray([row["wall_seconds"] for row in selected])
        print(
            case.name,
            {
                "mean_logz_error": float(np.mean(errors)),
                "median_logz_error": float(np.median(errors)),
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "logz_std": float(np.std([row["logz"] for row in selected])),
                "median_iterations": float(
                    np.median([row["niter"] for row in selected])
                ),
                "median_likelihood_calls": float(
                    np.median([row["n_likelihood_calls"] for row in selected])
                ),
                "median_wall_seconds": float(np.median(runtimes)),
                "failure_rate": float(
                    np.mean([not row["success"] for row in selected])
                ),
                "runtime_speedup_vs_dlogz_1e-3": baseline_runtime
                / float(np.median(runtimes)),
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n-live", type=int, default=200)
    parser.add_argument("--max-iterations", type=int, default=20_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("stopping_policy_runs.csv"),
    )
    args = parser.parse_args()

    truth = reference_logz()
    rows = [
        run_case(
            case=case,
            seed=seed,
            n_live=args.n_live,
            max_iterations=args.max_iterations,
            truth=truth,
        )
        for seed in range(args.seeds)
        for case in POLICIES
    ]
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"reference logZ: {truth:.12g}")
    print(f"wrote {len(rows)} runs to {args.output}")
    summarize_rows(rows)


if __name__ == "__main__":
    main()
