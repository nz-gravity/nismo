"""Profile optimized ``s-rwalk`` components across dimensions and live counts.

Examples
--------
python benchmarks/profile_srwalk.py --dimensions 10 50 100 200 \
    --n-live 100 500 --covariance-intervals 1 10 25

Rows are written as CSV to stdout. ``--likelihood-delays`` accepts seconds per
evaluated point, so ``0`` represents a cheap model and a positive value exposes
the likelihood-dominated regime. Component timings require
``SRWalkSettings(profile=True)`` and include queue dispatch/serialization wait.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nismo import NISMOSampler, SRWalkSettings


@dataclass(frozen=True, slots=True)
class StandardNormal:
    ndim: int

    def sample(
        self,
        n: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        return rng.normal(size=(n, self.ndim))

    def log_prob(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        return -0.5 * np.sum(theta**2, axis=1) - 0.5 * self.ndim * np.log(2.0 * np.pi)


@dataclass(frozen=True, slots=True)
class ConstantLikelihoodModel:
    ndim: int
    delay: float

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(f"x{index}" for index in range(self.ndim))

    def log_likelihood(
        self,
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if self.delay:
            time.sleep(self.delay * len(theta))
        return np.zeros(len(theta))

    def log_prior(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        return StandardNormal(self.ndim).log_prob(theta)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", nargs="+", type=int, default=(10, 50, 100, 200))
    parser.add_argument("--n-live", nargs="+", type=int, default=(100, 500))
    parser.add_argument(
        "--covariance-intervals",
        nargs="+",
        type=int,
        default=(1, 10, 25),
    )
    parser.add_argument("--likelihood-delays", nargs="+", type=float, default=(0.0,))
    parser.add_argument("--n-steps", type=int, default=25)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--queue-size", type=int)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser


def main() -> None:
    args = _parser().parse_args()
    fieldnames = (
        "dimension",
        "n_live",
        "likelihood_delay",
        "covariance_update_interval",
        "n_workers",
        "queue_size",
        "wall_seconds",
        "speedup_vs_interval_1",
        "iterations",
        "likelihood_calls",
        "geometry_update_seconds",
        "geometry_rebuild_seconds",
        "factorization_seconds",
        "proposal_linear_algebra_seconds",
        "prior_seconds",
        "likelihood_seconds",
        "q0_seconds",
        "queue_setup_seconds",
        "worker_dispatch_seconds",
        "factor_refreshes",
        "stale_candidate_fraction",
        "mean_squared_displacement",
        "mean_mh_acceptance",
        "mean_constraint_pass",
    )
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    for dimension in args.dimensions:
        for n_live in args.n_live:
            for delay in args.likelihood_delays:
                baseline_seconds: float | None = None
                for interval in args.covariance_intervals:
                    model = ConstantLikelihoodModel(dimension, delay)
                    proposal = StandardNormal(dimension)
                    sampler = NISMOSampler(
                        model=model,
                        importance_morph=proposal,
                        proposal_scheme="s-rwalk",
                        srwalk_settings=SRWalkSettings(
                            n_steps=args.n_steps,
                            covariance_update_interval=interval,
                            profile=True,
                        ),
                        n_live=n_live,
                        rng=args.seed,
                        n_workers=args.workers,
                        queue_size=args.queue_size,
                        tie_policy="randomized_plateau",
                    )
                    started = time.perf_counter()
                    result = sampler.run(
                        dlogz=0.5,
                        max_iterations=args.max_iterations,
                        max_proposals_per_replacement=args.n_steps,
                    )
                    wall_seconds = time.perf_counter() - started
                    if interval == 1:
                        baseline_seconds = wall_seconds
                    diagnostics = result.srwalk_diagnostics
                    if diagnostics is None:  # pragma: no cover - profile is enabled
                        raise RuntimeError("s-rwalk profiling produced no diagnostics")
                    writer.writerow(
                        {
                            "dimension": dimension,
                            "n_live": n_live,
                            "likelihood_delay": delay,
                            "covariance_update_interval": interval,
                            "n_workers": args.workers,
                            "queue_size": sampler.queue_size,
                            "wall_seconds": wall_seconds,
                            "speedup_vs_interval_1": (
                                float("nan")
                                if baseline_seconds is None
                                else baseline_seconds / wall_seconds
                            ),
                            "iterations": result.niter,
                            "likelihood_calls": result.n_likelihood_calls,
                            "geometry_update_seconds": (
                                diagnostics.geometry_update_seconds
                            ),
                            "geometry_rebuild_seconds": (
                                diagnostics.geometry_rebuild_seconds
                            ),
                            "factorization_seconds": diagnostics.factorization_seconds,
                            "proposal_linear_algebra_seconds": (
                                diagnostics.proposal_linear_algebra_seconds
                            ),
                            "prior_seconds": diagnostics.prior_seconds,
                            "likelihood_seconds": diagnostics.likelihood_seconds,
                            "q0_seconds": diagnostics.q0_seconds,
                            "queue_setup_seconds": diagnostics.queue_setup_seconds,
                            "worker_dispatch_seconds": (
                                diagnostics.worker_dispatch_seconds
                            ),
                            "factor_refreshes": diagnostics.factor_refreshes,
                            "stale_candidate_fraction": (
                                diagnostics.stale_candidate_fraction
                            ),
                            "mean_squared_displacement": (
                                diagnostics.mean_squared_displacement
                            ),
                            "mean_mh_acceptance": float(
                                np.mean(result.history.mh_acceptance_fraction)
                            ),
                            "mean_constraint_pass": float(
                                np.mean(result.history.constraint_pass_fraction)
                            ),
                        }
                    )
                    sys.stdout.flush()


if __name__ == "__main__":
    main()
