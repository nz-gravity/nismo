"""Benchmark replacement-level queue parallelism on an analytic Gaussian.

Example
-------
python benchmarks/parallel_replacement_queue.py \
    --workers 1 2 4 8 --queue-sizes 1 2 4 8 --delay 0.002

The artificial delay is per likelihood row and represents an expensive model.
CSV is written to stdout so callers can redirect it to a durable artifact.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from nismo import (
    EnsembleRWalkSettings,
    NISMOSampler,
    ParallelSettings,
    RWalkSettings,
    SRWalkSettings,
)


class StandardNormalProposal:
    """Pickleable one-dimensional normalized proposal."""

    ndim = 1

    def sample(
        self,
        n: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        return rng.normal(size=(n, 1))

    def log_prob(
        self,
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return -0.5 * theta[:, 0] ** 2 - 0.5 * np.log(2.0 * np.pi)


@dataclass(frozen=True, slots=True)
class DelayedGaussianModel:
    """Normalized prior with an unnormalized shifted Gaussian likelihood."""

    delay: float
    shift: float = 1.0
    likelihood_scale: float = 1.0
    ndim: int = 1
    parameter_names: tuple[str, ...] = ("x",)

    def log_likelihood(
        self,
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if self.delay:
            time.sleep(self.delay * len(theta))
        return -0.5 * ((theta[:, 0] - self.shift) / self.likelihood_scale) ** 2

    def log_prior(
        self,
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return StandardNormalProposal().log_prob(theta)

    @property
    def logz(self) -> float:
        variance = 1.0 + self.likelihood_scale**2
        return float(
            np.log(self.likelihood_scale)
            - 0.5 * np.log(variance)
            - 0.5 * self.shift**2 / variance
        )


def _settings(scheme: str) -> dict[str, Any]:
    if scheme == "rwalk":
        return {"rwalk_settings": RWalkSettings(walks=24)}
    if scheme == "s-rwalk":
        return {"srwalk_settings": SRWalkSettings(n_steps=24)}
    if scheme == "en-rwalk":
        return {
            "ensemble_rwalk_settings": EnsembleRWalkSettings(
                n_walkers=8,
                n_sweeps=3,
            )
        }
    raise ValueError(f"unsupported benchmark scheme: {scheme!r}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schemes",
        nargs="+",
        choices=("rwalk", "s-rwalk", "en-rwalk"),
        default=("rwalk", "s-rwalk", "en-rwalk"),
    )
    parser.add_argument(
        "--workers",
        nargs="+",
        type=int,
        default=(1, 2, 4, 8, 16, 32),
    )
    parser.add_argument(
        "--queue-sizes",
        nargs="+",
        type=int,
        default=(1, 2, 4, 8, 16, 32),
    )
    parser.add_argument("--delay", type=float, default=0.001)
    parser.add_argument("--n-live", type=int, default=80)
    parser.add_argument("--dlogz", type=float, default=0.1)
    parser.add_argument("--max-iterations", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260808)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.delay < 0.0:
        raise SystemExit("--delay must be non-negative")
    model = DelayedGaussianModel(delay=args.delay)
    fieldnames = (
        "scheme",
        "n_workers",
        "queue_size",
        "wall_seconds",
        "speedup",
        "likelihood_calls",
        "calls_per_second",
        "queue_efficiency",
        "stale_fraction",
        "wasted_likelihood_calls",
        "logz",
        "logz_error",
        "reported_logzerr",
        "termination_reason",
    )
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    for scheme in args.schemes:
        baseline_seconds: float | None = None
        for n_workers in args.workers:
            for queue_size in args.queue_sizes:
                sampler = NISMOSampler(
                    model=model,
                    importance_morph=StandardNormalProposal(),
                    proposal_scheme=scheme,
                    n_live=args.n_live,
                    rng=args.seed,
                    parallel=ParallelSettings(
                        n_workers=n_workers,
                        queue_size=queue_size,
                    ),
                    **_settings(scheme),
                )
                started = time.perf_counter()
                result = sampler.run(
                    dlogz=args.dlogz,
                    max_iterations=args.max_iterations,
                    max_proposals_per_replacement=100_000,
                )
                wall_seconds = time.perf_counter() - started
                if n_workers == 1 and queue_size == 1:
                    baseline_seconds = wall_seconds
                speedup = (
                    float("nan")
                    if baseline_seconds is None
                    else baseline_seconds / wall_seconds
                )
                diagnostics = result.queue_diagnostics
                stale_denominator = (
                    diagnostics.queue_candidates_consumed
                    + diagnostics.queue_candidates_stale
                )
                stale_fraction = (
                    diagnostics.queue_candidates_stale / stale_denominator
                    if stale_denominator
                    else 0.0
                )
                writer.writerow(
                    {
                        "scheme": scheme,
                        "n_workers": n_workers,
                        "queue_size": queue_size,
                        "wall_seconds": wall_seconds,
                        "speedup": speedup,
                        "likelihood_calls": result.n_likelihood_calls,
                        "calls_per_second": (result.n_likelihood_calls / wall_seconds),
                        "queue_efficiency": diagnostics.queue_efficiency,
                        "stale_fraction": stale_fraction,
                        "wasted_likelihood_calls": (
                            diagnostics.wasted_prefetch_likelihood_calls
                        ),
                        "logz": result.logz,
                        "logz_error": result.logz - model.logz,
                        "reported_logzerr": result.logzerr,
                        "termination_reason": result.termination_reason,
                    }
                )
                sys.stdout.flush()


if __name__ == "__main__":
    main()
