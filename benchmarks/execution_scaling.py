"""Reproducible execution comparison with fixed model/reference across backends.

Run from the repository with PYTHONPATH=src. JSONL contains environment, run and
aggregate records; timing is never asserted in CI. A --factory module:function
may return (model, frozen_proposal, exact_logz_or_None) for a real shell/PTA run.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from nismo import MorphProposal, NISMOSampler, ParallelSettings, SRWalkSettings


@dataclass
class NormalReference:
    ndim: int
    scale: float = 1.0

    def sample(self, n, rng):
        return rng.normal(scale=self.scale, size=(n, self.ndim))

    def log_prob(self, theta):
        return -0.5 * np.sum((theta / self.scale) ** 2, axis=1) - self.ndim * np.log(
            self.scale * np.sqrt(2 * np.pi)
        )


@dataclass
class AnalyticModel:
    ndim: int
    target: str
    delay: float = 0.0

    @property
    def parameter_names(self):
        return tuple(f"x{i}" for i in range(self.ndim))

    @property
    def precision(self):
        # Dense, rotated geometry with eigenvalues spanning six decades.
        rotation, _ = np.linalg.qr(
            np.random.default_rng(710).normal(size=(self.ndim, self.ndim))
        )
        return (rotation * np.geomspace(1e-3, 1e3, self.ndim)) @ rotation.T

    @property
    def prior_scale(self):
        return 5.0 if self.target == "mixture" else 1.0

    def log_prior(self, theta):
        return NormalReference(self.ndim, self.prior_scale).log_prob(theta)

    def log_likelihood(self, theta):
        if self.delay:
            # Deterministic per-row, state-dependent scalar workload.
            until = time.process_time() + self.delay * float(
                np.sum(1 + (theta[:, 0] > 0))
            )
            while time.process_time() < until:
                pass
        if self.target == "correlated":
            return -0.5 * np.einsum("bi,ij,bj->b", theta, self._precision, theta)
        if self.target == "mixture":
            center = np.zeros(self.ndim)
            center[0] = 5
            return logsumexp(
                np.array(
                    [
                        -0.5 * np.sum((theta - center) ** 2, axis=1),
                        -0.5 * np.sum((theta + center) ** 2, axis=1),
                    ]
                ),
                axis=0,
            ) - np.log(2)
        return -0.05 * np.sum(theta**2, axis=1)

    @property
    def logz(self):
        if self.target == "correlated":
            return -0.5 * np.linalg.slogdet(np.eye(self.ndim) + self._precision)[1]
        if self.target == "mixture":
            return -0.5 * self.ndim * np.log(26) - 25 / 52
        return -0.5 * self.ndim * np.log(1.1)

    def training_samples(self, n, rng):
        if self.target == "correlated":
            covariance = np.linalg.inv(np.eye(self.ndim) + self._precision)
            return rng.normal(size=(n, self.ndim)) @ np.linalg.cholesky(covariance).T
        if self.target == "mixture":
            theta = rng.normal(scale=np.sqrt(25 / 26), size=(n, self.ndim))
            theta[:, 0] += rng.choice([-1, 1], size=n) * 125 / 26
            return theta
        return rng.normal(scale=1 / np.sqrt(1.1), size=(n, self.ndim))


def cpu_seconds():
    return sum(
        x.ru_utime + x.ru_stime
        for x in (
            resource.getrusage(resource.RUSAGE_SELF),
            resource.getrusage(resource.RUSAGE_CHILDREN),
        )
    )


def available_cpus():
    cpus = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count()
    )
    quota = Path("/sys/fs/cgroup/cpu.max")
    if quota.exists():
        amount, period = quota.read_text().split()
        if amount != "max":
            cpus = min(cpus, max(1, int(amount) // int(period)))
    return cpus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["compatibility", "vectorized", "ordered", "rolling"],
        default=["compatibility", "vectorized", "ordered", "rolling"],
    )
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--n-live", nargs="+", type=int, default=[100, 400])
    parser.add_argument("--dimension", type=int, default=60)
    parser.add_argument("--chains-per-task", type=int, default=8)
    parser.add_argument("--queue-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=75)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--seed", type=int, default=400)
    parser.add_argument("--training-seed", type=int, default=71)
    parser.add_argument("--training-size", type=int, default=1000)
    parser.add_argument(
        "--target", choices=["gaussian", "correlated", "mixture"], default="gaussian"
    )
    parser.add_argument("--morph", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--delay", type=float, default=0)
    parser.add_argument("--dlogz", type=float, default=0.1)
    parser.add_argument("--max-iterations", type=int, default=100000)
    parser.add_argument("--max-wall-time", type=float)
    parser.add_argument(
        "--factory",
        help="Importable factory(args) -> (model, fixed proposal, logz or None)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(w > available_cpus() for w in args.workers):
        parser.error(f"workers exceed the {available_cpus()} allocated CPUs")
    with threadpool_limits(1):
        fit_start = time.monotonic()
        if args.factory:
            module, function = args.factory.split(":")
            model, proposal, truth = getattr(importlib.import_module(module), function)(
                args
            )
        else:
            model = AnalyticModel(args.dimension, args.target, args.delay)
            model._precision = model.precision
            proposal = NormalReference(args.dimension, model.prior_scale)
            if args.morph:
                training = model.training_samples(
                    args.training_size, np.random.default_rng(args.training_seed)
                )
                proposal = MorphProposal.fit(
                    training, param_names=model.parameter_names, groups=[]
                )
            truth = float(model.logz)
        fit_seconds = time.monotonic() - fit_start
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as stream:

            def emit(record):
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()

            emit(
                dict(
                    kind="environment",
                    python=platform.python_version(),
                    numpy=np.__version__,
                    scipy=scipy.__version__,
                    allocated_cpus=available_cpus(),
                    settings={
                        k: str(v) if isinstance(v, Path) else v
                        for k, v in vars(args).items()
                    },
                    preparation_seconds=fit_seconds,
                    exact_logz=truth,
                    morph_metadata=asdict(proposal.metadata)
                    if hasattr(proposal, "metadata")
                    else None,
                )
            )
            records = []
            for nlive in args.n_live:
                for mode in args.backends:
                    for workers in [1] if mode == "vectorized" else args.workers:
                        parallel = ParallelSettings(
                            backend=mode
                            if mode in ("compatibility", "vectorized")
                            else "process",
                            scheduler=mode
                            if mode in ("ordered", "rolling")
                            else "epoch",
                            n_workers=workers,
                            queue_size=1
                            if mode == "compatibility" and workers == 1
                            else args.queue_size,
                            chains_per_task=1
                            if mode == "compatibility"
                            else args.chains_per_task,
                            adaptation_interval=args.queue_size,
                            worker_threads=1,
                            initialization="process"
                            if args.delay
                            and workers > 1
                            and mode in ("ordered", "rolling")
                            else "auto",
                        )
                        for repeat in range(args.repeat):
                            sampler = NISMOSampler(
                                model=model,
                                importance_morph=proposal,
                                proposal_scheme="s-rwalk",
                                n_live=nlive,
                                rng=args.seed + repeat,
                                parallel=parallel,
                                srwalk_settings=SRWalkSettings(
                                    n_steps=args.steps,
                                    dynamic_steps=False,
                                    profile=args.profile,
                                ),
                            )
                            start, cpu_start = time.monotonic(), cpu_seconds()
                            result = sampler.run(
                                dlogz=args.dlogz,
                                max_iterations=args.max_iterations,
                                max_wall_time=args.max_wall_time,
                            )
                            wall, cpu = (
                                time.monotonic() - start,
                                cpu_seconds() - cpu_start,
                            )
                            history = result.history
                            cut = max(1, result.niter // 5)
                            steady = (
                                (result.niter - cut)
                                / (
                                    history.elapsed_seconds[-1]
                                    - history.elapsed_seconds[cut - 1]
                                )
                                if result.niter > cut
                                else None
                            )
                            values, weights = (
                                result.all_points,
                                result.posterior_weights,
                            )
                            mean = weights @ values
                            variance = weights @ ((values - mean) ** 2)
                            record = dict(
                                kind="run",
                                backend=mode,
                                workers=workers,
                                nlive=nlive,
                                repeat=repeat,
                                wall_seconds=wall,
                                cpu_seconds=cpu,
                                cpu_hours=cpu / 3600,
                                niter=result.niter,
                                ncall=result.n_likelihood_calls,
                                logz=result.logz,
                                bias=None if truth is None else result.logz - truth,
                                logzerr=result.logzerr,
                                success=result.success,
                                termination=result.termination_reason,
                                steady_replacements_per_second=steady,
                                volume_progress_per_second=None
                                if steady is None
                                else steady / nlive,
                                posterior_mean=mean.tolist(),
                                posterior_variance=variance.tolist(),
                                positive_first_axis_weight=float(
                                    weights @ (values[:, 0] > 0)
                                ),
                                queue=asdict(result.queue_diagnostics),
                                phases=dict(result.execution_diagnostics.phase_seconds),
                                worker_execution_seconds=result.execution_diagnostics.worker_execution_seconds,
                            )
                            records.append(record)
                            emit(record)
                            print(
                                f"{mode} workers={workers} N={nlive} repeat={repeat}: "
                                f"{wall:.3f}s logZ={result.logz:.4f} "
                                f"success={result.success}",
                                flush=True,
                            )
            for nlive in args.n_live:
                for mode in args.backends:
                    for workers in args.workers:
                        group = [
                            r
                            for r in records
                            if (r["nlive"], r["backend"], r["workers"])
                            == (nlive, mode, workers)
                        ]
                        if not group:
                            continue
                        emit(
                            dict(
                                kind="aggregate",
                                backend=mode,
                                workers=workers,
                                nlive=nlive,
                                repeats=len(group),
                                completed=sum(r["success"] for r in group),
                                mean_wall_seconds=float(
                                    np.mean([r["wall_seconds"] for r in group])
                                ),
                                mean_logz=float(np.mean([r["logz"] for r in group])),
                                logz_scatter=float(np.std([r["logz"] for r in group])),
                                one_sigma_coverage=None
                                if truth is None
                                else float(
                                    np.mean(
                                        [abs(r["bias"]) <= r["logzerr"] for r in group]
                                    )
                                ),
                            )
                        )


if __name__ == "__main__":
    main()
