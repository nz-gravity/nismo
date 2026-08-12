#!/usr/bin/env python3
"""
Repeated twin-Gaussian-shell benchmark: Dynesty ``rwalk`` versus NISMO.

The benchmark follows the model and NISMO setup in:
    examples/gaussian shell.ipynb
    https://github.com/nz-gravity/nismo/blob/main/examples/gaussian%20shell.ipynb

For each n_live value, the script:

1. Runs one pilot Dynesty ``rwalk`` calculation.
2. Converts that run's weighted samples to equal-weight posterior samples.
3. Fits one fixed MorphZ proposal from the thinned posterior, as in the notebook.
4. Runs N independent Dynesty calculations and N independent NISMO calculations.
5. Saves every run immediately to CSV, writes aggregate statistics, and creates
   a two-panel publication-style figure:
      top:    log(Z) versus n_live
      bottom: likelihood-call count versus n_live

The pilot Dynesty run is also Dynesty repeat 0, so exactly N Dynesty results are
reported for each n_live. All N NISMO repeats at a given n_live use the same
fixed Morph proposal, matching the "one NS run supplies the posterior samples"
design of the reference figure.

Likelihood-call accounting
--------------------------
NISMO requires posterior samples to construct its fixed Morph proposal. The raw
CSV therefore stores three call-count definitions:

* ncall_direct:
    Calls made by the sampler itself. This is the default plotted quantity.
* ncall_amortized:
    NISMO calls plus 1/N of the pilot Dynesty training cost.
* ncall_cold_start:
    NISMO calls plus the complete pilot Dynesty training cost.

Choose the plotted definition with ``--ncall-metric``.

Example
-------
Install from the current NISMO repository and install Dynesty:

    pip install dynesty matplotlib scipy tqdm
    pip install "nismo[morph,plot,progress] @ git+https://github.com/nz-gravity/nismo.git@main"

Run a small test:

    python benchmark_dynesty_rwalk_vs_nismo.py \
        --nlive 50 100 \
        --repeats 2 \
        --progress

Run the full paper-style grid:

    python benchmark_dynesty_rwalk_vs_nismo.py \
        --nlive 50 100 200 300 400 500 \
        --repeats 10 \
        --ncall-metric direct \
        --progress

Regenerate the summaries and figure without rerunning samplers:

    python benchmark_dynesty_rwalk_vs_nismo.py \
        --output-dir gaussian_shell_benchmark \
        --plot-only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import logsumexp

RAW_FIELDS = [
    "method",
    "method_label",
    "nlive",
    "repeat",
    "seed",
    "status",
    "termination_reason",
    "logz",
    "logzerr",
    "true_logz",
    "error",
    "abs_error",
    "squared_error",
    "ncall_direct",
    "ncall_training",
    "ncall_amortized",
    "ncall_cold_start",
    "niter",
    "nproposals",
    "runtime_s",
    "proposal_fit_time_s",
    "n_training_samples",
    "training_pilot_seed",
    "training_pilot_repeat",
    "message",
    "timestamp_utc",
]

METHOD_ORDER = {
    "dynesty_rwalk": 0,
    "nismo_fixed_morph": 1,
    "nismo_adaptive_morph": 1,
    "nismo_rwalk": 1,
    "nismo_s-rwalk": 1,
    "nismo_en-rwalk": 1,
}

TRUE_LOGZ_BY_DIMENSION = {
    10: -14.59,
    20: -36.09,
    30: -60.13,
    60: -139.95,
    100: -255.84,
}


class TwinGaussianShell:
    """Twin Gaussian-shell likelihood under a normalized uniform box prior."""

    def __init__(
        self,
        *,
        ndim: int,
        prior_half_width: float,
        shell_width: float,
        shell_radius: float,
        center_offset: float,
    ) -> None:
        if ndim < 1:
            raise ValueError("ndim must be positive")
        if prior_half_width <= 0.0:
            raise ValueError("prior_half_width must be positive")
        if shell_width <= 0.0:
            raise ValueError("shell_width must be positive")
        if shell_radius <= 0.0:
            raise ValueError("shell_radius must be positive")
        if center_offset < 0.0:
            raise ValueError("center_offset must be non-negative")

        self.ndim = int(ndim)
        self.prior_half_width = float(prior_half_width)
        self.shell_width = float(shell_width)
        self.shell_radius = float(shell_radius)
        self.center_offset = float(center_offset)

        self.center_1 = np.zeros(self.ndim, dtype=float)
        self.center_2 = np.zeros(self.ndim, dtype=float)
        self.center_1[0] = -self.center_offset
        self.center_2[0] = self.center_offset

        self._log_shell_norm = -0.5 * np.log(2.0 * np.pi * self.shell_width**2)
        self._log_prior_density = -self.ndim * np.log(2.0 * self.prior_half_width)

    def log_likelihood(self, theta: np.ndarray) -> float | np.ndarray:
        """Log likelihood for one point or a batch shaped ``(n, ndim)``."""
        point = np.asarray(theta, dtype=float)
        single_point = point.ndim == 1
        if single_point:
            point = point.reshape(1, -1)
        if point.ndim != 2 or point.shape[1] != self.ndim:
            raise ValueError(
                f"theta must have shape ({self.ndim},) or (n, {self.ndim})"
            )
        radius_1 = np.linalg.norm(point - self.center_1, axis=1)
        radius_2 = np.linalg.norm(point - self.center_2, axis=1)
        component_1 = (
            self._log_shell_norm
            - 0.5 * ((radius_1 - self.shell_radius) / self.shell_width) ** 2
        )
        component_2 = (
            self._log_shell_norm
            - 0.5 * ((radius_2 - self.shell_radius) / self.shell_width) ** 2
        )
        values = np.asarray(logsumexp((component_1, component_2), axis=0), dtype=float)
        return float(values[0]) if single_point else values

    def log_prior(self, theta: np.ndarray) -> float | np.ndarray:
        """Normalized log prior for one point or a batch on [-L, L]^D."""
        point = np.asarray(theta, dtype=float)
        single_point = point.ndim == 1
        if single_point:
            point = point.reshape(1, -1)
        if point.ndim != 2 or point.shape[1] != self.ndim:
            raise ValueError(
                f"theta must have shape ({self.ndim},) or (n, {self.ndim})"
            )
        values = np.where(
            np.all(np.abs(point) <= self.prior_half_width, axis=1),
            self._log_prior_density,
            -np.inf,
        )
        return float(values[0]) if single_point else np.asarray(values, dtype=float)

    def prior_transform(self, unit_cube: np.ndarray) -> np.ndarray:
        """Map [0, 1]^D to [-L, L]^D."""
        u = np.asarray(unit_cube, dtype=float)
        return -self.prior_half_width + 2.0 * self.prior_half_width * u


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_version(distribution_name: str) -> str:
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def deterministic_seed(
    base_seed: int,
    *,
    method_code: int,
    nlive: int,
    repeat: int,
) -> int:
    """Generate stable, independent uint32 seeds from benchmark coordinates."""
    sequence = np.random.SeedSequence(
        [int(base_seed), int(method_code), int(nlive), int(repeat)]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def integer_or_zero(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def normalize_row(row: dict[str, Any]) -> dict[str, str]:
    return {field: str(row.get(field, "")) for field in RAW_FIELDS}


def row_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row["method"]),
        integer_or_zero(row["nlive"]),
        integer_or_zero(row["repeat"]),
    )


def read_raw_rows(path: Path) -> dict[tuple[str, int, int], dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = {}
        for row in csv.DictReader(stream):
            normalized = normalize_row(row)
            rows[row_key(normalized)] = normalized
        return rows


def write_raw_rows(
    path: Path,
    rows: dict[tuple[str, int, int], dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        rows.values(),
        key=lambda row: (
            integer_or_zero(row["nlive"]),
            METHOD_ORDER.get(row["method"], 99),
            integer_or_zero(row["repeat"]),
        ),
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS)
        writer.writeheader()
        writer.writerows(ordered)
    temporary.replace(path)


def upsert_row(
    rows: dict[tuple[str, int, int], dict[str, str]],
    row: dict[str, Any],
    raw_path: Path,
) -> None:
    normalized = normalize_row(row)
    rows[row_key(normalized)] = normalized
    write_raw_rows(raw_path, rows)


def is_successful(
    rows: dict[tuple[str, int, int], dict[str, str]],
    method: str,
    nlive: int,
    repeat: int,
) -> bool:
    row = rows.get((method, nlive, repeat))
    return row is not None and row.get("status") == "success"


def make_success_row(
    *,
    method: str,
    method_label: str,
    nlive: int,
    repeat: int,
    seed: int,
    logz: float,
    logzerr: float,
    true_logz: float,
    ncall_direct: int,
    ncall_training: int,
    ncall_amortized: float,
    ncall_cold_start: int,
    niter: int,
    nproposals: int,
    runtime_s: float,
    proposal_fit_time_s: float = 0.0,
    n_training_samples: int = 0,
    training_pilot_seed: int | None = None,
    training_pilot_repeat: int | None = None,
    termination_reason: str = "",
    status: str = "success",
    message: str = "",
) -> dict[str, Any]:
    error = float(logz - true_logz)
    return {
        "method": method,
        "method_label": method_label,
        "nlive": nlive,
        "repeat": repeat,
        "seed": seed,
        "status": status,
        "termination_reason": termination_reason,
        "logz": logz,
        "logzerr": logzerr,
        "true_logz": true_logz,
        "error": error,
        "abs_error": abs(error),
        "squared_error": error**2,
        "ncall_direct": ncall_direct,
        "ncall_training": ncall_training,
        "ncall_amortized": ncall_amortized,
        "ncall_cold_start": ncall_cold_start,
        "niter": niter,
        "nproposals": nproposals,
        "runtime_s": runtime_s,
        "proposal_fit_time_s": proposal_fit_time_s,
        "n_training_samples": n_training_samples,
        "training_pilot_seed": (
            "" if training_pilot_seed is None else training_pilot_seed
        ),
        "training_pilot_repeat": (
            "" if training_pilot_repeat is None else training_pilot_repeat
        ),
        "message": message,
        "timestamp_utc": utc_now(),
    }


def make_failure_row(
    *,
    method: str,
    method_label: str,
    nlive: int,
    repeat: int,
    seed: int,
    true_logz: float,
    error: BaseException | str,
    termination_reason: str = "exception",
) -> dict[str, Any]:
    message = str(error)
    if isinstance(error, BaseException):
        message = "".join(traceback.format_exception_only(type(error), error)).strip()
    return {
        "method": method,
        "method_label": method_label,
        "nlive": nlive,
        "repeat": repeat,
        "seed": seed,
        "status": "failed",
        "termination_reason": termination_reason,
        "logz": "",
        "logzerr": "",
        "true_logz": true_logz,
        "error": "",
        "abs_error": "",
        "squared_error": "",
        "ncall_direct": "",
        "ncall_training": "",
        "ncall_amortized": "",
        "ncall_cold_start": "",
        "niter": "",
        "nproposals": "",
        "runtime_s": "",
        "proposal_fit_time_s": "",
        "n_training_samples": "",
        "training_pilot_seed": "",
        "training_pilot_repeat": "",
        "message": message,
        "timestamp_utc": utc_now(),
    }


def run_dynesty(
    *,
    model: TwinGaussianShell,
    nlive: int,
    seed: int,
    dlogz: float,
    bound: str,
    walks: int | None,
    facc: float,
    progress: bool,
    collect_posterior: bool,
) -> tuple[dict[str, Any], np.ndarray | None]:
    """Run one static Dynesty rwalk calculation."""
    import dynesty
    from dynesty import utils as dyfunc

    sampler_kwargs: dict[str, Any] = {
        "nlive": nlive,
        "sample": "rwalk",
        "bound": bound,
        "rstate": np.random.default_rng(seed),
        "facc": facc,
    }
    if walks is not None:
        sampler_kwargs["walks"] = walks

    start = time.perf_counter()
    sampler = dynesty.NestedSampler(
        model.log_likelihood,
        model.prior_transform,
        model.ndim,
        **sampler_kwargs,
    )
    sampler.run_nested(dlogz=dlogz, print_progress=progress)
    runtime_s = time.perf_counter() - start
    result = sampler.results

    logz = float(np.asarray(result.logz)[-1])
    logzerr = float(np.asarray(result.logzerr)[-1])
    ncall = int(np.sum(np.asarray(result.ncall, dtype=np.int64)))
    niter = int(getattr(result, "niter", len(result.logl)))

    posterior: np.ndarray | None = None
    if collect_posterior:
        samples = np.asarray(result.samples, dtype=float)
        log_weights = np.asarray(result.logwt, dtype=float)
        weights = np.exp(log_weights - logsumexp(log_weights))
        posterior = np.asarray(
            dyfunc.resample_equal(
                samples,
                weights,
                rstate=np.random.default_rng(seed ^ 0xA5A5A5A5),
            ),
            dtype=float,
        )

    statistics = {
        "logz": logz,
        "logzerr": logzerr,
        "ncall": ncall,
        "niter": niter,
        "runtime_s": runtime_s,
    }
    return statistics, posterior


def prepare_training_samples(
    posterior: np.ndarray,
    *,
    thin: int,
    max_samples: int,
    seed: int,
) -> np.ndarray:
    """Apply notebook-style thinning and an optional deterministic cap."""
    if thin < 1:
        raise ValueError("training_thin must be at least one")
    samples = np.asarray(posterior, dtype=float)[::thin]
    if max_samples > 0 and len(samples) > max_samples:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(samples), size=max_samples, replace=False))
        samples = samples[indices]
    if samples.ndim != 2 or len(samples) < 2:
        raise ValueError(
            "Morph training requires at least two posterior samples after thinning"
        )
    if not np.all(np.isfinite(samples)):
        raise ValueError("Morph training samples contain NaN or infinity")
    return np.array(samples, copy=True)


def parse_kde_bandwidth(value: str) -> str | float:
    try:
        return float(value)
    except ValueError:
        return value


def build_nismo_model(model: TwinGaussianShell) -> Any:
    from nismo import CallableModel

    return CallableModel(
        ndim=model.ndim,
        parameter_names=tuple(f"x{index}" for index in range(model.ndim)),
        log_likelihood_fn=model.log_likelihood,
        log_prior_fn=model.log_prior,
        vectorized=True,
    )


def fit_morph_proposal(
    *,
    nismo_model: Any,
    training_samples: np.ndarray,
    morph_type: str,
    kde_bw: str | float,
    min_tc: float | None,
    top_k_greedy: int,
) -> tuple[Any, float]:
    from nismo import MorphProposal

    start = time.perf_counter()
    proposal = MorphProposal.fit(
        training_samples,
        param_names=nismo_model.parameter_names,
        morph_type=morph_type,
        kde_bw=kde_bw,
        min_tc=min_tc,
        top_k_greedy=top_k_greedy,
    )
    fit_time_s = time.perf_counter() - start
    return proposal, fit_time_s


def run_nismo(
    *,
    nismo_model: Any,
    importance_proposal: Any,
    nlive: int,
    seed: int,
    proposal_scheme: str,
    dlogz: float,
    proposal_batch_size: int,
    tie_policy: str,
    ensemble_walkers: int,
    ensemble_sweeps: int,
    ensemble_de_weight: float,
    ensemble_stretch_weight: float,
    ensemble_gaussian_weight: float,
    rwalk_walks: int | None,
    rwalk_facc: float,
    srwalk_steps: int,
    max_iterations: int,
    max_proposals_per_replacement: int,
    max_likelihood_calls: int | None,
    max_wall_time: float | None,
    progress: bool,
) -> dict[str, Any]:
    """Run one NISMO calculation with the already fitted fixed importance q0."""
    from nismo import (
        EnsembleMoveWeights,
        EnsembleRWalkSettings,
        NISMOSampler,
        RWalkSettings,
        SRWalkSettings,
    )

    sampler_kwargs: dict[str, Any] = {
        "model": nismo_model,
        "importance_morph": importance_proposal,
        "proposal_scheme": proposal_scheme,
        "n_live": nlive,
        "rng": seed,
        "proposal_batch_size": proposal_batch_size,
        "tie_policy": tie_policy,
    }

    if proposal_scheme == "en-rwalk":
        sampler_kwargs["ensemble_rwalk_settings"] = EnsembleRWalkSettings(
            n_walkers=ensemble_walkers,
            n_sweeps=ensemble_sweeps,
            move_weights=EnsembleMoveWeights(
                de=ensemble_de_weight,
                stretch=ensemble_stretch_weight,
                gaussian=ensemble_gaussian_weight,
            ),
        )
    elif proposal_scheme == "rwalk":
        sampler_kwargs["rwalk_settings"] = RWalkSettings(
            walks=rwalk_walks,
            facc=rwalk_facc,
        )
    elif proposal_scheme == "s-rwalk":
        sampler_kwargs["srwalk_settings"] = SRWalkSettings(
            n_steps=srwalk_steps,
        )

    sampler = NISMOSampler(**sampler_kwargs)
    start = time.perf_counter()
    result = sampler.run(
        dlogz=dlogz,
        max_iterations=max_iterations,
        max_proposals_per_replacement=max_proposals_per_replacement,
        max_likelihood_calls=max_likelihood_calls,
        max_wall_time=max_wall_time,
        progress=progress,
    )
    runtime_s = time.perf_counter() - start

    return {
        "logz": float(result.logz),
        "logzerr": float(result.logzerr),
        "ncall": int(result.n_likelihood_calls),
        "niter": int(result.niter),
        "nproposals": int(result.n_proposals),
        "runtime_s": runtime_s,
        "success": bool(result.success),
        "termination_reason": str(result.termination_reason),
        "warnings": tuple(str(item) for item in result.warnings),
    }


def sample_standard_deviation(values: np.ndarray) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def summarize_rows(
    *,
    rows: dict[tuple[str, int, int], dict[str, str]],
    repeats: int,
    ncall_metric: str,
    summary_path: Path,
    requested_nlive: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate successful runs by method and n_live."""
    ncall_field = {
        "direct": "ncall_direct",
        "amortized": "ncall_amortized",
        "cold-start": "ncall_cold_start",
    }[ncall_metric]

    groups: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in rows.values():
        nlive = integer_or_zero(row["nlive"])
        repeat = integer_or_zero(row["repeat"])
        if requested_nlive is not None and nlive not in requested_nlive:
            continue
        if repeat < 0 or repeat >= repeats:
            continue
        if row.get("status") != "success":
            continue
        logz = finite_float(row.get("logz"))
        if not math.isfinite(logz):
            continue
        key = (row["method"], nlive)
        groups.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for (method, nlive), group in sorted(
        groups.items(),
        key=lambda item: (
            item[0][1],
            METHOD_ORDER.get(item[0][0], 99),
        ),
    ):
        logz = np.asarray([float(row["logz"]) for row in group], dtype=float)
        logzerr = np.asarray(
            [finite_float(row["logzerr"]) for row in group],
            dtype=float,
        )
        true_logz = float(group[0]["true_logz"])
        errors = logz - true_logz
        ncall_direct = np.asarray(
            [float(row["ncall_direct"]) for row in group],
            dtype=float,
        )
        ncall_training = np.asarray(
            [float(row["ncall_training"]) for row in group],
            dtype=float,
        )
        ncall_amortized = np.asarray(
            [float(row["ncall_amortized"]) for row in group],
            dtype=float,
        )
        ncall_cold_start = np.asarray(
            [float(row["ncall_cold_start"]) for row in group],
            dtype=float,
        )
        ncall_plot = np.asarray(
            [float(row[ncall_field]) for row in group],
            dtype=float,
        )
        runtime = np.asarray(
            [finite_float(row["runtime_s"]) for row in group],
            dtype=float,
        )
        fit_time = np.asarray(
            [finite_float(row["proposal_fit_time_s"], 0.0) for row in group],
            dtype=float,
        )
        n = len(group)

        summaries.append(
            {
                "method": method,
                "method_label": group[0]["method_label"],
                "nlive": nlive,
                "n_success": n,
                "n_requested": repeats,
                "true_logz": true_logz,
                "logz_mean": float(np.mean(logz)),
                "logz_std": sample_standard_deviation(logz),
                "logz_sem": (
                    sample_standard_deviation(logz) / math.sqrt(n)
                    if n > 0
                    else math.nan
                ),
                "logz_median": float(np.median(logz)),
                "logz_q16": float(np.quantile(logz, 0.16)),
                "logz_q84": float(np.quantile(logz, 0.84)),
                "logzerr_mean": float(np.nanmean(logzerr)),
                "bias": float(np.mean(errors)),
                "mean_abs_error": float(np.mean(np.abs(errors))),
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "ncall_direct_mean": float(np.mean(ncall_direct)),
                "ncall_direct_std": sample_standard_deviation(ncall_direct),
                "ncall_training_mean": float(np.mean(ncall_training)),
                "ncall_amortized_mean": float(np.mean(ncall_amortized)),
                "ncall_amortized_std": sample_standard_deviation(ncall_amortized),
                "ncall_cold_start_mean": float(np.mean(ncall_cold_start)),
                "ncall_cold_start_std": sample_standard_deviation(ncall_cold_start),
                "ncall_metric": ncall_metric,
                "ncall_plot_mean": float(np.mean(ncall_plot)),
                "ncall_plot_std": sample_standard_deviation(ncall_plot),
                "runtime_mean_s": float(np.nanmean(runtime)),
                "runtime_std_s": sample_standard_deviation(runtime),
                "proposal_fit_time_mean_s": float(np.nanmean(fit_time)),
            }
        )

    fields = [
        "method",
        "method_label",
        "nlive",
        "n_success",
        "n_requested",
        "true_logz",
        "logz_mean",
        "logz_std",
        "logz_sem",
        "logz_median",
        "logz_q16",
        "logz_q84",
        "logzerr_mean",
        "bias",
        "mean_abs_error",
        "rmse",
        "ncall_direct_mean",
        "ncall_direct_std",
        "ncall_training_mean",
        "ncall_amortized_mean",
        "ncall_amortized_std",
        "ncall_cold_start_mean",
        "ncall_cold_start_std",
        "ncall_metric",
        "ncall_plot_mean",
        "ncall_plot_std",
        "runtime_mean_s",
        "runtime_std_s",
        "proposal_fit_time_mean_s",
    ]
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    temporary.replace(summary_path)
    return summaries


def plot_summary(
    *,
    summaries: list[dict[str, Any]],
    true_logz: float,
    ncall_metric: str,
    logarithmic_ncall_axis: bool,
    output_dir: Path,
    show: bool,
) -> tuple[Path, Path]:
    """Create the requested logZ and likelihood-call comparison figure."""
    import matplotlib.pyplot as plt

    if not summaries:
        raise RuntimeError("No successful runs are available to plot")

    methods = sorted(
        {str(row["method"]) for row in summaries},
        key=lambda method: METHOD_ORDER.get(method, 99),
    )
    styles = {
        "dynesty_rwalk": {
            "color": "black",
            "marker": "o",
            "linestyle": "-",
        },
    }

    fig, (axis_logz, axis_calls) = plt.subplots(
        2,
        1,
        figsize=(7.2, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": (3.0, 1.15), "hspace": 0.06},
    )

    axis_logz.axhline(
        true_logz,
        color="black",
        linestyle=":",
        linewidth=1.1,
        label="Reference log(Z)",
        zorder=1,
    )

    for method_index, method in enumerate(methods):
        group = sorted(
            (row for row in summaries if row["method"] == method),
            key=lambda row: int(row["nlive"]),
        )
        x = np.asarray([row["nlive"] for row in group], dtype=float)
        logz_mean = np.asarray([row["logz_mean"] for row in group], dtype=float)
        logz_std = np.asarray([row["logz_std"] for row in group], dtype=float)
        ncall_mean = np.asarray([row["ncall_plot_mean"] for row in group], dtype=float)
        ncall_std = np.asarray([row["ncall_plot_std"] for row in group], dtype=float)

        style = styles.get(
            method,
            {
                "color": "tab:red" if method_index == 1 else f"C{method_index}",
                "marker": "o",
                "linestyle": "--",
            },
        )
        label = str(group[0]["method_label"])

        axis_logz.fill_between(
            x,
            logz_mean - logz_std,
            logz_mean + logz_std,
            color=style["color"],
            alpha=0.12,
            linewidth=0,
            zorder=2,
        )
        axis_logz.errorbar(
            x,
            logz_mean,
            yerr=logz_std,
            label=label,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.1,
            markersize=4.5,
            capsize=3,
            zorder=3,
        )

        lower_error = np.minimum(
            ncall_std,
            np.maximum(0.0, 0.95 * ncall_mean),
        )
        axis_calls.errorbar(
            x,
            ncall_mean,
            yerr=np.vstack((lower_error, ncall_std)),
            label=label,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.1,
            markersize=4.5,
            capsize=3,
        )

    axis_logz.set_ylabel(r"$\log Z$")
    axis_calls.set_ylabel("Likelihood calls")
    axis_calls.set_xlabel(r"$n_{\rm live}$")

    if logarithmic_ncall_axis:
        axis_calls.set_yscale("log")

    metric_note = {
        "direct": "direct sampler calls",
        "amortized": "calls incl. amortized Morph training",
        "cold-start": "calls incl. full Morph training",
    }[ncall_metric]
    axis_calls.text(
        0.015,
        0.06,
        metric_note,
        transform=axis_calls.transAxes,
        fontsize=8,
        va="bottom",
    )

    for axis in (axis_logz, axis_calls):
        axis.grid(True, which="major", linestyle=":", alpha=0.35)
        axis.minorticks_on()
        axis.tick_params(direction="in", top=True, right=True)

    axis_logz.legend(loc="best", frameon=True)
    fig.align_ylabels((axis_logz, axis_calls))

    png_path = output_dir / "dynesty_rwalk_vs_nismo.png"
    pdf_path = output_dir / "dynesty_rwalk_vs_nismo.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return png_path, pdf_path


def print_summary(summaries: list[dict[str, Any]]) -> None:
    if not summaries:
        print("No successful runs were available for the summary.")
        return
    print("\nAggregate results")
    print("-" * 108)
    print(
        f"{'method':<22} {'nlive':>7} {'runs':>7} "
        f"{'mean logZ':>13} {'std':>10} {'bias':>10} "
        f"{'mean ncall':>15} {'runtime [s]':>14}"
    )
    print("-" * 108)
    for row in summaries:
        print(
            f"{row['method_label']:<22} "
            f"{int(row['nlive']):>7d} "
            f"{int(row['n_success']):>3d}/{int(row['n_requested']):<3d} "
            f"{float(row['logz_mean']):>13.5f} "
            f"{float(row['logz_std']):>10.5f} "
            f"{float(row['bias']):>10.5f} "
            f"{float(row['ncall_plot_mean']):>15.1f} "
            f"{float(row['runtime_mean_s']):>14.1f}"
        )
    print("-" * 108)


def print_incomplete_summary(
    *,
    rows: dict[tuple[str, int, int], dict[str, str]],
    nlive_values: list[int],
    repeats: int,
) -> None:
    """Make excluded partial runs visible in terminal benchmark output."""
    incomplete: dict[tuple[str, int], list[str]] = {}
    labels: dict[tuple[str, int], str] = {}
    for row in rows.values():
        nlive = integer_or_zero(row["nlive"])
        repeat = integer_or_zero(row["repeat"])
        if nlive not in nlive_values or repeat < 0 or repeat >= repeats:
            continue
        if row.get("status") == "success":
            continue
        key = (row["method"], nlive)
        labels[key] = row["method_label"]
        reason = row.get("termination_reason") or row.get("message") or "unknown"
        incomplete.setdefault(key, []).append(reason)
    if not incomplete:
        return
    print("\nExcluded incomplete/failed runs")
    for key, reasons in sorted(
        incomplete.items(),
        key=lambda item: (item[0][1], METHOD_ORDER.get(item[0][0], 99)),
    ):
        counts: dict[str, int] = {}
        for reason in reasons:
            counts[reason] = counts.get(reason, 0) + 1
        rendered_reasons = ", ".join(
            f"{reason} ({count})" for reason, count in sorted(counts.items())
        )
        print(f"  {labels[key]}, nlive={key[1]}: {rendered_reasons}")


_PRESENTATION_ARGUMENTS = frozenset(
    {
        "output_dir",
        "overwrite",
        "plot_only",
        "progress",
        "resume",
        "show",
        "linear_ncall_axis",
        "ncall_metric",
        "nlive",
        "repeats",
    }
)


def benchmark_fingerprint(args: argparse.Namespace) -> str:
    """Return a stable identity for raw samples and training caches.

    Grid and presentation arguments are deliberately omitted so a compatible
    campaign can add live-point values or repeats without mixing numerical
    settings from another experiment.
    """
    payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in _PRESENTATION_ARGUMENTS
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def validate_resume_config(
    *,
    output_dir: Path,
    fingerprint: str,
) -> None:
    """Reject a resume that would combine incompatible numerical runs."""
    config_path = output_dir / "benchmark_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            "cannot safely resume without benchmark_config.json; use a new "
            "output directory or --overwrite"
        )
    with config_path.open(encoding="utf-8") as stream:
        previous = json.load(stream)
    if previous.get("run_fingerprint") != fingerprint:
        raise ValueError(
            "benchmark configuration does not match this output directory; "
            "use a new output directory or --overwrite"
        )


def save_config(
    *,
    args: argparse.Namespace,
    true_logz: float,
    output_dir: Path,
    fingerprint: str,
) -> None:
    config = {
        "created_utc": utc_now(),
        "command": " ".join(sys.argv),
        "run_fingerprint": fingerprint,
        "true_logz": true_logz,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": package_version("numpy"),
            "scipy": package_version("scipy"),
            "matplotlib": package_version("matplotlib"),
            "dynesty": package_version("dynesty"),
            "nismo": package_version("nismo"),
            "morphZ": package_version("morphZ"),
        },
        "design": {
            "proposal_training": (
                "One pilot Dynesty rwalk posterior per nlive; the pilot is also "
                "Dynesty repeat 0. All NISMO repeats at that nlive share the fixed "
                "Morph proposal fitted from the thinned equal-weight posterior."
            ),
            "logz_error_bars": "Empirical sample standard deviation across repeats.",
            "ncall_error_bars": "Empirical sample standard deviation across repeats.",
            "ncall_default": args.ncall_metric,
        },
    }
    path = output_dir / "benchmark_config.json"
    with path.open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2, sort_keys=True)


def cache_paths(output_dir: Path, nlive: int) -> tuple[Path, Path]:
    return (
        output_dir / f"training_posterior_nlive_{nlive}.npz",
        output_dir / f"training_posterior_nlive_{nlive}.json",
    )


def load_training_cache(
    output_dir: Path,
    nlive: int,
    fingerprint: str,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    samples_path, metadata_path = cache_paths(output_dir, nlive)
    if not samples_path.exists() or not metadata_path.exists():
        return None
    with np.load(samples_path) as data:
        posterior = np.asarray(data["posterior_samples"], dtype=float)
    with metadata_path.open("r", encoding="utf-8") as stream:
        cache_metadata = json.load(stream)
    if cache_metadata.get("run_fingerprint") != fingerprint:
        return None
    return posterior, cache_metadata


def save_training_cache(
    *,
    output_dir: Path,
    nlive: int,
    posterior: np.ndarray,
    pilot_row: dict[str, Any],
    fingerprint: str,
) -> None:
    samples_path, metadata_path = cache_paths(output_dir, nlive)
    np.savez_compressed(
        samples_path,
        posterior_samples=np.asarray(posterior, dtype=float),
    )
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "nlive": nlive,
                "created_utc": utc_now(),
                "run_fingerprint": fingerprint,
                "pilot_row": normalize_row(pilot_row),
                "posterior_shape": list(posterior.shape),
            },
            stream,
            indent=2,
            sort_keys=True,
        )


def resolve_true_logz(args: argparse.Namespace) -> float:
    if args.true_logz is not None:
        return float(args.true_logz)
    if args.ndim in TRUE_LOGZ_BY_DIMENSION:
        return TRUE_LOGZ_BY_DIMENSION[args.ndim]
    raise ValueError(
        f"No built-in reference logZ is available for ndim={args.ndim}. "
        "Pass --true-logz explicitly."
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.repeats < 1:
        raise ValueError("--repeats must be at least one")
    if not args.nlive or any(value < 2 for value in args.nlive):
        raise ValueError("all --nlive values must be integers >= 2")
    if len(set(args.nlive)) != len(args.nlive):
        raise ValueError("--nlive values must be unique")
    if args.dlogz <= 0.0:
        raise ValueError("--dlogz must be positive")
    if args.training_thin < 1:
        raise ValueError("--training-thin must be at least one")
    if args.max_training_samples < 0:
        raise ValueError("--max-training-samples cannot be negative")
    if args.nismo_proposal_scheme == "en-rwalk" and any(
        value < args.ensemble_walkers for value in args.nlive
    ):
        raise ValueError("each nlive must be at least --ensemble-walkers for en-rwalk")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Repeated Dynesty rwalk versus NISMO benchmark on the "
            "twin Gaussian-shell problem."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--nlive",
        type=int,
        nargs="+",
        default=[50, 100, 200, 300, 400, 500],
        help="Live-point values to benchmark.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Independent runs of each method per nlive.",
    )
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--ndim", type=int, default=30)
    parser.add_argument("--prior-half-width", type=float, default=6.0)
    parser.add_argument("--shell-width", type=float, default=0.1)
    parser.add_argument("--shell-radius", type=float, default=2.0)
    parser.add_argument("--center-offset", type=float, default=3.5)
    parser.add_argument(
        "--true-logz",
        type=float,
        default=None,
        help=(
            "Reference log evidence. If omitted, the built-in literature value "
            "for ndim=10, 20, or 30 is used."
        ),
    )
    parser.add_argument(
        "--dlogz",
        type=float,
        default=0.1,
        help="Common stopping tolerance passed to both samplers.",
    )

    parser.add_argument(
        "--dynesty-bound",
        choices=("none", "single", "multi", "balls", "cubes"),
        default="multi",
    )
    parser.add_argument(
        "--dynesty-walks",
        type=int,
        default=None,
        help="Optional explicit Dynesty rwalk length.",
    )
    parser.add_argument("--dynesty-facc", type=float, default=0.5)

    parser.add_argument("--morph-type", default="2_group")
    parser.add_argument(
        "--kde-bw",
        default="silverman",
        help="MorphZ KDE bandwidth name or a numeric value.",
    )
    parser.add_argument("--min-tc", type=float, default=None)
    parser.add_argument("--top-k-greedy", type=int, default=1)
    parser.add_argument(
        "--training-thin",
        type=int,
        default=10,
        help="Use posterior_samples[::training_thin], as in the notebook.",
    )
    parser.add_argument(
        "--max-training-samples",
        type=int,
        default=0,
        help="Optional cap after thinning; zero means no cap.",
    )

    parser.add_argument(
        "--nismo-proposal-scheme",
        choices=("fixed_morph", "adaptive_morph", "rwalk", "s-rwalk", "en-rwalk"),
        default="en-rwalk",
        help="NISMO constrained replacement scheme.",
    )
    parser.add_argument("--proposal-batch-size", type=int, default=64)
    parser.add_argument(
        "--tie-policy",
        choices=("strict", "randomized_plateau"),
        default="strict",
        help="NISMO constrained-ordering policy.",
    )
    parser.add_argument("--ensemble-walkers", type=int, default=8)
    parser.add_argument("--ensemble-sweeps", type=int, default=2)
    parser.add_argument("--ensemble-de-weight", type=float, default=0.70)
    parser.add_argument("--ensemble-stretch-weight", type=float, default=0.15)
    parser.add_argument("--ensemble-gaussian-weight", type=float, default=0.15)
    parser.add_argument("--nismo-rwalk-walks", type=int, default=None)
    parser.add_argument("--nismo-rwalk-facc", type=float, default=0.5)
    parser.add_argument("--nismo-srwalk-steps", type=int, default=25)
    parser.add_argument("--max-iterations", type=int, default=10_000)
    parser.add_argument(
        "--max-proposals-per-replacement",
        type=int,
        default=100_000,
    )
    parser.add_argument("--max-likelihood-calls", type=int, default=None)
    parser.add_argument("--max-wall-time", type=float, default=None)

    parser.add_argument(
        "--ncall-metric",
        choices=("direct", "amortized", "cold-start"),
        default="direct",
        help="Call-count definition used in the lower panel.",
    )
    parser.add_argument(
        "--linear-ncall-axis",
        action="store_true",
        help="Use a linear rather than logarithmic likelihood-call axis.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("gaussian_shell_benchmark"),
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse successful rows and cached pilot posterior samples.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory before starting.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Read the existing raw CSV and regenerate summary/plots only.",
    )
    parser.add_argument("--show", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    true_logz = resolve_true_logz(args)

    output_dir = args.output_dir.expanduser().resolve()
    fingerprint = benchmark_fingerprint(args)
    if args.overwrite and output_dir.exists() and not args.plot_only:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "raw_runs.csv"
    summary_path = output_dir / "summary.csv"
    rows = read_raw_rows(raw_path)

    if args.plot_only:
        if not raw_path.exists():
            raise FileNotFoundError(
                f"--plot-only requested, but {raw_path} does not exist"
            )
        summaries = summarize_rows(
            rows=rows,
            repeats=args.repeats,
            ncall_metric=args.ncall_metric,
            summary_path=summary_path,
            requested_nlive=set(args.nlive),
        )
        png_path, pdf_path = plot_summary(
            summaries=summaries,
            true_logz=true_logz,
            ncall_metric=args.ncall_metric,
            logarithmic_ncall_axis=not args.linear_ncall_axis,
            output_dir=output_dir,
            show=args.show,
        )
        print_summary(summaries)
        print_incomplete_summary(
            rows=rows,
            nlive_values=args.nlive,
            repeats=args.repeats,
        )
        print(f"\nSaved summary: {summary_path}")
        print(f"Saved plots:   {png_path} and {pdf_path}")
        return 0

    if raw_path.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(
            f"{raw_path} already exists. Use --resume or --overwrite."
        )
    if raw_path.exists() and args.resume:
        validate_resume_config(output_dir=output_dir, fingerprint=fingerprint)

    save_config(
        args=args,
        true_logz=true_logz,
        output_dir=output_dir,
        fingerprint=fingerprint,
    )

    model = TwinGaussianShell(
        ndim=args.ndim,
        prior_half_width=args.prior_half_width,
        shell_width=args.shell_width,
        shell_radius=args.shell_radius,
        center_offset=args.center_offset,
    )
    nismo_model = build_nismo_model(model)
    nismo_method = f"nismo_{args.nismo_proposal_scheme}"
    nismo_label = f"NISMO ({args.nismo_proposal_scheme})"

    print(f"Twin Gaussian shell: ndim={args.ndim}, reference logZ={true_logz:.3f}")
    print(
        f"Grid: nlive={args.nlive}, repeats={args.repeats}; "
        f"NISMO scheme={args.nismo_proposal_scheme}"
    )
    print(f"Incremental results: {raw_path}")

    for nlive in args.nlive:
        print(f"\n{'=' * 72}\nnlive = {nlive}\n{'=' * 72}")

        cached = load_training_cache(output_dir, nlive, fingerprint)
        posterior: np.ndarray | None = None
        cache_metadata: dict[str, Any] | None = None

        if cached is not None:
            posterior, cache_metadata = cached
            pilot_row = cache_metadata.get("pilot_row")
            if pilot_row and not is_successful(rows, "dynesty_rwalk", nlive, 0):
                upsert_row(rows, pilot_row, raw_path)
            print(f"Loaded cached pilot posterior with shape {tuple(posterior.shape)}")

        if posterior is None:
            pilot_seed = deterministic_seed(
                args.seed,
                method_code=1,
                nlive=nlive,
                repeat=0,
            )
            print(
                f"[Dynesty rwalk] nlive={nlive}, repeat=1/{args.repeats}, "
                f"seed={pilot_seed} (pilot for Morph)"
            )
            try:
                statistics, posterior = run_dynesty(
                    model=model,
                    nlive=nlive,
                    seed=pilot_seed,
                    dlogz=args.dlogz,
                    bound=args.dynesty_bound,
                    walks=args.dynesty_walks,
                    facc=args.dynesty_facc,
                    progress=args.progress,
                    collect_posterior=True,
                )
                pilot_row = make_success_row(
                    method="dynesty_rwalk",
                    method_label="Dynesty (rwalk)",
                    nlive=nlive,
                    repeat=0,
                    seed=pilot_seed,
                    logz=statistics["logz"],
                    logzerr=statistics["logzerr"],
                    true_logz=true_logz,
                    ncall_direct=statistics["ncall"],
                    ncall_training=0,
                    ncall_amortized=statistics["ncall"],
                    ncall_cold_start=statistics["ncall"],
                    niter=statistics["niter"],
                    nproposals=0,
                    runtime_s=statistics["runtime_s"],
                    termination_reason="dlogz",
                )
                upsert_row(rows, pilot_row, raw_path)
                if posterior is None:
                    raise RuntimeError("Dynesty pilot did not return posterior samples")
                save_training_cache(
                    output_dir=output_dir,
                    nlive=nlive,
                    posterior=posterior,
                    pilot_row=pilot_row,
                    fingerprint=fingerprint,
                )
                cache_metadata = {"pilot_row": normalize_row(pilot_row)}
                print(
                    f"  logZ={statistics['logz']:.5f} ± "
                    f"{statistics['logzerr']:.5f}; "
                    f"ncall={statistics['ncall']:,}"
                )
            except BaseException as error:
                failure = make_failure_row(
                    method="dynesty_rwalk",
                    method_label="Dynesty (rwalk)",
                    nlive=nlive,
                    repeat=0,
                    seed=pilot_seed,
                    true_logz=true_logz,
                    error=error,
                )
                upsert_row(rows, failure, raw_path)
                print(f"  FAILED: {failure['message']}", file=sys.stderr)
                posterior = None

        # Complete the remaining independent Dynesty repeats.
        for repeat in range(1, args.repeats):
            if args.resume and is_successful(rows, "dynesty_rwalk", nlive, repeat):
                print(
                    f"[Dynesty rwalk] nlive={nlive}, "
                    f"repeat={repeat + 1}/{args.repeats}: cached"
                )
                continue
            seed = deterministic_seed(
                args.seed,
                method_code=1,
                nlive=nlive,
                repeat=repeat,
            )
            print(
                f"[Dynesty rwalk] nlive={nlive}, "
                f"repeat={repeat + 1}/{args.repeats}, seed={seed}"
            )
            try:
                statistics, _ = run_dynesty(
                    model=model,
                    nlive=nlive,
                    seed=seed,
                    dlogz=args.dlogz,
                    bound=args.dynesty_bound,
                    walks=args.dynesty_walks,
                    facc=args.dynesty_facc,
                    progress=args.progress,
                    collect_posterior=False,
                )
                row = make_success_row(
                    method="dynesty_rwalk",
                    method_label="Dynesty (rwalk)",
                    nlive=nlive,
                    repeat=repeat,
                    seed=seed,
                    logz=statistics["logz"],
                    logzerr=statistics["logzerr"],
                    true_logz=true_logz,
                    ncall_direct=statistics["ncall"],
                    ncall_training=0,
                    ncall_amortized=statistics["ncall"],
                    ncall_cold_start=statistics["ncall"],
                    niter=statistics["niter"],
                    nproposals=0,
                    runtime_s=statistics["runtime_s"],
                    termination_reason="dlogz",
                )
                upsert_row(rows, row, raw_path)
                print(
                    f"  logZ={statistics['logz']:.5f} ± "
                    f"{statistics['logzerr']:.5f}; "
                    f"ncall={statistics['ncall']:,}"
                )
            except BaseException as error:
                failure = make_failure_row(
                    method="dynesty_rwalk",
                    method_label="Dynesty (rwalk)",
                    nlive=nlive,
                    repeat=repeat,
                    seed=seed,
                    true_logz=true_logz,
                    error=error,
                )
                upsert_row(rows, failure, raw_path)
                print(f"  FAILED: {failure['message']}", file=sys.stderr)

        if posterior is None or cache_metadata is None:
            reason = "NISMO skipped because the pilot Dynesty posterior was unavailable"
            print(reason, file=sys.stderr)
            for repeat in range(args.repeats):
                seed = deterministic_seed(
                    args.seed,
                    method_code=2,
                    nlive=nlive,
                    repeat=repeat,
                )
                failure = make_failure_row(
                    method=nismo_method,
                    method_label=nismo_label,
                    nlive=nlive,
                    repeat=repeat,
                    seed=seed,
                    true_logz=true_logz,
                    error=reason,
                    termination_reason="missing_training_posterior",
                )
                upsert_row(rows, failure, raw_path)
            continue

        pilot_row = cache_metadata["pilot_row"]
        pilot_ncall = integer_or_zero(pilot_row["ncall_direct"])
        pilot_seed = integer_or_zero(pilot_row["seed"])

        try:
            training_seed = deterministic_seed(
                args.seed,
                method_code=3,
                nlive=nlive,
                repeat=0,
            )
            training_samples = prepare_training_samples(
                posterior,
                thin=args.training_thin,
                max_samples=args.max_training_samples,
                seed=training_seed,
            )
            print(
                f"Fitting Morph proposal from {len(training_samples):,} "
                f"training samples..."
            )
            importance_proposal, proposal_fit_time_s = fit_morph_proposal(
                nismo_model=nismo_model,
                training_samples=training_samples,
                morph_type=args.morph_type,
                kde_bw=parse_kde_bandwidth(args.kde_bw),
                min_tc=args.min_tc,
                top_k_greedy=args.top_k_greedy,
            )
            print(f"  Morph fit completed in {proposal_fit_time_s:.1f} s")
        except BaseException as error:
            reason = "".join(
                traceback.format_exception_only(type(error), error)
            ).strip()
            print(f"Morph fitting FAILED: {reason}", file=sys.stderr)
            for repeat in range(args.repeats):
                seed = deterministic_seed(
                    args.seed,
                    method_code=2,
                    nlive=nlive,
                    repeat=repeat,
                )
                failure = make_failure_row(
                    method=nismo_method,
                    method_label=nismo_label,
                    nlive=nlive,
                    repeat=repeat,
                    seed=seed,
                    true_logz=true_logz,
                    error=error,
                    termination_reason="morph_fit_failed",
                )
                upsert_row(rows, failure, raw_path)
            continue

        for repeat in range(args.repeats):
            if args.resume and is_successful(rows, nismo_method, nlive, repeat):
                print(
                    f"[{nismo_label}] nlive={nlive}, "
                    f"repeat={repeat + 1}/{args.repeats}: cached"
                )
                continue

            seed = deterministic_seed(
                args.seed,
                method_code=2,
                nlive=nlive,
                repeat=repeat,
            )
            print(
                f"[{nismo_label}] nlive={nlive}, "
                f"repeat={repeat + 1}/{args.repeats}, seed={seed}"
            )
            try:
                statistics = run_nismo(
                    nismo_model=nismo_model,
                    importance_proposal=importance_proposal,
                    nlive=nlive,
                    seed=seed,
                    proposal_scheme=args.nismo_proposal_scheme,
                    dlogz=args.dlogz,
                    proposal_batch_size=args.proposal_batch_size,
                    tie_policy=args.tie_policy,
                    ensemble_walkers=args.ensemble_walkers,
                    ensemble_sweeps=args.ensemble_sweeps,
                    ensemble_de_weight=args.ensemble_de_weight,
                    ensemble_stretch_weight=args.ensemble_stretch_weight,
                    ensemble_gaussian_weight=args.ensemble_gaussian_weight,
                    rwalk_walks=args.nismo_rwalk_walks,
                    rwalk_facc=args.nismo_rwalk_facc,
                    srwalk_steps=args.nismo_srwalk_steps,
                    max_iterations=args.max_iterations,
                    max_proposals_per_replacement=(args.max_proposals_per_replacement),
                    max_likelihood_calls=args.max_likelihood_calls,
                    max_wall_time=args.max_wall_time,
                    progress=args.progress,
                )
                direct = int(statistics["ncall"])
                status = "success" if statistics["success"] else "incomplete"
                warning_message = " | ".join(statistics["warnings"])
                row = make_success_row(
                    method=nismo_method,
                    method_label=nismo_label,
                    nlive=nlive,
                    repeat=repeat,
                    seed=seed,
                    logz=statistics["logz"],
                    logzerr=statistics["logzerr"],
                    true_logz=true_logz,
                    ncall_direct=direct,
                    ncall_training=pilot_ncall,
                    ncall_amortized=direct + pilot_ncall / args.repeats,
                    ncall_cold_start=direct + pilot_ncall,
                    niter=statistics["niter"],
                    nproposals=statistics["nproposals"],
                    runtime_s=statistics["runtime_s"],
                    proposal_fit_time_s=proposal_fit_time_s,
                    n_training_samples=len(training_samples),
                    training_pilot_seed=pilot_seed,
                    training_pilot_repeat=0,
                    termination_reason=statistics["termination_reason"],
                    status=status,
                    message=warning_message,
                )
                upsert_row(rows, row, raw_path)
                print(
                    f"  logZ={statistics['logz']:.5f} ± "
                    f"{statistics['logzerr']:.5f}; "
                    f"ncall={direct:,}; "
                    f"termination={statistics['termination_reason']}"
                )
                if not statistics["success"]:
                    print(
                        "  Result saved as incomplete and excluded from "
                        "aggregate statistics.",
                        file=sys.stderr,
                    )
            except BaseException as error:
                failure = make_failure_row(
                    method=nismo_method,
                    method_label=nismo_label,
                    nlive=nlive,
                    repeat=repeat,
                    seed=seed,
                    true_logz=true_logz,
                    error=error,
                )
                upsert_row(rows, failure, raw_path)
                print(f"  FAILED: {failure['message']}", file=sys.stderr)

        # Refresh intermediate summaries after every nlive value.
        summarize_rows(
            rows=rows,
            repeats=args.repeats,
            ncall_metric=args.ncall_metric,
            summary_path=summary_path,
            requested_nlive=set(args.nlive),
        )

    summaries = summarize_rows(
        rows=rows,
        repeats=args.repeats,
        ncall_metric=args.ncall_metric,
        summary_path=summary_path,
        requested_nlive=set(args.nlive),
    )
    png_path, pdf_path = plot_summary(
        summaries=summaries,
        true_logz=true_logz,
        ncall_metric=args.ncall_metric,
        logarithmic_ncall_axis=not args.linear_ncall_axis,
        output_dir=output_dir,
        show=args.show,
    )
    print_summary(summaries)
    print_incomplete_summary(
        rows=rows,
        nlive_values=args.nlive,
        repeats=args.repeats,
    )
    print(f"\nSaved raw runs: {raw_path}")
    print(f"Saved summary:  {summary_path}")
    print(f"Saved plots:    {png_path} and {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
