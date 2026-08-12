"""Persistence helpers for complete NISMO run-output bundles."""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Mapping
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .diagnostics import summarize
from .plotting import plot_nested_progress, plot_run, plot_weight_health

if TYPE_CHECKING:
    from .results import NISMOResult


def normalize_output_path(
    output_path: str | os.PathLike[str] | None,
) -> Path | None:
    """Validate and normalize an optional run-output directory."""
    if output_path is None:
        return None
    if isinstance(output_path, str) and not output_path.strip():
        raise ValueError("output_path must not be empty")
    if not isinstance(output_path, (str, os.PathLike)):
        raise TypeError("output_path must be a path-like value or None")
    path = Path(output_path).expanduser()
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"output_path is not a directory: {path}")
    return path


def prepare_output_directory(output_path: Path) -> None:
    """Create a validated output directory before an expensive run starts."""
    if output_path.exists() and not output_path.is_dir():
        raise NotADirectoryError(f"output_path is not a directory: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)


def _json_compatible(value: Any) -> Any:
    """Convert nested dataclass and NumPy values to strict JSON values."""
    if is_dataclass(value) and not isinstance(value, type):
        return _json_compatible(asdict(value))
    if isinstance(value, np.generic):
        return _json_compatible(value.item())
    if isinstance(value, float):
        if np.isnan(value):
            return "NaN"
        if np.isposinf(value):
            return "Infinity"
        if np.isneginf(value):
            return "-Infinity"
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _save_weighted_samples(result: NISMOResult, output_path: Path) -> None:
    log_likelihood = np.concatenate(
        (result.dead_log_likelihood, result.final_live_log_likelihood)
    )
    log_prior = np.concatenate((result.dead_log_prior, result.final_live_log_prior))
    log_q0 = np.concatenate((result.dead_log_q0, result.final_live_log_q0))
    tie_breakers = np.concatenate(
        (result.dead_tie_breakers, result.final_live_tie_breakers)
    )
    is_live = np.concatenate(
        (
            np.zeros(result.niter, dtype=np.bool_),
            np.ones(result.nlive, dtype=np.bool_),
        )
    )
    np.savez_compressed(
        output_path / "weighted_samples.npz",
        parameter_names=np.asarray(result.parameter_names, dtype=np.str_),
        samples=result.all_points,
        posterior_weights=result.posterior_weights,
        log_posterior_weights=result.log_posterior_weights,
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        log_q0=log_q0,
        log_psi0=result.all_log_psi0,
        tie_breakers=tie_breakers,
        is_live=is_live,
    )


def _save_history(result: NISMOResult, output_path: Path) -> None:
    arrays = {
        field.name: getattr(result.history, field.name)
        for field in fields(result.history)
    }
    if result.ensemble_move_history is not None:
        arrays["ensemble_move_names"] = np.asarray(
            result.ensemble_move_history.names,
            dtype=np.str_,
        )
        for name in ("proposed", "valid", "accepted", "moved"):
            arrays[f"ensemble_{name}"] = getattr(result.ensemble_move_history, name)
    np.savez_compressed(output_path / "run_history.npz", **arrays)


def _save_plots(result: NISMOResult, output_path: Path) -> tuple[str, ...]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn(
            "Matplotlib is unavailable; saved samples and diagnostics without "
            "plots. Install 'nismo[plot]' to save diagnostic figures.",
            RuntimeWarning,
            stacklevel=3,
        )
        return ()

    plotters = (
        ("run_diagnostics.png", plot_run),
        ("nested_progress.png", plot_nested_progress),
        ("weight_health.png", plot_weight_health),
    )
    saved: list[str] = []
    for filename, plotter in plotters:
        figure, _ = plotter(result)
        try:
            figure.savefig(output_path / filename, dpi=160, bbox_inches="tight")
        finally:
            plt.close(figure)
        saved.append(filename)
    return tuple(saved)


def _diagnostic_payload(
    result: NISMOResult,
    *,
    plot_files: tuple[str, ...],
) -> dict[str, Any]:
    queue = asdict(result.queue_diagnostics)
    queue["queue_efficiency"] = result.queue_diagnostics.queue_efficiency
    queue["compute_efficiency"] = result.queue_diagnostics.compute_efficiency
    ensemble_totals = None
    if result.ensemble_move_history is not None:
        history = result.ensemble_move_history
        ensemble_totals = {
            name: {
                metric: int(np.sum(getattr(history, metric)[:, index]))
                for metric in ("proposed", "valid", "accepted", "moved")
            }
            for index, name in enumerate(history.names)
        }
    return {
        "schema_version": 1,
        "result": {
            "logz": result.logz,
            "logzerr": result.logzerr,
            "information": result.information,
            "success": result.success,
            "termination_reason": result.termination_reason,
            "niter": result.niter,
            "nlive": result.nlive,
            "ndim": len(result.parameter_names),
            "parameter_names": result.parameter_names,
            "n_likelihood_calls": result.n_likelihood_calls,
            "n_prior_calls": result.n_prior_calls,
            "n_proposals": result.n_proposals,
        },
        "run_diagnostics": summarize(result),
        "queue_diagnostics": queue,
        "srwalk_diagnostics": result.srwalk_diagnostics,
        "ensemble_move_totals": ensemble_totals,
        "config": result.config,
        "proposal_updates": result.proposal_updates,
        "nonfinite_counts": dict(result.nonfinite_counts),
        "warnings": result.warnings,
        "reproducibility": {
            "rng_bit_generator": result.rng_bit_generator,
            "rng_state_initial": result.rng_state_initial,
            "rng_state_final": result.rng_state_final,
            "importance_morph_description": result.importance_morph_description,
        },
        "files": {
            "weighted_samples": "weighted_samples.npz",
            "run_history": "run_history.npz",
            "diagnostics": "diagnostics.json",
            "plots": plot_files,
        },
    }


def save_run_outputs(
    result: NISMOResult,
    output_path: str | os.PathLike[str],
    *,
    plots: bool = True,
) -> Path:
    """Save weighted samples, run history, diagnostics, and optional plots.

    Existing files with NISMO's standard output names are replaced. Other
    files in the directory are left untouched.
    """
    directory = normalize_output_path(output_path)
    if directory is None:  # pragma: no cover - excluded by the public type
        raise TypeError("output_path must not be None")
    prepare_output_directory(directory)
    _save_weighted_samples(result, directory)
    _save_history(result, directory)
    plot_files = _save_plots(result, directory) if plots else ()
    payload = _json_compatible(_diagnostic_payload(result, plot_files=plot_files))
    (directory / "diagnostics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return directory
