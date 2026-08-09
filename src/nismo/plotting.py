"""Optional Matplotlib plots consuming only stored results."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .results import NISMOResult


def plot_run(result: NISMOResult) -> tuple[Any, Any]:
    """Create the three-panel Phase 2 run diagnostic.

    Returns
    -------
    tuple
        ``(figure, axes)`` where ``axes`` contains three aligned Matplotlib
        axes. The function never calls ``show`` or writes files.

    Raises
    ------
    ImportError
        If the optional Matplotlib dependency is unavailable.
    """
    import matplotlib.pyplot as plt

    iteration = result.history.iteration
    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
    axes[0].plot(
        iteration,
        result.history.discarded_log_psi,
        label="dead threshold",
    )
    axes[0].fill_between(
        iteration,
        result.history.live_min_log_psi,
        result.history.live_max_log_psi,
        alpha=0.25,
        label="live range",
    )
    axes[0].set_ylabel(r"$\log\Psi$")
    axes[0].legend()

    axes[1].plot(iteration, result.history.logz_total, label="total logz")
    axes[1].plot(iteration, result.history.logz_live, label="live remainder")
    axes[1].set_ylabel("log evidence")
    axes[1].legend()

    axes[2].plot(
        iteration,
        result.history.likelihood_calls,
        label="likelihood calls",
    )
    proposals_axis = axes[2].twinx()
    proposals_axis.plot(
        iteration,
        result.history.proposals,
        color="tab:orange",
        alpha=0.7,
        label="proposals / iteration",
    )
    axes[2].set_xlabel("iteration")
    axes[2].set_ylabel("cumulative calls")
    proposals_axis.set_ylabel("proposals")
    figure.tight_layout()
    return figure, axes


def plot_nested_progress(result: NISMOResult) -> tuple[Any, Any]:
    """Plot live-set, remaining-evidence, and threshold progression.

    The live-set panel uses the stored minimum, median, and maximum
    pseudo-likelihood rather than retaining an ``(niter, nlive)`` trace.  The
    remaining-evidence panel shows ``logz_live``, the logarithm of the current
    mean-live estimate of evidence still inside the active volume.

    Returns
    -------
    tuple
        ``(figure, axes)`` where ``axes`` contains three aligned Matplotlib
        axes. The function never calls ``show`` or writes files.
    """
    import matplotlib.pyplot as plt

    history = result.history
    iteration = history.iteration
    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 9))

    axes[0].fill_between(
        iteration,
        history.live_min_log_psi,
        history.live_max_log_psi,
        color="tab:blue",
        alpha=0.2,
        label="live min-max",
    )
    axes[0].plot(
        iteration,
        history.live_min_log_psi,
        color="tab:blue",
        alpha=0.65,
        linewidth=0.8,
        label="live minimum",
    )
    axes[0].plot(
        iteration,
        history.live_median_log_psi,
        color="tab:blue",
        linewidth=1.5,
        label="live median",
    )
    axes[0].plot(
        iteration,
        history.live_max_log_psi,
        color="tab:blue",
        alpha=0.65,
        linewidth=0.8,
        label="live maximum",
    )
    axes[0].set_ylabel(r"live $\log\Psi_0$")
    axes[0].set_title(f"Live-set progress (n_live={result.nlive})")
    axes[0].legend()

    axes[1].plot(
        iteration,
        history.logz_live,
        color="tab:green",
        label=r"$\log Z_{\rm live}$",
    )
    axes[1].set_ylabel(r"remaining $\log Z$")
    axes[1].legend()

    axes[2].plot(
        iteration,
        history.discarded_log_psi,
        color="tab:red",
        label="discarded threshold",
    )
    axes[2].set_xlabel("iteration")
    axes[2].set_ylabel(r"threshold $\log\Psi_0$")
    axes[2].legend()

    figure.tight_layout()
    return figure, axes


def plot_weight_health(result: NISMOResult) -> tuple[Any, Any]:
    """Plot sorted posterior weights and cumulative mass."""
    import matplotlib.pyplot as plt

    weights = np.sort(result.posterior_weights)[::-1]
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(weights)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("sorted contribution")
    axes[0].set_ylabel("posterior weight")
    axes[1].plot(np.cumsum(weights))
    axes[1].set_xlabel("sorted contribution")
    axes[1].set_ylabel("cumulative mass")
    figure.tight_layout()
    return figure, axes


def plot_posterior_1d(
    result: NISMOResult,
    *,
    parameter: int = 0,
    bins: int = 30,
    truth_x: NDArray[np.float64] | None = None,
    truth_density: NDArray[np.float64] | None = None,
) -> tuple[Any, Any]:
    """Plot one weighted marginal posterior and an optional known density.

    Parameters
    ----------
    result
        Stored NISMO result.
    parameter
        Zero-based parameter column.
    bins
        Weighted histogram bin count.
    truth_x, truth_density
        Optional equally shaped coordinates and normalized truth density.
    """
    import matplotlib.pyplot as plt

    if parameter < 0 or parameter >= result.all_points.shape[1]:
        raise ValueError("parameter index is out of range")
    if (truth_x is None) != (truth_density is None):
        raise ValueError("truth_x and truth_density must be provided together")
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.hist(
        result.all_points[:, parameter],
        bins=bins,
        weights=result.posterior_weights,
        density=True,
        alpha=0.45,
        label="NISMO weighted posterior",
    )
    if truth_x is not None and truth_density is not None:
        coordinates = np.asarray(truth_x, dtype=float)
        density = np.asarray(truth_density, dtype=float)
        if coordinates.shape != density.shape or coordinates.ndim != 1:
            raise ValueError(
                "truth arrays must be equally shaped one-dimensional arrays"
            )
        axis.plot(coordinates, density, label="analytic posterior")
    axis.set_xlabel(f"parameter {parameter}")
    axis.set_ylabel("density")
    axis.legend()
    figure.tight_layout()
    return figure, axis
