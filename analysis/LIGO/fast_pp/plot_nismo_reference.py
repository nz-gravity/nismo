#!/usr/bin/env python3
"""Compare saved weighted NISMO posteriors with a saved reference; no sampling.

uv run --extra lvk python analysis/LIGO/fast_pp/plot_nismo_reference.py \
  --reference RUN/rough.npz --runs RUN1 RUN2 --output-dir OUTPUT --index 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, NullLocator
from scipy.ndimage import gaussian_filter

LABELS = [
    r"$\mathcal{M}\ [M_\odot]$",
    r"$q$",
    r"$a_1$",
    r"$a_2$",
    r"$\theta_1\ [rad]$",
    r"$\theta_2\ [rad]$",
    r"$\phi_{12}\ [rad]$",
    r"$\phi_{JL}\ [rad]$",
]


def history_ess(history):
    """Total Kish ESS from cumulative dead weights plus live-set weight moments."""
    dead_log_weights = history["discarded_log_psi"] + history["log_delta_x"]
    dead_log_squared_sum = np.logaddexp.accumulate(2 * dead_log_weights)
    live_log_squared_sum = 2 * history["logz_live"] - np.log(history["live_ess"])
    return np.exp(
        2 * history["logz_total"]
        - np.logaddexp(dead_log_squared_sum, live_log_squared_sum)
    )


def plot_corner(reference, rw, points, weights, truth, ranges, indices, title, path):
    n = len(indices)
    fig, axes = plt.subplots(n, n, figsize=(2 * n, 2 * n), squeeze=False)
    cmap = LinearSegmentedColormap.from_list("nismo", ["white", "#C17D11"])
    for row, i in enumerate(indices):
        for col, j in enumerate(indices):
            ax = axes[row, col]
            if col > row:
                ax.set_visible(False)
                continue
            xbins = np.linspace(*ranges[j], 33)
            if row == col:
                for x, w, color, lw in [
                    (reference, rw, "#333333", 1.5),
                    (points, weights, "#C17D11", 1.3),
                ]:
                    ax.hist(
                        x[:, i],
                        bins=xbins,
                        weights=w,
                        density=True,
                        histtype="step",
                        color=color,
                        lw=lw,
                    )
                ax.set_yticks([])
            else:
                ybins = np.linspace(*ranges[i], 33)
                density, _, _ = np.histogram2d(
                    points[:, j], points[:, i], bins=(xbins, ybins), weights=weights
                )
                ax.pcolormesh(
                    xbins,
                    ybins,
                    density.T,
                    cmap=cmap,
                    vmin=0,
                    shading="flat",
                    rasterized=True,
                )
                ref_hist, _, _ = np.histogram2d(
                    reference[:, j], reference[:, i], bins=(xbins, ybins), weights=rw
                )
                ref_hist = gaussian_filter(ref_hist, 1, mode="constant")
                ranked = np.sort(ref_hist.ravel())[::-1]
                cdf = np.cumsum(ranked) / ranked.sum()
                levels = sorted(
                    set(
                        ranked[min(np.searchsorted(cdf, p), len(ranked) - 1)]
                        for p in [0.68, 0.95]
                    )
                )
                ax.contour(
                    (xbins[1:] + xbins[:-1]) / 2,
                    (ybins[1:] + ybins[:-1]) / 2,
                    ref_hist.T,
                    levels=levels,
                    colors="#333333",
                    linewidths=0.9,
                )
                ax.axhline(truth[i], color="#555555", ls=":", lw=0.8)
                ax.plot(truth[j], truth[i], "+", color="#333333", ms=5)
                ax.set_ylim(ranges[i])
                ax.yaxis.set_major_locator(MaxNLocator(3, prune="both"))
            ax.axvline(truth[j], color="#555555", ls=":", lw=0.8)
            ax.set_xlim(ranges[j])
            ax.xaxis.set_major_locator(MaxNLocator(3, prune="both"))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_minor_locator(NullLocator())
            ax.tick_params(labelsize=9, length=3)
            if row == n - 1:
                ax.set_xlabel(LABELS[j], fontsize=12)
            else:
                ax.set_xticklabels([])
            if col == 0 and row:
                ax.set_ylabel(LABELS[i], fontsize=12)
            elif col:
                ax.set_yticklabels([])
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=14 if n == 4 else 19, y=0.985)
    handles = [
        Line2D([], [], color="#333333", label="Dynesty-4000 reference"),
        Line2D([], [], color="#C17D11", label="NISMO"),
        Line2D([], [], color="#555555", ls=":", label="Injected truth"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    fig.text(
        0.5,
        0.014,
        "Reference: 68/95% contours, 1-bin smoothing. "
        "NISMO: weighted histograms, no smoothing.",
        ha="center",
        fontsize=8 if n == 4 else 11,
    )
    fig.subplots_adjust(
        left=0.09, right=0.985, bottom=0.085, top=0.85, hspace=0.06, wspace=0.06
    )
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix("." + ext), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--runs", required=True, nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.reference) as d:
        reference, rw, names = d["samples"], d["weights"], tuple(d["parameter_names"])
    truth = (
        pd.read_csv(Path(__file__).with_name("injections.csv"), index_col=0)
        .iloc[args.index][list(names)]
        .to_numpy()
    )
    ref_info = json.loads(args.reference.with_suffix(".json").read_text())
    if ref_info["index"] != args.index or not np.isclose(rw.sum(), 1):
        raise ValueError("reference injection or weights mismatch")
    runs = []
    for directory in args.runs:
        info = json.loads((directory / "summary.json").read_text())
        if (
            info["rough"]["index"] != args.index
            or info["fingerprint"] != ref_info["fingerprint"]
        ):
            raise ValueError("injection mismatch")
        with np.load(directory / "nismo/weighted_samples.npz") as d:
            if tuple(d["parameter_names"]) != names:
                raise ValueError("coordinate mismatch")
            points, weights, live = d["samples"], d["posterior_weights"], d["is_live"]
        if not np.isclose(weights.sum(), 1):
            raise ValueError("unnormalized weights")
        runs.append((directory, info, points, weights, live))
    lo = min(
        truth[0],
        np.quantile(reference[:, 0], 0.001),
        *[np.quantile(x[:, 0], 0.001) for _, _, x, _, _ in runs],
    )
    hi = max(
        truth[0],
        np.quantile(reference[:, 0], 0.999),
        *[np.quantile(x[:, 0], 0.999) for _, _, x, _, _ in runs],
    )
    ranges = [
        (lo, hi),
        (0.125, 1),
        (0, 0.99),
        (0, 0.99),
        (0, np.pi),
        (0, np.pi),
        (0, 2 * np.pi),
        (0, 2 * np.pi),
    ]
    report = []
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for run_index, (directory, info, points, weights, live) in enumerate(runs):
        color = ["#256CA6", "#C17D11", "#555555"][run_index % 3]
        label = info["rough"].get("training_label", directory.name)
        status = "stopping target met" if info["nismo"]["success"] else "incomplete"
        elapsed = info["nismo"]["seconds"]
        ess = 1 / np.sum(weights**2)
        mass = weights[live].sum()
        for indices, suffix in [
            (list(range(8)), "full"),
            (list(range(4)), "mass_spin"),
        ]:
            plot_corner(
                reference,
                rw,
                points,
                weights,
                truth,
                ranges,
                indices,
                f"Seed {args.index}: trained on {label}\n"
                f"NISMO {status}, {elapsed:.0f} s; weight ESS = {ess:.1f}",
                args.output_dir / f"{directory.name}_{suffix}",
            )
        with np.load(directory / "nismo/run_history.npz") as h:
            total_ess = history_ess(h)
            np.testing.assert_allclose(total_ess[-1], ess, rtol=1e-8)
            axes[0].plot(h["elapsed_seconds"], total_ess, color=color, label=label)
            axes[1].plot(
                h["elapsed_seconds"], 100 * h["remaining_fraction"], color=color
            )
            proposals = h["proposals"]
            window = 50
            acceptance = window / np.convolve(proposals, np.ones(window), mode="valid")
            axes[2].semilogy(
                h["elapsed_seconds"][window - 1 :], acceptance, color=color
            )
            finite = h["discarded_log_psi"][np.isfinite(h["discarded_log_psi"])]
            monotone = bool(np.all(np.diff(finite) >= 0))
        report.append(
            dict(
                source=label,
                posterior_ess=ess,
                live_weight_fraction=float(mass),
                live_points=int(live.sum()),
                ess_upper_bound_from_live_mass=float(live.sum() / mass**2),
                top5_weight_fraction=float(np.sort(weights)[-5:].sum()),
                finite_thresholds_monotone=monotone,
            )
        )
    for ax, ylabel in zip(
        axes,
        [
            "Total posterior weight ESS",
            "Weight in live points (%)",
            "Replacements / proposals (50-step window)",
        ],
        strict=True,
    ):
        ax.set_xlabel("NISMO elapsed time (s)")
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].set_ylim(0, 102)
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False
    )
    fig.subplots_adjust(top=0.82, bottom=0.16, wspace=0.32)
    for ext in ["png", "pdf"]:
        fig.savefig(args.output_dir / f"weight_history.{ext}", dpi=160)
    plt.close(fig)
    (args.output_dir / "weight_analysis.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
