"""Minimal MorphZ-backed analytic Gaussian evidence example."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from nismo import CallableModel, MorphProposal, NISMOSampler
from nismo.diagnostics import summarize
from nismo.plotting import plot_posterior_1d, plot_run, plot_weight_health


def build_model() -> CallableModel:
    """Return a normalized ``N(0, 2²)`` prior and ``N(theta; 0, 1)`` likelihood."""
    constant = -0.5 * np.log(2.0 * np.pi)
    return CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=lambda theta: constant - 0.5 * theta[:, 0] ** 2,
        log_prior_fn=lambda theta: (
            constant - np.log(2.0) - 0.5 * (theta[:, 0] / 2.0) ** 2
        ),
    )


def main() -> None:
    """Fit Morph, run NISMO, report diagnostics, and optionally save plots."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    posterior_variance = 0.8
    training_rng = np.random.default_rng(50)
    training = training_rng.normal(scale=np.sqrt(posterior_variance), size=(300, 1))
    importance_morph = MorphProposal.fit(
        training,
        groups=[],
        param_names=("x",),
        kde_bw="silverman",
    )
    result = NISMOSampler(
        model=build_model(),
        importance_morph=importance_morph,
        n_live=25,
        rng=12,
        proposal_batch_size=16,
    ).run(
        dlogz=0.08,
        max_iterations=400,
        max_proposals_per_replacement=5_000,
        progress=True,
    )
    expected_logz = -0.5 * np.log(10.0 * np.pi)
    diagnostics = summarize(result)
    print(f"expected logz: {expected_logz:.8f}")
    print(f"obtained logz: {result.logz:.8f} +/- {result.logzerr:.8f}")
    print(f"termination: {result.termination_reason}")
    print(f"iterations/calls: {result.niter}/{result.n_likelihood_calls}")
    print(f"posterior ESS: {diagnostics.posterior_ess:.2f}")

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        run_figure, _ = plot_run(result)
        run_figure.savefig(args.output_dir / "gaussian_run.png", dpi=140)
        weight_figure, _ = plot_weight_health(result)
        weight_figure.savefig(args.output_dir / "gaussian_weight_health.png", dpi=140)
        x = np.linspace(-3.0, 3.0, 300)
        density = np.exp(-0.5 * x**2 / posterior_variance) / np.sqrt(
            2.0 * np.pi * posterior_variance
        )
        posterior_figure, _ = plot_posterior_1d(
            result,
            bins=15,
            truth_x=x,
            truth_density=density,
        )
        posterior_figure.savefig(args.output_dir / "gaussian_posterior.png", dpi=140)


if __name__ == "__main__":
    main()
