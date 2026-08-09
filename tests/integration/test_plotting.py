from __future__ import annotations

import matplotlib
import numpy as np
import pytest
from tests.helpers import StandardNormalProposal

from nismo import CallableModel, NISMOSampler, plot_nested_progress
from nismo.plotting import plot_posterior_1d, plot_run, plot_weight_health

matplotlib.use("Agg")
pytestmark = pytest.mark.integration


def test_plots_return_figures_without_showing() -> None:
    proposal = StandardNormalProposal()
    model = CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=lambda x: np.full(len(x), np.log(1.5)),
        log_prior_fn=proposal.log_prob,
    )
    result = NISMOSampler(
        model=model,
        importance_morph=proposal,
        n_live=8,
        rng=3,
        tie_policy="randomized_plateau",
    ).run(dlogz=0.4, max_iterations=50)
    run_figure, axes = plot_run(result)
    progress_figure, progress_axes = plot_nested_progress(result)
    weight_figure, weight_axes = plot_weight_health(result)
    posterior_figure, posterior_axis = plot_posterior_1d(
        result,
        bins=5,
        truth_x=np.linspace(-2.0, 2.0),
        truth_density=np.exp(-0.5 * np.linspace(-2.0, 2.0) ** 2) / np.sqrt(2.0 * np.pi),
    )
    assert len(axes) == 3
    assert len(progress_axes) == 3
    assert progress_axes[0].get_ylabel() == r"live $\log\Psi_0$"
    assert progress_axes[1].get_ylabel() == r"remaining $\log Z$"
    assert progress_axes[2].get_ylabel() == r"threshold $\log\Psi_0$"
    assert len(weight_axes) == 2
    assert posterior_axis.get_ylabel() == "density"
    run_figure.canvas.draw()
    progress_figure.canvas.draw()
    weight_figure.canvas.draw()
    posterior_figure.canvas.draw()
