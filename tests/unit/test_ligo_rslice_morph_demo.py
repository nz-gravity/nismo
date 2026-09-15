"""Check actual MorphZ bandwidth semantics against independent SciPy KDEs."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import gaussian_kde

pytest.importorskip("morphZ")
RUNNER_DIR = Path(__file__).resolve().parents[2] / "analysis/LIGO/fast_pp"
sys.path.insert(0, str(RUNNER_DIR))
try:
    SPEC = importlib.util.spec_from_file_location(
        "rslice_morph_demo", RUNNER_DIR / "rslice_morph_demo.py"
    )
    RUNNER = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(RUNNER)
finally:
    sys.path.pop(0)


def test_wide_morph_density_matches_scaled_scipy_kernels():
    rng = np.random.default_rng(42)
    samples = rng.normal(size=(100, 3))
    samples[:, 1] = samples[:, 0] + 0.1 * samples[:, 1]
    names = ("x", "y", "z")
    wide, factors = RUNNER.fit_wide_morph(samples, names, 2.0)
    groups = list(wide.metadata.selected_groups)
    groups += [(name,) for name in wide.metadata.single_parameters]
    points = rng.normal(size=(11, 3))
    expected = np.zeros(len(points))
    for group in groups:
        indices = [names.index(name) for name in group]
        kde = gaussian_kde(samples[:, indices].T, bw_method="silverman")
        original_covariance = kde.covariance.copy()
        kde.set_bandwidth(2 * kde.factor)
        np.testing.assert_allclose(kde.covariance, 4 * original_covariance)
        np.testing.assert_allclose([factors[name] for name in group], kde.factor)
        expected += kde.logpdf(points[:, indices].T)
    np.testing.assert_allclose(wide.log_prob(points), expected, atol=1e-10)
    assert wide.sample(10, rng).shape == (10, 3)


@pytest.mark.parametrize("scale", [0, -1, np.nan, np.inf])
def test_invalid_bandwidth_rejected(scale):
    with pytest.raises(ValueError, match="finite and positive"):
        RUNNER.fit_wide_morph(np.ones((10, 2)), ("x", "y"), scale)


def test_mass_constraints_must_be_redundant():
    from types import SimpleNamespace

    class Priors(dict):
        constraint_keys = ("mass_1", "mass_2")

    priors = Priors(
        chirp_mass=SimpleNamespace(minimum=14, maximum=20),
        mass_ratio=SimpleNamespace(minimum=0.125, maximum=1),
        mass_1=SimpleNamespace(minimum=1.001398, maximum=1000),
        mass_2=SimpleNamespace(minimum=1.001398, maximum=1000),
    )
    RUNNER.validate_mass_constraints(priors)
    priors["mass_1"].maximum = 25
    with pytest.raises(ValueError, match="prior normalization"):
        RUNNER.validate_mass_constraints(priors)


def test_waveform_overlap_preserves_phase_and_residual_sees_amplitude():
    reference = np.array([1 + 2j, 3 - 1j])
    np.testing.assert_allclose(RUNNER.waveform_metrics(reference, reference), [1, 0, 1])
    overlap, residual, amplitude = RUNNER.waveform_metrics(reference, 2 * reference)
    assert overlap == pytest.approx(1)
    assert residual == pytest.approx(np.linalg.norm(reference))
    assert amplitude == pytest.approx(2)
    assert RUNNER.waveform_metrics(reference, -reference)[0] == pytest.approx(-1)
    assert RUNNER.waveform_metrics(reference, 1j * reference)[0] == pytest.approx(0)


def test_weighted_recovery_check_uses_psd_mask_and_does_not_claim_convergence():
    from types import SimpleNamespace

    class Generator:
        def frequency_domain_strain(self, parameters):
            return {"plus": np.array([999, parameters["x"]], dtype=complex)}

    class Detector:
        frequency_mask = np.array([False, True])
        power_spectral_density_array = np.array([0, 4])
        strain_data = SimpleNamespace(duration=4)

        def get_detector_response(self, polarizations, parameters):
            return polarizations["plus"]

    model = SimpleNamespace(
        ndim=1,
        parameter_names=("x",),
        fixed_values={},
        likelihood=SimpleNamespace(
            waveform_generator=Generator(), interferometers=[Detector(), Detector()]
        ),
        log_likelihood=lambda points: -(points[:, 0] ** 2),
    )
    report = RUNNER.posterior_check(
        model,
        {"x": 1},
        np.array([[1], [2]]),
        np.array([0.9, 0.1]),
        np.array([-1, -4]),
        stage="test",
        stopping_met=False,
        draws=100,
        min_ess=1,
    )
    assert report["injected_network_snr"] == pytest.approx(np.sqrt(0.5))
    assert report["posterior_ess"] == pytest.approx(1 / 0.82)
    assert report["unique_waveforms"] == 2
    assert report["residual_snr_q05_q50_q95"] == pytest.approx([0, 0, np.sqrt(0.5)])
    assert report["overlap_screen_passed"]
    assert not report["screen_passed"]  # Perfect overlap cannot override stopping.
    assert report["diagnostic_likelihood_calls"] == 1


@pytest.mark.parametrize(
    "args",
    [["--check-draws", "0"], ["--overlap-min", "1.1"], ["--check-min-ess", "nan"]],
)
def test_invalid_check_settings_rejected(args):
    with pytest.raises(SystemExit):
        RUNNER.parse_args(["48", "--output-dir", "/tmp/test-check", *args])


@pytest.mark.parametrize(
    "bad_prior,bad_likelihood", [(False, False), (True, False), (False, True)]
)
def test_bilby_import_audits_prior_and_likelihood(
    tmp_path, monkeypatch, bad_prior, bad_likelihood
):
    import json
    from types import SimpleNamespace

    import pandas as pd

    bilby_result = pytest.importorskip("bilby.core.result")
    x = np.linspace(-1, 1, 40)
    posterior = pd.DataFrame(
        {
            "x": x,
            "log_prior": -(x**2),
            "log_likelihood": -(x**2) + 100 + (x if bad_likelihood else 0),
        }
    )
    source = SimpleNamespace(
        search_parameter_keys=["x"],
        priors={"x": "wrong" if bad_prior else "uniform"},
        posterior=posterior,
        log_noise_evidence=-100,
        sampler="dynesty",
        log_evidence=-10,
        log_evidence_err=0.1,
    )
    model = SimpleNamespace(
        parameter_names=("x",),
        log_prior=lambda rows: -(rows[:, 0] ** 2),
        log_likelihood=lambda rows: -(rows[:, 0] ** 2),
    )
    monkeypatch.setattr(bilby_result, "read_in_result", lambda path: source)
    path = tmp_path / "input.json"
    path.write_text(
        json.dumps(
            {"sampler_kwargs": {"nlive": 1000, "sample": "rslice"}, "sampling_time": 10}
        )
    )
    arguments = dict(
        path=path,
        model=model,
        priors={"x": "uniform"},
        args=SimpleNamespace(index=0),
        outdir=tmp_path,
        fingerprint="test",
    )
    if bad_prior or bad_likelihood:
        with pytest.raises(ValueError, match=r"differ|audit failed"):
            RUNNER.import_bilby_posterior(**arguments)
        assert not (tmp_path / "rough.npz").exists()
    else:
        samples, metadata = RUNNER.import_bilby_posterior(**arguments)
        assert metadata["nlive"] == 1000
        assert metadata["reached_target"] is None
        assert samples.shape == (40, 1)
        with np.load(tmp_path / "rough.npz") as archive:
            np.testing.assert_allclose(archive["log_likelihood"], -(x**2), atol=1e-12)
            np.testing.assert_allclose(archive["weights"].sum(), 1)
