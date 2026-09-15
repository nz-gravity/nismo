#!/usr/bin/env python3
"""Rough rslice posterior -> broadened, fixed Morph nested importance sampling.

Run from the repository root with ``uv run --extra lvk python
analysis/LIGO/fast_pp/rslice_morph_demo.py 48 --output-dir /tmp/lvk-rslice``.
The existing eight-dimensional PP injection/likelihood is used unchanged.
This is a pilot; agreement with its own training run is not accuracy validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
from nismo_computation import build_model, fixed_parameter_values

from nismo import MorphProposal, NISMOSampler, MORWalkSettings
from nismo.diagnostics import summarize


def fit_wide_morph(samples, names, bandwidth_scale):
    """Multiply each selected KDE's Silverman kernel width, not data spread."""
    if not np.isfinite(bandwidth_scale) or bandwidth_scale <= 0:
        raise ValueError("bandwidth_scale must be finite and positive")
    selected = MorphProposal.fit(samples, param_names=names, morph_type="2_group")
    groups = selected.metadata.selected_groups
    dimensions = dict.fromkeys(names, 1)
    for group in groups:
        dimensions.update(dict.fromkeys(group, len(group)))
    # scipy gaussian_kde Silverman factor: (n * (d + 2) / 4)^(-1/(d + 4)).
    factors = {
        name: float(bandwidth_scale * (len(samples) * (d + 2) / 4) ** (-1 / (d + 4)))
        for name, d in dimensions.items()
    }
    proposal = MorphProposal.fit(
        samples,
        param_names=names,
        groups=[[list(group), 1.0] for group in groups],
        kde_bw=factors,
    )
    return proposal, factors


def write_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    temporary.replace(path)


def model_fingerprint(likelihood, priors):
    """Identify the exact noise realization, PSD, priors and waveform settings."""
    digest = hashlib.sha256(repr(priors).encode())
    digest.update(repr(likelihood.waveform_generator.waveform_arguments).encode())
    for ifo in likelihood.interferometers:
        digest.update(ifo.name.encode())
        digest.update(str(ifo.strain_data.start_time).encode())
        for values in (
            ifo.frequency_array,
            ifo.frequency_domain_strain,
            ifo.power_spectral_density_array,
            ifo.frequency_mask,
        ):
            digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def validate_mass_constraints(priors):
    """Reject changes that would invalidate the sampled-coordinate prior density."""
    if set(priors.constraint_keys) != {"mass_1", "mass_2"}:
        raise ValueError(
            "this demo requires only the existing component-mass constraints"
        )
    mc = priors["chirp_mass"]
    q = priors["mass_ratio"]
    for chirp_mass in (mc.minimum, mc.maximum):
        for mass_ratio in (q.minimum, q.maximum):
            m1 = chirp_mass * (1 + mass_ratio) ** 0.2 / mass_ratio**0.6
            for key, mass in (("mass_1", m1), ("mass_2", m1 * mass_ratio)):
                if not priors[key].minimum <= mass <= priors[key].maximum:
                    raise ValueError(
                        "non-redundant mass constraints need prior normalization"
                    )


def run_rough(model, priors, args, outdir, fingerprint):
    import dynesty
    from dynesty.utils import resample_equal

    started = time.perf_counter()
    rng = np.random.default_rng(args.rough_seed)
    names = model.parameter_names
    # The mass constraints in pp.prior are redundant throughout this Mc/q box.
    # Use the same normalized sampled-coordinate prior in both stages.
    sampler = dynesty.NestedSampler(
        lambda x: float(model.log_likelihood(x[None, :])[0]),
        lambda u: np.asarray(priors.rescale(names, u)),
        model.ndim,
        nlive=args.nlive,
        sample="rslice",
        bound="multi",
        slices=args.slices,
        rstate=rng,
        periodic=[
            i for i, name in enumerate(names) if priors[name].boundary == "periodic"
        ]
        or None,
    )
    # Capture pre-final-live stopping so a call cap cannot masquerade as convergence.
    for _ in sampler.sample(dlogz=args.rough_dlogz, maxcall=args.rough_maxcall):
        pass
    dead = sampler.results
    delta_logz = float(
        np.logaddexp(0, np.max(sampler.live_logl) + dead.logvol[-1] - dead.logz[-1])
    )
    for _ in sampler.add_live_points():
        pass
    result = sampler.results
    weights = np.exp(result.logwt - result.logz[-1])
    weights /= weights.sum()
    training = resample_equal(result.samples, weights, rstate=rng)
    np.savez_compressed(
        outdir / "rough.npz",
        samples=result.samples,
        log_likelihood=result.logl,
        weights=weights,
        training=training,
        parameter_names=names,
    )
    info = dict(
        index=args.index,
        fingerprint=fingerprint,
        nlive=args.nlive,
        seed=args.rough_seed,
        sample="rslice",
        slices=args.slices,
        dlogz=args.rough_dlogz,
        maxcall=args.rough_maxcall,
        final_remaining_dlogz=delta_logz,
        reached_target=delta_logz < args.rough_dlogz,
        logz=float(result.logz[-1]),
        logzerr=float(result.logzerr[-1]),
        n_likelihood_calls=int(np.sum(result.ncall)),
        seconds=time.perf_counter() - started,
        posterior_ess=float(1 / np.sum(weights**2)),
        n_training=len(training),
        n_unique_training=len(np.unique(training, axis=0)),
    )
    write_json(outdir / "rough.json", info)
    return training, info


def import_bilby_posterior(path, model, priors, args, outdir, fingerprint):
    """Audit a completed Bilby posterior and preserve source settings separately."""
    from bilby.core.result import read_in_result
    from nismo_computation import audit_posterior_contract, training_samples

    started = time.perf_counter()
    result = read_in_result(path)
    if tuple(result.search_parameter_keys) != model.parameter_names:
        raise ValueError("Bilby search coordinates differ from the current demo")
    if set(result.priors) != set(priors) or any(
        str(result.priors[key]) != str(priors[key]) for key in priors
    ):
        raise ValueError("Bilby priors/fixed values differ from the current demo")
    samples = training_samples(result, model.parameter_names)
    audit = audit_posterior_contract(
        model=model, result=result, samples=samples, n_points=32
    )
    if (
        audit["max_abs_log_prior_difference"] > 1e-6
        or audit["max_abs_log_likelihood_residual"] > 1e-6
    ):
        raise ValueError(f"Bilby posterior density audit failed: {audit}")
    offset = audit["log_likelihood_offset"]
    if not min(abs(offset), abs(offset - float(result.log_noise_evidence))) < 1e-6:
        raise ValueError("stored likelihood offset is neither zero nor noise evidence")
    stored_logl = result.posterior["log_likelihood"].to_numpy(dtype=float)
    if not np.all(np.isfinite(stored_logl)):
        raise ValueError("Bilby posterior has nonfinite likelihoods")
    np.savez_compressed(
        outdir / "rough.npz",
        samples=samples,
        training=samples,
        parameter_names=model.parameter_names,
        log_likelihood=stored_logl + offset,
        weights=np.full(len(samples), 1 / len(samples)),
    )
    raw = json.loads(path.read_text())
    kwargs = raw.get("sampler_kwargs", {})
    label = f"{result.sampler} {kwargs.get('sample', '')}".strip()
    source_nlive = kwargs.get("nlive")
    if source_nlive is not None:
        label += f" ({source_nlive} live)"
    source_seconds = raw.get("sampling_time")
    source_calls = raw.get("num_likelihood_evaluations")
    # A result file has posterior samples, but need not retain termination reason.
    info = dict(
        index=args.index,
        fingerprint=fingerprint,
        source_type="bilby_posterior",
        source_path=str(path.resolve()),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        training_label=label,
        sampler=result.sampler,
        sample=kwargs.get("sample"),
        nlive=source_nlive,
        dlogz=kwargs.get("dlogz"),
        reached_target=None,
        termination_note="Not certified by the result JSON",
        logz=float(result.log_evidence),
        logzerr=float(result.log_evidence_err),
        seconds=source_seconds,
        n_likelihood_calls=source_calls,
        n_training=len(samples),
        posterior_ess=float(len(samples)),
        ess_note="Equal-weight row count; not an autocorrelation/mixing ESS",
        n_unique_training=len(np.unique(samples, axis=0)),
        posterior_audit=audit,
        import_seconds=time.perf_counter() - started,
    )
    write_json(outdir / "rough.json", info)
    return samples, info


def waveform_metrics(reference, prediction):
    """Real normalized inner product and residual norm of whitened FD signals.

    Inputs concatenate detectors after weighting each active frequency bin by
    sqrt(4 / (duration * PSD)), as in Bilby's noise-weighted inner product.
    No time/phase/amplitude maximization: these extrinsics are fixed in this demo.
    """
    reference = np.asarray(reference, dtype=complex)
    prediction = np.asarray(prediction, dtype=complex)
    if reference.shape != prediction.shape or not np.all(np.isfinite(prediction)):
        raise ValueError("waveforms must have matching shapes and finite values")
    ref_power = float(np.vdot(reference, reference).real)
    pred_power = float(np.vdot(prediction, prediction).real)
    if (
        not np.isfinite(ref_power)
        or ref_power <= 0
        or not np.isfinite(pred_power)
        or pred_power <= 0
    ):
        raise ValueError("waveform norms must be finite and positive")
    overlap = float(
        np.vdot(reference, prediction).real / np.sqrt(ref_power * pred_power)
    )
    residual = float(np.linalg.norm(prediction - reference))
    return (
        float(np.clip(overlap, -1, 1)),
        residual,
        float(np.sqrt(pred_power / ref_power)),
    )


def posterior_check(
    model,
    truth,
    points,
    weights,
    log_likelihood,
    *,
    stage,
    stopping_met,
    draws=128,
    overlap_min=0.99,
    min_ess=100,
):
    """A bounded injection-recovery screen, not a posterior convergence test."""
    started = time.perf_counter()
    points, weights = np.asarray(points), np.asarray(weights, dtype=float)
    log_likelihood = np.asarray(log_likelihood)
    if (
        points.ndim != 2
        or points.shape[1] != model.ndim
        or weights.shape != (len(points),)
        or log_likelihood.shape != weights.shape
        or not np.all(np.isfinite(points))
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0)
        or weights.sum() <= 0
    ):
        raise ValueError("invalid weighted posterior for recovery check")
    weights = weights / weights.sum()
    # Deterministic systematic quadrature of the weighted empirical posterior.
    # Repeated indices are evaluated once; this does NOT increase posterior ESS.
    indices = np.searchsorted(np.cumsum(weights), (np.arange(draws) + 0.5) / draws)
    unique, inverse = np.unique(indices, return_inverse=True)
    likelihood = model.likelihood
    generator = likelihood.waveform_generator

    def whitened_signal(parameters):
        polarizations = generator.frequency_domain_strain(parameters)
        if polarizations is None:
            raise ValueError("waveform generation failed in recovery check")
        parts = []
        for ifo in likelihood.interferometers:
            mask = ifo.frequency_mask
            psd = ifo.power_spectral_density_array[mask]
            if np.any(~np.isfinite(psd)) or np.any(psd <= 0):
                raise ValueError("active detector PSD must be finite and positive")
            response = ifo.get_detector_response(polarizations, parameters)[mask]
            parts.append(response * np.sqrt(4 / (ifo.strain_data.duration * psd)))
        return np.concatenate(parts)

    reference = whitened_signal(truth)
    values = []
    for index in unique:
        parameters = dict(zip(model.parameter_names, points[index], strict=True))
        parameters.update(model.fixed_values)
        values.append(waveform_metrics(reference, whitened_signal(parameters)))
    values = np.asarray(values)[inverse]
    ess = float(1 / np.sum(weights**2))
    overlap_q = np.quantile(values[:, 0], [0.05, 0.5, 0.95]).tolist()
    residual_q = np.quantile(values[:, 1], [0.05, 0.5, 0.95]).tolist()
    truth_theta = np.array([[truth[name] for name in model.parameter_names]])
    truth_logl = float(model.log_likelihood(truth_theta)[0])
    overlap_ok = overlap_q[0] >= overlap_min
    ess_ok = ess >= min_ess
    result = dict(
        stage=stage,
        meaning="Heuristic recovery screen; not proof of convergence",
        stopping_target_met=None if stopping_met is None else bool(stopping_met),
        posterior_ess=ess,
        min_ess=min_ess,
        ess_screen_passed=ess_ok,
        overlap_q05_q50_q95=overlap_q,
        overlap_min=overlap_min,
        overlap_screen_passed=overlap_ok,
        estimated_mass_above_overlap_min=float(np.mean(values[:, 0] >= overlap_min)),
        residual_snr_q05_q50_q95=residual_q,
        amplitude_ratio_q05_q50_q95=np.quantile(
            values[:, 2], [0.05, 0.5, 0.95]
        ).tolist(),
        injected_network_snr=float(np.linalg.norm(reference)),
        truth_log_likelihood=truth_logl,
        best_stored_log_likelihood=float(np.max(log_likelihood)),
        truth_minus_best_log_likelihood=truth_logl - float(np.max(log_likelihood)),
        screen_passed=bool(stopping_met and ess_ok and overlap_ok),
        diagnostic_draws=draws,
        unique_waveforms=len(unique),
        integration="deterministic systematic weighted posterior quadrature",
        maximized_over_extrinsics=False,
        diagnostic_likelihood_calls=1,
        diagnostic_signal_evaluations=1 + len(unique),
        seconds=time.perf_counter() - started,
    )
    stopping_label = (
        "UNKNOWN" if stopping_met is None else ("met" if stopping_met else "NOT MET")
    )
    flags = [
        f"screen={'PASS' if result['screen_passed'] else 'FAIL'}",
        f"stopping={stopping_label}",
        f"weight ESS={ess:.1f} ({'pass' if ess_ok else 'LOW'}; minimum {min_ess:g})",
        f"overlap 5/50/95%={overlap_q[0]:.6f}/{overlap_q[1]:.6f}/{overlap_q[2]:.6f}",
        f"overlap screen={'PASS' if overlap_ok else 'FAIL'} (5% >= {overlap_min:g})",
        f"residual SNR 5/50/95%={residual_q[0]:.2f}/"
        f"{residual_q[1]:.2f}/{residual_q[2]:.2f}",
        f"truth-minus-best logL={result['truth_minus_best_log_likelihood']:+.2f}",
    ]
    print(
        f"[{stage} recovery check] "
        + "; ".join(flags)
        + "; convergence NOT established by this screen.",
        flush=True,
    )
    return result


def check_saved_stage(model, truth, archive_path, *, stage, stopping_met, args):
    with np.load(archive_path) as archive:
        if tuple(archive["parameter_names"]) != model.parameter_names:
            raise ValueError("saved parameter order differs from recovery model")
        return posterior_check(
            model,
            truth,
            archive["samples"],
            archive["weights" if "weights" in archive else "posterior_weights"],
            archive["log_likelihood"],
            stage=stage,
            stopping_met=stopping_met,
            draws=args.check_draws,
            overlap_min=args.overlap_min,
            min_ess=args.check_min_ess,
        )


def save_corner_plots(outdir, index, rough_dir=None):
    """Plot stored weighted samples against injection truth without rerunning PE."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator, NullLocator

    report = json.loads((outdir / "summary.json").read_text())
    if report["rough"]["index"] != index:
        raise ValueError("plot index differs from the saved injection index")
    if rough_dir is None:
        rough_dir = (
            outdir if (outdir / "rough.npz").exists() else Path(report["rough_source"])
        )
    with np.load(rough_dir / "rough.npz") as data:
        before = data["samples"].copy()
        before_w = data["weights"].copy()
        names = tuple(data["parameter_names"])
    rough_info = json.loads((rough_dir / "rough.json").read_text())
    if rough_info["fingerprint"] != report["fingerprint"]:
        raise ValueError("rough and NISMO model fingerprints differ")
    with np.load(outdir / "nismo" / "weighted_samples.npz") as data:
        if tuple(data["parameter_names"]) != names:
            raise ValueError("rough and NISMO coordinate orders differ")
        after = data["samples"].copy()
        after_w = data["posterior_weights"].copy()
    truth = pd.read_csv(Path(__file__).with_name("injections.csv"), index_col=0)
    truth = truth.iloc[index][list(names)].to_numpy(dtype=float)
    datasets = [(before, before_w / before_w.sum()), (after, after_w / after_w.sum())]
    labels = [
        r"$\mathcal{M}\ [M_\odot]$",
        r"$q$",
        r"$a_1$",
        r"$a_2$",
        r"$\theta_1\ [\mathrm{rad}]$",
        r"$\theta_2\ [\mathrm{rad}]$",
        r"$\phi_{12}\ [\mathrm{rad}]$",
        r"$\phi_{JL}\ [\mathrm{rad}]$",
    ]
    expected = (
        "chirp_mass",
        "mass_ratio",
        "a_1",
        "a_2",
        "tilt_1",
        "tilt_2",
        "phi_12",
        "phi_jl",
    )
    if names != expected:
        raise ValueError("unexpected parameter order for the demo corner labels")
    # Full prior axes for spins and angles; a shared posterior/truth mass zoom.
    mc_limits = [truth[0], truth[0]]
    summaries = []
    for stage, (points, weights) in zip(
        ("rough", "nismo_partial"), datasets, strict=True
    ):
        if not np.all(np.isfinite(points)) or np.any(weights < 0):
            raise ValueError("invalid stored samples or weights")
        for i, name in enumerate(names):
            order = np.argsort(points[:, i])
            cumulative = np.cumsum(weights[order])
            quantiles = np.interp(
                [0.001, 0.05, 0.5, 0.95, 0.999], cumulative, points[order, i]
            )
            if i == 0:
                mc_limits[0] = min(mc_limits[0], quantiles[0])
                mc_limits[1] = max(mc_limits[1], quantiles[-1])
            summaries.append(
                dict(
                    stage=stage,
                    parameter=name,
                    truth=truth[i],
                    q05=quantiles[1],
                    median=quantiles[2],
                    q95=quantiles[3],
                    truth_cdf=weights[points[:, i] <= truth[i]].sum(),
                )
            )
    pad = 0.08 * (mc_limits[1] - mc_limits[0])
    ranges = [
        (mc_limits[0] - pad, mc_limits[1] + pad),
        (0.125, 1),
        (0, 0.99),
        (0, 0.99),
        (0, np.pi),
        (0, np.pi),
        (0, 2 * np.pi),
        (0, 2 * np.pi),
    ]
    bins = [np.linspace(lo, hi, 37) for lo, hi in ranges]
    colors = ["#256CA6", "#C17D11"]
    ess = [1 / np.sum(w**2) for _, w in datasets]
    plots = outdir / "plots"
    plots.mkdir(exist_ok=True)
    pd.DataFrame(summaries).to_csv(plots / "weighted_marginals.csv", index=False)

    def draw_corner(figure, points, weights, indices, color):
        n = len(indices)
        axes = figure.subplots(n, n, squeeze=False)
        cmap = LinearSegmentedColormap.from_list("posterior", ["white", color])
        for row, i in enumerate(indices):
            for col, j in enumerate(indices):
                ax = axes[row, col]
                if row < col:
                    ax.set_visible(False)
                    continue
                if row == col:
                    ax.hist(
                        points[:, i],
                        bins=bins[i],
                        weights=weights,
                        density=True,
                        histtype="step",
                        color=color,
                        linewidth=1.5,
                    )
                    ax.set_yticks([])
                else:
                    hist, xedges, yedges = np.histogram2d(
                        points[:, j],
                        points[:, i],
                        bins=(bins[j], bins[i]),
                        weights=weights,
                    )
                    ax.pcolormesh(
                        xedges,
                        yedges,
                        hist.T,
                        cmap=cmap,
                        vmin=0,
                        shading="flat",
                        rasterized=True,
                    )
                    ax.axhline(truth[i], color="#333333", linestyle="--", linewidth=0.8)
                    ax.plot(truth[j], truth[i], "+", color="#111111", ms=6, mew=1)
                    ax.set_ylim(ranges[i])
                ax.axvline(truth[j], color="#333333", linestyle="--", linewidth=0.8)
                ax.set_xlim(ranges[j])
                ax.xaxis.set_major_locator(MaxNLocator(3, prune="both"))
                ax.xaxis.set_minor_locator(NullLocator())
                ax.yaxis.set_minor_locator(NullLocator())
                if row != col:
                    ax.yaxis.set_major_locator(MaxNLocator(3, prune="both"))
                ax.tick_params(labelsize=9, length=3)
                if row == n - 1:
                    ax.set_xlabel(labels[j], fontsize=12)
                else:
                    ax.set_xticklabels([])
                if col == 0 and row != col:
                    ax.set_ylabel(labels[i], fontsize=12)
                elif col:
                    ax.set_yticklabels([])
                ax.spines[["top", "right"]].set_visible(False)
        figure.subplots_adjust(
            left=0.085, bottom=0.08, right=0.985, top=0.91, wspace=0.06, hspace=0.06
        )

    training_label = report["rough"].get("training_label", "rough rslice")
    titles = [
        f"Before: {training_label} | weight ESS = {ess[0]:.1f}",
        f"After: NISMO, 2x bandwidth | INCOMPLETE | posterior ESS = {ess[1]:.1f}",
    ]
    # Read the actual width so plot-only also works on the bandwidth control.
    proposal = json.loads((outdir / "proposal.json").read_text())
    titles[1] = titles[1].replace("2x", f"{proposal['bandwidth_scale']:g}x")
    if report["nismo"]["success"]:
        titles[1] = titles[1].replace("INCOMPLETE", "stopping target reached")
    for k, ((points, weights), name) in enumerate(
        zip(datasets, ("corner_before", "corner_after"), strict=True)
    ):
        fig = plt.figure(figsize=(15, 15))
        draw_corner(fig, points, weights, list(range(8)), colors[k])
        fig.suptitle(f"Injection {index} — {titles[k]}", fontsize=19, y=0.985)
        fig.text(
            0.5,
            0.951,
            "Dashed lines / crosses: injected truth · Weighted histograms, "
            "no smoothing · Extrinsics fixed",
            ha="center",
            fontsize=11,
        )
        fig.text(
            0.985,
            0.018,
            "Shared axes; chirp mass zoom includes truth. "
            "Angles use their native periodic coordinates.",
            ha="right",
            fontsize=10,
        )
        for suffix in ("png", "pdf"):
            fig.savefig(plots / f"{name}.{suffix}", dpi=150)
        plt.close(fig)
    fig = plt.figure(figsize=(15, 7.5))
    subs = fig.subfigures(1, 2, wspace=0.04)
    for k, (points, weights) in enumerate(datasets):
        draw_corner(subs[k], points, weights, list(range(4)), colors[k])
        subs[k].suptitle(titles[k], fontsize=13, y=0.98)
    fig.legend(
        handles=[
            Line2D([], [], color="#333333", linestyle="--", label="Injected truth")
        ],
        loc="lower center",
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
    )
    for suffix in ("png", "pdf"):
        fig.savefig(plots / f"corner_comparison.{suffix}", dpi=160)
    plt.close(fig)
    print(f"Saved weighted corner plots to {plots}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--rough-from", type=Path, help="reuse rough.npz/json from this run directory"
    )
    parser.add_argument(
        "--nlive",
        type=int,
        default=100,
        help="NISMO live points (also used by fresh rslice)",
    )
    parser.add_argument("--rough-seed", type=int, default=1234)
    parser.add_argument("--nismo-seed", type=int, default=5678)
    parser.add_argument("--rough-dlogz", type=float, default=1.0)
    parser.add_argument("--slices", type=int, default=3)
    parser.add_argument("--rough-maxcall", type=int, default=100000)
    parser.add_argument("--bandwidth-scale", type=float, default=2.0)
    parser.add_argument("--dlogz", type=float, default=0.1)
    parser.add_argument("--max-seconds", type=float, default=600, help="NISMO time cap")
    parser.add_argument("--max-calls", type=int, default=200000)
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="plot an existing output directory without sampling",
    )
    parser.add_argument(
        "--checks-only",
        action="store_true",
        help="check saved posteriors without sampling",
    )
    parser.add_argument(
        "--bilby-result",
        type=Path,
        help="use an audited Bilby posterior instead of running rslice",
    )
    parser.add_argument(
        "--training-only",
        action="store_true",
        help="import/check a Bilby result without running NISMO",
    )
    parser.add_argument("--check-draws", type=int, default=128)
    parser.add_argument("--overlap-min", type=float, default=0.99)
    parser.add_argument("--check-min-ess", type=float, default=100)
    args = parser.parse_args(argv)
    if args.bilby_result and (args.rough_from or args.checks_only or args.plots_only):
        parser.error("--bilby-result cannot combine with reuse/checks-only/plots-only")
    if args.training_only and not args.bilby_result:
        parser.error("--training-only requires --bilby-result")
    if args.checks_only and args.plots_only:
        parser.error("choose either --checks-only or --plots-only")
    if not np.isfinite(args.overlap_min) or not -1 <= args.overlap_min <= 1:
        parser.error("--overlap-min must be between -1 and 1")
    for key in (
        "nlive",
        "slices",
        "rough_maxcall",
        "bandwidth_scale",
        "rough_dlogz",
        "dlogz",
        "max_seconds",
        "max_calls",
        "check_draws",
        "check_min_ess",
    ):
        if not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be finite and positive")
    if args.nlive < 20:
        parser.error("--nlive must be at least 20 for this eight-dimensional demo")
    return args


def main(argv=None):
    from pp_setup import load_simulation
    from threadpoolctl import threadpool_limits

    args = parse_args(argv)
    outdir = args.output_dir.resolve()
    if args.plots_only:
        save_corner_plots(outdir, args.index, args.rough_from)
        return 0
    if not args.checks_only:
        outdir.mkdir(parents=True, exist_ok=False)
        write_json(outdir / "settings.json", vars(args))
    elif not (outdir / "summary.json").is_file():
        raise FileNotFoundError(outdir / "summary.json")
    started = time.perf_counter()
    likelihood, priors, _, _, _ = load_simulation(
        args.index, output_root=outdir, plot_data=False
    )
    validate_mass_constraints(priors)
    names = tuple(priors.non_fixed_keys)
    model = build_model(
        likelihood=likelihood,
        priors=priors,
        names=names,
        fixed_values=fixed_parameter_values(priors),
    )
    fingerprint = model_fingerprint(likelihood, priors)
    import pandas as pd

    truth = (
        pd.read_csv(Path(__file__).with_name("injections.csv"), index_col=0)
        .iloc[args.index]
        .to_dict()
    )
    checks = {"fingerprint": fingerprint}
    if args.checks_only:
        saved = json.loads((outdir / "summary.json").read_text())
        if saved["fingerprint"] != fingerprint or saved["rough"]["index"] != args.index:
            raise ValueError("saved run does not match reconstructed injection/model")
        rough_dir = args.rough_from or (
            outdir if (outdir / "rough.npz").exists() else Path(saved["rough_source"])
        )
        rough_info = json.loads((rough_dir / "rough.json").read_text())
        if rough_info["fingerprint"] != fingerprint:
            raise ValueError("saved rough posterior does not match model")
        with threadpool_limits(limits=1):
            checks["rslice"] = check_saved_stage(
                model,
                truth,
                rough_dir / "rough.npz",
                stage="rslice",
                stopping_met=rough_info["reached_target"],
                args=args,
            )
            checks["nismo"] = check_saved_stage(
                model,
                truth,
                outdir / "nismo" / "weighted_samples.npz",
                stage="nismo",
                stopping_met=saved["nismo"]["success"],
                args=args,
            )
        write_json(outdir / "stage_checks.json", checks)
        return 0
    with threadpool_limits(limits=1):
        if args.bilby_result:
            training, rough = import_bilby_posterior(
                args.bilby_result, model, priors, args, outdir, fingerprint
            )
        elif args.rough_from:
            rough = json.loads((args.rough_from / "rough.json").read_text())
            if (
                rough["index"] != args.index
                or rough["fingerprint"] != fingerprint
                or (
                    rough.get("source_type") != "bilby_posterior"
                    and rough["nlive"] != args.nlive
                )
            ):
                raise ValueError("reused rough run does not match model/index/nlive")
            with np.load(args.rough_from / "rough.npz") as archive:
                if tuple(archive["parameter_names"]) != names:
                    raise ValueError("reused rough coordinate order differs")
                training = archive["training"]
        else:
            print("Running rough rslice stage...", flush=True)
            training, rough = run_rough(model, priors, args, outdir, fingerprint)
        print(f"Rough stage: {rough}", flush=True)
        checks["rslice"] = check_saved_stage(
            model,
            truth,
            (args.rough_from or outdir) / "rough.npz",
            stage=rough.get("training_label", "rslice"),
            stopping_met=rough["reached_target"],
            args=args,
        )
        write_json(outdir / "stage_checks.json", checks)
        if args.training_only:
            return 0
        fit_start = time.perf_counter()
        proposal, factors = fit_wide_morph(training, names, args.bandwidth_scale)
        fit_seconds = time.perf_counter() - fit_start
        write_json(
            outdir / "proposal.json",
            dict(
                metadata=asdict(proposal.metadata),
                bandwidth_scale=args.bandwidth_scale,
                factors=factors,
            ),
        )
        print("Running fixed_morph NISMO stage...", flush=True)
        nis_start = time.perf_counter()
        last_update = nis_start

        def progress(info):
            nonlocal last_update
            now = time.perf_counter()
            if now - last_update >= 15:
                write_json(outdir / "status.json", dict(stage="nismo", **info))
                print(
                    f"NISMO: {int(info['iteration'])} iterations, "
                    f"{int(info['likelihood_calls'])} calls, "
                    f"remaining dlogz={info['remaining_dlogz']:.3g}",
                    flush=True,
                )
                last_update = now

        result = NISMOSampler(
            model=model,
            importance_morph=proposal,
            proposal_scheme="mor-rwalk",
             mor_rwalk_settings=MORWalkSettings(
            n_proposals=100_000,
            refill=False,
                         ),
            n_live=args.nlive,
            rng=args.nismo_seed,
            proposal_batch_size=4*64,
        ).run(
            dlogz=args.dlogz,
            max_iterations=max(10000, 25 * args.nlive),
            max_likelihood_calls=args.max_calls,
            max_wall_time=args.max_seconds,
            progress=True,
        )
        nis_seconds = time.perf_counter() - nis_start
    diagnostics = asdict(summarize(result))
    result.save(outdir / "nismo", plots=False)
    with threadpool_limits(limits=1):
        checks["nismo"] = check_saved_stage(
            model,
            truth,
            outdir / "nismo" / "weighted_samples.npz",
            stage="nismo",
            stopping_met=result.success,
            args=args,
        )
    write_json(outdir / "stage_checks.json", checks)
    check_seconds = checks["rslice"]["seconds"] + checks["nismo"]["seconds"]
    report = dict(
        scope="Eight intrinsic parameters; injected extrinsics fixed. Pilot only.",
        versions={
            p: version(p) for p in ("nismo", "morphZ", "bilby", "dynesty", "lalsuite")
        },
        fingerprint=fingerprint,
        parameter_names=names,
        rough_source=str(args.rough_from.resolve()) if args.rough_from else "fresh",
        rough=rough,
        fit_seconds=fit_seconds,
        nismo=dict(
            logz=result.logz,
            logzerr=result.logzerr,
            success=result.success,
            termination_reason=result.termination_reason,
            n_likelihood_calls=result.n_likelihood_calls,
            n_proposals=result.n_proposals,
            niter=result.niter,
            seconds=nis_seconds,
            diagnostics=diagnostics,
            warnings=result.warnings,
        ),
        stage_checks=checks,
        diagnostic_seconds=check_seconds,
        diagnostic_likelihood_calls=2,
        pipeline_seconds=(
            rough["seconds"]
            + fit_seconds
            + nis_seconds
            + check_seconds
            + rough.get("import_seconds", 0)
            if rough["seconds"] is not None
            else None
        ),
        import_seconds=rough.get("import_seconds", 0),
        invocation_seconds=time.perf_counter() - started,
        pipeline_likelihood_calls=(
            rough["n_likelihood_calls"] + result.n_likelihood_calls
            if rough["n_likelihood_calls"] is not None
            else None
        ),
    )
    write_json(outdir / "summary.json", report)
    write_json(
        outdir / "status.json",
        dict(
            stage="finished",
            success=result.success,
            termination_reason=result.termination_reason,
        ),
    )
    print(json.dumps(report, indent=2, default=str), flush=True)
    save_corner_plots(outdir, args.index, args.rough_from)
    # Preserve bounded partial runs, but make them visible to batch automation.
    return 0 if result.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
