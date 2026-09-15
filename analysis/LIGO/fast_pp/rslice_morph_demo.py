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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--rough-from", type=Path, help="reuse rough.npz/json from this run directory"
    )
    parser.add_argument(
        "--nlive", type=int, default=100, help="live points in both stages"
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
    args = parser.parse_args(argv)
    for key in (
        "nlive",
        "slices",
        "rough_maxcall",
        "bandwidth_scale",
        "rough_dlogz",
        "dlogz",
        "max_seconds",
        "max_calls",
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
    outdir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    write_json(outdir / "settings.json", vars(args))
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
    with threadpool_limits(limits=1):
        if args.rough_from:
            rough = json.loads((args.rough_from / "rough.json").read_text())
            if (
                rough["index"] != args.index
                or rough["fingerprint"] != fingerprint
                or rough["nlive"] != args.nlive
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
        pipeline_seconds=rough["seconds"] + fit_seconds + nis_seconds,
        invocation_seconds=time.perf_counter() - started,
        pipeline_likelihood_calls=rough["n_likelihood_calls"]
        + result.n_likelihood_calls,
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
    # Preserve bounded partial runs, but make them visible to batch automation.
    return 0 if result.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
