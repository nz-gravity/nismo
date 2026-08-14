#!/usr/bin/env python

"""
pp_analysis.py

Load simulation produced by pp_setup.load_simulation
and run either Dynesty or Bilby-MCMC.

Usage:
    python pp_analysis.py --index 37 --sampler dynesty
"""

import argparse
import multiprocessing as mp
import os
from pathlib import Path

import bilby
from pp_setup import load_simulation

NPOOL = min(mp.cpu_count(), int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))


def run_dynesty(
    index,
    *,
    nlive=2000,
    output_dir=None,
    label=None,
    resume=False,
    plot_corner=True,
):
    likelihood, priors, default_outdir, _, checkpoint_delta_t = load_simulation(
        index,
        plot_data=output_dir is None,
    )
    outdir = Path(output_dir).resolve() if output_dir is not None else default_outdir
    outdir.mkdir(parents=True, exist_ok=True)
    run_label = label or ("dynesty" if nlive == 2000 else f"dynesty_nlive{nlive}")
    result_path = outdir / f"{run_label}_result.json"
    if result_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing result: {result_path}. "
            "Choose a new --output-dir or --label."
        )

    print(
        f"Running Dynesty seed={index}, nlive={nlive}, label={run_label}, "
        f"outdir={outdir}..."
    )
    result = bilby.run_sampler(
        likelihood,
        priors,
        sampler="dynesty",
        outdir=outdir,
        label=run_label,
        nlive=nlive,
        nact=20,
        sample="rwalk",
        resume=resume,
        npool=NPOOL,
        check_point_delta_t=checkpoint_delta_t,
        check_point_plot=True,
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
        result_class=bilby.gw.result.CBCResult,
    )
    if plot_corner:
        result.plot_corner()
    return result


def run_mcmc(index):
    likelihood, priors, outdir, _, checkpoint_delta_t = load_simulation(index)
    print(f"Running Bilby-MCMC {index}...")
    result = bilby.run_sampler(
        likelihood,
        priors,
        sampler="bilby_mcmc",
        outdir=outdir,
        label="mcmc",
        nsamples=2000,
        thin_by_nact=0.2,
        ntemps=8,
        npool=NPOOL,
        Tmax_from_SNR=20,
        adapt=True,
        proposal_cycle="gwA",
        L1steps=100,
        L2steps=5,
        check_point_delta_t=checkpoint_delta_t,
        check_point_plot=True,
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
        result_class=bilby.gw.result.CBCResult,
    )
    result.plot_corner()


def main():
    print(f"[pp_analysis] Using NPOOL = {NPOOL}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--sampler", choices=["dynesty", "mcmc"], required=True)
    parser.add_argument("--nlive", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--label")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from the matching Bilby/Dynesty resume checkpoint",
    )
    parser.add_argument("--no-corner", action="store_true")
    args = parser.parse_args()

    if args.nlive <= 0:
        parser.error("--nlive must be a positive integer")
    if args.sampler == "mcmc" and (
        args.nlive != 2000
        or args.output_dir is not None
        or args.label is not None
        or args.resume
    ):
        parser.error(
            "--nlive, --output-dir, and --label currently apply only to dynesty"
        )

    if args.sampler == "dynesty":
        run_dynesty(
            args.index,
            nlive=args.nlive,
            output_dir=args.output_dir,
            label=args.label,
            resume=args.resume,
            plot_corner=not args.no_corner,
        )
    else:
        run_mcmc(args.index)

    print("Analysis complete.")


if __name__ == "__main__":
    main()
