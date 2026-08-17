#!/bin/bash
#SBATCH --job-name=pp_dyn4k
#SBATCH --output=logs/dynesty_nlive4000/index_%a.out
#SBATCH --error=logs/dynesty_nlive4000/index_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1G
#SBATCH --array=0-99

# Dynesty PP analysis at nlive=4000, the high-live-point companion to the
# stored nlive=2000 production run.
#
# Results are written to outdir/seed_<index>/dynesty_nlive4000/ so the
# Dynesty-2000 baseline (outdir/seed_<index>/dynesty_result.json) is never
# touched; pp_analysis.py also refuses to overwrite an existing result.
#
# Array indices are 0-99 to match injections.csv rows and the seed_<index>
# directories.  (The nlive=2000 dynesty.slurm in this directory came from the
# original MorphZ machine and is left as-is for provenance.)
#
# OzSTAR's sbatch stages this script into a per-job /jobfs directory, so
# BASH_SOURCE does not point at the repository; resolve from SLURM_SUBMIT_DIR.

set -uo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unset; submit with sbatch from analysis/LIGO/fast_pp}"

module purge
module load gcc/13.3.0 python/3.12.3

NISMO_VENV="${NISMO_VENV:-/fred/oz200/avajpeyi/projects/MORPH/nismo/.venv}"
source "${NISMO_VENV}/bin/activate"

NLIVE="${NLIVE:-4000}"
LABEL="dynesty_nlive${NLIVE}"
OUTPUT_DIR="${SCRIPT_DIR}/outdir/seed_${SLURM_ARRAY_TASK_ID}/${LABEL}"

echo "------------------------------------------------------------"
echo " Dynesty PP ANALYSIS (nlive=${NLIVE})"
echo " SLURM_JOB_ID:        ${SLURM_JOB_ID}"
echo " SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo " SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK}"
echo " Output dir:          ${OUTPUT_DIR}"
echo " Node:                $(hostname)"
echo "------------------------------------------------------------"
date

srun python "${SCRIPT_DIR}/pp_analysis.py" \
	--index "${SLURM_ARRAY_TASK_ID}" \
	--sampler dynesty \
	--nlive "${NLIVE}" \
	--output-dir "${OUTPUT_DIR}" \
	--label "${LABEL}"
rc=$?

date
echo "Finished task ${SLURM_ARRAY_TASK_ID} with exit code ${rc}"
exit ${rc}
