#!/bin/bash
#SBATCH --job-name=nismo_pp
#SBATCH --output=nismo_%A_%a.out
#SBATCH --error=nismo_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-99

set -euo pipefail

# This job only reads dynesty_result.json.  It does not resume Dynesty and
# never reads dynesty_resume.pickle or dynesty_dynesty.pickle.
# OzSTAR's sbatch stages this script into a per-job /jobfs directory before
# running it, so BASH_SOURCE no longer points at the real repo. Resolve
# SCRIPT_DIR from SLURM_SUBMIT_DIR (the directory `sbatch` was invoked from)
# instead.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unset; submit with sbatch from analysis/LIGO/fast_pp}"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# OzSTAR runtime: load the requested toolchain and reuse the repository venv.
# The Dynesty-2000 result for each seed is read from outdir/seed_<index>/.
module purge
module load gcc/13.3.0 python/3.12.3

MORPHZ_VENV="${MORPHZ_VENV:-/fred/oz200/avajpeyi/projects/MORPH/nismo/.venv}"
source "${MORPHZ_VENV}/bin/activate"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

NISMO_SEED="${NISMO_SEED:-20260811}"

srun python "${SCRIPT_DIR}/nismo_computation.py" \
	"${SLURM_ARRAY_TASK_ID}" \
	--nismo-seed "${NISMO_SEED}"
