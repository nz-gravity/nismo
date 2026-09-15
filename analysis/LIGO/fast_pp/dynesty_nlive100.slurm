#!/bin/bash
#SBATCH --job-name=pp_dyn100
#SBATCH --output=dynesty_nlive100_%A_%a.out
#SBATCH --error=dynesty_nlive100_%A_%a.err
# Seed 0 took 43:56 wall at nlive=100 (the rwalk proposal hits maxmcmc
# constantly at low live counts, so this is NOT faster than nlive=2000 per
# sample).  4h leaves headroom for slower seeds.
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1G
#SBATCH --array=0-99

set -euo pipefail

# OzSTAR's sbatch stages this script into a per-job /jobfs directory, so
# BASH_SOURCE points there rather than at the repository.  Resolve from a
# fixed absolute path (override with NISMO_FAST_PP_DIR) instead.
SCRIPT_DIR="${NISMO_FAST_PP_DIR:-/fred/oz200/avajpeyi/projects/MORPH/nismo/analysis/LIGO/fast_pp}"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LVK_SEED="${SLURM_ARRAY_TASK_ID}"
RUN_DIR="${SCRIPT_DIR}/outdir/seed_${LVK_SEED}/dynesty_nlive100"

module purge
module load gcc/13.3.0 python/3.12.3

NISMO_VENV="${NISMO_VENV:-/fred/oz200/avajpeyi/projects/MORPH/nismo/.venv}"
source "${NISMO_VENV}/bin/activate"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "Dynesty-100: LVK seed ${LVK_SEED}; output ${RUN_DIR}"
srun python "${SCRIPT_DIR}/pp_analysis.py" \
    --index "${LVK_SEED}" \
    --sampler dynesty \
    --nlive 100 \
    --output-dir "${RUN_DIR}" \
    --label dynesty_nlive100 \
    --resume \
    --no-corner
