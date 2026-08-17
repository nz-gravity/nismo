#!/bin/bash
#SBATCH --job-name=pp_dyn100
#SBATCH --output=dynesty_nlive100_%A_%a.out
#SBATCH --error=dynesty_nlive100_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1G
#SBATCH --array=0-99

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
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
