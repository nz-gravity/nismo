#!/bin/bash
#SBATCH --job-name=nismo_dyn100
#SBATCH --output=nismo_dynesty_nlive100_%A_%a.out
#SBATCH --error=nismo_dynesty_nlive100_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-99

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LVK_SEED="${SLURM_ARRAY_TASK_ID}"
NISMO_SEED="${NISMO_SEED:-20260811}"

module purge
module load gcc/13.3.0 python/3.12.3

NISMO_VENV="${NISMO_VENV:-/fred/oz200/avajpeyi/projects/MORPH/nismo/.venv}"
source "${NISMO_VENV}/bin/activate"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "NISMO: LVK seed ${LVK_SEED}; trained on Dynesty-100; NISMO seed ${NISMO_SEED}"
srun python "${SCRIPT_DIR}/nismo_computation.py" \
    "${LVK_SEED}" \
    --dynesty-nlive 100 \
    --nismo-seed "${NISMO_SEED}"
