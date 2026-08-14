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
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# OzSTAR runtime: load the requested toolchain and reuse the repository venv.
# The Dynesty-2000 result for each seed is read from outdir/seed_<index>/
# unless RESULT_ROOT is overridden at submit time.
module purge
module load gcc/13.3.0 python/3.12.3

MORPHZ_VENV="${MORPHZ_VENV:-/fred/oz200/avajpeyi/projects/MORPH/nismo/.venv}"
source "${MORPHZ_VENV}/bin/activate"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

RESULT_ROOT="${RESULT_ROOT:-${SCRIPT_DIR}/outdir}"
RESULT_PATH="${RESULT_ROOT}/seed_${SLURM_ARRAY_TASK_ID}/dynesty_result.json"

if [[ ! -f "${RESULT_PATH}" ]]; then
	echo "Missing Dynesty result: ${RESULT_PATH}" >&2
	exit 1
fi

srun python "${SCRIPT_DIR}/nismo_computation.py" \
	"${SLURM_ARRAY_TASK_ID}" \
	--result-path "${RESULT_PATH}" \
	--output-dir "${RESULT_ROOT}/seed_${SLURM_ARRAY_TASK_ID}"
