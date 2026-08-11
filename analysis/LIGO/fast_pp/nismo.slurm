#!/bin/bash
#SBATCH --job-name=nismo_pp
#SBATCH --output=nismo_%A_%a.out
#SBATCH --error=nismo_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --array=1-100

set -euo pipefail

# This job only reads dynesty_result.json.  It does not resume Dynesty and
# never reads dynesty_resume.pickle or dynesty_dynesty.pickle.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Reuse the environment that previously ran Bilby + MorphZ.  NISMO is loaded
# directly from this checkout, so no installation or current-directory layout
# is required on the cluster.
MORPHZ_VENV="${MORPHZ_VENV:-/morphZ_casestudy_CBC_pe/.venv}"
source "${MORPHZ_VENV}/bin/activate"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python "${SCRIPT_DIR}/nismo_computation.py" "${SLURM_ARRAY_TASK_ID}"
