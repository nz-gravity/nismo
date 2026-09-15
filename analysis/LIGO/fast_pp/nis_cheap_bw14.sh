#!/bin/bash
set -euo pipefail
module purge; module load gcc/13.3.0 python/3.12.3
source /fred/oz200/avajpeyi/projects/MORPH/nismo/.venv/bin/activate
cd /fred/oz200/avajpeyi/projects/MORPH/nismo/analysis/LIGO/fast_pp
export PYTHONPATH=/fred/oz200/avajpeyi/projects/MORPH/nismo/src${PYTHONPATH:+:$PYTHONPATH}
python /fred/oz200/avajpeyi/projects/MORPH/nismo/analysis/LIGO/fast_pp/nismo_from_result.py 0 --result-path /fred/oz200/avajpeyi/projects/MORPH/nismo/analysis/LIGO/fast_pp/outdir/seed_0/cheap_nlive1000_rslice/cheap_nlive1000_rslice_result.json \
  --output-dir /fred/oz200/avajpeyi/projects/MORPH/nismo/analysis/LIGO/fast_pp/outdir/seed_0/nismo_cheap_bw14 --kde-bw 1.4 --n-workers 1 
