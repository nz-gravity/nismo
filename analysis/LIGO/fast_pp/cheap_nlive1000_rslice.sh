#!/bin/bash
set -euo pipefail
module purge; module load gcc/13.3.0 python/3.12.3
source /fred/oz200/avajpeyi/projects/MORPH/nismo/.venv/bin/activate
cd /fred/oz200/avajpeyi/projects/MORPH/nismo/analysis/LIGO/fast_pp
export PYTHONPATH=/fred/oz200/avajpeyi/projects/MORPH/nismo/src${PYTHONPATH:+:$PYTHONPATH}
/usr/bin/time -v python /fred/oz200/avajpeyi/projects/MORPH/nismo/analysis/LIGO/fast_pp/pp_analysis.py --index 0 --sampler dynesty \
  --output-dir /fred/oz200/avajpeyi/projects/MORPH/nismo/analysis/LIGO/fast_pp/outdir/seed_0/cheap_nlive1000_rslice --label cheap_nlive1000_rslice \
  --no-corner --nlive 1000 --sample rslice --slices 3
