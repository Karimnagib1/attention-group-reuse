#!/usr/bin/env bash
set -euo pipefail
review_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${review_dir}:${review_dir}/../src${PYTHONPATH:+:${PYTHONPATH}}"
# Avoid excessive BLAS threads during the small-matrix bootstrap analysis.
export OPENBLAS_NUM_THREADS="${DONOR_BLAS_THREADS:-1}"
export OMP_NUM_THREADS="${DONOR_OMP_THREADS:-1}"
exec python -m donor_ablation.run "$@"
