#!/usr/bin/env bash

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && [[ -f "${SLURM_SUBMIT_DIR}/src/quantitative_experiments/multilingual_head_patch/variance_decomposition_continuous.py" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"

python -m quantitative_experiments.multilingual_head_patch.variance_decomposition_continuous \
  --config configs/multilingual_head_patch/analysis/config_variance_decomposition_continuous.yaml \
  "$@"
