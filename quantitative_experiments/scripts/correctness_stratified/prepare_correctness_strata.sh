#!/usr/bin/env bash
# Step 1: prepare correctness strata, sweep ID lists, and exclusion summaries.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUANT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${QUANT_ROOT}/configs/correctness_stratified/analysis.yaml"

source ~/.bashrc
conda activate thesis

cd "${QUANT_ROOT}"
export PYTHONPATH="${QUANT_ROOT}/src:${QUANT_ROOT}:${PYTHONPATH:-}"

python "${SCRIPT_DIR}/prepare_correctness_strata.py" \
  --config "${CONFIG_PATH}" \
  "$@"
