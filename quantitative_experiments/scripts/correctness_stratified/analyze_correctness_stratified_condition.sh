#!/usr/bin/env bash
# Analyze one completed correctness-stratified model/dataset sweep.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUANT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${QUANT_ROOT}/configs/correctness_stratified/analysis.yaml"

source ~/.bashrc
conda activate thesis

cd "${QUANT_ROOT}"
export PYTHONPATH="${QUANT_ROOT}/src:${QUANT_ROOT}:${PYTHONPATH:-}"

python "${SCRIPT_DIR}/analyze_correctness_stratified_condition.py" \
  --config "${CONFIG_PATH}" \
  "$@"
