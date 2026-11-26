#!/usr/bin/env bash
# Visualize combined or single-condition correctness-stratified results.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUANT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source ~/.bashrc
conda activate thesis

cd "${QUANT_ROOT}"
export PYTHONPATH="${QUANT_ROOT}/src:${QUANT_ROOT}:${PYTHONPATH:-}"

python "${SCRIPT_DIR}/visualize_correctness_stratified.py" "$@"
