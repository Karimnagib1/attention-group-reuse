#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/eval/run_eval.sh [extra args]
# Example: bash scripts/eval/run_eval.sh --set model.name_or_path=meta-llama/Llama-3.1-8B

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"






cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"

python -m quantitative_experiments.mmluprox_eval.launch \
  --config configs/eval/base.yaml \
  "$@"

  # --config configs/eval/model.yaml \
  # --config configs/eval/data.yaml \
  # --config configs/eval/eval.yaml \
  # --config configs/eval/run.yaml \