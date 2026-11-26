#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUANT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${QUANT_ROOT}/configs/statistical_support/paper_statistics.yaml"

if [[ "$(hostname -s)" != "cn-09" ]] && [[ "${ALLOW_NON_CN09:-0}" != "1" ]]; then
  echo "This paper analysis is configured to run on cn-09. SSH to cn-09 first." >&2
  exit 2
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "thesis" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "Conda is unavailable; activate the thesis environment before running." >&2
    exit 2
  fi
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate thesis
fi

cd "${QUANT_ROOT}"
echo "Host: $(hostname -s)"
echo "Conda environment: ${CONDA_DEFAULT_ENV}"
echo "Config: ${CONFIG_PATH}"

python "${SCRIPT_DIR}/compute_paper_statistics.py" --config "${CONFIG_PATH}" "$@"
