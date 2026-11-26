#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"

DEFAULT_CONFIG="configs/multilingual_head_patch/baseline/baseline_layerwise_config.yaml"
CONDA_ENV="thesis"
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

CONFIG_FILE="${1:-$DEFAULT_CONFIG}"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file not found: $CONFIG_FILE"
    exit 1
fi

if command -v conda &> /dev/null; then
    echo "Activating conda environment: ${CONDA_ENV}"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/baseline_layerwise_${TIMESTAMP}.log"

echo "Running layer-wise baseline analysis..."
echo "Config: ${CONFIG_FILE}"
echo "Log file: ${LOG_FILE}"
echo "----------------------------------------"

python -m quantitative_experiments.multilingual_head_patch.run_baseline_layerwise --config "${CONFIG_FILE}" 2>&1 | tee "${LOG_FILE}"

echo "----------------------------------------"
echo "Layer-wise baseline run complete."
