#!/usr/bin/env bash

# Exit on error
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"

########################################
# CONFIGURATION
########################################

# Default config file (can override via first argument)
DEFAULT_CONFIG="configs/multilingual_head_patch/baseline/baseline_config.yaml"

# Optional: set your conda environment name here
CONDA_ENV="thesis"

# Optional: log directory
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

########################################
# ARGUMENT PARSING
########################################

CONFIG_FILE="${1:-$DEFAULT_CONFIG}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file not found: $CONFIG_FILE"
    exit 1
fi

########################################
# ENVIRONMENT SETUP (optional)
########################################

# Uncomment if using conda
if command -v conda &> /dev/null; then
    echo "Activating conda environment: ${CONDA_ENV}"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

########################################
# RUN BASELINE SCRIPT
########################################

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/baseline_${TIMESTAMP}.log"

echo "Running baseline analysis..."
echo "Config: ${CONFIG_FILE}"
echo "Log file: ${LOG_FILE}"
echo "----------------------------------------"

python -m quantitative_experiments.multilingual_head_patch.run_baseline --config "${CONFIG_FILE}" 2>&1 | tee "${LOG_FILE}"

echo "----------------------------------------"
echo "Baseline run complete."
echo "Results saved according to config."
