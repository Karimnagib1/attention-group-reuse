#!/usr/bin/env bash
#SBATCH --job-name=sweeping_llama
#SBATCH --output=logs/sweeping_%j.log
#SBATCH --error=logs/sweeping_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16        
#SBATCH --gres=gpu:4             
#SBATCH --mem=180G               
#SBATCH --partition=cscc-gpu-p    # GPU partition
#SBATCH --qos=cscc-gpu-qos        # GPU QoS
#SBATCH --time=48:00:00           # 48 Hours

set -euo pipefail

source ~/.bashrc
conda activate thesis

module load nvidia/cuda/12.0
export CUDA_HOME="${CUDA_HOME:-${CUDA_DIR:-${EBROOTCUDA:-/apps/local/cuda}}}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

log_with_timestamp() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

SECONDS=0

on_exit() {
  local exit_code=$?
  log_with_timestamp "Job finished with exit code ${exit_code} after ${SECONDS}s"
}

trap on_exit EXIT

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && [[ -f "${SLURM_SUBMIT_DIR}/src/quantitative_experiments/multilingual_head_patch/run.py" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p logs

RUN_MODULE="quantitative_experiments.multilingual_head_patch.run"
# CONFIG_PATH="configs/multilingual_head_patch/patch/language_contrast_belebele.yaml"
CONFIG_PATH="configs/multilingual_head_patch/patch/config_belebele.yaml"

if [[ ! -f "src/quantitative_experiments/multilingual_head_patch/run.py" ]]; then
  log_with_timestamp "Missing runner module file: ${REPO_ROOT}/src/quantitative_experiments/multilingual_head_patch/run.py"
  exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  log_with_timestamp "Missing config file: ${REPO_ROOT}/${CONFIG_PATH}"
  exit 1
fi

log_with_timestamp "Starting Belebele patch job"
log_with_timestamp "Repository root: ${REPO_ROOT}"
log_with_timestamp "Slurm job id: ${SLURM_JOB_ID:-local}"
log_with_timestamp "Config: ${CONFIG_PATH}"


export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub


cmd=(
  python
  -m
  "${RUN_MODULE}"
  --config "${CONFIG_PATH}"
  "$@"
)

printf -v cmd_pretty '%q ' "${cmd[@]}"
log_with_timestamp "Command: ${cmd_pretty% }"

"${cmd[@]}"
