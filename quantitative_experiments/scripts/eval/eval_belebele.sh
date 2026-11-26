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

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && [[ -f "${SLURM_SUBMIT_DIR}/src/quantitative_experiments/mmluprox_eval/launch.py" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p logs



# Usage: bash scripts/eval/eval_belebele.sh [extra args]
# Example: bash scripts/eval/eval_belebele.sh --set model.name_or_path=meta-llama/Llama-3.1-8B

python -m quantitative_experiments.mmluprox_eval.launch \
  --config configs/eval/model.yaml \
  --config configs/eval/data_belebele.yaml \
  --config configs/eval/eval_belebele.yaml \
  --config configs/eval/run_belebele.yaml \
  "$@"
