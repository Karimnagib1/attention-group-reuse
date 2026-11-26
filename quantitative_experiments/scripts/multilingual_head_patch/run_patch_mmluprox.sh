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

########################################
# Environment setup
########################################

# If your cluster needs ~/.bashrc to get conda, uncomment:
source ~/.bashrc

# Activate your conda environment
conda activate thesis

# Load CUDA module (cluster-specific)
module load nvidia/cuda/12.0

# Optional: set CUDA_HOME explicitly (often not strictly necessary)
export CUDA_HOME="${CUDA_HOME:-${CUDA_DIR:-${EBROOTCUDA:-/apps/local/cuda}}}"


export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && [[ -f "${SLURM_SUBMIT_DIR}/src/quantitative_experiments/multilingual_head_patch/run.py" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p logs

python -m quantitative_experiments.multilingual_head_patch.run \
  --config configs/multilingual_head_patch/patch/config_mmluprox.yaml \
  "$@"
