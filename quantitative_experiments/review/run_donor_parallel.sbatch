#!/bin/bash -l
#SBATCH --job-name=donor_parallel
#SBATCH --output=quantitative_experiments/review/logs/donor_parallel_%j.out
#SBATCH --error=quantitative_experiments/review/logs/donor_parallel_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=180G
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --exclude=gpu-05
#SBATCH --open-mode=append

set -euo pipefail
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'HELP'
From the thesis3 repository root:
  mkdir -p quantitative_experiments/review/logs
  sbatch quantitative_experiments/review/run_donor_parallel.sbatch qwen_mmluprox --samples-per-language 100

Conditions: qwen_mmluprox (default), qwen_belebele, llama_mmluprox, llama_belebele.
Defaults: 4 workers, one GPU each, FP32, batch size 4 per GPU, donor batch 4.
Extra arguments go to the parallel runner, e.g. --batch-size 2 or --dtype bfloat16.
Outputs default to runs/CONDITION_parallel_float32 (or _bfloat16).
Repeat the same command to resume. Wall-time expiry requires resubmission.
DONOR_SLURM_DRY_RUN=1 prints the command without activation or GPU work.
HELP
  exit 0
fi
condition="${1:-qwen_mmluprox}"
if [[ "$#" -gt 0 ]]; then shift; fi
case "${condition}" in
  qwen_mmluprox|qwen_belebele|llama_mmluprox|llama_belebele) ;;
  *) echo "Unknown condition '${condition}'." >&2; exit 2 ;;
esac
donor_repo_root="${DONOR_REPO_ROOT:-${SLURM_SUBMIT_DIR:-${PWD}}}"
donor_wrapper="${donor_repo_root}/quantitative_experiments/review/run_donor_parallel.sh"
donor_config="${donor_repo_root}/quantitative_experiments/review/configs/${condition}.yaml"
if [[ ! -f "${donor_wrapper}" || ! -f "${donor_config}" ]]; then
  echo "Submit from the thesis3 repository root, or set DONOR_REPO_ROOT." >&2
  exit 2
fi
cd "${donor_repo_root}"
cmd=(bash "${donor_wrapper}" run --config "${donor_config}" "$@")
printf 'Condition: %s\nCommand:' "${condition}"
printf ' %q' "${cmd[@]}"
printf '\n'
if [[ "${DONOR_SLURM_DRY_RUN:-0}" == "1" ]]; then exit 0; fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Use sbatch (or DONOR_SLURM_DRY_RUN=1 to inspect the command)." >&2
  exit 2
fi
printf 'Node: %s; job: %s; allocated nodes: %s; CUDA_VISIBLE_DEVICES: %s\n' \
  "$(hostname)" "${SLURM_JOB_ID}" "${SLURM_JOB_NODELIST:-unknown}" "${CUDA_VISIBLE_DEVICES:-unset}"
if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
  donor_conda_base="$("${CONDA_EXE}" info --base)"
elif command -v conda >/dev/null 2>&1; then
  donor_conda_base="$(conda info --base)"
elif [[ -f /apps/local/anaconda3/etc/profile.d/conda.sh ]]; then
  donor_conda_base=/apps/local/anaconda3
else
  echo "Conda is unavailable." >&2; exit 2
fi
source "${donor_conda_base}/etc/profile.d/conda.sh"
conda activate thesis
if command -v module >/dev/null 2>&1; then module load nvidia/cuda/12.0; fi
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-${HOME}/.cache/huggingface/token}"
export PYTHONUNBUFFERED=1
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
fi
exec "${cmd[@]}"
