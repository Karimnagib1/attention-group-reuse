#!/usr/bin/env bash
# Shared runner for the two sbatch scripts; defaults to the 14B BF16 config.
# Usage: run.sh {evaluate|patch} {mmluprox|belebele} [extra Python CLI arguments]
set -eo pipefail

if [[ $# -lt 2 ]] || [[ "$1" != evaluate && "$1" != patch ]] ||
   [[ "$2" != mmluprox && "$2" != belebele ]]; then
  echo "Usage: $0 {evaluate|patch} {mmluprox|belebele} [--config ...] [--set ...]" >&2
  exit 2
fi
mode="$1"
dataset="$2"
shift 2

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

if ! command -v conda >/dev/null 2>&1 && [[ -f "${HOME}/.bashrc" ]]; then
  source "${HOME}/.bashrc"
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "Conda is unavailable. Activate thesis on the login node before submitting." >&2
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis
set -u

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"
export PYTHONPATH="${repo_root}/src:${repo_root}:${PYTHONPATH:-}"
mkdir -p "$HF_HOME"

command=(python -u -m quantitative_experiments.ministral3 "$mode"
         --config configs/ministral3/mmluprox.yaml)
if [[ "$dataset" == belebele ]]; then
  command+=(--config configs/ministral3/belebele.yaml)
fi
command+=("$@")

printf 'Ministral %s: dataset=%s, conda=%s, cache=%s, job=%s\n' \
  "$mode" "$dataset" "$CONDA_DEFAULT_ENV" "$HF_HOME" "${SLURM_JOB_ID:-local}"
printf -v command_display '%q ' "${command[@]}"
printf 'Command: %s\n' "$command_display"

if [[ "$mode" == patch ]]; then
  command=(bash scripts/correctness_stratified/run_with_slurm_requeue.sh "${command[@]}")
fi
exec python -u -m quantitative_experiments.ministral3_process "${command[@]}"
