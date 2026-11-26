#!/usr/bin/env bash
#SBATCH --job-name=ministral3_base_patch
#SBATCH --output=logs/ministral3_base_patch_%j.log
#SBATCH --error=logs/ministral3_base_patch_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=180G
#SBATCH --exclude=gpu-54,gpu-05
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 {mmluprox|belebele} [--set ...]" >&2
  exit 2
fi
dataset="$1"
shift

# Keep the batch-shell PID so the existing requeue helper receives USR1.
# Patching uses the Base evaluation's own all_correct_ids.json by default.
# Direct bash runs can inherit SLURM_SUBMIT_DIR from an interactive allocation.
# Prefer this script's location; sbatch copies the script into its spool directory.
script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
repo_root="${REPO_ROOT:-$script_root}"
if [[ -z "${REPO_ROOT:-}" && ! -f "$repo_root/scripts/ministral3/run.sh" ]]; then
  repo_root="${SLURM_SUBMIT_DIR:-$PWD}"
fi
if [[ ! -f "$repo_root/scripts/ministral3/run.sh" ]]; then
  echo "Cannot find Ministral run.sh under $repo_root. Set REPO_ROOT to the quantitative_experiments directory." >&2
  exit 1
fi
exec bash "$repo_root/scripts/ministral3/run.sh" \
  patch "$dataset" \
  --set model.name_or_path=mistralai/Ministral-3-14B-Base-2512 "$@"
