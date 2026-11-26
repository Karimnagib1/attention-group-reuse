#!/bin/bash -l
#SBATCH --job-name=donor_ablation
#SBATCH --output=quantitative_experiments/review/logs/donor_ablation_%j.out
#SBATCH --error=quantitative_experiments/review/logs/donor_ablation_%j.err
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
Submit from the thesis3 repository root:
  mkdir -p quantitative_experiments/review/logs
  sbatch quantitative_experiments/review/run_donor_ablation.sbatch qwen_mmluprox --samples-per-language 100

Conditions: qwen_mmluprox (default), qwen_belebele, llama_mmluprox, llama_belebele.
Remaining arguments are forwarded to the existing donor-ablation run command.
An unchanged configuration resumes its existing output directory automatically.
The job does not depend on the submitting SSH connection staying open.

Slurm options go before the script path, e.g. sbatch --time=24:00:00 SCRIPT CONDITION.
Resubmit the same command if the job reaches its wall-time limit; --requeue permits
scheduler-initiated requeues but does not automatically extend that time limit.

For a local command-only check (no environment activation, model, or submission):
  DONOR_SLURM_DRY_RUN=1 bash quantitative_experiments/review/run_donor_ablation.sbatch qwen_mmluprox
HELP
  exit 0
fi

condition="${1:-qwen_mmluprox}"
if [[ "$#" -gt 0 ]]; then
  shift
fi
case "${condition}" in
  qwen_mmluprox|qwen_belebele|llama_mmluprox|llama_belebele) ;;
  *) echo "Unknown condition '${condition}'; run this script with --help." >&2; exit 2 ;;
esac

# Slurm copies batch scripts into its spool directory, so BASH_SOURCE cannot
# locate the checkout during a batch run. Resolve it from the submission directory.
donor_repo_root="${DONOR_REPO_ROOT:-${SLURM_SUBMIT_DIR:-${PWD}}}"
donor_wrapper="${donor_repo_root}/quantitative_experiments/review/run_donor_ablation.sh"
donor_config="${donor_repo_root}/quantitative_experiments/review/configs/${condition}.yaml"
if [[ ! -f "${donor_wrapper}" || ! -f "${donor_config}" ]]; then
  echo "Submit from the thesis3 repository root, or set DONOR_REPO_ROOT to that checkout." >&2
  exit 2
fi
cd "${donor_repo_root}"
cmd=(bash "${donor_wrapper}" run --config "${donor_config}" "$@")
printf 'Condition: %s\n' "${condition}"
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'
if [[ "${DONOR_SLURM_DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Use sbatch to launch this file (or set DONOR_SLURM_DRY_RUN=1 to inspect the command)." >&2
  exit 2
fi

# Initialize Conda explicitly: noninteractive shells may skip ~/.bashrc setup.
if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
  donor_conda_base="$("${CONDA_EXE}" info --base)"
elif command -v conda >/dev/null 2>&1; then
  donor_conda_base="$(conda info --base)"
elif [[ -f /apps/local/anaconda3/etc/profile.d/conda.sh ]]; then
  donor_conda_base=/apps/local/anaconda3
else
  echo "Conda is unavailable; activate thesis before submitting the job." >&2
  exit 2
fi
source "${donor_conda_base}/etc/profile.d/conda.sh"
conda activate thesis

if command -v module >/dev/null 2>&1; then
  module load nvidia/cuda/12.0
fi
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-${HOME}/.cache/huggingface/token}"
export PYTHONUNBUFFERED=1

# Existing atomic layer/group checkpoints provide resume after interruption.
# Do not use the old SIGUSR1 requeue wrapper: this runner has no USR1 handler.
exec "${cmd[@]}"
