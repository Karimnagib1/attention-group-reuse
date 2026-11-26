#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/eval/run_eval_belebele.sh [extra args]
# Example: bash scripts/eval/run_eval_belebele.sh --set model.name_or_path=meta-llama/Llama-3.1-8B

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  EXTRA_ARGS=$(printf '%q ' "$@")
  sbatch --chdir="${REPO_ROOT}" --export=ALL,EXTRA_ARGS="$EXTRA_ARGS",REPO_ROOT="${REPO_ROOT}" <<'EOF'
#!/usr/bin/env bash
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --mem=80G
#SBATCH -J belebele-eval
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
set -euo pipefail

cd "$REPO_ROOT"
eval "set -- $EXTRA_ARGS"
bash scripts/eval/run_eval_belebele.sh "$@"
EOF
  exit 0
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"

python -m quantitative_experiments.mmluprox_eval.launch \
  --config configs/eval/model.yaml \
  --config configs/eval/data_belebele.yaml \
  --config configs/eval/eval_belebele.yaml \
  --config configs/eval/run_belebele.yaml \
  "$@"
