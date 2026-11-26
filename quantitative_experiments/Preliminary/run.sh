#!/usr/bin/env bash
#SBATCH --job-name=prelim_head_patch
#SBATCH --output=logs/prelim_head_patch_%j.log
#SBATCH --error=logs/prelim_head_patch_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=90G
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --open-mode=append
#
# Submit from quantitative_experiments/:  sbatch Preliminary/run.sh
# Runs the head-level sweep (1,152 units) on the 50 questions, then the CPU analysis.
# Group-level numbers come from the paper's saved run (sliced to the same 50 questions);
# to also rerun the group level here, add kv_group to MODES. Requeueing resumes from checkpoints.
set -euo pipefail

source ~/.bashrc
conda activate thesis
module load nvidia/cuda/12.0 2>/dev/null || true

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"
mkdir -p logs

# Works on any GPU count (sbatch --gres=gpu:N overrides the line above). Keep the paper
# run's 8 prompts per GPU: the paper config's batch of 32 was split across 4 GPUs.
# On 1 GPU the runner skips DataParallel automatically. Later --set arguments win.
IFS=, read -ra visible <<< "${CUDA_VISIBLE_DEVICES:-0}"
n_gpus=${#visible[@]}
batch=$((8 * n_gpus))
echo "Node: $(hostname); GPUs: ${CUDA_VISIBLE_DEVICES:-none} ($n_gpus); batch size $batch; job: ${SLURM_JOB_ID:-local}"
set -- --set "patching.batch_size=$batch" --set "patching.donor_batch_size=$batch" "$@"

# Slurm sends USR1 to this batch shell; forward it to the requeue helper, which
# asks the runner to checkpoint and then requeues the job.
child=""
trap 'echo "USR1 received; forwarding."; [[ -n "$child" ]] && kill -USR1 "$child" 2>/dev/null || true' USR1

MODES=(head)
for mode in "${MODES[@]}"; do
  if [[ -f "Preliminary/runs/qwen_mmluprox_50/$mode/_SUCCESS" ]]; then
    echo "$mode already complete; skipping."
    continue
  fi
  set +e
  bash scripts/correctness_stratified/run_with_slurm_requeue.sh \
    python -u Preliminary/run_head_patch.py --mode "$mode" "$@" &
  child=$!
  while true; do
    wait "$child"; status=$?
    kill -0 "$child" 2>/dev/null || break   # wait returns early when a trapped signal arrives
  done
  set -e
  child=""
  [[ "$status" -eq 0 ]] || exit "$status"
  # The helper also exits 0 after requeueing; stop so the requeued job resumes.
  [[ -f "Preliminary/runs/qwen_mmluprox_50/$mode/_SUCCESS" ]] || exit 0
done

python -u Preliminary/analyze.py
