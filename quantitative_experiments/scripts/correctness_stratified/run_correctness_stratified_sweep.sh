#!/usr/bin/env bash
#SBATCH --job-name=correctness_signature
#SBATCH --output=logs/correctness_signature_%A_%a.log
#SBATCH --error=logs/correctness_signature_%A_%a.err
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=180G
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail

source ~/.bashrc
conda activate thesis
module load nvidia/cuda/12.0

export CUDA_HOME="${CUDA_HOME:-${CUDA_DIR:-${EBROOTCUDA:-/apps/local/cuda}}}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_TOKEN_PATH="${HOME}/.cache/huggingface/token"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && [[ -f "${SLURM_SUBMIT_DIR}/src/quantitative_experiments/multilingual_head_patch/run.py" ]]; then
  QUANT_ROOT="${SLURM_SUBMIT_DIR}"
else
  QUANT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

cd "${QUANT_ROOT}"
export PYTHONPATH="${QUANT_ROOT}/src:${QUANT_ROOT}:${PYTHONPATH:-}"
mkdir -p logs

CONDITION_INDEX="${SLURM_ARRAY_TASK_ID:-${CONDITION_INDEX:-}}"
if [[ -z "${CONDITION_INDEX}" ]]; then
  echo "Set CONDITION_INDEX=0..3 for a local launch, or submit this script as a SLURM array." >&2
  exit 2
fi

case "${CONDITION_INDEX}" in
  0)
    CONDITION_ID=llama_belebele
    CONFIG_PATH=configs/correctness_stratified/belebele_sweep.yaml
    MODEL_PATH=meta-llama/Llama-3.1-8B
    ID_PATH=results/multilingual_head_patch/correctness_stratified/selection/llama_belebele/selected_ids.json
    OUTPUT_PATH=llama_outputs/belebele/multilingual_head_patch/correctness_stratified
    ;;
  1)
    CONDITION_ID=llama_mmluprox
    CONFIG_PATH=configs/correctness_stratified/mmluprox_sweep.yaml
    MODEL_PATH=meta-llama/Llama-3.1-8B
    ID_PATH=results/multilingual_head_patch/correctness_stratified/selection/llama_mmluprox/selected_ids.json
    OUTPUT_PATH=llama_outputs/mmluprox/multilingual_head_patch/correctness_stratified
    ;;
  2)
    CONDITION_ID=qwen_belebele
    CONFIG_PATH=configs/correctness_stratified/belebele_sweep.yaml
    MODEL_PATH=Qwen/Qwen3-8B
    ID_PATH=results/multilingual_head_patch/correctness_stratified/selection/qwen_belebele/selected_ids.json
    OUTPUT_PATH=qwen_outputs/belebele/multilingual_head_patch/correctness_stratified
    ;;
  3)
    CONDITION_ID=qwen_mmluprox
    CONFIG_PATH=configs/correctness_stratified/mmluprox_sweep.yaml
    MODEL_PATH=Qwen/Qwen3-8B
    ID_PATH=results/multilingual_head_patch/correctness_stratified/selection/qwen_mmluprox/selected_ids.json
    OUTPUT_PATH=qwen_outputs/mmluprox/multilingual_head_patch/correctness_stratified
    ;;
  *)
    echo "Invalid condition index '${CONDITION_INDEX}'; expected 0, 1, 2, or 3." >&2
    exit 2
    ;;
esac

for required_path in "${CONFIG_PATH}" "${ID_PATH}"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "Missing required input: ${QUANT_ROOT}/${required_path}" >&2
    exit 2
  fi
done

echo "Condition: ${CONDITION_ID}"
echo "Config: ${CONFIG_PATH}"
echo "Selected IDs: ${ID_PATH}"
echo "Output: ${OUTPUT_PATH}"

cmd=(
  python -m quantitative_experiments.multilingual_head_patch.run
  --config "${CONFIG_PATH}"
  --set "model.name_or_path=${MODEL_PATH}"
  --set "data.allowed_parallel_ids_path=${ID_PATH}"
  --set "run.output_dir=${OUTPUT_PATH}"
  "$@"
)

printf -v cmd_pretty '%q ' "${cmd[@]}"
echo "Command: ${cmd_pretty% }"
exec bash \
  "${QUANT_ROOT}/scripts/correctness_stratified/run_with_slurm_requeue.sh" \
  "${cmd[@]}"
