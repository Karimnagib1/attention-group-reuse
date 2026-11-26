#!/usr/bin/env bash
# Submit the complete correctness-stratified workflow.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/correctness_stratified/run_correctness_stratified_all.sh [options]

Runs preparation now, submits the four GPU sweeps as a SLURM array, and submits
the final CPU analysis with an afterok dependency on the complete sweep array.

Options:
  --max-questions-per-k N  Override the configured positive per-K sweep cap.
  --no-cap                 Select every available question in every K stratum.
  --bootstrap-resamples N  Override final-analysis bootstrap resamples.
  --permutation-draws N    Override final-analysis permutation draws.
  -h, --help               Show this help message.
EOF
}

PREPARE_ARGS=()
ANALYSIS_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-questions-per-k)
      if [[ $# -lt 2 ]] || ! [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
        echo "--max-questions-per-k requires a positive integer." >&2
        exit 2
      fi
      PREPARE_ARGS+=("$1" "$2")
      shift 2
      ;;
    --no-cap)
      PREPARE_ARGS+=("$1")
      shift
      ;;
    --bootstrap-resamples)
      if [[ $# -lt 2 ]] || ! [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
        echo "--bootstrap-resamples requires a positive integer." >&2
        exit 2
      fi
      ANALYSIS_ARGS+=("$1" "$2")
      shift 2
      ;;
    --permutation-draws)
      if [[ $# -lt 2 ]] || ! [[ "$2" =~ ^[0-9]+$ ]]; then
        echo "--permutation-draws requires a non-negative integer." >&2
        exit 2
      fi
      ANALYSIS_ARGS+=("$1" "$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable. Run this orchestration script on the SLURM login node." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUANT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${QUANT_ROOT}/configs/correctness_stratified/analysis.yaml"
PREPARE_SCRIPT="${SCRIPT_DIR}/prepare_correctness_strata.py"
SWEEP_SCRIPT="${SCRIPT_DIR}/run_correctness_stratified_sweep.sh"
ANALYSIS_SCRIPT="${SCRIPT_DIR}/analyze_correctness_stratified.py"

for required_path in "${CONFIG_PATH}" "${PREPARE_SCRIPT}" "${SWEEP_SCRIPT}" "${ANALYSIS_SCRIPT}"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "Missing required workflow file: ${required_path}" >&2
    exit 2
  fi
done

source ~/.bashrc
conda activate thesis

cd "${QUANT_ROOT}"
export PYTHONPATH="${QUANT_ROOT}/src:${QUANT_ROOT}:${PYTHONPATH:-}"
mkdir -p logs

echo "Step 1/3: preparing correctness strata and selected IDs"
python "${PREPARE_SCRIPT}" --config "${CONFIG_PATH}" "${PREPARE_ARGS[@]}"

echo "Step 2/3: submitting four pooled GPU sweeps"
SWEEP_JOB_ID="$(sbatch --parsable --array=0-3 "${SWEEP_SCRIPT}")"
SWEEP_JOB_ID="${SWEEP_JOB_ID%%;*}"
if ! [[ "${SWEEP_JOB_ID}" =~ ^[0-9]+$ ]]; then
  echo "Could not parse sweep job ID from sbatch output: ${SWEEP_JOB_ID}" >&2
  exit 2
fi

analysis_cmd=(
  python "${ANALYSIS_SCRIPT}"
  --config "${CONFIG_PATH}"
  "${ANALYSIS_ARGS[@]}"
)
printf -v analysis_cmd_quoted '%q ' "${analysis_cmd[@]}"
printf -v quant_root_quoted '%q' "${QUANT_ROOT}"
analysis_wrap="source ~/.bashrc && conda activate thesis && cd ${quant_root_quoted} && export PYTHONPATH=${quant_root_quoted}/src:${quant_root_quoted}:\${PYTHONPATH:-} && ${analysis_cmd_quoted% }"

echo "Step 3/3: submitting final analysis after successful completion of job ${SWEEP_JOB_ID}"
ANALYSIS_JOB_ID="$(
  sbatch \
    --parsable \
    --dependency="afterok:${SWEEP_JOB_ID}" \
    --job-name=correctness_analysis \
    --output=logs/correctness_analysis_%j.log \
    --error=logs/correctness_analysis_%j.err \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=64G \
    --time=04:00:00 \
    --wrap="${analysis_wrap}"
)"
ANALYSIS_JOB_ID="${ANALYSIS_JOB_ID%%;*}"

echo "Workflow submitted."
echo "GPU sweep array job: ${SWEEP_JOB_ID}"
echo "Dependent analysis job: ${ANALYSIS_JOB_ID}"
echo "The analysis job will run only if all four GPU array tasks succeed."
