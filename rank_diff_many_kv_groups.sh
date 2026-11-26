#!/bin/bash

set -euo pipefail

mkdir -p logs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"

source ~/.bashrc
conda activate thesis

module load nvidia/cuda/12.0
export CUDA_HOME="${CUDA_HOME:-${CUDA_DIR:-${EBROOTCUDA:-/apps/local/cuda}}}"

NUM_PAIRS=100
NUM_EXAMPLES=20
PYTHON_SCRIPT="$SCRIPT_DIR/rank_diff_many_kv_groups.py"
DEFAULT_NORMAL_LANGUAGE=0
DEFAULT_PATCH_SPEC="24:6;31:0"
DEFAULT_PATCH_SPEC_LLAMA="14:6;15:4;17:5;24:0"


run_experiment() {
  local MODEL_NAME="$1"
  local NORMAL_SRC="$2"
  local NORMAL_TGT="$3"
  local ALT_SRC="$4"
  local ALT_TGT="$5"
  local BASE_FOLDER="$6"
  local DATA1_PATH="$7"
  local DATA2_PATH="$8"
  local PATCH_SPEC="${9:-$DEFAULT_PATCH_SPEC}"
  local NORMAL_LANGUAGE="${10:-$DEFAULT_NORMAL_LANGUAGE}"
  local NUM_PAIRS_LOCAL="${11:-$NUM_PAIRS}"
  local NUM_EXAMPLES_LOCAL="${12:-$NUM_EXAMPLES}"

  mkdir -p "$BASE_FOLDER"

  echo "=========================================="
  echo "Running rank-diff experiment:"
  echo "  MODEL_NAME       = $MODEL_NAME"
  echo "  NORMAL_SRC       = $NORMAL_SRC"
  echo "  NORMAL_TGT       = $NORMAL_TGT"
  echo "  ALT_SRC          = $ALT_SRC"
  echo "  ALT_TGT          = $ALT_TGT"
  echo "  BASE_FOLDER      = $BASE_FOLDER"
  echo "  DATA1_PATH       = $DATA1_PATH"
  echo "  DATA2_PATH       = $DATA2_PATH"
  echo "  PATCH_SPEC       = $PATCH_SPEC"
  echo "  NORMAL_LANGUAGE  = $NORMAL_LANGUAGE"
  echo "  NUM_PAIRS        = $NUM_PAIRS_LOCAL"
  echo "  NUM_EXAMPLES     = $NUM_EXAMPLES_LOCAL"
  echo "=========================================="

  export MODEL_NAME
  export NORMAL_SRC NORMAL_TGT ALT_SRC ALT_TGT
  export BASE_FOLDER DATA1_PATH DATA2_PATH
  export PATCH_SPEC NORMAL_LANGUAGE
  export NUM_PAIRS="$NUM_PAIRS_LOCAL"
  export NUM_EXAMPLES="$NUM_EXAMPLES_LOCAL"

  python3 "$PYTHON_SCRIPT"
}

# Qwen individual-language runs
run_experiment "Qwen/Qwen3-8B" "city" "capital" "university" "country calling code" "$SCRIPT_DIR/Qwen_individual_language_rank_diff/" "$DATA_DIR/ar_data.json" "$DATA_DIR/ar_data.json" "$DEFAULT_PATCH_SPEC" 1
run_experiment "Qwen/Qwen3-8B" "city" "capital" "landmark" "country calling code" "$SCRIPT_DIR/Qwen_individual_language_rank_diff/" "$DATA_DIR/ar_data.json" "$DATA_DIR/ar_data.json" "$DEFAULT_PATCH_SPEC" 1
run_experiment "Qwen/Qwen3-8B" "city" "capital" "university" "country calling code" "$SCRIPT_DIR/Qwen_individual_language_rank_diff/" "$DATA_DIR/de_data.json" "$DATA_DIR/de_data.json" "$DEFAULT_PATCH_SPEC" 1
run_experiment "Qwen/Qwen3-8B" "city" "capital" "landmark" "country calling code" "$SCRIPT_DIR/Qwen_individual_language_rank_diff/" "$DATA_DIR/de_data.json" "$DATA_DIR/de_data.json" "$DEFAULT_PATCH_SPEC" 1
run_experiment "Qwen/Qwen3-8B" "city" "capital" "university" "country calling code" "$SCRIPT_DIR/Qwen_individual_language_rank_diff/" "$DATA_DIR/en_data.json" "$DATA_DIR/en_data.json" "$DEFAULT_PATCH_SPEC" 1
run_experiment "Qwen/Qwen3-8B" "city" "capital" "landmark" "country calling code" "$SCRIPT_DIR/Qwen_individual_language_rank_diff/" "$DATA_DIR/en_data.json" "$DATA_DIR/en_data.json" "$DEFAULT_PATCH_SPEC" 1
run_experiment "Qwen/Qwen3-8B" "city" "capital" "university" "country calling code" "$SCRIPT_DIR/Qwen_individual_language_rank_diff/" "$DATA_DIR/es_data.json" "$DATA_DIR/es_data.json" "$DEFAULT_PATCH_SPEC" 1
run_experiment "Qwen/Qwen3-8B" "city" "capital" "landmark" "country calling code" "$SCRIPT_DIR/Qwen_individual_language_rank_diff/" "$DATA_DIR/es_data.json" "$DATA_DIR/es_data.json" "$DEFAULT_PATCH_SPEC" 1
run_experiment "Qwen/Qwen3-8B" "city" "capital" "university" "country calling code" "$SCRIPT_DIR/Qwen_individual_language_rank_diff/" "$DATA_DIR/ru_data.json" "$DATA_DIR/ru_data.json" "24:6" 1
run_experiment "Qwen/Qwen3-8B" "city" "capital" "landmark" "country calling code" "$SCRIPT_DIR/Qwen_individual_language_rank_diff/" "$DATA_DIR/ru_data.json" "$DATA_DIR/ru_data.json" "24:6" 1

# Qwen cross-lingual runs
run_experiment "Qwen/Qwen3-8B" "city" "capital" "university" "country calling code" "$SCRIPT_DIR/Qwen_cross_lingual_rank_diff/" "$DATA_DIR/ar_data.json" "$DATA_DIR/en_data.json" "$DEFAULT_PATCH_SPEC" 1
run_experiment "Qwen/Qwen3-8B" "city" "capital" "landmark" "country calling code" "$SCRIPT_DIR/Qwen_cross_lingual_rank_diff/" "$DATA_DIR/ar_data.json" "$DATA_DIR/en_data.json" "$DEFAULT_PATCH_SPEC" 1
run_experiment "Qwen/Qwen3-8B" "city" "capital" "university" "country calling code" "$SCRIPT_DIR/Qwen_cross_lingual_rank_diff/" "$DATA_DIR/de_data.json" "$DATA_DIR/en_data.json" "$DEFAULT_PATCH_SPEC" 1
run_experiment "Qwen/Qwen3-8B" "city" "capital" "landmark" "country calling code" "$SCRIPT_DIR/Qwen_cross_lingual_rank_diff/" "$DATA_DIR/de_data.json" "$DATA_DIR/en_data.json" "$DEFAULT_PATCH_SPEC" 1
run_experiment "Qwen/Qwen3-8B" "city" "capital" "university" "country calling code" "$SCRIPT_DIR/Qwen_cross_lingual_rank_diff/" "$DATA_DIR/en_data.json" "$DATA_DIR/ar_data.json" "$DEFAULT_PATCH_SPEC" 1
run_experiment "Qwen/Qwen3-8B" "city" "capital" "landmark" "country calling code" "$SCRIPT_DIR/Qwen_cross_lingual_rank_diff/" "$DATA_DIR/en_data.json" "$DATA_DIR/ar_data.json" "$DEFAULT_PATCH_SPEC" 1
run_experiment "Qwen/Qwen3-8B" "city" "capital" "university" "country calling code" "$SCRIPT_DIR/Qwen_cross_lingual_rank_diff/" "$DATA_DIR/en_data.json" "$DATA_DIR/de_data.json" "$DEFAULT_PATCH_SPEC" 1
run_experiment "Qwen/Qwen3-8B" "city" "capital" "landmark" "country calling code" "$SCRIPT_DIR/Qwen_cross_lingual_rank_diff/" "$DATA_DIR/en_data.json" "$DATA_DIR/de_data.json" "$DEFAULT_PATCH_SPEC" 1
run_experiment "Qwen/Qwen3-8B" "city" "capital" "university" "country calling code" "$SCRIPT_DIR/Qwen_cross_lingual_rank_diff/" "$DATA_DIR/en_data.json" "$DATA_DIR/es_data.json" "$DEFAULT_PATCH_SPEC" 1
run_experiment "Qwen/Qwen3-8B" "city" "capital" "landmark" "country calling code" "$SCRIPT_DIR/Qwen_cross_lingual_rank_diff/" "$DATA_DIR/en_data.json" "$DATA_DIR/es_data.json" "$DEFAULT_PATCH_SPEC" 1
run_experiment "Qwen/Qwen3-8B" "city" "capital" "university" "country calling code" "$SCRIPT_DIR/Qwen_cross_lingual_rank_diff/" "$DATA_DIR/en_data.json" "$DATA_DIR/ru_data.json" "$DEFAULT_PATCH_SPEC" 1
run_experiment "Qwen/Qwen3-8B" "city" "capital" "landmark" "country calling code" "$SCRIPT_DIR/Qwen_cross_lingual_rank_diff/" "$DATA_DIR/en_data.json" "$DATA_DIR/ru_data.json" "$DEFAULT_PATCH_SPEC" 1
run_experiment "Qwen/Qwen3-8B" "city" "capital" "university" "country calling code" "$SCRIPT_DIR/Qwen_cross_lingual_rank_diff/" "$DATA_DIR/es_data.json" "$DATA_DIR/en_data.json" "$DEFAULT_PATCH_SPEC" 1
run_experiment "Qwen/Qwen3-8B" "city" "capital" "landmark" "country calling code" "$SCRIPT_DIR/Qwen_cross_lingual_rank_diff/" "$DATA_DIR/es_data.json" "$DATA_DIR/en_data.json" "$DEFAULT_PATCH_SPEC" 1
run_experiment "Qwen/Qwen3-8B" "city" "capital" "university" "country calling code" "$SCRIPT_DIR/Qwen_cross_lingual_rank_diff/" "$DATA_DIR/ru_data.json" "$DATA_DIR/en_data.json" "24:6" 1
run_experiment "Qwen/Qwen3-8B" "city" "capital" "landmark" "country calling code" "$SCRIPT_DIR/Qwen_cross_lingual_rank_diff/" "$DATA_DIR/ru_data.json" "$DATA_DIR/en_data.json" "24:6" 1

# Llama individual-language runs
run_experiment "meta-llama/Meta-Llama-3.1-8B" "city" "capital" "university" "country calling code" "$SCRIPT_DIR/Llama_individual_language_rank_diff/" "$DATA_DIR/ar_data.json" "$DATA_DIR/ar_data.json" "$DEFAULT_PATCH_SPEC_LLAMA" 1
run_experiment "meta-llama/Meta-Llama-3.1-8B" "city" "capital" "landmark" "country calling code" "$SCRIPT_DIR/Llama_individual_language_rank_diff/" "$DATA_DIR/ar_data.json" "$DATA_DIR/ar_data.json" "$DEFAULT_PATCH_SPEC_LLAMA" 1
run_experiment "meta-llama/Meta-Llama-3.1-8B" "city" "capital" "university" "country calling code" "$SCRIPT_DIR/Llama_individual_language_rank_diff/" "$DATA_DIR/de_data.json" "$DATA_DIR/de_data.json" "$DEFAULT_PATCH_SPEC_LLAMA" 1
run_experiment "meta-llama/Meta-Llama-3.1-8B" "city" "capital" "landmark" "country calling code" "$SCRIPT_DIR/Llama_individual_language_rank_diff/" "$DATA_DIR/de_data.json" "$DATA_DIR/de_data.json" "$DEFAULT_PATCH_SPEC_LLAMA" 1
run_experiment "meta-llama/Meta-Llama-3.1-8B" "city" "capital" "university" "country calling code" "$SCRIPT_DIR/Llama_individual_language_rank_diff/" "$DATA_DIR/en_data.json" "$DATA_DIR/en_data.json" "$DEFAULT_PATCH_SPEC_LLAMA" 1
run_experiment "meta-llama/Meta-Llama-3.1-8B" "city" "capital" "landmark" "country calling code" "$SCRIPT_DIR/Llama_individual_language_rank_diff/" "$DATA_DIR/en_data.json" "$DATA_DIR/en_data.json" "$DEFAULT_PATCH_SPEC_LLAMA" 1
run_experiment "meta-llama/Meta-Llama-3.1-8B" "city" "capital" "university" "country calling code" "$SCRIPT_DIR/Llama_individual_language_rank_diff/" "$DATA_DIR/es_data.json" "$DATA_DIR/es_data.json" "$DEFAULT_PATCH_SPEC_LLAMA" 1
run_experiment "meta-llama/Meta-Llama-3.1-8B" "city" "capital" "landmark" "country calling code" "$SCRIPT_DIR/Llama_individual_language_rank_diff/" "$DATA_DIR/es_data.json" "$DATA_DIR/es_data.json" "$DEFAULT_PATCH_SPEC_LLAMA" 1
run_experiment "meta-llama/Meta-Llama-3.1-8B" "city" "capital" "university" "country calling code" "$SCRIPT_DIR/Llama_individual_language_rank_diff/" "$DATA_DIR/ru_data.json" "$DATA_DIR/ru_data.json" "$DEFAULT_PATCH_SPEC_LLAMA" 1
run_experiment "meta-llama/Meta-Llama-3.1-8B" "city" "capital" "landmark" "country calling code" "$SCRIPT_DIR/Llama_individual_language_rank_diff/" "$DATA_DIR/ru_data.json" "$DATA_DIR/ru_data.json" "$DEFAULT_PATCH_SPEC_LLAMA" 1

# Llama cross-lingual runs
run_experiment "meta-llama/Meta-Llama-3.1-8B" "city" "capital" "university" "country calling code" "$SCRIPT_DIR/Llama_cross_lingual_rank_diff/" "$DATA_DIR/ar_data.json" "$DATA_DIR/en_data.json" "$DEFAULT_PATCH_SPEC_LLAMA" 1
run_experiment "meta-llama/Meta-Llama-3.1-8B" "city" "capital" "landmark" "country calling code" "$SCRIPT_DIR/Llama_cross_lingual_rank_diff/" "$DATA_DIR/ar_data.json" "$DATA_DIR/en_data.json" "$DEFAULT_PATCH_SPEC_LLAMA" 1
run_experiment "meta-llama/Meta-Llama-3.1-8B" "city" "capital" "university" "country calling code" "$SCRIPT_DIR/Llama_cross_lingual_rank_diff/" "$DATA_DIR/de_data.json" "$DATA_DIR/en_data.json" "$DEFAULT_PATCH_SPEC_LLAMA" 1
run_experiment "meta-llama/Meta-Llama-3.1-8B" "city" "capital" "landmark" "country calling code" "$SCRIPT_DIR/Llama_cross_lingual_rank_diff/" "$DATA_DIR/de_data.json" "$DATA_DIR/en_data.json" "$DEFAULT_PATCH_SPEC_LLAMA" 1
run_experiment "meta-llama/Meta-Llama-3.1-8B" "city" "capital" "university" "country calling code" "$SCRIPT_DIR/Llama_cross_lingual_rank_diff/" "$DATA_DIR/en_data.json" "$DATA_DIR/ar_data.json" "$DEFAULT_PATCH_SPEC_LLAMA" 1
run_experiment "meta-llama/Meta-Llama-3.1-8B" "city" "capital" "landmark" "country calling code" "$SCRIPT_DIR/Llama_cross_lingual_rank_diff/" "$DATA_DIR/en_data.json" "$DATA_DIR/ar_data.json" "$DEFAULT_PATCH_SPEC_LLAMA" 1
run_experiment "meta-llama/Meta-Llama-3.1-8B" "city" "capital" "university" "country calling code" "$SCRIPT_DIR/Llama_cross_lingual_rank_diff/" "$DATA_DIR/en_data.json" "$DATA_DIR/de_data.json" "$DEFAULT_PATCH_SPEC_LLAMA" 1
run_experiment "meta-llama/Meta-Llama-3.1-8B" "city" "capital" "landmark" "country calling code" "$SCRIPT_DIR/Llama_cross_lingual_rank_diff/" "$DATA_DIR/en_data.json" "$DATA_DIR/de_data.json" "$DEFAULT_PATCH_SPEC_LLAMA" 1
run_experiment "meta-llama/Meta-Llama-3.1-8B" "city" "capital" "university" "country calling code" "$SCRIPT_DIR/Llama_cross_lingual_rank_diff/" "$DATA_DIR/en_data.json" "$DATA_DIR/es_data.json" "$DEFAULT_PATCH_SPEC_LLAMA" 1
run_experiment "meta-llama/Meta-Llama-3.1-8B" "city" "capital" "landmark" "country calling code" "$SCRIPT_DIR/Llama_cross_lingual_rank_diff/" "$DATA_DIR/en_data.json" "$DATA_DIR/es_data.json" "$DEFAULT_PATCH_SPEC_LLAMA" 1
run_experiment "meta-llama/Meta-Llama-3.1-8B" "city" "capital" "university" "country calling code" "$SCRIPT_DIR/Llama_cross_lingual_rank_diff/" "$DATA_DIR/en_data.json" "$DATA_DIR/ru_data.json" "$DEFAULT_PATCH_SPEC_LLAMA" 1
run_experiment "meta-llama/Meta-Llama-3.1-8B" "city" "capital" "landmark" "country calling code" "$SCRIPT_DIR/Llama_cross_lingual_rank_diff/" "$DATA_DIR/en_data.json" "$DATA_DIR/ru_data.json" "$DEFAULT_PATCH_SPEC_LLAMA" 1
run_experiment "meta-llama/Meta-Llama-3.1-8B" "city" "capital" "university" "country calling code" "$SCRIPT_DIR/Llama_cross_lingual_rank_diff/" "$DATA_DIR/es_data.json" "$DATA_DIR/en_data.json" "$DEFAULT_PATCH_SPEC_LLAMA" 1
run_experiment "meta-llama/Meta-Llama-3.1-8B" "city" "capital" "landmark" "country calling code" "$SCRIPT_DIR/Llama_cross_lingual_rank_diff/" "$DATA_DIR/es_data.json" "$DATA_DIR/en_data.json" "$DEFAULT_PATCH_SPEC_LLAMA" 1
run_experiment "meta-llama/Meta-Llama-3.1-8B" "city" "capital" "university" "country calling code" "$SCRIPT_DIR/Llama_cross_lingual_rank_diff/" "$DATA_DIR/ru_data.json" "$DATA_DIR/en_data.json" "$DEFAULT_PATCH_SPEC_LLAMA" 1
run_experiment "meta-llama/Meta-Llama-3.1-8B" "city" "capital" "landmark" "country calling code" "$SCRIPT_DIR/Llama_cross_lingual_rank_diff/" "$DATA_DIR/ru_data.json" "$DATA_DIR/en_data.json" "$DEFAULT_PATCH_SPEC_LLAMA" 1

