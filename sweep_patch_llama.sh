#!/bin/bash

# Always good practice in Slurm scripts
set -euo pipefail


# Make sure log directory exists (otherwise Slurm can't write logs)
mkdir -p logs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"

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



########################################
# Global config – shared across runs
########################################

MODEL_NAME="meta-llama/Meta-Llama-3.1-8B"
NUM_PAIRS=100
BATCH_SIZE=20
PATCH_MODE="group"          # "head" or "group"

PYTHON_SCRIPT="sweep_patch.py"

# default language flag (per-experiment override possible)
DEFAULT_NORMAL_LANGUAGE=0   # 0 = False, 1 = True

########################################
# Helper: run a single experiment
# args:
#   1: NORMAL_SRC
#   2: NORMAL_TGT
#   3: ALT_SRC
#   4: ALT_TGT
#   5: BASE_FOLDER
#   6: DATA1_PATH
#   7: DATA2_PATH
#   8: NORMAL_LANGUAGE (0/1)   [optional, default: DEFAULT_NORMAL_LANGUAGE]
#   9: NUM_PAIRS override       [optional, default: global NUM_PAIRS]
########################################
run_experiment() {
  local NORMAL_SRC="$1"
  local NORMAL_TGT="$2"
  local ALT_SRC="$3"
  local ALT_TGT="$4"
  local BASE_FOLDER="$5"
  local DATA1_PATH="$6"
  local DATA2_PATH="$7"
  local NORMAL_LANGUAGE="${8:-$DEFAULT_NORMAL_LANGUAGE}"
  local NUM_PAIRS_LOCAL="${9:-$NUM_PAIRS}"

  # Make sure base folder exists
  mkdir -p "$BASE_FOLDER"

  # Safe versions for filenames (spaces -> _)
  local NORMAL_SRC_SAFE="${NORMAL_SRC// /_}"
  local NORMAL_TGT_SAFE="${NORMAL_TGT// /_}"
  local ALT_SRC_SAFE="${ALT_SRC// /_}"
  local ALT_TGT_SAFE="${ALT_TGT// /_}"

  # Build output paths (same injected variables)
  # local OUT_PKL="${BASE_FOLDER}clean_icl_llama_${NUM_PAIRS_LOCAL}_${PATCH_MODE}_${NORMAL_SRC_SAFE}-${NORMAL_TGT_SAFE}_${ALT_SRC_SAFE}-${ALT_TGT_SAFE}.pkl"
  # local FIG_PATH="${BASE_FOLDER}ICL_patch_llama_${PATCH_MODE}_${NUM_PAIRS_LOCAL}_${NORMAL_SRC_SAFE}-${NORMAL_TGT_SAFE}-${ALT_SRC_SAFE}-${ALT_TGT_SAFE}.png"

  echo "=========================================="
  echo "Running experiment:"
  echo "  NORMAL_SRC       = $NORMAL_SRC"
  echo "  NORMAL_TGT       = $NORMAL_TGT"
  echo "  ALT_SRC          = $ALT_SRC"
  echo "  ALT_TGT          = $ALT_TGT"
  echo "  BASE_FOLDER      = $BASE_FOLDER"
  echo "  DATA1_PATH       = $DATA1_PATH"
  echo "  DATA2_PATH       = $DATA2_PATH"
  echo "  NORMAL_LANGUAGE  = $NORMAL_LANGUAGE"
  echo "  NUM_PAIRS        = $NUM_PAIRS_LOCAL"
  # echo "  OUT_PKL          = $OUT_PKL"
  # echo "  FIG_PATH         = $FIG_PATH"
  echo "=========================================="

  # Export env vars for Python
  export MODEL_NAME
  export BATCH_SIZE
  export PATCH_MODE

  export NORMAL_SRC NORMAL_TGT ALT_SRC ALT_TGT
  export NORMAL_LANGUAGE
  export BASE_FOLDER DATA1_PATH DATA2_PATH

  # export OUT_PKL FIG_PATH

  export NUM_PAIRS="$NUM_PAIRS_LOCAL"

  # Actually run
  python3 "$PYTHON_SCRIPT"
}

########################################
# Define experiments
########################################

# run_experiment \
#   "city" \
#   "capital" \
#   "landmark" \
#   "country calling code" \
#   "Llama_individual_language/" \
#   "/home/kareem.elzeky/thesis2/data/en_data.json" \
#   "/home/kareem.elzeky/thesis2/data/en_data.json" \
#   1   # normal_language = 0 (use alt type answer in non-normal language)

# run_experiment \
#   "city" \
#   "capital" \
#   "university" \
#   "country calling code" \
#   "Llama_individual_language/" \
#   "/home/kareem.elzeky/thesis2/data/en_data.json" \
#   "/home/kareem.elzeky/thesis2/data/en_data.json" \
#   1   # normal_language = 0 (use alt type answer in non-normal language)



# run_experiment \
#   "city" \
#   "capital" \
#   "landmark" \
#   "country calling code" \
#   "Llama_individual_language/" \
#   "/home/kareem.elzeky/thesis2/data/es_data.json" \
#   "/home/kareem.elzeky/thesis2/data/es_data.json" \
#   1   # normal_language = 0 (use alt type answer in non-normal language)

# run_experiment \
#   "city" \
#   "capital" \
#   "university" \
#   "country calling code" \
#   "Llama_individual_language/" \
#   "/home/kareem.elzeky/thesis2/data/es_data.json" \
#   "/home/kareem.elzeky/thesis2/data/es_data.json" \
#   1   # normal_language = 0 (use alt type answer in non-normal language)


# run_experiment \
#   "city" \
#   "capital" \
#   "landmark" \
#   "country calling code" \
#   "Llama_individual_language/" \
#   "$DATA_DIR/ru_data.json" \
#   "$DATA_DIR/ru_data.json" \
#   1   # normal_language = 1 (use normal-language answer source)

# run_experiment \
#   "city" \
#   "capital" \
#   "university" \
#   "country calling code" \
#   "Llama_individual_language/" \
#   "$DATA_DIR/ru_data.json" \
#   "$DATA_DIR/ru_data.json" \
#   1   # normal_language = 1 (use normal-language answer source)


# run_experiment \
#   "city" \
#   "capital" \
#   "landmark" \
#   "country calling code" \
#   "Llama_individual_language/" \
#   "$DATA_DIR/de_data.json" \
#   "$DATA_DIR/de_data.json" \
#   1   # normal_language = 1 (use normal-language answer source)

# run_experiment \
#   "city" \
#   "capital" \
#   "university" \
#   "country calling code" \
#   "Llama_individual_language/" \
#   "$DATA_DIR/de_data.json" \
#   "$DATA_DIR/de_data.json" \
#   1   # normal_language = 1 (use normal-language answer source)

# run_experiment \
#   "city" \
#   "capital" \
#   "university" \
#   "country calling code" \
#   "figs_llama_de_to_en_alt_language/" \
#   "/home/kareem.elzeky/thesis2/data/en_data.json" \
#   "/home/kareem.elzeky/thesis2/data/de_data.json" \
#   0   # normal_language = 0 (use alt type answer in non-normal language)

# run_experiment \
#   "city" \
#   "capital" \
#   "university" \
#   "country calling code" \
#   "figs_llama_en_to_de_alt_language/" \
#   "/home/kareem.elzeky/thesis2/data/de_data.json" \
#   "/home/kareem.elzeky/thesis2/data/en_data.json" \
#   0   # normal_language = 0 (use alt type answer in non-normal language)


# run_experiment \
#   "city" \
#   "capital" \
#   "landmark" \
#   "country calling code" \
#   "figs_llama_en_to_de_alt_language/" \
#   "/home/kareem.elzeky/thesis2/data/de_data.json" \
#   "/home/kareem.elzeky/thesis2/data/en_data.json" \
#   0   # normal_language = 0 (use alt type answer in non-normal language)


# run_experiment \
#   "city" \
#   "capital" \
#   "landmark" \
#   "country calling code" \
#   "figs_llama_en_to_de_normal_language/" \
#   "/home/kareem.elzeky/thesis2/data/de_data.json" \
#   "/home/kareem.elzeky/thesis2/data/en_data.json" \
#   1   # normal_language = 0 (use alt type answer in non-normal language)

# run_experiment \
#   "city" \
#   "capital" \
#   "university" \
#   "country calling code" \
#   "figs_llama_en_to_de_normal_language/" \
#   "/home/kareem.elzeky/thesis2/data/de_data.json" \
#   "/home/kareem.elzeky/thesis2/data/en_data.json" \
#   1   # normal_language = 0 (use alt type answer in non-normal language)


# run_experiment \
#   "city" \
#   "capital" \
#   "university" \
#   "country calling code" \
#   "figs_llama_de_to_en_normal_language/" \
#   "/home/kareem.elzeky/thesis2/data/en_data.json" \
#   "/home/kareem.elzeky/thesis2/data/de_data.json" \
#   1   # normal_language = 0 (use alt type answer in non-normal language)

# run_experiment \
#   "city" \
#   "capital" \
#   "landmark" \
#   "country calling code" \
#   "figs_llama_de_to_en_normal_language/" \
#   "/home/kareem.elzeky/thesis2/data/en_data.json" \
#   "/home/kareem.elzeky/thesis2/data/de_data.json" \
#   1   # normal_language = 0 (use alt type answer in non-normal language)



# run_experiment \
#   "city" \
#   "capital" \
#   "landmark" \
#   "country calling code" \
#   "figs_llama_es_to_en_alt_language/" \
#   "/home/kareem.elzeky/thesis2/data/en_data.json" \
#   "/home/kareem.elzeky/thesis2/data/es_data.json" \
#   0   # normal_language = 0 (use alt type answer in non-normal language)


# run_experiment \
#   "city" \
#   "capital" \
#   "university" \
#   "country calling code" \
#   "figs_llama_es_to_en_alt_language/" \
#   "/home/kareem.elzeky/thesis2/data/en_data.json" \
#   "/home/kareem.elzeky/thesis2/data/es_data.json" \
#   0   # normal_language = 0 (use alt type answer in non-normal language)

# run_experiment \
#   "city" \
#   "capital" \
#   "university" \
#   "country calling code" \
#   "figs_llama_es_to_de_alt_language/" \
#   "/home/kareem.elzeky/thesis2/data/de_data.json" \
#   "/home/kareem.elzeky/thesis2/data/es_data.json" \
#   0   # normal_language = 0 (use alt type answer in non-normal language)


# run_experiment \
#   "city" \
#   "capital" \
#   "landmark" \
#   "country calling code" \
#   "figs_llama_en_to_es_alt_language/" \
#   "/home/kareem.elzeky/thesis2/data/es_data.json" \
#   "/home/kareem.elzeky/thesis2/data/en_data.json" \
#   0   # normal_language = 0 (use alt type answer in non-normal language)


# run_experiment \
#   "city" \
#   "capital" \
#   "landmark" \
#   "country calling code" \
#   "figs_llama_en_to_es_normal_language/" \
#   "/home/kareem.elzeky/thesis2/data/es_data.json" \
#   "/home/kareem.elzeky/thesis2/data/en_data.json" \
#   1   # normal_language = 0 (use alt type answer in non-normal language)

# run_experiment \
#   "city" \
#   "capital" \
#   "university" \
#   "country calling code" \
#   "figs_llama_en_to_es_normal_language/" \
#   "/home/kareem.elzeky/thesis2/data/es_data.json" \
#   "/home/kareem.elzeky/thesis2/data/en_data.json" \
#   1   # normal_language = 0 (use alt type answer in non-normal language)


# run_experiment \
#   "city" \
#   "capital" \
#   "university" \
#   "country calling code" \
#   "figs_llama_es_to_en_normal_language/" \
#   "/home/kareem.elzeky/thesis2/data/en_data.json" \
#   "/home/kareem.elzeky/thesis2/data/es_data.json" \
#   1   # normal_language = 0 (use alt type answer in non-normal language)

# run_experiment \
#   "city" \
#   "capital" \
#   "landmark" \
#   "country calling code" \
#   "figs_llama_es_to_en_normal_language/" \
#   "/home/kareem.elzeky/thesis2/data/en_data.json" \
#   "/home/kareem.elzeky/thesis2/data/es_data.json" \
#   1   # normal_language = 0 (use alt type answer in non-normal language)




# run_experiment \
#   "city" \
#   "capital" \
#   "landmark" \
#   "country calling code" \
#   "figs_llama_ru_to_en_alt_language/" \
#   "/home/kareem.elzeky/thesis2/data/en_data.json" \
#   "/home/kareem.elzeky/thesis2/data/ru_data.json" \
#   0   # normal_language = 0 (use alt type answer in non-normal language)


# run_experiment \
#   "city" \
#   "capital" \
#   "university" \
#   "country calling code" \
#   "figs_llama_ru_to_en_alt_language/" \
#   "/home/kareem.elzeky/thesis2/data/en_data.json" \
#   "/home/kareem.elzeky/thesis2/data/ru_data.json" \
#   0   # normal_language = 0 (use alt type answer in non-normal language)

# run_experiment \
#   "city" \
#   "capital" \
#   "university" \
#   "country calling code" \
#   "figs_llama_en_to_ru_alt_language/" \
#   "/home/kareem.elzeky/thesis2/data/ru_data.json" \
#   "/home/kareem.elzeky/thesis2/data/en_data.json" \
#   0   # normal_language = 0 (use alt type answer in non-normal language)


# run_experiment \
#   "city" \
#   "capital" \
#   "landmark" \
#   "country calling code" \
#   "figs_llama_en_to_ru_alt_language/" \
#   "/home/kareem.elzeky/thesis2/data/ru_data.json" \
#   "/home/kareem.elzeky/thesis2/data/en_data.json" \
#   0   # normal_language = 0 (use alt type answer in non-normal language)


# run_experiment \
#   "city" \
#   "capital" \
#   "landmark" \
#   "country calling code" \
#   "figs_llama_en_to_ru_normal_language/" \
#   "/home/kareem.elzeky/thesis2/data/ru_data.json" \
#   "/home/kareem.elzeky/thesis2/data/en_data.json" \
#   1   # normal_language = 0 (use alt type answer in non-normal language)

# run_experiment \
#   "city" \
#   "capital" \
#   "university" \
#   "country calling code" \
#   "figs_llama_en_to_ru_normal_language/" \
#   "/home/kareem.elzeky/thesis2/data/ru_data.json" \
#   "/home/kareem.elzeky/thesis2/data/en_data.json" \
#   1   # normal_language = 0 (use alt type answer in non-normal language)


# run_experiment \
#   "city" \
#   "capital" \
#   "university" \
#   "country calling code" \
#   "figs_llama_ru_to_en_normal_language/" \
#   "/home/kareem.elzeky/thesis2/data/en_data.json" \
#   "/home/kareem.elzeky/thesis2/data/ru_data.json" \
#   1   # normal_language = 0 (use alt type answer in non-normal language)

# run_experiment \
#   "city" \
#   "capital" \
#   "landmark" \
#   "country calling code" \
#   "figs_llama_ru_to_en_normal_language/" \
#   "/home/kareem.elzeky/thesis2/data/en_data.json" \
#   "/home/kareem.elzeky/thesis2/data/ru_data.json" \
#   1   # normal_language = 0 (use alt type answer in non-normal language)


# run_experiment \
#   "city" \
#   "capital" \
#   "landmark" \
#   "country calling code" \
#   "Llama_individual_language/" \
#   "$DATA_DIR/ar_data.json" \
#   "$DATA_DIR/ar_data.json" \
#   1   # normal_language = 1 (use normal-language answer source)

# run_experiment \
#   "city" \
#   "capital" \
#   "university" \
#   "country calling code" \
#   "Llama_individual_language/" \
#   "$DATA_DIR/ar_data.json" \
#   "$DATA_DIR/ar_data.json" \
#   1   # normal_language = 1 (use normal-language answer source)


# run_experiment \
#   "city" \
#   "capital" \
#   "landmark" \
#   "country calling code" \
#   "Llama_individual_language/" \
#   "$DATA_DIR/de_data.json" \
#   "$DATA_DIR/de_data.json" \
#   1   # normal_language = 1 (use normal-language answer source)

# run_experiment \
#   "city" \
#   "capital" \
#   "university" \
#   "country calling code" \
#   "Llama_individual_language/" \
#   "$DATA_DIR/de_data.json" \
#   "$DATA_DIR/de_data.json" \
#   1   # normal_language = 1 (use normal-language answer source)







# run_experiment \
#   "city" \
#   "capital" \
#   "university" \
#   "country calling code" \
#   "Llama_cross_lingual/" \
#   "$DATA_DIR/ar_data.json" \
#   "$DATA_DIR/en_data.json" \
#   1   # normal_language = 0 (use alt type answer in non-normal language)

# run_experiment \
#   "city" \
#   "capital" \
#   "landmark" \
#   "country calling code" \
#   "Llama_cross_lingual/" \
#   "$DATA_DIR/ar_data.json" \
#   "$DATA_DIR/en_data.json" \
#   1   # normal_language = 0 (use alt type answer in non-normal language


# run_experiment \
#   "city" \
#   "capital" \
#   "university" \
#   "country calling code" \
#   "Llama_cross_lingual/" \
#   "$DATA_DIR/en_data.json" \
#   "$DATA_DIR/ar_data.json" \
#   1   # normal_language = 0 (use alt type answer in non-normal language)

# run_experiment \
#   "city" \
#   "capital" \
#   "landmark" \
#   "country calling code" \
#   "Llama_cross_lingual/" \
#   "$DATA_DIR/en_data.json" \
#   "$DATA_DIR/ar_data.json" \
#   1   # normal_language = 0 (use alt type answer in non-normal language








# run_experiment \
#   "city" \
#   "capital" \
#   "university" \
#   "country calling code" \
#   "Llama_cross_lingual/" \
#   "$DATA_DIR/es_data.json" \
#   "$DATA_DIR/en_data.json" \
#   1   # normal_language = 0 (use alt type answer in non-normal language)

# run_experiment \
#   "city" \
#   "capital" \
#   "landmark" \
#   "country calling code" \
#   "Llama_cross_lingual/" \
#   "$DATA_DIR/es_data.json" \
#   "$DATA_DIR/en_data.json" \
#   1   # normal_language = 0 (use alt type answer in non-normal language


# run_experiment \
#   "city" \
#   "capital" \
#   "university" \
#   "country calling code" \
#   "Llama_cross_lingual/" \
#   "$DATA_DIR/en_data.json" \
#   "$DATA_DIR/es_data.json" \
#   1   # normal_language = 0 (use alt type answer in non-normal language)

# run_experiment \
#   "city" \
#   "capital" \
#   "landmark" \
#   "country calling code" \
#   "Llama_cross_lingual/" \
#   "$DATA_DIR/en_data.json" \
#   "$DATA_DIR/es_data.json" \
#   1   # normal_language = 0 (use alt type answer in non-normal language







# run_experiment \
#   "city" \
#   "capital" \
#   "university" \
#   "country calling code" \
#   "Llama_cross_lingual/" \
#   "$DATA_DIR/de_data.json" \
#   "$DATA_DIR/en_data.json" \
#   1   # normal_language = 0 (use alt type answer in non-normal language)

# run_experiment \
#   "city" \
#   "capital" \
#   "landmark" \
#   "country calling code" \
#   "Llama_cross_lingual/" \
#   "$DATA_DIR/de_data.json" \
#   "$DATA_DIR/en_data.json" \
#   1   # normal_language = 0 (use alt type answer in non-normal language


# run_experiment \
#   "city" \
#   "capital" \
#   "university" \
#   "country calling code" \
#   "Llama_cross_lingual/" \
#   "$DATA_DIR/en_data.json" \
#   "$DATA_DIR/de_data.json" \
#   1   # normal_language = 0 (use alt type answer in non-normal language)

# run_experiment \
#   "city" \
#   "capital" \
#   "landmark" \
#   "country calling code" \
#   "Llama_cross_lingual/" \
#   "$DATA_DIR/en_data.json" \
#   "$DATA_DIR/de_data.json" \
#   1   # normal_language = 0 (use alt type answer in non-normal language






# run_experiment \
#   "city" \
#   "capital" \
#   "university" \
#   "country calling code" \
#   "Llama_cross_lingual/" \
#   "$DATA_DIR/ru_data.json" \
#   "$DATA_DIR/en_data.json" \
#   1   # normal_language = 0 (use alt type answer in non-normal language)

run_experiment \
  "city" \
  "capital" \
  "landmark" \
  "country calling code" \
  "Llama_cross_lingual/" \
  "$DATA_DIR/ru_data.json" \
  "$DATA_DIR/en_data.json" \
  1   # normal_language = 0 (use alt type answer in non-normal language


run_experiment \
  "city" \
  "capital" \
  "university" \
  "country calling code" \
  "Llama_cross_lingual/" \
  "$DATA_DIR/en_data.json" \
  "$DATA_DIR/ru_data.json" \
  1   # normal_language = 0 (use alt type answer in non-normal language)

run_experiment \
  "city" \
  "capital" \
  "landmark" \
  "country calling code" \
  "Llama_cross_lingual/" \
  "$DATA_DIR/en_data.json" \
  "$DATA_DIR/ru_data.json" \
  1   # normal_language = 0 (use alt type answer in non-normal language


