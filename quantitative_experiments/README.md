# Cross-Lingual Circuits Analysis

This is a modular evaluator for the MMLU-Prox benchmark using Hugging Face models.

## Install

```bash
pip install -r requirements.txt
```

## Run (all GPUs)

```bash
PYTHONPATH="$PWD/src:$PWD" python -m quantitative_experiments.mmluprox_eval.launch \
  --config configs/eval/model.yaml \
  --config configs/eval/data.yaml \
  --config configs/eval/eval.yaml \
  --config configs/eval/run.yaml
```

## Run (single GPU or CPU)

```bash
PYTHONPATH="$PWD/src:$PWD" python -m quantitative_experiments.mmluprox_eval.run \
  --config configs/eval/base.yaml
```

## Overrides

```bash
PYTHONPATH="$PWD/src:$PWD" python -m quantitative_experiments.mmluprox_eval.launch \
  --config configs/eval/model.yaml \
  --config configs/eval/data.yaml \
  --config configs/eval/eval.yaml \
  --config configs/eval/run.yaml \
  --set model.name_or_path=meta-llama/Llama-3.1-8B \
  --set eval.batch_size=4 \
  --set data.max_samples=100
```

## Language-specific outputs

When `data.languages` is set (default: `[en, ru, es, de, ar]`), the evaluator writes:
- `per_language.json`: accuracy stats per language.
- `all_correct_ids.json`: list of question IDs correct across all specified languages.

Outputs are written to `run.output_dir` (default: `outputs/mmluprox/`).
