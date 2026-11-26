# Same Question, Shared Attention Groups? A Causal Study of Cross-Lingual Generalization

Code and data for the paper by Kareem Elzeky, Kentaro Inui, Ekaterina Shutova and Thamar Solorio.

The paper builds an *attention-group usage signature* for each question in each language. It patches the final-token output of one grouped-query-attention group at a time and records how much the model's preference for the correct answer changes. It then compares these signatures across translations of the same question. The repository contains the code for every experiment and the saved results behind every table and figure. Every number in the paper can be recomputed on CPU from these results; GPU is needed only to regenerate the results themselves.

Model weights and benchmark data are not included. The models and benchmarks are downloaded from the Hugging Face Hub: Qwen/Qwen3-8B, meta-llama/Llama-3.1-8B (gated), mistralai/Ministral-3-14B-Base-2512, li-lab/MMLU-ProX and facebook/belebele. The saved results store only question IDs, labels and numbers, not question text.

## Layout

The repository has two parts. Paths inside each part are relative, so run commands from the directory shown.

- **`quantitative_experiments/`** contains the evaluation, the usage-signature sweeps, the statistics, the correctness-stratified analysis, the robustness checks and the paper figures.
- **The repository root** contains the multilingual latent-country case study (§3.5, §5.4, Appendix E).

### Where each paper result comes from

Paths in the main-analyses tables are relative to `quantitative_experiments/`.

**Main analyses**

| Paper | Code | Saved results |
|---|---|---|
| Table 6 (accuracy, all-correct sets) | `src/quantitative_experiments/mmluprox_eval/`, `configs/eval/`, `scripts/eval/` | `{qwen,llama}_outputs/{mmluprox,belebele}/{predictions.jsonl,metrics.json,per_language.json,all_correct_ids.json}` |
| Usage signatures (§3.1) | `src/quantitative_experiments/multilingual_head_patch/run.py`, `configs/multilingual_head_patch/patch/`, `scripts/multilingual_head_patch/run_patch_*.sh` | `*_outputs/*/multilingual_head_patch/test/importance_tensors.npz` |
| Tables 1 and 7, Table 2 CIs | `scripts/statistical_support/compute_paper_statistics.py`, `configs/statistical_support/paper_statistics.yaml` | `results/multilingual_head_patch/statistical_support/` |
| Saved 500-shuffle baselines read by the statistics script | `multilingual_head_patch/run_baseline.py`, `configs/multilingual_head_patch/baseline/baseline_config.yaml` | `*/multilingual_head_patch/test_baselines/baseline_report.json` |
| Table 2 and Figure 9 (variance decomposition) | `multilingual_head_patch/variance_decomposition_continuous.py`, `paper_visualizations/scripts/plot_model_variance_fractions.py` | `*/test/variance_decomposition_continuous/pct_var_mean_incorrect/`, `paper_visualizations/figures/variance_fraction_by_layer_*` |
| Figure 3 (layerwise similarity) | `multilingual_head_patch/run_baseline_layerwise.py`, `paper_visualizations/scripts/plot_layerwise_signature_similarity_ci.py` | `*/test_baselines_layerwise/`, `paper_visualizations/figures/layerwise_signature_similarity_gap_ci.*` |
| Figures 4–7 (correctness-stratified) | `scripts/correctness_stratified/`, `configs/correctness_stratified/` | `*/multilingual_head_patch/correctness_stratified/`, `results/multilingual_head_patch/correctness_stratified/` |
| Figure 8 (example signatures) | `scripts/correctness_stratified/visualize_matched_signature_overlap.py` | `results/.../correctness_stratified/analysis/usage_signature_overlap/qwen_mmluprox/` |
| Table 8, Appendix B.5 (donor robustness) | `review/donor_ablation/`, `review/donor_parallel/`, `review/configs/qwen_mmluprox.yaml`, `review/run_donor_parallel.*` | `review/runs/qwen_mmluprox_parallel_float32/`, `review/reports/qwen_mmluprox_donor_handoff/` |
| Tables 5, 6 and 9, Appendix B.6 (Ministral-3-14B) | `src/quantitative_experiments/ministral3*.py`, `configs/ministral3/`, `scripts/ministral3/`, `review/ministral3_belebele/analyze_saved_run.py` | `ministral_outputs/Ministral-3-14B-Base-2512/belebele/`, `review/ministral3_belebele/` |
| Table 10, Appendix B.7 (head-level patching) | `Preliminary/` | `Preliminary/runs/qwen_mmluprox_50/` |

**Latent-country case study** (paths relative to the repository root)

| Paper | Code | Saved results |
|---|---|---|
| Dataset (Appendix E.1) | `ICL_prompt_generator.py` | `data/{en,es,de,ru,ar}_data.json` |
| Table 3 (localization) | `sweep_patch.py`, `sweep_patch_{qwen,llama}.sh` | `{Qwen,Llama}_{individual_language,cross_lingual}/` |
| Table 12 (localization pairs) | `rank_diff_many_kv_groups.py`, `.sh`, `plot_rank_diff_results.py` | `*_rank_diff/`, `paper_summary_final/`; CIs in `quantitative_experiments/results/.../statistical_support/table5_latent_country_aggregate.csv` |
| Tables 4 and 11 (held-out pairs) | `review/latent_country/`, `review/run_latent_country.sh`, `review/configs/latent_country_*.json`, `review/latent_country_heldout_handoff/make_tables.py` | `review/runs/latent_country/`, `review/latent_country_heldout_handoff/` |

## Setup

`environment.yml` is the conda environment used for the experiments. Create it with:

```bash
conda env create -f environment.yml
```

Alternatively, install `quantitative_experiments/requirements.txt` plus numpy, scipy, pandas, matplotlib and pytest. The latent-country localization and Table 12 scripts also need `nnsight`. Ministral-3 needs `requirements-ministral3.txt` (`transformers[mistral-common]==5.0.0`).

Models and datasets are cached under `$HF_HOME` (default `~/.cache/huggingface`). Downloading Llama-3.1-8B requires accepting its license on the Hugging Face Hub.

The shell launchers are Slurm batch scripts for the cluster we used. Adjust the `#SBATCH` partition and QoS lines, and the `conda activate` / `module load` lines, for your system. Each one can also run directly with `bash`.

## Recomputing the paper's numbers from saved results (CPU)

From `quantitative_experiments/`:

```bash
export PYTHONPATH="$PWD/src:$PWD"

# Tables 1, 2, 7 and 12 CIs, permutation tests with Holm correction.
# The launcher checks the hostname; ALLOW_NON_CN09=1 skips that check.
ALLOW_NON_CN09=1 bash scripts/statistical_support/run_paper_statistics.sh

# Figures 3 and 9
python paper_visualizations/scripts/plot_layerwise_signature_similarity_ci.py
python paper_visualizations/scripts/plot_model_variance_fractions.py

# Figures 4-7
bash scripts/correctness_stratified/analyze_correctness_stratified_condition_offdiagonal.sh
python scripts/correctness_stratified/visualize_correctness_stratified_aggregate.py

# Table 10 (head vs group patching)
python Preliminary/analyze.py

# Table 9 (Ministral-3-14B on Belebele)
python review/ministral3_belebele/analyze_saved_run.py
```

From the repository root:

```bash
# Tables 4 and 11
python review/latent_country_heldout_handoff/make_tables.py

# Table 12 summary tables and plots
python plot_rank_diff_results.py
```

**Donor-robustness analysis (Table 8).** The saved results are in `review/runs/qwen_mmluprox_parallel_float32/`:
- per-donor importance tensors in `variants/`
- analysis outputs in `analysis/`
- the tables in `review/reports/qwen_mmluprox_donor_handoff/`

The run's `protocol.json` is not included because it stores full MMLU-ProX question text, and the analysis scripts need it. To recompute Table 8, rerun the donor experiment (see below).

## Regenerating the results (GPU)

All paths below are relative to `quantitative_experiments/`.

**Evaluation (Table 6).**

```bash
bash scripts/eval/eval_mmluprox.sh --set model.name_or_path=<model> --set run.output_dir=<qwen|llama>_outputs/mmluprox
```

Use `eval_belebele.sh` for Belebele. The questions answered correctly in all five languages are written to `all_correct_ids.json`.

**Usage-signature sweeps.** `scripts/multilingual_head_patch/run_patch_{mmluprox,belebele}.sh` launch them. The configs are set up for Qwen3-8B. For Llama-3.1-8B, override three settings with `--set`: `model.name_or_path`, `data.allowed_parallel_ids_path` (to `llama_outputs/<dataset>/all_correct_ids.json`) and `run.output_dir`.

The paper sweeps ran in **float32**: they used `dtype: auto`, which Transformers 4.x loads as float32. With Transformers 5.x, `auto` loads bfloat16 instead, so add `--set model.dtype=float32`.

**Per-condition analyses.** The baseline, layerwise-baseline and variance-decomposition configs each name one condition. Set their `input_dir` and `output_dir` (or `run_output_dir`) to each of the four `{qwen,llama}_outputs/{mmluprox,belebele}/multilingual_head_patch/test` directories in turn.

**Correctness-stratified sweeps.**

```bash
python scripts/correctness_stratified/prepare_correctness_strata.py
CONDITION_INDEX=<0..3> bash scripts/correctness_stratified/run_correctness_stratified_sweep.sh
```

**Robustness checks.**

| Check | Command |
|---|---|
| Ministral-3-14B (Appendix B.6) | `bash scripts/ministral3/evaluate_base.sh belebele`, then `patch_base.sh belebele` |
| Donor robustness (Appendix B.5) | `sbatch review/run_donor_parallel.sbatch qwen_mmluprox --samples-per-language 100 --batch-size 8 --donor-batch-size 8` (4 GPUs, float32) |
| Head-level patching (Appendix B.7) | `Preliminary/run.sh` |

**Latent-country case study** (from the repository root):
- **Localization:** `sweep_patch_{qwen,llama}.sh` run `sweep_patch.py`, with settings passed as environment variables.
- **Steering on the localization pairs:** `rank_diff_many_kv_groups.sh`.
- **Held-out evaluation:** see `review/README.md`.

## Tests

```bash
cd quantitative_experiments && PYTHONPATH=src:. python -m pytest tests review/tests
cd .. && PYTHONPATH=. python -m pytest review/tests
```

These are CPU tests with tiny randomly initialized models; they do not download weights.

## Citation

Please cite the paper if you use this code or data.
