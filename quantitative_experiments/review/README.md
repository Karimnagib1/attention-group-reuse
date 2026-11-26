# Donor-baseline ablations

This is a separate workflow for Reviewer 1's donor-baseline robustness request.
It imports the existing dataset, prompt, model, and scoring helpers without editing
them. All default experiment outputs go under this directory's `runs/` folder.

## Run

From the repository root, on a machine with the models and datasets available:

```bash
conda activate thesis
bash quantitative_experiments/review/run_donor_ablation.sh run \
  --config quantitative_experiments/review/configs/qwen_belebele.yaml
```

This selects **100 parallel questions**, each evaluated in all five languages:
500 recipient samples, or 2,000 prompts after the original plus three answer
orderings. The same recipients and orderings are used for every donor variant.
Selection uses the model/dataset's existing `all_correct_ids.json`, so this tests
the main all-correct analysis, not correctness strata. Original-order correctness
does not imply correctness under every permutation; there is no additional
permutation-correctness filter here.

There are four ready-to-use configs:

| Model | Dataset | Config |
| --- | --- | --- |
| Qwen3-8B | Belebele | `configs/qwen_belebele.yaml` |
| Qwen3-8B | MMLU-ProX | `configs/qwen_mmluprox.yaml` |
| Llama-3.1-8B | Belebele | `configs/llama_belebele.yaml` |
| Llama-3.1-8B | MMLU-ProX | `configs/llama_mmluprox.yaml` |

Change the recipient count with `--samples-per-language`, or edit
`selection.samples_per_language` in one of these new configs. Use a separate
output directory for a different experiment:

```bash
bash quantitative_experiments/review/run_donor_ablation.sh run \
  --config quantitative_experiments/review/configs/qwen_mmluprox.yaml \
  --samples-per-language 200 \
  --output-dir quantitative_experiments/review/runs/qwen_mmluprox_n200
```

To prepare and inspect the choices before loading a model, replace `run` with
`prepare`. It loads the dataset and writes `protocol.json`, containing the exact
recipient IDs, donor IDs, original and permuted prompts, resolved config, and a
fingerprint. Then use `run` with the **same config and overrides**. Preparation is
optional; `run` does it automatically if no saved protocol exists.

Config paths passed through `--config` and `--output-dir` are relative to the
current working directory. Paths inside YAML, including `base_config`,
`data.allowed_parallel_ids_path`, and `run.output_dir`, are relative to the YAML
file. For a dataset cache override, use an absolute path.

## Run with Slurm (`sbatch`)

The separate batch launcher uses the same runner, configs, output directories,
and checkpoints. From the **thesis3 repository root**:

```bash
mkdir -p quantitative_experiments/review/logs
sbatch quantitative_experiments/review/run_donor_ablation.sbatch \
  qwen_mmluprox --samples-per-language 100
```

The launcher activates `thesis` in the batch job. A previously interrupted run
with this configuration resumes automatically, reusing completed donor capture,
clean predictions, and saved layer/group units. There is no separate resume flag.
The job continues after you disconnect from SSH.

Defaults match the project's existing Slurm scripts: partition `cscc-gpu-p`, QoS
`cscc-gpu-qos`, one node, four GPUs, 16 CPUs, 180 GB RAM, and 48 hours. Override
resource requests with `sbatch` options **before** the script path, for example
`sbatch --time=24:00:00 ...`. These are resource requests, not a runtime estimate.
The first script argument chooses `qwen_mmluprox`, `qwen_belebele`,
`llama_mmluprox`, or `llama_belebele`; remaining arguments go to the runner.
When omitted, the condition defaults to `qwen_mmluprox`.

Slurm returns a job ID. Monitor it from the repository root:

```bash
squeue -j JOB_ID
tail -f quantitative_experiments/review/logs/donor_ablation_JOB_ID.out
# Errors are in donor_ablation_JOB_ID.err in the same directory.
```

The log directory must exist **before submission** because Slurm opens logs
before executing the script. If the job reaches its time limit, resubmit the
same command to continue; the in-progress layer/group unit is repeated.
`--requeue` permits scheduler-initiated requeues but does not automatically
resubmit a job at its wall-time limit. The launcher does not use the older
SIGUSR1 requeue wrapper or change the inference code, preserving the code
fingerprint of saved donor-ablation checkpoints.

## Donor definitions

The default configuration runs eight exhaustive attention-group sweeps:

| Variant | Definition | Default repetitions |
| --- | --- | --- |
| `language_mean` | Language-specific mean over the complete eligible all-correct pool, including recipients | 1 |
| `leave_one_out` | The same mean with that recipient question removed | 1 |
| `disjoint_mean_00`, etc. | Mean over 100 questions sampled from the eligible pool after excluding every recipient question | 3 |
| `random_in_language_00`, etc. | One other question's activation in the same language for each recipient | 3 |

The **100 recipient questions** and the **100 questions in each disjoint donor
pool** are separate settings. The reference mean uses the full eligible pool,
not just the 100 recipients. Thus the donor-capture stage processes more than 100
questions per language. A disjoint-pool request fails explicitly if there are
too few remaining questions; it never silently allows overlap.

Disjoint pool IDs are shared across languages to keep pool composition parallel.
The independently seeded repetitions may overlap each other but never overlap
recipients. Random donors are sampled independently for each recipient, language,
and repetition, excluding that recipient's question. Sampling uses replacement
across recipients. A random donor may be another recipient; it need not belong to
a disjoint pool. Donor selection never uses patching results.

All donors use **original-order prompts**. For a given recipient and repetition,
the donor prompt stays fixed across answer orderings, layers, and groups. Each
layer supplies its own captured activation. Leave-one-out subtracts the
recipient's original-order activation from the full language sum, then divides
by `N - 1`; it does not subtract a different activation for each answer ordering.

Means are accumulated in float64 and stored as float32 activations, then cast to
the model's dtype during patching. The reference is recomputed with the same
definition as the original mean baseline; it is not an assertion of bitwise
identity with old artifacts.

Settings can also be changed from the CLI:

```bash
bash quantitative_experiments/review/run_donor_ablation.sh run \
  --config quantitative_experiments/review/configs/llama_belebele.yaml \
  --samples-per-language 100 \
  --set donors.pool_questions=150 \
  --set donors.mean_repeats=2 \
  --set donors.random_repeats=5 \
  --set patching.batch_size=4 \
  --output-dir quantitative_experiments/review/runs/llama_belebele_custom
```

For fewer sweeps, lower the repeat counts or select strategies, keeping the
reference, for example:

```bash
--set 'donors.strategies=[language_mean,disjoint_mean,random_in_language]'
--set donors.mean_repeats=1
--set donors.random_repeats=1
```

The two seeds have separate purposes: `selection.seed` chooses recipients;
`donors.seed` chooses donor pools and random donor assignments.

## Patching and analysis

Each sweep patches one KV group at one layer, replacing only its final-token
attention output **before `o_proj`**. It uses the existing next-token option
scoring, prompt truncation, and correct-answer target. For each answer ordering,
the primary importance score is:

```text
100 * (clean_margin - patched_margin) / (abs(clean_margin) + 1e-8)
margin = correct_option_logit - mean(incorrect_option_logits)
```

Scores are averaged across orderings after computing the per-order percentage
drop. The strongest-incorrect margin and correct-answer rank change are also
saved, matching the original sweep's three importance metrics. The automatic
robustness analysis uses `pct_var_mean_incorrect` as the primary signature.

For each donor variant/repetition, the analysis reports:

- Aligned and shuffled cross-language Spearman similarity and top-10 Jaccard,
  their difference, and 95% bootstrap intervals for that difference.
- Results for every language pair and their equally weighted aggregate.
- The aggregate gap's change from the reference, with a paired bootstrap interval.
- Spearman correlation of each question/language signature with its signature
  under the reference mean, with per-language and overall summaries.

The shuffled mean is computed exactly over all question pairings, including
fixed points, rather than approximated with random permutations. Bootstrap
resampling is by parallel question ID, jointly across languages and donor
variants; both sides of the shuffled comparison are resampled. Default:
10,000 draws. Constant signatures are excluded from Spearman comparisons;
counts are reported and undefined estimates are represented as null/empty.
Top-k uses the original code's selection and tie behavior.

Intervals are conditional on the saved donor pools and assignments. Repetitions
are reported separately: they are not extra independent recipient observations,
and their signatures are not averaged before computing correlations. This
workflow provides robustness estimates, not new permutation-test p-values or a
claim that reuse causes correctness.

The run performs analysis automatically. To skip it, add `--skip-analysis`.
To rerun only the CPU analysis:

```bash
bash quantitative_experiments/review/run_donor_ablation.sh analyze \
  --run-dir quantitative_experiments/review/runs/qwen_belebele \
  --bootstrap-resamples 10000
```

## Outputs and resuming

```text
runs/<condition>/
  protocol.json                  # frozen question/donor assignments and prompts
  run_manifest.json              # code, model, protocol, and dimension identity
  donor_cache/                   # mean sums and required individual activations
  baseline.npz                   # clean logits/targets for every answer ordering
  checkpoints/<variant>/         # one atomic checkpoint per layer/group
  variants/<variant>/
    importance_tensors.npz       # [recipient samples, layers, groups] and flat forms
    samples.json
    metadata.json
  _SUCCESS                       # all sweeps exported successfully
  analysis/
    summary.json                 # aggregate gaps and signature stability
    similarity.csv              # aggregate and language-pair estimates
    signature_stability.csv
    signature_stability_per_question.csv
```

Repeat the same `run` command to resume completed layer/group units. Completed
donor capture and clean logits are reused. If interrupted during initial donor
capture, that stage restarts; if interrupted within a patch unit, only that unit
needs repeating. A changed config, code fingerprint, model identity, or protocol
requires a new output directory. A writer lock prevents simultaneous patching or
analysis of the same run directory.

Recipient count limits patching cost, not donor capture. This implementation
uses full prompt forwards and does **not** add prefix/KV caching. At the default
eight variants and four orderings, 100 questions across five languages require
4,608,000 patched prompt evaluations for Qwen or 4,096,000 for Llama, processed
in batches. Start with a smaller separate run if you need a runtime estimate.

Defaults use batch size 8 and retain the existing `dtype: auto` setting. Set
`--set model.dtype=bfloat16` for a separate run if desired. Use
`CUDA_VISIBLE_DEVICES` to control available GPUs. The original model loader
supports a single device or automatic sharding across visible GPUs;
`DataParallel` is deliberately rejected because capture hooks here are designed
for one model, possibly sharded. The activation bank is disk-backed and stores
only recipient/random-donor rows; mean accumulation streams over the full pool.

The existing `thesis` environment contains the needed dependencies. Dataset
cache paths are inherited from the existing project configs; override
`data.cache_dir` if running elsewhere. Model authentication/cache environment
variables are inherited from your shell, just as for the existing scripts.

## CPU verification

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
  -s quantitative_experiments/review/tests -v
```

Tests check donor formulas, disjointness, alignment, configuration validation,
reproducibility, score aggregation, checkpoint resumption, bootstrap/null
calculations, and patch scope/hook cleanup on tiny random Llama and Qwen models.
They do not download weights or run the full GPU experiments.
