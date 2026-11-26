# Ministral-3-14B-Base on Belebele: completed run

Analysis date: 2026-09-20. The completed matching job is **264947**.
Its log records four replicas, patch batch size 16 per GPU, all 320 intervention
units completed, and saved outputs. All 320 unit checkpoint files exist and the
final `_SUCCESS` fingerprint matches the checkpoint manifest. It resumed 18
previously completed units. An earlier matching job, 264935, failed because CUDA
GPUs were not visible. Slurm accounting could not be queried from this session;
completion is established by the job log and saved artifacts.

Sources:

- [Completed job log](../../logs/ministral3_base_patch_264947.log)
- [Original run outputs](../../ministral_outputs/Ministral-3-14B-Base-2512/belebele/)
- [Paper](../../../paper/Same_questions_shared_attention_groups__A_causal_study_of_multilingual_generalization.pdf), sections 3.1–3.2, Table 1, and Appendix B.
- [Derived statistics and verification](summary.json)
- [All language-pair statistics](language_pairs.csv)
- [Reproducible analysis](analyze_saved_run.py)

## Run population and baseline

The model is `mistralai/Ministral-3-14B-Base-2512`. It has 40 layers and eight
attention groups per layer, yielding 320 intervention coordinates. Patching
used 529 of 900 questions answered correctly in all five languages (58.78%):
2,645 question-language samples and 10,580 samples before averaging the original
answer order and three perturbations. This is the model's own all-correct subset,
selected on original-order generation, consistent with the paper's subset rule.

The preceding evaluation, rather than the patch job, produced these accuracies:

| Language | Correct / total | Accuracy |
|---|---:|---:|
| English | 841 / 900 | 93.44% |
| Spanish | 790 / 900 | 87.78% |
| German | 819 / 900 | 91.00% |
| Russian | 812 / 900 | 90.22% |
| Mesopotamian Arabic (`acm_Arab`) | 593 / 900 | 65.89% |
| Overall | 3,855 / 4,500 | 85.67% |

## Main result

Using the paper's correct-versus-average-incorrect normalized margin-drop
importance, averaged across answer-order variants, then comparing signatures
across the ten unordered language pairs:

| Similarity | Aligned | Cross-language control | Within-language control | Aligned minus cross-language control [95% CI] |
|---|---:|---:|---:|---:|
| Spearman | 0.354538 | 0.219536 | 0.222766 | 0.135002 [0.131273, 0.138283] |
| Jaccard@10 | 0.532277 | 0.394711 | 0.415205 | 0.137566 [0.130618, 0.144157] |

These control statistics, confidence intervals, and permutation tests were
computed during this review from saved tensors; they were not outputs of the
submitted patch job. The job's default Jaccard summary uses **k=5**, yielding
0.432595 for the main margin metric, so it is not directly comparable with
the paper's Jaccard@10.

All ten language pairs have positive gaps for both metrics and positive lower
95% confidence bounds. Each of the 20 permutation tests had zero exceedances
in 10,000 draws: unadjusted p = 1/10,001; Holm-adjusted p = 0.0019998 across
these **20 new tests**. The paper used a separate 80-test family; these are not
claimed to be adjusted over a combined 100-test family.

## Comparison with the attached paper

Llama and Qwen values below are those reported in Table 1 of the supplied PDF.

| Model | Retained questions | Aligned Spearman | Spearman gap | Aligned Jaccard@10 | Jaccard@10 gap |
|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B | 380 | 0.448 | 0.256 | 0.513 | 0.159 |
| Qwen3-8B | 478 | 0.345 | 0.232 | 0.456 | 0.113 |
| Ministral-3-14B-Base | 529 | 0.355 | 0.135 | 0.532 | 0.138 |

The result supports an extension of the same-question similarity finding to
Ministral on Belebele. Its Spearman gap is smaller than both existing models;
its Jaccard@10 gap lies between theirs. Higher raw top-10 overlap also comes
with a higher shuffled baseline. These are descriptive model comparisons:
the models differ in size, group count, and retained question subset. No
matched-question comparison or statistical test of between-model differences
was performed.

## Verification and limits

- All saved tensor entries are finite; question-language rows are unique and
  balanced; prediction totals and all-correct IDs reconcile with the selected samples.
- Independent SciPy rank calculations reproduce every saved main-metric
  Spearman pair mean within 2.5e-7. Set-based Jaccard calculations reconcile
  the saved top-5 results and the new top-10 results.
- The analysis reuses the paper's statistical-support functions with 10,000
  question-cluster bootstrap resamples and 10,000 permutations. Bootstrap
  samples retain all five language versions and recompute both statistics.
- Cross-language control point estimates are the exact expectations under
  unrestricted question permutations, including possible fixed points, as in
  the paper's statistical-support implementation. Within-language controls
  use the exact different-question mean, excluding self-pairs. The former
  avoids Monte Carlo noise in the control point estimate.
- Three of 2,645 main-metric vectors have a tied top-10 boundary. The same
  top-k convention as the paper's statistical-support script is retained.
- The separate `rank_drop_correct` Jaccard@5 of **0.914058 is misleading as
  evidence of shared influential groups**: 2,500 of 2,645 vectors are entirely
  zero and 99.52% of entries are zero. Arbitrary identical indices selected
  from ties inflate the overlap. The main result above uses the paper's
  continuous margin-drop metric, which has no constant vectors.
- This extends the paper's all-correct similarity analysis only. It does not
  establish the correctness-stratified, layerwise, variance-decomposition,
  or latent-country findings for Ministral.

Reproduce using a Python environment with NumPy, SciPy, and PyYAML:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python review/ministral3_belebele/analyze_saved_run.py
```

Run from `quantitative_experiments/`. No model inference is required and the
original run outputs are not modified. The reviewed similarity extension is
ready to share with the stated population and comparison caveats.
