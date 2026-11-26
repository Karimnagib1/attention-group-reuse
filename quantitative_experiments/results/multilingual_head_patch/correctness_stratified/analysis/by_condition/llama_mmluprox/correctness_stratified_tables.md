# Correctness-Stratified Usage-Signature Results

All estimates use the newly pooled correctness-stratified sweeps. The shuffled null is the exact mean over all question-by-question comparisons within each K stratum.

## Similarity by number of correctly answered languages

| Model | Dataset | Metric | K | Questions | Aligned [95% CI] | Gap [95% CI] |
|---|---|---|---:|---:|---:|---:|
| Llama-3.1-8B | MMLU-ProX | spearman | 0 | 250 | 0.421 [0.404, 0.437] | 0.412 [0.396, 0.429] |
| Llama-3.1-8B | MMLU-ProX | jaccard_at_10 | 0 | 250 | 0.316 [0.302, 0.331] | 0.199 [0.187, 0.212] |
| Llama-3.1-8B | MMLU-ProX | spearman | 1 | 250 | 0.376 [0.359, 0.394] | 0.345 [0.328, 0.362] |
| Llama-3.1-8B | MMLU-ProX | jaccard_at_10 | 1 | 250 | 0.319 [0.304, 0.334] | 0.167 [0.154, 0.181] |
| Llama-3.1-8B | MMLU-ProX | spearman | 2 | 250 | 0.408 [0.392, 0.425] | 0.360 [0.345, 0.375] |
| Llama-3.1-8B | MMLU-ProX | jaccard_at_10 | 2 | 250 | 0.350 [0.335, 0.365] | 0.180 [0.167, 0.192] |
| Llama-3.1-8B | MMLU-ProX | spearman | 3 | 250 | 0.439 [0.423, 0.456] | 0.362 [0.347, 0.378] |
| Llama-3.1-8B | MMLU-ProX | jaccard_at_10 | 3 | 250 | 0.391 [0.376, 0.406] | 0.193 [0.180, 0.205] |
| Llama-3.1-8B | MMLU-ProX | spearman | 4 | 250 | 0.488 [0.472, 0.502] | 0.382 [0.368, 0.396] |
| Llama-3.1-8B | MMLU-ProX | jaccard_at_10 | 4 | 250 | 0.449 [0.433, 0.464] | 0.212 [0.200, 0.224] |
| Llama-3.1-8B | MMLU-ProX | spearman | 5 | 250 | 0.593 [0.580, 0.604] | 0.402 [0.391, 0.413] |
| Llama-3.1-8B | MMLU-ProX | jaccard_at_10 | 5 | 250 | 0.555 [0.540, 0.571] | 0.239 [0.227, 0.250] |

## Ordinal trends

| Model | Dataset | Metric | Outcome | Slope per language [95% CI] | Holm p |
|---|---|---|---|---:|---:|
| Llama-3.1-8B | MMLU-ProX | spearman | aligned | 0.0350 [0.0313, 0.0386] | — |
| Llama-3.1-8B | MMLU-ProX | spearman | gap | 0.0019 [-0.0017, 0.0053] | 0.3142 |
| Llama-3.1-8B | MMLU-ProX | jaccard_at_10 | aligned | 0.0465 [0.0428, 0.0501] | — |
| Llama-3.1-8B | MMLU-ProX | jaccard_at_10 | gap | 0.0099 [0.0068, 0.0128] | 0.0002 |

## Matched partially correct subsets

| Model | Dataset | Match | K | Metric | Partial questions | Partial − all-correct [95% CI] |
|---|---|---|---:|---|---:|---:|
| Llama-3.1-8B | MMLU-ProX | confidence | 2 | spearman | 236 | -0.034 [-0.056, -0.012] |
| Llama-3.1-8B | MMLU-ProX | confidence | 2 | jaccard_at_10 | 236 | -0.018 [-0.046, 0.008] |
| Llama-3.1-8B | MMLU-ProX | confidence | 3 | spearman | 247 | -0.034 [-0.051, -0.017] |
| Llama-3.1-8B | MMLU-ProX | confidence | 3 | jaccard_at_10 | 247 | -0.020 [-0.037, -0.001] |
| Llama-3.1-8B | MMLU-ProX | confidence | 4 | spearman | 250 | -0.031 [-0.045, -0.017] |
| Llama-3.1-8B | MMLU-ProX | confidence | 4 | jaccard_at_10 | 250 | -0.011 [-0.027, 0.006] |
| Llama-3.1-8B | MMLU-ProX | difficulty | 2 | spearman | 116 | -0.059 [-0.093, -0.026] |
| Llama-3.1-8B | MMLU-ProX | difficulty | 2 | jaccard_at_10 | 116 | -0.053 [-0.099, -0.007] |
| Llama-3.1-8B | MMLU-ProX | difficulty | 3 | spearman | 140 | -0.061 [-0.086, -0.036] |
| Llama-3.1-8B | MMLU-ProX | difficulty | 3 | jaccard_at_10 | 140 | -0.090 [-0.123, -0.057] |
| Llama-3.1-8B | MMLU-ProX | difficulty | 4 | spearman | 141 | -0.038 [-0.065, -0.012] |
| Llama-3.1-8B | MMLU-ProX | difficulty | 4 | jaccard_at_10 | 141 | -0.031 [-0.062, 0.001] |

Confidence is an uncalibrated clean gold-option margin. Difficulty is a leave-model-out proxy based on the other model's five-language correctness count, with structural covariates. Matching uses replacement; intervals resample partial questions while holding the selected matches fixed.
