# Correctness-Stratified Usage-Signature Results

All estimates use the newly pooled correctness-stratified sweeps. The shuffled null is the exact mean over all question-by-question comparisons within each K stratum.

## Similarity by number of correctly answered languages

| Model | Dataset | Metric | K | Questions | Aligned [95% CI] | Gap [95% CI] |
|---|---|---|---:|---:|---:|---:|
| Llama-3.1-8B | Belebele | spearman | 0 | 43 | 0.254 [0.226, 0.283] | 0.241 [0.216, 0.269] |
| Llama-3.1-8B | Belebele | jaccard_at_10 | 0 | 43 | 0.251 [0.227, 0.276] | 0.122 [0.100, 0.145] |
| Llama-3.1-8B | Belebele | spearman | 1 | 49 | 0.186 [0.159, 0.213] | 0.178 [0.153, 0.205] |
| Llama-3.1-8B | Belebele | jaccard_at_10 | 1 | 49 | 0.239 [0.220, 0.258] | 0.097 [0.080, 0.114] |
| Llama-3.1-8B | Belebele | spearman | 2 | 68 | 0.216 [0.193, 0.241] | 0.186 [0.165, 0.206] |
| Llama-3.1-8B | Belebele | jaccard_at_10 | 2 | 68 | 0.278 [0.259, 0.297] | 0.098 [0.084, 0.112] |
| Llama-3.1-8B | Belebele | spearman | 3 | 103 | 0.236 [0.216, 0.257] | 0.173 [0.156, 0.190] |
| Llama-3.1-8B | Belebele | jaccard_at_10 | 3 | 103 | 0.322 [0.304, 0.341] | 0.099 [0.085, 0.113] |
| Llama-3.1-8B | Belebele | spearman | 4 | 250 | 0.306 [0.294, 0.318] | 0.181 [0.171, 0.191] |
| Llama-3.1-8B | Belebele | jaccard_at_10 | 4 | 250 | 0.416 [0.404, 0.428] | 0.114 [0.105, 0.123] |
| Llama-3.1-8B | Belebele | spearman | 5 | 250 | 0.465 [0.454, 0.476] | 0.242 [0.234, 0.250] |
| Llama-3.1-8B | Belebele | jaccard_at_10 | 5 | 250 | 0.545 [0.532, 0.557] | 0.141 [0.133, 0.150] |

## Ordinal trends

| Model | Dataset | Metric | Outcome | Slope per language [95% CI] | Holm p |
|---|---|---|---|---:|---:|
| Llama-3.1-8B | Belebele | spearman | aligned | 0.0572 [0.0527, 0.0617] | — |
| Llama-3.1-8B | Belebele | spearman | gap | 0.0076 [0.0035, 0.0116] | 0.0003 |
| Llama-3.1-8B | Belebele | jaccard_at_10 | aligned | 0.0699 [0.0658, 0.0741] | — |
| Llama-3.1-8B | Belebele | jaccard_at_10 | gap | 0.0080 [0.0047, 0.0113] | 0.0002 |

## Matched partially correct subsets

| Model | Dataset | Match | K | Metric | Partial questions | Partial − all-correct [95% CI] |
|---|---|---|---:|---|---:|---:|
| Llama-3.1-8B | Belebele | confidence | 2 | spearman | 63 | -0.005 [-0.043, 0.030] |
| Llama-3.1-8B | Belebele | confidence | 2 | jaccard_at_10 | 63 | -0.042 [-0.096, 0.010] |
| Llama-3.1-8B | Belebele | confidence | 3 | spearman | 103 | -0.011 [-0.029, 0.007] |
| Llama-3.1-8B | Belebele | confidence | 3 | jaccard_at_10 | 103 | -0.029 [-0.054, -0.004] |
| Llama-3.1-8B | Belebele | confidence | 4 | spearman | 250 | -0.008 [-0.018, 0.001] |
| Llama-3.1-8B | Belebele | confidence | 4 | jaccard_at_10 | 250 | -0.001 [-0.015, 0.013] |
| Llama-3.1-8B | Belebele | difficulty | 2 | spearman | 39 | -0.064 [-0.119, -0.015] |
| Llama-3.1-8B | Belebele | difficulty | 2 | jaccard_at_10 | 39 | -0.123 [-0.186, -0.061] |
| Llama-3.1-8B | Belebele | difficulty | 3 | spearman | 71 | -0.062 [-0.087, -0.036] |
| Llama-3.1-8B | Belebele | difficulty | 3 | jaccard_at_10 | 71 | -0.078 [-0.115, -0.041] |
| Llama-3.1-8B | Belebele | difficulty | 4 | spearman | 205 | -0.027 [-0.043, -0.012] |
| Llama-3.1-8B | Belebele | difficulty | 4 | jaccard_at_10 | 205 | -0.023 [-0.043, -0.003] |

Confidence is an uncalibrated clean gold-option margin. Difficulty is a leave-model-out proxy based on the other model's five-language correctness count, with structural covariates. Matching uses replacement; intervals resample partial questions while holding the selected matches fixed.
