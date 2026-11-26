# Correctness-Stratified Usage-Signature Results

All estimates use the newly pooled correctness-stratified sweeps. The shuffled null is the exact mean over all question-by-question comparisons within each K stratum.

## Similarity by number of correctly answered languages

| Model | Dataset | Metric | K | Questions | Aligned [95% CI] | Gap [95% CI] |
|---|---|---|---:|---:|---:|---:|
| Qwen3-8B | Belebele | spearman | 0 | 32 | 0.207 [0.179, 0.237] | 0.197 [0.170, 0.225] |
| Qwen3-8B | Belebele | jaccard_at_10 | 0 | 32 | 0.231 [0.208, 0.255] | 0.106 [0.086, 0.128] |
| Qwen3-8B | Belebele | spearman | 1 | 30 | 0.153 [0.127, 0.180] | 0.146 [0.121, 0.172] |
| Qwen3-8B | Belebele | jaccard_at_10 | 1 | 30 | 0.237 [0.211, 0.266] | 0.088 [0.068, 0.107] |
| Qwen3-8B | Belebele | spearman | 2 | 42 | 0.185 [0.160, 0.210] | 0.160 [0.138, 0.181] |
| Qwen3-8B | Belebele | jaccard_at_10 | 2 | 42 | 0.292 [0.261, 0.324] | 0.099 [0.077, 0.123] |
| Qwen3-8B | Belebele | spearman | 3 | 62 | 0.176 [0.151, 0.202] | 0.143 [0.120, 0.166] |
| Qwen3-8B | Belebele | jaccard_at_10 | 3 | 62 | 0.285 [0.260, 0.310] | 0.078 [0.060, 0.097] |
| Qwen3-8B | Belebele | spearman | 4 | 250 | 0.251 [0.240, 0.261] | 0.172 [0.163, 0.181] |
| Qwen3-8B | Belebele | jaccard_at_10 | 4 | 250 | 0.374 [0.362, 0.386] | 0.087 [0.079, 0.096] |
| Qwen3-8B | Belebele | spearman | 5 | 250 | 0.351 [0.344, 0.358] | 0.224 [0.217, 0.230] |
| Qwen3-8B | Belebele | jaccard_at_10 | 5 | 250 | 0.482 [0.472, 0.493] | 0.104 [0.096, 0.112] |

## Ordinal trends

| Model | Dataset | Metric | Outcome | Slope per language [95% CI] | Holm p |
|---|---|---|---|---:|---:|
| Qwen3-8B | Belebele | spearman | aligned | 0.0417 [0.0374, 0.0459] | — |
| Qwen3-8B | Belebele | spearman | gap | 0.0135 [0.0094, 0.0174] | 0.0002 |
| Qwen3-8B | Belebele | jaccard_at_10 | aligned | 0.0577 [0.0533, 0.0622] | — |
| Qwen3-8B | Belebele | jaccard_at_10 | gap | 0.0017 [-0.0018, 0.0053] | 0.3739 |

## Matched partially correct subsets

| Model | Dataset | Match | K | Metric | Partial questions | Partial − all-correct [95% CI] |
|---|---|---|---:|---|---:|---:|
| Qwen3-8B | Belebele | confidence | 2 | spearman | 28 | -0.033 [-0.075, 0.010] |
| Qwen3-8B | Belebele | confidence | 2 | jaccard_at_10 | 28 | 0.029 [-0.031, 0.093] |
| Qwen3-8B | Belebele | confidence | 3 | spearman | 59 | -0.008 [-0.038, 0.021] |
| Qwen3-8B | Belebele | confidence | 3 | jaccard_at_10 | 59 | -0.038 [-0.070, -0.005] |
| Qwen3-8B | Belebele | confidence | 4 | spearman | 250 | -0.005 [-0.014, 0.003] |
| Qwen3-8B | Belebele | confidence | 4 | jaccard_at_10 | 250 | -0.011 [-0.023, -0.000] |
| Qwen3-8B | Belebele | difficulty | 2 | spearman | 24 | -0.069 [-0.123, -0.018] |
| Qwen3-8B | Belebele | difficulty | 2 | jaccard_at_10 | 24 | -0.035 [-0.119, 0.048] |
| Qwen3-8B | Belebele | difficulty | 3 | spearman | 48 | -0.042 [-0.075, -0.010] |
| Qwen3-8B | Belebele | difficulty | 3 | jaccard_at_10 | 48 | -0.099 [-0.148, -0.051] |
| Qwen3-8B | Belebele | difficulty | 4 | spearman | 195 | -0.011 [-0.024, 0.000] |
| Qwen3-8B | Belebele | difficulty | 4 | jaccard_at_10 | 195 | -0.012 [-0.029, 0.004] |

Confidence is an uncalibrated clean gold-option margin. Difficulty is a leave-model-out proxy based on the other model's five-language correctness count, with structural covariates. Matching uses replacement; intervals resample partial questions while holding the selected matches fixed.
