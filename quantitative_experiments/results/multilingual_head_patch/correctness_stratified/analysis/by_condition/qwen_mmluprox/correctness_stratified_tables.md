# Correctness-Stratified Off-Diagonal Results

The shuffled baseline is the exact mean over different-question cells only. Same-question diagonal cells are excluded. Bootstrap replicates reweight whole parallel-question IDs and recompute the aligned, null, and paired gap estimates.

## Similarity by number of correctly answered languages

| Model | Dataset | Metric | K | Questions | Aligned [95% CI] | Off-diagonal null [95% CI] | Gap [95% CI] |
|---|---|---|---:|---:|---:|---:|---:|
| Qwen3-8B | MMLU-ProX | spearman | 0 | 250 | 0.255 [0.242, 0.267] | 0.003 [0.001, 0.005] | 0.252 [0.240, 0.264] |
| Qwen3-8B | MMLU-ProX | jaccard_at_10 | 0 | 250 | 0.211 [0.199, 0.223] | 0.096 [0.090, 0.102] | 0.115 [0.106, 0.123] |
| Qwen3-8B | MMLU-ProX | spearman | 1 | 250 | 0.258 [0.245, 0.271] | 0.021 [0.017, 0.026] | 0.237 [0.225, 0.249] |
| Qwen3-8B | MMLU-ProX | jaccard_at_10 | 1 | 250 | 0.253 [0.241, 0.266] | 0.142 [0.133, 0.151] | 0.112 [0.104, 0.120] |
| Qwen3-8B | MMLU-ProX | spearman | 2 | 250 | 0.276 [0.263, 0.290] | 0.039 [0.034, 0.046] | 0.237 [0.226, 0.248] |
| Qwen3-8B | MMLU-ProX | jaccard_at_10 | 2 | 250 | 0.283 [0.271, 0.295] | 0.180 [0.169, 0.190] | 0.103 [0.096, 0.111] |
| Qwen3-8B | MMLU-ProX | spearman | 3 | 250 | 0.305 [0.293, 0.318] | 0.061 [0.055, 0.067] | 0.244 [0.234, 0.255] |
| Qwen3-8B | MMLU-ProX | jaccard_at_10 | 3 | 250 | 0.319 [0.306, 0.332] | 0.213 [0.203, 0.223] | 0.107 [0.099, 0.115] |
| Qwen3-8B | MMLU-ProX | spearman | 4 | 250 | 0.342 [0.331, 0.352] | 0.083 [0.076, 0.090] | 0.259 [0.250, 0.268] |
| Qwen3-8B | MMLU-ProX | jaccard_at_10 | 4 | 250 | 0.363 [0.351, 0.376] | 0.240 [0.229, 0.252] | 0.123 [0.116, 0.131] |
| Qwen3-8B | MMLU-ProX | spearman | 5 | 250 | 0.405 [0.396, 0.413] | 0.133 [0.124, 0.142] | 0.272 [0.263, 0.280] |
| Qwen3-8B | MMLU-ProX | jaccard_at_10 | 5 | 250 | 0.434 [0.422, 0.446] | 0.306 [0.295, 0.317] | 0.128 [0.120, 0.137] |

## Ordinal gap trends

| Model | Dataset | Metric | Slope per language [95% CI] | Holm p |
|---|---|---|---:|---:|
| Qwen3-8B | MMLU-ProX | spearman | 0.0049 [0.0024, 0.0074] | 0.0005999 |
| Qwen3-8B | MMLU-ProX | jaccard_at_10 | 0.0030 [0.0010, 0.0050] | 0.007399 |

## Matched subsets

The matched-subset estimand is unchanged from the original analysis.

Rows produced: 12.
