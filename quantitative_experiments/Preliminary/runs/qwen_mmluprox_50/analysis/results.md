# Head vs group patching, Qwen3-8B / MMLU-ProX, 50 questions

Means over the 10 language pairs; 95% percentile bootstrap CI over questions (10,000 resamples).

| Level | Metric | Aligned | Cross-language control | Gap [95% CI] |
|---|---|---:|---:|---|
| group (paper run, saved) | spearman | 0.3990 | 0.1097 | 0.2893 [0.2615, 0.3045] |
| group (paper run, saved) | jaccard_at_10 | 0.4065 | 0.2722 | 0.1343 [0.1178, 0.1459] |
| head (1,152 heads) | spearman | 0.3859 | 0.0902 | 0.2957 [0.2713, 0.3077] |
| head (1,152 heads) | jaccard_at_10 | 0.5385 | 0.3986 | 0.1399 [0.1162, 0.1576] |
| head summed within group (288) | spearman | 0.4002 | 0.1142 | 0.2860 [0.2585, 0.3013] |
| head summed within group (288) | jaccard_at_10 | 0.4499 | 0.3165 | 0.1335 [0.1158, 0.1458] |

- Spearman(group importance, head importance summed within group) — group (paper run, saved): 0.9670 [0.9653, 0.9687]
