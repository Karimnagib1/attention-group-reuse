# Paper-ready statistical tables

Values are point estimates with 95% confidence intervals in brackets.

## Revised Table 2: Multilingual signature similarity

| Dataset | Model | Spearman aligned | Spearman null | Spearman gap | Jaccard@10 aligned | Jaccard@10 null | Jaccard@10 gap |
|---|---|---:|---:|---:|---:|---:|---:|
| Belebele | Llama-3.1-8B | 0.448 [0.439, 0.456] | 0.191 [0.186, 0.198] | 0.256 [0.249, 0.262] | 0.513 [0.502, 0.523] | 0.354 [0.345, 0.363] | 0.159 [0.152, 0.166] |
| MMLU-ProX | Llama-3.1-8B | 0.569 [0.564, 0.574] | 0.142 [0.137, 0.149] | 0.427 [0.420, 0.433] | 0.529 [0.522, 0.535] | 0.284 [0.277, 0.291] | 0.245 [0.239, 0.250] |
| Belebele | Qwen3-8B | 0.345 [0.339, 0.350] | 0.112 [0.110, 0.116] | 0.232 [0.227, 0.237] | 0.456 [0.449, 0.464] | 0.344 [0.338, 0.350] | 0.113 [0.106, 0.119] |
| MMLU-ProX | Qwen3-8B | 0.395 [0.392, 0.398] | 0.104 [0.101, 0.107] | 0.291 [0.288, 0.294] | 0.412 [0.408, 0.417] | 0.275 [0.271, 0.279] | 0.138 [0.135, 0.141] |

## Revised Table 3: SS-weighted variance decomposition

| Dataset | Model | Language | Question | Interaction |
|---|---|---:|---:|---:|
| Belebele | Llama-3.1-8B | 0.005 [0.004, 0.005] | 0.453 [0.446, 0.459] | 0.542 [0.536, 0.549] |
| MMLU-ProX | Llama-3.1-8B | 0.003 [0.003, 0.004] | 0.598 [0.593, 0.602] | 0.399 [0.395, 0.404] |
| Belebele | Qwen3-8B | 0.014 [0.014, 0.015] | 0.407 [0.403, 0.411] | 0.578 [0.574, 0.583] |
| MMLU-ProX | Qwen3-8B | 0.008 [0.007, 0.008] | 0.459 [0.456, 0.461] | 0.534 [0.531, 0.536] |

## Revised Table 5: Multilingual latent-country case study

Rates are percentages; patched rank is in vocabulary-rank units.

| Model | Setting | Baseline top-1 (%) | Patched top-1 (%) | Top-1 change (pp) | Patched top-10 (%) | Patched median rank |
|---|---|---:|---:|---:|---:|---:|
| Qwen | Monolingual | 0.8 [0.2, 1.6] | 34.3 [29.1, 39.6] | 33.5 [28.3, 38.9] | 76.7 [72.7, 80.6] | 3.2 [2.7, 4.2] |
| Qwen | Cross-Lingual | 0.5 [0.1, 1.0] | 35.6 [30.2, 41.2] | 35.1 [29.6, 40.8] | 78.0 [73.6, 82.2] | 2.4 [2.2, 3.3] |
| Llama | Monolingual | 0.6 [0.0, 1.6] | 12.3 [8.9, 16.1] | 11.7 [8.4, 15.4] | 61.9 [57.1, 66.6] | 6.8 [5.3, 8.9] |
| Llama | Cross-Lingual | 0.4 [0.0, 1.0] | 12.9 [9.6, 16.6] | 12.6 [9.3, 16.2] | 64.6 [59.7, 69.4] | 5.9 [4.7, 7.9] |

## Permutation-test reporting note

The language-pair CSV contains the exact exceedance count, the +1-smoothed Monte Carlo p-value, and the Holm-adjusted value across the 80 paper-used pairwise tests. Report zero-exceedance results as `0/10,000 exceedances; Monte Carlo p < 1e-4` rather than as `p = 0`.
