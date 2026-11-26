# Held-out latent-country results

We evaluated the previously identified attention groups on fresh donor–recipient prompt pairs, holding the group sets fixed across all language conditions. Evaluation comprised 100 pairs per condition for two donor families and 13 directed language pairs (2,600 evaluations per model), with 20 demonstrations per prompt. Relative to 20 random group-set draws matched for group count within each selected layer, patching the published groups increased the rank-1 rate of the first token of the donor-country capital in the recipient language. Across cross-language conditions, Qwen3-8B achieved 31.00% versus 7.68% under random patching, a gain of 23.33 percentage points (95% CI [19.53, 27.10]); Llama-3.1-8B achieved 15.13% versus 0.95%, a gain of 14.18 points [11.09, 17.47]. Within-language rates were 29.80% and 14.50%, respectively, compared with random-control rates of 8.36% and 1.56%. The gain over random controls was positive in every evaluated condition for both models. These results support generalization of the published groups’ steering effects to new prompt combinations across languages. The evaluation holds out complete prompts while allowing countries and entities to recur, and measures first-token steering rather than full-answer generation.

| Model | Evaluation | Mean rank ↓, before → after | Median rank† ↓, before → after | Top-1 (%), before → after | Top-10 (%), before → after | Random top-1 (%) | Gain over random (pp), 95% CI |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3-8B | All conditions | 750.07 → 31.52 | 161.87 → 3.75 | 1.54 → 30.54 | 10.81 → 73.50 | 7.94 | +22.60 [18.94, 26.21] |
| Qwen3-8B | Within language | 725.12 → 26.92 | 140.90 → 3.85 | 2.00 → 29.80 | 13.60 → 73.20 | 8.36 | +21.45 [17.74, 25.31] |
| Qwen3-8B | Cross language | 765.67 → 34.40 | 174.97 → 3.69 | 1.25 → 31.00 | 9.06 → 73.69 | 7.68 | +23.33 [19.53, 27.10] |
| Llama-3.1-8B | All conditions | 298.25 → 37.96 | 155.65 → 7.58 | 1.31 → 14.88 | 9.65 → 59.58 | 1.18 | +13.70 [10.72, 16.82] |
| Llama-3.1-8B | Within language | 277.27 → 38.01 | 133.65 → 7.95 | 1.70 → 14.50 | 12.10 → 57.40 | 1.56 | +12.95 [9.96, 16.16] |
| Llama-3.1-8B | Cross language | 311.36 → 37.93 | 169.41 → 7.34 | 1.06 → 15.13 | 8.13 → 60.94 | 0.95 | +14.18 [11.09, 17.47] |

All rates in the table concern the donor-country capital’s first-token rank, not ordinary task accuracy. “Top-1 before” measures how often that counterfactual target already had rank 1 before intervention. Before → after compares the unpatched recipient with patching the fixed published groups. Lower ranks and higher top-10 rates indicate stronger steering. †Aggregate median rank is the average of the within-condition medians, matching the existing summary definition; it is not the median of all pooled records.

## Additional observed results

- Cross-language top-10 rates increased from 9.06% to 73.69% for Qwen and from 8.13% to 60.94% for Llama.
- Directional variation remained: English→other-language top-1 rates were 33.25% for Qwen and 22.00% for Llama; the reverse directions gave 28.75% and 8.25%. These are descriptive aggregates.
- All 26 condition-specific gains over random groups had pointwise 95% bootstrap intervals above zero in each model. These intervals are not adjusted for multiple comparisons.
- Main results retain answer-token collisions: 22/2,600 records for Qwen and 44/2,600 for Llama. In these records the original capital and donor-country capital share their first token, so this metric cannot distinguish the two answers.

## Methods and interpretation

The fixed zero-based layer/group sites were Qwen {(24,6), (31,0)} and Llama {(14,6), (15,4), (17,5), (24,0)}. Both Qwen groups were used in every condition, including Russian. No new localization was used for these completed runs. Donor tasks were landmark→calling code and university→calling code; recipients performed city→capital. Query countries differed within each pair. All five same-language directions and the eight directed English↔{Arabic, German, Russian, Spanish} directions were evaluated. Each control draw preserved the selected sites’ layer counts; draws were with replacement across group sets and could contain published groups by chance. The controls comprised 20 draws, with 14 unique sets for Qwen and 20 for Llama. Ranks used one plus the number of vocabulary logits strictly larger than the target token’s logit. All conditions received equal weight. The 95% percentile intervals used 10,000 bootstrap resamples of the 100 shared pair indices, jointly across language directions and donor families, conditional on the selected groups and frozen random-group bank. The 2,600 records per model were therefore not treated as independent observations.

Preparation excludes exact prompts from the replayed original 100-pair/20-demonstration seed schedule, as well as overlap with the prepared English-only localization split. This does not claim exclusion against every arbitrary historical run. These completed experiments establish held-out-prompt steering for previously selected groups; they do not establish held-out-country generalization, English-only localization transfer, or a causal relationship between reuse and multilingual correctness.

## Per-condition results

Each row uses 100 prompt pairs. Language arrows run from donor to recipient. Random rates average 20 draws.

| Model | Direction | Donor family | Top-1 before (%) | Random top-1 (%) | Top-1 after (%) | Gain over random (pp), 95% CI | Mean rank, before → after | Median rank, before → after | Top-10 (%), before → after |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| Qwen3-8B | ar→ar | landmark | 3.00 | 6.05 | 16.00 | +9.95 [4.90, 15.55] | 118.40 → 25.02 | 26.00 → 10.00 | 29.00 → 52.00 |
| Qwen3-8B | ar→ar | university | 4.00 | 9.20 | 23.00 | +13.80 [7.45, 20.65] | 94.79 → 14.63 | 33.50 → 5.00 | 25.00 → 68.00 |
| Qwen3-8B | ar→en | landmark | 0.00 | 2.15 | 11.00 | +8.85 [4.15, 14.10] | 871.97 → 104.80 | 247.00 → 15.50 | 2.00 → 42.00 |
| Qwen3-8B | ar→en | university | 0.00 | 4.80 | 24.00 | +19.20 [12.55, 26.40] | 794.54 → 49.73 | 216.50 → 4.00 | 1.00 → 62.00 |
| Qwen3-8B | de→de | landmark | 0.00 | 7.40 | 32.00 | +24.60 [17.65, 31.70] | 1343.51 → 28.80 | 319.50 → 3.00 | 1.00 → 78.00 |
| Qwen3-8B | de→de | university | 0.00 | 6.45 | 29.00 | +22.55 [15.60, 29.90] | 1519.05 → 37.19 | 260.50 → 3.00 | 2.00 → 78.00 |
| Qwen3-8B | de→en | landmark | 0.00 | 8.35 | 35.00 | +26.65 [19.30, 34.15] | 871.97 → 22.16 | 247.00 → 2.00 | 2.00 → 85.00 |
| Qwen3-8B | de→en | university | 0.00 | 7.10 | 34.00 | +26.90 [19.40, 34.60] | 794.54 → 21.48 | 216.50 → 2.00 | 1.00 → 84.00 |
| Qwen3-8B | en→ar | landmark | 3.00 | 9.10 | 34.00 | +24.90 [17.55, 32.30] | 118.40 → 8.18 | 26.00 → 2.00 | 29.00 → 71.00 |
| Qwen3-8B | en→ar | university | 4.00 | 10.65 | 31.00 | +20.35 [12.75, 28.20] | 94.79 → 22.64 | 33.50 → 2.50 | 25.00 → 74.00 |
| Qwen3-8B | en→de | landmark | 0.00 | 7.80 | 36.00 | +28.20 [20.65, 36.10] | 1343.51 → 23.51 | 319.50 → 2.50 | 1.00 → 74.00 |
| Qwen3-8B | en→de | university | 0.00 | 5.50 | 26.00 | +20.50 [13.55, 27.90] | 1519.05 → 52.83 | 260.50 → 4.00 | 2.00 → 73.00 |
| Qwen3-8B | en→en | landmark | 0.00 | 9.25 | 41.00 | +31.75 [24.35, 39.60] | 871.97 → 8.79 | 247.00 → 2.00 | 2.00 → 86.00 |
| Qwen3-8B | en→en | university | 0.00 | 6.90 | 34.00 | +27.10 [19.75, 34.80] | 794.54 → 21.91 | 216.50 → 2.50 | 1.00 → 82.00 |
| Qwen3-8B | en→es | landmark | 0.00 | 8.60 | 36.00 | +27.40 [19.85, 35.20] | 1188.03 → 10.92 | 126.00 → 2.00 | 3.00 → 80.00 |
| Qwen3-8B | en→es | university | 0.00 | 7.35 | 31.00 | +23.65 [16.65, 30.85] | 1235.01 → 52.95 | 147.00 → 2.50 | 3.00 → 72.00 |
| Qwen3-8B | en→ru | landmark | 5.00 | 11.95 | 34.00 | +22.05 [14.80, 29.25] | 46.69 → 7.42 | 17.00 → 3.50 | 38.00 → 86.00 |
| Qwen3-8B | en→ru | university | 8.00 | 13.95 | 38.00 | +24.05 [16.75, 31.60] | 39.21 → 9.29 | 16.00 → 2.00 | 32.00 → 82.00 |
| Qwen3-8B | es→en | landmark | 0.00 | 7.35 | 39.00 | +31.65 [23.70, 39.85] | 871.97 → 18.58 | 247.00 → 2.50 | 2.00 → 85.00 |
| Qwen3-8B | es→en | university | 0.00 | 7.75 | 35.00 | +27.25 [20.10, 34.75] | 794.54 → 47.88 | 216.50 → 3.00 | 1.00 → 78.00 |
| Qwen3-8B | es→es | landmark | 0.00 | 7.10 | 34.00 | +26.90 [19.50, 34.65] | 1188.03 → 26.62 | 126.00 → 3.00 | 3.00 → 72.00 |
| Qwen3-8B | es→es | university | 0.00 | 8.10 | 34.00 | +25.90 [18.95, 33.05] | 1235.01 → 82.28 | 147.00 → 2.00 | 3.00 → 70.00 |
| Qwen3-8B | ru→en | landmark | 0.00 | 4.25 | 21.00 | +16.75 [10.55, 23.50] | 871.97 → 50.53 | 247.00 → 6.00 | 2.00 → 60.00 |
| Qwen3-8B | ru→en | university | 0.00 | 6.15 | 31.00 | +24.85 [17.65, 32.40] | 794.54 → 47.49 | 216.50 → 3.00 | 1.00 → 71.00 |
| Qwen3-8B | ru→ru | landmark | 5.00 | 9.60 | 24.00 | +14.40 [7.85, 21.45] | 46.69 → 11.84 | 17.00 → 5.00 | 38.00 → 73.00 |
| Qwen3-8B | ru→ru | university | 8.00 | 13.50 | 31.00 | +17.50 [10.75, 24.60] | 39.21 → 12.16 | 16.00 → 3.00 | 32.00 → 73.00 |
| Llama-3.1-8B | ar→ar | landmark | 5.00 | 4.45 | 13.00 | +8.55 [3.60, 14.25] | 67.03 → 24.89 | 24.50 → 11.50 | 28.00 → 49.00 |
| Llama-3.1-8B | ar→ar | university | 6.00 | 5.95 | 16.00 | +10.05 [4.85, 16.00] | 52.61 → 21.83 | 26.00 → 8.00 | 30.00 → 56.00 |
| Llama-3.1-8B | ar→en | landmark | 0.00 | 0.00 | 4.00 | +4.00 [1.00, 8.00] | 370.98 → 77.93 | 267.00 → 25.50 | 0.00 → 33.00 |
| Llama-3.1-8B | ar→en | university | 0.00 | 0.00 | 6.00 | +6.00 [2.00, 11.00] | 365.37 → 63.03 | 191.00 → 8.50 | 3.00 → 54.00 |
| Llama-3.1-8B | de→de | landmark | 0.00 | 0.10 | 14.00 | +13.90 [7.80, 20.90] | 483.35 → 54.74 | 265.00 → 11.00 | 1.00 → 48.00 |
| Llama-3.1-8B | de→de | university | 0.00 | 0.15 | 11.00 | +10.85 [5.00, 17.55] | 486.31 → 107.20 | 184.00 → 13.50 | 4.00 → 44.00 |
| Llama-3.1-8B | de→en | landmark | 0.00 | 0.10 | 14.00 | +13.90 [7.80, 20.90] | 370.98 → 31.29 | 267.00 → 5.00 | 0.00 → 61.00 |
| Llama-3.1-8B | de→en | university | 0.00 | 0.00 | 10.00 | +10.00 [5.00, 16.00] | 365.37 → 69.40 | 191.00 → 6.00 | 3.00 → 58.00 |
| Llama-3.1-8B | en→ar | landmark | 5.00 | 4.50 | 38.00 | +33.50 [24.60, 42.60] | 67.03 → 7.03 | 24.50 → 2.00 | 28.00 → 81.00 |
| Llama-3.1-8B | en→ar | university | 6.00 | 5.30 | 20.00 | +14.70 [8.40, 21.70] | 52.61 → 11.72 | 26.00 → 5.00 | 30.00 → 69.00 |
| Llama-3.1-8B | en→de | landmark | 0.00 | 0.10 | 20.00 | +19.90 [12.80, 28.00] | 483.35 → 31.61 | 265.00 → 5.00 | 1.00 → 67.00 |
| Llama-3.1-8B | en→de | university | 0.00 | 0.15 | 9.00 | +8.85 [3.85, 14.70] | 486.31 → 53.31 | 184.00 → 11.50 | 4.00 → 48.00 |
| Llama-3.1-8B | en→en | landmark | 0.00 | 0.10 | 19.00 | +18.90 [11.90, 26.90] | 370.98 → 15.96 | 267.00 → 4.00 | 0.00 → 79.00 |
| Llama-3.1-8B | en→en | university | 0.00 | 0.10 | 10.00 | +9.90 [4.70, 16.00] | 365.37 → 28.13 | 191.00 → 6.00 | 3.00 → 61.00 |
| Llama-3.1-8B | en→es | landmark | 0.00 | 0.10 | 24.00 | +23.90 [15.90, 32.80] | 403.21 → 24.51 | 173.00 → 3.50 | 1.00 → 72.00 |
| Llama-3.1-8B | en→es | university | 0.00 | 0.00 | 15.00 | +15.00 [8.00, 22.00] | 467.98 → 46.98 | 157.00 → 8.00 | 3.00 → 57.00 |
| Llama-3.1-8B | en→ru | landmark | 3.00 | 2.55 | 33.00 | +30.45 [21.75, 39.20] | 38.88 → 7.12 | 25.00 → 2.00 | 28.00 → 82.00 |
| Llama-3.1-8B | en→ru | university | 3.00 | 2.25 | 17.00 | +14.75 [8.20, 21.90] | 36.98 → 13.59 | 24.00 → 5.00 | 23.00 → 64.00 |
| Llama-3.1-8B | es→en | landmark | 0.00 | 0.10 | 16.00 | +15.90 [9.00, 23.00] | 370.98 → 25.58 | 267.00 → 4.00 | 0.00 → 67.00 |
| Llama-3.1-8B | es→en | university | 0.00 | 0.00 | 7.00 | +7.00 [3.00, 12.00] | 365.37 → 37.76 | 191.00 → 6.50 | 3.00 → 59.00 |
| Llama-3.1-8B | es→es | landmark | 0.00 | 0.10 | 21.00 | +20.90 [13.00, 28.90] | 403.21 → 37.80 | 173.00 → 5.50 | 1.00 → 61.00 |
| Llama-3.1-8B | es→es | university | 0.00 | 0.00 | 11.00 | +11.00 [5.00, 17.00] | 467.98 → 56.02 | 157.00 → 10.00 | 3.00 → 51.00 |
| Llama-3.1-8B | ru→en | landmark | 0.00 | 0.00 | 5.00 | +5.00 [1.00, 10.00] | 370.98 → 57.32 | 267.00 → 11.50 | 0.00 → 49.00 |
| Llama-3.1-8B | ru→en | university | 0.00 | 0.00 | 4.00 | +4.00 [1.00, 8.00] | 365.37 → 48.66 | 191.00 → 8.50 | 3.00 → 54.00 |
| Llama-3.1-8B | ru→ru | landmark | 3.00 | 2.25 | 15.00 | +12.75 [6.35, 19.60] | 38.88 → 16.17 | 25.00 → 5.00 | 28.00 → 60.00 |
| Llama-3.1-8B | ru→ru | university | 3.00 | 2.35 | 15.00 | +12.65 [6.70, 19.40] | 36.98 → 17.31 | 24.00 → 5.00 | 23.00 → 65.00 |

## Sources and verification

- [Qwen summary](runs/latent_country/qwen_published/summary.json) and [run manifest](runs/latent_country/qwen_published/run_manifest.json).
- [Llama summary](runs/latent_country/llama_published/summary.json) and [run manifest](runs/latent_country/llama_published/run_manifest.json).
- [Frozen prompt protocol](runs/latent_country/protocol.json).
- Verified all 2,600 records per model against expected IDs, run fingerprints, prompt fingerprints, completion markers, and 20 controls per record.
- Independently recalculated all top-1 point estimates and all 30 aggregate top-1 confidence intervals (five metrics × three evaluation scopes × two models).
- [Before/after rank statistics](latent_country_rank_statistics.json) were computed directly from saved target ranks for all 58 aggregate and per-condition rows. All post-patching mean ranks, median ranks, and top-10 rates reproduce the existing summaries. Added before-patching statistics are point estimates; no new confidence intervals are implied.
- Verified current entity-source hashes against the saved protocol. Historical-prompt overlap was checked by replaying the original seed schedule.
- Reported numbers retain the original analysis population and weighting. No experiment code or run results were changed.
