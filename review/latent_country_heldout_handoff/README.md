# Held-out latent-country evaluation: handoff package

Models: Qwen3-8B and Llama-3.1-8B.

This package answers a structured information request about the held-out latent-country evaluation. All tables were computed on CPU from the saved per-pair records of two completed runs, without loading either model:

- `review/runs/latent_country/qwen_published`
- `review/runs/latent_country/llama_published`

The script uses the runs' own bootstrap procedure and seeds. Before writing anything, it checks that the top-1 values it computes reproduce each run's saved `summary.json`: top-1 after, top-1 change and gain over random, each with its CI. All 168 checks pass (2 models × 28 rows, meaning 2 settings plus 26 conditions, × 3 values). The values also match the rebuttal table in `review/latent_country_paper_summary.md`.

To regenerate, run `~/.conda/envs/thesis/bin/python review/latent_country_heldout_handoff/make_tables.py` from the `thesis3/` root. It is deterministic and takes about 10 seconds.

## Files

| Request item | File |
|---|---|
| 1. One row per model × setting | `1_heldout_aggregate.csv` |
| Context: original Table 5 next to the held-out values | `1b_original_vs_heldout.csv` |
| 2–4. Plain-text answers | this README |
| 5. Per language pair and donor family (26 conditions per model) | `5_heldout_per_condition.csv` |

**Units:**
- Rates are percentages and changes are percentage points.
- Ranks are vocabulary ranks of the first token of the donor-country capital, written in the recipient language. Rank 1 is best.
- Ties are counted as 1 + the number of strictly higher logits.

**Columns of file 1 that the request asked for:** `n_pairs`, `top1_before`, `top1_after`, `delta_top1`, `delta_top1_ci_low`, `delta_top1_ci_high`, `top10_before`, `top10_after`, `median_rank_before`, `median_rank_after`.

**Columns added for context:**

| Column | Meaning |
|---|---|
| `n_conditions`, `n_records` | Number of conditions and of records behind each row (see below). |
| `mean_rank_before`, `mean_rank_after` | Mean target rank before and after patching. |
| `delta_top10` and its CI | Change in top-10 rate. |
| CIs for `top1_after`, `top10_after`, `median_rank_after` | Same bootstrap as the top-1 change. |
| `random_top1` | Top-1 rate when patching random group sets: 20 draws, each with the same number of groups per layer as the published set. |
| `gain_over_random` and its CI | `top1_after − random_top1`. This is the headline effect in the rebuttal. |
| `answer_token_collisions` | Number of pairs whose original and target answers share their first token. These pairs are kept. |

**What `n_pairs = 100` means.** It is the number of prompt pairs (`pair_index` 0–99). These are shared across conditions, so each pair index is reused in every language pair and donor family. Each setting averages several conditions with equal weight:
- Monolingual: 5 languages × 2 donor families = 10 conditions, 1,000 records.
- Cross-lingual: 8 directed en↔{es, de, ru, ar} pairs × 2 donor families = 16 conditions, 1,600 records.

The records are not independent observations. The bootstrap treats the 100 pair indices as the units.

**CI method.** The CI matches the method behind Table 5:
- 95% percentile paired bootstrap with 10,000 resamples of the 100 pair indices.
- Each resample uses the same indices across every condition in the setting, and before and after are computed on the same resampled pairs.
- The base seed is 31415, combined with the setting or condition name.
- The original Table 5 used the same method with different seeds.
- CIs are conditional on the fixed group sets and the frozen set of random control groups.

## Headline numbers (from file 1)

| Model | Setting | Top-1 before → after | Change in top-1 [95% CI] | Top-10 before → after | Median rank† before → after |
|---|---|---|---|---|---|
| Qwen3-8B | Monolingual | 2.0 → 29.8 | +27.8 [22.9, 32.9] | 13.6 → 73.2 | 140.9 → 3.85 |
| Qwen3-8B | Cross-lingual | 1.25 → 31.0 | +29.75 [24.9, 34.7] | 9.1 → 73.7 | 175.0 → 3.69 |
| Llama-3.1-8B | Monolingual | 1.7 → 14.5 | +12.8 [9.8, 16.0] | 12.1 → 57.4 | 133.7 → 7.95 |
| Llama-3.1-8B | Cross-lingual | 1.06 → 15.1 | +14.1 [10.9, 17.4] | 8.1 → 60.9 | 169.4 → 7.34 |

**Comparison with the original, non-held-out Table 5** (file `1b`):

| Model | Setting | Original change in top-1 [95% CI] | Held-out change [95% CI] |
|---|---|---|---|
| Qwen | Monolingual | 33.5 [28.3, 38.9] | 27.8 [22.9, 32.9] |
| Qwen | Cross-lingual | 35.1 [29.6, 40.8] | 29.75 [24.9, 34.7] |
| Llama | Monolingual | 11.7 [8.4, 15.4] | 12.8 [9.8, 16.0] |
| Llama | Cross-lingual | 12.6 [9.3, 16.2] | 14.1 [10.9, 17.4] |

- The held-out Qwen change is about 5–6 points lower. The Llama change is about 1–1.5 points higher.
- For every row, the two intervals overlap.
- No formal test of original versus held-out was run.
- Before-patching top-1 is slightly higher on the held-out prompts (for example, Qwen monolingual 0.8 → 2.0).

## Item 2: how the held-out pairs differ from the 100 localization pairs, and how they were sampled

**Plain-sentence answer.** The held-out pairs are new prompts, not new countries: no held-out prompt in any language matches a prompt from the original 100-pair schedule. The countries and relation families are the same, and the split is made by whole prompt, not by country or entity. Each held-out pair was generated from a new seed (20260917) by shuffling the entity list and drawing 21 entities (20 demonstrations plus a query). A draw was rejected and redrawn if its prompt in any language matched an original prompt, or if the donor country equalled the recipient country.

**Details:**

- **Countries:** the entity data contains only 25 countries. All 24 recipient countries and all 25 donor countries in the held-out set also appear in the original 100 pairs. **No countries are held out.**
- **Entities:** in English, the held-out recipient prompts use 68 distinct query cities. 36 of these were also query cities in the original pairs. Demonstration entities can recur too.
- **Relation families:** same as the original. Recipient task is city → capital. Donor tasks are landmark → country calling code and university → country calling code. Prompts use the same format with 20 demonstrations.
- **Language conditions:** the same 26 conditions per model as the original Table 5.
- **Excluded prompts:** the original seed schedule was replayed over all ordered language pairs, and 1,470 prompts were excluded. The replay covers that specific schedule only; it does not exclude prompts from every other historical run.
- **Entity alignment across languages:** each held-out pair uses one entity plan, rendered in all five languages. Entities are aligned by country key and list index.
- **English localization split:** the protocol also prepared a separate split of 100 English-only localization pairs, which shares no prompts with the evaluation split. **It was not used** for these results; see item 3.
- **Overlap in the original paper:** in the original experiment, localization and evaluation used the same 100 pairs. The reviewers' comment on that overlap is what motivated this held-out evaluation.

## Item 3: which attention groups were patched

**The fixed, previously published groups** — the ones in the paper's Table 4 and used for the original Table 5. Groups were **not** re-localized for each language pair or on the held-out data. Each model uses one group set for all conditions (layers and groups are zero-based):

- **Qwen3-8B:** layer 24 / group 6 and layer 31 / group 0.
- **Llama-3.1-8B:** layer 14 / group 6, layer 15 / group 4, layer 17 / group 5, layer 24 / group 0.

**One difference from the original Table 5.** In the original, the four Qwen conditions with a Russian recipient (ru→ru and en→ru, both donor families) patched only layer 24 / group 6. In the held-out evaluation, both Qwen groups were patched in every condition, including those four. Llama used the same four groups everywhere in both evaluations.

**Intervention details:**
- At the final token, the donor prompt's attention output for each selected KV group is taken **before `o_proj`** and written into the recipient's forward pass at the same position.
- Both models run in BF16 with eager attention.

## Item 4: the † on "median target rank"

The aggregate median rank is the **average of the within-condition medians**, not the median of all pooled records. For example, Qwen monolingual is the mean of the 10 condition medians. This matches how the original Table 5 summarised median rank. The per-condition rows in file 5 are ordinary medians over each condition's 100 pairs.

## Caveats

- **Steering of the first token only.** The outcome is the rank of the first tokenizer token of the donor-country capital. It is not a full generated answer or factual accuracy.
- **Answer-token collisions are kept.** These are records where the recipient's original capital and the donor-country capital share their first token: 22 of 2,600 records for Qwen and 44 of 2,600 for Llama. On these records the metric cannot tell the two answers apart.
- **What this evaluation does and does not show.** It shows held-out-*prompt* generalization of groups that were already chosen. It does not show held-out-*country* generalization or transfer from an English-only localization.
- **No multiple-comparison correction.** The per-condition CIs are pointwise. All 26 per-condition gains over random have CIs above 0 for both models.
