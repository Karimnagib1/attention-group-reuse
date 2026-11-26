# Donor-robustness experiment: Qwen3-8B / MMLU-ProX, handoff package

This folder answers a structured information request about the donor-robustness experiment. Every number comes from a single completed run. Nothing was rerun on a GPU. The tables were computed on CPU from the saved importance tensors, using the run's own similarity and bootstrap code. The script checks that it reproduces every gap and CI in the run's saved analysis before writing anything.

- **Run:** `quantitative_experiments/review/runs/qwen_mmluprox_parallel_float32` (Slurm job 259028, FP32, all 2,304 patch units completed)
- **Protocol SHA-256:** `e2469cb819fc58de017718bb82d8b7e52d5f7c1f468baa708743960ec9589723`
- **Run fingerprint:** `759d969c2f5c5e60707a7f50ad207ade63e0b08b20f165f615faa7ea6cc4fd76`
- **Regenerate:** run `~/.conda/envs/thesis/bin/python review/reports/qwen_mmluprox_donor_handoff/make_tables.py` from `quantitative_experiments/`. It is deterministic and takes about 1 minute on CPU.

## Files

| Request item | File | Rows |
|---|---|---|
| 1. Results table | `1_results_table.csv` | 8 donors × 2 metrics = 16 |
| 2. Per-language-pair gaps | `2_per_pair_gaps.csv` | 8 × 2 × 10 = 160 |
| 2. Minimum per-pair gap | `2b_min_pair_gap.csv` | 16 |
| 3. Setup | this README, "Setup" section | n/a |
| 4. Mean \|importance\| | `4_importance_magnitude.csv` | 8 donors × (5 languages + ALL) |
| 5. Test-retest across random seeds | `5_test_retest_across_repeats.csv` | random pairs, plus disjoint pairs for comparison |
| 6. Agreement with the reference signature | `6_agreement_with_reference.csv` | 8 donors × (5 languages + ALL) |
| Reproducibility | `make_tables.py` | n/a |

**Donor labels:**

| Label | Variant name in the run directory |
|---|---|
| `reference_mean` | `language_mean` |
| `leave_one_out` | `leave_one_out` |
| `disjoint_1`, `disjoint_2`, `disjoint_3` | `disjoint_mean_00`, `disjoint_mean_01`, `disjoint_mean_02` |
| `random_1`, `random_2`, `random_3` | `random_in_language_00`, `random_in_language_01`, `random_in_language_02` |

**Metrics:**
- `spearman` is the Spearman correlation between two 288-dimensional signatures (36 layers × 8 KV groups).
- `jaccard_at_10` is the Jaccard overlap of the 10 groups with the highest signed importance in each signature.

## Column definitions (files 1 and 2)

A **signature** is the 288-dimensional importance vector for one (question, language) sample. For a language pair (a, b), let M[q, q'] be the similarity between question q's signature in language a and question q''s signature in language b. There are 100 questions.

| Column | Definition |
|---|---|
| `aligned_mean` | Mean of the diagonal of M, i.e. the same question in both languages. |
| `cross_lang_control_mean` | Mean over **all 100 × 100** entries of M, including the 100 diagonal entries. This is the exact expected value under a uniform random shuffle of question pairing, and it is the control used in the run's saved analysis. |
| `cross_lang_control_offdiag_mean` | Mean over the 9,900 off-diagonal entries only, i.e. strictly different questions. Added for this handoff. |
| `within_lang_control_mean` | **Added for this handoff; the original analysis did not include it.** For each language, the mean similarity between signatures of *different* questions in that language (off-diagonal of the within-language matrix). The per-pair value is the average over the pair's two languages. The table-1 value is the average over the 10 pairs, which equals the plain average over the 5 languages. |
| `gap` | `aligned_mean − cross_lang_control_mean`. This is the headline gap, as specified in the request. |
| `gap_ci_low`, `gap_ci_high` | 95% percentile bootstrap CI of `gap` (see "Setup", item g). |
| `gap_vs_offdiag_control` | `aligned_mean − cross_lang_control_offdiag_mean`. About 0.001–0.003 larger than `gap`. |
| `gap_minus_reference` and its CI | Paired difference between this donor's gap and the reference-mean gap. The bootstrap uses the same resampled questions for both. |
| `perm_p_one_sided`, `perm_p_holm_16` / `perm_p_holm_160` | Permutation test added for this handoff (see "Setup", item g). |

In table 1, every value is an equal-weight average over the 10 language pairs. `2b_min_pair_gap.csv` gives the smallest and largest per-pair gap for each donor and metric. It also gives the number of pairs (out of 10) whose CI excludes 0, which is 10 for every donor and metric. The smallest gap is always en–ar.

## Setup

### a. Choosing the 100 questions
- **Pool:** the 1,839 questions Qwen3-8B answered correctly in all five languages (en, ru, es, de, ar), from `qwen_outputs/mmluprox/all_correct_ids.json`.
- **Correctness criterion:** correctness was judged with the original answer order only. Correctness under the three permuted orders was not required.
- **Sampling:** each ID was ranked by SHA-256 of `[42, "recipient", question_id]` and the 100 lowest were kept. This is deterministic and equivalent to uniform sampling without replacement. The selection seed is 42.
- **Recipient set:** each of the 100 questions in all 5 languages, giving 500 samples.

### b. Reference mean
The reference mean is the per-language mean of donor activations over **all 1,839 pool questions**, including the 100 recipients. It is not a mean over the 100 recipients only.

### c. Leave-one-out
Leave-one-out uses the same 1,839-question pool, minus only the current recipient question in the same language. The formula is (sum over 1,839 − recipient's own activation) / 1,838.

### d. Disjoint donors
- **Pools:** there are three pools, and each contains n = 100 questions.
- **Exclusion:** every pool excludes **all 100 recipients**. Each pool is drawn from the remaining 1,739 questions.
- **Draw method:** pool r keeps the 100 lowest IDs ranked by SHA-256 of `[123, "disjoint-r", question_id]`, with r = 0, 1, 2 and donor seed 123.
- **Overlap:** the three pools were drawn independently, so they overlap each other slightly (pools 1 and 2 share 7 questions, pools 1 and 3 share 5, pools 2 and 3 share 9).
- **Mean:** the donor is the per-language mean over the pool, so each language's mean uses the same 100 questions.

### e. Random donors
- **One donor per recipient sample:** a donor question is drawn separately for every recipient (question, language) sample and every repeat. There is no single fixed donor question.
- **Pool:** the donor is a uniform draw from the other 1,838 questions in the 1,839 pool, in the **same language** as the recipient. The recipient itself is excluded. Other recipients are not excluded, and 23–27 of the 500 draws per repeat landed on another recipient.
- **Seeds:** each draw's generator is seeded by hashing `(123, repeat, question_id, language)`, with repeat = 0, 1, 2 and base donor seed 123. So "seeds 1–3" are repeats 0–2 under one base seed, not three unrelated integer seeds.
- **Fixed assignment:** each sample's assignment is fixed across all layers, KV groups and answer orders.
- **Independent draws across languages:** the draw is independent per language, so one recipient question usually gets a *different* donor question in each language. Of the 1,000 aligned cross-language comparisons per repeat (100 questions × 10 pairs), 0, 0 and 1 shared the same donor question in repeats 1, 2 and 3. No recipient sample got the same donor in two repeats.

### f. Answer-order permutations
- **Orders:** each recipient is evaluated under 4 answer orders: the original plus 3 random permutations. The permutation seed is 42, keyed on the question ID, so a question gets the same 4 orders in all 5 languages.
- **Importance per sample:** importance is computed per order and then averaged over the 4.
- **Donor activations always come from donor prompts in the original answer order.** The same donor vector is patched into all 4 orders of a recipient. It is not matched to the recipient's permutation. This applies to every donor type, including the means.

### g. Statistics
- **Bootstrap CIs:** 10,000 resamples (seed 42), 95% percentile intervals. The 100 question IDs are resampled jointly across languages, language pairs and donors, so donor comparisons are paired. The cross-language control is recomputed within each resample, using the resampled questions on both axes.
- **Scope of the CIs:** CIs are conditional on the fixed donor pools and assignments. Each disjoint or random repeat is analysed separately; repeats are not pooled as extra samples.
- **Cross-language control:** the control was **not** estimated by sampling permutations. It is the exact expectation over all 100 × 100 pairings, so there is no number of permutations B for it.
- **Permutation tests and Holm correction:**
  - **Not run in the original analysis.** They were added for this handoff.
  - **Method:** B = 10,000 permutations, seed 20261122. Each permutation independently reorders the questions within each language. The test statistic is the aligned mean, equivalently the gap, because the exact null expectation is fixed. Tests are one-sided (aligned > shuffled), with p = (1 + #{permuted ≥ observed}) / (B + 1). The same permutations are used for every donor and metric.
  - **Holm correction:** applied over the 16 aggregate tests in file 1, and separately over the 160 per-pair tests in file 2.
  - **Results:**
    - **Aggregate tests:** all 16 have p = 1/10,001 ≈ 1.0 × 10⁻⁴, meaning no permutation reached the observed value. After Holm correction, all 16 are 0.0016.
    - **Per-pair tests:** 159 of 160 have p ≈ 1.0 × 10⁻⁴. The exception is random_1 / Jaccard@10 / en–ar, with p ≈ 2.0 × 10⁻⁴. After Holm correction, all 160 are 0.016.
  - **Caution:** with B = 10,000, these p-values are floors set by B, not exact tail probabilities.

### Other setup details the writer may need
- **Intervention:** one KV group (4 query heads, 128 dimensions of the attention output) at one layer, at the **final token position**, before `o_proj`. That group's slice is replaced with the donor activation. There are 36 layers × 8 groups = 288 interventions per variant, applied to 2,000 recipient prompts (500 samples × 4 answer orders).
- **Importance metric:** `pct_var_mean_incorrect` = 100 × (base − patched) / (|base| + ε), where the margin is the correct-option logit minus the mean of the incorrect-option logits. The margins are computed at the answer-letter position for options A–J. Positive values mean the intervention hurt the correct answer. This is **activation patching with a donor**, not zero ablation.
- **Prompt and precision:** a raw prompt (`Question: …\nOptions:\n…\nAnswer:`), FP32 weights.

## Optional items: what the numbers show

These are observations to support an explanation. They are not causal proofs.

- **Item 4, magnitude.** Random donors make larger and more variable intervention effects. Mean |importance| over all 288 groups and 500 samples:

  | Donors | Mean \|importance\| | Mean within-signature SD |
  |---|---|---|
  | Mean-type donors (reference, leave-one-out, disjoint) | 1.76–1.77 | 4.60–4.68 |
  | Random donors | 2.42–2.67 (≈1.4–1.5× larger) | 6.24–8.14 |

  The 99th-percentile |importance| is about 23 for mean-type donors and 29–34 for random donors.

- **Item 5, test-retest across random seeds.** For the same (question, language), signatures from two random-donor repeats agree only moderately. Spearman is 0.47–0.48 (95% CI about 0.46–0.49) and Jaccard@10 is 0.39–0.40. The same comparison between disjoint-mean repeats gives Spearman 0.98–0.99 and Jaccard@10 0.87–0.89. So roughly half of a random-donor signature's rank structure depends on which donor question was drawn.

- **Item 6, agreement with the reference signature.** Mean Spearman correlation with the reference-mean signature for the same (question, language):

  | Donor | Spearman | Jaccard@10 |
  |---|---|---|
  | Leave-one-out | 0.9995 | 0.996 |
  | Disjoint 1–3 | 0.990–0.993 | 0.90–0.92 |
  | Random 1–3 | 0.665–0.667 | 0.49–0.50 |

  The Spearman values were saved by the run; the Jaccard@10 values were added for this handoff.

- **How these connect to the halved gap (for the writer to judge).** Under random donors, aligned cross-language similarity falls from about 0.39 to about 0.21 (Spearman). The cross-language and within-language controls fall less in absolute terms, from about 0.11 to about 0.07. This fits the following picture:
  - Each language's signature picks up a large donor-specific component (items 4–6).
  - That component is not shared across languages, because the donor question differs per language (setup item e).
  - That non-shared component would dilute the aligned similarity.

  The data are consistent with this account, but no experiment tested it directly. For example, a random-donor variant that uses the *same* donor question in all five languages was not run.

## Caveats
- **Scope:** one model, one dataset and one recipient subset (questions correct in all five languages). Correctness strata, other models and datasets, and zero ablation are not covered.
- **Added analyses:** the permutation tests, the within-language and off-diagonal controls, item 4, item 5 and the Jaccard part of item 6 were computed after the experiment finished (on 2026-11-22) to answer this request. They were not pre-specified.
- **Disjoint differences:** in some disjoint repeats the paired differences from the reference are small (−0.003 Spearman), and their CIs exclude 0. The evidence shows close agreement, not formal equivalence.
- **Earlier write-up:** the run's earlier summary is in `../qwen_mmluprox_donor_summary.md`, and its numbers match these tables.
