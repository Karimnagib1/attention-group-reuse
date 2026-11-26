# Qwen3-8B / MMLU-ProX donor robustness

The aligned-versus-shuffled advantage remains positive under every tested donor variant. Leave-one-out and disjoint language means produce closely similar effects and highly stable signatures. Random single-question donors preserve the qualitative result but reduce the Spearman gap by 49–52%; the effect magnitude is therefore sensitive to donor choice.

Completed run: `qwen_mmluprox_parallel_float32`, Slurm job 259028 on gpu-55. All 2,304 patch units completed. The patching phase took about 12 hours 53 minutes.

## Design

Qwen3-8B, MMLU-ProX, 100 parallel questions across English, Russian, Spanish, German, and Arabic (500 question-language samples). Each sample uses the original plus three answer orderings, giving 2,000 prompts. FP32; 36 layers × 8 KV groups. The intervention changes only one group’s final-position output before the attention output projection. Importance is the percentage change in the correct-minus-mean-incorrect option-logit margin, computed per ordering and then averaged.

The language-mean reference uses 1,839 eligible questions per language, including the 100 recipients. Leave-one-out removes the current recipient from this mean. Each disjoint mean uses 100 questions excluding all recipients; three independently selected pools are evaluated. Each random-donor repetition uses another question from the same language for each recipient, with assignments fixed across layers/groups/orderings. There are three random-donor repetitions. Recipient eligibility is based on original-order all-language correctness, not correctness under all 20 question-language/order combinations.

## Aggregate results

Results average the ten language pairs equally. “Signature stability” is the mean within-question/language Spearman correlation with its reference language-mean signature; it is distinct from cross-language aligned similarity. Brackets are 95% percentile confidence intervals from 10,000 bootstraps over the 100 parallel question IDs, jointly across languages and donor variants.

| Donor | Aligned Spearman | Shuffled Spearman | Spearman gap [95% CI] | Top-10 Jaccard gap [95% CI] | Signature stability [95% CI] |
|---|---:|---:|---:|---:|---:|
| Language mean (reference) | 0.3938 | 0.1071 | 0.2867 [0.2707, 0.2969] | 0.1367 [0.1215, 0.1501] | 1.0000 [1.0000, 1.0000] |
| Leave-one-out | 0.3939 | 0.1070 | 0.2869 [0.2709, 0.2971] | 0.1367 [0.1215, 0.1501] | 0.9995 [0.9994, 0.9995] |
| Disjoint mean 1 | 0.3921 | 0.1043 | 0.2878 [0.2717, 0.2982] | 0.1370 [0.1224, 0.1499] | 0.9931 [0.9927, 0.9934] |
| Disjoint mean 2 | 0.3958 | 0.1118 | 0.2840 [0.2683, 0.2939] | 0.1322 [0.1166, 0.1461] | 0.9895 [0.9891, 0.9900] |
| Disjoint mean 3 | 0.3948 | 0.1110 | 0.2838 [0.2678, 0.2940] | 0.1337 [0.1187, 0.1472] | 0.9929 [0.9925, 0.9932] |
| Random donor 1 | 0.2143 | 0.0718 | 0.1426 [0.1317, 0.1507] | 0.0623 [0.0520, 0.0718] | 0.6649 [0.6550, 0.6746] |
| Random donor 2 | 0.2163 | 0.0693 | 0.1470 [0.1365, 0.1545] | 0.0639 [0.0548, 0.0727] | 0.6652 [0.6563, 0.6742] |
| Random donor 3 | 0.2137 | 0.0748 | 0.1388 [0.1292, 0.1458] | 0.0583 [0.0486, 0.0672] | 0.6665 [0.6570, 0.6757] |

## Interpretation and scope

- Leave-one-out changes the Spearman gap by only +0.000168 and gives signature stability 0.99946. Including the recipient in the large reference mean has negligible practical impact in this condition.
- Disjoint means give Spearman gaps 0.28383–0.28783, compared with 0.28673 for the reference, and signature stability 0.98954–0.99305. Two repetitions have small negative paired differences whose intervals exclude zero; these results show close agreement, not statistical equivalence.
- Random donors give Spearman gaps 0.13884–0.14699 and stability 0.66487–0.66653. All paired gap-difference intervals versus the reference are negative. The corresponding top-10 Jaccard gaps decline from 0.13672 to 0.05828–0.06391.
- All eight aggregate gap intervals exclude zero for both metrics. All 80 language-pair/variant intervals are also positive for each metric; these are pointwise intervals without a multiple-comparison correction.
- Confidence intervals are conditional on the fixed donor pools/assignments. The three repetitions are not treated as additional independent questions. Results currently cover one model/dataset condition and the main all-correct recipient subset; they do not test correctness strata, other models/datasets, or zero ablation.

## Suggested paper wording

> On 100 parallel MMLU-ProX questions in five languages, Qwen3-8B retained a positive aligned-minus-shuffled usage-signature similarity gap under every tested donor baseline. The Spearman gap was 0.2867 [95% CI: 0.2707, 0.2969] with the language mean and 0.2869 [0.2709, 0.2971] with leave-one-out means. Across three disjoint 100-question donor pools, gaps ranged from 0.2838 to 0.2878, with within-question signature correlations of 0.9895–0.9931 relative to the reference. Random in-language single-question donors reduced the gaps to 0.1388–0.1470, while all corresponding confidence intervals remained above zero. Top-10 Jaccard overlap showed the same pattern. These results support robustness of the qualitative aligned-versus-shuffled finding, while demonstrating sensitivity of its magnitude to the intervention baseline.

## Verification and sources

Checked protocol/manifest/success-marker identity, balanced sample ordering, disjointness and random-donor language/self-exclusion constraints, and all 2,304 checkpoint-to-export matches for all three importance metrics. Independently recomputed ranks using unique-value counts, Spearman correlations as Pearson correlations of ranks, Jaccard similarities using explicit set intersections/unions, and aggregate bootstrap gaps as weighted diagonal means minus weighted all-pairs means. All aggregate point estimates, gap intervals, paired difference intervals, and stability intervals matched the saved analysis. No constant signatures or missing comparisons occurred. No model inference was rerun and no experiment code or run artifacts were changed.

- [Saved analysis summary](../runs/qwen_mmluprox_parallel_float32/analysis/summary.json)
- [All language-pair similarities](../runs/qwen_mmluprox_parallel_float32/analysis/similarity.csv)
- [Signature stability](../runs/qwen_mmluprox_parallel_float32/analysis/signature_stability.csv)
- [Full-precision aggregate table](qwen_mmluprox_donor_summary.csv)
