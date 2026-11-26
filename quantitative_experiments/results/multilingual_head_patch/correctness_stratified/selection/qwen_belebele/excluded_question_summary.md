# Excluded-Question Characterization: Qwen3-8B / Belebele

The all-correct filter retains 478/900 questions (53.1%) and excludes 422.

## Correct-language count

| K correct | Available | Selected for sweep |
|---:|---:|---:|
| 0 | 32 | 32 |
| 1 | 30 | 30 |
| 2 | 42 | 42 |
| 3 | 62 | 62 |
| 4 | 256 | 250 |
| 5 | 478 | 250 |

## Language-level accuracy

| Language | All questions | Excluded questions only |
|---|---:|---:|
| acm_Arab | 57.8% | 10.0% |
| eng_Latn | 91.2% | 81.3% |
| deu_Latn | 89.9% | 78.4% |
| rus_Cyrl | 86.6% | 71.3% |
| spa_Latn | 87.2% | 72.7% |

## Retained-versus-excluded numeric characteristics

| Characteristic | Retained mean | Excluded mean | Standardized difference |
|---|---:|---:|---:|
| question_chars | 74.215 | 80.846 | -0.278 |
| question_words | 12.262 | 13.526 | -0.323 |
| context_chars | 469.347 | 489.268 | -0.125 |
| context_words | 77.197 | 81.201 | -0.154 |
| option_count | 4.000 | 4.000 | 0.000 |
| mean_option_chars | 22.708 | 28.347 | -0.385 |
| mean_option_words | 3.749 | 4.695 | -0.374 |
| other_model_k | 4.404 | 3.121 | 0.989 |

`other_model_k` is the other evaluated model's number of correct language versions and is used only as a behavioral difficulty proxy.
