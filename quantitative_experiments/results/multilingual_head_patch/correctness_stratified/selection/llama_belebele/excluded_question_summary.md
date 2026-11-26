# Excluded-Question Characterization: Llama-3.1-8B / Belebele

The all-correct filter retains 380/900 questions (42.2%) and excludes 520.

## Correct-language count

| K correct | Available | Selected for sweep |
|---:|---:|---:|
| 0 | 43 | 43 |
| 1 | 49 | 49 |
| 2 | 68 | 68 |
| 3 | 103 | 103 |
| 4 | 257 | 250 |
| 5 | 380 | 250 |

## Language-level accuracy

| Language | All questions | Excluded questions only |
|---|---:|---:|
| acm_Arab | 54.1% | 20.6% |
| eng_Latn | 87.0% | 77.5% |
| deu_Latn | 79.6% | 64.6% |
| rus_Cyrl | 81.2% | 67.5% |
| spa_Latn | 78.3% | 62.5% |

## Retained-versus-excluded numeric characteristics

| Characteristic | Retained mean | Excluded mean | Standardized difference |
|---|---:|---:|---:|
| question_chars | 74.595 | 79.319 | -0.198 |
| question_words | 12.318 | 13.246 | -0.236 |
| context_chars | 470.411 | 484.737 | -0.090 |
| context_words | 77.553 | 80.187 | -0.101 |
| option_count | 4.000 | 4.000 | 0.000 |
| mean_option_chars | 22.202 | 27.654 | -0.375 |
| mean_option_words | 3.678 | 4.569 | -0.355 |
| other_model_k | 4.729 | 3.687 | 0.931 |

`other_model_k` is the other evaluated model's number of correct language versions and is used only as a behavioral difficulty proxy.
