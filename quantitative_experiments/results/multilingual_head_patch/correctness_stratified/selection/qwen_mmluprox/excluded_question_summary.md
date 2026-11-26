# Excluded-Question Characterization: Qwen3-8B / MMLU-ProX

The all-correct filter retains 1839/11759 questions (15.6%) and excludes 9920.

## Correct-language count

| K correct | Available | Selected for sweep |
|---:|---:|---:|
| 0 | 5016 | 250 |
| 1 | 1606 | 250 |
| 2 | 1148 | 250 |
| 3 | 996 | 250 |
| 4 | 1154 | 250 |
| 5 | 1839 | 250 |

## Language-level accuracy

| Language | All questions | Excluded questions only |
|---|---:|---:|
| en | 39.4% | 28.2% |
| ru | 35.7% | 23.7% |
| es | 37.7% | 26.1% |
| de | 35.8% | 23.9% |
| ar | 27.5% | 14.1% |

## Retained-versus-excluded numeric characteristics

| Characteristic | Retained mean | Excluded mean | Standardized difference |
|---|---:|---:|---:|
| question_chars | 164.739 | 274.705 | -0.411 |
| question_words | 27.825 | 47.440 | -0.432 |
| context_chars | 0.000 | 0.000 | 0.000 |
| context_words | 0.000 | 0.000 | 0.000 |
| option_count | 9.120 | 9.573 | -0.281 |
| mean_option_chars | 49.790 | 35.149 | 0.359 |
| mean_option_words | 7.878 | 5.888 | 0.311 |
| other_model_k | 3.212 | 1.121 | 1.180 |

`other_model_k` is the other evaluated model's number of correct language versions and is used only as a behavioral difficulty proxy.
