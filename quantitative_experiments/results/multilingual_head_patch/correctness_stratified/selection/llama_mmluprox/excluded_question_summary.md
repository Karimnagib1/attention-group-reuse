# Excluded-Question Characterization: Llama-3.1-8B / MMLU-ProX

The all-correct filter retains 1420/11759 questions (12.1%) and excludes 10339.

## Correct-language count

| K correct | Available | Selected for sweep |
|---:|---:|---:|
| 0 | 5923 | 250 |
| 1 | 1583 | 250 |
| 2 | 1056 | 250 |
| 3 | 871 | 250 |
| 4 | 906 | 250 |
| 5 | 1420 | 250 |

## Language-level accuracy

| Language | All questions | Excluded questions only |
|---|---:|---:|
| en | 33.0% | 23.8% |
| ru | 28.2% | 18.3% |
| es | 29.4% | 19.8% |
| de | 29.0% | 19.2% |
| ar | 25.2% | 15.0% |

## Retained-versus-excluded numeric characteristics

| Characteristic | Retained mean | Excluded mean | Standardized difference |
|---|---:|---:|---:|
| question_chars | 192.031 | 266.500 | -0.271 |
| question_words | 32.712 | 45.974 | -0.284 |
| context_chars | 0.000 | 0.000 | 0.000 |
| context_words | 0.000 | 0.000 | 0.000 |
| option_count | 9.337 | 9.525 | -0.123 |
| mean_option_chars | 56.349 | 34.841 | 0.520 |
| mean_option_words | 8.955 | 5.821 | 0.482 |
| other_model_k | 3.696 | 1.495 | 1.245 |

`other_model_k` is the other evaluated model's number of correct language versions and is used only as a behavioral difficulty proxy.
