# Held-out latent-country evaluation

This workflow addresses the localization/evaluation overlap in the latent-country
case study. All new code and default experiment paths are under `review/`.
Existing project code and results are not modified.

The intervention follows the original scripts: take a donor prompt's final-token
attention output **before `o_proj`**, replace the selected KV-group slices in the
recipient forward pass, and rank the **first tokenizer token** of the donor
country's capital expressed in the **recipient language**. This evaluates
first-token steering, not full-answer generation or complete factual correctness.

## Two evaluation modes

1. **Published fixed groups:** freeze Qwen's L24/G6 + L31/G0 or Llama's
   L14/G6 + L15/G4 + L17/G5 + L24/G0 and evaluate fresh prompts. These sites
   were identified using the original experiments, including multiple languages.
   This mode tests held-out prompt performance, not English-only localization.
   The fixed Qwen set is deliberately the same in every condition, including
   Russian, where some original evaluations used only L24/G6.
2. **English-only localization:** sweep individual groups on new English→English
   localization prompts, choose the top 2 groups for Qwen or top 4 for Llama,
   freeze them in `selection.json`, then evaluate held-out prompts across languages.
   The group count is a prespecified hyperparameter, not chosen on evaluation outcomes.

Both modes use the same prepared protocol and include layer-matched random controls.

## Protocol and controls

- Default: 100 localization pairs per donor family and 100 evaluation pairs per
  condition; 20 demonstrations per prompt; landmark/university→calling-code donors
  and city→capital recipients. The donor and recipient query countries always differ.
- Evaluation covers all five monolingual conditions and eight directed
  English↔other-language conditions: 13 language pairs × 2 donor families = 26
  conditions / 2,600 records per model. Set `language_pairs` to `all` for all 25
  ordered language pairs. Localization uses English only, 200 records in total.
- All languages render the same country/entity-index prompt plans for each pair.
  This assumes corresponding positions in the existing translated entity lists
  describe the same entity; list lengths and country keys are validated.
- Localization and evaluation exclude identical prompts, including renderings in
  other languages. Preparation also replays and excludes prompts from the original
  100-pair / 20-demonstration seed schedule over all ordered language pairs.
  It does not claim exclusion against arbitrary historical configurations.
- Every manifest stores exact prompts, answers, countries, entity plans, seeds,
  source-data hashes, and a content fingerprint. Preparation uses no model outcomes.
- The split is by complete prompt, **not country or entity**. Countries and entities
  may recur with different demonstrations. No held-out-country claim is supported.
- Each random control preserves the selected groups' layer counts, sampling
  without replacement within each layer. Twenty independent site sets are frozen
  for the entire run and shared across conditions. Selected sites may recur by
  chance; excluding them would alter the null distribution.
- Localization uses the original signed normalized logit-margin change. Very small
  baseline denominators and first-token collisions are explicitly counted.
  Individual scores and the full ranking are saved for inspection. The localization
  step does not read evaluation outcomes or change the selected group count.
- Evaluation retains pairs whose original and type-corrected answers share the
  first token, matching the original rank outcome; their count is reported because
  they cannot establish a distinction between the two answers.

## Commands

Use the existing `thesis` Conda environment (Python 3.10+, PyTorch, Transformers,
NumPy, and Accelerate for automatic device placement). No NNsight dependency is
needed. The existing prompt format and rank metric are retained using native
PyTorch hooks. From the project root:

```bash
conda activate thesis

# CPU only; same protocol is usable with either model config.
bash review/run_latent_country.sh prepare \
  --config review/configs/latent_country_qwen.json \
  --output review/runs/latent_country/protocol.json

# GPU: evaluate frozen published Qwen groups on fresh prompts.
bash review/run_latent_country.sh evaluate \
  --config review/configs/latent_country_qwen.json \
  --protocol review/runs/latent_country/protocol.json \
  --output review/runs/latent_country/qwen_published

# Optional stronger test: select only on English localization prompts.
bash review/run_latent_country.sh localize \
  --config review/configs/latent_country_qwen.json \
  --protocol review/runs/latent_country/protocol.json \
  --output review/runs/latent_country/qwen_english_localization

bash review/run_latent_country.sh evaluate \
  --config review/configs/latent_country_qwen.json \
  --protocol review/runs/latent_country/protocol.json \
  --selection review/runs/latent_country/qwen_english_localization/selection.json \
  --output review/runs/latent_country/qwen_english_selected
```

For Llama, use `review/configs/latent_country_llama.json` and distinct output
directories. Its model weights must already be accessible to your account. Pin
`model.revision` before a production run if a specific checkpoint revision is
required; the resolved model commit is recorded regardless.

The wrapper uses the active environment and visible devices. It neither connects
to a host nor submits scheduler jobs. On `gpu-07`, activate `thesis` and run it
inside your allocated resources. Model weights are loaded only by `localize` and
`evaluate`. No experiment has been automatically submitted by this addition.

For a smaller pilot, copy a config **under `review/`**, reduce the pair counts and
random draws, and prepare a new protocol/output directory. Do not tune selection
on pilot evaluation outcomes and then report the same prompts as held out.

## Artifacts, resume, and interpretation

Each inference run writes:

- `run_manifest.json`: complete settings, model identity, selected/control sites,
  protocol/code fingerprints, and expected record IDs.
- `records/<condition>__<pair_index>.json`: atomic per-pair results. Rerunning the
  identical command skips completed records. At most one unfinished batch is lost.
- `selection.json` for localization: frozen sites and complete training ranking.
- `summary.json` and `summary.csv` for evaluation: per-condition and equal-weight
  monolingual/cross-lingual/all-condition summaries.
- `_SUCCESS`, written after the stage has completed and final artifacts exist.

Concurrent writers to the same run are rejected. Changing config, code, model
identity, prompt manifest, or selection requires a new run directory. Runs use
full forward passes, with no experimental KV-cache optimization. Prompts exceeding
the configured token limit fail explicitly instead of silently truncating queries.

Summaries include baseline and selected top-1, paired top-1 change, selected top-10,
mean/median rank, random-control top-1, and selected-minus-random top-1. Rates are
percentages and differences are percentage points. The rank tie convention is
one plus the count of strictly higher logits, matching the old evaluator.

The 95% percentile intervals resample `pair_index` jointly across language pairs
and donor families, preserving the shared recipient prompt and multilingual
pairing. There are 100 prompt clusters, **not 2,600 independent observations**.
The median-rank aggregate is the mean of within-condition medians. Intervals are
conditional on the selected groups and frozen random-site bank; they do not
measure uncertainty over localization samples, control-set selection, model
training, or countries. Localization outcomes are never included in test summaries.

Rebuild summaries without loading a model:

```bash
bash review/run_latent_country.sh summarize \
  --run-dir review/runs/latent_country/qwen_published
```

## Validation

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s review/tests -v
```

The tests replay the original prompt generator, check split separation and
language-correct targets, verify group replacement with tiny randomly initialized
Llama and Qwen models on CPU, check hook cleanup, and exercise localization,
evaluation, resume, provenance rejection, random controls, and paired statistics.
They do not download model weights or run the 8B experiments.
