#!/usr/bin/env python3
"""Prepare correctness strata and characterize the paper's excluded questions.

This script is CPU-only. It reads evaluation predictions and dataset metadata,
then writes standalone ID lists/manifests for a pooled attention-group sweep.
It does not modify or invoke an existing experiment runner.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from common import (
    load_yaml,
    proportional_pattern_sample,
    read_jsonl,
    resolve_path,
    standardized_mean_difference,
    write_csv,
    write_json,
    write_jsonl,
)


NUMERIC_FEATURES = (
    "question_chars",
    "question_words",
    "context_chars",
    "context_words",
    "option_count",
    "mean_option_chars",
    "mean_option_words",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build correctness strata, runner ID lists, and excluded-question summaries."
    )
    parser.add_argument("--config", required=True, help="Correctness-stratified analysis YAML.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--max-questions-per-k",
        type=int,
        help="Override the configurable per-K cap with a positive integer.",
    )
    group.add_argument(
        "--no-cap",
        action="store_true",
        help="Select every balanced question in every K stratum.",
    )
    return parser.parse_args()


def _words(text: str) -> int:
    return len(text.split())


def _dataset_features(dataset_config_path: Path, quantitative_root: Path, reference_language: str) -> dict[str, dict[str, Any]]:
    src_path = quantitative_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from datasets import Dataset
    from quantitative_experiments.mmluprox_eval.data import extract_options, load_eval_dataset

    dataset_cfg = load_yaml(dataset_config_path)
    data_cfg = dataset_cfg["data"]
    language_field = str(data_cfg.get("language_field", "language"))
    id_field = str(data_cfg.get("parallel_id_field", data_cfg.get("id_field", "id")))
    question_field = str(data_cfg.get("question_field", "question"))
    context_field = data_cfg.get("context_field")
    option_fields = list(data_cfg.get("option_fields", []))

    # Prefer an already prepared reference-language Arrow shard. This keeps the
    # characterization stage network-free on compute nodes. Fall back to the
    # repository loader when no prepared cache is available.
    dataset = None
    cache_dir = data_cfg.get("cache_dir")
    if cache_dir:
        cache_root = Path(str(cache_dir)).expanduser()
        requested_key = re.sub(r"[^a-z0-9]", "", str(data_cfg["dataset_name"]).lower())
        search_roots = [cache_root, cache_root / "datasets"]
        dataset_roots = [
            path
            for search_root in search_roots
            if search_root.is_dir()
            for path in search_root.iterdir()
            if path.is_dir() and re.sub(r"[^a-z0-9]", "", path.name.lower()) == requested_key
        ]
        split = str(data_cfg.get("split", "test"))
        candidates = sorted(
            path
            for dataset_root in dataset_roots
            for path in (dataset_root / reference_language).glob(f"**/*-{split}.arrow")
            if not path.name.startswith("cache-")
        )
        if candidates:
            dataset = Dataset.from_file(str(candidates[0]))
    if dataset is None:
        dataset = load_eval_dataset(dataset_cfg)

    features: dict[str, dict[str, Any]] = {}
    for index in range(len(dataset)):
        record = dict(dataset[index])
        record_language = record.get(language_field, reference_language)
        if str(record_language) != reference_language:
            continue
        if record.get(id_field) is None and data_cfg.get("id_fields"):
            separator = str(data_cfg.get("id_separator", "::"))
            parallel_id = separator.join(str(record.get(field, "")) for field in data_cfg["id_fields"])
        else:
            parallel_id = str(record.get(id_field))
        if parallel_id in features:
            raise ValueError(f"Duplicate reference-language dataset row for ID {parallel_id}")
        question = str(record.get(question_field, "") or "").strip()
        context = str(record.get(context_field, "") or "").strip() if context_field else ""
        options = [str(value).strip() for value in extract_options(record, option_fields) if str(value).strip()]
        option_chars = [len(value) for value in options]
        option_words = [_words(value) for value in options]
        features[parallel_id] = {
            "question_chars": len(question),
            "question_words": _words(question),
            "context_chars": len(context),
            "context_words": _words(context),
            "option_count": len(options),
            "mean_option_chars": float(np.mean(option_chars)) if option_chars else 0.0,
            "mean_option_words": float(np.mean(option_words)) if option_words else 0.0,
            "category": str(record.get("category", "") or ""),
            "source": str(record.get("src", "") or ""),
        }
    if not features:
        raise ValueError(f"No {reference_language} rows found with {dataset_config_path}")
    return features


def _prediction_manifest(condition: Mapping[str, Any], predictions_path: Path) -> list[dict[str, Any]]:
    languages = [str(item) for item in condition["languages"]]
    required = set(languages)
    predictions = read_jsonl(predictions_path)
    index: dict[tuple[str, str], dict[str, Any]] = {}
    observed_languages: set[str] = set()
    for row in predictions:
        for field in ("id", "language", "correct"):
            if field not in row:
                raise ValueError(f"Missing '{field}' in {predictions_path}")
        parallel_id = str(row["id"])
        language = str(row["language"])
        key = (parallel_id, language)
        if key in index:
            if index[key] == row:
                # DistributedSampler can pad evaluation with one byte-identical
                # record. It is one observation, not an additional question.
                continue
            raise ValueError(f"Conflicting duplicate prediction for {key} in {predictions_path}")
        index[key] = row
        observed_languages.add(language)
    if observed_languages != required:
        raise ValueError(
            f"Language mismatch in {predictions_path}: expected {sorted(required)}, "
            f"found {sorted(observed_languages)}"
        )

    all_ids = sorted({parallel_id for parallel_id, _ in index})
    manifest: list[dict[str, Any]] = []
    for parallel_id in all_ids:
        missing = [language for language in languages if (parallel_id, language) not in index]
        if missing:
            raise ValueError(f"Incomplete five-language question {parallel_id}; missing {missing}")
        per_language = {language: index[(parallel_id, language)] for language in languages}
        correct = {language: bool(per_language[language]["correct"]) for language in languages}
        pattern = "".join("1" if correct[language] else "0" for language in languages)
        gold_values = {str(row.get("gold", "")) for row in per_language.values()}
        manifest.append(
            {
                "condition_id": str(condition["id"]),
                "model": str(condition["model"]),
                "dataset": str(condition["dataset"]),
                "parallel_id": parallel_id,
                "k_correct": sum(correct.values()),
                "correctness_pattern": pattern,
                "correct_by_language": correct,
                "prediction_by_language": {
                    language: str(per_language[language].get("pred", "")) for language in languages
                },
                "gold_by_language": {
                    language: str(per_language[language].get("gold", "")) for language in languages
                },
                "gold_label": next(iter(gold_values)) if len(gold_values) == 1 else "mixed",
            }
        )
    return manifest


def _characteristic_rows(
    condition: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    selected_ids: set[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    total = len(rows)
    languages = [str(item) for item in condition["languages"]]

    k_counts = Counter(int(row["k_correct"]) for row in rows)
    for k_correct in range(len(languages) + 1):
        count = k_counts[k_correct]
        output.append(
            {
                "condition_id": condition["id"],
                "section": "k_distribution",
                "group": f"K={k_correct}",
                "value": "questions",
                "count": count,
                "fraction": count / total,
                "selected_count": sum(
                    str(row["parallel_id"]) in selected_ids and int(row["k_correct"]) == k_correct for row in rows
                ),
            }
        )

    pattern_counts = Counter(str(row["correctness_pattern"]) for row in rows)
    for pattern, count in sorted(pattern_counts.items(), key=lambda item: (-item[1], item[0])):
        output.append(
            {
                "condition_id": condition["id"],
                "section": "correctness_pattern",
                "group": pattern,
                "value": "questions",
                "count": count,
                "fraction": count / total,
                "selected_count": sum(
                    str(row["parallel_id"]) in selected_ids and str(row["correctness_pattern"]) == pattern
                    for row in rows
                ),
            }
        )

    for language in languages:
        correct_count = sum(bool(row["correct_by_language"][language]) for row in rows)
        excluded = [row for row in rows if int(row["k_correct"]) < len(languages)]
        excluded_correct = sum(bool(row["correct_by_language"][language]) for row in excluded)
        output.append(
            {
                "condition_id": condition["id"],
                "section": "language_accuracy",
                "group": language,
                "value": "correct",
                "count": correct_count,
                "fraction": correct_count / total,
                "excluded_count": excluded_correct,
                "excluded_fraction": excluded_correct / len(excluded) if excluded else float("nan"),
            }
        )

    retained = [row for row in rows if int(row["k_correct"]) == len(languages)]
    excluded = [row for row in rows if int(row["k_correct"]) < len(languages)]
    for feature in NUMERIC_FEATURES + ("other_model_k",):
        retained_values = np.asarray([float(row[feature]) for row in retained], dtype=np.float64)
        excluded_values = np.asarray([float(row[feature]) for row in excluded], dtype=np.float64)
        output.append(
            {
                "condition_id": condition["id"],
                "section": "retained_vs_excluded_numeric",
                "group": feature,
                "value": "mean",
                "retained_count": len(retained_values),
                "retained_mean": float(np.mean(retained_values)),
                "retained_sd": float(np.std(retained_values, ddof=1)) if len(retained_values) > 1 else 0.0,
                "excluded_count": len(excluded_values),
                "excluded_mean": float(np.mean(excluded_values)),
                "excluded_sd": float(np.std(excluded_values, ddof=1)) if len(excluded_values) > 1 else 0.0,
                "standardized_mean_difference": standardized_mean_difference(retained_values, excluded_values),
            }
        )

    for feature in ("category", "source", "gold_label"):
        values = sorted({str(row.get(feature, "")) for row in rows})
        for value in values:
            retained_count = sum(str(row.get(feature, "")) == value for row in retained)
            excluded_count = sum(str(row.get(feature, "")) == value for row in excluded)
            output.append(
                {
                    "condition_id": condition["id"],
                    "section": "retained_vs_excluded_categorical",
                    "group": feature,
                    "value": value or "(missing)",
                    "retained_count": retained_count,
                    "retained_fraction": retained_count / len(retained) if retained else float("nan"),
                    "excluded_count": excluded_count,
                    "excluded_fraction": excluded_count / len(excluded) if excluded else float("nan"),
                }
            )
    return output


def _write_population_markdown(
    path: Path,
    condition: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    selected_ids: set[str],
) -> None:
    languages = [str(item) for item in condition["languages"]]
    retained = [row for row in rows if int(row["k_correct"]) == len(languages)]
    excluded = [row for row in rows if int(row["k_correct"]) < len(languages)]
    lines = [
        f"# Excluded-Question Characterization: {condition['model']} / {condition['dataset']}",
        "",
        f"The all-correct filter retains {len(retained)}/{len(rows)} questions "
        f"({len(retained) / len(rows):.1%}) and excludes {len(excluded)}.",
        "",
        "## Correct-language count",
        "",
        "| K correct | Available | Selected for sweep |",
        "|---:|---:|---:|",
    ]
    for k_correct in range(len(languages) + 1):
        available = sum(int(row["k_correct"]) == k_correct for row in rows)
        selected = sum(
            int(row["k_correct"]) == k_correct and str(row["parallel_id"]) in selected_ids for row in rows
        )
        lines.append(f"| {k_correct} | {available} | {selected} |")
    lines.extend(
        [
            "",
            "## Language-level accuracy",
            "",
            "| Language | All questions | Excluded questions only |",
            "|---|---:|---:|",
        ]
    )
    for language in languages:
        overall = np.mean([bool(row["correct_by_language"][language]) for row in rows])
        excluded_accuracy = np.mean([bool(row["correct_by_language"][language]) for row in excluded])
        lines.append(f"| {language} | {overall:.1%} | {excluded_accuracy:.1%} |")
    lines.extend(
        [
            "",
            "## Retained-versus-excluded numeric characteristics",
            "",
            "| Characteristic | Retained mean | Excluded mean | Standardized difference |",
            "|---|---:|---:|---:|",
        ]
    )
    for feature in NUMERIC_FEATURES + ("other_model_k",):
        retained_values = np.asarray([float(row[feature]) for row in retained])
        excluded_values = np.asarray([float(row[feature]) for row in excluded])
        lines.append(
            f"| {feature} | {np.mean(retained_values):.3f} | {np.mean(excluded_values):.3f} | "
            f"{standardized_mean_difference(retained_values, excluded_values):.3f} |"
        )
    lines.extend(
        [
            "",
            "`other_model_k` is the other evaluated model's number of correct language versions and is used only as a behavioral difficulty proxy.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    cfg = load_yaml(config_path)
    quantitative_root = resolve_path(config_dir, cfg["paths"]["quantitative_root"])
    selection_root = resolve_path(config_dir, cfg["paths"]["selection_root"])
    selection_root.mkdir(parents=True, exist_ok=True)

    configured_cap = cfg["selection"].get("max_questions_per_k")
    cap: int | None
    if args.no_cap:
        cap = None
    elif args.max_questions_per_k is not None:
        cap = args.max_questions_per_k
    else:
        cap = None if configured_cap is None else int(configured_cap)
    if cap is not None and cap <= 0:
        raise ValueError("The per-K cap must be positive; use null or --no-cap for all questions.")
    selection_seed = int(cfg["selection"].get("seed", 42))

    conditions = [dict(item) for item in cfg["conditions"]]
    by_id = {str(condition["id"]): condition for condition in conditions}
    if len(by_id) != len(conditions):
        raise ValueError("Condition IDs must be unique.")

    manifests: dict[str, list[dict[str, Any]]] = {}
    dataset_feature_cache: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for condition in conditions:
        condition_id = str(condition["id"])
        predictions_path = resolve_path(config_dir, condition["predictions_path"])
        dataset_config_path = resolve_path(config_dir, condition["dataset_config"])
        rows = _prediction_manifest(condition, predictions_path)
        feature_key = (str(dataset_config_path), str(condition["reference_language"]))
        if feature_key not in dataset_feature_cache:
            dataset_feature_cache[feature_key] = _dataset_features(
                dataset_config_path, quantitative_root, str(condition["reference_language"])
            )
        features = dataset_feature_cache[feature_key]
        for row in rows:
            parallel_id = str(row["parallel_id"])
            if parallel_id not in features:
                raise ValueError(f"Prediction ID {parallel_id} is absent from {dataset_config_path}")
            row.update(features[parallel_id])
        manifests[condition_id] = rows

    # Cross-model correctness is a leave-model-out behavioral difficulty proxy.
    for condition in conditions:
        condition_id = str(condition["id"])
        counterpart_id = str(condition["counterpart_condition"])
        if counterpart_id not in manifests:
            raise ValueError(f"Unknown counterpart condition: {counterpart_id}")
        counterpart = {str(row["parallel_id"]): row for row in manifests[counterpart_id]}
        for row in manifests[condition_id]:
            other = counterpart.get(str(row["parallel_id"]))
            if other is None:
                raise ValueError(f"Counterpart {counterpart_id} lacks ID {row['parallel_id']}")
            row["other_model_k"] = int(other["k_correct"])
            row["other_model_correctness_pattern"] = str(other["correctness_pattern"])

    all_characteristics: list[dict[str, Any]] = []
    run_audit: list[dict[str, Any]] = []
    for condition in conditions:
        condition_id = str(condition["id"])
        rows = manifests[condition_id]
        condition_dir = selection_root / condition_id
        selected_ids: list[str] = []
        audit_rows: list[dict[str, Any]] = []
        for k_correct in range(len(condition["languages"]) + 1):
            stratum = [row for row in rows if int(row["k_correct"]) == k_correct]
            selected, quotas = proportional_pattern_sample(
                stratum,
                cap=cap,
                seed=selection_seed,
                condition_id=condition_id,
                k_correct=k_correct,
            )
            selected_ids.extend(selected)
            full_pattern_counts = Counter(str(row["correctness_pattern"]) for row in stratum)
            for pattern in sorted(full_pattern_counts):
                audit_rows.append(
                    {
                        "condition_id": condition_id,
                        "k_correct": k_correct,
                        "correctness_pattern": pattern,
                        "available_questions": full_pattern_counts[pattern],
                        "selected_questions": quotas.get(pattern, 0),
                        "configured_cap_per_k": cap,
                        "selection_seed": selection_seed,
                    }
                )
        selected_ids = sorted(set(selected_ids))
        selected_set = set(selected_ids)
        for row in rows:
            row["selected_for_sweep"] = str(row["parallel_id"]) in selected_set
        write_jsonl(condition_dir / "question_manifest.jsonl", rows)
        write_json(condition_dir / "selected_ids.json", selected_ids)
        write_csv(condition_dir / "selection_audit.csv", audit_rows)
        characteristics = _characteristic_rows(condition, rows, selected_set)
        write_csv(condition_dir / "population_characteristics.csv", characteristics)
        _write_population_markdown(
            condition_dir / "excluded_question_summary.md", condition, rows, selected_set
        )
        all_characteristics.extend(characteristics)
        selected_by_k = Counter(int(row["k_correct"]) for row in rows if row["selected_for_sweep"])
        run_audit.append(
            {
                "condition_id": condition_id,
                "model": condition["model"],
                "dataset": condition["dataset"],
                "predictions_path": str(resolve_path(config_dir, condition["predictions_path"])),
                "dataset_config": str(resolve_path(config_dir, condition["dataset_config"])),
                "sweep_output_dir": str(resolve_path(config_dir, condition["sweep_output_dir"])),
                "available_questions": len(rows),
                "selected_questions": len(selected_ids),
                "available_by_k": {str(k): sum(int(row["k_correct"]) == k for row in rows) for k in range(6)},
                "selected_by_k": {str(k): selected_by_k[k] for k in range(6)},
            }
        )
        print(f"{condition_id}: selected {len(selected_ids)}/{len(rows)} questions; cap={cap}")

    write_csv(selection_root / "population_characteristics_all_conditions.csv", all_characteristics)
    write_json(
        selection_root / "selection_manifest.json",
        {
            "config_path": str(config_path),
            "max_questions_per_k": cap,
            "selection_seed": selection_seed,
            "conditions": run_audit,
        },
    )


if __name__ == "__main__":
    main()
