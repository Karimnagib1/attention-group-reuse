#!/usr/bin/env python3
"""Analyze correctness-stratified usage signatures and matched subsets.

The script consumes only saved prediction manifests and attention-group sweep
artifacts. It never loads a language model or modifies an experiment runner.
"""

from __future__ import annotations

import argparse
import itertools
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from common import (
    bootstrap_mean_interval,
    holm_adjust,
    load_yaml,
    normalized_rank_rows,
    read_json,
    read_jsonl,
    resolve_path,
    slope,
    stable_seed,
    standardized_mean_difference,
    topk_binary_rows,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute similarity-by-correctness and matched partial-correct results."
    )
    parser.add_argument("--config", required=True, help="Correctness-stratified analysis YAML.")
    parser.add_argument("--bootstrap-resamples", type=int, help="Override bootstrap resamples.")
    parser.add_argument("--permutation-draws", type=int, help="Override ordinal-trend permutations.")
    parser.add_argument("--output-dir", help="Override the configured analysis output directory.")
    return parser.parse_args()


def _bootstrap_means(values: np.ndarray, draws: int, seed: int, block_size: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.full(draws, np.nan, dtype=np.float64)
    if values.size == 1:
        return np.full(draws, float(values[0]), dtype=np.float64)
    rng = np.random.default_rng(seed)
    output = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, block_size):
        stop = min(draws, start + block_size)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        output[start:stop] = values[indices].mean(axis=1)
    return output


class SweepData:
    def __init__(
        self,
        condition: Mapping[str, Any],
        sweep_dir: Path,
        selected_ids: set[str],
        metric: str,
        jaccard_k: int,
        margin_key: str = "base_margin_mean",
    ) -> None:
        required = [sweep_dir / name for name in ("metadata.json", "samples.json", "importance_tensors.npz")]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing correctness-stratified sweep artifacts: " + ", ".join(missing))
        metadata = read_json(sweep_dir / "metadata.json")
        samples = read_json(sweep_dir / "samples.json")
        self.languages = [str(item) for item in metadata.get("languages", [])]
        expected_languages = [str(item) for item in condition["languages"]]
        if self.languages != expected_languages:
            raise ValueError(
                f"{condition['id']} language order mismatch: expected {expected_languages}, found {self.languages}"
            )
        if metadata.get("target_mode") != "correct":
            raise ValueError(f"{condition['id']} sweep must use patching.target_mode=correct")
        if metadata.get("patch_mode") != "kv_group":
            raise ValueError(f"{condition['id']} sweep must use patching.patch_mode=kv_group")
        if not isinstance(samples, list):
            raise ValueError(f"Invalid samples.json in {sweep_dir}")

        with np.load(sweep_dir / "importance_tensors.npz") as archive:
            flat_key = f"flat_{metric}"
            if flat_key not in archive:
                raise KeyError(f"Missing '{flat_key}' in {sweep_dir / 'importance_tensors.npz'}")
            if margin_key not in archive:
                raise KeyError(
                    f"Missing '{margin_key}' in {sweep_dir / 'importance_tensors.npz'}"
                )
            flat = np.asarray(archive[flat_key], dtype=np.float64)
            margins = np.asarray(archive[margin_key], dtype=np.float64)
        if flat.ndim != 2 or margins.ndim != 1 or flat.shape[0] != margins.shape[0]:
            raise ValueError(f"Invalid tensor shapes in {sweep_dir}")
        if not np.all(np.isfinite(flat)) or not np.all(np.isfinite(margins)):
            raise ValueError(f"Non-finite importance values or margins in {sweep_dir}")

        sample_index: dict[tuple[str, str], int] = {}
        observed_ids: set[str] = set()
        for row in samples:
            key = (str(row["parallel_id"]), str(row["language"]))
            index = int(row["sample_index"])
            if key in sample_index:
                raise ValueError(f"Duplicate sweep sample {key} in {sweep_dir}")
            if index < 0 or index >= flat.shape[0]:
                raise ValueError(f"Out-of-range sample_index {index} in {sweep_dir}")
            sample_index[key] = index
            observed_ids.add(key[0])
        if observed_ids != selected_ids:
            missing_ids = sorted(selected_ids - observed_ids)[:10]
            extra_ids = sorted(observed_ids - selected_ids)[:10]
            raise ValueError(
                f"{condition['id']} sweep/selection ID mismatch; missing={missing_ids}, extra={extra_ids}"
            )
        for parallel_id in selected_ids:
            missing_languages = [
                language for language in self.languages if (parallel_id, language) not in sample_index
            ]
            if missing_languages:
                raise ValueError(f"Sweep ID {parallel_id} lacks languages {missing_languages}")

        self.ids = sorted(selected_ids)
        self.id_to_position = {parallel_id: index for index, parallel_id in enumerate(self.ids)}
        vectors = np.stack(
            [flat[[sample_index[(parallel_id, language)] for parallel_id in self.ids]] for language in self.languages],
            axis=0,
        )
        self.margins = np.stack(
            [margins[[sample_index[(parallel_id, language)] for parallel_id in self.ids]] for language in self.languages],
            axis=0,
        )
        normalized: list[np.ndarray] = []
        valid: list[np.ndarray] = []
        binary: list[np.ndarray] = []
        for language_index in range(len(self.languages)):
            norm, ok = normalized_rank_rows(vectors[language_index])
            normalized.append(norm)
            valid.append(ok)
            binary.append(topk_binary_rows(vectors[language_index], jaccard_k))
        self.normalized = np.stack(normalized, axis=0)
        self.valid = np.stack(valid, axis=0)
        self.binary = np.stack(binary, axis=0)
        self.jaccard_k = min(jaccard_k, vectors.shape[2])
        self.margin_key = margin_key

    def positions(self, parallel_ids: Sequence[str]) -> np.ndarray:
        return np.asarray([self.id_to_position[str(parallel_id)] for parallel_id in parallel_ids], dtype=np.int64)

    def matrix(self, metric: str, first: int, second: int, positions: np.ndarray) -> np.ndarray:
        if metric == "spearman":
            result = self.normalized[first, positions] @ self.normalized[second, positions].T
            valid = self.valid[first, positions][:, None] & self.valid[second, positions][None, :]
            result[~valid] = np.nan
            return result
        if metric == "jaccard_at_10":
            intersection = self.binary[first, positions] @ self.binary[second, positions].T
            return intersection / (2.0 * self.jaccard_k - intersection)
        raise ValueError(f"Unknown similarity metric: {metric}")

    def direct(self, metric: str, parallel_id: str, first: int, second: int) -> float:
        position = self.id_to_position[parallel_id]
        if metric == "spearman":
            if not (self.valid[first, position] and self.valid[second, position]):
                return float("nan")
            return float(np.dot(self.normalized[first, position], self.normalized[second, position]))
        intersection = float(np.dot(self.binary[first, position], self.binary[second, position]))
        return intersection / (2.0 * self.jaccard_k - intersection)

    def margin(self, parallel_id: str, language_index: int) -> float:
        return float(self.margins[language_index, self.id_to_position[parallel_id]])


def _matrix_contributions(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    diagonal = np.diag(matrix).astype(np.float64)
    row_mean = np.nanmean(matrix, axis=1)
    column_mean = np.nanmean(matrix, axis=0)
    null_contribution = 0.5 * (row_mean + column_mean)
    return diagonal, null_contribution, diagonal - null_contribution


def _summary_row(
    base: Mapping[str, Any],
    aligned: np.ndarray,
    null: np.ndarray,
    bootstrap_resamples: int,
    ci_level: float,
    bootstrap_seed: int,
    block_size: int,
) -> dict[str, Any]:
    aligned = np.asarray(aligned, dtype=np.float64)
    null = np.asarray(null, dtype=np.float64)
    valid = np.isfinite(aligned) & np.isfinite(null)
    aligned = aligned[valid]
    null = null[valid]
    gap = aligned - null
    aligned_low, aligned_high = bootstrap_mean_interval(
        aligned, bootstrap_resamples, ci_level, stable_seed(bootstrap_seed, *base.values(), "aligned"), block_size
    )
    null_low, null_high = bootstrap_mean_interval(
        null, bootstrap_resamples, ci_level, stable_seed(bootstrap_seed, *base.values(), "null"), block_size
    )
    gap_low, gap_high = bootstrap_mean_interval(
        gap, bootstrap_resamples, ci_level, stable_seed(bootstrap_seed, *base.values(), "gap"), block_size
    )
    return {
        **dict(base),
        "n_questions": int(valid.sum()),
        "aligned_estimate": float(np.mean(aligned)) if aligned.size else float("nan"),
        "aligned_ci_low": aligned_low,
        "aligned_ci_high": aligned_high,
        "null_exact_estimate": float(np.mean(null)) if null.size else float("nan"),
        "null_ci_low": null_low,
        "null_ci_high": null_high,
        "gap_estimate": float(np.mean(gap)) if gap.size else float("nan"),
        "gap_ci_low": gap_low,
        "gap_ci_high": gap_high,
    }


def _stratum_similarity(
    condition: Mapping[str, Any],
    manifest_rows: Sequence[Mapping[str, Any]],
    sweep: SweepData,
    analysis_cfg: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[int, dict[str, np.ndarray]]]]:
    bootstrap_resamples = int(analysis_cfg["bootstrap_resamples"])
    ci_level = float(analysis_cfg["ci_level"])
    bootstrap_seed = int(analysis_cfg["bootstrap_seed"])
    block_size = int(analysis_cfg.get("bootstrap_block_size", 256))
    min_cell = int(analysis_cfg.get("min_cell_questions", 10))
    metrics = ("spearman", "jaccard_at_10")
    language_pairs = list(itertools.combinations(range(len(sweep.languages)), 2))
    by_k_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    trend_values: dict[str, dict[int, dict[str, np.ndarray]]] = {metric: {} for metric in metrics}

    manifest_by_id = {str(row["parallel_id"]): row for row in manifest_rows}
    for k_correct in range(len(sweep.languages) + 1):
        ids = sorted(
            str(row["parallel_id"])
            for row in manifest_rows
            if bool(row["selected_for_sweep"]) and int(row["k_correct"]) == k_correct
        )
        if not ids:
            continue
        positions = sweep.positions(ids)
        for metric in metrics:
            aligned_by_question: list[list[float]] = [[] for _ in ids]
            null_by_question: list[list[float]] = [[] for _ in ids]
            status_aligned: dict[str, dict[str, list[float]]] = {
                status: defaultdict(list) for status in ("correct_correct", "correct_incorrect", "incorrect_incorrect")
            }
            status_null: dict[str, dict[str, list[float]]] = {
                status: defaultdict(list) for status in ("correct_correct", "correct_incorrect", "incorrect_incorrect")
            }
            low_cells = Counter()

            for first, second in language_pairs:
                matrix = sweep.matrix(metric, first, second, positions)
                aligned, null, _ = _matrix_contributions(matrix)
                for question_index in range(len(ids)):
                    if math.isfinite(aligned[question_index]) and math.isfinite(null[question_index]):
                        aligned_by_question[question_index].append(float(aligned[question_index]))
                        null_by_question[question_index].append(float(null[question_index]))
                pair_base = {
                    "condition_id": condition["id"],
                    "model": condition["model"],
                    "dataset": condition["dataset"],
                    "metric": metric,
                    "k_correct": k_correct,
                    "scope": "language_pair",
                    "language_pair": f"{sweep.languages[first]}__{sweep.languages[second]}",
                }
                by_k_rows.append(
                    _summary_row(
                        pair_base,
                        aligned,
                        null,
                        bootstrap_resamples,
                        ci_level,
                        bootstrap_seed,
                        block_size,
                    )
                )

                states: dict[str, list[int]] = defaultdict(list)
                for question_index, parallel_id in enumerate(ids):
                    correct = manifest_by_id[parallel_id]["correct_by_language"]
                    state = (bool(correct[sweep.languages[first]]), bool(correct[sweep.languages[second]]))
                    state_name = {True: "1", False: "0"}[state[0]] + {True: "1", False: "0"}[state[1]]
                    states[state_name].append(question_index)
                for state_name, question_indices in states.items():
                    if not question_indices:
                        continue
                    status = {
                        "11": "correct_correct",
                        "10": "correct_incorrect",
                        "01": "correct_incorrect",
                        "00": "incorrect_incorrect",
                    }[state_name]
                    subset = np.asarray(question_indices, dtype=np.int64)
                    state_matrix = matrix[np.ix_(subset, subset)]
                    state_aligned, state_null, _ = _matrix_contributions(state_matrix)
                    if len(subset) < min_cell:
                        low_cells[status] += 1
                    for local_index, question_index in enumerate(subset):
                        parallel_id = ids[int(question_index)]
                        if math.isfinite(state_aligned[local_index]) and math.isfinite(state_null[local_index]):
                            status_aligned[status][parallel_id].append(float(state_aligned[local_index]))
                            status_null[status][parallel_id].append(float(state_null[local_index]))

            aggregate_aligned = np.asarray(
                [np.mean(values) if values else np.nan for values in aligned_by_question], dtype=np.float64
            )
            aggregate_null = np.asarray(
                [np.mean(values) if values else np.nan for values in null_by_question], dtype=np.float64
            )
            aggregate_base = {
                "condition_id": condition["id"],
                "model": condition["model"],
                "dataset": condition["dataset"],
                "metric": metric,
                "k_correct": k_correct,
                "scope": "aggregate",
                "language_pair": "all_10_pairs",
            }
            by_k_rows.append(
                _summary_row(
                    aggregate_base,
                    aggregate_aligned,
                    aggregate_null,
                    bootstrap_resamples,
                    ci_level,
                    bootstrap_seed,
                    block_size,
                )
            )
            valid = np.isfinite(aggregate_aligned) & np.isfinite(aggregate_null)
            trend_values[metric][k_correct] = {
                "aligned": aggregate_aligned[valid],
                "gap": (aggregate_aligned - aggregate_null)[valid],
            }

            for status in ("correct_correct", "correct_incorrect", "incorrect_incorrect"):
                status_ids = sorted(set(status_aligned[status]) & set(status_null[status]))
                aligned_values = np.asarray(
                    [np.mean(status_aligned[status][parallel_id]) for parallel_id in status_ids], dtype=np.float64
                )
                null_values = np.asarray(
                    [np.mean(status_null[status][parallel_id]) for parallel_id in status_ids], dtype=np.float64
                )
                status_base = {
                    "condition_id": condition["id"],
                    "model": condition["model"],
                    "dataset": condition["dataset"],
                    "metric": metric,
                    "k_correct": k_correct,
                    "pair_status": status,
                }
                row = _summary_row(
                    status_base,
                    aligned_values,
                    null_values,
                    bootstrap_resamples,
                    ci_level,
                    bootstrap_seed,
                    block_size,
                )
                row["low_n_language_pair_cells"] = low_cells[status]
                row["estimable"] = bool(len(status_ids))
                status_rows.append(row)
    return by_k_rows, status_rows, trend_values


def _trend_rows(
    condition: Mapping[str, Any],
    trend_values: Mapping[str, Mapping[int, Mapping[str, np.ndarray]]],
    analysis_cfg: Mapping[str, Any],
) -> list[dict[str, Any]]:
    bootstrap_draws = int(analysis_cfg["bootstrap_resamples"])
    permutation_draws = int(analysis_cfg["permutation_draws"])
    bootstrap_seed = int(analysis_cfg["bootstrap_seed"])
    permutation_seed = int(analysis_cfg["permutation_seed"])
    ci_level = float(analysis_cfg["ci_level"])
    block_size = int(analysis_cfg.get("bootstrap_block_size", 256))
    output: list[dict[str, Any]] = []

    for metric, per_k in trend_values.items():
        for outcome in ("aligned", "gap"):
            ks = sorted(k for k, values in per_k.items() if np.asarray(values[outcome]).size)
            arrays = [np.asarray(per_k[k][outcome], dtype=np.float64) for k in ks]
            x = np.concatenate([np.full(values.size, k, dtype=np.float64) for k, values in zip(ks, arrays)])
            y = np.concatenate(arrays)
            estimate = slope(x, y)

            weights = np.asarray([values.size for values in arrays], dtype=np.float64)
            k_values = np.asarray(ks, dtype=np.float64)
            x_mean = float(np.average(k_values, weights=weights))
            denominator = float(np.sum(weights * np.square(k_values - x_mean)))
            means = np.stack(
                [
                    _bootstrap_means(
                        values,
                        bootstrap_draws,
                        stable_seed(bootstrap_seed, condition["id"], metric, outcome, k),
                        block_size,
                    )
                    for k, values in zip(ks, arrays)
                ],
                axis=1,
            )
            bootstrap_slopes = np.sum(
                weights[None, :] * (k_values - x_mean)[None, :] * means, axis=1
            ) / denominator
            alpha = 1.0 - ci_level
            ci_low, ci_high = np.quantile(bootstrap_slopes, [alpha / 2.0, 1.0 - alpha / 2.0])

            permutation_p = float("nan")
            exceedances: int | None = None
            if outcome == "gap" and permutation_draws > 0:
                rng = np.random.default_rng(stable_seed(permutation_seed, condition["id"], metric, outcome))
                exceedances = 0
                for _ in range(permutation_draws):
                    permuted = rng.permutation(x)
                    candidate = slope(permuted, y)
                    if abs(candidate) >= abs(estimate):
                        exceedances += 1
                permutation_p = (exceedances + 1.0) / (permutation_draws + 1.0)
            output.append(
                {
                    "condition_id": condition["id"],
                    "model": condition["model"],
                    "dataset": condition["dataset"],
                    "metric": metric,
                    "outcome": outcome,
                    "n_questions": len(y),
                    "k_min": min(ks),
                    "k_max": max(ks),
                    "slope_per_additional_correct_language": estimate,
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "permutation_draws": permutation_draws if outcome == "gap" else None,
                    "permutation_exceedances": exceedances,
                    "permutation_p_raw": permutation_p,
                    "permutation_p_holm": None,
                }
            )
    return output


def _standardized_features(treated: np.ndarray, controls: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pooled = np.concatenate([treated, controls], axis=0)
    center = np.mean(pooled, axis=0)
    scale = np.std(pooled, axis=0)
    active = scale > 1.0e-8
    if not np.any(active):
        return np.zeros((len(treated), 1)), np.zeros((len(controls), 1)), active
    return (treated[:, active] - center[active]) / scale[active], (
        controls[:, active] - center[active]
    ) / scale[active], active


def _match_condition(
    condition: Mapping[str, Any],
    manifest_rows: Sequence[Mapping[str, Any]],
    sweep: SweepData,
    analysis_cfg: Mapping[str, Any],
    match_types: Sequence[str] = ("confidence", "difficulty"),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    row_by_id = {str(row["parallel_id"]): row for row in manifest_rows if bool(row["selected_for_sweep"])}
    all_correct_ids = sorted(pid for pid, row in row_by_id.items() if int(row["k_correct"]) == 5)
    language_pairs = list(itertools.combinations(range(len(sweep.languages)), 2))
    assignments: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    structural_features = ("question_words", "context_words", "option_count", "mean_option_words")

    invalid_match_types = sorted(set(match_types) - {"confidence", "difficulty"})
    if invalid_match_types:
        raise ValueError(f"Unknown matching types: {', '.join(invalid_match_types)}")

    for match_type in match_types:
        caliper = float(
            analysis_cfg["confidence_caliper_sd"]
            if match_type == "confidence"
            else analysis_cfg["difficulty_caliper_sd"]
        )
        for k_correct in (2, 3, 4):
            current_assignments: list[dict[str, Any]] = []
            for first, second in language_pairs:
                first_language = sweep.languages[first]
                second_language = sweep.languages[second]
                treated_ids = sorted(
                    pid
                    for pid, row in row_by_id.items()
                    if int(row["k_correct"]) == k_correct
                    and bool(row["correct_by_language"][first_language])
                    and bool(row["correct_by_language"][second_language])
                )
                if not treated_ids or not all_correct_ids:
                    continue
                if match_type == "confidence":
                    feature_names = (f"margin_{first_language}", f"margin_{second_language}")

                    def features(parallel_id: str) -> list[float]:
                        return [sweep.margin(parallel_id, first), sweep.margin(parallel_id, second)]

                    exact_key = lambda parallel_id: ()
                else:
                    feature_names = structural_features

                    def features(parallel_id: str) -> list[float]:
                        return [float(row_by_id[parallel_id][name]) for name in structural_features]

                    def exact_key(parallel_id: str) -> tuple[Any, ...]:
                        row = row_by_id[parallel_id]
                        base: tuple[Any, ...] = (int(row["other_model_k"]),)
                        if str(condition["dataset_family"]) == "mmluprox":
                            base += (str(row.get("category", "")),)
                        return base

                treated_matrix = np.asarray([features(pid) for pid in treated_ids], dtype=np.float64)
                control_matrix = np.asarray([features(pid) for pid in all_correct_ids], dtype=np.float64)
                treated_z, control_z, active = _standardized_features(treated_matrix, control_matrix)
                control_min = np.min(control_z, axis=0)
                control_max = np.max(control_z, axis=0)

                for treated_index, treated_id in enumerate(treated_ids):
                    candidate_indices = [
                        index
                        for index, control_id in enumerate(all_correct_ids)
                        if exact_key(control_id) == exact_key(treated_id)
                    ]
                    if not candidate_indices:
                        continue
                    treated_vector = treated_z[treated_index]
                    if np.any(treated_vector < control_min) or np.any(treated_vector > control_max):
                        continue
                    candidate_z = control_z[candidate_indices]
                    distances = np.linalg.norm(candidate_z - treated_vector[None, :], axis=1)
                    best_local = int(np.argmin(distances))
                    distance = float(distances[best_local])
                    if distance > caliper:
                        continue
                    control_index = candidate_indices[best_local]
                    control_id = all_correct_ids[control_index]
                    assignment = {
                        "condition_id": condition["id"],
                        "model": condition["model"],
                        "dataset": condition["dataset"],
                        "match_type": match_type,
                        "confidence_margin_key": (
                            sweep.margin_key if match_type == "confidence" else None
                        ),
                        "k_correct": k_correct,
                        "language_pair": f"{first_language}__{second_language}",
                        "first_language_index": first,
                        "second_language_index": second,
                        "partial_parallel_id": treated_id,
                        "control_parallel_id": control_id,
                        "distance": distance,
                        "feature_names": list(feature_names),
                        "partial_features": treated_matrix[treated_index].tolist(),
                        "control_features": control_matrix[control_index].tolist(),
                        "other_model_k": int(row_by_id[treated_id]["other_model_k"]),
                        "category": str(row_by_id[treated_id].get("category", "")),
                    }
                    assignments.append(assignment)
                    current_assignments.append(assignment)

                # Balance diagnostics are emitted for every attempted pair.
                pair_assignments = [
                    item
                    for item in current_assignments
                    if item["language_pair"] == f"{first_language}__{second_language}"
                ]
                for feature_index, feature_name in enumerate(feature_names):
                    matched_treated = np.asarray(
                        [item["partial_features"][feature_index] for item in pair_assignments], dtype=np.float64
                    )
                    matched_control = np.asarray(
                        [item["control_features"][feature_index] for item in pair_assignments], dtype=np.float64
                    )
                    diagnostics.append(
                        {
                            "condition_id": condition["id"],
                            "match_type": match_type,
                            "confidence_margin_key": (
                                sweep.margin_key if match_type == "confidence" else None
                            ),
                            "k_correct": k_correct,
                            "language_pair": f"{first_language}__{second_language}",
                            "feature": feature_name,
                            "eligible_partial_pairs": len(treated_ids),
                            "matched_partial_pairs": len(pair_assignments),
                            "unique_controls": len({item["control_parallel_id"] for item in pair_assignments}),
                            "mean_match_distance": float(
                                np.mean([item["distance"] for item in pair_assignments])
                            )
                            if pair_assignments
                            else float("nan"),
                            "smd_before": standardized_mean_difference(
                                treated_matrix[:, feature_index], control_matrix[:, feature_index]
                            ),
                            "smd_after": standardized_mean_difference(matched_treated, matched_control),
                            "caliper_sd": caliper,
                            "active_distance_feature": bool(active[feature_index]) if feature_index < len(active) else False,
                        }
                    )

            for metric in ("spearman", "jaccard_at_10"):
                per_question_partial: dict[str, list[float]] = defaultdict(list)
                per_question_control: dict[str, list[float]] = defaultdict(list)
                for assignment in current_assignments:
                    first = int(assignment["first_language_index"])
                    second = int(assignment["second_language_index"])
                    partial_id = str(assignment["partial_parallel_id"])
                    control_id = str(assignment["control_parallel_id"])
                    partial_value = sweep.direct(metric, partial_id, first, second)
                    control_value = sweep.direct(metric, control_id, first, second)
                    if math.isfinite(partial_value) and math.isfinite(control_value):
                        per_question_partial[partial_id].append(partial_value)
                        per_question_control[partial_id].append(control_value)
                matched_ids = sorted(set(per_question_partial) & set(per_question_control))
                partial_values = np.asarray(
                    [np.mean(per_question_partial[parallel_id]) for parallel_id in matched_ids], dtype=np.float64
                )
                control_values = np.asarray(
                    [np.mean(per_question_control[parallel_id]) for parallel_id in matched_ids], dtype=np.float64
                )
                differences = partial_values - control_values
                ci_low, ci_high = bootstrap_mean_interval(
                    differences,
                    int(analysis_cfg["bootstrap_resamples"]),
                    float(analysis_cfg["ci_level"]),
                    stable_seed(
                        int(analysis_cfg["bootstrap_seed"]),
                        condition["id"],
                        match_type,
                        k_correct,
                        metric,
                    ),
                    int(analysis_cfg.get("bootstrap_block_size", 256)),
                )
                result_rows.append(
                    {
                        "condition_id": condition["id"],
                        "model": condition["model"],
                        "dataset": condition["dataset"],
                        "match_type": match_type,
                        "confidence_margin_key": (
                            sweep.margin_key if match_type == "confidence" else None
                        ),
                        "k_correct": k_correct,
                        "metric": metric,
                        "eligible_correct_correct_pair_rows": sum(
                            int(row["eligible_partial_pairs"])
                            for row in diagnostics
                            if row["condition_id"] == condition["id"]
                            and row["match_type"] == match_type
                            and int(row["k_correct"]) == k_correct
                            and row["feature"]
                            == (
                                f"margin_{row['language_pair'].split('__')[0]}"
                                if match_type == "confidence"
                                else structural_features[0]
                            )
                        ),
                        "matched_pair_rows": len(current_assignments),
                        "matched_partial_questions": len(matched_ids),
                        "unique_all_correct_controls": len(
                            {item["control_parallel_id"] for item in current_assignments}
                        ),
                        "partial_similarity": float(np.mean(partial_values)) if partial_values.size else float("nan"),
                        "matched_all_correct_similarity": float(np.mean(control_values))
                        if control_values.size
                        else float("nan"),
                        "partial_minus_all_correct": float(np.mean(differences))
                        if differences.size
                        else float("nan"),
                        "difference_ci_low": ci_low,
                        "difference_ci_high": ci_high,
                        "bootstrap_unit": "partial_question_fixed_match",
                    }
                )
    return result_rows, diagnostics, assignments


def _write_markdown(
    path: Path,
    similarity_rows: Sequence[Mapping[str, Any]],
    trend_rows: Sequence[Mapping[str, Any]],
    matched_rows: Sequence[Mapping[str, Any]],
) -> None:
    aggregate = [row for row in similarity_rows if row["scope"] == "aggregate"]
    confidence_margin_keys = {
        str(row.get("confidence_margin_key"))
        for row in matched_rows
        if row.get("match_type") == "confidence" and row.get("confidence_margin_key")
    }
    if confidence_margin_keys == {"base_margin_max"}:
        confidence_note = (
            "Confidence is the uncalibrated clean correct-option logit minus the "
            "strongest incorrect-option logit."
        )
    elif confidence_margin_keys == {"base_margin_mean"}:
        confidence_note = (
            "Confidence is the uncalibrated clean correct-option logit minus the "
            "mean incorrect-option logit."
        )
    else:
        confidence_note = "Confidence is an uncalibrated clean gold-option margin."
    lines = [
        "# Correctness-Stratified Usage-Signature Results",
        "",
        "All estimates use the newly pooled correctness-stratified sweeps. The shuffled null is the exact mean over all question-by-question comparisons within each K stratum.",
        "",
        "## Similarity by number of correctly answered languages",
        "",
        "| Model | Dataset | Metric | K | Questions | Aligned [95% CI] | Gap [95% CI] |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['model']} | {row['dataset']} | {row['metric']} | {row['k_correct']} | "
            f"{row['n_questions']} | {row['aligned_estimate']:.3f} "
            f"[{row['aligned_ci_low']:.3f}, {row['aligned_ci_high']:.3f}] | "
            f"{row['gap_estimate']:.3f} [{row['gap_ci_low']:.3f}, {row['gap_ci_high']:.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Ordinal trends",
            "",
            "| Model | Dataset | Metric | Outcome | Slope per language [95% CI] | Holm p |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for row in trend_rows:
        holm = row.get("permutation_p_holm")
        holm_text = "—" if holm is None else f"{float(holm):.4g}"
        lines.append(
            f"| {row['model']} | {row['dataset']} | {row['metric']} | {row['outcome']} | "
            f"{row['slope_per_additional_correct_language']:.4f} "
            f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}] | {holm_text} |"
        )
    lines.extend(
        [
            "",
            "## Matched partially correct subsets",
            "",
            "| Model | Dataset | Match | K | Metric | Partial questions | Partial − all-correct [95% CI] |",
            "|---|---|---|---:|---|---:|---:|",
        ]
    )
    for row in matched_rows:
        lines.append(
            f"| {row['model']} | {row['dataset']} | {row['match_type']} | {row['k_correct']} | "
            f"{row['metric']} | {row['matched_partial_questions']} | "
            f"{row['partial_minus_all_correct']:.3f} "
            f"[{row['difference_ci_low']:.3f}, {row['difference_ci_high']:.3f}] |"
        )
    lines.extend(
        [
            "",
            f"{confidence_note} Difficulty is a leave-model-out proxy based on the other model's five-language correctness count, with structural covariates. Matching uses replacement; intervals resample partial questions while holding the selected matches fixed.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    cfg = load_yaml(config_path)
    selection_root = resolve_path(config_dir, cfg["paths"]["selection_root"])
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else resolve_path(config_dir, cfg["paths"]["analysis_output_dir"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_cfg = dict(cfg["analysis"])
    if args.bootstrap_resamples is not None:
        analysis_cfg["bootstrap_resamples"] = args.bootstrap_resamples
    if args.permutation_draws is not None:
        analysis_cfg["permutation_draws"] = args.permutation_draws
    if int(analysis_cfg["bootstrap_resamples"]) <= 0 or int(analysis_cfg["permutation_draws"]) < 0:
        raise ValueError("Bootstrap resamples must be positive and permutation draws non-negative.")

    similarity_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    trend_rows: list[dict[str, Any]] = []
    matched_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    resolved_inputs: list[dict[str, Any]] = []

    for raw_condition in cfg["conditions"]:
        condition = dict(raw_condition)
        condition_id = str(condition["id"])
        condition_dir = selection_root / condition_id
        manifest_path = condition_dir / "question_manifest.jsonl"
        selected_path = condition_dir / "selected_ids.json"
        if not manifest_path.is_file() or not selected_path.is_file():
            raise FileNotFoundError(
                f"Run prepare_correctness_strata.py first; missing inputs under {condition_dir}"
            )
        manifest_rows = read_jsonl(manifest_path)
        selected_ids = {str(item) for item in read_json(selected_path)}
        manifest_selected = {
            str(row["parallel_id"]) for row in manifest_rows if bool(row.get("selected_for_sweep"))
        }
        if manifest_selected != selected_ids:
            raise ValueError(f"Selection manifest/ID list mismatch for {condition_id}")
        sweep_dir = resolve_path(config_dir, condition["sweep_output_dir"])
        sweep = SweepData(
            condition,
            sweep_dir,
            selected_ids,
            str(analysis_cfg["importance_metric"]),
            int(analysis_cfg["jaccard_k"]),
        )
        condition_similarity, condition_status, condition_trend_values = _stratum_similarity(
            condition, manifest_rows, sweep, analysis_cfg
        )
        condition_trends = _trend_rows(condition, condition_trend_values, analysis_cfg)
        condition_matched, condition_diagnostics, condition_assignments = _match_condition(
            condition, manifest_rows, sweep, analysis_cfg
        )
        similarity_rows.extend(condition_similarity)
        status_rows.extend(condition_status)
        trend_rows.extend(condition_trends)
        matched_rows.extend(condition_matched)
        diagnostic_rows.extend(condition_diagnostics)
        assignment_rows.extend(condition_assignments)
        resolved_inputs.append(
            {
                "condition_id": condition_id,
                "manifest_path": str(manifest_path.resolve()),
                "selected_ids_path": str(selected_path.resolve()),
                "sweep_output_dir": str(sweep_dir),
                "selected_questions": len(selected_ids),
            }
        )
        print(f"Analyzed {condition_id}: {len(selected_ids)} selected questions")

    gap_indices = [
        index
        for index, row in enumerate(trend_rows)
        if row["outcome"] == "gap" and math.isfinite(float(row["permutation_p_raw"]))
    ]
    if gap_indices:
        adjusted = holm_adjust([float(trend_rows[index]["permutation_p_raw"]) for index in gap_indices])
        for index, value in zip(gap_indices, adjusted):
            trend_rows[index]["permutation_p_holm"] = value

    write_csv(output_dir / "similarity_by_k.csv", similarity_rows)
    write_csv(output_dir / "similarity_by_pair_status.csv", status_rows)
    write_csv(output_dir / "ordinal_trends.csv", trend_rows)
    write_csv(output_dir / "matched_subset_results.csv", matched_rows)
    write_csv(output_dir / "matching_diagnostics.csv", diagnostic_rows)
    write_csv(output_dir / "matching_assignments.csv", assignment_rows)
    write_json(
        output_dir / "correctness_stratified_results.json",
        {
            "config_path": str(config_path),
            "settings": analysis_cfg,
            "resolved_inputs": resolved_inputs,
            "similarity_by_k": similarity_rows,
            "similarity_by_pair_status": status_rows,
            "ordinal_trends": trend_rows,
            "matched_subset_results": matched_rows,
            "matching_diagnostics": diagnostic_rows,
        },
    )
    _write_markdown(output_dir / "correctness_stratified_tables.md", similarity_rows, trend_rows, matched_rows)
    print(f"Saved analysis to {output_dir}")


if __name__ == "__main__":
    main()
