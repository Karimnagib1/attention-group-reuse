#!/usr/bin/env python3
"""Standalone uncertainty analysis for the paper's reported experiments.

This script deliberately does not import or modify the experiment runners. It reads
only the artifact paths explicitly listed in its configuration and writes new
statistical-support tables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import yaml


EPS = 1.0e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute confidence intervals and higher-resolution permutation tests for the paper."
    )
    parser.add_argument("--config", required=True, help="Path to the statistical-support YAML configuration.")
    parser.add_argument("--bootstrap-resamples", type=int, help="Override the configured bootstrap count.")
    parser.add_argument("--permutation-draws", type=int, help="Override the configured permutation count.")
    parser.add_argument("--output-dir", help="Override the configured output directory.")
    parser.add_argument(
        "--permutation-source",
        choices=["resample", "saved"],
        default="resample",
        help=(
            "'resample' (default) draws fresh permutations at --permutation-draws. "
            "'saved' reuses the paper's original 500-shuffle exceedance p-value, null mean/SD, "
            "and z-score from each sweep's baseline_report.json instead of drawing anything new; "
            "--permutation-draws is ignored in this mode."
        ),
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    return cfg


def resolve_path(config_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def percentile_interval(values: np.ndarray, ci_level: float) -> Tuple[float, float]:
    alpha = 1.0 - ci_level
    low, high = np.quantile(values[np.isfinite(values)], [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def point_centered_bootstrap_interval(
    estimate: float,
    bootstrap_values: np.ndarray,
    ci_level: float,
) -> Tuple[float, float, float]:
    """Normal interval centered on the estimate using the bootstrap standard error."""
    bootstrap_values = bootstrap_values[np.isfinite(bootstrap_values)]
    if bootstrap_values.size < 2:
        raise ValueError("Bootstrap standard-error intervals require at least two finite samples.")
    standard_error = float(np.std(bootstrap_values, ddof=1))
    alpha = 1.0 - ci_level
    critical_value = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    low = max(0.0, estimate - critical_value * standard_error)
    high = min(1.0, estimate + critical_value * standard_error)
    return float(low), float(high), standard_error


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, original_idx in enumerate(order):
        candidate = min(1.0, (m - rank) * float(p[original_idx]))
        running = max(running, candidate)
        adjusted[original_idx] = running
    return adjusted.tolist()


def load_balanced_tensor(input_dir: Path, metric: str) -> Tuple[np.ndarray, List[str], List[str]]:
    samples = json.loads((input_dir / "samples.json").read_text(encoding="utf-8"))
    metadata = json.loads((input_dir / "metadata.json").read_text(encoding="utf-8"))
    languages = [str(item) for item in metadata.get("languages", [])]
    if not languages:
        raise ValueError(f"No language list in {input_dir / 'metadata.json'}")

    with np.load(input_dir / "importance_tensors.npz") as archive:
        flat_key = f"flat_{metric}"
        if flat_key in archive:
            vectors = np.asarray(archive[flat_key], dtype=np.float32)
        elif metric in archive and archive[metric].ndim == 3:
            raw = np.asarray(archive[metric], dtype=np.float32)
            vectors = raw.reshape(raw.shape[0], -1)
        else:
            raise KeyError(f"Metric '{metric}' not found in {input_dir / 'importance_tensors.npz'}")

    index: Dict[Tuple[str, str], int] = {}
    all_parallel_ids: set[str] = set()
    for row in samples:
        parallel_id = str(row["parallel_id"])
        language = str(row["language"])
        sample_index = int(row["sample_index"])
        if (parallel_id, language) in index:
            raise ValueError(f"Duplicate sample ({parallel_id}, {language}) in {input_dir}")
        index[(parallel_id, language)] = sample_index
        all_parallel_ids.add(parallel_id)

    parallel_ids = sorted(
        parallel_id
        for parallel_id in all_parallel_ids
        if all((parallel_id, language) in index for language in languages)
    )
    if len(parallel_ids) != len(all_parallel_ids):
        raise ValueError(
            f"The paper analysis expects a balanced design, but {input_dir} has "
            f"{len(all_parallel_ids) - len(parallel_ids)} incomplete parallel questions."
        )

    tensor = np.stack(
        [vectors[[index[(parallel_id, language)] for parallel_id in parallel_ids]] for language in languages],
        axis=0,
    )
    return tensor, languages, parallel_ids


def rankdata_last_axis(values: np.ndarray) -> np.ndarray:
    """Return 1-based average ranks along the last axis, including ties."""
    original_shape = values.shape
    flat = np.asarray(values).reshape(-1, original_shape[-1])
    width = flat.shape[1]
    order = np.argsort(flat, axis=1, kind="mergesort")
    ranks = np.empty_like(flat, dtype=np.float32)
    ordinal = np.broadcast_to(np.arange(1, width + 1, dtype=np.float32), flat.shape)
    np.put_along_axis(ranks, order, ordinal, axis=1)

    for row_index in range(flat.shape[0]):
        sorted_values = flat[row_index, order[row_index]]
        starts = np.concatenate(
            [np.array([0], dtype=np.int64), np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
        )
        ends = np.concatenate([starts[1:], np.array([width], dtype=np.int64)])
        tied = (ends - starts) > 1
        for start, end in zip(starts[tied], ends[tied]):
            average_rank = 0.5 * (float(start) + float(end) - 1.0) + 1.0
            ranks[row_index, order[row_index, start:end]] = average_rank
    return ranks.reshape(original_shape)


def rank_normalized_rows(tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ranks = rankdata_last_axis(tensor)
    ranks -= ranks.mean(axis=2, keepdims=True)
    norms = np.linalg.norm(ranks, axis=2, keepdims=True)
    valid = norms[..., 0] > EPS
    normalized = np.divide(ranks, norms, out=np.zeros_like(ranks), where=norms > EPS)
    return normalized, valid


def rank_percentile_tensor(tensor: np.ndarray) -> np.ndarray:
    n_features = tensor.shape[2]
    if n_features <= 1:
        return np.zeros_like(tensor, dtype=np.float32)
    ranks = rankdata_last_axis(tensor)
    return (ranks - 1.0) / float(n_features - 1)


def jaccard_binary_rows(tensor: np.ndarray, k: int) -> np.ndarray:
    k = min(max(int(k), 1), tensor.shape[2])
    top = np.argpartition(tensor, -k, axis=2)[:, :, -k:]
    binary = np.zeros_like(tensor, dtype=np.float32)
    np.put_along_axis(binary, top, 1.0, axis=2)
    return binary


def spearman_similarity_matrix(
    normalized_ranks: np.ndarray,
    valid_rows: np.ndarray,
    first: int,
    second: int,
) -> np.ndarray:
    similarity = normalized_ranks[first] @ normalized_ranks[second].T
    valid = valid_rows[first][:, None] & valid_rows[second][None, :]
    similarity = similarity.astype(np.float32, copy=False)
    similarity[~valid] = np.nan
    return similarity


def jaccard_similarity_matrix(binary: np.ndarray, first: int, second: int, k: int) -> np.ndarray:
    intersection = binary[first] @ binary[second].T
    return (intersection / (2.0 * float(k) - intersection)).astype(np.float32, copy=False)


def make_bootstrap_counts(n_items: int, n_resamples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    probabilities = np.full(n_items, 1.0 / float(n_items), dtype=np.float64)
    return rng.multinomial(n_items, probabilities, size=n_resamples).astype(np.float32)


def bootstrap_similarity_matrix(
    similarity: np.ndarray,
    counts: np.ndarray,
    ci_level: float,
    block_size: int,
) -> Dict[str, float]:
    n_resamples, n_items = counts.shape
    diag = np.diag(similarity).astype(np.float32)
    diag_valid = np.isfinite(diag)
    matrix_valid = np.isfinite(similarity)
    matrix_zero = np.nan_to_num(similarity, nan=0.0).astype(np.float32, copy=False)

    aligned_point = float(np.mean(diag[diag_valid]))
    null_point = float(np.mean(similarity[matrix_valid]))
    aligned_samples = np.empty(n_resamples, dtype=np.float64)
    null_samples = np.empty(n_resamples, dtype=np.float64)

    all_finite = bool(np.all(matrix_valid) and np.all(diag_valid))
    diag_zero = np.nan_to_num(diag, nan=0.0)
    diag_valid_float = diag_valid.astype(np.float32)
    matrix_valid_float = matrix_valid.astype(np.float32) if not all_finite else None

    for start in range(0, n_resamples, block_size):
        stop = min(start + block_size, n_resamples)
        block = counts[start:stop]
        aligned_num = block @ diag_zero
        aligned_den = block @ diag_valid_float
        aligned_samples[start:stop] = aligned_num / aligned_den

        null_num = np.sum((block @ matrix_zero) * block, axis=1)
        if all_finite:
            null_den = np.square(block.sum(axis=1))
        else:
            assert matrix_valid_float is not None
            null_den = np.sum((block @ matrix_valid_float) * block, axis=1)
        null_samples[start:stop] = null_num / null_den

    gap_samples = aligned_samples - null_samples
    aligned_low, aligned_high = percentile_interval(aligned_samples, ci_level)
    null_low, null_high = percentile_interval(null_samples, ci_level)
    gap_low, gap_high = percentile_interval(gap_samples, ci_level)
    return {
        "aligned_estimate": aligned_point,
        "aligned_ci_low": aligned_low,
        "aligned_ci_high": aligned_high,
        "null_exact_estimate": null_point,
        "null_ci_low": null_low,
        "null_ci_high": null_high,
        "gap_estimate": aligned_point - null_point,
        "gap_ci_low": gap_low,
        "gap_ci_high": gap_high,
    }


def permutation_test(
    similarity: np.ndarray,
    n_draws: int,
    seed: int,
    block_size: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    n_items = similarity.shape[0]
    rows = np.arange(n_items, dtype=np.int64)[None, :]
    aligned = float(np.nanmean(np.diag(similarity)))
    exceedances = 0
    null_sum = 0.0
    null_sum_sq = 0.0
    produced = 0

    while produced < n_draws:
        current = min(block_size, n_draws - produced)
        permutations = np.stack([rng.permutation(n_items) for _ in range(current)], axis=0)
        sampled = similarity[rows, permutations]
        null_values = np.nanmean(sampled, axis=1)
        exceedances += int(np.sum(null_values >= aligned))
        null_sum += float(np.sum(null_values))
        null_sum_sq += float(np.sum(np.square(null_values)))
        produced += current

    null_mean = null_sum / n_draws
    variance = max(0.0, (null_sum_sq - n_draws * null_mean * null_mean) / max(n_draws - 1, 1))
    return {
        "permutation_draws": int(n_draws),
        "permutation_exceedances": exceedances,
        "permutation_p_value": (exceedances + 1.0) / (n_draws + 1.0),
        "permutation_null_mean": null_mean,
        "permutation_null_sd": math.sqrt(variance),
    }


def saved_null_matrices(input_dir: Path, metric: str, k: int) -> Dict[str, Any]:
    report_path = input_dir.parent / f"{input_dir.name}_baselines" / "baseline_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Expected paper baseline report at {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metric_block = report["results"][metric]

    def shuffle_block(similarity_key: str) -> Dict[str, np.ndarray]:
        key = "spearman" if similarity_key == "spearman" else "jaccard_topk"
        sub = metric_block["spearman"] if similarity_key == "spearman" else metric_block["jaccard_topk"][str(k)]
        null = sub["null_cross_language_shuffle"]
        return {
            "null_mean": np.asarray(null["null_mean"], dtype=np.float64),
            "null_std": np.asarray(null["null_std"], dtype=np.float64),
            "p_value_greater": np.asarray(null["p_value_greater"], dtype=np.float64),
            "z_score": np.asarray(null["z_score"], dtype=np.float64),
        }

    return {
        "languages": [str(item) for item in report["languages"]],
        "n_shuffles": int(report["n_shuffles"]),
        "spearman": shuffle_block("spearman")["null_mean"],
        "jaccard": shuffle_block("jaccard")["null_mean"],
        "spearman_shuffle": shuffle_block("spearman"),
        "jaccard_shuffle": shuffle_block("jaccard"),
    }


def permutation_stats_from_saved(
    shuffle: Mapping[str, np.ndarray], first: int, second: int, n_shuffles: int
) -> Dict[str, Any]:
    p_value = float(shuffle["p_value_greater"][first, second])
    exceedances_float = p_value * (n_shuffles + 1) - 1.0
    exceedances = int(round(exceedances_float))
    if exceedances < 0 or abs(exceedances - exceedances_float) > 1.0e-6:
        raise ValueError(
            f"Saved p_value_greater={p_value} is inconsistent with the +1-corrected formula "
            f"for n_shuffles={n_shuffles} (implied exceedances={exceedances_float})."
        )
    return {
        "permutation_draws": int(n_shuffles),
        "permutation_exceedances": exceedances,
        "permutation_p_value": p_value,
        "permutation_null_mean": float(shuffle["null_mean"][first, second]),
        "permutation_null_sd": float(shuffle["null_std"][first, second]),
        "z_score": float(shuffle["z_score"][first, second]),
    }


def compute_similarity_rows(
    conditions: Sequence[Mapping[str, Any]],
    quantitative_root: Path,
    settings: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    metric = str(settings["metric"])
    k = int(settings["jaccard_k"])
    n_bootstrap = int(settings["bootstrap_resamples"])
    n_permutations = int(settings["permutation_draws"])
    ci_level = float(settings["ci_level"])
    bootstrap_seed = int(settings["bootstrap_seed"])
    permutation_seed = int(settings["permutation_seed"])
    block_size = int(settings.get("matrix_block_size", 256))
    permutation_source = str(settings.get("permutation_source", "resample"))

    rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    for condition in conditions:
        input_dir = quantitative_root / str(condition["input_dir"])
        tensor, languages, parallel_ids = load_balanced_tensor(input_dir, metric)
        normalized_ranks, valid_ranks = rank_normalized_rows(tensor)
        jaccard_binary = jaccard_binary_rows(tensor, k)
        saved = saved_null_matrices(input_dir, metric, k)
        if saved["languages"] != languages:
            raise ValueError(f"Language order mismatch between sweep and baseline report for {input_dir}")
        if saved["n_shuffles"] != 500:
            raise ValueError(f"Expected the paper's 500-shuffle baseline in {input_dir}, got {saved['n_shuffles']}")

        counts = make_bootstrap_counts(
            len(parallel_ids),
            n_bootstrap,
            stable_seed(bootstrap_seed, condition["model"], condition["dataset"]),
        )
        aggregate_matrices = {
            "spearman": np.zeros((len(parallel_ids), len(parallel_ids)), dtype=np.float32),
            "jaccard@10": np.zeros((len(parallel_ids), len(parallel_ids)), dtype=np.float32),
        }
        pair_count = 0

        for first in range(len(languages)):
            for second in range(first + 1, len(languages)):
                matrices = {
                    "spearman": spearman_similarity_matrix(normalized_ranks, valid_ranks, first, second),
                    "jaccard@10": jaccard_similarity_matrix(jaccard_binary, first, second, k),
                }
                for similarity_name, matrix in matrices.items():
                    aggregate_matrices[similarity_name] += matrix
                    summary = bootstrap_similarity_matrix(matrix, counts, ci_level, block_size)
                    if permutation_source == "saved":
                        shuffle = saved["spearman_shuffle" if similarity_name == "spearman" else "jaccard_shuffle"]
                        perm = permutation_stats_from_saved(shuffle, first, second, saved["n_shuffles"])
                    else:
                        perm = permutation_test(
                            matrix,
                            n_permutations,
                            stable_seed(
                                permutation_seed,
                                condition["model"],
                                condition["dataset"],
                                similarity_name,
                                languages[first],
                                languages[second],
                            ),
                            block_size,
                        )
                    saved_matrix = saved["spearman" if similarity_name == "spearman" else "jaccard"]
                    row = {
                        "scope": "language_pair",
                        "model": condition["model"],
                        "dataset": condition["dataset"],
                        "metric": metric,
                        "similarity": similarity_name,
                        "language_a": languages[first],
                        "language_b": languages[second],
                        "n_questions": len(parallel_ids),
                        "bootstrap_resamples": n_bootstrap,
                        "ci_level": ci_level,
                        "null_saved_500_estimate": float(saved_matrix[first, second]),
                        **summary,
                        **perm,
                    }
                    rows.append(row)
                pair_count += 1

        for similarity_name, matrix_sum in aggregate_matrices.items():
            matrix = matrix_sum / float(pair_count)
            summary = bootstrap_similarity_matrix(matrix, counts, ci_level, block_size)
            if permutation_source == "saved":
                # baseline_report.json only stores per-language-pair 500-shuffle statistics; there is
                # no saved shuffle of the 10-pair-averaged matrix, so no permutation stats are fabricated
                # here. Per-pair saved statistics are reported in the appendix rows instead.
                perm: Dict[str, Any] = {}
            else:
                perm = permutation_test(
                    matrix,
                    n_permutations,
                    stable_seed(
                        permutation_seed,
                        condition["model"],
                        condition["dataset"],
                        similarity_name,
                        "aggregate",
                    ),
                    block_size,
                )
            saved_matrix = saved["spearman" if similarity_name == "spearman" else "jaccard"]
            upper = np.triu_indices(len(languages), k=1)
            rows.append(
                {
                    "scope": "aggregate_10_language_pairs",
                    "model": condition["model"],
                    "dataset": condition["dataset"],
                    "metric": metric,
                    "similarity": similarity_name,
                    "language_a": "ALL",
                    "language_b": "ALL",
                    "n_questions": len(parallel_ids),
                    "bootstrap_resamples": n_bootstrap,
                    "ci_level": ci_level,
                    "null_saved_500_estimate": float(np.mean(saved_matrix[upper])),
                    **summary,
                    **perm,
                }
            )

        audit_rows.append(
            {
                "model": condition["model"],
                "dataset": condition["dataset"],
                "input_dir": str(input_dir),
                "n_questions": len(parallel_ids),
                "languages": languages,
                "metric": metric,
            }
        )

    pair_indices = [index for index, row in enumerate(rows) if row["scope"] == "language_pair"]
    adjusted = holm_adjust([float(rows[index]["permutation_p_value"]) for index in pair_indices])
    for index, value in zip(pair_indices, adjusted):
        rows[index]["permutation_p_value_holm_80_tests"] = value
    for row in rows:
        row.setdefault("permutation_p_value_holm_80_tests", None)
    return rows, audit_rows


def variance_fraction_samples(
    x: np.ndarray,
    counts: np.ndarray,
    block_size: int,
) -> np.ndarray:
    n_resamples, _ = counts.shape
    n_languages = x.shape[0]
    question_means = x.mean(axis=0)
    question_mean_sq = np.sum(np.square(question_means), axis=1)
    total_sq = np.sum(np.square(x), axis=(0, 2))
    samples = np.empty((n_resamples, 3), dtype=np.float64)

    for start in range(0, n_resamples, block_size):
        stop = min(start + block_size, n_resamples)
        block = counts[start:stop].astype(np.float64, copy=False)
        total_weight = block.sum(axis=1)
        grand = (block @ question_means) / total_weight[:, None]
        grand_sq = np.sum(np.square(grand), axis=1)

        language_sq = np.zeros(stop - start, dtype=np.float64)
        for language in range(n_languages):
            language_mean = (block @ x[language]) / total_weight[:, None]
            language_sq += np.sum(np.square(language_mean), axis=1)

        ss_total = block @ total_sq - n_languages * total_weight * grand_sq
        ss_language = total_weight * language_sq - n_languages * total_weight * grand_sq
        ss_question = n_languages * (block @ question_mean_sq - total_weight * grand_sq)
        ss_interaction = ss_total - ss_language - ss_question
        components = np.stack([ss_language, ss_question, ss_interaction], axis=1)
        components[np.abs(components) < 1.0e-9] = 0.0
        samples[start:stop] = components / ss_total[:, None]
    return samples


def compute_variance_rows(
    conditions: Sequence[Mapping[str, Any]],
    quantitative_root: Path,
    settings: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    metric = str(settings["metric"])
    n_bootstrap = int(settings["bootstrap_resamples"])
    ci_level = float(settings["ci_level"])
    seed = int(settings["bootstrap_seed"])
    block_size = int(settings.get("variance_block_size", 256))
    component_names = ["language", "question", "interaction"]
    output: List[Dict[str, Any]] = []

    for condition in conditions:
        input_dir = quantitative_root / str(condition["input_dir"])
        tensor, languages, parallel_ids = load_balanced_tensor(input_dir, metric)
        normalized = rank_percentile_tensor(tensor).astype(np.float64)
        point_counts = np.ones((1, len(parallel_ids)), dtype=np.float64)
        point = variance_fraction_samples(normalized, point_counts, 1)[0]
        counts = make_bootstrap_counts(
            len(parallel_ids),
            n_bootstrap,
            stable_seed(seed, condition["model"], condition["dataset"], "variance"),
        )
        samples = variance_fraction_samples(normalized, counts, block_size)

        published_summary_path = (
            input_dir / "variance_decomposition_continuous" / metric / "summary.json"
        )
        if not published_summary_path.exists():
            raise FileNotFoundError(f"Missing paper variance summary: {published_summary_path}")
        published = json.loads(published_summary_path.read_text(encoding="utf-8"))[
            "global_fraction_ss_weighted"
        ]
        published_values = [
            float(published["f_lang"]),
            float(published["f_question"]),
            float(published["f_interaction"]),
        ]
        if not np.allclose(point, published_values, atol=2.0e-6, rtol=0.0):
            raise ValueError(
                f"Variance reproduction failed for {condition['model']} / {condition['dataset']}: "
                f"computed={point.tolist()}, saved={published_values}"
            )

        for component_index, component in enumerate(component_names):
            low, high, standard_error = point_centered_bootstrap_interval(
                float(point[component_index]),
                samples[:, component_index],
                ci_level,
            )
            output.append(
                {
                    "model": condition["model"],
                    "dataset": condition["dataset"],
                    "metric": metric,
                    "component": component,
                    "estimate": float(point[component_index]),
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_mean": float(np.mean(samples[:, component_index])),
                    "bootstrap_standard_error": standard_error,
                    "ci_level": ci_level,
                    "ci_method": "point_centered_normal_interval_from_question_cluster_bootstrap_se",
                    "bootstrap_resamples": n_bootstrap,
                    "n_questions": len(parallel_ids),
                    "n_languages": len(languages),
                    "normalization": "rank_percentile",
                    "aggregation": "global_ss_weighted",
                    "saved_paper_estimate": published_values[component_index],
                }
            )
    return output


def latent_metric_values(baseline: np.ndarray, patched: np.ndarray) -> Dict[str, float]:
    return {
        "baseline_top1_rate": float(np.mean(baseline == 1)),
        "patched_top1_rate": float(np.mean(patched == 1)),
        "delta_top1_rate": float(np.mean((patched == 1).astype(np.float64) - (baseline == 1))),
        "patched_top10_rate": float(np.mean(patched <= 10)),
        "patched_median_rank": float(np.median(patched)),
    }


def bootstrap_latent(
    baseline: np.ndarray,
    patched: np.ndarray,
    n_resamples: int,
    seed: int,
    ci_level: float,
    block_size: int,
    macro_conditions: bool,
) -> Dict[str, Tuple[float, float]]:
    if baseline.shape != patched.shape:
        raise ValueError("Baseline and patched latent-country arrays must have identical shapes.")
    if baseline.ndim == 1:
        baseline = baseline[None, :]
        patched = patched[None, :]
    n_conditions, n_pairs = baseline.shape
    rng = np.random.default_rng(seed)
    samples = {
        "baseline_top1_rate": np.empty(n_resamples),
        "patched_top1_rate": np.empty(n_resamples),
        "delta_top1_rate": np.empty(n_resamples),
        "patched_top10_rate": np.empty(n_resamples),
        "patched_median_rank": np.empty(n_resamples),
    }

    produced = 0
    while produced < n_resamples:
        current = min(block_size, n_resamples - produced)
        indices = rng.integers(0, n_pairs, size=(current, n_pairs))
        sampled_baseline = baseline[:, indices]
        sampled_patched = patched[:, indices]
        base_t1 = sampled_baseline == 1
        patch_t1 = sampled_patched == 1
        if macro_conditions:
            reduction_axes = (0, 2)
            median_values = np.median(sampled_patched, axis=2).mean(axis=0)
        else:
            reduction_axes = (0, 2)
            median_values = np.median(sampled_patched, axis=2)[0]
        sl = slice(produced, produced + current)
        samples["baseline_top1_rate"][sl] = np.mean(base_t1, axis=reduction_axes)
        samples["patched_top1_rate"][sl] = np.mean(patch_t1, axis=reduction_axes)
        samples["delta_top1_rate"][sl] = np.mean(
            patch_t1.astype(np.float64) - base_t1.astype(np.float64), axis=reduction_axes
        )
        samples["patched_top10_rate"][sl] = np.mean(sampled_patched <= 10, axis=reduction_axes)
        samples["patched_median_rank"][sl] = median_values
        produced += current

    return {metric: percentile_interval(values, ci_level) for metric, values in samples.items()}


def load_latent_manifest(
    manifest_path: Path,
    workspace_root: Path,
    settings: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    expected_total = int(settings["expected_total_conditions"])
    if len(manifest_rows) != expected_total:
        raise ValueError(f"Expected {expected_total} paper-used latent conditions, found {len(manifest_rows)}")

    allowed_folders = set(str(item) for item in settings["allowed_folders"])
    expected_by_family = {str(key): int(value) for key, value in settings["expected_conditions_by_family"].items()}
    actual_by_family: Dict[str, int] = {}
    loaded: List[Dict[str, Any]] = []

    for manifest_row in manifest_rows:
        if manifest_row["folder"] not in allowed_folders:
            raise ValueError(f"Manifest contains non-paper folder: {manifest_row['folder']}")
        actual_by_family[manifest_row["family"]] = actual_by_family.get(manifest_row["family"], 0) + 1
        json_path = Path(manifest_row["json_path"])
        if not json_path.exists():
            json_path = workspace_root / manifest_row["folder"] / Path(manifest_row["json_path"]).name
        if not json_path.exists():
            raise FileNotFoundError(f"Missing paper-used latent result: {json_path}")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        per_pair = sorted(payload["per_pair"], key=lambda row: int(row["pair_index"]))
        pair_indices = [int(row["pair_index"]) for row in per_pair]
        if pair_indices != list(range(100)):
            raise ValueError(f"Expected pair_index 0..99 in {json_path}, got {pair_indices[:5]}...")
        baseline = np.asarray([row["baseline"]["alt_rank"] for row in per_pair], dtype=np.int64)
        patched = np.asarray([row["patched"]["alt_rank"] for row in per_pair], dtype=np.int64)
        loaded.append(
            {
                **manifest_row,
                "json_path_resolved": str(json_path.resolve()),
                "baseline": baseline,
                "patched": patched,
            }
        )

    if actual_by_family != expected_by_family:
        raise ValueError(f"Latent manifest family counts do not match the paper: {actual_by_family}")
    return loaded


def compute_latent_rows(
    manifest_path: Path,
    workspace_root: Path,
    settings: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    loaded = load_latent_manifest(manifest_path, workspace_root, settings)
    n_bootstrap = int(settings["bootstrap_resamples"])
    ci_level = float(settings["ci_level"])
    seed = int(settings["bootstrap_seed"])
    block_size = int(settings.get("bootstrap_block_size", 512))
    metrics = [
        "baseline_top1_rate",
        "patched_top1_rate",
        "delta_top1_rate",
        "patched_top10_rate",
        "patched_median_rank",
    ]

    condition_rows: List[Dict[str, Any]] = []
    for item in loaded:
        point = latent_metric_values(item["baseline"], item["patched"])
        intervals = bootstrap_latent(
            item["baseline"],
            item["patched"],
            n_bootstrap,
            stable_seed(seed, item["experiment_id"]),
            ci_level,
            block_size,
            macro_conditions=False,
        )
        for metric in metrics:
            condition_rows.append(
                {
                    "scope": "condition",
                    "model": item["model"],
                    "setting": item["setting"],
                    "family": item["family"],
                    "experiment_id": item["experiment_id"],
                    "language_pair": item["language_pair"],
                    "task": item["task"],
                    "patch_spec": item["patch_spec"],
                    "metric": metric,
                    "estimate": point[metric],
                    "ci_low": intervals[metric][0],
                    "ci_high": intervals[metric][1],
                    "ci_level": ci_level,
                    "bootstrap_resamples": n_bootstrap,
                    "n_pairs": len(item["baseline"]),
                    "json_path": item["json_path_resolved"],
                }
            )

    aggregate_rows: List[Dict[str, Any]] = []
    families = list(dict.fromkeys(str(item["family"]) for item in loaded))
    for family in families:
        family_items = [item for item in loaded if item["family"] == family]
        baseline = np.stack([item["baseline"] for item in family_items])
        patched = np.stack([item["patched"] for item in family_items])
        point = {
            "baseline_top1_rate": float(np.mean(baseline == 1)),
            "patched_top1_rate": float(np.mean(patched == 1)),
            "delta_top1_rate": float(np.mean((patched == 1).astype(float) - (baseline == 1))),
            "patched_top10_rate": float(np.mean(patched <= 10)),
            "patched_median_rank": float(np.median(patched, axis=1).mean()),
        }
        intervals = bootstrap_latent(
            baseline,
            patched,
            n_bootstrap,
            stable_seed(seed, family, "aggregate"),
            ci_level,
            block_size,
            macro_conditions=True,
        )
        for metric in metrics:
            aggregate_rows.append(
                {
                    "scope": "model_setting_macro_average",
                    "model": family_items[0]["model"],
                    "setting": family_items[0]["setting"],
                    "family": family,
                    "metric": metric,
                    "estimate": point[metric],
                    "ci_low": intervals[metric][0],
                    "ci_high": intervals[metric][1],
                    "ci_level": ci_level,
                    "bootstrap_resamples": n_bootstrap,
                    "n_conditions": len(family_items),
                    "n_shared_pair_indices": baseline.shape[1],
                    "aggregation": "macro_average_over_conditions_clustered_by_pair_index",
                }
            )

    audit_rows = [
        {
            "experiment_id": item["experiment_id"],
            "family": item["family"],
            "json_path": item["json_path_resolved"],
            "n_pairs": len(item["baseline"]),
        }
        for item in loaded
    ]
    return aggregate_rows, condition_rows, audit_rows


def format_ci(estimate: float, low: float, high: float, digits: int = 3, percent: bool = False) -> str:
    scale = 100.0 if percent else 1.0
    return f"{estimate * scale:.{digits}f} [{low * scale:.{digits}f}, {high * scale:.{digits}f}]"


def render_paper_tables(
    similarity_rows: Sequence[Mapping[str, Any]],
    variance_rows: Sequence[Mapping[str, Any]],
    latent_rows: Sequence[Mapping[str, Any]],
    permutation_source: str = "resample",
) -> str:
    lines = [
        "# Paper-ready statistical tables",
        "",
        "Values are point estimates with 95% confidence intervals in brackets.",
        "",
        "## Revised Table 2: Multilingual signature similarity",
        "",
        "| Dataset | Model | Spearman aligned | Spearman null | Spearman gap | Jaccard@10 aligned | Jaccard@10 null | Jaccard@10 gap |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    aggregate_similarity = [row for row in similarity_rows if row["scope"] == "aggregate_10_language_pairs"]
    condition_order = list(dict.fromkeys((row["dataset"], row["model"]) for row in aggregate_similarity))
    for dataset, model in condition_order:
        by_metric = {
            row["similarity"]: row
            for row in aggregate_similarity
            if row["dataset"] == dataset and row["model"] == model
        }
        sp = by_metric["spearman"]
        ja = by_metric["jaccard@10"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(dataset),
                    str(model),
                    format_ci(sp["aligned_estimate"], sp["aligned_ci_low"], sp["aligned_ci_high"]),
                    format_ci(sp["null_exact_estimate"], sp["null_ci_low"], sp["null_ci_high"]),
                    format_ci(sp["gap_estimate"], sp["gap_ci_low"], sp["gap_ci_high"]),
                    format_ci(ja["aligned_estimate"], ja["aligned_ci_low"], ja["aligned_ci_high"]),
                    format_ci(ja["null_exact_estimate"], ja["null_ci_low"], ja["null_ci_high"]),
                    format_ci(ja["gap_estimate"], ja["gap_ci_low"], ja["gap_ci_high"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Revised Table 3: SS-weighted variance decomposition",
            "",
            "| Dataset | Model | Language | Question | Interaction |",
            "|---|---|---:|---:|---:|",
        ]
    )
    variance_order = list(dict.fromkeys((row["dataset"], row["model"]) for row in variance_rows))
    for dataset, model in variance_order:
        by_component = {
            row["component"]: row
            for row in variance_rows
            if row["dataset"] == dataset and row["model"] == model
        }
        values = [
            format_ci(
                by_component[component]["estimate"],
                by_component[component]["ci_low"],
                by_component[component]["ci_high"],
            )
            for component in ["language", "question", "interaction"]
        ]
        lines.append(f"| {dataset} | {model} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Revised Table 5: Multilingual latent-country case study",
            "",
            "Rates are percentages; patched rank is in vocabulary-rank units.",
            "",
            "| Model | Setting | Baseline top-1 (%) | Patched top-1 (%) | Top-1 change (pp) | Patched top-10 (%) | Patched median rank |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    latent_order = list(dict.fromkeys((row["model"], row["setting"]) for row in latent_rows))
    for model, setting in latent_order:
        by_metric = {
            row["metric"]: row
            for row in latent_rows
            if row["model"] == model and row["setting"] == setting
        }
        rate_values = []
        for metric in ["baseline_top1_rate", "patched_top1_rate", "delta_top1_rate", "patched_top10_rate"]:
            row = by_metric[metric]
            rate_values.append(format_ci(row["estimate"], row["ci_low"], row["ci_high"], digits=1, percent=True))
        rank_row = by_metric["patched_median_rank"]
        rank_value = format_ci(rank_row["estimate"], rank_row["ci_low"], rank_row["ci_high"], digits=1)
        lines.append(f"| {model} | {setting.title()} | " + " | ".join([*rate_values, rank_value]) + " |")

    lines.extend(
        [
            "",
            "## Permutation-test reporting note",
            "",
            (
                "The language-pair CSV contains the exact exceedance count, the +1-smoothed Monte Carlo "
                "p-value, the z-score against the saved null's mean/SD, and the Holm-adjusted value across "
                "the 80 paper-used pairwise tests. These are the paper's original 500 shuffles, not freshly "
                "drawn permutations. Report zero-exceedance results as `0/500 exceedances; Monte Carlo "
                "p < 1/501` rather than as `p = 0`; treat the z-score, not a normal-approximation tail "
                "p-value, as the non-saturating evidence."
                if permutation_source == "saved"
                else "The language-pair CSV contains the exact exceedance count, the +1-smoothed Monte Carlo p-value, and the Holm-adjusted value across the 80 paper-used pairwise tests. Report zero-exceedance results as `0/10,000 exceedances; Monte Carlo p < 1e-4` rather than as `p = 0`."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    cfg = load_config(config_path)

    analysis = cfg["analysis"]
    if args.bootstrap_resamples is not None:
        analysis["bootstrap_resamples"] = int(args.bootstrap_resamples)
    if args.permutation_draws is not None:
        analysis["permutation_draws"] = int(args.permutation_draws)
    if int(analysis["bootstrap_resamples"]) <= 0 or int(analysis["permutation_draws"]) <= 0:
        raise ValueError("Bootstrap and permutation counts must be positive.")
    analysis["permutation_source"] = args.permutation_source
    if args.permutation_source == "saved" and args.permutation_draws is not None:
        raise ValueError("--permutation-draws has no effect with --permutation-source saved; omit one of them.")

    workspace_root = resolve_path(config_dir, cfg["paths"]["workspace_root"])
    quantitative_root = resolve_path(config_dir, cfg["paths"]["quantitative_root"])
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else resolve_path(config_dir, cfg["paths"]["output_dir"])
    )
    manifest_path = resolve_path(config_dir, cfg["paths"]["latent_manifest_csv"])
    output_dir.mkdir(parents=True, exist_ok=True)

    common_settings = {
        **analysis,
        "metric": cfg["paper_scope"]["importance_metric"],
        "jaccard_k": cfg["paper_scope"]["jaccard_k"],
    }
    conditions = cfg["paper_scope"]["conditions"]
    print("Computing aligned-versus-null confidence intervals and permutation tests...")
    similarity_rows, similarity_audit = compute_similarity_rows(
        conditions, quantitative_root, common_settings
    )
    print("Computing variance-decomposition confidence intervals...")
    variance_rows = compute_variance_rows(conditions, quantitative_root, common_settings)
    print("Computing latent-country confidence intervals from the paper manifest...")
    latent_aggregate_rows, latent_condition_rows, latent_audit = compute_latent_rows(
        manifest_path,
        workspace_root,
        {**cfg["latent_country"], **analysis},
    )

    write_csv(output_dir / "table2_similarity_aggregate.csv", [row for row in similarity_rows if row["scope"].startswith("aggregate")])
    write_csv(output_dir / "appendix_similarity_language_pairs.csv", [row for row in similarity_rows if row["scope"] == "language_pair"])
    write_csv(output_dir / "table3_variance_decomposition.csv", variance_rows)
    write_csv(output_dir / "table5_latent_country_aggregate.csv", latent_aggregate_rows)
    write_csv(output_dir / "appendix_latent_country_conditions.csv", latent_condition_rows)

    paper_tables = render_paper_tables(
        similarity_rows, variance_rows, latent_aggregate_rows, permutation_source=args.permutation_source
    )
    (output_dir / "paper_tables.md").write_text(paper_tables, encoding="utf-8")
    payload = {
        "config_path": str(config_path),
        "settings": analysis,
        "paper_scope": cfg["paper_scope"],
        "artifact_audit": {
            "similarity_and_variance": similarity_audit,
            "latent_manifest": str(manifest_path),
            "latent_conditions": latent_audit,
        },
        "similarity": similarity_rows,
        "variance_decomposition": variance_rows,
        "latent_country_aggregate": latent_aggregate_rows,
        "latent_country_conditions": latent_condition_rows,
    }
    (output_dir / "statistical_support.json").write_text(
        json.dumps(json_ready(payload), indent=2), encoding="utf-8"
    )
    print(f"Wrote statistical-support outputs to {output_dir}")


if __name__ == "__main__":
    main()
