#!/usr/bin/env python3
"""Analyze one condition with a different-question, off-diagonal null.

This entry point is intentionally separate from the original condition
analysis.  It excludes every same-question cell from the shuffled baseline
and uses one set of multinomial question weights to recompute the aligned,
off-diagonal-null, and gap estimates in each bootstrap replicate.
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from analyze_correctness_stratified import (
    SweepData,
    _match_condition,
    _stratum_similarity,
    _trend_rows,
)
from analyze_correctness_stratified_condition import (
    _apply_holm_within_condition,
    select_condition,
)
from common import (
    load_yaml,
    read_json,
    read_jsonl,
    resolve_path,
    stable_seed,
    write_csv,
    write_json,
)


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "correctness_stratified"
    / "analysis.yaml"
)
NULL_DEFINITION = "different_question_offdiagonal"
BOOTSTRAP_METHOD = "multinomial_parallel_question_weights_recomputed"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze one correctness-stratified condition using an exact "
            "off-diagonal different-question null."
        )
    )
    parser.add_argument(
        "--condition",
        required=True,
        help="Configured condition ID, for example qwen_belebele.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"Correctness-stratified analysis YAML (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument("--bootstrap-resamples", type=int, help="Override bootstrap resamples.")
    parser.add_argument("--permutation-draws", type=int, help="Override trend permutations.")
    parser.add_argument(
        "--output-dir",
        help=(
            "Override the output directory. By default results are isolated under "
            "the configured analysis directory at "
            "by_condition_offdiagonal/<condition_id>."
        ),
    )
    return parser.parse_args(argv)


def _finite_mean_matrix(matrices: Sequence[np.ndarray]) -> np.ndarray:
    """Average language-pair matrices cellwise without emitting all-NaN warnings."""
    stacked = np.asarray(matrices, dtype=np.float64)
    finite = np.isfinite(stacked)
    counts = finite.sum(axis=0)
    totals = np.where(finite, stacked, 0.0).sum(axis=0)
    return np.divide(
        totals,
        counts,
        out=np.full(totals.shape, np.nan, dtype=np.float64),
        where=counts > 0,
    )


def _offdiagonal_contributions(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned and symmetric different-question contributions by question."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Similarity matrix must be square.")
    offdiagonal = matrix.copy()
    np.fill_diagonal(offdiagonal, np.nan)
    finite = np.isfinite(offdiagonal)
    row_counts = finite.sum(axis=1)
    column_counts = finite.sum(axis=0)
    values = np.where(finite, offdiagonal, 0.0)
    row_means = np.divide(
        values.sum(axis=1),
        row_counts,
        out=np.full(matrix.shape[0], np.nan, dtype=np.float64),
        where=row_counts > 0,
    )
    column_means = np.divide(
        values.sum(axis=0),
        column_counts,
        out=np.full(matrix.shape[0], np.nan, dtype=np.float64),
        where=column_counts > 0,
    )
    return np.diag(matrix).astype(np.float64), 0.5 * (row_means + column_means)


def _percentile_interval(values: np.ndarray, ci_level: float) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    alpha = 1.0 - ci_level
    low, high = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def _cluster_bootstrap_summary(
    matrix: np.ndarray,
    *,
    draws: int,
    ci_level: float,
    seed: int,
    block_size: int,
    retain_draws: bool = False,
) -> dict[str, Any]:
    """Summarize a matrix using an off-diagonal null and question weights.

    A bootstrap draw gives each original parallel-question ID a multinomial
    count.  Diagonal and off-diagonal means are recomputed from those weights.
    Same original IDs remain excluded from the null even when their bootstrap
    count is greater than one.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Similarity matrix must be square.")
    n_questions = matrix.shape[0]
    if n_questions < 2:
        raise ValueError("The off-diagonal null requires at least two questions.")
    if draws <= 0:
        raise ValueError("Bootstrap resamples must be positive.")
    if not 0.0 < ci_level < 1.0:
        raise ValueError("CI level must lie strictly between zero and one.")
    if block_size <= 0:
        raise ValueError("Bootstrap block size must be positive.")

    diagonal = np.diag(matrix).astype(np.float64)
    diagonal_valid = np.isfinite(diagonal)
    diagonal_values = np.where(diagonal_valid, diagonal, 0.0)

    offdiagonal = matrix.copy()
    np.fill_diagonal(offdiagonal, np.nan)
    offdiagonal_valid = np.isfinite(offdiagonal)
    offdiagonal_values = np.where(offdiagonal_valid, offdiagonal, 0.0)

    aligned_estimate = (
        float(diagonal_values.sum() / diagonal_valid.sum())
        if diagonal_valid.any()
        else float("nan")
    )
    null_exact_estimate = (
        float(offdiagonal_values.sum() / offdiagonal_valid.sum())
        if offdiagonal_valid.any()
        else float("nan")
    )
    gap_estimate = aligned_estimate - null_exact_estimate

    aligned_draws = np.full(draws, np.nan, dtype=np.float64)
    null_draws = np.full(draws, np.nan, dtype=np.float64)
    rng = np.random.default_rng(seed)
    probabilities = np.full(n_questions, 1.0 / n_questions, dtype=np.float64)
    validity = offdiagonal_valid.astype(np.float64)

    for start in range(0, draws, block_size):
        stop = min(draws, start + block_size)
        weights = rng.multinomial(
            n_questions,
            probabilities,
            size=stop - start,
        ).astype(np.float64)

        aligned_denominator = weights @ diagonal_valid.astype(np.float64)
        aligned_numerator = weights @ diagonal_values
        aligned_draws[start:stop] = np.divide(
            aligned_numerator,
            aligned_denominator,
            out=np.full(stop - start, np.nan, dtype=np.float64),
            where=aligned_denominator > 0,
        )

        # w_i w_j reweights entire question clusters. The zero diagonal in
        # both arrays ensures that every i == j comparison remains excluded.
        null_numerator = np.sum((weights @ offdiagonal_values) * weights, axis=1)
        null_denominator = np.sum((weights @ validity) * weights, axis=1)
        null_draws[start:stop] = np.divide(
            null_numerator,
            null_denominator,
            out=np.full(stop - start, np.nan, dtype=np.float64),
            where=null_denominator > 0,
        )

    gap_draws = aligned_draws - null_draws
    aligned_low, aligned_high = _percentile_interval(aligned_draws, ci_level)
    null_low, null_high = _percentile_interval(null_draws, ci_level)
    gap_low, gap_high = _percentile_interval(gap_draws, ci_level)
    summary: dict[str, Any] = {
        "n_questions": int(diagonal_valid.sum()),
        "aligned_estimate": aligned_estimate,
        "aligned_ci_low": aligned_low,
        "aligned_ci_high": aligned_high,
        "null_exact_estimate": null_exact_estimate,
        "null_ci_low": null_low,
        "null_ci_high": null_high,
        "gap_estimate": gap_estimate,
        "gap_ci_low": gap_low,
        "gap_ci_high": gap_high,
        "null_definition": NULL_DEFINITION,
        "null_includes_diagonal": False,
        "null_finite_cells": int(offdiagonal_valid.sum()),
        "bootstrap_unit": "parallel_question_id",
        "bootstrap_method": BOOTSTRAP_METHOD,
        "bootstrap_resamples": draws,
    }
    if retain_draws:
        summary["_aligned_bootstrap_draws"] = aligned_draws
        summary["_gap_bootstrap_draws"] = gap_draws
    return summary


def _aggregate_offdiagonal_similarity(
    condition: Mapping[str, Any],
    manifest_rows: Sequence[Mapping[str, Any]],
    sweep: SweepData,
    analysis_cfg: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[int, dict[str, np.ndarray]]],
    dict[str, dict[int, dict[str, np.ndarray]]],
]:
    metrics = ("spearman", "jaccard_at_10")
    language_pairs = list(itertools.combinations(range(len(sweep.languages)), 2))
    rows: list[dict[str, Any]] = []
    trend_values: dict[str, dict[int, dict[str, np.ndarray]]] = {
        metric: {} for metric in metrics
    }
    trend_bootstrap_means: dict[str, dict[int, dict[str, np.ndarray]]] = {
        metric: {} for metric in metrics
    }

    for k_correct in range(len(sweep.languages) + 1):
        ids = sorted(
            str(row["parallel_id"])
            for row in manifest_rows
            if bool(row["selected_for_sweep"])
            and int(row["k_correct"]) == k_correct
        )
        if len(ids) < 2:
            continue
        positions = sweep.positions(ids)
        for metric in metrics:
            matrices = [
                sweep.matrix(metric, first, second, positions)
                for first, second in language_pairs
            ]
            aggregate_matrix = _finite_mean_matrix(matrices)
            summary = _cluster_bootstrap_summary(
                aggregate_matrix,
                draws=int(analysis_cfg["bootstrap_resamples"]),
                ci_level=float(analysis_cfg["ci_level"]),
                seed=stable_seed(
                    int(analysis_cfg["bootstrap_seed"]),
                    condition["id"],
                    metric,
                    k_correct,
                    NULL_DEFINITION,
                ),
                block_size=int(analysis_cfg.get("bootstrap_block_size", 256)),
                retain_draws=True,
            )
            aligned_bootstrap_draws = np.asarray(
                summary.pop("_aligned_bootstrap_draws"), dtype=np.float64
            )
            gap_bootstrap_draws = np.asarray(
                summary.pop("_gap_bootstrap_draws"), dtype=np.float64
            )
            rows.append(
                {
                    "condition_id": condition["id"],
                    "model": condition["model"],
                    "dataset": condition["dataset"],
                    "metric": metric,
                    "k_correct": k_correct,
                    "scope": "aggregate",
                    "language_pair": "mean_of_all_10_pairs",
                    **summary,
                }
            )

            aligned, null = _offdiagonal_contributions(aggregate_matrix)
            valid = np.isfinite(aligned) & np.isfinite(null)
            trend_values[metric][k_correct] = {
                "aligned": aligned[valid],
                "gap": (aligned - null)[valid],
            }
            trend_bootstrap_means[metric][k_correct] = {
                "aligned": aligned_bootstrap_draws,
                "gap": gap_bootstrap_draws,
            }
    return rows, trend_values, trend_bootstrap_means


def _replace_trend_bootstrap_intervals(
    trend_rows: list[dict[str, Any]],
    trend_values: Mapping[str, Mapping[int, Mapping[str, np.ndarray]]],
    trend_bootstrap_means: Mapping[
        str, Mapping[int, Mapping[str, np.ndarray]]
    ],
    ci_level: float,
) -> None:
    """Use the recomputed per-K matrix bootstraps for ordinal slope intervals."""
    for row in trend_rows:
        metric = str(row["metric"])
        outcome = str(row["outcome"])
        ks = sorted(
            k
            for k, values in trend_values[metric].items()
            if np.asarray(values[outcome]).size
        )
        if len(ks) < 2:
            continue
        weights = np.asarray(
            [np.asarray(trend_values[metric][k][outcome]).size for k in ks],
            dtype=np.float64,
        )
        k_values = np.asarray(ks, dtype=np.float64)
        x_mean = float(np.average(k_values, weights=weights))
        denominator = float(np.sum(weights * np.square(k_values - x_mean)))
        means = np.stack(
            [
                np.asarray(trend_bootstrap_means[metric][k][outcome], dtype=np.float64)
                for k in ks
            ],
            axis=1,
        )
        slopes = np.sum(
            weights[None, :] * (k_values - x_mean)[None, :] * means,
            axis=1,
        ) / denominator
        ci_low, ci_high = _percentile_interval(slopes, ci_level)
        row["ci_low"] = ci_low
        row["ci_high"] = ci_high
        row["bootstrap_method"] = BOOTSTRAP_METHOD


def _aligned_only_status_rows(
    condition: Mapping[str, Any],
    manifest_rows: Sequence[Mapping[str, Any]],
    sweep: SweepData,
    analysis_cfg: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Retain the status visualization while avoiding a mismatched null."""
    _, status_rows, _ = _stratum_similarity(
        condition,
        manifest_rows,
        sweep,
        analysis_cfg,
    )
    for row in status_rows:
        for key in (
            "null_exact_estimate",
            "null_ci_low",
            "null_ci_high",
            "gap_estimate",
            "gap_ci_low",
            "gap_ci_high",
        ):
            row[key] = float("nan")
        row["null_definition"] = "not_reported_for_pair_status"
    return status_rows


def _write_markdown(
    path: Path,
    similarity_rows: Sequence[Mapping[str, Any]],
    trend_rows: Sequence[Mapping[str, Any]],
    matched_rows: Sequence[Mapping[str, Any]],
) -> None:
    def formatted(value: Any, digits: int) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "not estimated"
        return f"{numeric:.{digits}g}" if math.isfinite(numeric) else "not estimated"

    lines = [
        "# Correctness-Stratified Off-Diagonal Results",
        "",
        (
            "The shuffled baseline is the exact mean over different-question "
            "cells only. Same-question diagonal cells are excluded. Bootstrap "
            "replicates reweight whole parallel-question IDs and recompute the "
            "aligned, null, and paired gap estimates."
        ),
        "",
        "## Similarity by number of correctly answered languages",
        "",
        "| Model | Dataset | Metric | K | Questions | Aligned [95% CI] | Off-diagonal null [95% CI] | Gap [95% CI] |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in similarity_rows:
        lines.append(
            f"| {row['model']} | {row['dataset']} | {row['metric']} | "
            f"{row['k_correct']} | {row['n_questions']} | "
            f"{row['aligned_estimate']:.3f} [{row['aligned_ci_low']:.3f}, {row['aligned_ci_high']:.3f}] | "
            f"{row['null_exact_estimate']:.3f} [{row['null_ci_low']:.3f}, {row['null_ci_high']:.3f}] | "
            f"{row['gap_estimate']:.3f} [{row['gap_ci_low']:.3f}, {row['gap_ci_high']:.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Ordinal gap trends",
            "",
            "| Model | Dataset | Metric | Slope per language [95% CI] | Holm p |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in trend_rows:
        if row["outcome"] != "gap":
            continue
        lines.append(
            f"| {row['model']} | {row['dataset']} | {row['metric']} | "
            f"{row['slope_per_additional_correct_language']:.4f} "
            f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}] | "
            f"{formatted(row['permutation_p_holm'], 4)} |"
        )
    lines.extend(
        [
            "",
            "## Matched subsets",
            "",
            "The matched-subset estimand is unchanged from the original analysis.",
            "",
            f"Rows produced: {len(matched_rows)}.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_condition_analysis(
    *,
    config_path: Path,
    condition_id: str,
    bootstrap_resamples: Optional[int] = None,
    permutation_draws: Optional[int] = None,
    output_dir_override: Optional[Path] = None,
) -> Path:
    config_path = config_path.resolve()
    config_dir = config_path.parent
    cfg = load_yaml(config_path)
    condition = select_condition(cfg, condition_id)

    analysis_cfg = dict(cfg["analysis"])
    if bootstrap_resamples is not None:
        analysis_cfg["bootstrap_resamples"] = bootstrap_resamples
    if permutation_draws is not None:
        analysis_cfg["permutation_draws"] = permutation_draws
    if int(analysis_cfg["bootstrap_resamples"]) <= 0:
        raise ValueError("Bootstrap resamples must be positive.")
    if int(analysis_cfg["permutation_draws"]) < 0:
        raise ValueError("Permutation draws must be non-negative.")

    selection_root = resolve_path(config_dir, cfg["paths"]["selection_root"])
    configured_output_root = resolve_path(config_dir, cfg["paths"]["analysis_output_dir"])
    output_dir = (
        output_dir_override.resolve()
        if output_dir_override is not None
        else configured_output_root / "by_condition_offdiagonal" / condition_id
    )

    condition_dir = selection_root / condition_id
    manifest_path = condition_dir / "question_manifest.jsonl"
    selected_path = condition_dir / "selected_ids.json"
    if not manifest_path.is_file() or not selected_path.is_file():
        raise FileNotFoundError(
            "Run prepare_correctness_strata.py first; expected "
            f"{manifest_path} and {selected_path}"
        )
    manifest_rows = read_jsonl(manifest_path)
    selected_ids = {str(item) for item in read_json(selected_path)}
    manifest_selected = {
        str(row["parallel_id"])
        for row in manifest_rows
        if bool(row.get("selected_for_sweep"))
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
    similarity_rows, trend_values, trend_bootstrap_means = _aggregate_offdiagonal_similarity(
        condition,
        manifest_rows,
        sweep,
        analysis_cfg,
    )
    status_rows = _aligned_only_status_rows(
        condition,
        manifest_rows,
        sweep,
        analysis_cfg,
    )
    trend_rows = _trend_rows(condition, trend_values, analysis_cfg)
    _replace_trend_bootstrap_intervals(
        trend_rows,
        trend_values,
        trend_bootstrap_means,
        float(analysis_cfg["ci_level"]),
    )
    matched_rows, diagnostic_rows, assignment_rows = _match_condition(
        condition,
        manifest_rows,
        sweep,
        analysis_cfg,
    )
    holm_test_count = _apply_holm_within_condition(trend_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
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
            "condition_id": condition_id,
            "settings": analysis_cfg,
            "null_analysis": {
                "definition": NULL_DEFINITION,
                "includes_same_question_diagonal": False,
                "bootstrap_unit": "parallel_question_id",
                "bootstrap_method": BOOTSTRAP_METHOD,
                "paired_aligned_null_weights": True,
            },
            "multiple_testing": {
                "method": "Holm",
                "scope": "selected_condition_gap_trend_tests",
                "number_of_tests": holm_test_count,
            },
            "resolved_inputs": [
                {
                    "condition_id": condition_id,
                    "manifest_path": str(manifest_path.resolve()),
                    "selected_ids_path": str(selected_path.resolve()),
                    "sweep_output_dir": str(sweep_dir),
                    "selected_questions": len(selected_ids),
                }
            ],
            "similarity_by_k": similarity_rows,
            "similarity_by_pair_status": status_rows,
            "ordinal_trends": trend_rows,
            "matched_subset_results": matched_rows,
            "matching_diagnostics": diagnostic_rows,
        },
    )
    _write_markdown(
        output_dir / "correctness_stratified_tables.md",
        similarity_rows,
        trend_rows,
        matched_rows,
    )
    print(f"Analyzed {condition_id}: {len(selected_ids)} selected questions")
    print("Null: exact different-question off-diagonal mean (diagonal excluded)")
    print(
        "Bootstrap: multinomial parallel-question weights; aligned, null, and "
        "gap recomputed together"
    )
    print(f"Saved analysis to {output_dir}")
    return output_dir


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    try:
        run_condition_analysis(
            config_path=Path(args.config),
            condition_id=str(args.condition),
            bootstrap_resamples=args.bootstrap_resamples,
            permutation_draws=args.permutation_draws,
            output_dir_override=Path(args.output_dir) if args.output_dir else None,
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
