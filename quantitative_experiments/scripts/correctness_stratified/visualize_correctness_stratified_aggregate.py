#!/usr/bin/env python3
"""Numerically aggregate and visualize four model-dataset analyses."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from visualize_correctness_stratified import (
    DEFAULT_ANALYSIS_ROOT,
    REQUIRED_FILES,
    _read_csv,
    visualize_rows,
)


DEFAULT_CONDITIONS = (
    "llama_belebele",
    "llama_mmluprox",
    "qwen_belebele",
    "qwen_mmluprox",
)
AGGREGATE_CONDITION_ID = "four_condition_macro_average"
AGGREGATE_MODEL_LABEL = "Four model-dataset combinations average"
AGGREGATE_DATASET_LABEL = ""
AGGREGATION_METHOD = "equal_weight_mean_across_model_dataset_conditions"
INTERVAL_METHOD = "equal_weight_mean_of_condition_ci_endpoints_descriptive"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute an equal-weight numerical macro-average of separate "
            "model-dataset correctness-stratified results and visualize it."
        )
    )
    parser.add_argument(
        "--analysis-root",
        default=str(DEFAULT_ANALYSIS_ROOT),
        help=f"Analysis root containing by_condition/ (default: {DEFAULT_ANALYSIS_ROOT}).",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=DEFAULT_CONDITIONS,
        metavar="CONDITION",
        help="Condition directories to combine (default: the four model-dataset combinations).",
    )
    parser.add_argument(
        "--output-dir",
        help="Figure directory (default: <analysis-root>/aggregate_viz).",
    )
    parser.add_argument(
        "--table-dir",
        help=(
            "Numerically aggregated CSV directory "
            "(default: <analysis-root>/aggregate_tables)."
        ),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png", "pdf"),
        help="One or more output formats (default: png pdf).",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Raster output DPI (default: 300).")
    return parser.parse_args(argv)


def _validate_condition_rows(
    condition: str,
    filename: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if not rows:
        raise ValueError(f"No rows found in {condition}/{filename}")
    observed = {str(row.get("condition_id", "")) for row in rows}
    if observed != {condition}:
        raise ValueError(
            f"Expected condition_id={condition!r} in {condition}/{filename}, "
            f"found {sorted(observed)!r}"
        )


def load_condition_tables(
    analysis_root: Path,
    conditions: Sequence[str],
) -> Dict[str, List[Dict[str, str]]]:
    """Load and concatenate each required table in the requested order."""
    if not conditions:
        raise ValueError("At least one condition is required.")
    if len(set(conditions)) != len(conditions):
        raise ValueError("Condition names must be unique.")

    combined: Dict[str, List[Dict[str, str]]] = {
        filename: [] for filename in REQUIRED_FILES
    }
    for condition in conditions:
        condition_dir = analysis_root / "by_condition" / condition
        for filename in REQUIRED_FILES:
            rows = _read_csv(condition_dir / filename)
            _validate_condition_rows(condition, filename, rows)
            combined[filename].extend(rows)
    return combined


def _float(row: Mapping[str, Any], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [_float(row, key) for row in rows]
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else float("nan")


def _sum_int(rows: Sequence[Mapping[str, Any]], key: str) -> int:
    values = [_float(row, key) for row in rows]
    return int(sum(value for value in values if math.isfinite(value)))


def _groups(
    rows: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
    conditions: Sequence[str],
) -> List[Tuple[Tuple[str, ...], List[Mapping[str, Any]]]]:
    grouped: Dict[Tuple[str, ...], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row[name]) for name in keys)
        grouped.setdefault(key, []).append(row)

    expected = set(conditions)
    for key, group in grouped.items():
        observed = {str(row["condition_id"]) for row in group}
        if observed != expected or len(group) != len(conditions):
            raise ValueError(
                f"Incomplete aggregate cell {dict(zip(keys, key))}: "
                f"expected {sorted(expected)}, found {sorted(observed)}"
            )
    return list(grouped.items())


def _identity(n_conditions: int) -> Dict[str, Any]:
    return {
        "condition_id": AGGREGATE_CONDITION_ID,
        "model": AGGREGATE_MODEL_LABEL,
        "dataset": AGGREGATE_DATASET_LABEL,
        "n_conditions": n_conditions,
        "aggregation_method": AGGREGATION_METHOD,
        "interval_method": INTERVAL_METHOD,
    }


def _aggregate_similarity(
    rows: Sequence[Mapping[str, Any]], conditions: Sequence[str]
) -> List[Dict[str, Any]]:
    rows = [row for row in rows if str(row.get("scope")) == "aggregate"]
    output: List[Dict[str, Any]] = []
    for (metric, k_correct), group in _groups(
        rows, ("metric", "k_correct"), conditions
    ):
        output.append(
            {
                **_identity(len(conditions)),
                "metric": metric,
                "k_correct": k_correct,
                "scope": "aggregate",
                "language_pair": "four_condition_macro_average",
                "n_questions": _sum_int(group, "n_questions"),
                **{
                    key: _mean(group, key)
                    for key in (
                        "aligned_estimate",
                        "aligned_ci_low",
                        "aligned_ci_high",
                        "null_exact_estimate",
                        "null_ci_low",
                        "null_ci_high",
                        "gap_estimate",
                        "gap_ci_low",
                        "gap_ci_high",
                    )
                },
            }
        )
    return output


def _aggregate_status(
    rows: Sequence[Mapping[str, Any]], conditions: Sequence[str]
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for (metric, k_correct, pair_status), group in _groups(
        rows, ("metric", "k_correct", "pair_status"), conditions
    ):
        output.append(
            {
                **_identity(len(conditions)),
                "metric": metric,
                "k_correct": k_correct,
                "pair_status": pair_status,
                "n_questions": _sum_int(group, "n_questions"),
                **{
                    key: _mean(group, key)
                    for key in (
                        "aligned_estimate",
                        "aligned_ci_low",
                        "aligned_ci_high",
                        "null_exact_estimate",
                        "null_ci_low",
                        "null_ci_high",
                        "gap_estimate",
                        "gap_ci_low",
                        "gap_ci_high",
                    )
                },
            }
        )
    return output


def _aggregate_trends(
    rows: Sequence[Mapping[str, Any]], conditions: Sequence[str]
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for (metric, outcome), group in _groups(rows, ("metric", "outcome"), conditions):
        output.append(
            {
                **_identity(len(conditions)),
                "metric": metric,
                "outcome": outcome,
                "n_questions": _sum_int(group, "n_questions"),
                "k_min": min(int(float(row["k_min"])) for row in group),
                "k_max": max(int(float(row["k_max"])) for row in group),
                "slope_per_additional_correct_language": _mean(
                    group, "slope_per_additional_correct_language"
                ),
                "ci_low": _mean(group, "ci_low"),
                "ci_high": _mean(group, "ci_high"),
                "permutation_p_raw": float("nan"),
                "permutation_p_holm": float("nan"),
            }
        )
    return output


def _aggregate_matched(
    rows: Sequence[Mapping[str, Any]], conditions: Sequence[str]
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for (match_type, metric, k_correct), group in _groups(
        rows, ("match_type", "metric", "k_correct"), conditions
    ):
        output.append(
            {
                **_identity(len(conditions)),
                "match_type": match_type,
                "metric": metric,
                "k_correct": k_correct,
                **{
                    key: _sum_int(group, key)
                    for key in (
                        "eligible_correct_correct_pair_rows",
                        "matched_pair_rows",
                        "matched_partial_questions",
                        "unique_all_correct_controls",
                    )
                },
                **{
                    key: _mean(group, key)
                    for key in (
                        "partial_similarity",
                        "matched_all_correct_similarity",
                        "partial_minus_all_correct",
                        "difference_ci_low",
                        "difference_ci_high",
                    )
                },
            }
        )
    return output


def aggregate_condition_tables(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    conditions: Sequence[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """Return equal-weight numerical macro-averages for every plotted cell."""
    return {
        "similarity_by_k.csv": _aggregate_similarity(
            tables["similarity_by_k.csv"], conditions
        ),
        "similarity_by_pair_status.csv": _aggregate_status(
            tables["similarity_by_pair_status.csv"], conditions
        ),
        "ordinal_trends.csv": _aggregate_trends(
            tables["ordinal_trends.csv"], conditions
        ),
        "matched_subset_results.csv": _aggregate_matched(
            tables["matched_subset_results.csv"], conditions
        ),
    }


def write_aggregate_tables(
    tables: Mapping[str, Sequence[Mapping[str, Any]]], table_dir: Path
) -> List[Path]:
    table_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for filename, rows in tables.items():
        if not rows:
            raise ValueError(f"No aggregate rows available for {filename}")
        path = table_dir / filename
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        written.append(path)
    return written


def visualize_aggregate(
    analysis_root: Path,
    conditions: Sequence[str],
    output_dir: Path,
    table_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    source_tables = load_condition_tables(analysis_root, conditions)
    tables = aggregate_condition_tables(source_tables, conditions)
    write_aggregate_tables(tables, table_dir)
    return visualize_rows(
        similarity_rows=tables["similarity_by_k.csv"],
        status_rows=tables["similarity_by_pair_status.csv"],
        trend_rows=tables["ordinal_trends.csv"],
        matched_rows=tables["matched_subset_results.csv"],
        output_dir=output_dir,
        formats=formats,
        dpi=dpi,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    analysis_root = Path(args.analysis_root).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else analysis_root / "aggregate_viz"
    )
    table_dir = (
        Path(args.table_dir).resolve()
        if args.table_dir
        else analysis_root / "aggregate_tables"
    )
    conditions: Tuple[str, ...] = tuple(args.conditions)
    try:
        written = visualize_aggregate(
            analysis_root=analysis_root,
            conditions=conditions,
            output_dir=output_dir,
            table_dir=table_dir,
            formats=tuple(args.formats),
            dpi=int(args.dpi),
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error

    print(
        f"Macro-averaged {len(conditions)} conditions from "
        f"{analysis_root / 'by_condition'}"
    )
    print(f"Saved 4 numerical aggregate tables to {table_dir}")
    print(f"Saved {len(written)} visualization files to {output_dir}")
    for path in written:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
