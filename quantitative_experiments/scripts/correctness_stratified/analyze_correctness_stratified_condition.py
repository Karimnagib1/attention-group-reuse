#!/usr/bin/env python3
"""Analyze one configured correctness-stratified model/dataset condition.

This is a single-condition driver around the analysis functions in
``analyze_correctness_stratified.py``. It intentionally leaves the existing
all-condition entrypoint unchanged.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from analyze_correctness_stratified import (
    SweepData,
    _match_condition,
    _stratum_similarity,
    _trend_rows,
    _write_markdown,
)
from common import (
    holm_adjust,
    load_yaml,
    read_json,
    read_jsonl,
    resolve_path,
    write_csv,
    write_json,
)


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "correctness_stratified"
    / "analysis.yaml"
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze one model/dataset condition from the correctness-stratified sweep."
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
    parser.add_argument("--permutation-draws", type=int, help="Override ordinal-trend permutations.")
    parser.add_argument(
        "--output-dir",
        help=(
            "Override the output directory. By default results are written under "
            "the configured analysis directory at "
            "by_condition_on_diagonal/<condition_id>."
        ),
    )
    return parser.parse_args(argv)


def configured_condition_ids(cfg: Mapping[str, Any]) -> list[str]:
    conditions = cfg.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("The analysis config must contain a non-empty 'conditions' list.")
    ids = [str(condition["id"]) for condition in conditions]
    duplicates = sorted({condition_id for condition_id in ids if ids.count(condition_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate condition IDs in analysis config: {', '.join(duplicates)}")
    return ids


def select_condition(cfg: Mapping[str, Any], condition_id: str) -> dict[str, Any]:
    valid_ids = configured_condition_ids(cfg)
    if condition_id not in valid_ids:
        raise ValueError(
            f"Unknown condition '{condition_id}'. Valid condition IDs: {', '.join(valid_ids)}"
        )
    return dict(next(item for item in cfg["conditions"] if str(item["id"]) == condition_id))


def _apply_holm_within_condition(trend_rows: list[dict[str, Any]]) -> int:
    gap_indices = [
        index
        for index, row in enumerate(trend_rows)
        if row["outcome"] == "gap" and math.isfinite(float(row["permutation_p_raw"]))
    ]
    if gap_indices:
        adjusted = holm_adjust(
            [float(trend_rows[index]["permutation_p_raw"]) for index in gap_indices]
        )
        for index, value in zip(gap_indices, adjusted):
            trend_rows[index]["permutation_p_holm"] = value
    return len(gap_indices)


def run_condition_analysis(
    *,
    config_path: Path,
    condition_id: str,
    bootstrap_resamples: Optional[int] = None,
    permutation_draws: Optional[int] = None,
    output_dir_override: Optional[Path] = None,
    confidence_margin_key: str = "base_margin_mean",
    match_types: Sequence[str] = ("confidence", "difficulty"),
    default_output_subdir: str = "by_condition_on_diagonal",
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
        else configured_output_root / default_output_subdir / condition_id
    )

    condition_dir = selection_root / condition_id
    manifest_path = condition_dir / "question_manifest.jsonl"
    selected_path = condition_dir / "selected_ids.json"
    if not manifest_path.is_file() or not selected_path.is_file():
        raise FileNotFoundError(
            "Run prepare_correctness_strata.py first for the selected condition; "
            f"expected {manifest_path} and {selected_path}"
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
        confidence_margin_key,
    )
    similarity_rows, status_rows, trend_values = _stratum_similarity(
        condition, manifest_rows, sweep, analysis_cfg
    )
    trend_rows = _trend_rows(condition, trend_values, analysis_cfg)
    matched_rows, diagnostic_rows, assignment_rows = _match_condition(
        condition, manifest_rows, sweep, analysis_cfg, match_types
    )
    holm_test_count = _apply_holm_within_condition(trend_rows)

    resolved_inputs = [
        {
            "condition_id": condition_id,
            "manifest_path": str(manifest_path.resolve()),
            "selected_ids_path": str(selected_path.resolve()),
            "sweep_output_dir": str(sweep_dir),
            "selected_questions": len(selected_ids),
        }
    ]
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
            "matching_variant": {
                "match_types": list(match_types),
                "confidence_margin_key": confidence_margin_key,
            },
            "multiple_testing": {
                "method": "Holm",
                "scope": "selected_condition_gap_trend_tests",
                "number_of_tests": holm_test_count,
            },
            "resolved_inputs": resolved_inputs,
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
    print(
        "Holm correction scope: selected condition gap-trend tests "
        f"({holm_test_count} tests)"
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
