from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DRIVER = (
    ROOT
    / "scripts"
    / "correctness_stratified"
    / "visualize_correctness_stratified_aggregate.py"
)
CONDITIONS = (
    "llama_belebele",
    "llama_mmluprox",
    "qwen_belebele",
    "qwen_mmluprox",
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_condition(condition_dir: Path, condition: str) -> None:
    condition_dir.mkdir(parents=True)
    model, dataset = condition.split("_", maxsplit=1)
    condition_offset = 0.01 * CONDITIONS.index(condition)
    similarity = []
    statuses = []
    trends = []
    matched = []
    for metric in ("spearman", "jaccard_at_10"):
        for k_correct in range(6):
            estimate = 0.2 + 0.03 * k_correct + condition_offset
            similarity.append(
                {
                    "condition_id": condition,
                    "model": model,
                    "dataset": dataset,
                    "metric": metric,
                    "k_correct": k_correct,
                    "scope": "aggregate",
                    "aligned_estimate": estimate,
                    "aligned_ci_low": estimate - 0.01,
                    "aligned_ci_high": estimate + 0.01,
                    "null_exact_estimate": 0.1,
                    "null_ci_low": 0.09,
                    "null_ci_high": 0.11,
                    "gap_estimate": estimate - 0.1,
                    "gap_ci_low": estimate - 0.11,
                    "gap_ci_high": estimate - 0.09,
                }
            )
            for status in (
                "correct_correct",
                "correct_incorrect",
                "incorrect_incorrect",
            ):
                statuses.append(
                    {
                        "condition_id": condition,
                        "model": model,
                        "dataset": dataset,
                        "metric": metric,
                        "k_correct": k_correct,
                        "pair_status": status,
                        "aligned_estimate": estimate,
                        "aligned_ci_low": estimate - 0.01,
                        "aligned_ci_high": estimate + 0.01,
                    }
                )
        trends.append(
            {
                "condition_id": condition,
                "model": model,
                "dataset": dataset,
                "metric": metric,
                "outcome": "gap",
                "n_questions": 100,
                "k_min": 0,
                "k_max": 5,
                "slope_per_additional_correct_language": 0.03 + condition_offset,
                "ci_low": 0.02 + condition_offset,
                "ci_high": 0.04 + condition_offset,
                "permutation_p_holm": 0.01,
            }
        )
        for match_type in ("confidence", "difficulty"):
            for k_correct in (2, 3, 4):
                matched.append(
                    {
                        "condition_id": condition,
                        "model": model,
                        "dataset": dataset,
                        "match_type": match_type,
                        "metric": metric,
                        "k_correct": k_correct,
                        "partial_minus_all_correct": -0.02,
                        "difference_ci_low": -0.03,
                        "difference_ci_high": -0.01,
                    }
                )
    _write_csv(condition_dir / "similarity_by_k.csv", similarity)
    _write_csv(condition_dir / "similarity_by_pair_status.csv", statuses)
    _write_csv(condition_dir / "ordinal_trends.csv", trends)
    _write_csv(condition_dir / "matched_subset_results.csv", matched)


class AggregateCorrectnessVisualizationTests(unittest.TestCase):
    def test_cli_combines_the_four_condition_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            analysis_root = Path(temporary) / "analysis"
            for condition in CONDITIONS:
                _write_condition(
                    analysis_root / "by_condition" / condition,
                    condition,
                )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(DRIVER),
                    "--analysis-root",
                    str(analysis_root),
                    "--formats",
                    "png",
                    "--dpi",
                    "72",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            output_dir = analysis_root / "aggregate_viz"
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {
                    "aligned_vs_null_by_k.png",
                    "aligned_vs_null_spearman.png",
                    "aligned_vs_null_jaccard_at_10.png",
                    "gap_by_k.png",
                    "pair_status_by_k.png",
                    "pair_status_spearman.png",
                    "pair_status_jaccard_at_10.png",
                    "matched_subsets.png",
                    "matched_subsets_confidence_spearman.png",
                    "matched_subsets_confidence_jaccard_at_10.png",
                    "matched_subsets_difficulty_spearman.png",
                    "matched_subsets_difficulty_jaccard_at_10.png",
                },
            )
            table_dir = analysis_root / "aggregate_tables"
            self.assertEqual(
                {path.name for path in table_dir.iterdir()},
                {
                    "similarity_by_k.csv",
                    "similarity_by_pair_status.csv",
                    "ordinal_trends.csv",
                    "matched_subset_results.csv",
                },
            )
            with (table_dir / "similarity_by_k.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            first = next(
                row
                for row in rows
                if row["metric"] == "spearman" and row["k_correct"] == "0"
            )
            self.assertEqual(first["condition_id"], "four_condition_macro_average")
            self.assertAlmostEqual(float(first["aligned_estimate"]), 0.215)
            self.assertIn("Macro-averaged 4 conditions", completed.stdout)


if __name__ == "__main__":
    unittest.main()
