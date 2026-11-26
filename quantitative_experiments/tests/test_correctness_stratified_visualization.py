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
    / "visualize_correctness_stratified.py"
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class CorrectnessStratifiedVisualizationTests(unittest.TestCase):
    def test_visualization_cli_writes_all_figures_under_viz(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            analysis_dir = Path(temporary) / "analysis"
            analysis_dir.mkdir()
            similarity = []
            statuses = []
            for metric in ("spearman", "jaccard_at_10"):
                for k_correct in range(6):
                    aligned = 0.20 + 0.04 * k_correct
                    null = 0.10 + 0.02 * k_correct
                    similarity.append(
                        {
                            "condition_id": "test_condition",
                            "model": "Test Model",
                            "dataset": "Test Dataset",
                            "metric": metric,
                            "k_correct": k_correct,
                            "scope": "aggregate",
                            "aligned_estimate": aligned,
                            "aligned_ci_low": aligned - 0.02,
                            "aligned_ci_high": aligned + 0.02,
                            "null_exact_estimate": null,
                            "null_ci_low": null - 0.01,
                            "null_ci_high": null + 0.01,
                            "gap_estimate": aligned - null,
                            "gap_ci_low": aligned - null - 0.02,
                            "gap_ci_high": aligned - null + 0.02,
                        }
                    )
                    for status_index, status in enumerate(
                        (
                            "correct_correct",
                            "correct_incorrect",
                            "incorrect_incorrect",
                        )
                    ):
                        estimate = aligned - 0.04 * status_index
                        statuses.append(
                            {
                                "condition_id": "test_condition",
                                "model": "Test Model",
                                "dataset": "Test Dataset",
                                "metric": metric,
                                "k_correct": k_correct,
                                "pair_status": status,
                                "aligned_estimate": estimate,
                                "aligned_ci_low": estimate - 0.02,
                                "aligned_ci_high": estimate + 0.02,
                            }
                        )
            trends = [
                {
                    "condition_id": "test_condition",
                    "model": "Test Model",
                    "dataset": "Test Dataset",
                    "metric": metric,
                    "outcome": "gap",
                    "slope_per_additional_correct_language": 0.02,
                    "permutation_p_holm": 0.01,
                }
                for metric in ("spearman", "jaccard_at_10")
            ]
            matched = []
            for match_type in ("confidence", "difficulty"):
                for metric in ("spearman", "jaccard_at_10"):
                    for k_correct in (2, 3, 4):
                        estimate = -0.02 * (5 - k_correct)
                        matched.append(
                            {
                                "condition_id": "test_condition",
                                "model": "Test Model",
                                "dataset": "Test Dataset",
                                "match_type": match_type,
                                "metric": metric,
                                "k_correct": k_correct,
                                "partial_minus_all_correct": estimate,
                                "difference_ci_low": estimate - 0.02,
                                "difference_ci_high": estimate + 0.02,
                            }
                        )
            _write_csv(analysis_dir / "similarity_by_k.csv", similarity)
            _write_csv(
                analysis_dir / "similarity_by_pair_status.csv", statuses
            )
            _write_csv(analysis_dir / "ordinal_trends.csv", trends)
            _write_csv(analysis_dir / "matched_subset_results.csv", matched)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(DRIVER),
                    "--input-dir",
                    str(analysis_dir),
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
            viz_dir = analysis_dir / "viz"
            self.assertEqual(
                {path.name for path in viz_dir.iterdir()},
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
            for path in viz_dir.iterdir():
                self.assertGreater(path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
