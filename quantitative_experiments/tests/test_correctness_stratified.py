from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "correctness_stratified"
sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    bootstrap_mean_interval,
    normalized_rank_rows,
    proportional_pattern_sample,
    topk_binary_rows,
)
from analyze_correctness_stratified import (  # noqa: E402
    SweepData,
    _match_condition,
    _stratum_similarity,
    _trend_rows,
)


class CommonTests(unittest.TestCase):
    def test_pattern_cap_is_configurable_and_deterministic(self) -> None:
        rows = [
            {"parallel_id": f"a{index}", "correctness_pattern": "11000"} for index in range(7)
        ] + [{"parallel_id": f"b{index}", "correctness_pattern": "10100"} for index in range(3)]
        first, quotas = proportional_pattern_sample(rows, 5, 17, "condition", 2)
        second, _ = proportional_pattern_sample(rows, 5, 17, "condition", 2)
        all_rows, all_quotas = proportional_pattern_sample(rows, None, 17, "condition", 2)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertGreater(quotas["11000"], quotas["10100"])
        self.assertEqual(len(all_rows), 10)
        self.assertEqual(sum(all_quotas.values()), 10)

    def test_rank_and_topk_helpers_handle_ties(self) -> None:
        values = np.asarray([[1.0, 1.0, 3.0, 2.0], [4.0, 3.0, 2.0, 1.0]])
        normalized, valid = normalized_rank_rows(values)
        self.assertTrue(np.all(valid))
        self.assertTrue(np.allclose(np.linalg.norm(normalized, axis=1), 1.0))
        binary = topk_binary_rows(values, 2)
        self.assertEqual(binary[0].tolist(), [0.0, 0.0, 1.0, 1.0])

    def test_bootstrap_is_reproducible(self) -> None:
        values = np.asarray([0.0, 1.0, 2.0, 3.0])
        first = bootstrap_mean_interval(values, 100, 0.95, 123)
        second = bootstrap_mean_interval(values, 100, 0.95, 123)
        self.assertEqual(first, second)


class SyntheticWorkflowTests(unittest.TestCase):
    def _write_sweep(self, root: Path) -> tuple[dict, list[dict], set[str]]:
        languages = ["l0", "l1", "l2", "l3", "l4"]
        ids = [f"q{k}" for k in range(6)]
        samples = []
        vectors = []
        margins = []
        sample_index = 0
        for question_index, parallel_id in enumerate(ids):
            for language_index, language in enumerate(languages):
                samples.append(
                    {
                        "sample_index": sample_index,
                        "parallel_id": parallel_id,
                        "language": language,
                    }
                )
                vectors.append(np.roll(np.arange(1.0, 9.0), question_index + language_index))
                margins.append(float(language_index + 1))
                sample_index += 1
        metadata = {
            "languages": languages,
            "target_mode": "correct",
            "patch_mode": "kv_group",
        }
        (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        (root / "samples.json").write_text(json.dumps(samples), encoding="utf-8")
        np.savez_compressed(
            root / "importance_tensors.npz",
            flat_pct_var_mean_incorrect=np.asarray(vectors),
            base_margin_mean=np.asarray(margins),
            base_margin_max=np.asarray(margins) + 10.0,
        )
        condition = {
            "id": "synthetic",
            "model": "model",
            "dataset": "dataset",
            "dataset_family": "mmluprox",
            "languages": languages,
        }
        manifest = []
        for k_correct, parallel_id in enumerate(ids):
            correct = {language: index < k_correct for index, language in enumerate(languages)}
            manifest.append(
                {
                    "parallel_id": parallel_id,
                    "k_correct": k_correct,
                    "correctness_pattern": "".join("1" if correct[x] else "0" for x in languages),
                    "correct_by_language": correct,
                    "selected_for_sweep": True,
                    "other_model_k": 5,
                    "category": "category",
                    "question_words": 10,
                    "context_words": 20,
                    "option_count": 4,
                    "mean_option_words": 3,
                }
            )
        return condition, manifest, set(ids)

    def test_synthetic_analysis_and_matching(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            condition, manifest, selected = self._write_sweep(root)
            sweep = SweepData(condition, root, selected, "pct_var_mean_incorrect", 2)
            settings = {
                "bootstrap_resamples": 20,
                "permutation_draws": 20,
                "ci_level": 0.95,
                "bootstrap_seed": 42,
                "permutation_seed": 123,
                "bootstrap_block_size": 8,
                "min_cell_questions": 1,
                "confidence_caliper_sd": 1.0,
                "difficulty_caliper_sd": 1.0,
            }
            similarity, status, trend_values = _stratum_similarity(condition, manifest, sweep, settings)
            aggregate = [row for row in similarity if row["scope"] == "aggregate"]
            self.assertEqual(len(aggregate), 12)
            k_zero_cc = [
                row
                for row in status
                if row["k_correct"] == 0 and row["pair_status"] == "correct_correct"
            ]
            self.assertTrue(k_zero_cc)
            self.assertTrue(all(not row["estimable"] for row in k_zero_cc))
            trends = _trend_rows(condition, trend_values, settings)
            self.assertEqual(len(trends), 4)

            matched, diagnostics, assignments = _match_condition(condition, manifest, sweep, settings)
            self.assertTrue(assignments)
            self.assertTrue(diagnostics)
            self.assertEqual(len(matched), 12)
            self.assertTrue(all(row["matched_partial_questions"] == 1 for row in matched))

            max_margin_sweep = SweepData(
                condition,
                root,
                selected,
                "pct_var_mean_incorrect",
                2,
                margin_key="base_margin_max",
            )
            self.assertEqual(max_margin_sweep.margin("q0", 0), 11.0)
            max_matched, max_diagnostics, max_assignments = _match_condition(
                condition,
                manifest,
                max_margin_sweep,
                settings,
                match_types=("confidence",),
            )
            self.assertEqual(len(max_matched), 6)
            self.assertTrue(max_diagnostics)
            self.assertTrue(max_assignments)
            self.assertTrue(
                all(
                    row["confidence_margin_key"] == "base_margin_max"
                    for row in max_matched + max_diagnostics + max_assignments
                )
            )


if __name__ == "__main__":
    unittest.main()
