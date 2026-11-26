from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "correctness_stratified"
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_correctness_stratified_condition_offdiagonal import (  # noqa: E402
    _cluster_bootstrap_summary,
    _offdiagonal_contributions,
)


class OffDiagonalNullTests(unittest.TestCase):
    def test_point_null_excludes_every_diagonal_cell(self) -> None:
        matrix = np.asarray(
            [
                [10.0, 1.0, 2.0],
                [3.0, 20.0, 4.0],
                [5.0, 6.0, 30.0],
            ]
        )
        aligned, null = _offdiagonal_contributions(matrix)
        self.assertEqual(aligned.tolist(), [10.0, 20.0, 30.0])
        self.assertAlmostEqual(float(np.mean(null)), 3.5)

        summary = _cluster_bootstrap_summary(
            matrix,
            draws=50,
            ci_level=0.95,
            seed=17,
            block_size=8,
        )
        self.assertAlmostEqual(summary["aligned_estimate"], 20.0)
        self.assertAlmostEqual(summary["null_exact_estimate"], 3.5)
        self.assertAlmostEqual(summary["gap_estimate"], 16.5)
        self.assertFalse(summary["null_includes_diagonal"])
        self.assertEqual(summary["null_finite_cells"], 6)

    def test_bootstrap_recomputes_paired_gap_and_is_reproducible(self) -> None:
        # If any duplicate bootstrap copies were allowed back onto the null
        # diagonal, this null would become positive. It must remain exactly 0.
        matrix = np.diag([2.0, 4.0, 8.0, 16.0])
        first = _cluster_bootstrap_summary(
            matrix,
            draws=100,
            ci_level=0.95,
            seed=123,
            block_size=11,
        )
        second = _cluster_bootstrap_summary(
            matrix,
            draws=100,
            ci_level=0.95,
            seed=123,
            block_size=11,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["null_exact_estimate"], 0.0)
        self.assertEqual(first["null_ci_low"], 0.0)
        self.assertEqual(first["null_ci_high"], 0.0)
        self.assertAlmostEqual(first["gap_estimate"], first["aligned_estimate"])
        self.assertGreater(first["gap_ci_high"], first["gap_ci_low"])


if __name__ == "__main__":
    unittest.main()
