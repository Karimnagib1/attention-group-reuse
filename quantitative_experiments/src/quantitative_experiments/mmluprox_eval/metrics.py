from __future__ import annotations

from typing import Dict


def accuracy(correct: int, total: int) -> Dict[str, float]:
    return {
        "accuracy": 0.0 if total == 0 else correct / total,
        "total": float(total),
        "correct": float(correct),
    }

