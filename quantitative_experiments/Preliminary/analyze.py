"""CPU analysis for the head-level vs group-level preliminary run.

Reuses the estimator from the donor-robustness analysis (review/donor_ablation/analysis.py):
for each language pair, aligned = same question in both languages; control = mean over
all 50 x 50 question pairings (exact shuffled-pairing expectation); gap = aligned - control;
95% percentile CI from 10,000 bootstraps of the 50 question IDs, resampled jointly across
the 10 language pairs. Results are equal-weight means over the 10 pairs.

Writes Preliminary/runs/qwen_mmluprox_50/analysis/{results.csv,results.md}.
Works with whichever runs exist; the paper's saved group-level run is always included.
"""
from __future__ import annotations

import csv
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

QUANT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(QUANT_ROOT / "src"), str(QUANT_ROOT)]
from review.donor_ablation.analysis import (  # noqa: E402
    bootstrap_weights, ci, normalized_ranks, similarity_matrices, weighted_statistics,
)

RUN_DIR = QUANT_ROOT / "Preliminary/runs/qwen_mmluprox_50"
PAPER_GROUP = QUANT_ROOT / "qwen_outputs/mmluprox/multilingual_head_patch/test"
METRIC = "pct_var_mean_incorrect"
LANGUAGES = ["en", "ru", "es", "de", "ar"]
GROUP_SIZE = 4
K = 10
BOOTSTRAPS, SEED = 10_000, 42


def load(directory: Path, ids: list[str]) -> np.ndarray:
    """Importance tensor ordered as [question, language, layer, unit]."""
    samples = json.loads((directory / "samples.json").read_text())
    index = {(s["parallel_id"], s["language"]): i for i, s in enumerate(samples)}
    with np.load(directory / "importance_tensors.npz") as archive:
        tensor = archive[METRIC]
    rows = [index[(q, lang)] for q in ids for lang in LANGUAGES]
    return tensor[rows].reshape(len(ids), len(LANGUAGES), *tensor.shape[1:]).astype(np.float64)


def gap_summary(signatures: np.ndarray, weights: np.ndarray) -> dict:
    """signatures: [question, language, dims]."""
    per_metric: dict[str, list] = {}
    for i, j in combinations(range(len(LANGUAGES)), 2):
        for metric, matrix in similarity_matrices(signatures[:, i], signatures[:, j], K).items():
            per_metric.setdefault(metric, []).append(weighted_statistics(matrix, weights))
    out = {}
    for metric, stats in per_metric.items():
        aggregate = np.mean(stats, axis=0)
        out[metric] = (aggregate[0, 0], aggregate[0, 1], aggregate[0, 2], *ci(aggregate[1:, 2]))
    return out


def rowwise_spearman(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> tuple[float, float, float]:
    """Mean over questions of the language-averaged per-(question, language) Spearman, with CI."""
    n = a.shape[0]
    rho = np.sum(normalized_ranks(a.reshape(-1, a.shape[-1])) * normalized_ranks(b.reshape(-1, b.shape[-1])), axis=1)
    per_question = rho.reshape(n, -1).mean(axis=1)
    estimates = weights @ per_question / n
    return (estimates[0], *ci(estimates[1:]))


def main() -> None:
    ids = json.loads((RUN_DIR / "question_ids.json").read_text())
    n = len(ids)
    weights = np.vstack([np.ones(n), bootstrap_weights(n, BOOTSTRAPS, SEED)])
    out_dir = RUN_DIR / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors = {"group (paper run, saved)": load(PAPER_GROUP, ids)}
    if (RUN_DIR / "kv_group/_SUCCESS").exists():
        tensors["group (rerun, same env as head)"] = load(RUN_DIR / "kv_group", ids)
    head = load(RUN_DIR / "head", ids) if (RUN_DIR / "head/_SUCCESS").exists() else None
    if head is not None:
        tensors["head (1,152 heads)"] = head
        tensors["head summed within group (288)"] = head.reshape(*head.shape[:3], -1, GROUP_SIZE).sum(-1)

    rows, lines = [], [f"# Head vs group patching, Qwen3-8B / MMLU-ProX, {n} questions\n",
                       "Means over the 10 language pairs; 95% percentile bootstrap CI over questions "
                       f"({BOOTSTRAPS:,} resamples).\n",
                       "| Level | Metric | Aligned | Cross-language control | Gap [95% CI] |",
                       "|---|---|---:|---:|---|"]
    for level, tensor in tensors.items():
        signatures = tensor.reshape(n, len(LANGUAGES), -1)
        for metric, (aligned, control, gap, low, high) in gap_summary(signatures, weights).items():
            rows.append({"level": level, "metric": metric, "n_units": signatures.shape[-1], "aligned": aligned,
                         "control": control, "gap": gap, "gap_ci_low": low, "gap_ci_high": high})
            lines.append(f"| {level} | {metric} | {aligned:.4f} | {control:.4f} | {gap:.4f} [{low:.4f}, {high:.4f}] |")

    lines.append("")
    checks = []
    if head is not None:
        summed = tensors["head summed within group (288)"]
        for name in [k for k in tensors if k.startswith("group")]:
            checks.append((f"Spearman(group importance, head importance summed within group) — {name}",
                           rowwise_spearman(tensors[name], summed, weights)))
    if "group (rerun, same env as head)" in tensors:
        checks.append(("Reproducibility: Spearman(rerun group, paper group)",
                       rowwise_spearman(tensors["group (rerun, same env as head)"],
                                        tensors["group (paper run, saved)"], weights)))
    for label, (estimate, low, high) in checks:
        rows.append({"level": label, "metric": "mean_spearman", "aligned": estimate,
                     "gap_ci_low": low, "gap_ci_high": high})
        lines.append(f"- {label}: {estimate:.4f} [{low:.4f}, {high:.4f}]")

    with (out_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
