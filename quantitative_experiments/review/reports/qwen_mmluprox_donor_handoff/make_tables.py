"""Build the donor-robustness handoff tables from the saved Qwen3-8B / MMLU-ProX run.

CPU only; reads saved importance tensors, never reruns the model. Reuses the
run's own similarity/bootstrap functions and asserts that the saved analysis
values are reproduced before writing anything.

Run from quantitative_experiments/ with the thesis env:
    ~/.conda/envs/thesis/bin/python review/reports/qwen_mmluprox_donor_handoff/make_tables.py
"""
from __future__ import annotations

import csv
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

QUANT_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(QUANT_ROOT / "src"), str(QUANT_ROOT)]
from review.donor_ablation.analysis import (  # noqa: E402
    PRIMARY_METRIC, bootstrap_weights, ci, normalized_ranks, similarity_matrices, weighted_statistics,
)

RUN = QUANT_ROOT / "review/runs/qwen_mmluprox_parallel_float32"
OUT = Path(__file__).resolve().parent
PERMUTATIONS = 10_000
PERMUTATION_SEED = 20261122
LABELS = {
    "language_mean": "reference_mean", "leave_one_out": "leave_one_out",
    "disjoint_mean_00": "disjoint_1", "disjoint_mean_01": "disjoint_2", "disjoint_mean_02": "disjoint_3",
    "random_in_language_00": "random_1", "random_in_language_01": "random_2", "random_in_language_02": "random_3",
}
METRIC_LABELS = {"spearman": "spearman", "jaccard_at_10": "jaccard_at_10"}


def write(name: str, rows: list[dict]) -> None:
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in row.items()})


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted, running = np.empty(len(pvalues)), 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted.tolist()


def offdiag_mean(matrix: np.ndarray) -> float:
    mask = ~np.eye(len(matrix), dtype=bool)
    return float(matrix[mask].mean())


def main() -> None:
    protocol = json.loads((RUN / "protocol.json").read_text())
    config = protocol["config"]["analysis"]
    languages, n, k = protocol["languages"], len(protocol["recipient_ids"]), config["jaccard_k"]
    weights = np.vstack([np.ones(n), bootstrap_weights(n, config["bootstrap_resamples"], config["bootstrap_seed"])])
    saved = {(r["variant"], r["metric"], r["language_pair"]): r
             for r in csv.DictReader((RUN / "analysis/similarity.csv").open())}
    pairs = list(combinations(range(len(languages)), 2))
    rng = np.random.default_rng(PERMUTATION_SEED)
    # One independent uniform permutation of question order per language per draw (fixed points allowed,
    # matching the saved null). Shared across donors/metrics so tests are paired.
    perms = np.stack([np.stack([rng.permutation(n) for _ in languages]) for _ in range(PERMUTATIONS)])

    tensors = {}
    for variant in protocol["variants"]:
        with np.load(RUN / "variants" / variant["name"] / "importance_tensors.npz") as archive:
            tensors[variant["name"]] = archive[PRIMARY_METRIC].reshape(n, len(languages), -1).astype(np.float64)

    table, per_pair, min_pair, perm_rows = [], [], [], []
    for variant in protocol["variants"]:
        name, values = variant["name"], tensors[variant["name"]]
        within = {}
        for lang in range(len(languages)):
            for metric, matrix in similarity_matrices(values[:, lang], values[:, lang], k).items():
                within[(metric, lang)] = offdiag_mean(matrix)
        by_metric: dict[str, list] = {}
        for i, j in pairs:
            pair_label = f"{languages[i]}--{languages[j]}"
            for metric, matrix in similarity_matrices(values[:, i], values[:, j], k).items():
                stats = weighted_statistics(matrix, weights)
                low, high = ci(stats[1:, 2])
                ref = saved[(name, metric, pair_label)]
                assert np.isclose(stats[0, 2], float(ref["gap"]), atol=1e-9), (name, metric, pair_label)
                assert np.isclose(low, float(ref["gap_ci_low"]), atol=1e-9)
                aligned_perm = matrix[perms[:, i], perms[:, j]].mean(axis=1)  # (B,)
                p_pair = (1 + np.sum(aligned_perm >= stats[0, 0])) / (PERMUTATIONS + 1)
                row = {"donor": LABELS[name], "metric": METRIC_LABELS[metric], "language_pair": pair_label,
                       "aligned_mean": stats[0, 0], "cross_lang_control_mean": stats[0, 1],
                       "cross_lang_control_offdiag_mean": offdiag_mean(matrix),
                       "within_lang_control_mean": (within[(metric, i)] + within[(metric, j)]) / 2,
                       "gap": stats[0, 2], "gap_ci_low": low, "gap_ci_high": high, "perm_p_one_sided": p_pair}
                per_pair.append(row)
                by_metric.setdefault(metric, []).append((stats, row, aligned_perm))
        for metric, items in by_metric.items():
            aggregate = np.mean([s for s, _, _ in items], axis=0)
            low, high = ci(aggregate[1:, 2])
            ref = saved[(name, metric, "ALL")]
            assert np.isclose(aggregate[0, 2], float(ref["gap"]), atol=1e-9) and np.isclose(high, float(ref["gap_ci_high"]), atol=1e-9)
            perm_stat = np.mean([a for _, _, a in items], axis=0)
            p_all = (1 + np.sum(perm_stat >= aggregate[0, 0])) / (PERMUTATIONS + 1)
            rows = [r for _, r, _ in items]
            table.append({"donor": LABELS[name], "metric": METRIC_LABELS[metric],
                          "aligned_mean": aggregate[0, 0], "cross_lang_control_mean": aggregate[0, 1],
                          "within_lang_control_mean": float(np.mean([r["within_lang_control_mean"] for r in rows])),
                          "gap": aggregate[0, 2], "gap_ci_low": low, "gap_ci_high": high,
                          "cross_lang_control_offdiag_mean": float(np.mean([r["cross_lang_control_offdiag_mean"] for r in rows])),
                          "gap_vs_offdiag_control": aggregate[0, 0] - float(np.mean([r["cross_lang_control_offdiag_mean"] for r in rows])),
                          "gap_minus_reference": float(ref["gap_minus_reference"] or 0.0),
                          "gap_minus_reference_ci_low": float(ref["difference_ci_low"] or 0.0),
                          "gap_minus_reference_ci_high": float(ref["difference_ci_high"] or 0.0),
                          "perm_p_one_sided": p_all})
            worst = min(rows, key=lambda r: r["gap"])
            min_pair.append({"donor": LABELS[name], "metric": METRIC_LABELS[metric], "min_pair_gap": worst["gap"],
                             "min_pair": worst["language_pair"], "min_pair_gap_ci_low": worst["gap_ci_low"],
                             "min_pair_gap_ci_high": worst["gap_ci_high"],
                             "max_pair_gap": max(r["gap"] for r in rows),
                             "n_pairs_ci_excludes_zero": sum(r["gap_ci_low"] > 0 for r in rows)})

    for row, adjusted in zip(table, holm([r["perm_p_one_sided"] for r in table])):
        row["perm_p_holm_16"] = adjusted
    for row, adjusted in zip(per_pair, holm([r["perm_p_one_sided"] for r in per_pair])):
        row["perm_p_holm_160"] = adjusted
    write("1_results_table.csv", table)
    write("2_per_pair_gaps.csv", per_pair)
    write("2b_min_pair_gap.csv", min_pair)

    # 4. Intervention magnitude.
    magnitude = []
    for variant in protocol["variants"]:
        values = tensors[variant["name"]]  # (question, language, layer*group)
        for index, label in enumerate([*languages, "ALL"]):
            block = values[:, index] if index < len(languages) else values.reshape(-1, values.shape[-1])
            magnitude.append({"donor": LABELS[variant["name"]], "language": label,
                              "mean_abs_importance": float(np.abs(block).mean()),
                              "median_abs_importance": float(np.median(np.abs(block))),
                              "mean_signed_importance": float(block.mean()),
                              "mean_within_signature_sd": float(block.std(axis=-1).mean()),
                              "p99_abs_importance": float(np.quantile(np.abs(block), 0.99))})
    write("4_importance_magnitude.csv", magnitude)

    # 5. Test-retest between donor repetitions for the same question-language pair.
    def retest(names: list[str], kind: str) -> list[dict]:
        rows = []
        for a, b in combinations(names, 2):
            x = tensors[a].reshape(n * len(languages), -1)
            y = tensors[b].reshape(n * len(languages), -1)
            rho = np.clip(np.sum(normalized_ranks(x) * normalized_ranks(y), axis=1), -1, 1).reshape(n, len(languages))
            jac = np.array([similarity_matrices(x[r:r + 1], y[r:r + 1], k)[f"jaccard_at_{k}"][0, 0]
                            for r in range(len(x))]).reshape(n, len(languages))
            for metric, values in (("spearman", rho), ("jaccard_at_10", jac)):
                for index, label in enumerate([*languages, "ALL"]):
                    per_question = values[:, index] if index < len(languages) else values.mean(axis=1)
                    estimates = weights @ per_question / n
                    low, high = ci(estimates[1:])
                    rows.append({"donor_kind": kind, "repeat_a": LABELS[a], "repeat_b": LABELS[b], "metric": metric,
                                 "language": label, "mean_same_question_language": estimates[0],
                                 "ci_low": low, "ci_high": high})
        return rows
    names = [v["name"] for v in protocol["variants"]]
    write("5_test_retest_across_repeats.csv",
          retest([x for x in names if x.startswith("random")], "random") +
          retest([x for x in names if x.startswith("disjoint")], "disjoint (comparison)"))

    # 6. Agreement with the reference signature (Spearman saved by the run; Jaccard@10 added here).
    stability = list(csv.DictReader((RUN / "analysis/signature_stability.csv").open()))
    ref = tensors["language_mean"].reshape(n * len(languages), -1)
    rows = []
    for row in stability:
        name = row["variant"]
        x = tensors[name].reshape(n * len(languages), -1)
        jac = np.array([similarity_matrices(x[r:r + 1], ref[r:r + 1], k)[f"jaccard_at_{k}"][0, 0]
                        for r in range(len(x))]).reshape(n, len(languages))
        lang = row["language"]
        per_question = jac[:, languages.index(lang)] if lang in languages else jac.mean(axis=1)
        estimates = weights @ per_question / n
        low, high = ci(estimates[1:])
        rows.append({"donor": LABELS[name], "language": lang,
                     "spearman_vs_reference": float(row["spearman_vs_reference"]),
                     "spearman_ci_low": float(row["ci_low"]), "spearman_ci_high": float(row["ci_high"]),
                     "jaccard10_vs_reference": estimates[0], "jaccard10_ci_low": low, "jaccard10_ci_high": high})
    write("6_agreement_with_reference.csv", rows)
    print("Wrote tables to", OUT)


if __name__ == "__main__":
    main()
