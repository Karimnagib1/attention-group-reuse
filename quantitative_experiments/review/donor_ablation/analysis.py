from __future__ import annotations

import csv
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from .engine import locked
from .protocol import load_protocol
from quantitative_experiments.multilingual_head_patch.checkpoint import atomic_write_json
from quantitative_experiments.multilingual_head_patch.run import _rankdata_1d, _topk_index_set

PRIMARY_METRIC = "pct_var_mean_incorrect"


def normalized_ranks(values: np.ndarray) -> np.ndarray:
    ranks = np.stack([_rankdata_1d(row) for row in values])
    ranks -= ranks.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(ranks, axis=1, keepdims=True)
    return np.divide(ranks, norms, out=np.full_like(ranks, np.nan), where=norms > 1e-8)


def similarity_matrices(a: np.ndarray, b: np.ndarray, k: int) -> dict:
    if a.shape != b.shape or a.ndim != 2 or not 1 <= k <= a.shape[1]:
        raise ValueError("Aligned signature matrices or top-k are invalid.")
    ra, rb = normalized_ranks(a), normalized_ranks(b)
    masks = []
    for values in (a, b):
        mask = np.zeros_like(values, dtype=np.float64)
        for row, value in enumerate(values):
            mask[row, list(_topk_index_set(value, k))] = 1
        masks.append(mask)
    intersection = masks[0] @ masks[1].T
    return {"spearman": np.clip(ra @ rb.T, -1, 1), f"jaccard_at_{k}": intersection / (2 * k - intersection)}


def bootstrap_weights(n: int, repeats: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).multinomial(n, np.full(n, 1.0 / n), size=repeats).astype(np.float64)


def weighted_statistics(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Resample question IDs jointly on both axes; columns: aligned, null, gap."""
    valid = np.isfinite(matrix)
    clean = np.where(valid, matrix, 0.0)
    diag, diag_valid = np.diag(clean), np.diag(valid).astype(np.float64)
    results = []
    for start in range(0, len(weights), 512):
        w = weights[start:start + 512]
        aligned_den = w @ diag_valid
        null_den = np.sum((w @ valid.astype(np.float64)) * w, axis=1)
        aligned = np.divide(w @ diag, aligned_den, out=np.full(len(w), np.nan), where=aligned_den > 0)
        null = np.divide(np.sum((w @ clean) * w, axis=1), null_den,
                         out=np.full(len(w), np.nan), where=null_den > 0)
        results.append(np.column_stack([aligned, null, aligned - null]))
    return np.concatenate(results)


def finite_mean(values: np.ndarray, axis=0) -> np.ndarray:
    count = np.isfinite(values).sum(axis=axis)
    return np.divide(np.nansum(values, axis=axis), count,
                     out=np.full(np.shape(count), np.nan), where=count > 0)


def ci(values: np.ndarray) -> tuple[float, float]:
    good = values[np.isfinite(values)]
    return tuple(np.quantile(good, [0.025, 0.975])) if len(good) else (float("nan"), float("nan"))


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_csv(path: Path, rows: list[dict]):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(json_safe(rows))
    temporary.replace(path)


def analyze(output: Path, bootstrap_resamples: int | None = None) -> dict:
    with locked(output):
        return _analyze(output, bootstrap_resamples)


def _analyze(output: Path, bootstrap_resamples: int | None) -> dict:
    protocol = load_protocol(output / "protocol.json")
    manifest = json.loads((output / "run_manifest.json").read_text())
    if manifest["protocol_sha256"] != protocol["sha256"]:
        raise ValueError("Manifest and protocol disagree.")
    if json.loads((output / "_SUCCESS").read_text())["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Run is incomplete or has inconsistent artifacts.")
    config = protocol["config"]["analysis"]
    repeats = bootstrap_resamples if bootstrap_resamples is not None else config["bootstrap_resamples"]
    if repeats < 1:
        raise ValueError("bootstrap_resamples must be positive.")
    languages = protocol["languages"]
    n = len(protocol["recipient_ids"])
    if n < 2:
        raise ValueError("Similarity analysis requires at least two parallel questions.")
    weights = bootstrap_weights(n, repeats, config["bootstrap_seed"])
    weights = np.vstack([np.ones(n), weights])  # row zero is the observed sample
    all_rows, stability_rows, stability_samples = [], [], []
    reference, reference_stats = None, {}
    for variant in protocol["variants"]:
        directory = output / "variants" / variant["name"]
        meta = json.loads((directory / "metadata.json").read_text())
        if meta["fingerprint"] != manifest["fingerprint"] or meta["variant"] != variant:
            raise ValueError(f"Variant metadata mismatch: {directory}")
        sample_meta = json.loads((directory / "samples.json").read_text())
        expected = [(pid, lang) for pid in protocol["recipient_ids"] for lang in languages]
        if [(s["parallel_id"], s["language"]) for s in sample_meta] != expected:
            raise ValueError("Sample order differs from the saved protocol.")
        with np.load(directory / "importance_tensors.npz", allow_pickle=False) as archive:
            tensor = archive[PRIMARY_METRIC]
        shape = (n * len(languages), manifest["n_layers"], manifest["n_groups"])
        if tensor.shape != shape or not np.isfinite(tensor).all():
            raise ValueError(f"Invalid importance tensor: {directory}")
        values = tensor.reshape(n, len(languages), -1)
        pair_stats = {}
        for i, j in combinations(range(len(languages)), 2):
            for metric, matrix in similarity_matrices(values[:, i], values[:, j], config["jaccard_k"]).items():
                stats = weighted_statistics(matrix, weights)
                pair_stats.setdefault(metric, []).append(stats)
                low, high = ci(stats[1:, 2])
                all_rows.append({"variant": variant["name"], "kind": variant["kind"], "repeat": variant["repeat"],
                                 "metric": metric, "language_pair": f"{languages[i]}--{languages[j]}",
                                 "aligned": stats[0, 0], "shuffled": stats[0, 1], "gap": stats[0, 2],
                                 "gap_ci_low": low, "gap_ci_high": high,
                                 "valid_aligned": int(np.isfinite(np.diag(matrix)).sum()),
                                 "valid_shuffled_pairs": int(np.isfinite(matrix).sum())})
        for metric, stats in pair_stats.items():
            aggregate = finite_mean(np.stack(stats), axis=0)
            if variant["kind"] == "language_mean":
                reference_stats[metric] = aggregate
            delta = aggregate[:, 2] - reference_stats[metric][:, 2]
            low, high = ci(aggregate[1:, 2])
            delta_low, delta_high = ci(delta[1:])
            all_rows.append({"variant": variant["name"], "kind": variant["kind"], "repeat": variant["repeat"],
                             "metric": metric, "language_pair": "ALL", "aligned": aggregate[0, 0],
                             "shuffled": aggregate[0, 1], "gap": aggregate[0, 2],
                             "gap_ci_low": low, "gap_ci_high": high, "gap_minus_reference": delta[0],
                             "difference_ci_low": delta_low, "difference_ci_high": delta_high})
        if reference is None:
            reference = values.copy()
        rho = np.sum(normalized_ranks(values.reshape(n * len(languages), -1)) *
                     normalized_ranks(reference.reshape(n * len(languages), -1)), axis=1).reshape(n, len(languages))
        rho = np.clip(rho, -1, 1)
        for q, pid in enumerate(protocol["recipient_ids"]):
            for lang_index, language in enumerate(languages):
                stability_samples.append({"variant": variant["name"], "parallel_id": pid, "language": language,
                                          "spearman_vs_reference": rho[q, lang_index]})
        for index, label in enumerate([*languages, "ALL"]):
            per_question = rho[:, index] if index < len(languages) else finite_mean(rho, axis=1)
            valid = np.isfinite(per_question)
            denominator = weights @ valid.astype(float)
            estimates = np.divide(weights @ np.nan_to_num(per_question), denominator,
                                  out=np.full(len(weights), np.nan), where=denominator > 0)
            low, high = ci(estimates[1:])
            stability_rows.append({"variant": variant["name"], "language": label,
                                   "spearman_vs_reference": estimates[0], "ci_low": low, "ci_high": high,
                                   "valid_questions": int(valid.sum())})
        print(f"Analyzed {variant['name']}", flush=True)
    summary = json_safe({"protocol_sha256": protocol["sha256"], "fingerprint": manifest["fingerprint"],
                         "importance_metric": PRIMARY_METRIC, "questions_per_language": n,
                         "bootstrap_resamples": repeats, "bootstrap_seed": config["bootstrap_seed"],
                         "null": "Uniform shuffled question pairing including fixed points; exact all-pairs expectation.",
                         "uncertainty": "95% percentile CI; question IDs resampled jointly across languages and donors. Conditional on fixed donor pools/assignments.",
                         "degenerate_spearman": "Skip constant signatures; undefined results are null. Counts are in the CSV.",
                         "aggregation": "Equal mean across language pairs; each donor repetition analyzed separately.",
                         "similarity": [row for row in all_rows if row["language_pair"] == "ALL"],
                         "signature_stability": [row for row in stability_rows if row["language"] == "ALL"]})
    directory = output / "analysis"
    directory.mkdir(exist_ok=True)
    write_csv(directory / "similarity.csv", all_rows)
    write_csv(directory / "signature_stability.csv", stability_rows)
    write_csv(directory / "signature_stability_per_question.csv", stability_samples)
    atomic_write_json(directory / "summary.json", summary)
    return summary
