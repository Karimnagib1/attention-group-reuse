#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from tqdm.auto import tqdm


EPS = 1e-8


def _rankdata_1d_tie_aware(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    n = x.shape[0]
    i = 0
    while i < n:
        j = i + 1
        while j < n and x[order[j]] == x[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def spearman_corr(a: np.ndarray, b: np.ndarray, *, degenerate: str = "skip") -> Optional[float]:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size:
        raise ValueError("Spearman inputs must have same length.")

    ra = _rankdata_1d_tie_aware(a)
    rb = _rankdata_1d_tie_aware(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.linalg.norm(ra) * np.linalg.norm(rb)

    if denom < EPS:
        if degenerate == "skip":
            return None
        if degenerate == "zero":
            return 0.0
        if degenerate == "one_if_equal_else_zero":
            return 1.0 if np.array_equal(a, b) else 0.0
        raise ValueError(f"Unknown degenerate handling: {degenerate}")

    return float(np.dot(ra, rb) / denom)


def mean_spearman_over_pairs(vecs_a: np.ndarray, vecs_b: np.ndarray, *, degenerate: str) -> Tuple[float, int]:
    if vecs_a.shape != vecs_b.shape:
        raise ValueError(f"Shape mismatch: {vecs_a.shape} vs {vecs_b.shape}")
    vals: List[float] = []
    for i in range(vecs_a.shape[0]):
        c = spearman_corr(vecs_a[i], vecs_b[i], degenerate=degenerate)
        if c is None:
            continue
        vals.append(float(c))
    if not vals:
        return 0.0, 0
    return float(np.mean(vals)), len(vals)


def load_config(path: Path) -> Dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping.")
    return cfg


def load_sweep_inputs(input_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any], np.lib.npyio.NpzFile]:
    samples = json.loads((input_dir / "samples.json").read_text(encoding="utf-8"))
    metadata = json.loads((input_dir / "metadata.json").read_text(encoding="utf-8"))
    npz = np.load(input_dir / "importance_tensors.npz")
    return samples, metadata, npz


def _infer_layer_dims_from_metadata(metadata: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    n_layers = metadata.get("n_layers")
    n_patch_units = metadata.get("n_patch_units")
    return (int(n_layers), int(n_patch_units)) if n_layers and n_patch_units else (None, None)


def load_metric_layerwise(npz: np.lib.npyio.NpzFile, metadata: Dict[str, Any], metric: str) -> np.ndarray:
    flat_key = f"flat_{metric}"
    if metric in npz:
        arr = np.asarray(npz[metric])
        if arr.ndim == 3:
            return arr
        if arr.ndim == 2:
            n_layers, n_patch_units = _infer_layer_dims_from_metadata(metadata)
            if not n_layers or not n_patch_units:
                raise ValueError(f"Metric '{metric}' is flat, but n_layers/n_patch_units missing in metadata.")
            if arr.shape[1] != n_layers * n_patch_units:
                raise ValueError(f"Cannot reshape metric '{metric}' with shape {arr.shape} into ({n_layers}, {n_patch_units}).")
            return arr.reshape(arr.shape[0], n_layers, n_patch_units)
        raise ValueError(f"Unsupported ndim for metric '{metric}': {arr.ndim}")

    if flat_key in npz:
        arr = np.asarray(npz[flat_key])
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D flat tensor for key '{flat_key}', got shape {arr.shape}")
        n_layers, n_patch_units = _infer_layer_dims_from_metadata(metadata)
        if not n_layers or not n_patch_units:
            raise ValueError(f"Key '{flat_key}' found, but n_layers/n_patch_units missing in metadata.")
        if arr.shape[1] != n_layers * n_patch_units:
            raise ValueError(f"Cannot reshape key '{flat_key}' with shape {arr.shape} into ({n_layers}, {n_patch_units}).")
        return arr.reshape(arr.shape[0], n_layers, n_patch_units)

    raise ValueError(f"Could not find metric tensor for '{metric}' (checked '{metric}' and '{flat_key}').")


def build_sample_index(samples: List[Dict[str, Any]]) -> Dict[Tuple[str, str], int]:
    idx: Dict[Tuple[str, str], int] = {}
    for s in samples:
        pid = str(s["parallel_id"])
        lang = str(s["language"])
        idx[(pid, lang)] = int(s["sample_index"])
    return idx


def aligned_rows_for_language_pair(
    samples: List[Dict[str, Any]],
    lang_a: str,
    lang_b: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    idx = build_sample_index(samples)
    all_pids = sorted({str(s["parallel_id"]) for s in samples})

    rows_a: List[int] = []
    rows_b: List[int] = []
    common_pids: List[str] = []
    for pid in all_pids:
        ia = idx.get((pid, lang_a))
        ib = idx.get((pid, lang_b))
        if ia is None or ib is None:
            continue
        rows_a.append(ia)
        rows_b.append(ib)
        common_pids.append(pid)

    if not rows_a:
        raise ValueError(f"No shared parallel_id samples between languages '{lang_a}' and '{lang_b}'.")

    return np.array(rows_a, dtype=np.int64), np.array(rows_b, dtype=np.int64), common_pids


def aligned_layerwise_spearman(A: np.ndarray, B: np.ndarray, *, degenerate: str) -> Tuple[np.ndarray, np.ndarray]:
    if A.shape != B.shape:
        raise ValueError(f"Aligned tensor mismatch: {A.shape} vs {B.shape}")
    _, n_layers, _ = A.shape
    layer_corr = np.zeros(n_layers, dtype=np.float64)
    layer_counts = np.zeros(n_layers, dtype=np.int32)
    for l in range(n_layers):
        c, n = mean_spearman_over_pairs(A[:, l, :], B[:, l, :], degenerate=degenerate)
        layer_corr[l] = c
        layer_counts[l] = n
    return layer_corr, layer_counts


def _shuffle_worker(args: Tuple[np.ndarray, np.ndarray, int, int, str]) -> Tuple[np.ndarray, np.ndarray]:
    A, B, n_local, seed, degenerate = args
    rng = np.random.default_rng(int(seed))
    n_q, n_layers, _ = A.shape
    dists = np.zeros((n_local, n_layers), dtype=np.float64)
    used = np.zeros((n_local, n_layers), dtype=np.int32)
    for t in range(n_local):
        perm = rng.permutation(n_q)
        Bp = B[perm]
        for l in range(n_layers):
            c, n = mean_spearman_over_pairs(A[:, l, :], Bp[:, l, :], degenerate=degenerate)
            dists[t, l] = c
            used[t, l] = n
    return dists, used


def shuffled_layerwise_null(
    A: np.ndarray,
    B: np.ndarray,
    *,
    n_shuffles: int,
    seed: int,
    degenerate: str,
    n_workers: int,
    show_progress: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    if n_shuffles <= 0:
        raise ValueError("n_shuffles must be > 0")
    n_workers = max(1, int(n_workers))
    chunks = [n_shuffles // n_workers] * n_workers
    for i in range(n_shuffles % n_workers):
        chunks[i] += 1
    chunks = [c for c in chunks if c > 0]

    args: List[Tuple[np.ndarray, np.ndarray, int, int, str]] = []
    for w, n_local in enumerate(chunks):
        args.append((A, B, n_local, int(seed) + 10007 * w, degenerate))

    d_all: List[np.ndarray] = []
    c_all: List[np.ndarray] = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_shuffle_worker, a) for a in args]
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Layer-wise shuffles",
            leave=False,
            disable=not show_progress,
        ):
            d, c = fut.result()
            d_all.append(d)
            c_all.append(c)
    return np.concatenate(d_all, axis=0), np.concatenate(c_all, axis=0)


def p_and_z(aligned: np.ndarray, null_dists: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_shuffles = null_dists.shape[0]
    mean_null = null_dists.mean(axis=0)
    std_null = null_dists.std(axis=0, ddof=1) if n_shuffles > 1 else np.zeros_like(mean_null)

    ge = (null_dists >= aligned[None, :]).sum(axis=0)
    p = (ge + 1.0) / (n_shuffles + 1.0)

    z = np.zeros_like(mean_null)
    mask = std_null > 0
    z[mask] = (aligned[mask] - mean_null[mask]) / std_null[mask]
    return p, z


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Layer-wise Spearman baseline for language pairs.")
    p.add_argument("--config", required=True, help="Path to YAML config.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))

    input_dir = Path(cfg["io"]["input_dir"])
    output_dir = Path(cfg["io"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_cfg = cfg.get("baseline", {}) or {}
    metrics = list(
        baseline_cfg.get(
            "metrics",
            ["pct_var_mean_incorrect", "pct_var_max_incorrect", "rank_drop_correct"],
        )
    )
    samples, metadata, npz = load_sweep_inputs(input_dir)

    languages_in_samples = sorted({str(s["language"]) for s in samples})
    requested_languages = baseline_cfg.get("languages")
    if requested_languages is None:
        # Backward compatibility with previous single-pair config.
        language_pair = baseline_cfg.get("language_pair")
        if not isinstance(language_pair, list) or len(language_pair) != 2:
            raise ValueError(
                "Set either baseline.languages (>=2 languages) or baseline.language_pair (exactly 2 languages)."
            )
        requested_languages = [str(language_pair[0]), str(language_pair[1])]
    else:
        if not isinstance(requested_languages, list) or len(requested_languages) < 2:
            raise ValueError("baseline.languages must be a list with at least 2 language codes.")
        requested_languages = [str(x) for x in requested_languages]

    missing = [x for x in requested_languages if x not in languages_in_samples]
    if missing:
        raise ValueError(f"Languages not present in samples: {missing}. Available: {languages_in_samples}")
    selected_languages = list(dict.fromkeys(requested_languages))
    if len(selected_languages) < 2:
        raise ValueError("Need at least 2 unique languages.")
    language_pairs = list(itertools.combinations(selected_languages, 2))

    degenerate = str(baseline_cfg.get("degenerate_spearman", "skip"))
    n_shuffles = int(baseline_cfg.get("n_shuffles", 500))
    seed = int(baseline_cfg.get("seed", 123))
    n_workers = int(baseline_cfg.get("n_workers", 8))
    save_distributions = bool(baseline_cfg.get("save_distributions_npz", True))

    report: Dict[str, Any] = {
        "input_dir": str(input_dir),
        "metrics": metrics,
        "languages": selected_languages,
        "language_pairs": [[a, b] for (a, b) in language_pairs],
        "n_language_pairs": len(language_pairs),
        "n_layers": int(metadata.get("n_layers")) if metadata.get("n_layers") is not None else None,
        "n_patch_units": int(metadata.get("n_patch_units")) if metadata.get("n_patch_units") is not None else None,
        "degenerate_spearman": degenerate,
        "n_shuffles": n_shuffles,
        "seed": seed,
        "results": {},
    }

    dist_store: Dict[str, np.ndarray] = {}

    for metric in tqdm(metrics, desc="Metrics"):
        tensor = load_metric_layerwise(npz, metadata, metric)  # [N, L, G]
        per_pair: Dict[str, Any] = {}
        for pair_i, (lang_a, lang_b) in enumerate(
            tqdm(language_pairs, desc=f"Pairs [{metric}]", leave=False)
        ):
            rows_a, rows_b, common_pids = aligned_rows_for_language_pair(samples, lang_a=lang_a, lang_b=lang_b)
            A = tensor[rows_a]
            B = tensor[rows_b]

            aligned, layer_counts = aligned_layerwise_spearman(A, B, degenerate=degenerate)
            null_dists, null_counts = shuffled_layerwise_null(
                A,
                B,
                n_shuffles=n_shuffles,
                seed=seed + 100000 * pair_i,
                degenerate=degenerate,
                n_workers=n_workers,
                show_progress=True,
            )
            null_mean = null_dists.mean(axis=0)
            null_std = null_dists.std(axis=0, ddof=1) if null_dists.shape[0] > 1 else np.zeros_like(null_mean)
            p_val, z_score = p_and_z(aligned, null_dists)

            pair_key = f"{lang_a}__{lang_b}"
            per_pair[pair_key] = {
                "language_pair": [lang_a, lang_b],
                "n_common_parallel_ids": len(common_pids),
                "aligned": {
                    "layer_mean_spearman": aligned.tolist(),
                    "layer_pair_counts": layer_counts.tolist(),
                },
                "null_cross_language_shuffle": {
                    "n_shuffles": int(null_dists.shape[0]),
                    "null_mean": null_mean.tolist(),
                    "null_std": null_std.tolist(),
                    "null_pair_counts_mean": null_counts.mean(axis=0).tolist(),
                    "p_value_greater": p_val.tolist(),
                    "z_score": z_score.tolist(),
                },
            }

            if save_distributions:
                dist_store[f"{metric}__{pair_key}__layerwise__null_cross_language_shuffle"] = null_dists.astype(
                    np.float32
                )

        report["results"][metric] = {"pairs": per_pair}

    (output_dir / "baseline_layerwise_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if save_distributions and dist_store:
        np.savez_compressed(output_dir / "layerwise_distributions.npz", **dist_store)

    print(f"Wrote layer-wise baseline report to: {output_dir / 'baseline_layerwise_report.json'}")
    if save_distributions and dist_store:
        print(f"Wrote layer-wise distributions to: {output_dir / 'layerwise_distributions.npz'}")


if __name__ == "__main__":
    main()
