# #!/usr/bin/env python3
# from __future__ import annotations

# import argparse
# import json
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Any, Dict, List, Tuple, Optional

# import numpy as np
# import yaml
# from tqdm.auto import tqdm
# from concurrent.futures import ProcessPoolExecutor, as_completed
# import os


# EPS = 1e-8


# # -----------------------------
# # Utilities
# # -----------------------------

# def _rankdata_1d_tie_aware(x: np.ndarray) -> np.ndarray:
#     """
#     Assign average ranks (1-based) to a 1D array with ties.
#     Same approach as your main file, but kept local here.
#     """
#     x = np.asarray(x)
#     order = np.argsort(x, kind="mergesort")
#     ranks = np.empty_like(order, dtype=np.float64)
#     n = x.shape[0]
#     i = 0
#     while i < n:
#         j = i + 1
#         while j < n and x[order[j]] == x[order[i]]:
#             j += 1
#         avg_rank = 0.5 * (i + j - 1) + 1.0
#         ranks[order[i:j]] = avg_rank
#         i = j
#     return ranks


# def spearman_corr(
#     a: np.ndarray,
#     b: np.ndarray,
#     *,
#     degenerate: str = "skip",
# ) -> Optional[float]:
#     """
#     Tie-aware Spearman correlation of two 1D arrays.

#     degenerate:
#       - "skip": return None if denom too small (zero variance)
#       - "zero": return 0.0 if denom too small
#       - "one_if_equal_else_zero": if both arrays identical -> 1.0 else 0.0
#     """
#     a = np.asarray(a, dtype=np.float64).ravel()
#     b = np.asarray(b, dtype=np.float64).ravel()
#     if a.size != b.size:
#         raise ValueError("Spearman inputs must have same length.")

#     ra = _rankdata_1d_tie_aware(a)
#     rb = _rankdata_1d_tie_aware(b)
#     ra -= ra.mean()
#     rb -= rb.mean()
#     denom = (np.linalg.norm(ra) * np.linalg.norm(rb))

#     if denom < EPS:
#         if degenerate == "skip":
#             return None
#         if degenerate == "zero":
#             return 0.0
#         if degenerate == "one_if_equal_else_zero":
#             return 1.0 if np.array_equal(a, b) else 0.0
#         raise ValueError(f"Unknown degenerate handling: {degenerate}")

#     return float(np.dot(ra, rb) / denom)


# def mean_spearman_over_pairs(
#     vecs_a: np.ndarray,
#     vecs_b: np.ndarray,
#     *,
#     degenerate: str,
# ) -> Tuple[float, int]:
#     """
#     Compute mean Spearman over matched rows: vecs_a[i] with vecs_b[i].
#     Returns (mean_corr, n_used) where n_used excludes None if degenerate="skip".
#     """
#     assert vecs_a.shape == vecs_b.shape
#     vals: List[float] = []
#     for i in range(vecs_a.shape[0]):
#         c = spearman_corr(vecs_a[i], vecs_b[i], degenerate=degenerate)
#         if c is None:
#             continue
#         vals.append(float(c))
#     if not vals:
#         return 0.0, 0
#     return float(np.mean(vals)), len(vals)


# def _stable_rng(seed: int) -> np.random.Generator:
#     return np.random.default_rng(int(seed))


# # -----------------------------
# # Data loading
# # -----------------------------

# @dataclass
# class LoadedSweep:
#     samples: List[Dict[str, Any]]
#     metadata: Dict[str, Any]
#     vectors_by_metric: Dict[str, np.ndarray]  # metric -> [n_samples, D]
#     languages: List[str]


# def load_sweep_outputs(input_dir: Path, metrics: List[str]) -> LoadedSweep:
#     samples = json.loads((input_dir / "samples.json").read_text(encoding="utf-8"))
#     metadata = json.loads((input_dir / "metadata.json").read_text(encoding="utf-8"))

#     npz = np.load(input_dir / "importance_tensors.npz")
#     # Prefer flattened vectors if present (they are in your file)
#     metric_map: Dict[str, str] = {
#         "pct_var_mean_incorrect": "flat_pct_var_mean_incorrect",
#         "pct_var_max_incorrect": "flat_pct_var_max_incorrect",
#         "rank_drop_correct": "flat_rank_drop_correct",
#     }

#     vectors_by_metric: Dict[str, np.ndarray] = {}
#     for m in metrics:
#         if m not in metric_map:
#             raise ValueError(f"Unknown metric '{m}'. Choose from {list(metric_map.keys())}.")
#         key = metric_map[m]
#         if key not in npz:
#             raise ValueError(f"Missing '{key}' in importance_tensors.npz")
#         vecs = np.asarray(npz[key])
#         vectors_by_metric[m] = vecs

#     languages = list(metadata.get("languages", []))
#     if not languages:
#         # fallback: infer from samples
#         languages = sorted({str(s["language"]) for s in samples})

#     n = len(samples)
#     for m, v in vectors_by_metric.items():
#         if v.shape[0] != n:
#             raise ValueError(f"Vector rows mismatch for metric {m}: {v.shape[0]} vs samples {n}.")

#     return LoadedSweep(samples=samples, metadata=metadata, vectors_by_metric=vectors_by_metric, languages=languages)


# # -----------------------------
# # Index building
# # -----------------------------

# def build_index(samples: List[Dict[str, Any]]) -> Dict[Tuple[str, str], int]:
#     """
#     Map (parallel_id, language) -> sample_index
#     Assumes final samples contain one per (parallel_id, language).
#     """
#     idx: Dict[Tuple[str, str], int] = {}
#     for s in samples:
#         pid = str(s["parallel_id"])
#         lang = str(s["language"])
#         i = int(s["sample_index"])
#         idx[(pid, lang)] = i
#     return idx


# def parallel_ids_present(samples: List[Dict[str, Any]]) -> List[str]:
#     return sorted({str(s["parallel_id"]) for s in samples})


# # -----------------------------
# # Baseline computations
# # -----------------------------

# def aligned_mean_spearman(
#     vecs: np.ndarray,
#     languages: List[str],
#     pids: List[str],
#     index_by_pair: Dict[Tuple[str, str], int],
#     degenerate: str,
#     show_progress: bool = False,
# ) -> Dict[str, Any]:
#     """
#     Compute aligned mean Spearman for each language pair (li, lj)
#     using the same parallel_id mapping.
#     """
#     L = len(languages)
#     mat = np.zeros((L, L), dtype=np.float64)
#     counts = np.zeros((L, L), dtype=np.int32)

#     for i, li in enumerate(
#         tqdm(languages, desc="Aligned language pairs", leave=False, disable=not show_progress)
#     ):
#         for j, lj in enumerate(languages):
#             # Build paired vectors over pids
#             rows_a = []
#             rows_b = []
#             for pid in pids:
#                 ia = index_by_pair.get((pid, li))
#                 ib = index_by_pair.get((pid, lj))
#                 if ia is None or ib is None:
#                     continue
#                 rows_a.append(ia)
#                 rows_b.append(ib)

#             if not rows_a:
#                 mat[i, j] = 0.0
#                 counts[i, j] = 0
#                 continue

#             A = vecs[np.array(rows_a, dtype=np.int64)]
#             B = vecs[np.array(rows_b, dtype=np.int64)]
#             mean_corr, n_used = mean_spearman_over_pairs(A, B, degenerate=degenerate)
#             mat[i, j] = mean_corr
#             counts[i, j] = n_used

#     return {
#         "languages": list(languages),
#         "mean_spearman": mat.tolist(),
#         "pair_counts": counts.tolist(),
#     }


# def shuffled_mapping_null_parallel(
#     vecs: np.ndarray,
#     languages: List[str],
#     pids: List[str],
#     index_by_pair: Dict[Tuple[str, str], int],
#     n_shuffles: int,
#     seed: int,
#     degenerate: str,
#     *,
#     pair_mode: str = "cross_language",
#     n_workers: int = 8,
# ) -> Dict[str, Any]:
#     L = len(languages)

#     # Split shuffles across workers
#     n_workers = max(1, int(n_workers))
#     chunks = [n_shuffles // n_workers] * n_workers
#     for i in range(n_shuffles % n_workers):
#         chunks[i] += 1
#     chunks = [c for c in chunks if c > 0]

#     # Make distinct seeds per worker
#     worker_args = []
#     base_seed = int(seed)
#     for w, n_local in enumerate(chunks):
#         worker_args.append((vecs, languages, pids, index_by_pair, n_local, base_seed + 10007 * w, degenerate, pair_mode))

#     dists_list = []
#     counts_list = []

#     with ProcessPoolExecutor(max_workers=n_workers) as ex:
#         futs = [ex.submit(_shuffle_worker, a) for a in worker_args]
#         for fut in tqdm(as_completed(futs), total=len(futs), desc=f"Shuffles [{pair_mode}] (parallel)"):
#             d, c = fut.result()
#             dists_list.append(d)
#             counts_list.append(c)

#     dists = np.concatenate(dists_list, axis=0) if dists_list else np.zeros((0, L, L))
#     used_counts = np.concatenate(counts_list, axis=0) if counts_list else np.zeros((0, L, L), dtype=np.int32)

#     mean_mat = dists.mean(axis=0) if dists.shape[0] else np.zeros((L, L))
#     std_mat = dists.std(axis=0, ddof=1) if dists.shape[0] > 1 else np.zeros((L, L))

#     return {
#         "languages": list(languages),
#         "n_shuffles": int(dists.shape[0]),
#         "null_mean": mean_mat.tolist(),
#         "null_std": std_mat.tolist(),
#         "null_pair_counts_mean": used_counts.mean(axis=0).tolist() if used_counts.shape[0] else np.zeros((L, L)).tolist(),
#         "distributions": dists,
#     }

# def _shuffle_worker(args):
#     (vecs, languages, pids, index_by_pair, n_local, seed, degenerate, pair_mode) = args
#     rng = np.random.default_rng(seed)
#     L = len(languages)
#     dists = np.zeros((n_local, L, L), dtype=np.float64)
#     used_counts = np.zeros((n_local, L, L), dtype=np.int32)

#     for t in range(n_local):
#         for i, li in enumerate(languages):
#             pids_i = [pid for pid in pids if (pid, li) in index_by_pair]
#             if not pids_i:
#                 continue
#             for j, lj in enumerate(languages):
#                 if pair_mode == "within_language" and li != lj:
#                     continue
#                 pids_j = [pid for pid in pids if (pid, lj) in index_by_pair]
#                 if not pids_j:
#                     continue
#                 common = sorted(set(pids_i) & set(pids_j))
#                 if len(common) < 2:
#                     continue
#                 perm = rng.permutation(common)

#                 rows_a, rows_b = [], []
#                 for pid_a, pid_b in zip(common, perm):
#                     ia = index_by_pair.get((pid_a, li))
#                     ib = index_by_pair.get((pid_b, lj))
#                     if ia is None or ib is None:
#                         continue
#                     rows_a.append(ia)
#                     rows_b.append(ib)
#                 if not rows_a:
#                     continue

#                 A = vecs[np.array(rows_a, dtype=np.int64)]
#                 B = vecs[np.array(rows_b, dtype=np.int64)]
#                 mean_corr, n_used = mean_spearman_over_pairs(A, B, degenerate=degenerate)
#                 dists[t, i, j] = mean_corr
#                 used_counts[t, i, j] = n_used

#     return dists, used_counts

# def shuffled_mapping_null(
#     vecs: np.ndarray,
#     languages: List[str],
#     pids: List[str],
#     index_by_pair: Dict[Tuple[str, str], int],
#     n_shuffles: int,
#     seed: int,
#     degenerate: str,
#     *,
#     pair_mode: str = "cross_language",
#     show_progress: bool = False,
# ) -> Dict[str, Any]:
#     """
#     Compute shuffle-based null distributions of mean Spearman.

#     pair_mode:
#       - "cross_language": for each (li, lj), shuffle pid mapping for lj
#       - "within_language": for each (li, li), shuffle pid mapping within same language
#     """
#     rng = _stable_rng(seed)
#     L = len(languages)

#     # store distributions as [n_shuffles, L, L]
#     dists = np.zeros((n_shuffles, L, L), dtype=np.float64)
#     used_counts = np.zeros((n_shuffles, L, L), dtype=np.int32)

#     # Precompute indices per (language, pid) for fast lookup
#     # Also store pid list that exists for each language (should be all pids)
#     # But be robust anyway.
#     for t in tqdm(
#         range(n_shuffles),
#         desc=f"Shuffles [{pair_mode}]",
#         leave=False,
#         disable=not show_progress,
#     ):
#         # independent perm per pair can be noisy; but we do per (li, lj) to match null definition
#         for i, li in enumerate(languages):
#             # available pids for li
#             pids_i = [pid for pid in pids if (pid, li) in index_by_pair]
#             if not pids_i:
#                 continue

#             for j, lj in enumerate(languages):
#                 if pair_mode == "within_language" and li != lj:
#                     # only fill diagonal if within_language baseline requested
#                     dists[t, i, j] = 0.0
#                     used_counts[t, i, j] = 0
#                     continue

#                 pids_j = [pid for pid in pids if (pid, lj) in index_by_pair]
#                 if not pids_j:
#                     continue

#                 # Use the intersection to ensure equal length pairing
#                 common = sorted(set(pids_i) & set(pids_j))
#                 if len(common) < 2:
#                     dists[t, i, j] = 0.0
#                     used_counts[t, i, j] = 0
#                     continue

#                 perm = rng.permutation(common)

#                 # Build pairs: li(pid) vs lj(perm_pid)
#                 rows_a = []
#                 rows_b = []
#                 for pid_a, pid_b in zip(common, perm):
#                     ia = index_by_pair.get((pid_a, li))
#                     ib = index_by_pair.get((pid_b, lj))
#                     if ia is None or ib is None:
#                         continue
#                     rows_a.append(ia)
#                     rows_b.append(ib)

#                 if not rows_a:
#                     dists[t, i, j] = 0.0
#                     used_counts[t, i, j] = 0
#                     continue

#                 A = vecs[np.array(rows_a, dtype=np.int64)]
#                 B = vecs[np.array(rows_b, dtype=np.int64)]
#                 mean_corr, n_used = mean_spearman_over_pairs(A, B, degenerate=degenerate)
#                 dists[t, i, j] = mean_corr
#                 used_counts[t, i, j] = n_used

#     # Summaries: mean/std per pair
#     mean_mat = dists.mean(axis=0)
#     std_mat = dists.std(axis=0, ddof=1) if n_shuffles > 1 else np.zeros_like(mean_mat)

#     return {
#         "languages": list(languages),
#         "n_shuffles": int(n_shuffles),
#         "null_mean": mean_mat.tolist(),
#         "null_std": std_mat.tolist(),
#         "null_pair_counts_mean": used_counts.mean(axis=0).tolist(),
#         "distributions": dists,  # keep as ndarray for optional NPZ saving
#     }


# def compute_p_values_and_z(
#     aligned: np.ndarray,
#     null_dists: np.ndarray,
# ) -> Dict[str, Any]:
#     """
#     For each pair (i,j):
#       p = fraction of null >= aligned (one-sided, "greater")
#       z = (aligned - mean_null) / std_null
#     """
#     n_shuffles = null_dists.shape[0]
#     mean_null = null_dists.mean(axis=0)
#     std_null = null_dists.std(axis=0, ddof=1) if n_shuffles > 1 else np.zeros_like(mean_null)

#     # p-value with +1 smoothing
#     ge = (null_dists >= aligned[None, :, :]).sum(axis=0)
#     p = (ge + 1.0) / (n_shuffles + 1.0)

#     z = np.zeros_like(mean_null)
#     mask = std_null > 0
#     z[mask] = (aligned[mask] - mean_null[mask]) / std_null[mask]
#     # if std=0, keep z=0

#     return {
#         "p_value_greater": p.tolist(),
#         "z_score": z.tolist(),
#     }


# # -----------------------------
# # Main
# # -----------------------------

# def parse_args() -> argparse.Namespace:
#     p = argparse.ArgumentParser("Compute baseline/null distributions for cross-language Spearman similarities.")
#     p.add_argument("--config", required=True, help="Path to YAML config.")
#     return p.parse_args()


# def main() -> None:
#     args = parse_args()
#     cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
#     if not isinstance(cfg, dict):
#         raise ValueError("Config must be a YAML mapping.")

#     input_dir = Path(cfg["io"]["input_dir"])
#     output_dir = Path(cfg["io"]["output_dir"])
#     output_dir.mkdir(parents=True, exist_ok=True)

#     metrics = list(cfg["baseline"].get("metrics", ["pct_var_mean_incorrect", "pct_var_max_incorrect", "rank_drop_correct"]))
#     degenerate = str(cfg["baseline"].get("degenerate_spearman", "skip"))
#     n_shuffles = int(cfg["baseline"].get("n_shuffles", 200))
#     seed = int(cfg["baseline"].get("seed", 123))
#     compute_within_language = bool(cfg["baseline"].get("within_language_baseline", True))
#     save_distributions = bool(cfg["baseline"].get("save_distributions_npz", False))

#     sweep = load_sweep_outputs(input_dir, metrics=metrics)
#     index_by_pair = build_index(sweep.samples)
#     pids = parallel_ids_present(sweep.samples)
#     languages = sweep.languages

#     report: Dict[str, Any] = {
#         "input_dir": str(input_dir),
#         "metrics": metrics,
#         "languages": languages,
#         "n_parallel_ids": len(pids),
#         "degenerate_spearman": degenerate,
#         "n_shuffles": n_shuffles,
#         "seed": seed,
#         "results": {},
#     }

#     # optional NPZ distributions
#     dist_store: Dict[str, np.ndarray] = {}

#     for metric in tqdm(metrics, desc="Metrics"):
#         vecs = sweep.vectors_by_metric[metric]

#         # 1) aligned
#         aligned = aligned_mean_spearman(
#             vecs,
#             languages=languages,
#             pids=pids,
#             index_by_pair=index_by_pair,
#             degenerate=degenerate,
#             show_progress=True,
#         )
#         aligned_mat = np.array(aligned["mean_spearman"], dtype=np.float64)
#         n_workers = int(cfg["baseline"].get("n_workers", 8))

#         # 2) cross-language shuffled null
#         null_cross = shuffled_mapping_null_parallel(
#             vecs,
#             languages=languages,
#             pids=pids,
#             index_by_pair=index_by_pair,
#             n_shuffles=n_shuffles,
#             seed=seed,
#             degenerate=degenerate,
#             pair_mode="cross_language",
#             # show_progress=True,
#             n_workers=n_workers,
#         )
#         null_cross_dists = null_cross.pop("distributions")

#         stats_cross = compute_p_values_and_z(aligned_mat, null_cross_dists)

#         out_metric: Dict[str, Any] = {
#             "aligned": aligned,
#             "null_cross_language_shuffle": {
#                 **null_cross,
#                 **stats_cross,
#             },
#         }

#         if save_distributions:
#             dist_store[f"{metric}__null_cross_language_shuffle"] = null_cross_dists.astype(np.float32)

#         # 3) within-language shuffled null (diagonal baseline)
#         if compute_within_language:
#             null_within = shuffled_mapping_null_parallel(
#                 vecs,
#                 languages=languages,
#                 pids=pids,
#                 index_by_pair=index_by_pair,
#                 n_shuffles=n_shuffles,
#                 seed=seed + 999,  # offset to avoid reusing identical RNG stream
#                 degenerate=degenerate,
#                 pair_mode="within_language",
#                 # show_progress=True,
#                 n_workers = int(cfg["baseline"].get("n_workers", 8))
#             )
#             null_within_dists = null_within.pop("distributions")
#             stats_within = compute_p_values_and_z(aligned_mat, null_within_dists)

#             out_metric["null_within_language_shuffle"] = {
#                 **null_within,
#                 **stats_within,
#             }

#             if save_distributions:
#                 dist_store[f"{metric}__null_within_language_shuffle"] = null_within_dists.astype(np.float32)

#         report["results"][metric] = out_metric

#     # Save JSON report
#     (output_dir / "baseline_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

#     # Save distributions NPZ (optional)
#     if save_distributions and dist_store:
#         np.savez_compressed(output_dir / "pairwise_distributions.npz", **dist_store)

#     print(f"Wrote baseline report to: {output_dir / 'baseline_report.json'}")
#     if save_distributions and dist_store:
#         print(f"Wrote distributions to: {output_dir / 'pairwise_distributions.npz'}")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import yaml
from tqdm.auto import tqdm


EPS = 1e-8


# -----------------------------
# Spearman helpers (tie-aware)
# -----------------------------

def _rankdata_1d_tie_aware(x: np.ndarray) -> np.ndarray:
    """Assign average ranks (1-based) to a 1D array with ties."""
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
    """
    Tie-aware Spearman correlation of two 1D arrays.

    degenerate:
      - "skip": return None if denom too small (zero variance)
      - "zero": return 0.0 if denom too small
      - "one_if_equal_else_zero": if both arrays identical -> 1.0 else 0.0
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size:
        raise ValueError("Spearman inputs must have same length.")

    ra = _rankdata_1d_tie_aware(a)
    rb = _rankdata_1d_tie_aware(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = (np.linalg.norm(ra) * np.linalg.norm(rb))

    if denom < EPS:
        if degenerate == "skip":
            return None
        if degenerate == "zero":
            return 0.0
        if degenerate == "one_if_equal_else_zero":
            return 1.0 if np.array_equal(a, b) else 0.0
        raise ValueError(f"Unknown degenerate handling: {degenerate}")

    return float(np.dot(ra, rb) / denom)


def mean_spearman_over_pairs(
    vecs_a: np.ndarray,
    vecs_b: np.ndarray,
    *,
    degenerate: str,
) -> Tuple[float, int]:
    """Mean Spearman over matched rows: vecs_a[i] with vecs_b[i]."""
    assert vecs_a.shape == vecs_b.shape
    vals: List[float] = []
    for i in range(vecs_a.shape[0]):
        c = spearman_corr(vecs_a[i], vecs_b[i], degenerate=degenerate)
        if c is None:
            continue
        vals.append(float(c))
    if not vals:
        return 0.0, 0
    return float(np.mean(vals)), len(vals)


# -----------------------------
# Jaccard@k helpers
# -----------------------------

def _topk_indices(x: np.ndarray, k: int) -> np.ndarray:
    """
    Return top-k indices by value (descending).
    Deterministic tie-break: sort by (-value, index).
    """
    x = np.asarray(x)
    n = x.size
    k = int(min(max(k, 0), n))
    if k == 0:
        return np.array([], dtype=np.int64)

    # Candidate selection (fast)
    cand = np.argpartition(x, -k)[-k:]
    # Deterministic order among candidates
    # Sort by (-x[i], i)
    cand_sorted = cand[np.lexsort((cand, -x[cand]))]
    return cand_sorted.astype(np.int64)


def jaccard_topk(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """
    Jaccard similarity between sets of top-k indices of vectors a and b.
    If both sets empty => 1.0 (identical).
    """
    ia = _topk_indices(a, k)
    ib = _topk_indices(b, k)
    sa = set(int(i) for i in ia)
    sb = set(int(i) for i in ib)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 1.0
    return float(len(sa & sb) / len(union))


def mean_jaccard_over_pairs(
    vecs_a: np.ndarray,
    vecs_b: np.ndarray,
    *,
    k: int,
) -> Tuple[float, int]:
    """Mean Jaccard@k over matched rows."""
    assert vecs_a.shape == vecs_b.shape
    vals: List[float] = []
    for i in range(vecs_a.shape[0]):
        vals.append(jaccard_topk(vecs_a[i], vecs_b[i], k=k))
    if not vals:
        return 0.0, 0
    return float(np.mean(vals)), len(vals)


# -----------------------------
# Data loading
# -----------------------------

@dataclass
class LoadedSweep:
    samples: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    vectors_by_metric: Dict[str, np.ndarray]  # metric -> [n_samples, D]
    languages: List[str]


def load_sweep_outputs(input_dir: Path, metrics: List[str]) -> LoadedSweep:
    samples = json.loads((input_dir / "samples.json").read_text(encoding="utf-8"))
    metadata = json.loads((input_dir / "metadata.json").read_text(encoding="utf-8"))

    npz = np.load(input_dir / "importance_tensors.npz")

    metric_map: Dict[str, str] = {
        "pct_var_mean_incorrect": "flat_pct_var_mean_incorrect",
        "pct_var_max_incorrect": "flat_pct_var_max_incorrect",
        "rank_drop_correct": "flat_rank_drop_correct",
    }

    vectors_by_metric: Dict[str, np.ndarray] = {}
    for m in metrics:
        if m not in metric_map:
            raise ValueError(f"Unknown metric '{m}'. Choose from {list(metric_map.keys())}.")
        key = metric_map[m]
        if key not in npz:
            raise ValueError(f"Missing '{key}' in importance_tensors.npz")
        vectors_by_metric[m] = np.asarray(npz[key])

    languages = list(metadata.get("languages", []))
    if not languages:
        languages = sorted({str(s["language"]) for s in samples})

    n = len(samples)
    for m, v in vectors_by_metric.items():
        if v.shape[0] != n:
            raise ValueError(f"Vector rows mismatch for metric {m}: {v.shape[0]} vs samples {n}.")

    return LoadedSweep(samples=samples, metadata=metadata, vectors_by_metric=vectors_by_metric, languages=languages)


# -----------------------------
# Index building
# -----------------------------

def build_index(samples: List[Dict[str, Any]]) -> Dict[Tuple[str, str], int]:
    """Map (parallel_id, language) -> sample_index."""
    idx: Dict[Tuple[str, str], int] = {}
    for s in samples:
        pid = str(s["parallel_id"])
        lang = str(s["language"])
        i = int(s["sample_index"])
        idx[(pid, lang)] = i
    return idx


def parallel_ids_present(samples: List[Dict[str, Any]]) -> List[str]:
    return sorted({str(s["parallel_id"]) for s in samples})


def _stable_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


# -----------------------------
# Aligned computations
# -----------------------------

def aligned_mean_spearman(
    vecs: np.ndarray,
    languages: List[str],
    pids: List[str],
    index_by_pair: Dict[Tuple[str, str], int],
    degenerate: str,
    *,
    show_progress: bool = False,
) -> Dict[str, Any]:
    L = len(languages)
    mat = np.zeros((L, L), dtype=np.float64)
    counts = np.zeros((L, L), dtype=np.int32)

    for i, li in enumerate(
        tqdm(
            languages,
            desc="Aligned Spearman language pairs",
            leave=False,
            disable=not show_progress,
        )
    ):
        for j, lj in enumerate(languages):
            rows_a: List[int] = []
            rows_b: List[int] = []
            for pid in pids:
                ia = index_by_pair.get((pid, li))
                ib = index_by_pair.get((pid, lj))
                if ia is None or ib is None:
                    continue
                rows_a.append(ia)
                rows_b.append(ib)

            if not rows_a:
                mat[i, j] = 0.0
                counts[i, j] = 0
                continue

            A = vecs[np.array(rows_a, dtype=np.int64)]
            B = vecs[np.array(rows_b, dtype=np.int64)]
            mean_corr, n_used = mean_spearman_over_pairs(A, B, degenerate=degenerate)
            mat[i, j] = mean_corr
            counts[i, j] = n_used

    return {"languages": list(languages), "mean_spearman": mat.tolist(), "pair_counts": counts.tolist()}


def aligned_mean_jaccard(
    vecs: np.ndarray,
    languages: List[str],
    pids: List[str],
    index_by_pair: Dict[Tuple[str, str], int],
    k: int,
    *,
    show_progress: bool = False,
) -> Dict[str, Any]:
    L = len(languages)
    mat = np.zeros((L, L), dtype=np.float64)
    counts = np.zeros((L, L), dtype=np.int32)

    for i, li in enumerate(
        tqdm(
            languages,
            desc=f"Aligned Jaccard@{k} language pairs",
            leave=False,
            disable=not show_progress,
        )
    ):
        for j, lj in enumerate(languages):
            rows_a: List[int] = []
            rows_b: List[int] = []
            for pid in pids:
                ia = index_by_pair.get((pid, li))
                ib = index_by_pair.get((pid, lj))
                if ia is None or ib is None:
                    continue
                rows_a.append(ia)
                rows_b.append(ib)

            if not rows_a:
                mat[i, j] = 0.0
                counts[i, j] = 0
                continue

            A = vecs[np.array(rows_a, dtype=np.int64)]
            B = vecs[np.array(rows_b, dtype=np.int64)]
            mean_sim, n_used = mean_jaccard_over_pairs(A, B, k=k)
            mat[i, j] = mean_sim
            counts[i, j] = n_used

    return {"languages": list(languages), "k": int(k), "mean_jaccard": mat.tolist(), "pair_counts": counts.tolist()}


# -----------------------------
# Shuffle nulls
# -----------------------------

def shuffled_mapping_null_spearman(
    vecs: np.ndarray,
    languages: List[str],
    pids: List[str],
    index_by_pair: Dict[Tuple[str, str], int],
    n_shuffles: int,
    seed: int,
    degenerate: str,
    *,
    pair_mode: str = "cross_language",
    show_progress: bool = False,
    n_workers: int = 8,
) -> Dict[str, Any]:
    """
    pair_mode:
      - "cross_language": for each (li, lj), shuffle pid mapping for lj
      - "within_language": only diagonal meaningful; shuffle within same language
    """
    L = len(languages)
    n_workers = max(1, int(n_workers))
    chunks = [n_shuffles // n_workers] * n_workers
    for i in range(n_shuffles % n_workers):
        chunks[i] += 1
    chunks = [c for c in chunks if c > 0]

    worker_args: List[Tuple[Any, ...]] = []
    base_seed = int(seed)
    for w, n_local in enumerate(chunks):
        worker_args.append(
            (vecs, languages, pids, index_by_pair, n_local, base_seed + 10007 * w, degenerate, pair_mode)
        )

    dists_list: List[np.ndarray] = []
    counts_list: List[np.ndarray] = []

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_shuffle_worker_spearman, a) for a in worker_args]
        for fut in tqdm(
            as_completed(futs),
            total=len(futs),
            desc=f"Shuffles Spearman [{pair_mode}] (parallel)",
            leave=False,
            disable=not show_progress,
        ):
            d, c = fut.result()
            dists_list.append(d)
            counts_list.append(c)

    dists = np.concatenate(dists_list, axis=0) if dists_list else np.zeros((0, L, L), dtype=np.float64)
    used_counts = np.concatenate(counts_list, axis=0) if counts_list else np.zeros((0, L, L), dtype=np.int32)

    mean_mat = dists.mean(axis=0) if dists.shape[0] else np.zeros((L, L), dtype=np.float64)
    std_mat = dists.std(axis=0, ddof=1) if dists.shape[0] > 1 else np.zeros_like(mean_mat)

    return {
        "languages": list(languages),
        "n_shuffles": int(dists.shape[0]),
        "null_mean": mean_mat.tolist(),
        "null_std": std_mat.tolist(),
        "null_pair_counts_mean": (
            used_counts.mean(axis=0).tolist() if used_counts.shape[0] else np.zeros((L, L), dtype=np.float64).tolist()
        ),
        "distributions": dists,
    }


def _shuffle_worker_spearman(args: Tuple[Any, ...]) -> Tuple[np.ndarray, np.ndarray]:
    (vecs, languages, pids, index_by_pair, n_local, seed, degenerate, pair_mode) = args
    rng = _stable_rng(seed)
    L = len(languages)
    dists = np.zeros((n_local, L, L), dtype=np.float64)
    used_counts = np.zeros((n_local, L, L), dtype=np.int32)

    for t in range(n_local):
        for i, li in enumerate(languages):
            pids_i = [pid for pid in pids if (pid, li) in index_by_pair]
            if not pids_i:
                continue

            for j, lj in enumerate(languages):
                if pair_mode == "within_language" and li != lj:
                    continue

                pids_j = [pid for pid in pids if (pid, lj) in index_by_pair]
                if not pids_j:
                    continue

                common = sorted(set(pids_i) & set(pids_j))
                if len(common) < 2:
                    continue

                perm = rng.permutation(common)

                rows_a: List[int] = []
                rows_b: List[int] = []
                for pid_a, pid_b in zip(common, perm):
                    ia = index_by_pair.get((pid_a, li))
                    ib = index_by_pair.get((pid_b, lj))
                    if ia is None or ib is None:
                        continue
                    rows_a.append(ia)
                    rows_b.append(ib)

                if not rows_a:
                    continue

                A = vecs[np.array(rows_a, dtype=np.int64)]
                B = vecs[np.array(rows_b, dtype=np.int64)]
                mean_corr, n_used = mean_spearman_over_pairs(A, B, degenerate=degenerate)
                dists[t, i, j] = mean_corr
                used_counts[t, i, j] = n_used

    return dists, used_counts


def shuffled_mapping_null_jaccard(
    vecs: np.ndarray,
    languages: List[str],
    pids: List[str],
    index_by_pair: Dict[Tuple[str, str], int],
    n_shuffles: int,
    seed: int,
    *,
    k: int,
    pair_mode: str = "cross_language",
    show_progress: bool = False,
    n_workers: int = 8,
) -> Dict[str, Any]:
    """
    Jaccard@k null by shuffling pid mapping for the second language in each pair.

    pair_mode:
      - "cross_language": shuffle pid mapping for lj
      - "within_language": only diagonal meaningful; shuffle within same language
    """
    L = len(languages)
    n_workers = max(1, int(n_workers))
    chunks = [n_shuffles // n_workers] * n_workers
    for i in range(n_shuffles % n_workers):
        chunks[i] += 1
    chunks = [c for c in chunks if c > 0]

    worker_args: List[Tuple[Any, ...]] = []
    base_seed = int(seed)
    for w, n_local in enumerate(chunks):
        worker_args.append(
            (vecs, languages, pids, index_by_pair, n_local, base_seed + 10007 * w, k, pair_mode)
        )

    dists_list: List[np.ndarray] = []
    counts_list: List[np.ndarray] = []

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_shuffle_worker_jaccard, a) for a in worker_args]
        for fut in tqdm(
            as_completed(futs),
            total=len(futs),
            desc=f"Shuffles Jaccard@{k} [{pair_mode}] (parallel)",
            leave=False,
            disable=not show_progress,
        ):
            d, c = fut.result()
            dists_list.append(d)
            counts_list.append(c)

    dists = np.concatenate(dists_list, axis=0) if dists_list else np.zeros((0, L, L), dtype=np.float64)
    used_counts = np.concatenate(counts_list, axis=0) if counts_list else np.zeros((0, L, L), dtype=np.int32)

    mean_mat = dists.mean(axis=0) if dists.shape[0] else np.zeros((L, L), dtype=np.float64)
    std_mat = dists.std(axis=0, ddof=1) if dists.shape[0] > 1 else np.zeros_like(mean_mat)

    return {
        "languages": list(languages),
        "k": int(k),
        "n_shuffles": int(dists.shape[0]),
        "null_mean": mean_mat.tolist(),
        "null_std": std_mat.tolist(),
        "null_pair_counts_mean": (
            used_counts.mean(axis=0).tolist() if used_counts.shape[0] else np.zeros((L, L), dtype=np.float64).tolist()
        ),
        "distributions": dists,
    }


def _shuffle_worker_jaccard(args: Tuple[Any, ...]) -> Tuple[np.ndarray, np.ndarray]:
    (vecs, languages, pids, index_by_pair, n_local, seed, k, pair_mode) = args
    rng = _stable_rng(seed)
    L = len(languages)
    dists = np.zeros((n_local, L, L), dtype=np.float64)
    used_counts = np.zeros((n_local, L, L), dtype=np.int32)

    for t in range(n_local):
        for i, li in enumerate(languages):
            pids_i = [pid for pid in pids if (pid, li) in index_by_pair]
            if not pids_i:
                continue

            for j, lj in enumerate(languages):
                if pair_mode == "within_language" and li != lj:
                    continue

                pids_j = [pid for pid in pids if (pid, lj) in index_by_pair]
                if not pids_j:
                    continue

                common = sorted(set(pids_i) & set(pids_j))
                if len(common) < 1:
                    continue

                perm = rng.permutation(common)

                rows_a: List[int] = []
                rows_b: List[int] = []
                for pid_a, pid_b in zip(common, perm):
                    ia = index_by_pair.get((pid_a, li))
                    ib = index_by_pair.get((pid_b, lj))
                    if ia is None or ib is None:
                        continue
                    rows_a.append(ia)
                    rows_b.append(ib)

                if not rows_a:
                    continue

                A = vecs[np.array(rows_a, dtype=np.int64)]
                B = vecs[np.array(rows_b, dtype=np.int64)]
                mean_sim, n_used = mean_jaccard_over_pairs(A, B, k=k)
                dists[t, i, j] = mean_sim
                used_counts[t, i, j] = n_used

    return dists, used_counts


# -----------------------------
# Stats: p-values and z-scores
# -----------------------------

def compute_p_values_and_z(aligned: np.ndarray, null_dists: np.ndarray) -> Dict[str, Any]:
    """
    For each pair (i,j):
      p = fraction of null >= aligned (one-sided, "greater")
      z = (aligned - mean_null) / std_null
    """
    n_shuffles = null_dists.shape[0]
    mean_null = null_dists.mean(axis=0)
    std_null = null_dists.std(axis=0, ddof=1) if n_shuffles > 1 else np.zeros_like(mean_null)

    ge = (null_dists >= aligned[None, :, :]).sum(axis=0)
    p = (ge + 1.0) / (n_shuffles + 1.0)

    z = np.zeros_like(mean_null)
    mask = std_null > 0
    z[mask] = (aligned[mask] - mean_null[mask]) / std_null[mask]

    return {"p_value_greater": p.tolist(), "z_score": z.tolist()}


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Compute shuffle baselines for cross-language Spearman + Jaccard@k similarities.")
    p.add_argument("--config", required=True, help="Path to YAML config.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping.")

    input_dir = Path(cfg["io"]["input_dir"])
    output_dir = Path(cfg["io"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = list(cfg["baseline"].get("metrics", ["pct_var_mean_incorrect", "pct_var_max_incorrect", "rank_drop_correct"]))
    degenerate = str(cfg["baseline"].get("degenerate_spearman", "skip"))
    n_shuffles = int(cfg["baseline"].get("n_shuffles", 200))
    seed = int(cfg["baseline"].get("seed", 123))
    n_workers = int(cfg["baseline"].get("n_workers", 8))
    compute_within_language = bool(cfg["baseline"].get("within_language_baseline", True))
    save_distributions = bool(cfg["baseline"].get("save_distributions_npz", False))

    # Jaccard config
    compute_jaccard = bool(cfg["baseline"].get("compute_jaccard", True))
    jaccard_ks = [int(k) for k in cfg["baseline"].get("jaccard_topk", [10, 20, 50, 100])]
    # sanity: unique, sorted
    jaccard_ks = sorted(set(jaccard_ks))

    sweep = load_sweep_outputs(input_dir, metrics=metrics)
    index_by_pair = build_index(sweep.samples)
    pids = parallel_ids_present(sweep.samples)
    languages = sweep.languages

    report: Dict[str, Any] = {
        "input_dir": str(input_dir),
        "metrics": metrics,
        "languages": languages,
        "n_parallel_ids": len(pids),
        "degenerate_spearman": degenerate,
        "n_shuffles": n_shuffles,
        "seed": seed,
        "jaccard": {
            "enabled": compute_jaccard,
            "topk": jaccard_ks,
        },
        "results": {},
    }

    dist_store: Dict[str, np.ndarray] = {}

    for metric in tqdm(metrics, desc="Metrics"):
        vecs = sweep.vectors_by_metric[metric]
        out_metric: Dict[str, Any] = {}

        # -------- Spearman: aligned + nulls --------
        aligned_sp = aligned_mean_spearman(
            vecs,
            languages=languages,
            pids=pids,
            index_by_pair=index_by_pair,
            degenerate=degenerate,
            show_progress=True,
        )
        aligned_sp_mat = np.array(aligned_sp["mean_spearman"], dtype=np.float64)

        null_sp_cross = shuffled_mapping_null_spearman(
            vecs,
            languages=languages,
            pids=pids,
            index_by_pair=index_by_pair,
            n_shuffles=n_shuffles,
            seed=seed,
            degenerate=degenerate,
            pair_mode="cross_language",
            show_progress=True,
            n_workers=n_workers,
        )
        sp_cross_dists = null_sp_cross.pop("distributions")
        stats_sp_cross = compute_p_values_and_z(aligned_sp_mat, sp_cross_dists)

        out_metric["spearman"] = {
            "aligned": aligned_sp,
            "null_cross_language_shuffle": {**null_sp_cross, **stats_sp_cross},
        }

        if compute_within_language:
            null_sp_within = shuffled_mapping_null_spearman(
                vecs,
                languages=languages,
                pids=pids,
                index_by_pair=index_by_pair,
                n_shuffles=n_shuffles,
                seed=seed + 999,
                degenerate=degenerate,
                pair_mode="within_language",
                show_progress=True,
                n_workers=n_workers,
            )
            sp_within_dists = null_sp_within.pop("distributions")
            stats_sp_within = compute_p_values_and_z(aligned_sp_mat, sp_within_dists)
            out_metric["spearman"]["null_within_language_shuffle"] = {**null_sp_within, **stats_sp_within}

            if save_distributions:
                dist_store[f"{metric}__spearman__null_within_language_shuffle"] = sp_within_dists.astype(np.float32)

        if save_distributions:
            dist_store[f"{metric}__spearman__null_cross_language_shuffle"] = sp_cross_dists.astype(np.float32)

        # -------- Jaccard@k: aligned + nulls --------
        if compute_jaccard:
            out_metric["jaccard_topk"] = {}

            for k in tqdm(jaccard_ks, desc=f"Jaccard@k [{metric}]", leave=False):
                aligned_j = aligned_mean_jaccard(
                    vecs,
                    languages=languages,
                    pids=pids,
                    index_by_pair=index_by_pair,
                    k=k,
                    show_progress=True,
                )
                aligned_j_mat = np.array(aligned_j["mean_jaccard"], dtype=np.float64)

                null_j_cross = shuffled_mapping_null_jaccard(
                    vecs,
                    languages=languages,
                    pids=pids,
                    index_by_pair=index_by_pair,
                    n_shuffles=n_shuffles,
                    seed=seed + 10000 + k,
                    k=k,
                    pair_mode="cross_language",
                    show_progress=True,
                    n_workers=n_workers,
                )
                j_cross_dists = null_j_cross.pop("distributions")
                stats_j_cross = compute_p_values_and_z(aligned_j_mat, j_cross_dists)

                entry = {
                    "aligned": aligned_j,
                    "null_cross_language_shuffle": {**null_j_cross, **stats_j_cross},
                }

                if compute_within_language:
                    null_j_within = shuffled_mapping_null_jaccard(
                        vecs,
                        languages=languages,
                        pids=pids,
                        index_by_pair=index_by_pair,
                        n_shuffles=n_shuffles,
                        seed=seed + 20000 + k,
                        k=k,
                        pair_mode="within_language",
                        show_progress=True,
                        n_workers=n_workers,
                    )
                    j_within_dists = null_j_within.pop("distributions")
                    stats_j_within = compute_p_values_and_z(aligned_j_mat, j_within_dists)
                    entry["null_within_language_shuffle"] = {**null_j_within, **stats_j_within}

                    if save_distributions:
                        dist_store[f"{metric}__jaccard@{k}__null_within_language_shuffle"] = j_within_dists.astype(
                            np.float32
                        )

                if save_distributions:
                    dist_store[f"{metric}__jaccard@{k}__null_cross_language_shuffle"] = j_cross_dists.astype(np.float32)

                out_metric["jaccard_topk"][str(k)] = entry

        report["results"][metric] = out_metric

    (output_dir / "baseline_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if save_distributions and dist_store:
        np.savez_compressed(output_dir / "pairwise_distributions.npz", **dist_store)

    print(f"Wrote baseline report to: {output_dir / 'baseline_report.json'}")
    if save_distributions and dist_store:
        print(f"Wrote distributions to: {output_dir / 'pairwise_distributions.npz'}")


if __name__ == "__main__":
    main()
