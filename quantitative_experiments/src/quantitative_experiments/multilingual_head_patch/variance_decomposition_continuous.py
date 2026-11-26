from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import yaml


EPS = 1e-8
COMPONENT_NAMES = np.array(["lang", "question", "interaction"], dtype=object)


def _deep_merge(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(paths: Iterable[str]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    for p in paths:
        data = yaml.safe_load(Path(p).read_text(encoding="utf-8"))
        if data:
            cfg = _deep_merge(cfg, data)
    print(f"Loaded config from {list(paths)}:\n{yaml.dump(cfg, sort_keys=False)}")
    return cfg


def _set_by_path(cfg: Dict[str, Any], path: str, value: Any) -> None:
    cur = cfg
    parts = path.split(".")
    for key in parts[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[parts[-1]] = value


def apply_overrides(cfg: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Use key=value.")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        _set_by_path(cfg, key, value)
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuous variance decomposition of head importance (language/question/interaction)"
    )
    parser.add_argument("--config", action="append", required=True, help="YAML config path (can be repeated).")
    parser.add_argument("--set", action="append", default=[], help="Config override in key=value format.")
    return parser.parse_args()


@dataclass
class RunArtifacts:
    samples: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    tensors: Dict[str, np.ndarray]


def _rankdata_1d(x: np.ndarray) -> np.ndarray:
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


def _normalize_tensor(x: np.ndarray, method: str, eps: float) -> np.ndarray:
    if method == "none":
        return x.astype(np.float64, copy=False)

    if method == "zscore":
        mu = x.mean(axis=2, keepdims=True)
        sigma = x.std(axis=2, keepdims=True)
        sigma = np.where(sigma < eps, 1.0, sigma)
        return (x - mu) / sigma

    if method == "rank_percentile":
        l, q, h = x.shape
        out = np.zeros((l, q, h), dtype=np.float64)
        if h == 1:
            return out
        denom = float(h - 1)
        for li in range(l):
            for qi in range(q):
                ranks = _rankdata_1d(x[li, qi, :].astype(np.float64, copy=False))
                out[li, qi, :] = (ranks - 1.0) / denom
        return out

    raise ValueError("processing.normalization must be one of: none, zscore, rank_percentile")


def _load_artifacts(cfg: Dict[str, Any]) -> RunArtifacts:
    run_dir = Path(cfg["input"]["run_output_dir"])
    npz_path = run_dir / "importance_tensors.npz"
    samples_path = run_dir / "samples.json"
    metadata_path = run_dir / "metadata.json"

    if not npz_path.exists():
        raise FileNotFoundError(f"Missing tensor file: {npz_path}")
    if not samples_path.exists():
        raise FileNotFoundError(f"Missing samples file: {samples_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    with np.load(npz_path) as data:
        tensors = {k: np.asarray(data[k]) for k in data.files}

    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        raise ValueError("samples.json must contain a list.")

    return RunArtifacts(samples=samples, metadata=metadata, tensors=tensors)


def _select_metric_names(cfg: Dict[str, Any], tensors: Dict[str, np.ndarray]) -> List[str]:
    requested = cfg["input"].get("metrics")
    if requested:
        metric_names = [str(m) for m in requested]
    else:
        metric_names = [
            k
            for k, arr in tensors.items()
            if arr.ndim == 3 and k not in {"baseline_option_logits"}
        ]

    missing = [m for m in metric_names if m not in tensors]
    if missing:
        raise ValueError(f"Requested metrics not found in importance_tensors.npz: {missing}")

    for m in metric_names:
        if tensors[m].ndim != 3:
            raise ValueError(f"Metric '{m}' must have shape [N, layers, patch_units], got ndim={tensors[m].ndim}.")

    return metric_names


def _resolve_languages(cfg: Dict[str, Any], metadata: Dict[str, Any], samples: Sequence[Dict[str, Any]]) -> List[str]:
    from_cfg = cfg["input"].get("languages")
    if from_cfg:
        return [str(x) for x in from_cfg]

    from_meta = metadata.get("languages")
    if from_meta:
        return [str(x) for x in from_meta]

    return sorted({str(s.get("language")) for s in samples if s.get("language") is not None})


def _build_lqh_tensor(
    arr_nlu: np.ndarray,
    samples: Sequence[Dict[str, Any]],
    languages: Sequence[str],
    require_balanced: bool,
    drop_incomplete_questions: bool,
) -> Tuple[np.ndarray, List[str], Dict[str, int], int, int]:
    if arr_nlu.ndim != 3:
        raise ValueError(f"Expected [N, layers, patch_units], got shape {arr_nlu.shape}")

    n_samples, n_layers, n_units = arr_nlu.shape
    if len(samples) != n_samples:
        raise ValueError(
            f"Sample count mismatch: samples.json has {len(samples)} rows but tensor has {n_samples} rows."
        )

    arr_nh = arr_nlu.reshape(n_samples, n_layers * n_units)

    by_pair: Dict[Tuple[str, str], int] = {}
    all_parallel_ids: set[str] = set()
    for i, s in enumerate(samples):
        pid = str(s.get("parallel_id"))
        lang = str(s.get("language"))
        all_parallel_ids.add(pid)
        key = (pid, lang)
        if key in by_pair:
            raise ValueError(f"Duplicate sample for pair (parallel_id={pid}, language={lang}).")
        by_pair[key] = i

    kept_parallel_ids: List[str] = []
    missing_map: Dict[str, List[str]] = {}
    for pid in sorted(all_parallel_ids):
        missing_langs = [lang for lang in languages if (pid, lang) not in by_pair]
        if missing_langs:
            missing_map[pid] = missing_langs
            if require_balanced and not drop_incomplete_questions:
                raise ValueError(
                    f"Unbalanced design: parallel_id={pid} missing languages {missing_langs}. "
                    "Set input.drop_incomplete_questions=true to skip incomplete IDs."
                )
            if drop_incomplete_questions:
                continue
        kept_parallel_ids.append(pid)

    if not kept_parallel_ids:
        raise ValueError("No complete parallel questions left after alignment/filtering.")

    l = len(languages)
    q = len(kept_parallel_ids)
    h = n_layers * n_units
    x = np.zeros((l, q, h), dtype=np.float64)

    for qi, pid in enumerate(kept_parallel_ids):
        for li, lang in enumerate(languages):
            key = (pid, lang)
            if key not in by_pair:
                if require_balanced:
                    raise ValueError(f"Balanced tensor requested, but missing pair {key}.")
                continue
            x[li, qi, :] = arr_nh[by_pair[key], :]

    return x, kept_parallel_ids, {pid: i for i, pid in enumerate(kept_parallel_ids)}, n_layers, n_units


def _decompose(x: np.ndarray, eps: float) -> Dict[str, np.ndarray]:
    l, q, _ = x.shape
    mu = x.mean(axis=(0, 1))
    a = x.mean(axis=1) - mu
    b = x.mean(axis=0) - mu
    r = x - mu[None, None, :] - a[:, None, :] - b[None, :, :]

    ss_lang = float(q) * np.sum(a * a, axis=0)
    ss_question = float(l) * np.sum(b * b, axis=0)
    ss_interaction = np.sum(r * r, axis=(0, 1))
    ss_total = np.sum((x - mu[None, None, :]) ** 2, axis=(0, 1))

    f_lang = np.zeros_like(ss_total)
    f_question = np.zeros_like(ss_total)
    f_interaction = np.zeros_like(ss_total)

    valid = ss_total > eps
    f_lang[valid] = ss_lang[valid] / ss_total[valid]
    f_question[valid] = ss_question[valid] / ss_total[valid]
    f_interaction[valid] = ss_interaction[valid] / ss_total[valid]

    return {
        "ss_lang": ss_lang,
        "ss_question": ss_question,
        "ss_interaction": ss_interaction,
        "ss_total": ss_total,
        "f_lang": f_lang,
        "f_question": f_question,
        "f_interaction": f_interaction,
    }


def _continuous_head_stats(decomp: Dict[str, np.ndarray], eps: float) -> Dict[str, np.ndarray]:
    fractions = np.stack(
        [decomp["f_lang"], decomp["f_question"], decomp["f_interaction"]],
        axis=1,
    )
    valid = decomp["ss_total"] > eps

    order = np.argsort(fractions, axis=1)
    top_idx = order[:, -1]
    second_idx = order[:, -2]
    dominant_fraction = fractions[np.arange(fractions.shape[0]), top_idx]
    second_fraction = fractions[np.arange(fractions.shape[0]), second_idx]
    dominance_margin = dominant_fraction - second_fraction

    frac_sum = fractions.sum(axis=1, keepdims=True)
    probs = np.zeros_like(fractions)
    np.divide(fractions, frac_sum, out=probs, where=frac_sum > eps)
    log_probs = np.zeros_like(probs)
    positive = probs > 0.0
    log_probs[positive] = np.log(probs[positive])
    component_entropy = -np.sum(probs * log_probs, axis=1) / np.log(3.0)

    dominant_component = COMPONENT_NAMES[top_idx].astype(object)
    dominant_component[~valid] = "none"
    dominant_fraction = np.where(valid, dominant_fraction, 0.0)
    second_fraction = np.where(valid, second_fraction, 0.0)
    dominance_margin = np.where(valid, dominance_margin, 0.0)
    component_entropy = np.where(valid, component_entropy, 0.0)

    return {
        "dominant_component": dominant_component,
        "dominant_fraction": dominant_fraction,
        "second_fraction": second_fraction,
        "dominance_margin": dominance_margin,
        "component_entropy": component_entropy,
    }


def _fraction_summary(decomp: Dict[str, np.ndarray], eps: float) -> Dict[str, Any]:
    valid = decomp["ss_total"] > eps
    ss_total_sum = float(np.sum(decomp["ss_total"]))

    def _mean(name: str, mask: np.ndarray | None = None) -> float:
        values = decomp[name] if mask is None else decomp[name][mask]
        if values.size == 0:
            return float("nan")
        return float(np.mean(values))

    if ss_total_sum > eps:
        weighted = {
            "f_lang": float(np.sum(decomp["ss_lang"]) / ss_total_sum),
            "f_question": float(np.sum(decomp["ss_question"]) / ss_total_sum),
            "f_interaction": float(np.sum(decomp["ss_interaction"]) / ss_total_sum),
        }
    else:
        weighted = {"f_lang": float("nan"), "f_question": float("nan"), "f_interaction": float("nan")}

    return {
        "mean_fraction_all_heads": {
            "f_lang": _mean("f_lang"),
            "f_question": _mean("f_question"),
            "f_interaction": _mean("f_interaction"),
        },
        "mean_fraction_nonzero_heads": {
            "f_lang": _mean("f_lang", valid),
            "f_question": _mean("f_question", valid),
            "f_interaction": _mean("f_interaction", valid),
        },
        "global_fraction_ss_weighted": weighted,
        "global_f_lang": weighted["f_lang"],
        "global_f_question": weighted["f_question"],
        "global_f_interaction": weighted["f_interaction"],
        "ss_total_sum": ss_total_sum,
        "n_nonzero_heads": int(np.sum(valid)),
    }


def _dominant_component_counts(stats: Dict[str, np.ndarray]) -> Dict[str, int]:
    labels = stats["dominant_component"]
    names = ["lang", "question", "interaction", "none"]
    return {name: int(np.sum(labels == name)) for name in names}


def _layer_component_distribution(
    decomp: Dict[str, np.ndarray],
    stats: Dict[str, np.ndarray],
    n_layers: int,
    n_units: int,
    eps: float,
) -> Dict[str, Any]:
    per_layer = []
    for layer in range(n_layers):
        sl = slice(layer * n_units, (layer + 1) * n_units)
        ss_total_sum = float(np.sum(decomp["ss_total"][sl]))
        if ss_total_sum > eps:
            f_lang = float(np.sum(decomp["ss_lang"][sl]) / ss_total_sum)
            f_question = float(np.sum(decomp["ss_question"][sl]) / ss_total_sum)
            f_interaction = float(np.sum(decomp["ss_interaction"][sl]) / ss_total_sum)
        else:
            f_lang = 0.0
            f_question = 0.0
            f_interaction = 0.0

        labels = stats["dominant_component"][sl]
        row: Dict[str, Any] = {
            "layer": int(layer),
            "n_heads": int(n_units),
            "n_nonzero_heads": int(np.sum(decomp["ss_total"][sl] > eps)),
            "ss_total_sum": ss_total_sum,
            "f_lang": f_lang,
            "f_question": f_question,
            "f_interaction": f_interaction,
        }
        for name in ["lang", "question", "interaction", "none"]:
            row[f"dominant_count_{name}"] = int(np.sum(labels == name))
        per_layer.append(row)

    return {"component_names": ["lang", "question", "interaction"], "per_layer": per_layer}


def _pick_continuous_examples(
    stats: Dict[str, np.ndarray],
    decomp: Dict[str, np.ndarray],
    n_layers: int,
    n_units: int,
    max_per_component: int,
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for component in ["lang", "question", "interaction"]:
        idx = np.where(stats["dominant_component"] == component)[0]
        if idx.size == 0:
            out[component] = []
            continue
        order = np.lexsort((-stats["dominance_margin"][idx], -stats["dominant_fraction"][idx]))
        chosen = idx[order[:max_per_component]]
        rows = []
        for h in chosen.tolist():
            rows.append(
                {
                    "flat_head_idx": int(h),
                    "layer": int(h // n_units),
                    "patch_unit": int(h % n_units),
                    "f_lang": float(decomp["f_lang"][h]),
                    "f_question": float(decomp["f_question"][h]),
                    "f_interaction": float(decomp["f_interaction"][h]),
                    "dominant_fraction": float(stats["dominant_fraction"][h]),
                    "dominance_margin": float(stats["dominance_margin"][h]),
                    "component_entropy": float(stats["component_entropy"][h]),
                }
            )
        out[component] = rows
    return out


def _bootstrap_summary(
    x: np.ndarray,
    n_resamples: int,
    seed: int,
    eps: float,
) -> Dict[str, Any]:
    l, q, _ = x.shape
    if q <= 0:
        raise ValueError("Cannot bootstrap with zero questions.")

    rng = np.random.default_rng(seed)
    mean_samples = np.zeros((n_resamples, 3), dtype=np.float64)
    weighted_samples = np.zeros((n_resamples, 3), dtype=np.float64)

    for b in range(n_resamples):
        idx = rng.integers(0, q, size=q)
        db = _decompose(x[:, idx, :], eps)
        valid = db["ss_total"] > eps
        if np.any(valid):
            mean_samples[b, 0] = float(np.mean(db["f_lang"][valid]))
            mean_samples[b, 1] = float(np.mean(db["f_question"][valid]))
            mean_samples[b, 2] = float(np.mean(db["f_interaction"][valid]))
        else:
            mean_samples[b, :] = np.nan

        ss_total_sum = float(np.sum(db["ss_total"]))
        if ss_total_sum > eps:
            weighted_samples[b, 0] = float(np.sum(db["ss_lang"]) / ss_total_sum)
            weighted_samples[b, 1] = float(np.sum(db["ss_question"]) / ss_total_sum)
            weighted_samples[b, 2] = float(np.sum(db["ss_interaction"]) / ss_total_sum)
        else:
            weighted_samples[b, :] = np.nan

    def _ci(col: np.ndarray) -> Dict[str, float]:
        col = col[np.isfinite(col)]
        if col.size == 0:
            return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
        return {
            "mean": float(np.mean(col)),
            "ci95_low": float(np.quantile(col, 0.025)),
            "ci95_high": float(np.quantile(col, 0.975)),
        }

    return {
        "n_resamples": int(n_resamples),
        "n_languages": int(l),
        "n_questions": int(q),
        "mean_fraction_nonzero_heads": {
            "f_lang": _ci(mean_samples[:, 0]),
            "f_question": _ci(mean_samples[:, 1]),
            "f_interaction": _ci(mean_samples[:, 2]),
        },
        "global_fraction_ss_weighted": {
            "f_lang": _ci(weighted_samples[:, 0]),
            "f_question": _ci(weighted_samples[:, 1]),
            "f_interaction": _ci(weighted_samples[:, 2]),
        },
    }


def _write_head_csv(
    out_path: Path,
    n_layers: int,
    n_units: int,
    decomp: Dict[str, np.ndarray],
    stats: Dict[str, np.ndarray],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "flat_head_idx",
                "layer",
                "patch_unit",
                "ss_total",
                "ss_lang",
                "ss_question",
                "ss_interaction",
                "f_lang",
                "f_question",
                "f_interaction",
                "dominant_component",
                "dominant_fraction",
                "second_fraction",
                "dominance_margin",
                "component_entropy",
            ]
        )
        for h in range(n_layers * n_units):
            w.writerow(
                [
                    h,
                    h // n_units,
                    h % n_units,
                    float(decomp["ss_total"][h]),
                    float(decomp["ss_lang"][h]),
                    float(decomp["ss_question"][h]),
                    float(decomp["ss_interaction"][h]),
                    float(decomp["f_lang"][h]),
                    float(decomp["f_question"][h]),
                    float(decomp["f_interaction"][h]),
                    str(stats["dominant_component"][h]),
                    float(stats["dominant_fraction"][h]),
                    float(stats["second_fraction"][h]),
                    float(stats["dominance_margin"][h]),
                    float(stats["component_entropy"][h]),
                ]
            )


def _save_metric_outputs(
    out_dir: Path,
    metric_name: str,
    x_norm: np.ndarray,
    parallel_ids: Sequence[str],
    languages: Sequence[str],
    n_layers: int,
    n_units: int,
    decomp: Dict[str, np.ndarray],
    stats: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
) -> None:
    metric_dir = out_dir / metric_name
    metric_dir.mkdir(parents=True, exist_ok=True)
    eps = float(cfg["processing"].get("epsilon", EPS))

    summary = {
        "metric": metric_name,
        "analysis": "variance_decomposition_continuous",
        "normalization": cfg["processing"].get("normalization", "rank_percentile"),
        "epsilon": eps,
        "n_languages": int(len(languages)),
        "n_parallel_questions": int(len(parallel_ids)),
        "n_layers": int(n_layers),
        "n_patch_units": int(n_units),
        "n_total_heads": int(n_layers * n_units),
        "dominant_component_counts": _dominant_component_counts(stats),
    }
    summary.update(_fraction_summary(decomp, eps))

    bootstrap_cfg = cfg.get("bootstrap", {})
    if bool(bootstrap_cfg.get("enabled", False)):
        summary["bootstrap"] = _bootstrap_summary(
            x=x_norm,
            n_resamples=int(bootstrap_cfg.get("n_resamples", 200)),
            seed=int(bootstrap_cfg.get("seed", 42)),
            eps=eps,
        )

    max_examples = int(cfg.get("output", {}).get("save_top_examples_per_component", 3))
    signature_examples = _pick_continuous_examples(
        stats=stats,
        decomp=decomp,
        n_layers=n_layers,
        n_units=n_units,
        max_per_component=max_examples,
    )

    signature_arrays: Dict[str, np.ndarray] = {}
    for component, rows in signature_examples.items():
        for row in rows:
            h = int(row["flat_head_idx"])
            key = f"{component}__L{int(row['layer'])}_U{int(row['patch_unit'])}"
            signature_arrays[key] = x_norm[:, :, h].astype(np.float32, copy=False)

    (metric_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (metric_dir / "layer_component_distribution.json").write_text(
        json.dumps(_layer_component_distribution(decomp, stats, n_layers, n_units, eps), indent=2),
        encoding="utf-8",
    )
    (metric_dir / "signature_examples.json").write_text(json.dumps(signature_examples, indent=2), encoding="utf-8")

    dominant_codes = np.full(stats["dominant_component"].shape, -1, dtype=np.int16)
    for code, name in enumerate(["lang", "question", "interaction"]):
        dominant_codes[stats["dominant_component"] == name] = code

    np.savez_compressed(
        metric_dir / "decomposition_tensors.npz",
        f_lang=decomp["f_lang"].reshape(n_layers, n_units).astype(np.float32),
        f_question=decomp["f_question"].reshape(n_layers, n_units).astype(np.float32),
        f_interaction=decomp["f_interaction"].reshape(n_layers, n_units).astype(np.float32),
        ss_total=decomp["ss_total"].reshape(n_layers, n_units).astype(np.float32),
        ss_lang=decomp["ss_lang"].reshape(n_layers, n_units).astype(np.float32),
        ss_question=decomp["ss_question"].reshape(n_layers, n_units).astype(np.float32),
        ss_interaction=decomp["ss_interaction"].reshape(n_layers, n_units).astype(np.float32),
        dominant_component_code=dominant_codes.reshape(n_layers, n_units),
        dominant_fraction=stats["dominant_fraction"].reshape(n_layers, n_units).astype(np.float32),
        second_fraction=stats["second_fraction"].reshape(n_layers, n_units).astype(np.float32),
        dominance_margin=stats["dominance_margin"].reshape(n_layers, n_units).astype(np.float32),
        component_entropy=stats["component_entropy"].reshape(n_layers, n_units).astype(np.float32),
    )

    if signature_arrays:
        np.savez_compressed(metric_dir / "signature_matrices.npz", **signature_arrays)

    _write_head_csv(
        out_path=metric_dir / "head_decomposition.csv",
        n_layers=n_layers,
        n_units=n_units,
        decomp=decomp,
        stats=stats,
    )


def run(cfg: Dict[str, Any]) -> None:
    artifacts = _load_artifacts(cfg)
    metric_names = _select_metric_names(cfg, artifacts.tensors)
    languages = _resolve_languages(cfg, artifacts.metadata, artifacts.samples)

    output_dir = Path(cfg["output"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    run_manifest = {
        "analysis": "variance_decomposition_continuous",
        "input_run_output_dir": str(cfg["input"]["run_output_dir"]),
        "metrics": metric_names,
        "languages": languages,
        "normalization": cfg["processing"].get("normalization", "rank_percentile"),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    for metric_name in metric_names:
        arr = np.asarray(artifacts.tensors[metric_name], dtype=np.float64)
        x, parallel_ids, _, n_layers, n_units = _build_lqh_tensor(
            arr_nlu=arr,
            samples=artifacts.samples,
            languages=languages,
            require_balanced=bool(cfg["input"].get("require_balanced", True)),
            drop_incomplete_questions=bool(cfg["input"].get("drop_incomplete_questions", True)),
        )

        x_norm = _normalize_tensor(
            x,
            method=str(cfg["processing"].get("normalization", "rank_percentile")).strip().lower(),
            eps=float(cfg["processing"].get("epsilon", EPS)),
        )

        decomp = _decompose(x_norm, eps=float(cfg["processing"].get("epsilon", EPS)))
        stats = _continuous_head_stats(decomp, eps=float(cfg["processing"].get("epsilon", EPS)))
        _save_metric_outputs(
            out_dir=output_dir,
            metric_name=metric_name,
            x_norm=x_norm,
            parallel_ids=parallel_ids,
            languages=languages,
            n_layers=n_layers,
            n_units=n_units,
            decomp=decomp,
            stats=stats,
            cfg=cfg,
        )

        print(
            f"[{metric_name}] saved continuous decomposition for {n_layers * n_units} heads "
            f"(L={len(languages)}, Q={len(parallel_ids)})."
        )

    print(f"Saved continuous outputs to {output_dir.resolve()}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.set)
    run(cfg)


if __name__ == "__main__":
    main()
