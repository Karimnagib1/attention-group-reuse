#!/usr/bin/env python3
"""Shared helpers for the standalone correctness-stratified analysis."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml


EPS = 1.0e-8


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def resolve_path(config_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_ready(dict(row)), sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def stable_order(values: Iterable[str], seed: int, *parts: object) -> list[str]:
    def key(value: str) -> tuple[str, str]:
        payload = "|".join([str(seed), *(str(part) for part in parts), value])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest(), value

    return sorted((str(value) for value in values), key=key)


def proportional_pattern_sample(
    rows: Sequence[Mapping[str, Any]],
    cap: int | None,
    seed: int,
    condition_id: str,
    k_correct: int,
) -> tuple[list[str], dict[str, int]]:
    """Select a deterministic pattern-stratified sample within one K stratum."""
    groups: dict[str, list[str]] = {}
    for row in rows:
        groups.setdefault(str(row["correctness_pattern"]), []).append(str(row["parallel_id"]))
    sizes = {pattern: len(ids) for pattern, ids in groups.items()}
    total = sum(sizes.values())
    if cap is None or cap >= total:
        return sorted(pid for ids in groups.values() for pid in ids), dict(sizes)
    if cap <= 0:
        raise ValueError("max_questions_per_k must be positive or null.")

    patterns = sorted(groups)
    quotas = {pattern: 0 for pattern in patterns}
    if cap >= len(patterns):
        for pattern in patterns:
            quotas[pattern] = 1

    remaining = cap - sum(quotas.values())
    while remaining > 0:
        available = {pattern: sizes[pattern] - quotas[pattern] for pattern in patterns}
        available = {pattern: count for pattern, count in available.items() if count > 0}
        if not available:
            break
        available_total = sum(available.values())
        ideals = {pattern: remaining * count / available_total for pattern, count in available.items()}
        additions = {pattern: min(available[pattern], int(math.floor(value))) for pattern, value in ideals.items()}
        added = sum(additions.values())
        for pattern, count in additions.items():
            quotas[pattern] += count
        remaining -= added
        if remaining == 0:
            break
        candidates = sorted(
            available,
            key=lambda pattern: (-(ideals[pattern] - math.floor(ideals[pattern])), pattern),
        )
        progressed = False
        for pattern in candidates:
            if remaining == 0:
                break
            if quotas[pattern] < sizes[pattern]:
                quotas[pattern] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    selected: list[str] = []
    for pattern in patterns:
        ordered = stable_order(groups[pattern], seed, condition_id, k_correct, pattern)
        selected.extend(ordered[: quotas[pattern]])
    if len(selected) != cap:
        raise RuntimeError(f"Pattern sampling selected {len(selected)} IDs, expected {cap}.")
    return sorted(selected), quotas


def rankdata_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError("rankdata_rows expects a two-dimensional array.")
    n_rows, width = values.shape
    order = np.argsort(values, axis=1, kind="mergesort")
    ranks = np.empty((n_rows, width), dtype=np.float64)
    ordinal = np.broadcast_to(np.arange(1, width + 1, dtype=np.float64), (n_rows, width))
    np.put_along_axis(ranks, order, ordinal, axis=1)
    for row_index in range(n_rows):
        sorted_values = values[row_index, order[row_index]]
        starts = np.concatenate(([0], np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1))
        ends = np.concatenate((starts[1:], [width]))
        for start, end in zip(starts, ends):
            if end - start > 1:
                ranks[row_index, order[row_index, start:end]] = 0.5 * (start + end - 1) + 1.0
    return ranks


def normalized_rank_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ranks = rankdata_rows(values)
    ranks -= ranks.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(ranks, axis=1)
    valid = norms > EPS
    normalized = np.divide(ranks, norms[:, None], out=np.zeros_like(ranks), where=valid[:, None])
    return normalized, valid


def topk_binary_rows(values: np.ndarray, k: int) -> np.ndarray:
    values = np.asarray(values)
    k = min(max(int(k), 1), values.shape[1])
    binary = np.zeros(values.shape, dtype=np.float64)
    for row_index, row in enumerate(values):
        indices = np.argsort(-row, kind="mergesort")[:k]
        binary[row_index, indices] = 1.0
    return binary


def bootstrap_mean_interval(
    values: np.ndarray,
    n_resamples: int,
    ci_level: float,
    seed: int,
    block_size: int = 256,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1 or n_resamples <= 0:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=np.float64)
    for start in range(0, n_resamples, block_size):
        stop = min(n_resamples, start + block_size)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        samples[start:stop] = values[indices].mean(axis=1)
    alpha = 1.0 - ci_level
    low, high = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2:
        return float("nan")
    centered = x - x.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= EPS:
        return float("nan")
    return float(np.dot(centered, y - y.mean()) / denominator)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (len(p) - rank) * float(p[original_index]))
        running = max(running, candidate)
        adjusted[original_index] = running
    return adjusted.tolist()


def standardized_mean_difference(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.size == 0 or second.size == 0:
        return float("nan")
    pooled = math.sqrt(0.5 * (float(np.var(first)) + float(np.var(second))))
    if pooled <= EPS:
        return 0.0 if abs(float(np.mean(first) - np.mean(second))) <= EPS else float("inf")
    return float((np.mean(first) - np.mean(second)) / pooled)
