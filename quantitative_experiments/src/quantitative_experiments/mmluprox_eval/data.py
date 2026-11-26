from __future__ import annotations

from typing import Any, Dict, List, Optional

from datasets import concatenate_datasets, load_dataset


def _maybe_add_language_field(ds, language_field: str, language_value: str):
    if language_field in ds.column_names:
        return ds
    return ds.map(lambda ex: {language_field: language_value})


def load_mmluprox_dataset(cfg: Dict[str, Any]):
    name = cfg["data"]["dataset_name"]
    subset = cfg["data"].get("subset")
    subsets = cfg["data"].get("subsets")
    split = cfg["data"].get("split", "test")
    cache_dir = cfg["data"].get("cache_dir")
    language_field = cfg["data"].get("language_field", "language")
    option_fields = cfg["data"].get("option_fields", [f"option_{i}" for i in range(10)])
    drop_columns = cfg["data"].get("drop_columns", [])

    if subsets:
        parts = []
        for sub in subsets:
            ds = load_dataset(name, sub, split=split, cache_dir=cache_dir)
            ds = _maybe_add_language_field(ds, language_field, sub)
            parts.append(ds)
        ds = concatenate_datasets(parts)
    elif subset:
        ds = load_dataset(name, subset, split=split, cache_dir=cache_dir)
        ds = _maybe_add_language_field(ds, language_field, subset)
    else:
        ds = load_dataset(name, split=split, cache_dir=cache_dir)

    ds = _ensure_id_field(ds, cfg)
    if drop_columns:
        to_drop = [c for c in drop_columns if c in ds.column_names]
        if to_drop:
            ds = ds.remove_columns(to_drop)

    # Replace missing option values with empty strings to avoid None in collate.
    ds = _fill_missing_options(ds, option_fields)

    languages = cfg["data"].get("languages")
    if languages and language_field in ds.column_names:
        lang_set = set(languages)
        ds = ds.filter(lambda ex: ex.get(language_field) in lang_set)

    max_samples = cfg["data"].get("max_samples")
    seed = cfg["data"].get("seed", 42)
    if max_samples:
        ds = ds.shuffle(seed=seed).select(range(max_samples))

    return ds


def load_eval_dataset(cfg: Dict[str, Any]):
    """Backward-compatible alias used by experiment entrypoints."""
    return load_mmluprox_dataset(cfg)


def extract_options(example: Dict[str, Any], option_fields: List[str]) -> List[str]:
    options = []
    for field in option_fields:
        val = example.get(field)
        if val is None:
            options.append("")
        else:
            options.append(str(val))
    return options


def _fill_missing_options(ds, option_fields: List[str]):
    present = [f for f in option_fields if f in ds.column_names]
    if not present:
        return ds

    def _fill(ex):
        out = {}
        for f in present:
            if ex.get(f) is None:
                out[f] = ""
        return out

    return ds.map(_fill)


def _ensure_id_field(ds, cfg: Dict[str, Any]):
    data_cfg = cfg.get("data", {})
    id_fields = data_cfg.get("id_fields")
    if not id_fields:
        return ds

    id_field = data_cfg.get("id_field", "id")
    sep = data_cfg.get("id_separator", "::")

    def _build(ex):
        parts = [str(ex.get(field, "")) for field in id_fields]
        return {id_field: sep.join(parts)}

    return ds.map(_build)
