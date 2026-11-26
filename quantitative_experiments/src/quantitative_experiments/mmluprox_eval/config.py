from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


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
        data = yaml.safe_load(Path(p).read_text())
        if data:
            cfg = _deep_merge(cfg, data)
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
        # Parse value using YAML to allow lists/bools/numbers
        value = yaml.safe_load(raw)
        _set_by_path(cfg, key, value)
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MMLU-Prox evaluation")
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="Path to a YAML config file. Can be passed multiple times.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config entries, e.g. --set model.name_or_path=meta-llama/Llama-3.1-8B",
    )
    return parser.parse_args()

