from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch


CHECKPOINT_VERSION = 1
UNIT_KEYS = (
    "pct_var_mean_incorrect",
    "pct_var_max_incorrect",
    "rank_drop_correct",
)


class CheckpointMismatchError(ValueError):
    """Raised when an existing checkpoint belongs to a different sweep."""


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _file_fingerprint(path_value: Any) -> Dict[str, Any] | None:
    if path_value is None or str(path_value).strip() == "":
        return None
    path = Path(str(path_value))
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return {"path": str(path), "sha256": digest}


def scientific_config_fingerprint(cfg: Mapping[str, Any]) -> str:
    """Hash result-affecting configuration while excluding checkpoint controls."""
    payload = copy.deepcopy(dict(cfg))
    run_cfg = payload.get("run")
    if isinstance(run_cfg, dict):
        run_cfg.pop("checkpoint", None)
    return sha256_json(payload)


def ordered_sample_fingerprint(samples: Sequence[Any]) -> str:
    """Hash the exact ordered expanded samples consumed by the sweep."""
    rows = []
    for sample in samples:
        rows.append(
            {
                "sample_id": str(sample.sample_id),
                "parallel_id": str(sample.parallel_id),
                "language": str(sample.language),
                "prompt": str(sample.prompt),
                "correct_idx": int(sample.correct_idx),
                "question": str(sample.question),
                "context": str(sample.context),
                "options": [str(value) for value in sample.options],
            }
        )
    return sha256_json(rows)


def build_manifest(
    *,
    cfg: Mapping[str, Any],
    samples: Sequence[Any],
    model_commit: str | None,
    n_eval_samples: int,
    n_layers: int,
    n_patch_units: int,
    n_heads: int,
    n_kv_heads: int,
    d_head: int,
    n_options: int,
) -> Dict[str, Any]:
    """Build the identity and shape manifest for one resumable sweep."""
    model_identity = {
        "name_or_path": cfg.get("model", {}).get("name_or_path"),
        "revision": cfg.get("model", {}).get("revision"),
        "resolved_commit": model_commit,
        "n_layers": int(n_layers),
        "n_heads": int(n_heads),
        "n_kv_heads": int(n_kv_heads),
        "d_head": int(d_head),
    }
    shape = {
        "n_eval_samples": int(n_eval_samples),
        "n_layers": int(n_layers),
        "n_patch_units": int(n_patch_units),
        "n_options": int(n_options),
    }
    input_fingerprints = {
        "allowed_parallel_ids": _file_fingerprint(
            cfg.get("data", {}).get("allowed_parallel_ids_path")
        )
    }
    identity = {
        "config_fingerprint": scientific_config_fingerprint(cfg),
        "input_fingerprints": input_fingerprints,
        "sample_fingerprint": ordered_sample_fingerprint(samples),
        "model_fingerprint": sha256_json(model_identity),
        "shape": shape,
    }
    return {
        "version": CHECKPOINT_VERSION,
        **identity,
        "fingerprint": sha256_json(identity),
        "model": model_identity,
    }


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def atomic_save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


class SweepCheckpoint:
    """Atomic per-unit checkpoint store for a head-patching sweep."""

    def __init__(
        self,
        output_dir: Path,
        manifest: Mapping[str, Any],
        *,
        resume: bool,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.root = self.output_dir / ".sweep_checkpoint"
        self.units_dir = self.root / "units"
        self.manifest_path = self.root / "manifest.json"
        self.manifest = dict(manifest)
        self.n_eval_samples = int(self.manifest["shape"]["n_eval_samples"])
        self.n_layers = int(self.manifest["shape"]["n_layers"])
        self.n_patch_units = int(self.manifest["shape"]["n_patch_units"])
        self.n_options = int(self.manifest["shape"]["n_options"])

        self.root.mkdir(parents=True, exist_ok=True)
        self.units_dir.mkdir(parents=True, exist_ok=True)
        self._initialize_or_validate(resume=resume)

    def _initialize_or_validate(self, *, resume: bool) -> None:
        if not self.manifest_path.exists():
            unexpected = [
                path
                for path in self.root.iterdir()
                if path.name != "units" and not path.name.endswith(".tmp")
            ]
            unexpected.extend(
                path for path in self.units_dir.iterdir() if not path.name.endswith(".tmp")
            )
            if unexpected:
                raise CheckpointMismatchError(
                    f"Checkpoint directory {self.root} contains data but has no manifest. "
                    "Use a new run.output_dir or move the incomplete checkpoint aside."
                )
            atomic_write_json(self.manifest_path, self.manifest)
            return

        try:
            existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointMismatchError(
                f"Could not read checkpoint manifest {self.manifest_path}: {exc}. "
                "Use a new run.output_dir or move the incomplete checkpoint aside."
            ) from exc

        if existing != self.manifest:
            mismatched = [
                key
                for key in (
                    "version",
                    "config_fingerprint",
                    "input_fingerprints",
                    "sample_fingerprint",
                    "model_fingerprint",
                    "shape",
                )
                if existing.get(key) != self.manifest.get(key)
            ]
            raise CheckpointMismatchError(
                f"Checkpoint {self.root} does not match this sweep "
                f"(different: {', '.join(mismatched) or 'manifest'}). "
                "Refusing to mix results; use a new run.output_dir or move the old "
                ".sweep_checkpoint directory aside."
            )

        if not resume and self.has_saved_progress():
            raise CheckpointMismatchError(
                f"Checkpoint progress already exists in {self.root}, but run.checkpoint.resume=false. "
                "Set resume=true, use a new run.output_dir, or move the old checkpoint aside."
            )

    def has_saved_progress(self) -> bool:
        return (
            (self.root / "donors.npz").exists()
            or (self.root / "baseline.npz").exists()
            or any(self.units_dir.glob("layer_*_unit_*.npz"))
        )

    def _unit_path(self, layer_idx: int, patch_idx: int) -> Path:
        if not (0 <= layer_idx < self.n_layers):
            raise IndexError(f"Layer index {layer_idx} is outside [0, {self.n_layers}).")
        if not (0 <= patch_idx < self.n_patch_units):
            raise IndexError(f"Patch-unit index {patch_idx} is outside [0, {self.n_patch_units}).")
        return self.units_dir / f"layer_{layer_idx:04d}_unit_{patch_idx:04d}.npz"

    def load_unit(self, layer_idx: int, patch_idx: int) -> Dict[str, np.ndarray] | None:
        path = self._unit_path(layer_idx, patch_idx)
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as saved:
                if set(saved.files) != set(UNIT_KEYS):
                    raise ValueError(f"expected keys {UNIT_KEYS}, found {saved.files}")
                arrays = {
                    key: np.asarray(saved[key], dtype=np.float32).copy()
                    for key in UNIT_KEYS
                }
        except (OSError, ValueError, KeyError) as exc:
            raise CheckpointMismatchError(
                f"Checkpoint unit {path} is unreadable or incompatible: {exc}. "
                "Move that unit file aside so it can be recomputed."
            ) from exc

        for key, array in arrays.items():
            if array.shape != (self.n_eval_samples,):
                raise CheckpointMismatchError(
                    f"Checkpoint unit {path} has invalid {key} shape {array.shape}; "
                    f"expected {(self.n_eval_samples,)}."
                )
        return arrays

    def save_unit(
        self,
        layer_idx: int,
        patch_idx: int,
        arrays: Mapping[str, np.ndarray],
    ) -> None:
        if set(arrays) != set(UNIT_KEYS):
            raise ValueError(f"Unit arrays must contain exactly {UNIT_KEYS}.")
        normalized: Dict[str, np.ndarray] = {}
        for key in UNIT_KEYS:
            value = np.asarray(arrays[key], dtype=np.float32)
            if value.shape != (self.n_eval_samples,):
                raise ValueError(
                    f"Cannot checkpoint {key} with shape {value.shape}; "
                    f"expected {(self.n_eval_samples,)}."
                )
            normalized[key] = value
        atomic_save_npz(self._unit_path(layer_idx, patch_idx), **normalized)

    def load_baseline(self) -> np.ndarray | None:
        path = self.root / "baseline.npz"
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as saved:
                if saved.files != ["option_logits"]:
                    raise ValueError(f"expected option_logits, found {saved.files}")
                option_logits = np.asarray(saved["option_logits"], dtype=np.float32).copy()
        except (OSError, ValueError, KeyError) as exc:
            raise CheckpointMismatchError(f"Invalid baseline checkpoint {path}: {exc}.") from exc
        expected = (self.n_eval_samples, self.n_options)
        if option_logits.shape != expected:
            raise CheckpointMismatchError(
                f"Baseline checkpoint has shape {option_logits.shape}; expected {expected}."
            )
        return option_logits

    def save_baseline(self, option_logits: np.ndarray) -> None:
        value = np.asarray(option_logits, dtype=np.float32)
        expected = (self.n_eval_samples, self.n_options)
        if value.shape != expected:
            raise ValueError(f"Cannot checkpoint baseline shape {value.shape}; expected {expected}.")
        atomic_save_npz(self.root / "baseline.npz", option_logits=value)

    def load_donors(
        self,
        languages: Sequence[str],
        *,
        n_layers: int,
        n_heads: int,
        d_head: int,
    ) -> Dict[str, torch.Tensor] | None:
        path = self.root / "donors.npz"
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as saved:
                if set(saved.files) != {"languages", "values"}:
                    raise ValueError(f"expected languages and values, found {saved.files}")
                saved_languages = [str(value) for value in saved["languages"].tolist()]
                values = np.asarray(saved["values"], dtype=np.float32).copy()
        except (OSError, ValueError, KeyError) as exc:
            raise CheckpointMismatchError(f"Invalid donor checkpoint {path}: {exc}.") from exc

        expected_languages = [str(value) for value in languages]
        expected_shape = (len(expected_languages), n_layers, n_heads, d_head)
        if saved_languages != expected_languages or values.shape != expected_shape:
            raise CheckpointMismatchError(
                f"Donor checkpoint does not match this sweep: languages={saved_languages}, "
                f"shape={values.shape}; expected languages={expected_languages}, "
                f"shape={expected_shape}."
            )
        return {
            language: torch.from_numpy(values[index])
            for index, language in enumerate(expected_languages)
        }

    def save_donors(
        self,
        donors: Mapping[str, torch.Tensor],
        languages: Sequence[str],
    ) -> None:
        ordered_languages = [str(value) for value in languages]
        values = np.stack(
            [
                donors[language].detach().float().cpu().numpy()
                for language in ordered_languages
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        atomic_save_npz(
            self.root / "donors.npz",
            languages=np.asarray(ordered_languages),
            values=values,
        )

    def completed_unit_count(self) -> int:
        count = 0
        for layer_idx in range(self.n_layers):
            for patch_idx in range(self.n_patch_units):
                if self._unit_path(layer_idx, patch_idx).exists():
                    count += 1
        return count
