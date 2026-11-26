from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
from pathlib import Path

import numpy as np

from .backend import Backend
from .protocol import digest, implementation_digest, validate_protocol
from quantitative_experiments.multilingual_head_patch.checkpoint import atomic_save_npz, atomic_write_json
from quantitative_experiments.multilingual_head_patch.run import (
    _group_mean, _margin_vs_incorrect_mean, _margin_vs_incorrect_max, _percent_drop, _rank_from_desc,
)

METRICS = ("pct_var_mean_incorrect", "pct_var_max_incorrect", "rank_drop_correct")


@contextmanager
def locked(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another writer is using {directory}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def read_npz(path: Path, fingerprint: str) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        values = {k: archive[k] for k in archive.files}
    if str(values.pop("fingerprint")) != fingerprint:
        raise ValueError(f"Checkpoint belongs to a different run: {path}")
    if any(not np.isfinite(value).all() for value in values.values() if np.issubdtype(value.dtype, np.number)):
        raise ValueError(f"Non-finite checkpoint: {path}")
    return values


def prepare_donors(protocol: dict, backend, output: Path, fingerprint: str) -> tuple[dict, list]:
    directory = output / "donor_cache"
    directory.mkdir(exist_ok=True)
    ready = directory / "_READY.json"
    bank_rows = protocol["bank_rows"]
    shape = (len(bank_rows), backend.width)
    if ready.exists():
        marker = json.loads(ready.read_text())
        if marker != {"fingerprint": fingerprint, "shape": list(shape), "n_layers": backend.n_layers}:
            raise ValueError("Donor cache identity or dimensions differ.")
        sums = read_npz(directory / "sums.npz", fingerprint)
        banks = [np.load(directory / f"layer_{layer:03d}.npy", mmap_mode="r", allow_pickle=False)
                 for layer in range(backend.n_layers)]
        if any(bank.shape != shape or bank.dtype != np.float32 for bank in banks):
            raise ValueError("Donor activation bank has unexpected shape or dtype.")
        return sums, banks

    # Save only activations needed for recipients and sampled individual donors;
    # stream the rest into sums. This avoids keeping the full donor pool in RAM.
    banks = [np.lib.format.open_memmap(directory / f"layer_{layer:03d}.npy", mode="w+", dtype=np.float32, shape=shape)
             for layer in range(backend.n_layers)]
    bank_positions = {sample_index: position for position, sample_index in enumerate(bank_rows)}
    languages = protocol["languages"]
    language_index = {language: i for i, language in enumerate(languages)}
    mean_variants = [v for v in protocol["variants"] if v["kind"] in {"language_mean", "disjoint_mean"}]
    pools = [None if v["kind"] == "language_mean" else set(v["parallel_ids"]) for v in mean_variants]
    totals = np.zeros((len(pools), len(languages), backend.n_layers, backend.width), dtype=np.float64)
    counts = np.zeros((len(pools), len(languages)), dtype=np.int64)
    samples = protocol["samples"]
    batch_size = protocol["config"]["patching"]["donor_batch_size"]
    filled = set()
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        values = backend.capture([s["prompt"] for s in batch])
        if values.shape != (len(batch), backend.n_layers, backend.width) or not np.isfinite(values).all():
            raise ValueError("Donor capture has invalid shape or values.")
        for offset, sample in enumerate(batch):
            lang = language_index[sample["language"]]
            for index, pool in enumerate(pools):
                if pool is None or sample["parallel_id"] in pool:
                    totals[index, lang] += values[offset]
                    counts[index, lang] += 1
            sample_index = start + offset
            if sample_index in bank_positions:
                position = bank_positions[sample_index]
                for layer, bank in enumerate(banks):
                    bank[position] = values[offset, layer]
                filled.add(sample_index)
        print(f"Donor activations: {min(start + batch_size, len(samples))}/{len(samples)}", flush=True)
    if filled != set(bank_rows) or np.any(counts == 0):
        raise ValueError("Incomplete donor bank or empty language mean.")
    for index, pool in enumerate(pools):
        expected = len(protocol["reference_ids"]) if pool is None else len(pool)
        if not np.all(counts[index] == expected):
            raise ValueError("Donor pool is not balanced across languages.")
    for bank in banks:
        bank.flush()
    sums = {"totals": totals, "counts": counts, "names": np.array([v["name"] for v in mean_variants])}
    atomic_save_npz(directory / "sums.npz", fingerprint=np.array(fingerprint), **sums)
    atomic_write_json(ready, {"fingerprint": fingerprint, "shape": list(shape), "n_layers": backend.n_layers})
    return sums, banks


def donor_vectors(protocol: dict, variant: dict, sums: dict, bank: np.ndarray, layer: int) -> np.ndarray:
    positions = {index: pos for pos, index in enumerate(protocol["bank_rows"])}
    recipients = [protocol["samples"][index] for index in protocol["recipient_rows"]]
    lang_index = {lang: i for i, lang in enumerate(protocol["languages"])}
    langs = np.array([lang_index[s["language"]] for s in recipients])
    names = sums["names"].tolist()
    kind = variant["kind"]
    if kind == "random_in_language":
        values = bank[[positions[i] for i in variant["donor_indices"]]]
    elif kind == "leave_one_out":
        index = names.index("language_mean")
        counts = sums["counts"][index, langs] - 1
        if np.any(counts < 1):
            raise ValueError("Leave-one-out means need at least two reference questions.")
        originals = bank[[positions[i] for i in protocol["recipient_rows"]]]
        values = (sums["totals"][index, langs, layer] - originals) / counts[:, None]
    else:
        index = names.index(variant["name"])
        values = sums["totals"][index, langs, layer] / sums["counts"][index, langs, None]
    return np.asarray(values, dtype=np.float32)


def run(protocol: dict, output: Path, backend=None) -> None:
    validate_protocol(protocol)
    cfg = protocol["config"]
    with locked(output):
        backend = backend or Backend.load(cfg)
        identity = json.loads(json.dumps(backend.identity()))
        manifest = {"protocol_sha256": protocol["sha256"], "implementation_sha256": implementation_digest(),
                    "model_identity": identity, "n_layers": backend.n_layers, "n_groups": backend.n_groups,
                    "n_recipient_samples": len(protocol["recipient_rows"]), "n_eval_samples": len(protocol["expanded_samples"]),
                    "variants": [v["name"] for v in protocol["variants"]]}
        fingerprint = digest(manifest)
        manifest["fingerprint"] = fingerprint
        manifest_path = output / "run_manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError("Existing run differs in model, protocol, or code; use a new output directory.")
        else:
            if any(p.name not in {".lock", "protocol.json"} for p in output.iterdir()):
                raise ValueError("Output contains unrelated files without a run manifest.")
            atomic_write_json(manifest_path, manifest)
        protocol_path = output / "protocol.json"
        if protocol_path.exists() and json.loads(protocol_path.read_text())["sha256"] != protocol["sha256"]:
            raise ValueError("Output protocol does not match the requested run.")
        atomic_write_json(protocol_path, protocol)
        (output / "_SUCCESS").unlink(missing_ok=True)
        sums, banks = prepare_donors(protocol, backend, output, fingerprint)
        expanded = protocol["expanded_samples"]
        n_base = len(protocol["recipient_rows"])
        group_ids = np.asarray(protocol["eval_to_base"], dtype=np.int64)
        targets = np.array([s["correct_idx"] for s in expanded], dtype=np.int64)
        batch_size = cfg["patching"]["batch_size"]
        baseline_path = output / "baseline.npz"
        if baseline_path.exists():
            baseline = read_npz(baseline_path, fingerprint)["option_logits"]
            if baseline.shape != (len(expanded), backend.n_options):
                raise ValueError("Baseline checkpoint has incorrect shape.")
        else:
            baseline = np.concatenate([backend.logits([s["prompt"] for s in expanded[start:start + batch_size]])
                                       for start in range(0, len(expanded), batch_size)])
            atomic_save_npz(baseline_path, fingerprint=np.array(fingerprint), option_logits=baseline,
                            target_indices=targets, eval_to_base=group_ids)
        base_mean = _margin_vs_incorrect_mean(baseline, targets)
        base_max = _margin_vs_incorrect_max(baseline, targets)
        base_rank = _rank_from_desc(baseline)[np.arange(len(expanded)), targets]
        sample_meta = [{"sample_index": i, "sample_id": protocol["samples"][index]["sample_id"],
                        "parallel_id": protocol["samples"][index]["parallel_id"],
                        "language": protocol["samples"][index]["language"],
                        "correct_idx": protocol["samples"][index]["correct_idx"]}
                       for i, index in enumerate(protocol["recipient_rows"])]
        for variant in protocol["variants"]:
            checkpoint = output / "checkpoints" / variant["name"]
            checkpoint.mkdir(parents=True, exist_ok=True)
            importance = {key: np.empty((n_base, backend.n_layers, backend.n_groups), dtype=np.float32) for key in METRICS}
            for layer in range(backend.n_layers):
                donors = donor_vectors(protocol, variant, sums, banks[layer], layer)
                for group in range(backend.n_groups):
                    unit = checkpoint / f"layer_{layer:03d}_group_{group:03d}.npz"
                    if unit.exists():
                        scores = read_npz(unit, fingerprint)
                        if set(scores) != set(METRICS) or any(a.shape != (n_base,) for a in scores.values()):
                            raise ValueError(f"Invalid unit checkpoint {unit}")
                    else:
                        patched = np.concatenate([
                            backend.patched_logits([s["prompt"] for s in expanded[start:start + batch_size]], layer, group,
                                                   donors[group_ids[start:start + batch_size]])
                            for start in range(0, len(expanded), batch_size)])
                        rank = _rank_from_desc(patched)[np.arange(len(expanded)), targets]
                        scores = {
                            METRICS[0]: _percent_drop(base_mean, _margin_vs_incorrect_mean(patched, targets)),
                            METRICS[1]: _percent_drop(base_max, _margin_vs_incorrect_max(patched, targets)),
                            METRICS[2]: (rank - base_rank).astype(np.float32),
                        }
                        scores = {key: _group_mean(value, group_ids, n_base) for key, value in scores.items()}
                        if not all(np.isfinite(value).all() for value in scores.values()):
                            raise ValueError("Non-finite importance score; checkpoint not written.")
                        atomic_save_npz(unit, fingerprint=np.array(fingerprint), **scores)
                    for key in METRICS:
                        importance[key][:, layer, group] = scores[key]
                print(f"{variant['name']}: layer {layer + 1}/{backend.n_layers} complete", flush=True)
            directory = output / "variants" / variant["name"]
            directory.mkdir(parents=True, exist_ok=True)
            arrays = dict(importance)
            arrays.update({f"flat_{key}": value.reshape(n_base, -1) for key, value in importance.items()})
            arrays.update(base_margin_mean=_group_mean(base_mean, group_ids, n_base),
                          base_margin_max=_group_mean(base_max, group_ids, n_base),
                          base_correct_rank=_group_mean(base_rank, group_ids, n_base))
            atomic_save_npz(directory / "importance_tensors.npz", **arrays)
            atomic_write_json(directory / "samples.json", sample_meta)
            atomic_write_json(directory / "metadata.json", {**manifest, "variant": variant,
                              "languages": protocol["languages"], "n_samples": n_base,
                              "n_parallel_questions": len(protocol["recipient_ids"]),
                              "n_patch_units": backend.n_groups, "patch_mode": "kv_group", "target_mode": "correct"})
        atomic_write_json(output / "_SUCCESS", {"fingerprint": fingerprint})
