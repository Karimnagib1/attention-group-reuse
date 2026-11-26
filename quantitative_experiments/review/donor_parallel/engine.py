from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from donor_ablation.backend import Backend
from donor_ablation.engine import METRICS, donor_vectors, prepare_donors, read_npz
from donor_ablation.protocol import digest, implementation_digest, validate_protocol
from quantitative_experiments.multilingual_head_patch.checkpoint import atomic_save_npz, atomic_write_json
from quantitative_experiments.multilingual_head_patch.run import (
    _group_mean, _margin_vs_incorrect_mean, _margin_vs_incorrect_max, _percent_drop, _rank_from_desc,
)


def code_digest():
    return digest({"reused_code": implementation_digest(),
                   "parallel_code": {p.name: p.read_text() for p in sorted(Path(__file__).parent.glob("*.py"))}})


def load_backend(cfg):
    """The launcher masks the device before starting this fresh Python process."""
    import os
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Each worker must see exactly one functioning GPU. Check CUDA_VISIBLE_DEVICES and the node.")
    torch.cuda.set_device(0)
    torch.empty(1, device="cuda")  # Fail before loading weights if CUDA cannot initialize.
    if cfg["model"]["dtype"] == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support BF16; use --dtype float32 in a new output directory.")
    print(f"PID {os.getpid()}; visible GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}; "
          f"{torch.cuda.get_device_name(0)}; dtype {cfg['model']['dtype']}; "
          f"batch {cfg['patching']['batch_size']}; donor batch {cfg['patching']['donor_batch_size']}", flush=True)
    backend = Backend.load(cfg)
    parameters = list(backend.runner.base_model.parameters())
    expected_dtype = getattr(torch, cfg["model"]["dtype"])
    if any(p.device != torch.device("cuda:0") or p.dtype != expected_dtype for p in parameters):
        raise RuntimeError("Model must reside entirely on the worker's single GPU with the requested dtype.")
    print(f"Model ready; allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
    return backend


def make_manifest(protocol, backend):
    value = {"runner": "donor_parallel_v1", "protocol_sha256": protocol["sha256"],
             "implementation_sha256": code_digest(),
             "model_identity": json.loads(json.dumps(backend.identity())),
             "n_layers": backend.n_layers, "n_groups": backend.n_groups,
             "n_options": backend.n_options, "width": backend.width,
             "n_recipient_samples": len(protocol["recipient_rows"]),
             "n_eval_samples": len(protocol["expanded_samples"]),
             "variants": [v["name"] for v in protocol["variants"]]}
    value["fingerprint"] = digest(value)
    return value


def load_manifest(output, protocol):
    manifest = json.loads((output / "run_manifest.json").read_text())
    if manifest.get("runner") != "donor_parallel_v1":
        raise ValueError("Use a separate output directory for the parallel runner.")
    if manifest["fingerprint"] != digest({k: v for k, v in manifest.items() if k != "fingerprint"}):
        raise ValueError("Manifest fingerprint mismatch.")
    if manifest["protocol_sha256"] != protocol["sha256"] or manifest["implementation_sha256"] != code_digest():
        raise ValueError("Protocol or implementation changed; use a new output directory.")
    return manifest


def indices(protocol):
    return (np.array([s["correct_idx"] for s in protocol["expanded_samples"]], dtype=np.int64),
            np.asarray(protocol["eval_to_base"], dtype=np.int64))


def load_baseline(output, protocol, manifest):
    saved = read_npz(output / "baseline.npz", manifest["fingerprint"])
    targets, groups = indices(protocol)
    if (saved["option_logits"].shape != (len(targets), manifest["n_options"])
            or not np.array_equal(saved["target_indices"], targets)
            or not np.array_equal(saved["eval_to_base"], groups)):
        raise ValueError("Baseline shape or sample mapping differs from the protocol.")
    return saved["option_logits"]


def setup(protocol, output, backend=None):
    """One GPU captures donors and clean predictions once, before patch workers start."""
    validate_protocol(protocol)
    backend = backend or load_backend(protocol["config"])
    manifest = make_manifest(protocol, backend)
    path = output / "run_manifest.json"
    if path.exists():
        if load_manifest(output, protocol) != manifest:
            raise ValueError("Saved model identity differs; use a new output directory.")
    else:
        atomic_write_json(path, manifest)
    prepare_donors(protocol, backend, output, manifest["fingerprint"])
    if not (output / "baseline.npz").exists():
        samples = protocol["expanded_samples"]
        batch = protocol["config"]["patching"]["batch_size"]
        values = []
        for start in range(0, len(samples), batch):
            values.append(backend.logits([s["prompt"] for s in samples[start:start + batch]]))
        targets, groups = indices(protocol)
        atomic_save_npz(output / "baseline.npz", fingerprint=np.array(manifest["fingerprint"]),
                        option_logits=np.concatenate(values), target_indices=targets, eval_to_base=groups)
    load_baseline(output, protocol, manifest)
    atomic_write_json(output / "_PREPARED.json", {"fingerprint": manifest["fingerprint"]})
    print("Shared donor cache and clean predictions ready.", flush=True)
    return manifest


def tasks(protocol, manifest):
    return [(v, layer, group) for v in protocol["variants"]
            for layer in range(manifest["n_layers"]) for group in range(manifest["n_groups"])]


def unit_path(output, variant, layer, group):
    return output / "checkpoints" / variant["name"] / f"layer_{layer:03d}_group_{group:03d}.npz"


def read_unit(path, manifest):
    scores = read_npz(path, manifest["fingerprint"])
    if set(scores) != set(METRICS) or any(v.shape != (manifest["n_recipient_samples"],) for v in scores.values()):
        raise ValueError(f"Invalid unit checkpoint: {path}")
    return scores


def work(protocol, output, rank, workers, backend=None, limit=None):
    """Static round-robin assignment: no overlapping writes between workers."""
    if not 0 <= rank < workers:
        raise ValueError("Invalid worker index/count.")
    manifest = load_manifest(output, protocol)
    marker = json.loads((output / "_PREPARED.json").read_text())
    if marker != {"fingerprint": manifest["fingerprint"]}:
        raise ValueError("Shared preparation is incomplete or mismatched.")
    assigned = tasks(protocol, manifest)[rank::workers]
    pending = []
    for task in assigned:
        path = unit_path(output, *task)
        if path.exists():
            read_unit(path, manifest)
        else:
            pending.append(task)
    print(f"Worker {rank}: {len(assigned) - len(pending)} saved; {len(pending)} units pending", flush=True)
    if not pending:
        return 0
    if limit is not None:
        if limit < 1:
            raise ValueError("Unit limit must be positive.")
        pending = pending[:limit]
    backend = backend or load_backend(protocol["config"])
    if make_manifest(protocol, backend) != manifest:
        raise ValueError("Worker model differs from shared donor/baseline model.")
    if not (output / "donor_cache/_READY.json").exists():
        raise ValueError("Shared donor cache is missing; workers may not regenerate it concurrently.")
    sums, banks = prepare_donors(protocol, backend, output, manifest["fingerprint"])
    baseline = load_baseline(output, protocol, manifest)
    targets, groups = indices(protocol)
    base_mean = _margin_vs_incorrect_mean(baseline, targets)
    base_max = _margin_vs_incorrect_max(baseline, targets)
    base_rank = _rank_from_desc(baseline)[np.arange(len(targets)), targets]
    samples = protocol["expanded_samples"]
    batch = protocol["config"]["patching"]["batch_size"]
    # Identical token batches are used for every group, preserving sample order.
    prompt_batches = [[s["prompt"] for s in samples[start:start + batch]] for start in range(0, len(samples), batch)]
    previous = None
    for done, (variant, layer, group) in enumerate(pending, 1):
        started = time.monotonic()
        if previous != (variant["name"], layer):
            donors = donor_vectors(protocol, variant, sums, banks[layer], layer)
            previous = (variant["name"], layer)
        patched = np.concatenate([backend.patched_logits(prompts, layer, group, donors[groups[start:start + batch]])
                                  for start, prompts in zip(range(0, len(samples), batch), prompt_batches)])
        if patched.shape != baseline.shape or not np.isfinite(patched).all():
            raise ValueError("Invalid patched logits.")
        rank_values = _rank_from_desc(patched)[np.arange(len(targets)), targets]
        scores = {METRICS[0]: _percent_drop(base_mean, _margin_vs_incorrect_mean(patched, targets)),
                  METRICS[1]: _percent_drop(base_max, _margin_vs_incorrect_max(patched, targets)),
                  METRICS[2]: (rank_values - base_rank).astype(np.float32)}
        scores = {k: _group_mean(v, groups, manifest["n_recipient_samples"]) for k, v in scores.items()}
        if not all(np.isfinite(v).all() for v in scores.values()):
            raise ValueError("Non-finite importance scores; checkpoint not written.")
        atomic_save_npz(unit_path(output, variant, layer, group),
                        fingerprint=np.array(manifest["fingerprint"]), **scores)
        print(f"Worker {rank}: {done}/{len(pending)}; {variant['name']} layer {layer + 1} "
              f"group {group + 1}; {time.monotonic() - started:.1f}s", flush=True)
    return len(pending)


def finalize(protocol, output):
    """Require every checkpoint, then export the existing analysis format."""
    manifest = load_manifest(output, protocol)
    # Validate the complete set before producing any new success marker.
    for task in tasks(protocol, manifest):
        read_unit(unit_path(output, *task), manifest)
    baseline = load_baseline(output, protocol, manifest)
    targets, groups = indices(protocol)
    n = manifest["n_recipient_samples"]
    rank_values = _rank_from_desc(baseline)[np.arange(len(targets)), targets]
    base_arrays = {"base_margin_mean": _group_mean(_margin_vs_incorrect_mean(baseline, targets), groups, n),
                   "base_margin_max": _group_mean(_margin_vs_incorrect_max(baseline, targets), groups, n),
                   "base_correct_rank": _group_mean(rank_values, groups, n)}
    sample_meta = [{"sample_index": i, "sample_id": protocol["samples"][index]["sample_id"],
                    "parallel_id": protocol["samples"][index]["parallel_id"],
                    "language": protocol["samples"][index]["language"],
                    "correct_idx": protocol["samples"][index]["correct_idx"]}
                   for i, index in enumerate(protocol["recipient_rows"])]
    for variant in protocol["variants"]:
        arrays = {k: np.empty((n, manifest["n_layers"], manifest["n_groups"]), dtype=np.float32) for k in METRICS}
        for layer in range(manifest["n_layers"]):
            for group in range(manifest["n_groups"]):
                scores = read_unit(unit_path(output, variant, layer, group), manifest)
                for key in METRICS:
                    arrays[key][:, layer, group] = scores[key]
        arrays.update({f"flat_{key}": arrays[key].reshape(n, -1) for key in METRICS})
        arrays.update(base_arrays)
        directory = output / "variants" / variant["name"]
        atomic_save_npz(directory / "importance_tensors.npz", **arrays)
        atomic_write_json(directory / "samples.json", sample_meta)
        atomic_write_json(directory / "metadata.json", {**manifest, "variant": variant,
                          "languages": protocol["languages"], "n_samples": n,
                          "n_parallel_questions": len(protocol["recipient_ids"]),
                          "n_patch_units": manifest["n_groups"], "patch_mode": "kv_group", "target_mode": "correct"})
    atomic_write_json(output / "_SUCCESS", {"fingerprint": manifest["fingerprint"]})
