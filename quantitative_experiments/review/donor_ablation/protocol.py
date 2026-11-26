from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from . import QUANT_ROOT
from quantitative_experiments.multilingual_head_patch.run import (
    Sample, _build_parallel_samples, _deep_merge, _expand_samples_for_choice_order,
    apply_overrides, load_config,
)
from quantitative_experiments.multilingual_head_patch.checkpoint import atomic_write_json

STRATEGIES = {"language_mean", "leave_one_out", "disjoint_mean", "random_in_language"}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def seed_for(*parts) -> int:
    return int(digest(parts)[:16], 16)


def load_settings(path: Path, overrides=(), sample_count: int | None = None, output: Path | None = None) -> dict:
    path = path.resolve()
    custom = yaml.safe_load(path.read_text())
    base_path = (path.parent / custom.pop("base_config")).resolve()
    cfg = _deep_merge(load_config([str(base_path)]), custom)
    cfg = apply_overrides(cfg, overrides)
    if sample_count is not None:
        cfg["selection"]["samples_per_language"] = sample_count
    if output is not None:
        cfg["run"]["output_dir"] = str(output.resolve())
    for section, key in (("data", "allowed_parallel_ids_path"), ("run", "output_dir")):
        if cfg[section].get(key):
            cfg[section][key] = str((path.parent / cfg[section][key]).resolve())
    # Dataset limits would truncate the donor population, not just recipients.
    if cfg["data"].get("max_samples") is not None or cfg["data"].get("max_parallel_questions") is not None:
        raise ValueError("Use selection.samples_per_language; data limits would change the reference donor pool.")
    if cfg["run"].get("data_parallel", False):
        raise ValueError("Use a single device or device_map sharding; replicated DataParallel is not supported here.")
    if cfg["patching"].get("patch_mode") != "kv_group" or cfg["patching"].get("target_mode") != "correct":
        raise ValueError("This ablation fixes patch_mode=kv_group and target_mode=correct.")
    strategies = cfg["donors"]["strategies"]
    if len(strategies) != len(set(strategies)) or set(strategies) - STRATEGIES or "language_mean" not in strategies:
        raise ValueError("Choose unique known donor strategies, including the language_mean reference.")
    for section, key in (("selection", "samples_per_language"), ("patching", "batch_size"),
                         ("patching", "donor_batch_size"), ("analysis", "bootstrap_resamples"),
                         ("analysis", "jaccard_k")):
        if not isinstance(cfg[section][key], int) or cfg[section][key] < 1:
            raise ValueError(f"{section}.{key} must be a positive integer.")
    for key in ("pool_questions", "mean_repeats", "random_repeats"):
        if not isinstance(cfg["donors"][key], int) or cfg["donors"][key] < 1:
            raise ValueError(f"donors.{key} must be a positive integer.")
    return cfg


def choose_ids(ids: list[str], count: int, seed: int, role: str) -> list[str]:
    if count > len(ids):
        raise ValueError(f"Requested {count} {role} questions, but only {len(ids)} are available.")
    return sorted(sorted(ids, key=lambda q: (digest([seed, role, q]), q))[:count])


def build_protocol(cfg: dict, samples: list[Sample], languages: list[str]) -> dict:
    if not samples or len(languages) < 2 or len(set(languages)) != len(languages):
        raise ValueError("Need a nonempty balanced pool with at least two languages.")
    lookup = {(sample.parallel_id, sample.language): i for i, sample in enumerate(samples)}
    ids = sorted({sample.parallel_id for sample in samples})
    if len(lookup) != len(samples) or len(samples) != len(ids) * len(languages):
        raise ValueError("Reference pool must have exactly one sample per question/language.")
    if any((pid, lang) not in lookup for pid in ids for lang in languages):
        raise ValueError("Incomplete parallel question in reference pool.")
    if len(ids) < 2:
        raise ValueError("Need at least two questions for leave-one-out/random donors.")
    recipient_ids = choose_ids(ids, cfg["selection"]["samples_per_language"], cfg["selection"]["seed"], "recipient")
    recipient_rows = [lookup[(pid, lang)] for pid in recipient_ids for lang in languages]
    recipients = [samples[i] for i in recipient_rows]
    expanded, to_base, counts = _expand_samples_for_choice_order(recipients, cfg)
    variants = [{"name": "language_mean", "kind": "language_mean", "repeat": 0}]
    if "leave_one_out" in cfg["donors"]["strategies"]:
        variants.append({"name": "leave_one_out", "kind": "leave_one_out", "repeat": 0})
    if "disjoint_mean" in cfg["donors"]["strategies"]:
        remaining = sorted(set(ids) - set(recipient_ids))
        for repeat in range(cfg["donors"]["mean_repeats"]):
            pool = choose_ids(remaining, cfg["donors"]["pool_questions"], cfg["donors"]["seed"], f"disjoint-{repeat}")
            variants.append({"name": f"disjoint_mean_{repeat:02d}", "kind": "disjoint_mean", "repeat": repeat,
                             "parallel_ids": pool})
    if "random_in_language" in cfg["donors"]["strategies"]:
        for repeat in range(cfg["donors"]["random_repeats"]):
            donor_indices = []
            for sample in recipients:
                candidates = [pid for pid in ids if pid != sample.parallel_id]
                rng = np.random.default_rng(seed_for(cfg["donors"]["seed"], repeat, sample.parallel_id, sample.language))
                donor_id = candidates[int(rng.integers(len(candidates)))]
                donor_indices.append(lookup[(donor_id, sample.language)])
            variants.append({"name": f"random_in_language_{repeat:02d}", "kind": "random_in_language", "repeat": repeat,
                             "donor_indices": donor_indices})
    required = set(recipient_rows)
    for variant in variants:
        required.update(variant.get("donor_indices", []))
    payload = {"version": 1, "config": cfg, "languages": languages, "reference_ids": ids,
               "recipient_ids": recipient_ids, "recipient_rows": recipient_rows,
               "samples": [asdict(s) for s in samples], "expanded_samples": [asdict(s) for s in expanded],
               "eval_to_base": to_base.tolist(), "variants_per_base": counts.tolist(),
               "variants": variants, "bank_rows": sorted(required),
               "reference_definition": "Mean of original-order activations over the complete balanced eligible pool, including recipients.",
               "random_definition": "Different question in the same language, sampled independently per recipient/language/repeat and fixed across layers, groups and answer orders."}
    payload["sha256"] = digest(payload)
    validate_protocol(payload)
    return payload


def validate_protocol(payload: dict) -> None:
    if payload.get("sha256") != digest({k: v for k, v in payload.items() if k != "sha256"}):
        raise ValueError("Protocol fingerprint mismatch.")
    samples = payload["samples"]
    recipients = [samples[i] for i in payload["recipient_rows"]]
    for variant in payload["variants"]:
        if variant["kind"] == "disjoint_mean" and set(variant["parallel_ids"]) & set(payload["recipient_ids"]):
            raise ValueError("Disjoint donor pool overlaps recipients.")
        if variant["kind"] == "random_in_language":
            if len(variant["donor_indices"]) != len(recipients):
                raise ValueError("Missing random donor assignments.")
            for rec, index in zip(recipients, variant["donor_indices"]):
                donor = samples[index]
                if rec["language"] != donor["language"] or rec["parallel_id"] == donor["parallel_id"]:
                    raise ValueError("Random donor must be a different question in the recipient language.")


def prepare(cfg: dict, output: Path) -> dict:
    samples, languages = _build_parallel_samples(cfg)
    payload = build_protocol(cfg, samples, languages)
    if output.exists():
        previous = json.loads(output.read_text())
        validate_protocol(previous)
        if previous["sha256"] != payload["sha256"]:
            raise ValueError("Prepared protocol differs. Use a new output directory.")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output, payload)
    return payload


def load_protocol(path: Path, cfg: dict | None = None) -> dict:
    payload = json.loads(path.read_text())
    validate_protocol(payload)
    if cfg is not None and payload["config"] != cfg:
        raise ValueError("Configuration differs from the prepared protocol. Use a new output directory.")
    return payload


def implementation_digest() -> str:
    paths = list(Path(__file__).parent.glob("*.py"))
    base = QUANT_ROOT / "src/quantitative_experiments"
    paths += [base / "multilingual_head_patch/run.py", base / "multilingual_head_patch/checkpoint.py",
              base / "mmluprox_eval/data.py", base / "mmluprox_eval/prompt.py"]
    return digest({str(p.relative_to(QUANT_ROOT)): p.read_text() for p in sorted(paths)})
