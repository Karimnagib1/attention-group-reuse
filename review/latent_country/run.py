"""CLI: prepare -> [localize] -> evaluate -> summarize.

Every expensive stage resumes only against identical config, model, protocol,
and implementation fingerprints. All default artifacts live under review/.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import fcntl
import json
from pathlib import Path

import numpy as np

from .protocol import ROOT, build_protocol, digest, load_protocol, read_config, write_json
from .statistics import random_sites, summarize_records


@contextmanager
def output_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is writing {directory}.") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def implementation_digest() -> str:
    return digest({p.name: p.read_text() for p in sorted(Path(__file__).parent.glob("*.py"))})


def establish_run(output: Path, manifest: dict) -> str:
    path = output / "run_manifest.json"
    fingerprint = digest(manifest)
    payload = {**manifest, "fingerprint": fingerprint}
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError("Existing run differs in config, model, code, selection or prompts. Use a new output directory.")
    else:
        if any(output.iterdir()) and any(p.name != ".lock" for p in output.iterdir()):
            raise ValueError("Output directory contains files without a matching run manifest.")
        write_json(path, payload)
    (output / "records").mkdir(exist_ok=True)
    return fingerprint


def pending_rows(output: Path, rows: list[dict], fingerprint: str) -> list[dict]:
    expected = {r["id"]: r for r in rows}
    existing = {}
    for path in (output / "records").glob("*.json"):
        record = json.loads(path.read_text())
        if record.get("id") not in expected or path.stem != record["id"]:
            raise ValueError(f"Unexpected record {path}.")
        if record.get("run_fingerprint") != fingerprint or record.get("prompt_sha256") != digest(expected[record["id"]]):
            raise ValueError(f"Record belongs to a different run: {path}.")
        if record["id"] in existing:
            raise ValueError("Duplicate saved record.")
        existing[record["id"]] = record
    return [row for row in rows if row["id"] not in existing]


def read_records(output: Path, manifest: dict) -> list[dict]:
    records = [json.loads(p.read_text()) for p in sorted((output / "records").glob("*.json"))]
    if {r["id"] for r in records} != set(manifest["expected_ids"]) or len(records) != len(manifest["expected_ids"]):
        raise ValueError("Run is incomplete; resume its original command before summarizing.")
    if any(r["run_fingerprint"] != manifest["fingerprint"] for r in records):
        raise ValueError("Records do not match the run manifest.")
    return records


def selection_sites(cfg: dict, protocol: dict, backend, selection_path: Path | None) -> tuple[list, dict]:
    from .backend import validate_sites
    if selection_path is None:
        sites = cfg["selection"]["published_sites"]
        source = {"mode": "published_fixed", "description": "Frozen published group set; new held-out prompts."}
    else:
        source = json.loads(selection_path.read_text())
        if digest({k: v for k, v in source.items() if k != "sha256"}) != source.get("sha256"):
            raise ValueError("Selection content does not match its fingerprint.")
        if source["protocol_sha256"] != protocol["sha256"] or source["model"] != cfg["model"]:
            raise ValueError("Selection is for a different protocol or model.")
        if source["model_identity"] != backend.identity():
            raise ValueError("Selection and evaluation model identities differ.")
        if source["mode"] != "english_localization":
            raise ValueError("Selection was not produced by the English-only localizer.")
        sites = source["sites"]
    return validate_sites(sites, backend.n_layers, backend.n_groups), source


def execute(cfg: dict, protocol: dict, output: Path, stage: str,
            selection_path: Path | None = None, backend=None) -> None:
    from .backend import Backend
    if protocol["settings"] != cfg["protocol"]:
        raise ValueError("Config protocol settings differ from the prepared manifest.")
    backend = backend or Backend.load(cfg)
    if stage == "localize":
        rows = protocol["localization"]
        sites = [[layer, group] for layer in range(backend.n_layers) for group in range(backend.n_groups)]
        if not 0 < cfg["selection"]["count"] <= len(sites):
            raise ValueError("selection.count is outside the model's site count.")
        patch_sets, source = [[site] for site in sites], {"mode": "english_localization"}
        controls = []
    else:
        rows = protocol["evaluation"]
        sites, source = selection_sites(cfg, protocol, backend, selection_path)
        controls = random_sites(sites, backend.n_groups, cfg["evaluation"]["random_draws"],
                                cfg["evaluation"]["random_seed"])
        patch_sets = [sites] + controls
    manifest = {"stage": stage, "config": cfg, "protocol_sha256": protocol["sha256"],
                "implementation_sha256": implementation_digest(), "model_identity": backend.identity(),
                "selection": source, "sites": sites, "random_sites": controls,
                "expected_ids": [row["id"] for row in rows]}
    with output_lock(output):
        fingerprint = establish_run(output, manifest)
        pending = pending_rows(output, rows, fingerprint)
        if pending:
            # A previously completed run may have lost a record. Never leave a
            # stale completion marker visible while repairing an incomplete run.
            (output / "_SUCCESS").unlink(missing_ok=True)
        print(f"{stage}: {len(rows) - len(pending)}/{len(rows)} records already complete", flush=True)
        batch_size = cfg["inference"]["batch_size"]
        layers = sorted({layer for patch_set in patch_sets for layer, _ in patch_set})
        # Keep conditions together for interpretable progress and comparable lengths.
        for condition in sorted({row["condition"] for row in pending}):
            condition_rows = [row for row in pending if row["condition"] == condition]
            for start in range(0, len(condition_rows), batch_size):
                batch = condition_rows[start:start + batch_size]
                original, target = backend.token_ids(batch)
                encoded = backend.encode([row["recipient_prompt"] for row in batch])
                donors = backend.capture([row["donor_prompt"] for row in batch], layers)
                baseline = backend.score(encoded, original, target)
                patches = [backend.score(encoded, original, target, donors, patch_set) for patch_set in patch_sets]
                for index, row in enumerate(batch):
                    result = {k: row[k] for k in ("id", "condition", "pair_index", "donor_language", "recipient_language", "donor_source")}
                    result.update(run_fingerprint=fingerprint, prompt_sha256=digest(row),
                                  baseline=baseline[index], original_token_id=original[index], target_token_id=target[index],
                                  answer_token_collision=original[index] == target[index])
                    if stage == "localize":
                        clean = baseline[index]["original_minus_target_logit"]
                        denominator = clean if abs(clean) > 1e-8 else 1e-8
                        result["normalized_effects"] = [(clean - patch[index]["original_minus_target_logit"]) / denominator
                                                        for patch in patches]
                        result["near_zero_baseline_margin"] = abs(clean) <= 1e-8
                    else:
                        result.update(selected=patches[0][index], random=[patch[index] for patch in patches[1:]])
                    write_json(output / "records" / f"{row['id']}.json", result)
                print(f"{stage}: {condition} {min(start + batch_size, len(condition_rows))}/{len(condition_rows)} pending records saved", flush=True)
        saved_manifest = json.loads((output / "run_manifest.json").read_text())
        records = read_records(output, saved_manifest)
        if stage == "localize":
            means = np.asarray([r["normalized_effects"] for r in records], dtype=np.float64).mean(axis=0)
            if not np.isfinite(means).all():
                raise ValueError("Non-finite localization score.")
            if all(r["answer_token_collision"] for r in records):
                raise ValueError("All localization pairs have identical answer tokens; selection is uninformative.")
            order = sorted(range(len(sites)), key=lambda i: (-means[i], sites[i][0], sites[i][1]))
            payload = {"mode": "english_localization", "model": cfg["model"], "model_identity": backend.identity(),
                       "protocol_sha256": protocol["sha256"], "run_fingerprint": fingerprint,
                       "sites": [sites[i] for i in order[:cfg["selection"]["count"]]],
                       "selection_rule": "Top k mean signed normalized localization effects, pooled equally over English prompt pairs and donor families; ties by layer/group.",
                       "score_formula": "(clean_margin - patched_margin) / clean_margin; denominator replaced by +1e-8 only when abs(clean_margin) <= 1e-8, matching the original sweep.",
                       "ranking": [{"site": sites[i], "score": float(means[i])} for i in order],
                       "n_pairs": len(records), "answer_token_collisions": sum(r["answer_token_collision"] for r in records),
                       "near_zero_baseline_margins": sum(r["near_zero_baseline_margin"] for r in records)}
            payload["sha256"] = digest(payload)
            write_json(output / "selection.json", payload)
        else:
            write_summary(output, saved_manifest, records)
        write_json(output / "_SUCCESS", {"run_fingerprint": fingerprint, "n_records": len(records)})


def write_summary(output: Path, manifest: dict, records: list[dict]) -> None:
    settings = manifest["config"]["evaluation"]
    summary = summarize_records(records, settings["bootstrap_resamples"], settings["bootstrap_seed"])
    summary["run_fingerprint"] = manifest["fingerprint"]
    write_json(output / "summary.json", summary)
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "n_conditions", "n_pairs_per_condition", "metric", "estimate", "ci95_low", "ci95_high"])
        for condition, group in summary["groups"].items():
            for metric, value in group["metrics"].items():
                writer.writerow([condition, group["n_conditions"], group["n_pairs_per_condition"],
                                 metric, value["estimate"], *value["ci95"]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    default_config = ROOT / "review/configs/latent_country_qwen.json"
    for command in ("prepare", "localize", "evaluate"):
        p = sub.add_parser(command)
        p.add_argument("--config", type=Path, default=default_config)
        p.add_argument("--output", type=Path, required=True, help="New artifact file (prepare) or run directory.")
        if command == "prepare":
            p.add_argument("--data-dir", type=Path, default=ROOT / "data")
        else:
            p.add_argument("--protocol", type=Path, required=True)
        if command == "evaluate":
            p.add_argument("--selection", type=Path, help="Frozen selection.json from English-only localization; omit for published groups.")
    p = sub.add_parser("summarize")
    p.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "summarize":
        with output_lock(args.run_dir):
            manifest = json.loads((args.run_dir / "run_manifest.json").read_text())
            if manifest["stage"] != "evaluate":
                raise ValueError("Only evaluation runs have steering summaries.")
            write_summary(args.run_dir, manifest, read_records(args.run_dir, manifest))
        return
    cfg = read_config(args.config)
    if args.command == "prepare":
        payload = build_protocol(cfg["protocol"], args.data_dir)
        if args.output.exists():
            if load_protocol(args.output)["sha256"] != payload["sha256"]:
                raise ValueError("Existing protocol differs; choose a new output path.")
        else:
            write_json(args.output, payload)
        print(f"Prepared {len(payload['localization'])} English localization and {len(payload['evaluation'])} evaluation records at {args.output}")
        print(f"Protocol SHA256: {payload['sha256']}")
    else:
        execute(cfg, load_protocol(args.protocol), args.output, args.command, getattr(args, "selection", None))


if __name__ == "__main__":
    main()
