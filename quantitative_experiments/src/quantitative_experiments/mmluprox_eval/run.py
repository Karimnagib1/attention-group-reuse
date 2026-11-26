from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .config import apply_overrides, load_config, parse_args
from .data import extract_options, load_mmluprox_dataset
from .distributed import DistInfo, all_reduce_sum, destroy_process_group, gather_object_to_rank0, init_distributed
from .metrics import accuracy
from .prompt import build_prompt, option_continuation, option_text_continuation
from .scorer import Scorer


def _to_builtin(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    return value


def _get_answer_index(example: Dict[str, Any], cfg: Dict[str, Any]) -> int:
    field = cfg["data"].get("answer_index_field", "answer_index")
    if field in example:
        offset = int(cfg["data"].get("answer_index_offset", 0))
        return int(example[field]) + offset
    label_field = cfg["data"].get("answer_field", "answer")
    label = str(example[label_field]).strip()
    labels = cfg["eval"]["option_labels"]
    if label in labels:
        return labels.index(label)
    raise ValueError(f"Unknown answer label: {label}")


def _build_continuations(labels: List[str], options: List[str], cfg: Dict[str, Any]) -> List[str]:
    source = cfg["eval"].get("continuation_source", "label")
    if source == "label":
        return [option_continuation(label, cfg) for label in labels]
    if source == "option_text":
        if len(options) < len(labels):
            raise ValueError(
                f"continuation_source=option_text requires at least {len(labels)} options, got {len(options)}."
            )
        return [option_text_continuation(options[i], cfg) for i in range(len(labels))]
    raise ValueError("eval.continuation_source must be one of: label, option_text")


def _score_details_payload(
    labels: List[str],
    options: List[str],
    score_details: List[Dict[str, float | int]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, detail in enumerate(score_details):
        out.append(
            {
                "label": labels[i],
                "text": options[i] if i < len(options) else None,
                "score": float(detail["score"]),
                "sum_logprob": float(detail["sum_logprob"]),
                "token_count": int(detail["token_count"]),
            }
        )
    return out


def evaluate(cfg: Dict[str, Any], dist: DistInfo) -> Dict[str, Any]:
    device = torch.device("cuda", dist.local_rank) if torch.cuda.is_available() else torch.device("cpu")
    scorer = Scorer(cfg, device)

    ds = load_mmluprox_dataset(cfg)
    option_fields = cfg["data"].get("option_fields", [f"option_{i}" for i in range(10)])
    language_field = cfg["data"].get("language_field", "language")
    languages = cfg["data"].get("languages")
    id_field = cfg["data"].get("id_field", "id")

    method = cfg["eval"].get("method", "loglikelihood")
    gen_batch_size = cfg["eval"].get("generate_batch_size", 1)

    sampler = None
    if dist.enabled:
        sampler = DistributedSampler(ds, num_replicas=dist.world_size, rank=dist.rank, shuffle=False)

    batch_size = cfg["eval"]["batch_size"]
    labels = cfg["eval"]["option_labels"]
    gen_max_new_tokens = cfg["eval"].get("generate_max_new_tokens", 2)
    gen_temperature = cfg["eval"].get("generate_temperature", 0.0)
    gen_do_sample = cfg["eval"].get("generate_do_sample", False)
    continuation_source = cfg["eval"].get("continuation_source", "label")
    length_penalty_alpha = float(cfg["eval"].get("length_penalty_alpha", 0.0))

    dl_batch_size = gen_batch_size if method == "generate" else 1
    dl = DataLoader(ds, batch_size=dl_batch_size, sampler=sampler)

    correct = 0
    total = 0
    per_lang = {}
    per_item_records: List[Dict[str, Any]] = []

    pbar = tqdm(dl, disable=dist.rank != 0, desc="Evaluating")
    for batch in pbar:
        batch_len = len(next(iter(batch.values())))
        examples = [{k: _to_builtin(v[i]) for k, v in batch.items()} for i in range(batch_len)]

        if method == "generate":
            prompts: List[str] = []
            lang_list: List[str | None] = []
            gold_list: List[int] = []
            id_list: List[Any] = []

            for ex in examples:
                lang = ex.get(language_field)
                if languages and lang not in languages:
                    continue
                options = extract_options(ex, option_fields)
                prompts.append(build_prompt(ex, options, cfg))
                lang_list.append(lang)
                gold_list.append(_get_answer_index(ex, cfg))
                id_list.append(ex.get(id_field))

            if prompts:
                preds = scorer.generate_choices(
                    prompts,
                    labels,
                    max_new_tokens=gen_max_new_tokens,
                    temperature=gen_temperature,
                    do_sample=gen_do_sample,
                    batch_size=gen_batch_size,
                )
                for pred_label, lang, gold_idx, qid in zip(preds, lang_list, gold_list, id_list):
                    pred_idx = labels.index(pred_label) if pred_label in labels else -1
                    is_correct = int(pred_idx == gold_idx)
                    correct += is_correct
                    total += 1

                    if lang is None:
                        lang = "unknown"
                    if lang not in per_lang:
                        per_lang[lang] = {"correct": 0, "total": 0}
                    per_lang[lang]["correct"] += is_correct
                    per_lang[lang]["total"] += 1

                    if cfg["run"].get("save_predictions", False):
                        per_item_records.append(
                            {
                                "id": qid,
                                "language": lang,
                                "pred": pred_label if pred_label is not None else "unknown",
                                "gold": labels[gold_idx],
                                "correct": bool(is_correct),
                            }
                        )
                    else:
                        per_item_records.append(
                            {
                                "id": qid,
                                "language": lang,
                                "correct": bool(is_correct),
                            }
                        )
        else:
            for example in examples:
                lang = example.get(language_field)
                if languages and lang not in languages:
                    continue

                options = extract_options(example, option_fields)
                prompt = build_prompt(example, options, cfg)
                continuations = _build_continuations(labels, options, cfg)
                items = [(prompt, continuation) for continuation in continuations]
                score_details = scorer.score_detailed(
                    items,
                    batch_size=batch_size,
                    length_penalty_alpha=length_penalty_alpha,
                )
                scores = [float(entry["score"]) for entry in score_details]
                pred_idx = max(range(len(scores)), key=scores.__getitem__)
                gold_idx = _get_answer_index(example, cfg)

                is_correct = int(pred_idx == gold_idx)
                correct += is_correct
                total += 1

                if lang is None:
                    lang = "unknown"
                if lang not in per_lang:
                    per_lang[lang] = {"correct": 0, "total": 0}
                per_lang[lang]["correct"] += is_correct
                per_lang[lang]["total"] += 1

                if cfg["run"].get("save_predictions", False):
                    record = {
                        "id": example.get(id_field),
                        "language": lang,
                        "pred": labels[pred_idx],
                        "gold": labels[gold_idx],
                        "correct": bool(is_correct),
                    }
                    if continuation_source == "option_text":
                        record.update(
                            {
                                "pred_text": options[pred_idx],
                                "gold_text": options[gold_idx],
                                "continuation_source": continuation_source,
                                "length_penalty_alpha": length_penalty_alpha,
                                "scores": _score_details_payload(labels, options, score_details),
                            }
                        )
                    per_item_records.append(record)
                else:
                    per_item_records.append(
                        {
                            "id": example.get(id_field),
                            "language": lang,
                            "correct": bool(is_correct),
                        }
                    )

        if dist.rank == 0 and total % cfg["run"].get("log_every", 100) == 0:
            pbar.set_postfix({"acc": correct / total})

    if dist.rank == 0:
        print("Evaluation loop complete. Releasing unnecessary VRAM before aggregation...")

    # Free model memory before any distributed aggregation work.
    del scorer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()

    if dist.rank == 0:
        print("Gathering records across ranks...")

    correct_t = torch.tensor(correct, device=device, dtype=torch.long)
    total_t = torch.tensor(total, device=device, dtype=torch.long)
    correct_t = all_reduce_sum(correct_t)
    total_t = all_reduce_sum(total_t)

    if dist.rank == 0:
        print("Reduction complete. Computing metrics...")

    metrics = accuracy(int(correct_t.item()), int(total_t.item()))

    if dist.rank == 0:
        print("Gathering per-item records...")
    gathered_records = gather_object_to_rank0(per_item_records)
    per_lang_metrics = None
    all_correct_ids = None

    if dist.rank == 0:
        print("Aggregating per-language metrics...")
        flat = [item for sub in gathered_records for item in sub]

        # Per-language metrics
        per_lang_metrics = {}
        for rec in flat:
            lang = rec.get("language", "unknown")
            if lang not in per_lang_metrics:
                per_lang_metrics[lang] = {"correct": 0, "total": 0}
            per_lang_metrics[lang]["total"] += 1
            if rec.get("correct"):
                per_lang_metrics[lang]["correct"] += 1

        for lang, stats in per_lang_metrics.items():
            stats["accuracy"] = 0.0 if stats["total"] == 0 else stats["correct"] / stats["total"]

        print("Computing all-correct IDs across languages...")
        # IDs correct in all languages
        if languages:
            by_id = {}
            for rec in flat:
                qid = rec.get("id")
                lang = rec.get("language")
                if qid is None or lang is None:
                    continue
                if qid not in by_id:
                    by_id[qid] = {}
                by_id[qid][lang] = bool(rec.get("correct"))

            all_correct_ids = []
            required = set(languages)
            for qid, lang_map in by_id.items():
                if required.issubset(lang_map.keys()) and all(lang_map[l] for l in required):
                    all_correct_ids.append(qid)

    return {
        "metrics": metrics,
        "per_language": per_lang_metrics,
        "all_correct_ids": all_correct_ids,
        "records": None if dist.rank != 0 else (flat if cfg["run"].get("save_predictions", False) else None),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.set)

    dist = init_distributed(master_port=cfg["run"].get("master_port"))
    try:
        if dist.rank == 0:
            print(
                "Run config: save_predictions="
                f"{cfg['run'].get('save_predictions', False)}, "
                f"output_dir={cfg['run'].get('output_dir')}"
            )
        results = evaluate(cfg, dist)

        if dist.rank == 0:
            print("Writing output files...")
            output_dir = Path(cfg["run"]["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Output dir (cwd={os.getcwd()}): {output_dir.resolve()}")

            save_predictions = bool(cfg["run"].get("save_predictions", False))
            if save_predictions and results.get("records") is None:
                print("Warning: save_predictions is True but no records were returned. Writing empty file.")
                results["records"] = []

            (output_dir / "metrics.json").write_text(json.dumps(results["metrics"], indent=2))
            if results["per_language"] is not None:
                (output_dir / "per_language.json").write_text(json.dumps(results["per_language"], indent=2))
            if results["all_correct_ids"] is not None:
                (output_dir / "all_correct_ids.json").write_text(json.dumps(results["all_correct_ids"], indent=2))
            if save_predictions and results.get("records") is not None:
                pred_path = output_dir / "predictions.jsonl"
                print(f"Writing predictions.jsonl ({len(results['records'])} records)...")
                with pred_path.open("w", encoding="utf-8") as f:
                    f.write("\n".join(json.dumps(r) for r in results["records"]) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                if pred_path.exists():
                    print(f"predictions.jsonl written: {pred_path} ({pred_path.stat().st_size} bytes)")
                else:
                    print("Warning: predictions.jsonl was not found after write.")

    finally:
        if dist.rank == 0:
            print("Destroying process group...")
        destroy_process_group()


if __name__ == "__main__":
    main()
