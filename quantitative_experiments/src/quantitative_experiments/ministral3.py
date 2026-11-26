"""Ministral-only adapters for the existing accuracy evaluation and GQA sweep.

Run with ``python -m quantitative_experiments.ministral3 {evaluate,patch}``.
The legacy modules are only rebound inside this process, then restored.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from quantitative_experiments.mmluprox_eval import run as evaluation
from quantitative_experiments.mmluprox_eval.config import apply_overrides, load_config
from quantitative_experiments.mmluprox_eval.distributed import DistInfo
from quantitative_experiments.mmluprox_eval.scorer import Scorer
from quantitative_experiments.multilingual_head_patch import run as sweeping


class MinistralInputs:
    """Shared model loading and identical prompt encoding for both experiments."""

    def _load(self, cfg, device):
        from transformers import AutoConfig

        try:
            from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
            from transformers.models.ministral3 import Ministral3Config  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Install requirements-ministral3.txt in the active Python 3.11 environment."
            ) from exc

        self.cfg = cfg
        run = cfg.get("run", {})
        replicated = bool(run.get("data_parallel", False))
        if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
            raise ValueError("Launch one process; the Ministral entry point starts its own replicas.")
        if replicated and not dist.is_initialized():
            raise ValueError("Start replicated runs through the Ministral entry point.")
        self.device = torch.device(device if replicated else run.get("device", str(device)))
        model_cfg = cfg["model"]
        model_id = model_cfg["name_or_path"]
        source = {"revision": model_cfg.get("revision"), "cache_dir": model_cfg.get("cache_dir")}
        config = AutoConfig.from_pretrained(model_id, **source)
        if config.model_type != "mistral3" or config.text_config.model_type != "ministral3":
            raise ValueError("This adapter requires a Ministral 3 checkpoint in Transformers format.")
        if getattr(config, "quantization_config", None):
            raise ValueError("Use the -BF16 checkpoint for these intervention experiments.")

        self.tokenizer = MistralCommonBackend.from_pretrained(model_id, **source)
        self.tokenizer.padding_side = "left"
        self.option_labels = cfg["eval"]["option_labels"]
        self.answer_sep = cfg["eval"].get("answer_sep", " ")
        self.max_input_length = cfg["eval"].get("max_input_length")
        self.option_token_ids = sweeping.HeadPatchingRunner._prepare_option_token_ids(self)

        dtype_name = model_cfg.get("dtype", "bfloat16")
        dtype = "auto" if dtype_name == "auto" else getattr(torch, dtype_name)
        kwargs = dict(source, config=config, dtype=dtype,
                      attn_implementation=model_cfg.get("attn_implementation", "sdpa"))
        # Each replica worker owns one full model. Never shard it onto another
        # worker's GPU, even when the base config still contains device_map=auto.
        device_map = {"": str(self.device)} if replicated else run.get("device_map")
        if device_map is not None:
            kwargs["device_map"] = device_map
            if run.get("max_memory") and not replicated:
                kwargs["max_memory"] = sweeping.HeadPatchingRunner._normalize_max_memory(
                    self, run["max_memory"]
                )
        self.model = Mistral3ForConditionalGeneration.from_pretrained(model_id, **kwargs)
        if device_map is None:
            self.model.to(self.device)
        self.model.eval()
        self.device = self.model.get_input_embeddings().weight.device
        self.base_model = self.model
        self.uses_model_sharding = device_map is not None and not replicated
        self.uses_data_parallel = False
        print(f"Ministral device map: {getattr(self.model, 'hf_device_map', {'': str(self.device)})}", flush=True)

    def _encode_prompt(self, prompt):
        mode = self.cfg["eval"].get("prompt_format", "raw")
        if mode == "raw":
            ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
            if self.tokenizer.bos_token_id is not None:
                ids = [self.tokenizer.bos_token_id] + ids
        elif mode == "chat":
            ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True, return_dict=False
            )
        else:
            raise ValueError("eval.prompt_format must be raw or chat.")
        if self.max_input_length:
            ids = ids[-self.max_input_length:]
        return torch.tensor(ids, dtype=torch.long)

    def _encode_prompts(self, prompts):
        encoded = [self._encode_prompt(prompt).tolist() for prompt in prompts]
        batch = self.tokenizer.pad(
            {"input_ids": encoded}, padding=True, return_attention_mask=True, return_tensors="pt"
        )
        return batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device)


class MinistralScorer(MinistralInputs, Scorer):
    def __init__(self, cfg, device):
        if cfg["eval"].get("method", "generate") != "generate":
            raise ValueError("This baseline uses eval.method=generate, matching the paper.")
        self._load(cfg, device)

    @torch.no_grad()
    def generate_choices(self, prompts, labels, max_new_tokens, temperature, do_sample, batch_size):
        # Anchor to a standalone label: the A in 'Answer' must not count as A.
        pattern = re.compile(r"^\s*(" + "|".join(map(re.escape, labels)) + r")(?!\w)")
        predictions = []
        for start in range(0, len(prompts), batch_size):
            input_ids, mask = self._encode_prompts(prompts[start:start + batch_size])
            kwargs = {"temperature": temperature} if do_sample else {}
            generated = self.model.generate(
                input_ids=input_ids, attention_mask=mask, max_new_tokens=max_new_tokens,
                do_sample=do_sample, pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id, **kwargs,
            )
            for tokens in generated[:, input_ids.shape[1]:]:
                match = pattern.match(self.tokenizer.decode(tokens, skip_special_tokens=True))
                predictions.append(match.group(1) if match else None)
        return predictions


class MinistralRunner(MinistralInputs, sweeping.HeadPatchingRunner):
    def __init__(self, cfg):
        device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        self._load(cfg, device)
        # The shared sweep exports both clean and patched logits to NumPy,
        # which cannot accept BF16. Upcast only the small final-token logits;
        # model weights and attention activations keep the configured dtype.
        self.model.lm_head.register_forward_hook(lambda module, inputs, logits: logits.float())
        text = self.model.config.text_config
        self.layers = self.model.model.language_model.layers
        self.n_layers = len(self.layers)
        self.n_heads = int(text.num_attention_heads)
        self.n_kv_heads = int(text.num_key_value_heads)
        self.d_model = int(text.hidden_size)
        self.d_head = int(text.head_dim)
        if self.n_heads % self.n_kv_heads:
            raise ValueError("Query heads must divide evenly into KV groups.")
        for layer in self.layers:
            if layer.self_attn.o_proj.in_features != self.n_heads * self.d_head:
                raise ValueError("Unexpected o_proj input shape; refusing to patch incorrect slices.")


class ReplicaSampler(torch.utils.data.Sampler):
    """Split rows across workers without DistributedSampler's duplicate padding."""

    def __init__(self, dataset, num_replicas, rank, shuffle=False):
        if shuffle or num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError("Expected an unshuffled evaluation partition and a valid worker rank.")
        self.indices = range(rank, len(dataset), num_replicas)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def _evaluate(cfg, dist_info):
    # Both replacements are scoped to this Ministral process. The shared
    # evaluator already aggregates counts and predictions across workers.
    with patch.object(evaluation, "Scorer", MinistralScorer), \
         patch.object(evaluation, "DistributedSampler", ReplicaSampler):
        return evaluation.evaluate(cfg, dist_info)


def _evaluation_worker(rank, cfg, world_size, rendezvous, result_path, device_type="cuda"):
    """One full model per process; CPU/Gloo is also usable by offline tests."""
    if device_type == "cuda":
        torch.cuda.set_device(rank)
    try:
        dist.init_process_group(
            backend="nccl" if device_type == "cuda" else "gloo",
            init_method=rendezvous, rank=rank, world_size=world_size,
            # A worker can reach aggregation much earlier than another worker.
            timeout=timedelta(hours=24),
        )
        result = _evaluate(cfg, DistInfo(True, world_size, rank, rank))
        if rank == 0:
            Path(result_path).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _evaluate_replicated(cfg):
    world_size = torch.cuda.device_count()
    if not torch.cuda.is_available() or world_size == 0:
        raise ValueError("Replicated evaluation requires visible CUDA GPUs.")
    print(f"Replicated evaluation: {world_size} GPUs, one model per GPU, "
          f"batch size {cfg['eval'].get('generate_batch_size', 1)} per GPU.", flush=True)
    # Unique rendezvous and result files prevent collisions between jobs.
    # Only the parent writes public results after every worker succeeds.
    with TemporaryDirectory(prefix="ministral-replicas-") as temporary:
        root = Path(temporary)
        result_path = root / "result.json"
        mp.spawn(
            _evaluation_worker,
            args=(cfg, world_size, f"file://{root / 'rendezvous'}", str(result_path)),
            nprocs=world_size, join=True,
        )
        return json.loads(result_path.read_text(encoding="utf-8"))


def run(cfg, mode):
    """Reuse the existing evaluation/sweep; produce their normal output formats."""
    cfg = json.loads(json.dumps(cfg))
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 or dist.is_initialized():
        raise ValueError("Launch one process; use run.data_parallel=true to start replicas.")
    dataset = cfg["data"]["dataset_name"].rsplit("/", 1)[-1].lower()
    model = cfg["model"]["name_or_path"].rstrip("/").rsplit("/", 1)[-1]
    output = Path(cfg.setdefault("run", {}).setdefault(
        "output_dir", str(Path("ministral_outputs") / model / dataset)
    ))
    output.mkdir(parents=True, exist_ok=True)
    if mode == "evaluate":
        if not cfg["run"].get("save_predictions", False):
            raise ValueError("Keep run.save_predictions=true to compute per-language and all-correct results.")
        result = (_evaluate_replicated(cfg) if cfg["run"].get("data_parallel", False)
                  else _evaluate(cfg, DistInfo(False, 1, 0, 0)))
        for key, filename in [("metrics", "metrics.json"), ("per_language", "per_language.json"),
                              ("all_correct_ids", "all_correct_ids.json")]:
            (output / filename).write_text(json.dumps(result[key], indent=2) + "\n")
        if result["records"] is not None:
            (output / "predictions.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result["records"])
            )
    elif mode == "patch":
        use_baseline_ids = "allowed_parallel_ids_path" not in cfg["data"]
        ids_path = cfg["data"].setdefault("allowed_parallel_ids_path", str(output / "all_correct_ids.json"))
        if ids_path is not None and not Path(ids_path).is_file():
            raise FileNotFoundError(f"Run evaluate first to create {ids_path}.")
        if use_baseline_ids:
            previous = json.loads((output / "evaluate_config.json").read_text())
            # Batch sizes may change; the model, inputs, and scoring must match.
            for section, ignored in [
                ("model", {"cache_dir"}),
                ("data", {"cache_dir", "max_parallel_questions", "allowed_parallel_ids_path"}),
                ("eval", {"batch_size", "generate_batch_size"}),
            ]:
                before = {k: v for k, v in previous[section].items() if k not in ignored}
                after = {k: v for k, v in cfg[section].items() if k not in ignored}
                if before != after:
                    raise ValueError(f"Baseline {section} config differs. Run evaluate with this config first.")
        sweeping._install_graceful_stop_handler(cfg)
        try:
            if cfg["run"].get("data_parallel", False):
                from quantitative_experiments.ministral3_parallel_patch import run_replicated_patch

                result = run_replicated_patch(cfg)
            else:
                with patch.object(sweeping, "HeadPatchingRunner", MinistralRunner):
                    result = sweeping.run_sweep(cfg)
            sweeping._raise_if_graceful_stop_requested()
            sweeping.save_outputs(cfg, result)
        except sweeping.GracefulStopRequested:
            raise SystemExit(sweeping.GRACEFUL_STOP_EXIT_CODE)
    else:
        raise ValueError("mode must be evaluate or patch")
    (output / f"{mode}_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"Ministral {mode} outputs: {output.resolve()}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["evaluate", "patch"])
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    run(apply_overrides(load_config(args.config), args.set), args.mode)


if __name__ == "__main__":
    main()
