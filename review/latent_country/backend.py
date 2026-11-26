"""Hugging Face implementation of the original final-token o_proj-input patch.

No prefix caching or layer-skipping: this keeps the intervention directly
comparable to the original full-forward implementation.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Sequence

import torch


def validate_sites(sites: Sequence, n_layers: int, n_groups: int) -> list[list[int]]:
    result = [[int(layer), int(group)] for layer, group in sites]
    if not result or len({tuple(x) for x in result}) != len(result):
        raise ValueError("Patch sites must be nonempty and unique.")
    if any(not (0 <= layer < n_layers and 0 <= group < n_groups) for layer, group in result):
        raise ValueError("Patch site outside model dimensions.")
    return sorted(result)


def replace_groups(z: torch.Tensor, donor: torch.Tensor, groups: list[int], n_groups: int) -> torch.Tensor:
    if z.shape[-1] % n_groups or donor.shape != z[:, -1, :].shape:
        raise ValueError("Donor/group dimensions do not match attention output.")
    width = z.shape[-1] // n_groups
    result = z.clone()
    for group in groups:
        if not 0 <= group < n_groups:
            raise ValueError("Invalid group index.")
        start = group * width
        result[:, -1, start:start + width] = donor[:, start:start + width].to(device=z.device, dtype=z.dtype)
    return result


class Backend:
    def __init__(self, model, tokenizer, max_input_length: int):
        self.model, self.tokenizer = model.eval(), tokenizer
        self.layers = model.model.layers
        self.n_layers = len(self.layers)
        self.n_heads = int(model.config.num_attention_heads)
        self.n_groups = int(getattr(model.config, "num_key_value_heads", self.n_heads))
        self.max_input_length = max_input_length
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.n_heads % self.n_groups:
            raise ValueError("Query heads must divide evenly into KV groups.")
        expected_width = self.n_heads * int(getattr(model.config, "head_dim", None)
                                          or model.config.hidden_size // self.n_heads)
        for layer in self.layers:
            if layer.self_attn.o_proj.in_features != expected_width:
                raise ValueError("Unsupported attention output layout.")
        self.device = model.get_input_embeddings().weight.device

    @classmethod
    def load(cls, cfg: dict):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        settings = cfg["model"]
        options = {"revision": settings.get("revision"), "trust_remote_code": False}
        dtype = settings.get("dtype", "bfloat16")
        kwargs = dict(options, torch_dtype="auto" if dtype == "auto" else getattr(torch, dtype),
                      attn_implementation=settings.get("attn_implementation", "eager"))
        device = cfg["inference"].get("device", "auto")
        if device == "auto":
            kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(settings["name_or_path"], **kwargs)
        if device != "auto":
            model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(settings["name_or_path"], **options)
        return cls(model, tokenizer, cfg["inference"]["max_input_length"])

    def identity(self) -> dict:
        import transformers
        return {"model_class": type(self.model).__name__, "model_config": self.model.config.to_dict(),
                "resolved_commit": getattr(self.model.config, "_commit_hash", None),
                "torch": torch.__version__, "transformers": transformers.__version__,
                "n_layers": self.n_layers, "n_heads": self.n_heads, "n_groups": self.n_groups}

    def token_ids(self, rows: list[dict]) -> tuple[list[int], list[int]]:
        result = []
        for key in ("recipient_answer", "type_corrected_answer"):
            ids = [self.tokenizer(row[key], add_special_tokens=False)["input_ids"] for row in rows]
            if any(not x for x in ids):
                raise ValueError("Answer tokenized to an empty sequence.")
            result.append([x[0] for x in ids])
        return result[0], result[1]

    def encode(self, prompts: list[str]) -> dict:
        encoded = self.tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=True,
                                 truncation=False)
        if encoded["input_ids"].shape[1] > self.max_input_length:
            raise ValueError("Prompt exceeds max_input_length; refusing silent truncation of the final query.")
        return {k: v.to(self.device) for k, v in encoded.items()}

    @torch.inference_mode()
    def forward(self, encoded: dict) -> torch.Tensor:
        logits = self.model(**encoded, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite model logits.")
        return logits

    @torch.inference_mode()
    def capture(self, prompts: list[str], layers: list[int]) -> dict[int, torch.Tensor]:
        cache, hooks = {}, []
        try:
            for layer_index in layers:
                def capture(module, args, index=layer_index):
                    cache[index] = args[0][:, -1, :].detach().clone().cpu()
                hooks.append(self.layers[layer_index].self_attn.o_proj.register_forward_pre_hook(capture))
            self.forward(self.encode(prompts))
        finally:
            for hook in hooks:
                hook.remove()
        if set(cache) != set(layers):
            raise RuntimeError("Missing donor activations.")
        return cache

    @contextmanager
    def patch(self, donors: dict[int, torch.Tensor], sites: list):
        sites = validate_sites(sites, self.n_layers, self.n_groups)
        grouped = {}
        for layer, group in sites:
            grouped.setdefault(layer, []).append(group)
        hooks = []
        try:
            for layer, groups in grouped.items():
                def patch(module, args, layer=layer, groups=groups):
                    return (replace_groups(args[0], donors[layer], groups, self.n_groups), *args[1:])
                hooks.append(self.layers[layer].self_attn.o_proj.register_forward_pre_hook(patch))
            yield
        finally:
            for hook in hooks:
                hook.remove()

    @torch.inference_mode()
    def score(self, encoded: dict, original: list[int], target: list[int],
              donors: dict | None = None, sites: list | None = None) -> list[dict]:
        if sites is None:
            logits = self.forward(encoded)
        else:
            with self.patch(donors, sites):
                logits = self.forward(encoded)
        index = torch.arange(len(original), device=logits.device)
        original_logits, target_logits = logits[index, original], logits[index, target]
        original_ranks = 1 + (logits > original_logits[:, None]).sum(dim=1)
        target_ranks = 1 + (logits > target_logits[:, None]).sum(dim=1)
        return [{"original_rank": int(original_ranks[i]), "target_rank": int(target_ranks[i]),
                 "original_minus_target_logit": float(original_logits[i] - target_logits[i])}
                for i in range(len(original))]
