from __future__ import annotations

import numpy as np
import torch

from quantitative_experiments.multilingual_head_patch.run import HeadPatchingRunner


class Backend:
    """Read-only reuse of the original model/prompt/option-token wrapper."""
    def __init__(self, runner):
        self.runner = runner
        if runner.uses_data_parallel:
            raise ValueError("Use model sharding or a single device, not DataParallel.")
        self.n_layers, self.n_groups = runner.n_layers, runner.n_kv_heads
        self.width = runner.layers[0].self_attn.o_proj.in_features
        if self.width % self.n_groups or any(layer.self_attn.o_proj.in_features != self.width for layer in runner.layers):
            raise ValueError("Unsupported attention output layout.")
        self.n_options = len(runner.option_token_ids)

    @classmethod
    def load(cls, cfg: dict):
        return cls(HeadPatchingRunner(cfg))

    def identity(self):
        import transformers
        model = self.runner.base_model
        return {"config": model.config.to_dict(), "resolved_commit": getattr(model.config, "_commit_hash", None),
                "model_class": type(model).__name__, "torch": torch.__version__, "transformers": transformers.__version__,
                "n_layers": self.n_layers, "n_groups": self.n_groups, "width": self.width,
                "option_token_ids": list(self.runner.option_token_ids)}

    @torch.inference_mode()
    def logits(self, prompts: list[str]) -> np.ndarray:
        values = self.runner._forward_option_logits(prompts).detach().float().cpu().numpy()
        if not np.isfinite(values).all():
            raise ValueError("Non-finite option logits.")
        return values

    @torch.inference_mode()
    def capture(self, prompts: list[str]) -> np.ndarray:
        cache, handles = {}, []
        try:
            for layer, block in enumerate(self.runner.layers):
                def capture(module, args, layer=layer):
                    cache[layer] = args[0][:, -1, :].detach().float().cpu().numpy().copy()
                handles.append(block.self_attn.o_proj.register_forward_pre_hook(capture))
            self.logits(prompts)
        finally:
            for handle in handles:
                handle.remove()
        if len(cache) != self.n_layers:
            raise RuntimeError("Missing donor activations.")
        values = np.stack([cache[layer] for layer in range(self.n_layers)], axis=1)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite donor activation.")
        return values

    @torch.inference_mode()
    def patched_logits(self, prompts: list[str], layer: int, group: int, donors: np.ndarray) -> np.ndarray:
        if not 0 <= layer < self.n_layers or not 0 <= group < self.n_groups:
            raise ValueError("Patch location outside model dimensions.")
        if donors.shape != (len(prompts), self.width) or not np.isfinite(donors).all():
            raise ValueError("Donor batch shape or values invalid.")
        width = self.width // self.n_groups
        start, stop = group * width, (group + 1) * width
        vectors = torch.from_numpy(np.asarray(donors[:, start:stop]).copy())
        def patch(module, args):
            z = args[0].clone()
            z[:, -1, start:stop] = vectors.to(device=z.device, dtype=z.dtype)
            return (z, *args[1:])
        handle = self.runner.layers[layer].self_attn.o_proj.register_forward_pre_hook(patch)
        try:
            return self.logits(prompts)
        finally:
            handle.remove()
