from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class Scorer:
    def __init__(self, cfg: Dict[str, Any], device: torch.device):
        self.cfg = cfg
        self.device = device

        model_name = cfg["model"]["name_or_path"]
        print(model_name)
        revision = cfg["model"].get("revision")
        trust_remote_code = cfg["model"].get("trust_remote_code", True)
        dtype = cfg["model"].get("dtype", "auto")
        cache_dir = cfg["model"].get("cache_dir") or cfg.get("data", {}).get("cache_dir")

        torch_dtype = None
        if dtype == "auto":
            torch_dtype = None
        else:
            torch_dtype = getattr(torch, dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )
        # Decoder-only models should use left padding for generation correctness.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
        )
        self.model.to(self.device)
        self.model.eval()

        self.max_input_length = cfg["eval"].get("max_input_length")

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        bos = [self.tokenizer.bos_token_id] if self.tokenizer.bos_token_id is not None else []
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        input_ids = bos + prompt_ids

        if self.max_input_length and len(input_ids) > self.max_input_length:
            input_ids = input_ids[-self.max_input_length :]

        return torch.tensor(input_ids, dtype=torch.long)

    def _encode(self, prompt: str, continuation: str) -> Tuple[torch.Tensor, int]:
        bos = [self.tokenizer.bos_token_id] if self.tokenizer.bos_token_id is not None else []
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        cont_ids = self.tokenizer(continuation, add_special_tokens=False).input_ids
        input_ids = bos + prompt_ids + cont_ids
        prompt_len = len(bos) + len(prompt_ids)

        if self.max_input_length and len(input_ids) > self.max_input_length:
            # Truncate from the left to keep the end (answer) intact
            input_ids = input_ids[-self.max_input_length :]
            prompt_len = max(1, len(input_ids) - len(cont_ids))

        return torch.tensor(input_ids, dtype=torch.long), prompt_len

    @torch.no_grad()
    def generate_choice(
        self,
        prompt: str,
        labels: List[str],
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
    ) -> str | None:
        input_ids = self._encode_prompt(prompt).unsqueeze(0).to(self.device)

        gen_ids = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        new_ids = gen_ids[0, input_ids.size(1) :]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        if not text:
            return None

        for ch in text:
            if ch in labels:
                return ch
        return None

    @torch.no_grad()
    def generate_choices(
        self,
        prompts: List[str],
        labels: List[str],
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
        batch_size: int,
    ) -> List[str | None]:
        results: List[str | None] = []
        idx = 0
        while idx < len(prompts):
            chunk = prompts[idx : idx + batch_size]
            idx += batch_size

            encoded = [self._encode_prompt(p) for p in chunk]
            input_lens = [t.size(0) for t in encoded]
            max_len = max(input_lens)

            pad_id = self.tokenizer.pad_token_id
            padded = torch.full((len(encoded), max_len), pad_id, dtype=torch.long)
            attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.long)
            for i, ids in enumerate(encoded):
                seq_len = ids.size(0)
                if self.tokenizer.padding_side == "left":
                    padded[i, max_len - seq_len :] = ids
                    attention_mask[i, max_len - seq_len :] = 1
                else:
                    padded[i, :seq_len] = ids
                    attention_mask[i, :seq_len] = 1

            padded = padded.to(self.device)
            attention_mask = attention_mask.to(self.device)

            gen_ids = self.model.generate(
                input_ids=padded,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            prompt_width = padded.size(1)
            for i in range(gen_ids.size(0)):
                new_ids = gen_ids[i, prompt_width:]
                text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                choice = None
                if text:
                    for ch in text:
                        if ch in labels:
                            choice = ch
                            break
                results.append(choice)

        return results

    @torch.no_grad()
    def score_detailed(
        self,
        items: List[Tuple[str, str]],
        batch_size: int,
        length_penalty_alpha: float = 0.0,
    ) -> List[Dict[str, float | int]]:
        if length_penalty_alpha < 0:
            raise ValueError("length_penalty_alpha must be non-negative.")

        scores: List[Dict[str, float | int]] = []
        idx = 0
        while idx < len(items):
            batch = items[idx : idx + batch_size]
            idx += batch_size

            encoded = [self._encode(prompt, cont) for prompt, cont in batch]
            input_ids = [e[0] for e in encoded]
            prompt_lens = [e[1] for e in encoded]

            max_len = max(t.size(0) for t in input_ids)
            pad_id = self.tokenizer.pad_token_id
            padded = torch.full((len(input_ids), max_len), pad_id, dtype=torch.long)
            attention_mask = torch.zeros((len(input_ids), max_len), dtype=torch.long)
            for i, ids in enumerate(input_ids):
                padded[i, : ids.size(0)] = ids
                attention_mask[i, : ids.size(0)] = 1

            padded = padded.to(self.device)
            attention_mask = attention_mask.to(self.device)

            outputs = self.model(input_ids=padded, attention_mask=attention_mask)
            logits = outputs.logits

            log_probs = F.log_softmax(logits, dim=-1)
            target_ids = padded[:, 1:]
            log_probs = log_probs[:, :-1, :]
            token_logprobs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

            for b in range(len(batch)):
                seq_len = int(attention_mask[b].sum().item())
                start = max(prompt_lens[b] - 1, 0)
                end = seq_len - 1
                token_count = end - start
                if token_count <= 0:
                    raise ValueError("Continuation tokenized to zero tokens; cannot score candidate.")
                sum_logprob = float(token_logprobs[b, start:end].sum().item())
                denom = float(token_count) ** length_penalty_alpha
                score = sum_logprob / denom
                scores.append(
                    {
                        "score": score,
                        "sum_logprob": sum_logprob,
                        "token_count": token_count,
                    }
                )

        return scores

    @torch.no_grad()
    def score(self, items: List[Tuple[str, str]], batch_size: int) -> List[float]:
        detailed = self.score_detailed(items, batch_size=batch_size, length_penalty_alpha=0.0)
        return [float(entry["sum_logprob"]) for entry in detailed]
