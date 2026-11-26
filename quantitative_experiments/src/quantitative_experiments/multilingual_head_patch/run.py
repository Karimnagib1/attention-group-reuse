from __future__ import annotations

import argparse
import hashlib
import json
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantitative_experiments.mmluprox_eval.data import extract_options, load_eval_dataset
from quantitative_experiments.mmluprox_eval.prompt import build_prompt
from quantitative_experiments.multilingual_head_patch.checkpoint import (
    SweepCheckpoint,
    atomic_save_npz,
    atomic_write_json,
    atomic_write_text,
    build_manifest,
)


EPS = 1e-8
GRACEFUL_STOP_EXIT_CODE = 75
_GRACEFUL_STOP_REQUESTED = False


class GracefulStopRequested(RuntimeError):
    """Raised at a safe batch boundary after SLURM requests a checkpoint stop."""


def _handle_graceful_stop_signal(signum, frame) -> None:
    del frame
    global _GRACEFUL_STOP_REQUESTED
    _GRACEFUL_STOP_REQUESTED = True
    print(
        f"Received signal {signum}; stopping at the next safe batch boundary.",
        flush=True,
    )


def _install_graceful_stop_handler(cfg: Dict[str, Any]) -> None:
    checkpoint_cfg = cfg.get("run", {}).get("checkpoint", {})
    if bool(checkpoint_cfg.get("enabled", False)) and hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _handle_graceful_stop_signal)


def _raise_if_graceful_stop_requested() -> None:
    if _GRACEFUL_STOP_REQUESTED:
        raise GracefulStopRequested


def _deep_merge(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge nested dictionaries.

    Example input: base={"a": {"b": 1}}, upd={"a": {"c": 2}}
    Example output: {"a": {"b": 1, "c": 2}}
    """
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(paths: Iterable[str]) -> Dict[str, Any]:
    """Load and merge one or more YAML config files.

    Example input: ["configs/eval/model.yaml", "configs/eval/data.yaml"]
    Example output: {"model": {...}, "data": {...}}
    """
    cfg: Dict[str, Any] = {}
    for p in paths:
        data = yaml.safe_load(Path(p).read_text())
        if data:
            cfg = _deep_merge(cfg, data)
    return cfg


def _set_by_path(cfg: Dict[str, Any], path: str, value: Any) -> None:
    """Set a nested config value from a dotted key path.

    Example input: cfg={}, path="patching.batch_size", value=8
    Example output: cfg becomes {"patching": {"batch_size": 8}}
    """
    cur = cfg
    parts = path.split(".")
    for key in parts[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[parts[-1]] = value


def apply_overrides(cfg: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    """Apply CLI overrides in key=value format to config dict.

    Example input: overrides=["model.name_or_path=Qwen/Qwen3-8B"]
    Example output: cfg["model"]["name_or_path"] == "Qwen/Qwen3-8B"
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Use key=value.")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        _set_by_path(cfg, key, value)
    return cfg


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for config files and overrides.

    Example input: --config exp.yaml --set patching.patch_mode=kv_group
    Example output: Namespace(config=["exp.yaml"], set=["patching.patch_mode=kv_group"])
    """
    parser = argparse.ArgumentParser(description="Multilingual head-patching sweep for parallel MCQ")
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="YAML config path (can be repeated).",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Config override, e.g. --set model.name_or_path=Qwen/Qwen3-8B",
    )
    return parser.parse_args()


@dataclass
class Sample:
    """One aligned multilingual MCQ sample used in patching.

    Example input: question id+language record from dataset
    Example output: Sample(sample_id="42_en", parallel_id="42", language="en", ...)
    """
    sample_id: str
    parallel_id: str
    language: str
    prompt: str
    correct_idx: int
    question: str
    context: str
    options: Tuple[str, ...]


class HeadPatchingRunner:
    """Model/tokenizer wrapper with utilities for option-logit scoring.

    Example input: cfg with model/eval sections
    Example output: initialized runner with model dims and option token ids
    """

    def __init__(self, cfg: Dict[str, Any]):
        """Initialize model, tokenizer, layer handles, and MCQ option ids.

        Example input: cfg["model"]["name_or_path"] = "Qwen/Qwen3-8B"
        Example output: self.n_layers, self.n_heads, self.option_token_ids ready
        """
        self.cfg = cfg
        self.device = torch.device(cfg["run"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.uses_model_sharding = False
        self.uses_data_parallel = False

        model_name = cfg["model"]["name_or_path"]
        revision = cfg["model"].get("revision")
        trust_remote_code = cfg["model"].get("trust_remote_code", True)
        dtype_name = cfg["model"].get("dtype", "auto")
        attn_impl = cfg["model"].get("attn_implementation")

        torch_dtype = None if dtype_name == "auto" else getattr(torch, dtype_name)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: Dict[str, Any] = {
            "revision": revision,
            "trust_remote_code": trust_remote_code,
        }
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl

        # Multi-GPU mode selection:
        # - data_parallel=True => replicate full model on all visible GPUs.
        # - otherwise use device_map sharding when requested/available.
        run_cfg = cfg.get("run", {})
        use_data_parallel = bool(run_cfg.get("data_parallel", False))
        requested_device_map = run_cfg.get("device_map")
        multi_gpu_enabled = bool(run_cfg.get("multi_gpu", True))
        max_memory_cfg = run_cfg.get("max_memory")
        max_memory = self._normalize_max_memory(max_memory_cfg) if max_memory_cfg else None
        if use_data_parallel and requested_device_map is not None:
            raise ValueError("run.data_parallel=true is incompatible with run.device_map. Remove one of them.")

        if not use_data_parallel:
            if (
                requested_device_map is None
                and multi_gpu_enabled
                and self.device.type == "cuda"
                and torch.cuda.device_count() > 1
            ):
                requested_device_map = "auto"
            if max_memory is not None and requested_device_map is None:
                requested_device_map = "auto"
            if requested_device_map is not None:
                model_kwargs["device_map"] = requested_device_map
            if max_memory is not None:
                model_kwargs["max_memory"] = max_memory

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.uses_model_sharding = "device_map" in model_kwargs
        if self.uses_model_sharding:
            self.device = self._resolve_input_device()
        else:
            self.model.to(self.device)
            if use_data_parallel and self.device.type == "cuda" and torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)
                self.uses_data_parallel = True
        self.model.eval()
        self.base_model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model

        self.layers = self._resolve_layers()
        self.n_layers = len(self.layers)
        self.n_heads = int(self.base_model.config.num_attention_heads)
        self.n_kv_heads = int(getattr(self.base_model.config, "num_key_value_heads", self.n_heads))
        self.d_model = int(self.base_model.config.hidden_size)
        self.d_head = self.d_model // self.n_heads

        self.option_labels = cfg["eval"]["option_labels"]
        self.answer_sep = cfg["eval"].get("answer_sep", " ")
        self.option_token_ids = self._prepare_option_token_ids()

        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True

    def _resolve_layers(self):
        """Return the list of transformer blocks for supported architectures.

        Example input: loaded decoder-only HF model
        Example output: model.model.layers (or model.transformer.h)
        """
        if hasattr(self.base_model, "model") and hasattr(self.base_model.model, "layers"):
            return self.base_model.model.layers
        if hasattr(self.base_model, "transformer") and hasattr(self.base_model.transformer, "h"):
            return self.base_model.transformer.h
        raise ValueError("Unsupported architecture: cannot find transformer block list.")

    def _resolve_input_device(self) -> torch.device:
        """Pick a valid input device when model is dispatched across devices."""
        devmap = getattr(self.model, "hf_device_map", None)
        if isinstance(devmap, dict):
            for d in devmap.values():
                if isinstance(d, int):
                    return torch.device(f"cuda:{d}")
                if isinstance(d, str):
                    if d.isdigit():
                        return torch.device(f"cuda:{d}")
                    if d.startswith("cuda"):
                        return torch.device(d)
        return self.device

    def _normalize_max_memory(self, cfg_val: Any) -> Dict[Any, str]:
        """Normalize YAML max_memory keys for HF device placement."""
        if not isinstance(cfg_val, dict):
            raise ValueError("run.max_memory must be a mapping like {'cuda:0': '30GiB', 'cuda:1': '60GiB'}.")
        out: Dict[Any, str] = {}
        for k, v in cfg_val.items():
            key = str(k).strip()
            if key.startswith("cuda:"):
                out[int(key.split(":", 1)[1])] = str(v)
            elif key.isdigit():
                out[int(key)] = str(v)
            else:
                out[key] = str(v)
        return out

    def _prepare_option_token_ids(self) -> List[int]:
        """Map option labels to single-token ids for next-token scoring.

        Example input: option_labels=["A","B","C"], answer_sep=" "
        Example output: [362, 426, 334]
        """
        ids: List[int] = []
        for label in self.option_labels:
            text = f"{self.answer_sep}{label}"
            toks = self.tokenizer(text, add_special_tokens=False).input_ids
            if len(toks) != 1:
                raise ValueError(
                    f"Option label continuation '{text}' tokenized to {len(toks)} tokens; expected 1. "
                    "Adjust eval.answer_sep / eval.option_labels for this tokenizer."
                )
            ids.append(int(toks[0]))
        return ids

    def _encode_prompts(self, prompts: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize prompts with padding/truncation and move tensors to device.

        Example input: ["Q1... Answer:", "Q2... Answer:"]
        Example output: (input_ids[B,T], attention_mask[B,T])
        """
        enc = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg["eval"].get("max_input_length"),
            add_special_tokens=True,
        )
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    @torch.no_grad()
    def _forward_option_logits(self, prompts: Sequence[str]) -> torch.Tensor:
        """Compute next-token logits restricted to MCQ option tokens.

        Example input: prompts of length B
        Example output: tensor shape [B, num_options]
        """
        input_ids, attention_mask = self._encode_prompts(prompts)
        # out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # last_logits = out.logits[:, -1, :]
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=1,   # <--- KEY: only compute logits for last token
        )
        last_logits = out.logits[:, -1, :]  # now logits shape is [B, 1, vocab]

        option_ids = torch.tensor(self.option_token_ids, device=last_logits.device, dtype=torch.long)
        return last_logits.index_select(1, option_ids)


def _answer_index(example: Dict[str, Any], cfg: Dict[str, Any]) -> int:
    """Resolve gold answer index from label or numeric field.

    Example input: {"answer": "C"} with labels ["A","B","C","D","E"]
    Example output: 2
    """
    answer_index_offset = int(cfg["data"].get("answer_index_offset", 0))
    answer_index_field = cfg["data"].get("answer_index_field")
    if answer_index_field and answer_index_field in example:
        idx = int(example[answer_index_field]) + answer_index_offset
        labels = cfg["eval"]["option_labels"]
        if idx < 0 or idx >= len(labels):
            raise ValueError(f"Answer index {idx} out of bounds for labels {labels}.")
        return idx

    answer_field = cfg["data"].get("answer_field", "answer")
    labels = cfg["eval"]["option_labels"]
    raw = str(example[answer_field]).strip()
    if raw in labels:
        return labels.index(raw)

    # Numeric fallback
    try:
        idx = int(raw) + answer_index_offset
    except ValueError as exc:
        raise ValueError(f"Could not parse answer from '{raw}'.") from exc
    if idx < 0 or idx >= len(labels):
        raise ValueError(f"Answer index {idx} out of bounds for labels {labels}.")
    return idx


def _load_allowed_parallel_ids(cfg: Dict[str, Any]) -> set[str] | None:
    """Load an optional allow-list of parallel IDs from JSON.

    Example input: data.allowed_parallel_ids_path="outputs/mcsqa/all_correct_ids.json"
    Example output: {"q1", "q2", "q3"} or None
    """
    path = cfg["data"].get("allowed_parallel_ids_path")
    if not path:
        return None

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("data.allowed_parallel_ids_path must point to a JSON list of IDs.")
    return {str(x) for x in raw}


def _render_prompt_from_parts(question: str, context: str, options: Sequence[str], cfg: Dict[str, Any]) -> str:
    question_field = cfg["data"].get("question_field", "question")
    context_field = cfg["data"].get("context_field")
    ex: Dict[str, Any] = {question_field: question}
    if context_field:
        ex[context_field] = context
    return build_prompt(ex, list(options), cfg)


def _choice_order_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg.get("patching", {}).get("choice_order_perturbations", {})


def _stable_seed(base_seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:8], byteorder="big", signed=False)) % (2**32)


def _num_non_empty_options(options: Sequence[str]) -> int:
    return sum(1 for opt in options if str(opt).strip() != "")


def _num_choice_perturbations(sample: Sample, cfg: Dict[str, Any]) -> int:
    perturb_cfg = _choice_order_cfg(cfg)
    fixed_n = perturb_cfg.get("num_perturbations")
    if fixed_n is not None:
        return max(0, int(fixed_n))
    return max(_num_non_empty_options(sample.options) - 1, 5)


def _expand_samples_for_choice_order(
    base_samples: Sequence[Sample], cfg: Dict[str, Any]
) -> Tuple[List[Sample], np.ndarray, np.ndarray]:
    perturb_cfg = _choice_order_cfg(cfg)
    enabled = bool(perturb_cfg.get("enabled", False))
    if not enabled:
        ids = np.arange(len(base_samples), dtype=np.int64)
        counts = np.ones(len(base_samples), dtype=np.int64)
        return list(base_samples), ids, counts

    include_original = bool(perturb_cfg.get("include_original", False))
    base_seed = int(perturb_cfg.get("seed", 42))

    expanded: List[Sample] = []
    group_ids: List[int] = []
    counts = np.zeros(len(base_samples), dtype=np.int64)

    for base_idx, s in enumerate(base_samples):
        active_positions = [i for i, opt in enumerate(s.options) if str(opt).strip() != ""]
        if len(active_positions) <= 1:
            expanded.append(s)
            group_ids.append(base_idx)
            counts[base_idx] += 1
            continue

        n_perturb = _num_choice_perturbations(s, cfg)
        if n_perturb <= 0:
            expanded.append(s)
            group_ids.append(base_idx)
            counts[base_idx] += 1
            continue

        rng = np.random.default_rng(_stable_seed(base_seed, s.sample_id))
        seen: set[Tuple[int, ...]] = set()
        identity = tuple(active_positions)
        max_attempts = max(64, 20 * n_perturb)
        made = 0

        if include_original:
            expanded.append(s)
            group_ids.append(base_idx)
            counts[base_idx] += 1

        for _ in range(max_attempts):
            if made >= n_perturb:
                break
            shuffled = list(rng.permutation(active_positions))
            key = tuple(shuffled)
            if key == identity or key in seen:
                continue
            seen.add(key)

            new_options = list(s.options)
            for dest_pos, src_pos in zip(active_positions, shuffled):
                new_options[dest_pos] = s.options[src_pos]

            src_to_dest = {src: dest for dest, src in zip(active_positions, shuffled)}
            new_correct_idx = int(src_to_dest.get(s.correct_idx, s.correct_idx))
            new_prompt = _render_prompt_from_parts(
                question=s.question,
                context=s.context,
                options=new_options,
                cfg=cfg,
            )

            expanded.append(
                Sample(
                    sample_id=f"{s.sample_id}::perm{made + 1}",
                    parallel_id=s.parallel_id,
                    language=s.language,
                    prompt=new_prompt,
                    correct_idx=new_correct_idx,
                    question=s.question,
                    context=s.context,
                    options=tuple(str(x) for x in new_options),
                )
            )
            group_ids.append(base_idx)
            counts[base_idx] += 1
            made += 1

        if counts[base_idx] == 0:
            expanded.append(s)
            group_ids.append(base_idx)
            counts[base_idx] += 1

    return expanded, np.array(group_ids, dtype=np.int64), counts


def _group_mean(arr: np.ndarray, group_ids: np.ndarray, n_groups: int) -> np.ndarray:
    out_shape = (n_groups,) + arr.shape[1:]
    out = np.zeros(out_shape, dtype=np.float64)
    counts = np.zeros(n_groups, dtype=np.int64)
    for i, g in enumerate(group_ids):
        out[g] += arr[i]
        counts[g] += 1
    reshape = (n_groups,) + (1,) * (arr.ndim - 1)
    counts_safe = np.maximum(counts, 1).reshape(reshape)
    return (out / counts_safe).astype(np.float32, copy=False)


def _build_parallel_samples(cfg: Dict[str, Any]) -> Tuple[List[Sample], List[str]]:
    """Build aligned samples present in all requested languages.

    Example input: cfg with languages=["en","es"], parallel_id_field="id"
    Example output: ([Sample(...en), Sample(...es), ...], ["en", "es"])
    """
    ds = load_eval_dataset(cfg)
    language_field = cfg["data"].get("language_field", "language")
    id_field = cfg["data"].get("id_field", "id")
    parallel_id_field = cfg["data"].get("parallel_id_field", id_field)
    option_fields = cfg["data"].get("option_fields", [f"option_{i}" for i in range(5)])
    question_field = cfg["data"].get("question_field", "question")
    context_field = cfg["data"].get("context_field")
    requested_languages = cfg["data"].get("languages")

    records: List[Dict[str, Any]] = [dict(ds[i]) for i in range(len(ds))]

    if requested_languages:
        lang_set = set(requested_languages)
        records = [r for r in records if str(r.get(language_field)) in lang_set]
        languages = list(requested_languages)
    else:
        languages = sorted({str(r.get(language_field)) for r in records if r.get(language_field) is not None})

    by_parallel: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in records:
        pid = str(r.get(parallel_id_field))
        lang = str(r.get(language_field))
        if not pid or lang == "None":
            continue
        by_parallel.setdefault(pid, {})[lang] = r

    kept_ids = [pid for pid, per_lang in by_parallel.items() if all(lang in per_lang for lang in languages)]
    kept_ids.sort()

    allowed_ids = _load_allowed_parallel_ids(cfg)
    if allowed_ids is not None:
        kept_ids = [pid for pid in kept_ids if pid in allowed_ids]

    max_parallel = cfg["data"].get("max_parallel_questions")
    if max_parallel is not None:
        kept_ids = kept_ids[: int(max_parallel)]

    samples: List[Sample] = []
    for pid in kept_ids:
        for lang in languages:
            ex = by_parallel[pid][lang]
            options = extract_options(ex, option_fields)
            question = str(ex.get(question_field, "")).strip()
            context = str(ex.get(context_field, "")).strip() if context_field else ""
            prompt = _render_prompt_from_parts(question=question, context=context, options=options, cfg=cfg)
            sample_id = str(ex.get(id_field, f"{pid}:{lang}"))
            samples.append(
                Sample(
                    sample_id=sample_id,
                    parallel_id=pid,
                    language=lang,
                    prompt=prompt,
                    correct_idx=_answer_index(ex, cfg),
                    question=question,
                    context=context,
                    options=tuple(str(x) for x in options),
                )
            )

    return samples, languages


def _rank_from_desc(values: np.ndarray) -> np.ndarray:
    """Convert row-wise scores into 1-based descending ranks.

    Example input: [[0.2, 0.8, 0.1]]
    Example output: [[2, 1, 3]]
    """
    order = np.argsort(-values, axis=-1)
    ranks = np.empty_like(order)
    row_idx = np.arange(values.shape[0])[:, None]
    ranks[row_idx, order] = np.arange(1, values.shape[1] + 1)[None, :]
    return ranks


def _margin_vs_incorrect_mean(option_logits: np.ndarray, target_idx: np.ndarray) -> np.ndarray:
    """Margin: target logit minus mean of non-target logits.

    Example input: logits=[[5,2,1]], target_idx=[0]
    Example output: [3.5]
    """
    row = np.arange(option_logits.shape[0])
    tgt = option_logits[row, target_idx]
    sums = option_logits.sum(axis=1) - tgt
    mean_bad = sums / (option_logits.shape[1] - 1)
    return tgt - mean_bad


def _margin_vs_incorrect_max(option_logits: np.ndarray, target_idx: np.ndarray) -> np.ndarray:
    """Margin: target logit minus strongest non-target logit.

    Example input: logits=[[5,2,1]], target_idx=[0]
    Example output: [3.0]
    """
    masked = option_logits.copy()
    row = np.arange(option_logits.shape[0])
    masked[row, target_idx] = -np.inf
    return option_logits[row, target_idx] - masked.max(axis=1)


def _percent_drop(base: np.ndarray, patched: np.ndarray) -> np.ndarray:
    """Compute percent variation from base to patched values.

    Example input: base=[2.0], patched=[1.0]
    Example output: [50.0]
    """
    return 100.0 * (base - patched) / (np.abs(base) + EPS)


def _rankdata_1d(x: np.ndarray) -> np.ndarray:
    """Assign average ranks (1-based) to a 1D array with ties.

    Example input: [10, 10, 30]
    Example output: [1.5, 1.5, 3.0]
    """
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    n = x.shape[0]
    i = 0
    while i < n:
        j = i + 1
        while j < n and x[order[j]] == x[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Spearman correlation using tie-aware ranks.

    Example input: a=[1,2,3], b=[3,2,1]
    Example output: -1.0
    """
    ra = _rankdata_1d(a)
    rb = _rankdata_1d(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.linalg.norm(ra) * np.linalg.norm(rb)
    if denom < EPS:
        return 0.0
    return float(np.dot(ra, rb) / denom)


def _topk_index_set(x: np.ndarray, k: int) -> set[int]:
    """Return indices of the top-k largest values from a 1D vector."""
    if x.ndim != 1:
        raise ValueError("Expected a 1D vector for top-k extraction.")
    n = x.shape[0]
    if n == 0:
        return set()
    kk = max(1, min(int(k), n))
    idx = np.argpartition(-x, kk - 1)[:kk]
    return {int(i) for i in idx.tolist()}


def _jaccard(a: set[int], b: set[int]) -> float:
    """Compute Jaccard similarity between two index sets."""
    union = a | b
    if not union:
        return 0.0
    return float(len(a & b) / len(union))


def _batched_indices(n: int, batch_size: int):
    """Yield [start, end) slices for batching.

    Example input: n=10, batch_size=4
    Example output: (0,4), (4,8), (8,10)
    """
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield start, end


def _num_batches(n: int, batch_size: int) -> int:
    if n <= 0:
        return 0
    return (n + batch_size - 1) // batch_size


def _split_by_data_parallel_device(batch_tensor_cpu: torch.Tensor, device_ids: Sequence[int]) -> Dict[str, torch.Tensor]:
    """Split a [B, ...] CPU tensor into DataParallel device chunks matching scatter order."""
    chunks = torch.chunk(batch_tensor_cpu, len(device_ids), dim=0)
    out: Dict[str, torch.Tensor] = {}
    for did, chunk in zip(device_ids, chunks):
        if chunk.numel() == 0:
            continue
        out[f"cuda:{int(did)}"] = chunk
    return out


def _head_indices_for_patch(
    patch_mode: str,
    patch_idx: int,
    n_heads: int,
    n_kv_heads: int,
) -> List[int]:
    """Map a patch unit index to underlying head indices.

    Example input: patch_mode="kv_group", patch_idx=1, n_heads=16, n_kv_heads=4
    Example output: [4, 5, 6, 7]
    """
    if patch_mode == "head":
        return [patch_idx]

    if patch_mode == "kv_group":
        if n_heads % n_kv_heads != 0:
            raise ValueError(
                f"num_attention_heads ({n_heads}) must be divisible by num_key_value_heads ({n_kv_heads})."
            )
        group_size = n_heads // n_kv_heads
        start = patch_idx * group_size
        end = (patch_idx + 1) * group_size
        return list(range(start, end))

    raise ValueError("patching.patch_mode must be 'head' or 'kv_group'.")


@torch.no_grad()
def _compute_language_donors(
    runner: HeadPatchingRunner,
    samples: Sequence[Sample],
    languages: Sequence[str],
    batch_size: int,
) -> Dict[str, torch.Tensor]:
    """Compute per-language donor means at each layer/head (last token).

    Example input: samples across languages ["en","es"], batch_size=4
    Example output: {"en": tensor[L,H,Dh], "es": tensor[L,H,Dh]}
    """
    prompts_by_lang: Dict[str, List[str]] = {lang: [] for lang in languages}
    for s in samples:
        prompts_by_lang[s.language].append(s.prompt)

    donors: Dict[str, torch.Tensor] = {}
    for lang in tqdm(languages, desc="Donor languages"):
        sums = torch.zeros((runner.n_layers, runner.n_heads, runner.d_head), dtype=torch.float32)
        count = 0

        n_lang = len(prompts_by_lang[lang])
        for start, end in tqdm(
            _batched_indices(n_lang, batch_size),
            total=_num_batches(n_lang, batch_size),
            desc=f"Donor batches [{lang}]",
            leave=False,
        ):
            _raise_if_graceful_stop_requested()
            prompts = prompts_by_lang[lang][start:end]
            input_ids, attention_mask = runner._encode_prompts(prompts)
            layer_cache: List[List[torch.Tensor]] = [[] for _ in range(runner.n_layers)]
            hooks = []

            for layer_idx, layer in enumerate(runner.layers):
                o_proj = layer.self_attn.o_proj

                def _capture(module, layer_input, layer_idx=layer_idx):
                    z = layer_input[0][:, -1, :].detach().float().cpu()
                    z = z.view(z.size(0), runner.n_heads, runner.d_head)
                    layer_cache[layer_idx].append(z.sum(dim=0))
                    return None

                hooks.append(o_proj.register_forward_pre_hook(_capture))

            # _ = runner.model(input_ids=input_ids, attention_mask=attention_mask)
            _ = runner.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=1,
            )

            for h in hooks:
                h.remove()

            for layer_idx, layer_parts in enumerate(layer_cache):
                if not layer_parts:
                    raise RuntimeError(f"Missing donor activation for layer {layer_idx}.")
                layer_sum = torch.stack(layer_parts, dim=0).sum(dim=0)
                sums[layer_idx] += layer_sum

            count += end - start
            _raise_if_graceful_stop_requested()

        if count == 0:
            raise ValueError(f"No samples found for language '{lang}'.")

        donors[lang] = sums / float(count)

    return donors


@torch.no_grad()
def run_sweep(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run full head/KV-group sweep and compute importance metrics.

    Example input: config with patching.patch_mode="head"
    Example output: dict with metadata, baseline, importance tensors, Spearman and Jaccard summaries
    """
    base_samples, languages = _build_parallel_samples(cfg)
    if not base_samples:
        raise ValueError("No aligned parallel samples after filtering.")
    eval_samples, eval_to_base_ids, n_variants_per_base = _expand_samples_for_choice_order(base_samples, cfg)
    perturb_cfg = _choice_order_cfg(cfg)
    perturb_enabled = bool(perturb_cfg.get("enabled", False))

    runner = HeadPatchingRunner(cfg)
    n_base_samples = len(base_samples)
    n_eval_samples = len(eval_samples)

    print(
        f"Loaded {n_base_samples} base samples ({n_base_samples // len(languages)} parallel questions x {len(languages)} languages)."
    )
    if perturb_enabled:
        print(
            f"Choice-order perturbations enabled for baseline/patching: expanded to {n_eval_samples} eval samples "
            f"(avg {float(n_variants_per_base.mean()):.2f} variants per base sample)."
        )
    print(
        f"Model layers={runner.n_layers}, heads={runner.n_heads}, kv_heads={runner.n_kv_heads}, d_head={runner.d_head}."
    )
    if runner.uses_model_sharding:
        devmap = getattr(runner.model, "hf_device_map", {})
        gpu_devices = sorted(
            {
                f"cuda:{d}" if isinstance(d, int) else d
                for d in devmap.values()
                if (isinstance(d, int) or (isinstance(d, str) and d.startswith("cuda")))
            }
        )
        print(f"Model sharded across devices: {gpu_devices}")
    elif runner.uses_data_parallel:
        print(f"Model replicated with DataParallel across {torch.cuda.device_count()} GPUs.")

    donor_batch_size = int(cfg["patching"].get("donor_batch_size", cfg["patching"].get("batch_size", 4)))
    sweep_batch_size = int(cfg["patching"].get("batch_size", 4))
    patch_mode = str(cfg["patching"].get("patch_mode", "head")).strip().lower()
    if patch_mode not in {"head", "kv_group"}:
        raise ValueError("patching.patch_mode must be 'head' or 'kv_group'.")
    n_patch_units = runner.n_heads if patch_mode == "head" else runner.n_kv_heads
    group_size = runner.n_heads // runner.n_kv_heads if runner.n_kv_heads > 0 else 1

    checkpoint_cfg = cfg.get("run", {}).get("checkpoint", {})
    checkpoint_enabled = bool(checkpoint_cfg.get("enabled", False))
    checkpoint: SweepCheckpoint | None = None
    if checkpoint_enabled:
        model_commit = getattr(runner.base_model.config, "_commit_hash", None)
        manifest = build_manifest(
            cfg=cfg,
            samples=eval_samples,
            model_commit=str(model_commit) if model_commit is not None else None,
            n_eval_samples=n_eval_samples,
            n_layers=runner.n_layers,
            n_patch_units=n_patch_units,
            n_heads=runner.n_heads,
            n_kv_heads=runner.n_kv_heads,
            d_head=runner.d_head,
            n_options=len(runner.option_labels),
        )
        checkpoint = SweepCheckpoint(
            Path(cfg["run"]["output_dir"]),
            manifest,
            resume=bool(checkpoint_cfg.get("resume", True)),
        )
        completed_units = checkpoint.completed_unit_count()
        print(
            f"Checkpointing enabled at {checkpoint.root.resolve()} "
            f"({completed_units}/{runner.n_layers * n_patch_units} patch units complete)."
        )

    _raise_if_graceful_stop_requested()
    donors = (
        checkpoint.load_donors(
            languages,
            n_layers=runner.n_layers,
            n_heads=runner.n_heads,
            d_head=runner.d_head,
        )
        if checkpoint is not None
        else None
    )
    if donors is None:
        donors = _compute_language_donors(runner, base_samples, languages, donor_batch_size)
        if checkpoint is not None:
            checkpoint.save_donors(donors, languages)
            print("Saved donor checkpoint.")
    else:
        print("Restored donor checkpoint.")

    # Baseline logits and targets
    correct_indices = np.array([s.correct_idx for s in eval_samples], dtype=np.int64)
    baseline_option_logits = checkpoint.load_baseline() if checkpoint is not None else None
    if baseline_option_logits is None:
        baseline_option_logits = np.zeros(
            (n_eval_samples, len(runner.option_labels)),
            dtype=np.float32,
        )
        for start, end in tqdm(
            _batched_indices(n_eval_samples, sweep_batch_size),
            total=_num_batches(n_eval_samples, sweep_batch_size),
            desc="Baseline batches",
        ):
            _raise_if_graceful_stop_requested()
            prompts = [s.prompt for s in eval_samples[start:end]]
            logits = runner._forward_option_logits(prompts)
            baseline_option_logits[start:end] = logits.detach().cpu().numpy().astype(np.float32)
            _raise_if_graceful_stop_requested()
        if checkpoint is not None:
            checkpoint.save_baseline(baseline_option_logits)
            print("Saved baseline checkpoint.")
    else:
        print("Restored baseline checkpoint.")

    generated_indices = baseline_option_logits.argmax(axis=1).astype(np.int64)
    target_mode = cfg["patching"].get("target_mode", "generated")
    if target_mode not in {"generated", "correct"}:
        raise ValueError("patching.target_mode must be 'generated' or 'correct'.")
    target_indices = generated_indices if target_mode == "generated" else correct_indices

    base_margin_mean = _margin_vs_incorrect_mean(baseline_option_logits, target_indices)
    base_margin_max = _margin_vs_incorrect_max(baseline_option_logits, target_indices)
    base_ranks = _rank_from_desc(baseline_option_logits)
    base_correct_rank = base_ranks[np.arange(n_eval_samples), correct_indices]

    metric_mean = np.zeros((n_eval_samples, runner.n_layers, n_patch_units), dtype=np.float32)
    metric_max = np.zeros((n_eval_samples, runner.n_layers, n_patch_units), dtype=np.float32)
    metric_rank_drop = np.zeros((n_eval_samples, runner.n_layers, n_patch_units), dtype=np.float32)

    # Sweep layer and patch unit (head or KV-group)
    for layer_idx in tqdm(range(runner.n_layers), desc="Sweep layers"):
        print(f"Sweeping layer {layer_idx + 1}/{runner.n_layers}...")
        for patch_idx in tqdm(range(n_patch_units), desc=f"Patch units [L{layer_idx}]", leave=False):
            _raise_if_graceful_stop_requested()
            saved_unit = checkpoint.load_unit(layer_idx, patch_idx) if checkpoint is not None else None
            if saved_unit is not None:
                metric_mean[:, layer_idx, patch_idx] = saved_unit["pct_var_mean_incorrect"]
                metric_max[:, layer_idx, patch_idx] = saved_unit["pct_var_max_incorrect"]
                metric_rank_drop[:, layer_idx, patch_idx] = saved_unit["rank_drop_correct"]
                continue

            patched_option_logits = np.zeros((n_eval_samples, len(runner.option_labels)), dtype=np.float32)
            patch_heads = _head_indices_for_patch(
                patch_mode=patch_mode,
                patch_idx=patch_idx,
                n_heads=runner.n_heads,
                n_kv_heads=runner.n_kv_heads,
            )

            for start, end in tqdm(
                _batched_indices(n_eval_samples, sweep_batch_size),
                total=_num_batches(n_eval_samples, sweep_batch_size),
                desc=f"Patched batches [L{layer_idx} U{patch_idx}]",
                leave=False,
            ):
                _raise_if_graceful_stop_requested()
                chunk = eval_samples[start:end]
                prompts = [s.prompt for s in chunk]
                lang_chunk = [s.language for s in chunk]

                input_ids, attention_mask = runner._encode_prompts(prompts)
                layer = runner.layers[layer_idx]
                o_proj = layer.self_attn.o_proj

                donor_vectors_by_head_cpu = {
                    h: torch.stack([donors[lang][layer_idx, h] for lang in lang_chunk], dim=0)
                    for h in patch_heads
                }
                donor_vectors_by_head_by_device: Dict[int, Dict[str, torch.Tensor]] = {}
                if runner.uses_data_parallel:
                    device_ids = list(getattr(runner.model, "device_ids", []))
                    for h in patch_heads:
                        donor_vectors_by_head_by_device[h] = _split_by_data_parallel_device(
                            donor_vectors_by_head_cpu[h], device_ids
                        )

                def _patch(module, layer_input):
                    z = layer_input[0].clone()
                    z_device = str(z.device)
                    for h in patch_heads:
                        start_ix = h * runner.d_head
                        end_ix = (h + 1) * runner.d_head
                        if runner.uses_data_parallel:
                            donor_chunk = donor_vectors_by_head_by_device[h].get(z_device)
                            if donor_chunk is None:
                                raise RuntimeError(
                                    f"Missing donor chunk for device {z_device} in DataParallel patching."
                                )
                            z[:, -1, start_ix:end_ix] = donor_chunk.to(z.device, non_blocking=True)
                        else:
                            z[:, -1, start_ix:end_ix] = donor_vectors_by_head_cpu[h].to(
                                z.device, non_blocking=True
                            )
                    return (z,)

                hook = o_proj.register_forward_pre_hook(_patch)
                # out = runner.model(input_ids=input_ids, attention_mask=attention_mask)
                
                out = runner.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    logits_to_keep=1,
                )

                hook.remove()

                last_logits = out.logits[:, -1, :]
                option_ids = torch.tensor(runner.option_token_ids, device=last_logits.device, dtype=torch.long)
                opt = last_logits.index_select(1, option_ids)
                patched_option_logits[start:end] = opt.detach().cpu().numpy().astype(np.float32)
                _raise_if_graceful_stop_requested()

            patched_margin_mean = _margin_vs_incorrect_mean(patched_option_logits, target_indices)
            patched_margin_max = _margin_vs_incorrect_max(patched_option_logits, target_indices)
            patched_ranks = _rank_from_desc(patched_option_logits)
            patched_correct_rank = patched_ranks[np.arange(n_eval_samples), correct_indices]

            unit_metrics = {
                "pct_var_mean_incorrect": _percent_drop(base_margin_mean, patched_margin_mean),
                "pct_var_max_incorrect": _percent_drop(base_margin_max, patched_margin_max),
                "rank_drop_correct": (patched_correct_rank - base_correct_rank).astype(np.float32),
            }
            metric_mean[:, layer_idx, patch_idx] = unit_metrics["pct_var_mean_incorrect"]
            metric_max[:, layer_idx, patch_idx] = unit_metrics["pct_var_max_incorrect"]
            metric_rank_drop[:, layer_idx, patch_idx] = unit_metrics["rank_drop_correct"]
            if checkpoint is not None:
                checkpoint.save_unit(layer_idx, patch_idx, unit_metrics)
            _raise_if_graceful_stop_requested()

    if perturb_enabled:
        final_samples = list(base_samples)
        final_baseline_option_logits = _group_mean(baseline_option_logits, eval_to_base_ids, n_base_samples)
        final_base_margin_mean = _group_mean(base_margin_mean, eval_to_base_ids, n_base_samples)
        final_base_margin_max = _group_mean(base_margin_max, eval_to_base_ids, n_base_samples)
        final_base_correct_rank = _group_mean(base_correct_rank, eval_to_base_ids, n_base_samples)
        final_metric_mean = _group_mean(metric_mean, eval_to_base_ids, n_base_samples)
        final_metric_max = _group_mean(metric_max, eval_to_base_ids, n_base_samples)
        final_metric_rank_drop = _group_mean(metric_rank_drop, eval_to_base_ids, n_base_samples)
        final_correct_indices = np.array([s.correct_idx for s in base_samples], dtype=np.int64)
    else:
        final_samples = list(eval_samples)
        final_baseline_option_logits = baseline_option_logits
        final_base_margin_mean = base_margin_mean
        final_base_margin_max = base_margin_max
        final_base_correct_rank = base_correct_rank
        final_metric_mean = metric_mean
        final_metric_max = metric_max
        final_metric_rank_drop = metric_rank_drop
        final_correct_indices = correct_indices

    final_generated_indices = final_baseline_option_logits.argmax(axis=1).astype(np.int64)
    final_target_indices = final_generated_indices if target_mode == "generated" else final_correct_indices

    n_final_samples = len(final_samples)
    flat_mean = final_metric_mean.reshape(n_final_samples, -1)
    flat_max = final_metric_max.reshape(n_final_samples, -1)
    flat_rank = final_metric_rank_drop.reshape(n_final_samples, -1)

    sample_meta = []
    for i, s in enumerate(final_samples):
        sample_meta.append(
            {
                "sample_index": i,
                "sample_id": s.sample_id,
                "parallel_id": s.parallel_id,
                "language": s.language,
                "correct_idx": int(final_correct_indices[i]),
                "generated_idx": int(final_generated_indices[i]),
                "target_idx": int(final_target_indices[i]),
                "generated_label": runner.option_labels[int(final_generated_indices[i])],
                "correct_label": runner.option_labels[int(final_correct_indices[i])],
                "n_choice_order_variants": int(n_variants_per_base[i]) if perturb_enabled else 1,
            }
        )

    # Optional Spearman summary across languages for each parallel question.
    spearman_summary: Dict[str, Any] = {}
    if cfg["run"].get("compute_spearman", True):
        vectors_by_metric = {
            "pct_var_mean_incorrect": flat_mean,
            "pct_var_max_incorrect": flat_max,
            "rank_drop_correct": flat_rank,
        }

        index_by_pair: Dict[Tuple[str, str], int] = {}
        for i, s in enumerate(final_samples):
            index_by_pair[(s.parallel_id, s.language)] = i

        for metric_name, vecs in vectors_by_metric.items():
            mat = np.zeros((len(languages), len(languages)), dtype=np.float32)
            counts = np.zeros_like(mat, dtype=np.int32)
            for pid in sorted({s.parallel_id for s in final_samples}):
                for i, li in enumerate(languages):
                    idx_i = index_by_pair.get((pid, li))
                    if idx_i is None:
                        continue
                    for j, lj in enumerate(languages):
                        idx_j = index_by_pair.get((pid, lj))
                        if idx_j is None:
                            continue
                        corr = _spearman_corr(vecs[idx_i], vecs[idx_j])
                        mat[i, j] += corr
                        counts[i, j] += 1

            with np.errstate(invalid="ignore"):
                mat = np.divide(mat, np.maximum(counts, 1), where=np.maximum(counts, 1) > 0)
            spearman_summary[metric_name] = {
                "languages": list(languages),
                "mean_spearman": mat.tolist(),
                "pair_counts": counts.tolist(),
            }

    # Optional Jaccard overlap of top-k heads/KV-groups across languages per parallel question.
    jaccard_summary: Dict[str, Any] = {}
    if cfg["run"].get("compute_jaccard", True):
        top_k = int(cfg["run"].get("jaccard_top_k", 5))
        vectors_by_metric = {
            "pct_var_mean_incorrect": flat_mean,
            "pct_var_max_incorrect": flat_max,
            "rank_drop_correct": flat_rank,
        }

        index_by_pair: Dict[Tuple[str, str], int] = {}
        for i, s in enumerate(final_samples):
            index_by_pair[(s.parallel_id, s.language)] = i

        all_parallel_ids = sorted({s.parallel_id for s in final_samples})
        for metric_name, vecs in vectors_by_metric.items():
            per_pid_rows: List[Dict[str, Any]] = []
            mean_mat = np.zeros((len(languages), len(languages)), dtype=np.float32)
            counts = np.zeros_like(mean_mat, dtype=np.int32)
            overall_pair_vals: List[float] = []

            for pid in all_parallel_ids:
                top_sets: Dict[str, set[int]] = {}
                for lang in languages:
                    idx = index_by_pair.get((pid, lang))
                    if idx is None:
                        continue
                    top_sets[lang] = _topk_index_set(vecs[idx], top_k)

                pair_vals_for_pid: List[float] = []
                pairwise: Dict[str, float] = {}
                for i, li in enumerate(languages):
                    if li not in top_sets:
                        continue
                    for j, lj in enumerate(languages):
                        if lj not in top_sets:
                            continue
                        jac = _jaccard(top_sets[li], top_sets[lj])
                        mean_mat[i, j] += jac
                        counts[i, j] += 1
                        pairwise[f"{li}__{lj}"] = jac
                        if i < j:
                            pair_vals_for_pid.append(jac)
                            overall_pair_vals.append(jac)

                per_pid_rows.append(
                    {
                        "parallel_id": pid,
                        "pairwise_jaccard": pairwise,
                        "mean_jaccard_across_language_pairs": (
                            float(np.mean(pair_vals_for_pid)) if pair_vals_for_pid else None
                        ),
                    }
                )

            with np.errstate(invalid="ignore"):
                mean_mat = np.divide(mean_mat, np.maximum(counts, 1), where=np.maximum(counts, 1) > 0)

            jaccard_summary[metric_name] = {
                "languages": list(languages),
                "top_k": top_k,
                "mean_jaccard": mean_mat.tolist(),
                "pair_counts": counts.tolist(),
                "overall_mean_jaccard_across_language_pairs": (
                    float(np.mean(overall_pair_vals)) if overall_pair_vals else None
                ),
                "per_parallel_question": per_pid_rows,
            }

    # Mean heatmaps per language for quick inspection.
    lang_to_indices: Dict[str, List[int]] = {lang: [] for lang in languages}
    for i, s in enumerate(final_samples):
        lang_to_indices[s.language].append(i)

    language_means: Dict[str, Dict[str, List[List[float]]]] = {}
    for lang, idxs in lang_to_indices.items():
        arr = np.array(idxs, dtype=np.int64)
        language_means[lang] = {
            "pct_var_mean_incorrect": final_metric_mean[arr].mean(axis=0).tolist(),
            "pct_var_max_incorrect": final_metric_max[arr].mean(axis=0).tolist(),
            "rank_drop_correct": final_metric_rank_drop[arr].mean(axis=0).tolist(),
        }

    return {
        "metadata": {
            "n_samples": n_final_samples,
            "n_parallel_questions": n_final_samples // len(languages),
            "n_eval_samples_before_perturb_aggregation": n_eval_samples,
            "languages": list(languages),
            "n_layers": runner.n_layers,
            "n_heads": runner.n_heads,
            "n_kv_heads": runner.n_kv_heads,
            "group_size": group_size,
            "patch_mode": patch_mode,
            "n_patch_units": n_patch_units,
            "d_head": runner.d_head,
            "option_labels": list(runner.option_labels),
            "target_mode": target_mode,
            "choice_order_perturbations": {
                "enabled": perturb_enabled,
                "num_perturbations": perturb_cfg.get("num_perturbations"),
                "include_original": bool(perturb_cfg.get("include_original", False)),
                "seed": int(perturb_cfg.get("seed", 42)),
            },
        },
        "samples": sample_meta,
        "baseline": {
            "option_logits": final_baseline_option_logits,
            "base_margin_mean": final_base_margin_mean,
            "base_margin_max": final_base_margin_max,
            "base_correct_rank": final_base_correct_rank,
        },
        "importance": {
            "pct_var_mean_incorrect": final_metric_mean,
            "pct_var_max_incorrect": final_metric_max,
            "rank_drop_correct": final_metric_rank_drop,
        },
        "flat_vectors": {
            "pct_var_mean_incorrect": flat_mean,
            "pct_var_max_incorrect": flat_max,
            "rank_drop_correct": flat_rank,
        },
        "language_means": language_means,
        "spearman": spearman_summary,
        "jaccard": jaccard_summary,
        "_checkpoint_fingerprint": (
            checkpoint.manifest["fingerprint"] if checkpoint is not None else None
        ),
    }


def save_outputs(cfg: Dict[str, Any], results: Dict[str, Any]) -> None:
    """Write sweep outputs to NPZ/JSON files under run.output_dir.

    Example input: cfg["run"]["output_dir"]="outputs/multilingual_head_patch"
    Example output: files like importance_tensors.npz and metadata.json
    """
    output_dir = Path(cfg["run"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    success_path = output_dir / "_SUCCESS"
    success_path.unlink(missing_ok=True)

    atomic_write_json(output_dir / "samples.json", results["samples"])
    atomic_write_json(output_dir / "metadata.json", results["metadata"])
    atomic_write_json(output_dir / "language_means.json", results["language_means"])

    if results.get("spearman"):
        atomic_write_json(output_dir / "spearman_summary.json", results["spearman"])
    if results.get("jaccard"):
        atomic_write_json(output_dir / "jaccard_summary.json", results["jaccard"])

    # Save tensor-like arrays in NPZ last.
    atomic_save_npz(
        output_dir / "importance_tensors.npz",
        baseline_option_logits=results["baseline"]["option_logits"],
        base_margin_mean=results["baseline"]["base_margin_mean"],
        base_margin_max=results["baseline"]["base_margin_max"],
        base_correct_rank=results["baseline"]["base_correct_rank"],
        pct_var_mean_incorrect=results["importance"]["pct_var_mean_incorrect"],
        pct_var_max_incorrect=results["importance"]["pct_var_max_incorrect"],
        rank_drop_correct=results["importance"]["rank_drop_correct"],
        flat_pct_var_mean_incorrect=results["flat_vectors"]["pct_var_mean_incorrect"],
        flat_pct_var_max_incorrect=results["flat_vectors"]["pct_var_max_incorrect"],
        flat_rank_drop_correct=results["flat_vectors"]["rank_drop_correct"],
    )

    atomic_write_text(
        success_path,
        json.dumps(
            {
                "status": "complete",
                "checkpoint_fingerprint": results.get("_checkpoint_fingerprint"),
            },
            indent=2,
        )
        + "\n",
    )
    print(f"Saved outputs to {output_dir.resolve()}")


def main() -> None:
    """CLI entry point: load config, run sweep, save outputs.

    Example input: --config configs/multilingual_head_patch/patch/config.yaml
    Example output: output files saved to configured directory
    """
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.set)
    _install_graceful_stop_handler(cfg)

    try:
        results = run_sweep(cfg)
        _raise_if_graceful_stop_requested()
        save_outputs(cfg, results)
    except GracefulStopRequested:
        print(
            "Graceful stop complete; all previously completed patch units are checkpointed.",
            flush=True,
        )
        raise SystemExit(GRACEFUL_STOP_EXIT_CODE)


if __name__ == "__main__":
    main()
