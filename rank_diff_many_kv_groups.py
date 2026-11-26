#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Sequence

import einops
import torch
from nnsight import LanguageModel
from tqdm import trange

from ICL_prompt_generator import generate_icl_prompt


MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-8B")
NUM_PAIRS = int(os.getenv("NUM_PAIRS", "100"))
NUM_EXAMPLES = int(os.getenv("NUM_EXAMPLES", "20"))
NORMAL_SRC = os.getenv("NORMAL_SRC", "city")
NORMAL_TGT = os.getenv("NORMAL_TGT", "capital")
ALT_SRC = os.getenv("ALT_SRC", "university")
ALT_TGT = os.getenv("ALT_TGT", "country calling code")
PATCH_SPEC = os.getenv("PATCH_SPEC", "30:6;33:1,7;34:1,2")
BASE_FOLDER = os.getenv("BASE_FOLDER", "rank_diff_results/")

_normal_language_env = os.getenv("NORMAL_LANGUAGE", "0").lower()
NORMAL_LANGUAGE = _normal_language_env in ("1", "true", "yes", "y")

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA1_PATH = os.getenv("DATA1_PATH", str(DATA_DIR / "es_data.json"))
DATA2_PATH = os.getenv("DATA2_PATH", str(DATA_DIR / "en_data.json"))
OUT_JSON_ENV = os.getenv("OUT_JSON")

MODEL_SHORT_NAME = MODEL_NAME.split("/")[-1]
LANG1 = Path(DATA1_PATH).name.split("_")[0]
LANG2 = Path(DATA2_PATH).name.split("_")[0]


def prompt_direction_for_path(path: str) -> str:
    filename = os.path.basename(path).lower()
    return "rtl" if "ar" in filename else "ltr"


DATA1_PROMPT_DIR = prompt_direction_for_path(DATA1_PATH)
DATA2_PROMPT_DIR = prompt_direction_for_path(DATA2_PATH)


def sanitize_filename_part(value: str) -> str:
    return value.replace(" ", "_").replace("/", "-")


def parse_patch_spec(spec: str) -> Dict[int, List[int]]:
    """
    Parse a semicolon-separated patch spec like:
      "30:6;33:1,7;34:1,2"
    into:
      {30: [6], 33: [1, 7], 34: [1, 2]}
    """
    parsed: Dict[int, List[int]] = {}
    cleaned = spec.strip()
    if not cleaned:
        raise ValueError("PATCH_SPEC must not be empty.")

    for chunk in cleaned.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(
                f"Invalid PATCH_SPEC entry '{chunk}'. Expected format like '30:6,7'."
            )
        layer_str, groups_str = chunk.split(":", 1)
        layer_idx = int(layer_str.strip())
        groups = [int(part.strip()) for part in groups_str.split(",") if part.strip()]
        if not groups:
            raise ValueError(f"No KV groups provided for layer {layer_idx}.")
        parsed[layer_idx] = groups

    if not parsed:
        raise ValueError("PATCH_SPEC did not contain any valid layer/group entries.")

    return parsed


def patch_spec_for_filename(layers_to_kv_groups: Mapping[int, Sequence[int]]) -> str:
    parts: List[str] = []
    for layer_idx in sorted(layers_to_kv_groups):
        groups = ",".join(str(group) for group in layers_to_kv_groups[layer_idx])
        parts.append(f"L{layer_idx}-G{groups}")
    return "__".join(parts)


def token_rank(logits_1d: torch.Tensor, token_id: int) -> int:
    val = logits_1d[token_id]
    higher = (logits_1d > val).sum().item()
    return int(higher + 1)


def get_alt_answer(
    data: Any, bridge: str, target_type: Literal["capital", "country_code"]
) -> str:
    return " " + data[bridge][target_type]


def patch_many_kv_groups_and_get_ranks(
    llm: LanguageModel,
    normal_prompt: str,
    alt_prompt: str,
    layers_to_kv_groups: Mapping[int, Sequence[int]],
    normal_token_id: int,
    alt_token_id: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Patch multiple KV groups across layers using donors from alt_prompt, then
    measure ranks and logit diffs on the normal prompt.
    """
    n_layers = llm.model.config.num_hidden_layers
    n_heads = llm.model.config.num_attention_heads
    d_model = llm.model.config.hidden_size
    n_kv_heads = getattr(llm.model.config, "num_key_value_heads", n_heads)

    assert n_heads % n_kv_heads == 0, "N_HEADS must be divisible by N_KV_HEADS for GQA."
    d_head = d_model // n_heads
    group_size = n_heads // n_kv_heads

    for layer_idx, kv_groups in layers_to_kv_groups.items():
        assert 0 <= layer_idx < n_layers, f"layer_idx out of range: {layer_idx}"
        for group_idx in kv_groups:
            assert 0 <= group_idx < n_kv_heads, (
                f"kv_group_idx out of range: {group_idx} in layer {layer_idx}"
            )

    result: Dict[str, Any] = {
        "patched_kv_groups": {int(k): list(v) for k, v in layers_to_kv_groups.items()},
        "baseline": {"normal_rank": None, "alt_rank": None, "logit_diff": None},
        "patched": {"normal_rank": None, "alt_rank": None, "logit_diff": None},
    }

    donor_vecs: Dict[int, Dict[int, torch.Tensor]] = {
        layer_idx: {} for layer_idx in layers_to_kv_groups.keys()
    }

    with torch.no_grad():
        with llm.session(remote=False):
            if verbose:
                print(f"[donor] tracing alt prompt for {layers_to_kv_groups}")
            with llm.trace() as tracer:
                with tracer.invoke(alt_prompt):
                    for layer_idx, kv_groups in layers_to_kv_groups.items():
                        z = llm.model.layers[layer_idx].self_attn.o_proj.input
                        z_last = z[:, -1, :]
                        donors = einops.rearrange(
                            z_last, "b (h d) -> b h d", h=n_heads
                        )

                        for group_idx in kv_groups:
                            head_start = group_idx * group_size
                            head_end = (group_idx + 1) * group_size
                            donor_vecs[layer_idx][group_idx] = (
                                donors[:, head_start:head_end, :].detach().cpu()
                            )

            if verbose:
                print("[baseline] tracing normal prompt")
            with llm.trace() as tracer:
                with tracer.invoke(normal_prompt):
                    logits = llm.lm_head.output
                    if logits is None:
                        raise RuntimeError("lm_head.output is None during baseline trace.")
                    last = logits[0, -1, :]
                    result["baseline"]["normal_rank"] = token_rank(last, normal_token_id)
                    result["baseline"]["alt_rank"] = token_rank(last, alt_token_id)
                    result["baseline"]["logit_diff"] = (
                        last[normal_token_id] - last[alt_token_id]
                    ).item()

            if verbose:
                print(f"[patch] applying donors for {layers_to_kv_groups}")
            with llm.trace() as tracer:
                with tracer.invoke(normal_prompt):
                    for layer_idx, kv_groups in layers_to_kv_groups.items():
                        z_live = llm.model.layers[layer_idx].self_attn.o_proj.input
                        z_mod = z_live.clone()

                        for group_idx in kv_groups:
                            head_start = group_idx * group_size
                            head_end = (group_idx + 1) * group_size
                            donor_group = donor_vecs[layer_idx][group_idx].to(z_live.device)

                            for local_idx, head_idx in enumerate(range(head_start, head_end)):
                                start, end = head_idx * d_head, (head_idx + 1) * d_head
                                z_mod[:, -1, start:end] = donor_group[:, local_idx, :]

                        llm.model.layers[layer_idx].self_attn.o_proj.input = z_mod

                    patched = llm.lm_head.output
                    if patched is None:
                        raise RuntimeError("lm_head.output is None after patch.")
                    patched_last = patched[0, -1, :]
                    result["patched"]["normal_rank"] = token_rank(
                        patched_last, normal_token_id
                    )
                    result["patched"]["alt_rank"] = token_rank(patched_last, alt_token_id)
                    result["patched"]["logit_diff"] = (
                        patched_last[normal_token_id] - patched_last[alt_token_id]
                    ).item()

    torch.cuda.empty_cache()
    return result


def main() -> None:
    os.makedirs(BASE_FOLDER, exist_ok=True)

    layers_to_kv_groups = parse_patch_spec(PATCH_SPEC)
    patch_spec_name = patch_spec_for_filename(layers_to_kv_groups)

    if OUT_JSON_ENV is not None:
        out_json = OUT_JSON_ENV
    else:
        out_json = os.path.join(
            BASE_FOLDER,
            f"{LANG2}_to_{LANG1}_language_rank_diff_{MODEL_SHORT_NAME}_"
            f"{patch_spec_name}_{NUM_PAIRS}_"
            f"{sanitize_filename_part(NORMAL_SRC)}-{sanitize_filename_part(NORMAL_TGT)}_"
            f"{sanitize_filename_part(ALT_SRC)}-{sanitize_filename_part(ALT_TGT)}.json",
        )

    data1 = json.load(open(DATA1_PATH, "r", encoding="utf-8"))
    data2 = json.load(open(DATA2_PATH, "r", encoding="utf-8"))

    assert torch.cuda.is_available(), "CUDA is required for this script."
    torch.backends.cuda.matmul.allow_tf32 = True

    llm = LanguageModel(
        MODEL_NAME,
        device_map="auto",
    )
    llm.model.config.use_cache = False
    llm.model.eval()

    cfg = llm.model.config
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads)

    print(
        f"Model: {MODEL_NAME}\n"
        f"Layers: {n_layers}, Heads: {n_heads}, KV heads: {n_kv_heads}\n"
        f"Data1: {DATA1_PATH} ({DATA1_PROMPT_DIR}), Data2: {DATA2_PATH} ({DATA2_PROMPT_DIR})\n"
        f"Patch spec: {layers_to_kv_groups}\n"
        f"Output: {out_json}"
    )

    normal_prompts: List[str] = []
    normal_answers: List[str] = []
    alt_prompts: List[str] = []
    alt_answers: List[str] = []

    for i in range(NUM_PAIRS):
        normal_prompt, normal_bridge, normal_answer = generate_icl_prompt(
            NORMAL_SRC,
            NORMAL_TGT,
            num_examples=NUM_EXAMPLES,
            seed=i,
            dir=DATA1_PROMPT_DIR,
            dataset=data1,
        )
        normal_prompts.append(normal_prompt)
        normal_answers.append(normal_answer)

        found_alt = False
        t = 0
        while not found_alt:
            alt_seed = (2 * NUM_PAIRS - i) + t
            alt_prompt, alt_bridge, _ = generate_icl_prompt(
                ALT_SRC,
                ALT_TGT,
                num_examples=NUM_EXAMPLES,
                seed=alt_seed,
                dir=DATA2_PROMPT_DIR,
                dataset=data2,
            )
            if alt_bridge != normal_bridge:
                found_alt = True
            else:
                t += 1

        alt_prompts.append(alt_prompt)
        if NORMAL_LANGUAGE:
            alt_answers.append(get_alt_answer(data1, alt_bridge, "capital"))
        else:
            alt_answers.append(get_alt_answer(data2, alt_bridge, "capital"))

    normal_answer_token_indices = [
        llm.tokenizer(ans, add_special_tokens=False)["input_ids"][0]
        for ans in normal_answers
    ]
    alt_answer_token_indices = [
        llm.tokenizer(ans, add_special_tokens=False)["input_ids"][0]
        for ans in alt_answers
    ]

    all_results: List[Dict[str, Any]] = []
    for i in trange(NUM_PAIRS, desc="Rank diff pairs"):
        result = patch_many_kv_groups_and_get_ranks(
            llm=llm,
            normal_prompt=normal_prompts[i],
            alt_prompt=alt_prompts[i],
            layers_to_kv_groups=layers_to_kv_groups,
            normal_token_id=normal_answer_token_indices[i],
            alt_token_id=alt_answer_token_indices[i],
            verbose=False,
        )
        all_results.append(result)

    baseline_alt_ranks = [
        result["baseline"]["alt_rank"]
        for result in all_results
        if result["baseline"]["alt_rank"] is not None
    ]
    patched_alt_ranks = [
        result["patched"]["alt_rank"]
        for result in all_results
        if result["patched"]["alt_rank"] is not None
    ]

    n = len(patched_alt_ranks)
    num_baseline_rank1_alt = sum(rank == 1 for rank in baseline_alt_ranks)
    num_patched_rank1_alt = sum(rank == 1 for rank in patched_alt_ranks)
    num_patched_top10_alt = sum(rank <= 10 for rank in patched_alt_ranks)

    print(f"# pairs used: {n}")
    print(
        f"Baseline: alt ranked #1 in {num_baseline_rank1_alt} / {n} "
        f"({num_baseline_rank1_alt / max(n, 1):.3f})"
    )
    print(
        f"Patched : alt ranked #1 in {num_patched_rank1_alt} / {n} "
        f"({num_patched_rank1_alt / max(n, 1):.3f})"
    )
    print(
        f"Patched : alt in top 10 in {num_patched_top10_alt} / {n} "
        f"({num_patched_top10_alt / max(n, 1):.3f})"
    )

    per_pair_summary = [
        {
            "pair_index": i,
            "baseline": all_results[i]["baseline"],
            "patched": all_results[i]["patched"],
        }
        for i in range(len(all_results))
    ]

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "model_name": MODEL_NAME,
                    "num_pairs": NUM_PAIRS,
                    "num_examples": NUM_EXAMPLES,
                    "normal_src": NORMAL_SRC,
                    "normal_tgt": NORMAL_TGT,
                    "alt_src": ALT_SRC,
                    "alt_tgt": ALT_TGT,
                    "normal_language": NORMAL_LANGUAGE,
                    "data1_path": DATA1_PATH,
                    "data2_path": DATA2_PATH,
                    "data1_prompt_dir": DATA1_PROMPT_DIR,
                    "data2_prompt_dir": DATA2_PROMPT_DIR,
                    "layers_to_kv_groups": {
                        str(k): list(v) for k, v in layers_to_kv_groups.items()
                    },
                },
                "summary": {
                    "num_pairs_used": n,
                    "num_baseline_rank1_alt": num_baseline_rank1_alt,
                    "num_patched_rank1_alt": num_patched_rank1_alt,
                    "num_patched_top10_alt": num_patched_top10_alt,
                    "baseline_rank1_rate": num_baseline_rank1_alt / max(n, 1),
                    "patched_rank1_rate": num_patched_rank1_alt / max(n, 1),
                    "patched_top10_rate": num_patched_top10_alt / max(n, 1),
                },
                "per_pair": per_pair_summary,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved results to {out_json}")


if __name__ == "__main__":
    main()
