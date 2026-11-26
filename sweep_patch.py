#!/usr/bin/env python3
import os
from pathlib import Path
import torch
import einops
from nnsight import LanguageModel
import matplotlib.pyplot as plt
import numpy as np
from ICL_prompt_generator import generate_icl_prompt
from tqdm import trange
import json
import pickle
from typing import Literal, Sequence, Tuple, List, Dict, Any

# -----------------------------
# Config from environment
# -----------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-8B")
NUM_PAIRS = int(os.getenv("NUM_PAIRS", "100"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
PATCH_MODE = os.getenv("PATCH_MODE", "group")
CUDA_DEBUG_SYNC = os.getenv("CUDA_DEBUG_SYNC", "0").lower() in ("1", "true", "yes", "y")
model_name = MODEL_NAME.split("/")[-1]

normal_src = os.getenv("NORMAL_SRC", "city")
normal_tgt = os.getenv("NORMAL_TGT", "capital")
alt_src = os.getenv("ALT_SRC", "university")
alt_tgt = os.getenv("ALT_TGT", "country calling code")

_normal_language_env = os.getenv("NORMAL_LANGUAGE", "0").lower()
normal_language = _normal_language_env in ("1", "true", "yes", "y")

BASE_FOLDER = os.getenv("BASE_FOLDER", "figs_qwen_clean_en_to_es_alt_language/")

DATA_DIR = Path(__file__).resolve().parent / "data"

# create the base folder if it doesn't exist
os.makedirs(BASE_FOLDER, exist_ok=True)


DATA1_PATH = os.getenv("DATA1_PATH", str(DATA_DIR / "es_data.json"))
DATA2_PATH = os.getenv("DATA2_PATH", str(DATA_DIR / "en_data.json"))

OUT_PKL_ENV = os.getenv("OUT_PKL",None)
FIG_PATH_ENV = os.getenv("FIG_PATH", None)

lang1 = DATA1_PATH.split("/")[-1].split("_")[0]
lang2 = DATA2_PATH.split("/")[-1].split("_")[0]


def prompt_direction_for_path(path: str) -> str:
    filename = os.path.basename(path).lower()
    return "rtl" if "ar" in filename else "ltr"


DATA1_PROMPT_DIR = prompt_direction_for_path(DATA1_PATH)
DATA2_PROMPT_DIR = prompt_direction_for_path(DATA2_PATH)

if OUT_PKL_ENV is not None:
    out_pkl = OUT_PKL_ENV
else:
    out_pkl = (
        f"{BASE_FOLDER}{lang2}_to_{lang1}_language_icl_patch_{model_name}_"
        f"{PATCH_MODE}_{NUM_PAIRS}_{normal_src}-{normal_tgt}_{alt_src}-{alt_tgt}.pkl"
    )

if FIG_PATH_ENV is not None:
    fig_path = FIG_PATH_ENV
else:
    fig_path = (
        f"{BASE_FOLDER}{lang2}_to_{lang1}_language_icl_patch_{model_name}_"
        f"{PATCH_MODE}_{NUM_PAIRS}_{normal_src}-{normal_tgt}-{alt_src}-{alt_tgt}.png"
    )

# ---- rest of your script unchanged below this ----


# -----------------------------
# Plotting helper
# -----------------------------
def plot_icl_patching_results(
    ioi_patching_results,
    x_labels,
    plot_title="Normalized Logit Difference After Patching Residual Stream on the ICL Task",
    save_path="icl_patch_results.png",
):
    ioi_patching_results = np.array(ioi_patching_results)

    plt.figure(figsize=(10, 6))
    plt.imshow(
        ioi_patching_results,
        cmap="RdBu_r",
        aspect="auto",
        origin="upper",
        vmin=-0.8,
        vmax=0.8,
    )
    plt.colorbar(label="Norm. Logit Diff")

    plt.title(plot_title)
    plt.ylabel("Layer")
    plt.xlabel("Head / KV-group")
    plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels, rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Plot saved as {save_path}")

# -----------------------------
# Data & helpers
# -----------------------------
data1 = json.load(open(DATA1_PATH, "r"))
data2 = json.load(open(DATA2_PATH, "r"))

def get_alt_answer(
    data: Any, bridge: str, target_type: Literal["capital", "country_code"]
) -> str:
    # leading space for tokenization consistency
    return " " + data[bridge][target_type]

# -----------------------------
# Safety / environment checks
# -----------------------------
assert torch.cuda.is_available(), "CUDA is required for this script."
torch.backends.cuda.matmul.allow_tf32 = True  # minor perf/memory win on Ampere+

# -----------------------------
# Load Qwen model (multi-GPU friendly)
# -----------------------------
llm = LanguageModel(
    MODEL_NAME,
    device_map="auto",
    # torch_dtype=torch.bfloat16
)

llm.model.config.use_cache = False
llm.model.eval()

cfg = llm.model.config
N_LAYERS = cfg.num_hidden_layers
N_HEADS = cfg.num_attention_heads
N_KV_HEADS = getattr(cfg, "num_key_value_heads", N_HEADS)
GROUP_SIZE = N_HEADS // N_KV_HEADS

print(
    f"Model: {MODEL_NAME}\n"
    f"Layers: {N_LAYERS}, Heads: {N_HEADS}, KV heads: {N_KV_HEADS}, "
    f"Group size (heads per KV head): {GROUP_SIZE}"
)

# -----------------------------
# Build prompts & answers
# -----------------------------
normal_prompts: List[str] = []
normal_answers: List[str] = []
alt_prompts: List[str] = []
alt_answers: List[str] = []
counter = 0
# for i in range(NUM_PAIRS):
#     normal_prompt, normal_bridge, normal_answer = generate_icl_prompt(
#         normal_src, normal_tgt, num_examples=20, seed=i, dataset=data1
#     )
#     normal_prompts.append(normal_prompt)
#     normal_answers.append(normal_answer)  # ensure leading space

#     alt_prompt, alt_bridge, alt_answer = generate_icl_prompt(
#         alt_src, alt_tgt, num_examples=20, seed=2 * NUM_PAIRS - i, dataset=data2
#     )
#     if alt_bridge == normal_bridge:
#         counter+=1
#     alt_prompts.append(alt_prompt)
#     if normal_language:
#         alt_answers.append(get_alt_answer(data1, alt_bridge, "capital"))
#     else: 
#         # to make sure the attn head representation is language agnostic, should I use capital or country code?????????????
#         # alt_answers.append(get_alt_answer(data2, alt_bridge, "country_code"))
#         alt_answers.append(get_alt_answer(data2, alt_bridge, "capital"))


for i in range(NUM_PAIRS):
    normal_prompt, normal_bridge, normal_answer = generate_icl_prompt(
        normal_src,
        normal_tgt,
        num_examples=20,
        seed=i,
        dir=DATA1_PROMPT_DIR,
        dataset=data1,
    )
    normal_prompts.append(normal_prompt)
    normal_answers.append(normal_answer)
    found_alt = False
    t = 0
    # Keep resampling alt until bridge differs
    while not found_alt:
        alt_seed = (2 * NUM_PAIRS - i) + t  # deterministic but avoids collisions
        alt_prompt, alt_bridge, alt_answer = generate_icl_prompt(
            alt_src,
            alt_tgt,
            num_examples=20,
            seed=alt_seed,
            dir=DATA2_PROMPT_DIR,
            dataset=data2,
        )
        if alt_bridge != normal_bridge:
            break
        t += 1
        
    alt_prompts.append(alt_prompt)

    if normal_language:
        alt_answers.append(get_alt_answer(data1, alt_bridge, "capital"))
    else:
        alt_answers.append(get_alt_answer(data2, alt_bridge, "capital"))

        
print("num same:", counter)
# print("normal answers:", normal_answers)
# print("alt answers:", alt_answers)

# For Qwen: be explicit about not adding special tokens, and take the first token
normal_answer_token_indices = [
    llm.tokenizer(ans, add_special_tokens=False)["input_ids"][0] for ans in normal_answers
]
alt_answer_token_indices = [
    llm.tokenizer(ans, add_special_tokens=False)["input_ids"][0] for ans in alt_answers
]
# print("Tokenization Example:", llm.tokenizer(normal_answers, add_special_tokens=False))
# print("Decoded Example:", [llm.tokenizer.decode(i) for i in llm.tokenizer(normal_answers, add_special_tokens=False)["input_ids"][0]])
# print("normal answer token indices: ", normal_answer_token_indices)
# print("alt answer token indices: ", alt_answer_token_indices)

# -----------------------------
# Core patching function (Qwen-aware, supports group/head mode)
# -----------------------------
def sweep_last_token_patching_qwen(
    llm: LanguageModel,
    normal_prompts: Sequence[str],
    alt_prompts: Sequence[str],
    normal_ids: Sequence[int],
    alt_ids: Sequence[int],
    *,
    layer_range: Tuple[int, int] | None = None,
    patch_mode: Literal["head", "group"] = "head",
    verbose: bool = False,
) -> Tuple[List[List[float]], List[List[float]], List[float]]:
    """
    Batched last-token patching for Qwen models, with GQA support.

    patch_mode:
      - "head"  : treat each logical attention head independently (like standard MHA)
      - "group" : patch KV groups (all heads that share one KV head) together

    Returns:
      - results_mean: [layers x (heads or groups)] mean normalized effect across batch
      - results_per_example: same shape, each cell is list of per-example effects
      - base_diffs: list of baseline logit diffs (one per pair)
    """
    assert len(normal_prompts) == len(alt_prompts)
    assert len(normal_ids) == len(alt_ids) == len(normal_prompts)

    B = len(normal_prompts)
    N_LAYERS = llm.model.config.num_hidden_layers
    N_HEADS = llm.model.config.num_attention_heads
    N_KV_HEADS = getattr(llm.model.config, "num_key_value_heads", N_HEADS)
    
    print("N_KV_HEADS =", N_KV_HEADS)
    print("GROUP_SIZE =", N_HEADS // N_KV_HEADS)
    D_MODEL = llm.model.config.hidden_size
    D_HEAD = D_MODEL // N_HEADS
    
    

    start_L, end_L = layer_range or (0, N_LAYERS)

    normal_ids_t = torch.tensor(normal_ids, dtype=torch.long)
    alt_ids_t = torch.tensor(alt_ids, dtype=torch.long)

    results_mean: List[List[float]] = []
    results_per_example: List[List[List[float]]] = []
    base_diffs_list: List[float] = []

    with torch.no_grad():
        with llm.session(remote=False):

            # ---- STEP 1: ALT donors per layer (CPU), last token only ----
            alt_last_per_layer: List[torch.Tensor] = []
            with llm.trace() as tracer:
                with tracer.invoke(alt_prompts):
                    for L in range(N_LAYERS):
                        z = llm.model.layers[L].self_attn.o_proj.input  # [B, T, D_MODEL]
                        z_last = z[:, -1, :]                            # [B, D_MODEL]
                        donors = einops.rearrange(
                            z_last, "b (h d) -> b h d", h=N_HEADS
                        )  # [B, H, D_HEAD]
                        alt_last_per_layer.append(donors.detach().cpu())

            # ---- STEP 2: Baseline diffs on NORMAL (per example) ----
            with llm.trace() as tracer:
                with tracer.invoke(normal_prompts):
                    logits = llm.lm_head.output  # [B, T, V]
                    dev = logits.device
                    batch_ix = torch.arange(B, device=dev)
                    n_ids = normal_ids_t.to(dev)
                    a_ids = alt_ids_t.to(dev)
                    base = logits[batch_ix, -1, n_ids] - logits[batch_ix, -1, a_ids]  # [B]
                    base_diffs_list = base.detach().cpu().tolist()

            if verbose:
                mb = float(sum(base_diffs_list) / max(len(base_diffs_list), 1))
                print(f"Mean baseline logit diff: {mb:.3f}")

            # ---- STEP 3: Patch & measure for each head or group ----
            for L in trange(start_L, end_L, desc="Layers"):
                row_mean: List[float] = []
                row_per_ex: List[List[float]] = []

                donor_all_heads_cpu = alt_last_per_layer[L]  # [B, H, D_HEAD] on CPU

                if patch_mode == "head":
                    # treat each logical head separately
                    iterable = range(N_HEADS)
                    def heads_for_index(H_idx: int) -> List[int]:
                        return [H_idx]
                else:
                    # "group" mode: each index represents a KV group
                    group_size = N_HEADS // N_KV_HEADS
                    iterable = range(N_KV_HEADS)
                    def heads_for_index(g: int) -> List[int]:
                        start_h = g * group_size
                        end_h = (g + 1) * group_size
                        return list(range(start_h, end_h))

                for idx in iterable:
                    try:
                        with llm.trace() as tracer:
                            with tracer.invoke(normal_prompts):
                                z_live = llm.model.layers[L].self_attn.o_proj.input  # [B, T, D_MODEL]
                                live_dev = z_live.device
                                z_mod = z_live.clone()

                                # Patch all heads in this head/group index. Keep donors
                                # on the live device only for the duration of this trace.
                                for h in heads_for_index(idx):
                                    donor_vec = donor_all_heads_cpu[:, h, :].to(live_dev)

                                    start, end = h * D_HEAD, (h + 1) * D_HEAD
                                    z_mod[:, -1, start:end] = donor_vec

                                llm.model.layers[L].self_attn.o_proj.input = z_mod

                                patched_logits = llm.lm_head.output  # [B, T, V]
                                batch_ix = torch.arange(B, device=patched_logits.device)
                                n_ids = normal_ids_t.to(patched_logits.device)
                                a_ids = alt_ids_t.to(patched_logits.device)

                                pdiff = (
                                    patched_logits[batch_ix, -1, n_ids]
                                    - patched_logits[batch_ix, -1, a_ids]
                                )  # [B]
                                pdiff_cpu = pdiff.detach().cpu().tolist()

                                if CUDA_DEBUG_SYNC:
                                    torch.cuda.synchronize(patched_logits.device)

                                per_ex: List[float] = []
                                for b, p in zip(base_diffs_list, pdiff_cpu):
                                    denom = b if abs(b) > 1e-8 else 1e-8
                                    per_ex.append((b - p) / denom)

                                row_per_ex.append(per_ex)
                                row_mean.append(sum(per_ex) / len(per_ex))
                    except Exception as e:
                        raise RuntimeError(
                            f"CUDA/NNsight failure while patching layer={L}, "
                            f"{patch_mode}_index={idx}, heads={heads_for_index(idx)}"
                        ) from e

                results_per_example.append(row_per_ex)
                results_mean.append(row_mean)

            del alt_last_per_layer
    # clean vram

    torch.cuda.empty_cache()

    return results_mean, results_per_example, base_diffs_list

# -----------------------------
# Run batched sweep over all pairs
# -----------------------------
mean_results_all: np.ndarray | None = None
per_example_results_all: List[List[List[float]]] = []
base_diffs_all: List[float] = []
num_batches = 0

for batch_start in trange(0, NUM_PAIRS, BATCH_SIZE, desc="Batching pairs"):
    batch_end = min(batch_start + BATCH_SIZE, NUM_PAIRS)
    print(f"Processing pairs {batch_start} to {batch_end - 1}...")

    normal_prompts_batch = normal_prompts[batch_start:batch_end]
    alt_prompts_batch = alt_prompts[batch_start:batch_end]
    normal_answer_token_indices_batch = normal_answer_token_indices[batch_start:batch_end]
    alt_answer_token_indices_batch = alt_answer_token_indices[batch_start:batch_end]

    results_mean, results_per_examples, base_diffs_list = sweep_last_token_patching_qwen(
        llm,
        normal_prompts_batch,
        alt_prompts_batch,
        normal_answer_token_indices_batch,
        alt_answer_token_indices_batch,
        patch_mode=PATCH_MODE,
    )

    results_mean_arr = np.array(results_mean, dtype=np.float32)
    if mean_results_all is None:
        mean_results_all = results_mean_arr
    else:
        mean_results_all += results_mean_arr

    per_example_results_all.extend(results_per_examples)
    base_diffs_all.extend(base_diffs_list)
    num_batches += 1
    
    import gc
    del results_mean, results_per_examples, base_diffs_list
    torch.cuda.empty_cache()
    gc.collect()


# average over batches
mean_results_all /= max(num_batches, 1)

# -----------------------------
# Save results
# -----------------------------
with open(out_pkl, "wb") as f:
    pickle.dump(
        {
            "mean_results": mean_results_all,
            "per_example_results": per_example_results_all,
            "base_diffs": base_diffs_all,
        },
        f,
    )
print(f"✅ Saved results to {out_pkl}")


# load the pickle file

# with open(out_pkl, "rb") as f:
#     data = pickle.load(f)
#     mean_results_all = data["mean_results"]
#     per_example_results_all = data["per_example_results"]
#     base_diffs_all = data["base_diffs"]


# -----------------------------
# Report top heads/groups
# -----------------------------
mean_results_array = np.array(mean_results_all)
flat_indices = np.argsort(mean_results_array.flatten())[::-1]
top_k = 10

if PATCH_MODE == "head":
    label_type = "Head"
    num_x = N_HEADS
else:
    label_type = "KV-group"
    num_x = N_KV_HEADS

print(f"Top {top_k} {label_type}s with highest mean effect:")
for idx in flat_indices[:top_k]:
    layer = idx // mean_results_array.shape[1]
    col = idx % mean_results_array.shape[1]
    effect = mean_results_array[layer, col]
    print(f"Layer {layer}, {label_type} {col}: Mean Effect = {effect:.4f}")

# -----------------------------
# Plot & save heatmap
# -----------------------------
if PATCH_MODE == "head":
    x_labels = [f"Head {i}" for i in range(N_HEADS)]
else:
    x_labels = [f"G{g}" for g in range(N_KV_HEADS)]


plot_icl_patching_results(
    mean_results_all,
    x_labels,
    f"Patching {model_name} Attention ({PATCH_MODE}) on ICL task (last token)",
    fig_path,
)
