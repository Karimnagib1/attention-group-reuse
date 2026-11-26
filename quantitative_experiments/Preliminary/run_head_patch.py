"""Head-level vs group-level patching on a fixed 50-question Qwen3-8B / MMLU-ProX subset.

Thin wrapper around the unchanged group-level pipeline
(quantitative_experiments.multilingual_head_patch.run). Everything comes from the
paper config configs/multilingual_head_patch/patch/config_mmluprox.yaml:
language-mean donor, final-token o_proj input, target_mode=correct,
pct_var_mean_incorrect, original + 3 answer-order permutations.

Precision is pinned to float32. The config says dtype: auto, which the runner passes as
"no dtype". Transformers 4.x then loads float32, which is what the paper run used. Transformers
5.x, now installed in `thesis`, instead loads the checkpoint's bfloat16, and run_sweep's
.numpy() calls fail on bf16 logits. Pinning float32 reproduces the paper run's precision.

Otherwise only three things change:
  1. recipients are 50 questions sampled with a fixed seed from the all-five-correct list;
  2. patching.patch_mode is set from --mode (head | kv_group);
  3. the donor mean is still computed over the full 1,839-question pool, as in the
     paper run. The stock runner averages donors over whatever questions it loads,
     so a 50-question run would otherwise silently use 50-question means.

Usage (from quantitative_experiments/):
    python Preliminary/run_head_patch.py --mode head
    python Preliminary/run_head_patch.py --mode kv_group
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

QUANT_ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(QUANT_ROOT / "src"), str(QUANT_ROOT)]

from quantitative_experiments.multilingual_head_patch import run as hp  # noqa: E402

BASE_CONFIG = "configs/multilingual_head_patch/patch/config_mmluprox.yaml"
POOL = "qwen_outputs/mmluprox/all_correct_ids.json"  # 1,839 all-five-correct questions
RUN_DIR = Path("Preliminary/runs/qwen_mmluprox_50")
IDS_PATH = RUN_DIR / "question_ids.json"


def select_questions(n: int, seed: int) -> list[str]:
    pool = sorted(str(x) for x in json.loads(Path(POOL).read_text()))
    ids = sorted(random.Random(seed).sample(pool, n))
    if IDS_PATH.exists():
        if json.loads(IDS_PATH.read_text()) != ids:
            raise ValueError(f"{IDS_PATH} exists with a different selection; use a new RUN_DIR.")
    else:
        IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        IDS_PATH.write_text(json.dumps(ids, indent=2) + "\n")
    return ids


def use_full_pool_donors(cfg: dict, expected_pool: int) -> None:
    """Make run_sweep compute language-mean donors over the full eligible pool."""
    pool_cfg = copy.deepcopy(cfg)
    pool_cfg["data"]["allowed_parallel_ids_path"] = POOL
    original = hp._compute_language_donors

    def donors(runner, samples, languages, batch_size):
        pool_samples, pool_languages = hp._build_parallel_samples(pool_cfg)
        if list(pool_languages) != list(languages):
            raise ValueError("Donor pool languages differ from recipient languages.")
        if len(pool_samples) != expected_pool * len(languages):
            raise ValueError(f"Expected {expected_pool} pool questions, got {len(pool_samples) // len(languages)}.")
        print(f"Donor means over the full pool: {expected_pool} questions x {len(languages)} languages.", flush=True)
        return original(runner, pool_samples, languages, batch_size)

    hp._compute_language_donors = donors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["head", "kv_group"], required=True)
    parser.add_argument("--n-questions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--set", action="append", default=[], help="Extra config overrides, e.g. patching.batch_size=16")
    args = parser.parse_args()

    os.chdir(QUANT_ROOT)  # config paths are relative to quantitative_experiments/
    ids = select_questions(args.n_questions, args.seed)
    pool_size = len(json.loads(Path(POOL).read_text()))
    output_dir = RUN_DIR / args.mode

    cfg = hp.load_config([BASE_CONFIG])
    cfg = hp.apply_overrides(cfg, [
        "model.dtype=float32",  # see module docstring: matches the paper run under Transformers 4.x
        f"patching.patch_mode={args.mode}",
        f"data.allowed_parallel_ids_path={IDS_PATH}",
        f"run.output_dir={output_dir}",
        "run.checkpoint.enabled=true",
        "run.checkpoint.resume=true",
        *args.set,
    ])
    use_full_pool_donors(cfg, pool_size)

    import torch, transformers  # noqa: E401
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "preliminary_provenance.json").write_text(json.dumps({
        "mode": args.mode, "n_questions": len(ids), "selection_seed": args.seed,
        "selection": f"random.Random({args.seed}).sample(sorted({POOL}), {args.n_questions})",
        "question_ids_path": str(IDS_PATH), "donor_pool": POOL, "donor_pool_size": pool_size,
        "base_config": BASE_CONFIG, "overrides": args.set,
        "transformers": transformers.__version__, "torch": torch.__version__,
    }, indent=2) + "\n")

    hp._install_graceful_stop_handler(cfg)
    try:
        results = hp.run_sweep(cfg)
        hp._raise_if_graceful_stop_requested()
        hp.save_outputs(cfg, results)
    except hp.GracefulStopRequested:
        print("Graceful stop: completed patch units are checkpointed.", flush=True)
        raise SystemExit(hp.GRACEFUL_STOP_EXIT_CODE)


if __name__ == "__main__":
    main()
