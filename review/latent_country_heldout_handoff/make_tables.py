"""Build held-out latent-country handoff tables from the saved per-pair records.

CPU only; no model is loaded. Uses the run's own bootstrap (same seeds and
resampling as review/latent_country/statistics.py) and asserts that its top-1
estimates and CIs reproduce each run's saved summary.json before writing.

From the thesis3 root:  python3 review/latent_country_heldout_handoff/make_tables.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from review.latent_country.protocol import stable_seed  # noqa: E402

OUT = Path(__file__).resolve().parent
RUNS = {"Qwen3-8B": ROOT / "review/runs/latent_country/qwen_published",
        "Llama-3.1-8B": ROOT / "review/runs/latent_country/llama_published"}
ORIGINAL = ROOT / "quantitative_experiments/results/multilingual_head_patch/statistical_support/table5_latent_country_aggregate.csv"


def load(run: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((run / "records").glob("*.json"))]


def stats(records: list[dict], draws: int, seed: int) -> dict:
    conditions = sorted({r["condition"] for r in records})
    indices = sorted({r["pair_index"] for r in records})
    by_key = {(r["condition"], r["pair_index"]): r for r in records}
    assert len(by_key) == len(records) == len(conditions) * len(indices)
    grid = [[by_key[(c, i)] for i in indices] for c in conditions]
    before = np.array([[r["baseline"]["target_rank"] for r in row] for row in grid])
    after = np.array([[r["selected"]["target_rank"] for r in row] for row in grid])
    rand = np.array([[[x["target_rank"] for x in r["random"]] for r in row] for row in grid])

    def estimates(ids):
        b1 = (before[:, ids] == 1).mean(axis=(0, 2)) * 100
        a1 = (after[:, ids] == 1).mean(axis=(0, 2)) * 100
        r1 = (rand[:, ids, :] == 1).mean(axis=(0, 2, 3)) * 100
        b10 = (before[:, ids] <= 10).mean(axis=(0, 2)) * 100
        a10 = (after[:, ids] <= 10).mean(axis=(0, 2)) * 100
        return {"top1_before": b1, "top1_after": a1, "delta_top1": a1 - b1,
                "top10_before": b10, "top10_after": a10, "delta_top10": a10 - b10,
                "median_rank_before": np.median(before[:, ids], axis=2).mean(axis=0),
                "median_rank_after": np.median(after[:, ids], axis=2).mean(axis=0),
                "mean_rank_before": before[:, ids].mean(axis=(0, 2)),
                "mean_rank_after": after[:, ids].mean(axis=(0, 2)),
                "random_top1": r1, "gain_over_random": a1 - r1}

    point = estimates(np.arange(len(indices))[None, :])
    samples = {k: [] for k in point}
    rng = np.random.default_rng(seed)  # identical draw sequence to statistics.py
    for offset in range(0, draws, 128):
        ids = rng.integers(len(indices), size=(min(128, draws - offset), len(indices)))
        for k, v in estimates(ids).items():
            samples[k].extend(v.tolist())
    result = {k: float(v[0]) for k, v in point.items()}
    for k in point:
        result[f"{k}_ci_low"], result[f"{k}_ci_high"] = np.quantile(samples[k], [0.025, 0.975]).tolist()
    result.update(n_conditions=len(conditions), n_pair_clusters=len(indices), n_records=len(records),
                  answer_token_collisions=int(sum(r["answer_token_collision"] for r in records)))
    return result


COLUMNS = ["n_pairs", "n_conditions", "n_records", "top1_before", "top1_after", "delta_top1",
           "delta_top1_ci_low", "delta_top1_ci_high", "top10_before", "top10_after",
           "median_rank_before", "median_rank_after"]
EXTRA = ["mean_rank_before", "mean_rank_after", "delta_top10", "delta_top10_ci_low", "delta_top10_ci_high",
         "top1_after_ci_low", "top1_after_ci_high", "top10_after_ci_low", "top10_after_ci_high",
         "median_rank_after_ci_low", "median_rank_after_ci_high", "random_top1", "gain_over_random",
         "gain_over_random_ci_low", "gain_over_random_ci_high", "answer_token_collisions"]


def row(prefix: dict, s: dict) -> dict:
    out = dict(prefix)
    out["n_pairs"] = s["n_pair_clusters"]
    for k in COLUMNS[1:] + EXTRA:
        out[k] = s[k]
    return out


def write(name: str, rows: list[dict]) -> None:
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def main() -> None:
    aggregate, per_condition = [], []
    for model, run in RUNS.items():
        summary = json.loads((run / "summary.json").read_text())
        draws, seed = summary["bootstrap_resamples"], summary["bootstrap_seed"]
        records = load(run)
        groups = {"monolingual": [r for r in records if r["donor_language"] == r["recipient_language"]],
                  "cross_lingual": [r for r in records if r["donor_language"] != r["recipient_language"]]}
        groups.update({c: [r for r in records if r["condition"] == c] for c in sorted({r["condition"] for r in records})})
        for name, rows in groups.items():
            s = stats(rows, draws, stable_seed(seed, name))
            saved = summary["groups"][name]["metrics"]
            for mine, theirs in (("delta_top1", "selected_minus_baseline_top1_pp"),
                                 ("gain_over_random", "selected_minus_random_top1_pp"),
                                 ("top1_after", "selected_top1_percent")):
                assert np.isclose(s[mine], saved[theirs]["estimate"]), (model, name, mine)
                assert np.allclose([s[f"{mine}_ci_low"], s[f"{mine}_ci_high"]], saved[theirs]["ci95"]), (model, name, mine)
            if name in ("monolingual", "cross_lingual"):
                aggregate.append(row({"model": model, "setting": name}, s))
            else:
                first = rows[0]
                per_condition.append(row({"model": model, "setting": "monolingual" if first["donor_language"] == first["recipient_language"] else "cross_lingual",
                                          "donor_language": first["donor_language"], "recipient_language": first["recipient_language"],
                                          "donor_family": first["donor_source"]}, s))
    write("1_heldout_aggregate.csv", aggregate)
    write("5_heldout_per_condition.csv", per_condition)

    # Original (non-held-out) Table 5 next to the held-out values, for the writer's convenience.
    original = {}
    for r in csv.DictReader(ORIGINAL.open()):
        if r["scope"] == "model_setting_macro_average":
            original[(r["model"], r["setting"].replace("-", "_").lower(), r["metric"])] = r
    names = {"Qwen3-8B": "Qwen", "Llama-3.1-8B": "Llama"}
    comparison = []
    for a in aggregate:
        key = (names[a["model"]], a["setting"])
        get = lambda m, f="estimate": 100 * float(original[(*key, m)][f]) if f != "estimate" or m != "patched_median_rank" else float(original[(*key, m)][f])
        comparison.append({"model": a["model"], "setting": a["setting"],
                           "original_top1_before": get("baseline_top1_rate"), "heldout_top1_before": a["top1_before"],
                           "original_top1_after": get("patched_top1_rate"), "heldout_top1_after": a["top1_after"],
                           "original_delta_top1": get("delta_top1_rate"),
                           "original_delta_ci_low": get("delta_top1_rate", "ci_low"),
                           "original_delta_ci_high": get("delta_top1_rate", "ci_high"),
                           "heldout_delta_top1": a["delta_top1"], "heldout_delta_ci_low": a["delta_top1_ci_low"],
                           "heldout_delta_ci_high": a["delta_top1_ci_high"],
                           "original_top10_after": get("patched_top10_rate"), "heldout_top10_after": a["top10_after"],
                           "original_median_rank_after": float(original[(*key, "patched_median_rank")]["estimate"]),
                           "heldout_median_rank_after": a["median_rank_after"]})
    write("1b_original_vs_heldout.csv", comparison)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
