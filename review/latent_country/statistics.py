"""Paired question-cluster summaries, conditional on the frozen random-site bank."""
from __future__ import annotations

from collections import Counter
import numpy as np

from .protocol import stable_seed


def random_sites(sites: list, n_groups: int, draws: int, seed: int) -> list:
    counts = Counter(layer for layer, group in sites)
    if draws < 1 or any(count > n_groups for count in counts.values()):
        raise ValueError("Invalid random control specification.")
    rng = np.random.default_rng(seed)
    # Draw with replacement across sets, without replacement inside each layer.
    # Selected groups can occur by chance: do not bias the null by excluding them.
    return [[[layer, int(group)] for layer, count in sorted(counts.items())
             for group in sorted(rng.choice(n_groups, count, replace=False))] for _ in range(draws)]


def statistics(records: list[dict], draws: int, seed: int) -> dict:
    conditions = sorted({r["condition"] for r in records})
    indices = sorted({r["pair_index"] for r in records})
    by_key = {(r["condition"], r["pair_index"]): r for r in records}
    if len(by_key) != len(records) or len(records) != len(conditions) * len(indices):
        raise ValueError("Summary requires a complete, unique condition-by-pair grid.")
    grid = [[by_key[(condition, index)] for index in indices] for condition in conditions]
    baseline = np.array([[r["baseline"]["target_rank"] for r in row] for row in grid])
    selected = np.array([[r["selected"]["target_rank"] for r in row] for row in grid])
    random = np.array([[[x["target_rank"] for x in r["random"]] for r in row] for row in grid])
    collision = np.array([[r["answer_token_collision"] for r in row] for row in grid])

    def estimates(ids):
        # ids shape [bootstrap draws, pairs]; same pair indices in every condition.
        base_t1 = (baseline[:, ids] == 1).mean(axis=(0, 2)) * 100
        selected_t1 = (selected[:, ids] == 1).mean(axis=(0, 2)) * 100
        random_t1 = (random[:, ids, :] == 1).mean(axis=(0, 2, 3)) * 100
        return {"baseline_top1_percent": base_t1, "selected_top1_percent": selected_t1,
                "selected_minus_baseline_top1_pp": selected_t1 - base_t1,
                "selected_top10_percent": (selected[:, ids] <= 10).mean(axis=(0, 2)) * 100,
                "selected_mean_rank": selected[:, ids].mean(axis=(0, 2)),
                "selected_median_rank": np.median(selected[:, ids], axis=2).mean(axis=0),
                "random_top1_percent": random_t1,
                "selected_minus_random_top1_pp": selected_t1 - random_t1}

    point = estimates(np.arange(len(indices))[None, :])
    samples = {key: [] for key in point}
    rng = np.random.default_rng(seed)
    for offset in range(0, draws, 128):
        ids = rng.integers(len(indices), size=(min(128, draws - offset), len(indices)))
        for key, value in estimates(ids).items():
            samples[key].extend(value.tolist())
    return {"n_conditions": len(conditions), "n_pairs_per_condition": len(indices),
            "answer_token_collision_count": int(collision.sum()),
            "metrics": {key: {"estimate": float(value[0]),
                              "ci95": np.quantile(samples[key], [0.025, 0.975]).tolist()}
                        for key, value in point.items()}}


def summarize_records(records: list[dict], draws: int, seed: int) -> dict:
    groups = {condition: [r for r in records if r["condition"] == condition]
              for condition in sorted({r["condition"] for r in records})}
    groups.update({"all": records,
                   "monolingual": [r for r in records if r["donor_language"] == r["recipient_language"]],
                   "cross_lingual": [r for r in records if r["donor_language"] != r["recipient_language"]]})
    return {"method": "95% percentile bootstrap; resample pair_index jointly across conditions; equal condition weights",
            "conditioning": "Fixed model, countries, selected groups, and random-site bank; no held-out-country claim.",
            "outcome": "Vocabulary rank of the first token of the recipient-language, donor-country capital.",
            "ties": "Rank is one plus the number of strictly higher logits, matching the original evaluator.",
            "collisions": "Pairs with identical original/target first-token IDs are retained and counted explicitly.",
            "bootstrap_resamples": draws, "bootstrap_seed": seed,
            "groups": {name: statistics(rows, draws, stable_seed(seed, name)) for name, rows in groups.items() if rows}}
