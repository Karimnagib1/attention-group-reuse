#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RESULTS_DIRS = [
    "Qwen_individual_language_rank_diff",
    "Qwen_cross_lingual_rank_diff",
    "Llama_individual_language_rank_diff",
    "Llama_cross_lingual_rank_diff",
]

LANG_ORDER = ["ar", "de", "en", "es", "ru"]
ALT_TASK_ORDER = ["university", "landmark"]
METRIC_COLUMNS = [
    ("baseline_top1_rate", "Base T1"),
    ("patched_top1_rate", "Patch T1"),
    ("delta_top1_rate", "Delta T1"),
    ("patched_top10_rate", "Patch T10"),
]

A4_PAGE_WIDTH_IN = 8.27
HEATMAP_WIDTH_IN = A4_PAGE_WIDTH_IN
DUMBBELL_WIDTH_IN = A4_PAGE_WIDTH_IN
HEATMAP_ROW_HEIGHT_IN = 0.52
DUMBBELL_ROW_HEIGHT_IN = 0.48
HEATMAP_MIN_HEIGHT_IN = 5.2
DUMBBELL_MIN_HEIGHT_IN = 5.0
HEATMAP_TITLE_FONTSIZE = 16
DUMBBELL_TITLE_FONTSIZE = 16
AXIS_LABEL_FONTSIZE = 13
XTICK_FONTSIZE = 12
YTICK_FONTSIZE = 11
ANNOTATION_FONTSIZE = 10
LEGEND_FONTSIZE = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create paper-ready summary plots from rank-diff JSON results."
    )
    parser.add_argument(
        "--results-dirs",
        nargs="*",
        default=DEFAULT_RESULTS_DIRS,
        help="Directories containing rank-diff JSON outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default="paper_summary_final",
        help="Directory where summary CSVs and figures will be written.",
    )
    return parser.parse_args()


def model_display_name(model_name: str) -> str:
    short = model_name.split("/")[-1]
    if "Qwen" in short:
        return "Qwen"
    if "Llama" in short:
        return "Llama"
    return short


def parse_lang_from_data_path(path_str: str) -> str:
    return Path(path_str).name.split("_")[0]


def task_display_name(alt_src: str) -> str:
    return alt_src


def pretty_target_name(value: str) -> str:
    if value == "country calling code":
        return "country code"
    return value


def task_pair_label(
    normal_src: str,
    normal_tgt: str,
    alt_src: str,
    alt_tgt: str,
) -> str:
    return (
        f"{normal_src}->{pretty_target_name(normal_tgt)} | "
        f"{alt_src}->{pretty_target_name(alt_tgt)}"
    )


def format_patch_spec(
    layers_to_kv_groups: Union[Dict[str, List[int]], Dict[int, List[int]]]
) -> str:
    parts: List[str] = []
    for layer_key in sorted(layers_to_kv_groups, key=lambda value: int(value)):
        groups = ",".join(str(group) for group in layers_to_kv_groups[layer_key])
        parts.append(f"L{int(layer_key)}-G{groups}")
    return "__".join(parts)


def family_sort_key(family: str) -> Tuple[int, str]:
    family_order = {
        "Qwen | monolingual": 0,
        "Qwen | cross-lingual": 1,
        "Llama | monolingual": 2,
        "Llama | cross-lingual": 3,
    }
    return (family_order.get(family, 99), family)


def language_sort_key(language_pair: str) -> Tuple[int, int]:
    src_lang, tgt_lang = language_pair.split("->")
    return (
        LANG_ORDER.index(src_lang) if src_lang in LANG_ORDER else 99,
        LANG_ORDER.index(tgt_lang) if tgt_lang in LANG_ORDER else 99,
    )


def task_sort_key(task: str) -> int:
    return ALT_TASK_ORDER.index(task) if task in ALT_TASK_ORDER else 99


def short_experiment_label(row: pd.Series) -> str:
    return f"{row['language_pair']} | {row['task_label']}"


def load_rank_diff_results(results_dirs: Iterable[str], root_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    experiment_rows: List[Dict[str, object]] = []
    pair_rows: List[Dict[str, object]] = []

    for results_dir in results_dirs:
        dir_path = root_dir / results_dir
        if not dir_path.exists():
            print(f"Skipping missing results dir: {dir_path}")
            continue

        for json_path in sorted(dir_path.glob("*.json")):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            cfg = data["config"]
            per_pair = data["per_pair"]

            model = model_display_name(cfg["model_name"])
            src_lang = parse_lang_from_data_path(cfg["data2_path"])
            tgt_lang = parse_lang_from_data_path(cfg["data1_path"])
            language_pair = f"{src_lang}->{tgt_lang}"
            task = task_display_name(cfg["alt_src"])
            setting = "monolingual" if src_lang == tgt_lang else "cross-lingual"
            family = f"{model} | {setting}"
            patch_spec = format_patch_spec(cfg["layers_to_kv_groups"])
            experiment_id = f"{family}|{language_pair}|{task}|{patch_spec}"

            baseline_alt_ranks = np.array(
                [item["baseline"]["alt_rank"] for item in per_pair], dtype=np.int64
            )
            patched_alt_ranks = np.array(
                [item["patched"]["alt_rank"] for item in per_pair], dtype=np.int64
            )
            baseline_logit_diffs = np.array(
                [item["baseline"]["logit_diff"] for item in per_pair], dtype=np.float64
            )
            patched_logit_diffs = np.array(
                [item["patched"]["logit_diff"] for item in per_pair], dtype=np.float64
            )

            baseline_top1_rate = float(np.mean(baseline_alt_ranks == 1))
            patched_top1_rate = float(np.mean(patched_alt_ranks == 1))
            patched_top10_rate = float(np.mean(patched_alt_ranks <= 10))

            experiment_rows.append(
                {
                    "experiment_id": experiment_id,
                    "json_path": str(json_path),
                    "folder": dir_path.name,
                    "model": model,
                    "setting": setting,
                    "family": family,
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang,
                    "language_pair": language_pair,
                    "task": task,
                    "normal_src": cfg["normal_src"],
                    "normal_tgt": cfg["normal_tgt"],
                    "alt_src": cfg["alt_src"],
                    "alt_tgt": cfg["alt_tgt"],
                    "task_label": task_pair_label(
                        cfg["normal_src"],
                        cfg["normal_tgt"],
                        cfg["alt_src"],
                        cfg["alt_tgt"],
                    ),
                    "patch_spec": patch_spec,
                    "num_pairs": int(cfg["num_pairs"]),
                    "num_examples": int(cfg["num_examples"]),
                    "baseline_top1_rate": baseline_top1_rate,
                    "patched_top1_rate": patched_top1_rate,
                    "delta_top1_rate": patched_top1_rate - baseline_top1_rate,
                    "patched_top10_rate": patched_top10_rate,
                    "baseline_mean_rank": float(np.mean(baseline_alt_ranks)),
                    "patched_mean_rank": float(np.mean(patched_alt_ranks)),
                    "baseline_median_rank": float(np.median(baseline_alt_ranks)),
                    "patched_median_rank": float(np.median(patched_alt_ranks)),
                    "baseline_mean_logit_diff": float(np.mean(baseline_logit_diffs)),
                    "patched_mean_logit_diff": float(np.mean(patched_logit_diffs)),
                }
            )

            for item in per_pair:
                pair_rows.append(
                    {
                        "experiment_id": experiment_id,
                        "model": model,
                        "setting": setting,
                        "family": family,
                        "language_pair": language_pair,
                        "task": task,
                        "patch_spec": patch_spec,
                        "pair_index": int(item["pair_index"]),
                        "baseline_alt_rank": int(item["baseline"]["alt_rank"]),
                        "patched_alt_rank": int(item["patched"]["alt_rank"]),
                        "baseline_normal_rank": int(item["baseline"]["normal_rank"]),
                        "patched_normal_rank": int(item["patched"]["normal_rank"]),
                        "baseline_logit_diff": float(item["baseline"]["logit_diff"]),
                        "patched_logit_diff": float(item["patched"]["logit_diff"]),
                    }
                )

    experiments_df = pd.DataFrame(experiment_rows)
    pair_df = pd.DataFrame(pair_rows)

    if experiments_df.empty:
        raise RuntimeError("No rank-diff JSON files were found in the provided directories.")

    experiments_df = experiments_df.sort_values(
        by=["family", "language_pair", "task"],
        key=lambda series: series.map(
            lambda value: (
                family_sort_key(value)
                if series.name == "family"
                else language_sort_key(value)
                if series.name == "language_pair"
                else task_sort_key(value)
            )
        ),
    ).reset_index(drop=True)

    return experiments_df, pair_df


def write_summary_tables(experiments_df: pd.DataFrame, pair_df: pd.DataFrame, output_dir: Path) -> None:
    experiments_out = experiments_df.copy()
    rate_cols = [
        "baseline_top1_rate",
        "patched_top1_rate",
        "delta_top1_rate",
        "patched_top10_rate",
    ]
    for col in rate_cols:
        experiments_out[f"{col}_pct"] = 100.0 * experiments_out[col]

    experiments_out.to_csv(output_dir / "experiment_summary.csv", index=False)
    pair_df.to_csv(output_dir / "pair_summary.csv", index=False)


def model_output_stem(model_name: str) -> str:
    return model_name.lower()


def family_output_stem(family: str) -> str:
    model_name, setting = family.split(" | ", maxsplit=1)
    return f"{model_output_stem(model_name)}_{setting.replace('-', '_')}"


def plot_height(num_rows: int, row_height_in: float, min_height_in: float) -> float:
    return max(min_height_in, row_height_in * num_rows + 1.8)


def plot_summary_heatmap_for_model(experiments_df: pd.DataFrame, model_name: str, output_dir: Path) -> None:
    model_df = experiments_df[experiments_df["model"] == model_name].copy()
    families = list(dict.fromkeys(model_df["family"]))

    cmap_abs = plt.cm.Blues
    cmap_delta = plt.cm.RdBu_r
    metric_labels = [label for _, label in METRIC_COLUMNS]

    for family in families:
        family_df = model_df[model_df["family"] == family].copy()
        family_df["row_label"] = family_df.apply(short_experiment_label, axis=1)
        fig, ax = plt.subplots(
            figsize=(
                HEATMAP_WIDTH_IN,
                plot_height(len(family_df), HEATMAP_ROW_HEIGHT_IN, HEATMAP_MIN_HEIGHT_IN),
            ),
            constrained_layout=True,
        )
        matrix = np.column_stack(
            [
                100.0 * family_df["baseline_top1_rate"].to_numpy(),
                100.0 * family_df["patched_top1_rate"].to_numpy(),
                100.0 * family_df["delta_top1_rate"].to_numpy(),
                100.0 * family_df["patched_top10_rate"].to_numpy(),
            ]
        )

        rgba = np.zeros((matrix.shape[0], matrix.shape[1], 4))
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                if col_idx == 2:
                    rgba[row_idx, col_idx] = cmap_delta((np.clip(value, -100, 100) + 100) / 200)
                else:
                    rgba[row_idx, col_idx] = cmap_abs(np.clip(value / 100.0, 0, 1))

        ax.imshow(rgba, aspect="auto")
        ax.set_title(
            f"{family} Rank-Diff Summary",
            fontsize=HEATMAP_TITLE_FONTSIZE,
            pad=14,
        )
        ax.set_xticks(np.arange(len(metric_labels)))
        ax.set_xticklabels(metric_labels, rotation=0, fontsize=XTICK_FONTSIZE)
        ax.set_yticks(np.arange(len(family_df)))
        ax.set_yticklabels(family_df["row_label"], fontsize=YTICK_FONTSIZE)
        ax.tick_params(axis="x", pad=8)

        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                ax.text(
                    col_idx,
                    row_idx,
                    f"{matrix[row_idx, col_idx]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=ANNOTATION_FONTSIZE,
                    color="black",
                )

        ax.set_xticks(np.arange(-0.5, len(metric_labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(family_df), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)
        fig.savefig(
            output_dir / f"{family_output_stem(family)}_summary_heatmap.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_dumbbell_top1_for_model(experiments_df: pd.DataFrame, model_name: str, output_dir: Path) -> None:
    model_df = experiments_df[experiments_df["model"] == model_name].copy()
    families = list(dict.fromkeys(model_df["family"]))

    color_map = {"Qwen": "#0f766e", "Llama": "#9a3412"}

    for family in families:
        family_df = model_df[model_df["family"] == family].copy()
        family_df = family_df.sort_values(
            by=["language_pair", "task"],
            key=lambda series: series.map(
                language_sort_key if series.name == "language_pair" else task_sort_key
            ),
        ).reset_index(drop=True)
        family_df["label"] = family_df.apply(short_experiment_label, axis=1)
        fig, ax = plt.subplots(
            figsize=(
                DUMBBELL_WIDTH_IN,
                plot_height(len(family_df), DUMBBELL_ROW_HEIGHT_IN, DUMBBELL_MIN_HEIGHT_IN),
            ),
            constrained_layout=True,
        )

        y_positions = np.arange(len(family_df))
        baseline = 100.0 * family_df["baseline_top1_rate"].to_numpy()
        patched = 100.0 * family_df["patched_top1_rate"].to_numpy()
        family_model_name = family.split(" | ")[0]
        patched_color = color_map.get(family_model_name, "#1d4ed8")

        for idx, y_val in enumerate(y_positions):
            ax.plot([baseline[idx], patched[idx]], [y_val, y_val], color="#94a3b8", linewidth=2.2)

        ax.scatter(baseline, y_positions, color="#64748b", s=48, label="Baseline", zorder=3)
        ax.scatter(patched, y_positions, color=patched_color, s=48, label="Patched", zorder=3)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(family_df["label"], fontsize=YTICK_FONTSIZE)
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        ax.set_title(
            f"{family} Baseline vs Patched Top-1 Rate",
            fontsize=DUMBBELL_TITLE_FONTSIZE,
            pad=14,
        )
        ax.tick_params(axis="x", labelsize=XTICK_FONTSIZE)
        ax.set_xlim(left=min(0, baseline.min() - 2), right=max(100, patched.max() + 2))
        ax.set_xlabel("Alt answer ranked #1 (%)", fontsize=AXIS_LABEL_FONTSIZE)
        ax.legend(loc="upper right", frameon=False, fontsize=LEGEND_FONTSIZE)
        fig.savefig(
            output_dir / f"{family_output_stem(family)}_top1_dumbbell.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def ecdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def select_representative_experiments(experiments_df: pd.DataFrame) -> pd.DataFrame:
    selected_indices: List[int] = []

    sorted_delta = experiments_df.sort_values("delta_top1_rate", ascending=False)
    if not sorted_delta.empty:
        selected_indices.append(int(sorted_delta.index[0]))

    median_delta = experiments_df["delta_top1_rate"].median()
    median_idx = int((experiments_df["delta_top1_rate"] - median_delta).abs().idxmin())
    selected_indices.append(median_idx)

    weakest_idx = int(experiments_df["delta_top1_rate"].idxmin())
    selected_indices.append(weakest_idx)

    arabic_df = experiments_df[experiments_df["language_pair"].str.contains("ar")]
    if not arabic_df.empty:
        best_arabic_idx = int(arabic_df.sort_values("delta_top1_rate", ascending=False).index[0])
        selected_indices.append(best_arabic_idx)

    deduped: List[int] = []
    for idx in selected_indices:
        if idx not in deduped:
            deduped.append(idx)

    if len(deduped) < 4:
        for idx in sorted_delta.index.tolist():
            if int(idx) not in deduped:
                deduped.append(int(idx))
            if len(deduped) == 4:
                break

    return experiments_df.loc[deduped[:4]].copy()


def plot_selected_cdfs_for_model(
    experiments_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    model_name: str,
    output_dir: Path,
) -> None:
    selected = select_representative_experiments(
        experiments_df[experiments_df["model"] == model_name].copy()
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.flatten()

    color_map = {"Qwen": "#0f766e", "Llama": "#9a3412"}

    for ax, (_, row) in zip(axes, selected.iterrows()):
        experiment_pairs = pair_df[pair_df["experiment_id"] == row["experiment_id"]]
        baseline_x, baseline_y = ecdf(experiment_pairs["baseline_alt_rank"].to_numpy())
        patched_x, patched_y = ecdf(experiment_pairs["patched_alt_rank"].to_numpy())
        patched_color = color_map.get(row["model"], "#1d4ed8")

        ax.step(baseline_x, baseline_y, where="post", color="#64748b", linewidth=2, label="Baseline")
        ax.step(patched_x, patched_y, where="post", color=patched_color, linewidth=2, label="Patched")
        ax.set_xscale("log")
        ax.set_xlabel("Alt-answer rank (log scale)")
        ax.set_ylabel("ECDF")
        ax.set_title(
            f"{row['family']}\n{row['language_pair']} | {row['task_label']} | "
            f"Delta T1 = {100.0 * row['delta_top1_rate']:.1f}",
            fontsize=10,
        )
        ax.grid(True, linestyle="--", alpha=0.35)

    for ax in axes[len(selected):]:
        ax.axis("off")

    axes[0].legend(loc="lower right", frameon=False)
    fig.suptitle(f"{model_name} Representative Alt-Rank Distributions", fontsize=14)
    fig.savefig(
        output_dir / f"{model_output_stem(model_name)}_selected_cdfs.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def rank_bin_labels() -> List[str]:
    return ["1", "2-5", "6-10", "11-20", "21-50", "51-100", "101-500", "501+"]


def bin_rank_series(values: pd.Series) -> pd.Series:
    bins = [0, 1, 5, 10, 20, 50, 100, 500, np.inf]
    labels = rank_bin_labels()
    return pd.cut(values, bins=bins, labels=labels, include_lowest=True, right=True)


def plot_llama_demonstrative_histogram(experiments_df: pd.DataFrame, pair_df: pd.DataFrame, output_dir: Path) -> None:
    llama_df = experiments_df[experiments_df["model"] == "Llama"].copy()
    if llama_df.empty:
        return

    llama_df["mean_rank_drop"] = llama_df["baseline_mean_rank"] - llama_df["patched_mean_rank"]
    row = llama_df.sort_values("mean_rank_drop", ascending=False).iloc[0]
    experiment_pairs = pair_df[pair_df["experiment_id"] == row["experiment_id"]].copy()

    labels = rank_bin_labels()
    baseline_binned = (
        bin_rank_series(experiment_pairs["baseline_alt_rank"])
        .value_counts(normalize=True)
        .reindex(labels, fill_value=0.0)
    )
    patched_binned = (
        bin_rank_series(experiment_pairs["patched_alt_rank"])
        .value_counts(normalize=True)
        .reindex(labels, fill_value=0.0)
    )

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.bar(x - width / 2, 100.0 * baseline_binned.to_numpy(), width=width, color="#64748b", label="Baseline")
    ax.bar(x + width / 2, 100.0 * patched_binned.to_numpy(), width=width, color="#9a3412", label="Patched")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Examples (%)")
    ax.set_xlabel("Alt-answer rank bin")
    ax.set_title(
        "Demonstrative Llama Rank Shift\n"
        f"{row['family']} | {row['language_pair']} | {row['task_label']}\n"
        f"Mean rank: {row['baseline_mean_rank']:.1f} -> {row['patched_mean_rank']:.1f}, "
        f"Median rank: {row['baseline_median_rank']:.1f} -> {row['patched_median_rank']:.1f}",
        fontsize=11,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False)

    for idx, value in enumerate(100.0 * baseline_binned.to_numpy()):
        if value > 0:
            ax.text(idx - width / 2, value + 1.0, f"{value:.0f}", ha="center", va="bottom", fontsize=8)
    for idx, value in enumerate(100.0 * patched_binned.to_numpy()):
        if value > 0:
            ax.text(idx + width / 2, value + 1.0, f"{value:.0f}", ha="center", va="bottom", fontsize=8)

    fig.savefig(output_dir / "llama_demonstrative_histogram.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_readme(experiments_df: pd.DataFrame, output_dir: Path) -> None:
    families = experiments_df.groupby("family").size().to_dict()
    lines = [
        "# Paper Summary Outputs",
        "",
        f"Experiments summarized: {len(experiments_df)}",
        "",
        "## Families",
        "",
    ]
    for family, count in families.items():
        lines.append(f"- {family}: {count}")

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `experiment_summary.csv`: one row per experiment with paper-level metrics",
            "- `pair_summary.csv`: one row per prompt pair for distribution plots",
            "- `qwen_selected_cdfs.png`: representative Qwen pre/post rank distributions",
            "- `llama_selected_cdfs.png`: representative Llama pre/post rank distributions",
            "- `llama_demonstrative_histogram.png`: binned before/after histogram for a Llama case where top-1 misses a large rank shift",
        ]
    )

    for family in families:
        family_stem = family_output_stem(family)
        lines.append(f"- `{family_stem}_summary_heatmap.png`: {family} summary heatmap")
        lines.append(
            f"- `{family_stem}_top1_dumbbell.png`: {family} baseline vs patched top-1 comparison"
        )

    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root_dir = Path(__file__).resolve().parent
    output_dir = root_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments_df, pair_df = load_rank_diff_results(args.results_dirs, root_dir)
    write_summary_tables(experiments_df, pair_df, output_dir)
    for model_name in ["Qwen", "Llama"]:
        plot_summary_heatmap_for_model(experiments_df, model_name, output_dir)
        plot_dumbbell_top1_for_model(experiments_df, model_name, output_dir)
        plot_selected_cdfs_for_model(experiments_df, pair_df, model_name, output_dir)
    plot_llama_demonstrative_histogram(experiments_df, pair_df, output_dir)
    write_readme(experiments_df, output_dir)

    print(f"Wrote paper summary outputs to {output_dir}")


if __name__ == "__main__":
    main()
