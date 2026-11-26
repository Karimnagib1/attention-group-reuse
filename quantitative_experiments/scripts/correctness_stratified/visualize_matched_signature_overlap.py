#!/usr/bin/env python3
"""Visualize top-ranked usage-signature overlap for a matched language pair."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle
from scipy.stats import spearmanr


SCRIPT_DIR = Path(__file__).resolve().parent
QUANT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_SWEEP_DIR = (
    QUANT_ROOT
    / "qwen_outputs/mmluprox/multilingual_head_patch/correctness_stratified"
)
DEFAULT_OUTPUT_DIR = (
    QUANT_ROOT
    / "results/multilingual_head_patch/correctness_stratified/analysis"
    / "usage_signature_overlap/qwen_mmluprox"
)
ALL_LANGUAGES = ("en", "ru", "es", "de", "ar")
LANGUAGE_NAMES = {"en": "English", "ru": "Russian", "es": "Spanish", "de": "German", "ar": "Arabic"}
OPTION_LABELS = tuple("ABCDEFGHIJ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot two matched usage signatures and mark overlapping top-ranked attention groups."
        )
    )
    parser.add_argument("--sweep-dir", default=str(DEFAULT_SWEEP_DIR))
    parser.add_argument("--languages", nargs=2, default=("en", "es"))
    parser.add_argument(
        "--parallel-id",
        help=(
            "Exact all-correct parallel-question ID. By default, select a representative "
            "question near the median Spearman correlation with the median top-k overlap."
        ),
    )
    parser.add_argument("--metric", default="pct_var_mean_incorrect")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--clip-percentile", type=float, default=99.5)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"), default=("png", "pdf"))
    parser.add_argument("--dpi", type=int, default=250)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_metric_tensor(sweep_dir: Path, metric: str, metadata: Mapping[str, Any]) -> np.ndarray:
    with np.load(sweep_dir / "importance_tensors.npz") as archive:
        if metric in archive:
            tensor = np.asarray(archive[metric], dtype=np.float64)
        elif f"flat_{metric}" in archive:
            flat = np.asarray(archive[f"flat_{metric}"], dtype=np.float64)
            tensor = flat.reshape(
                flat.shape[0],
                int(metadata["n_layers"]),
                int(metadata["n_patch_units"]),
            )
        else:
            raise KeyError(f"Metric {metric!r} is not present in the sweep archive.")
    if tensor.ndim != 3 or not np.all(np.isfinite(tensor)):
        raise ValueError(f"Expected a finite samples-by-layers-by-groups tensor, got {tensor.shape}.")
    return tensor


def group_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Mapping[str, Any]]]:
    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for sample in samples:
        grouped[str(sample["parallel_id"])][str(sample["language"])] = sample
    return dict(grouped)


def is_all_correct(language_samples: Mapping[str, Mapping[str, Any]]) -> bool:
    return all(
        language in language_samples
        and int(language_samples[language]["generated_idx"])
        == int(language_samples[language]["correct_idx"])
        for language in ALL_LANGUAGES
    )


def top_indices(matrix: np.ndarray, top_k: int) -> np.ndarray:
    if not 1 <= top_k <= matrix.size:
        raise ValueError(f"--top-k must be between 1 and {matrix.size}.")
    return np.argsort(matrix.ravel(), kind="stable")[-top_k:][::-1]


def pair_statistics(first: np.ndarray, second: np.ndarray, top_k: int) -> dict[str, Any]:
    first_top = top_indices(first, top_k)
    second_top = top_indices(second, top_k)
    shared = set(int(index) for index in first_top) & set(int(index) for index in second_top)
    union = set(int(index) for index in first_top) | set(int(index) for index in second_top)
    return {
        "spearman_correlation": float(spearmanr(first.ravel(), second.ravel()).statistic),
        "top_k": top_k,
        "overlap_count": len(shared),
        "jaccard_at_k": len(shared) / len(union),
        "first_top": first_top,
        "second_top": second_top,
        "shared": shared,
    }


def select_question(
    grouped: Mapping[str, Mapping[str, Mapping[str, Any]]],
    tensor: np.ndarray,
    languages: Sequence[str],
    top_k: int,
    explicit_parallel_id: str | None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    candidates = []
    for parallel_id, language_samples in grouped.items():
        if not is_all_correct(language_samples):
            continue
        first = tensor[int(language_samples[languages[0]]["sample_index"])]
        second = tensor[int(language_samples[languages[1]]["sample_index"])]
        stats = pair_statistics(first, second, top_k)
        candidates.append((parallel_id, stats))
    if not candidates:
        raise ValueError("No questions answered correctly in all five languages were found.")

    correlations = np.asarray([item[1]["spearman_correlation"] for item in candidates])
    overlaps = np.asarray([item[1]["overlap_count"] for item in candidates])
    median_correlation = float(np.median(correlations))
    median_overlap = float(np.median(overlaps))
    selection_summary = {
        "n_all_five_correct_candidates": len(candidates),
        "median_spearman_correlation": median_correlation,
        "median_top_k_overlap_count": median_overlap,
    }

    if explicit_parallel_id is not None:
        matches = [item for item in candidates if item[0] == str(explicit_parallel_id)]
        if len(matches) != 1:
            raise ValueError(
                f"Parallel ID {explicit_parallel_id!r} is not an all-five-correct candidate."
            )
        parallel_id, stats = matches[0]
        selection_summary["reason"] = "explicit all-five-correct parallel-question ID"
        return parallel_id, stats, selection_summary

    median_overlap_candidates = [
        item for item in candidates if item[1]["overlap_count"] == int(round(median_overlap))
    ]
    parallel_id, stats = min(
        median_overlap_candidates,
        key=lambda item: (
            abs(item[1]["spearman_correlation"] - median_correlation),
            item[0],
        ),
    )
    selection_summary["reason"] = (
        "all-five-correct question with the median top-k overlap count and "
        "Spearman correlation closest to the all-correct median"
    )
    return parallel_id, stats, selection_summary


def cached_dataset_paths(language: str) -> list[Path]:
    root = (
        Path.home()
        / f".cache/huggingface/datasets/li-lab___mmlu-pro_x/{language}"
    )
    return sorted(root.glob("**/mmlu-pro_x-test.arrow"), reverse=True)


def load_question(language: str, parallel_id: str) -> tuple[Mapping[str, Any], Path]:
    for path in cached_dataset_paths(language):
        dataset = Dataset.from_file(str(path))
        for row in dataset:
            if str(row.get("question_id")) == parallel_id:
                return row, path
    raise KeyError(f"Question {parallel_id} was not found in the cached {language} dataset.")


def question_payload(row: Mapping[str, Any], sample: Mapping[str, Any]) -> dict[str, Any]:
    options = []
    for index, label in enumerate(OPTION_LABELS):
        text = row.get(f"option_{index}")
        if text is None or str(text).strip() == "":
            continue
        options.append(
            {
                "label": label,
                "text": str(text),
                "is_correct": index == int(sample["correct_idx"]),
                "is_model_choice": index == int(sample["generated_idx"]),
            }
        )
    return {
        "language": str(sample["language"]),
        "sample": dict(sample),
        "question": str(row["question"]),
        "options": options,
        "correct_answer": str(sample["correct_label"]),
        "model_choice": str(sample["generated_label"]),
        "category": row.get("category"),
        "source": row.get("src"),
    }


def ranked_locations(
    matrix: np.ndarray,
    indices: np.ndarray,
    shared: set[int],
) -> list[dict[str, Any]]:
    n_groups = matrix.shape[1]
    return [
        {
            "rank": rank,
            "layer": int(index) // n_groups,
            "attention_group": int(index) % n_groups,
            "value": float(matrix.ravel()[index]),
            "shared_top_k": int(index) in shared,
        }
        for rank, index in enumerate(indices, start=1)
    ]


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.labelsize": 18,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def annotate_top_groups(
    axis: plt.Axes,
    matrix: np.ndarray,
    ranked: Sequence[Mapping[str, Any]],
    color_limit: float,
) -> None:
    for location in ranked:
        layer = int(location["layer"])
        group = int(location["attention_group"])
        shared = bool(location["shared_top_k"])
        edge_color = "#FFD92F" if shared else "#202020"
        axis.add_patch(
            Rectangle(
                (group - 0.48, layer - 0.48),
                0.96,
                0.96,
                fill=False,
                edgecolor=edge_color,
                linewidth=2.4 if shared else 1.0,
                linestyle="-" if shared else "--",
            )
        )
        text_color = "white" if abs(float(matrix[layer, group])) > 0.55 * color_limit else "black"
        axis.text(
            group,
            layer,
            str(location["rank"]),
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            color=text_color,
        )


def plot_heatmaps(
    matrices: Sequence[np.ndarray],
    question_payloads: Sequence[Mapping[str, Any]],
    ranked: Sequence[Sequence[Mapping[str, Any]]],
    stats: Mapping[str, Any],
    *,
    clip_percentile: float,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
    stem: str,
) -> list[Path]:
    absolute = np.concatenate([np.abs(matrix).ravel() for matrix in matrices])
    color_limit = float(np.percentile(absolute, clip_percentile))
    if not math.isfinite(color_limit) or color_limit <= 0.0:
        color_limit = float(np.max(absolute)) if absolute.size else 1.0
    norm = TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit)

    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 8.6), sharey=True)
    image = None
    for axis, matrix, payload, locations in zip(axes, matrices, question_payloads, ranked):
        image = axis.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="RdBu_r",
            norm=norm,
        )
        language = str(payload["language"])
        axis.set_title(
            f"{LANGUAGE_NAMES.get(language, language)}\n"
            f'“{payload["question"]}”\n'
            f"gold {payload['correct_answer']} · model {payload['model_choice']}"
        )
        axis.set_xlabel("Attention group within layer", labelpad=8)
        axis.set_xticks(np.arange(matrix.shape[1]))
        axis.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
        axis.grid(which="minor", axis="x", color="white", linewidth=0.65, alpha=0.6)
        axis.tick_params(which="minor", bottom=False)
        annotate_top_groups(axis, matrix, locations, color_limit)
    axes[0].set_ylabel("Transformer layer")
    layer_step = max(1, matrices[0].shape[0] // 8)
    layer_ticks = list(range(0, matrices[0].shape[0], layer_step))
    if layer_ticks[-1] != matrices[0].shape[0] - 1:
        layer_ticks.append(matrices[0].shape[0] - 1)
    axes[0].set_yticks(layer_ticks)

    unique_patch = Patch(
        facecolor="none",
        edgecolor="#202020",
        linewidth=1.0,
        linestyle="--",
        label="Top-10 location; number gives within-language rank",
    )
    shared_patch = Patch(
        facecolor="none",
        edgecolor="#FFD92F",
        linewidth=2.4,
        label="Shared top-10 location",
    )
    fig.legend(
        handles=[unique_patch, shared_patch],
        loc="lower center",
        bbox_to_anchor=(0.45, 0.045),
        ncol=2,
        frameon=False,
    )

    assert image is not None
    fig.subplots_adjust(left=0.08, right=0.86, top=0.87, bottom=0.25, wspace=0.14)
    colorbar_axis = fig.add_axes((0.89, 0.25, 0.022, 0.62))
    colorbar = fig.colorbar(image, cax=colorbar_axis, extend="both")
    colorbar.set_label("Usage-signature value")
    fig.text(
        0.45,
        0.125,
        f"Spearman correlation = {stats['spearman_correlation']:.3f}; "
        f"shared top 10 = {stats['overlap_count']}/10; "
        f"Jaccard@10 = {stats['jaccard_at_k']:.3f}",
        ha="center",
        fontsize=12.5,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for extension in formats:
        path = output_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=dpi if extension == "png" else None, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def main() -> None:
    args = parse_args()
    if len(set(args.languages)) != 2:
        raise SystemExit("--languages must contain two distinct language codes.")
    if any(language not in ALL_LANGUAGES for language in args.languages):
        raise SystemExit(f"Languages must be drawn from {ALL_LANGUAGES}.")
    if not 0.0 < args.clip_percentile <= 100.0:
        raise SystemExit("--clip-percentile must lie in (0, 100].")

    sweep_dir = Path(args.sweep_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    metadata = read_json(sweep_dir / "metadata.json")
    samples = read_json(sweep_dir / "samples.json")
    tensor = load_metric_tensor(sweep_dir, args.metric, metadata)
    grouped = group_samples(samples)
    parallel_id, stats, selection = select_question(
        grouped,
        tensor,
        args.languages,
        int(args.top_k),
        args.parallel_id,
    )
    language_samples = grouped[parallel_id]
    selected_samples = [language_samples[language] for language in args.languages]
    matrices = [tensor[int(sample["sample_index"])] for sample in selected_samples]

    questions = []
    dataset_paths = {}
    for language, sample in zip(args.languages, selected_samples):
        row, dataset_path = load_question(language, parallel_id)
        questions.append(question_payload(row, sample))
        dataset_paths[language] = str(dataset_path)

    ranked = [
        ranked_locations(matrix, indices, stats["shared"])
        for matrix, indices in zip(matrices, (stats["first_top"], stats["second_top"]))
    ]
    n_groups = matrices[0].shape[1]
    shared_locations = []
    first_rank = {int(index): rank for rank, index in enumerate(stats["first_top"], start=1)}
    second_rank = {int(index): rank for rank, index in enumerate(stats["second_top"], start=1)}
    for index in sorted(stats["shared"]):
        shared_locations.append(
            {
                "layer": index // n_groups,
                "attention_group": index % n_groups,
                f"{args.languages[0]}_rank": first_rank[index],
                f"{args.languages[1]}_rank": second_rank[index],
                f"{args.languages[0]}_value": float(matrices[0].ravel()[index]),
                f"{args.languages[1]}_value": float(matrices[1].ravel()[index]),
            }
        )

    stem = (
        f"matched_usage_signature_overlap_{parallel_id}_"
        f"{args.languages[0]}_{args.languages[1]}_top{args.top_k}"
    )
    written = plot_heatmaps(
        matrices,
        questions,
        ranked,
        stats,
        clip_percentile=float(args.clip_percentile),
        output_dir=output_dir,
        formats=args.formats,
        dpi=int(args.dpi),
        stem=stem,
    )

    report_path = output_dir / f"{stem}.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "condition": "qwen_mmluprox",
                "model": "Qwen3-8B",
                "dataset": "MMLU-ProX",
                "parallel_id": parallel_id,
                "languages": list(args.languages),
                "answered_correctly_in_all_five_languages": True,
                "all_language_outcomes": {
                    language: {
                        "correct_answer": language_samples[language]["correct_label"],
                        "model_choice": language_samples[language]["generated_label"],
                        "correct": True,
                    }
                    for language in ALL_LANGUAGES
                },
                "metric": args.metric,
                "selection": selection,
                "statistics": {
                    "spearman_correlation": stats["spearman_correlation"],
                    "top_k": stats["top_k"],
                    "overlap_count": stats["overlap_count"],
                    "jaccard_at_k": stats["jaccard_at_k"],
                },
                "top_ranked_locations": {
                    args.languages[0]: ranked[0],
                    args.languages[1]: ranked[1],
                },
                "shared_top_ranked_locations": shared_locations,
                "questions": questions,
                "shared_color_scale": True,
                "clip_percentile": float(args.clip_percentile),
                "dataset_paths": dataset_paths,
                "sweep_dir": str(sweep_dir),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")

    for path in written:
        print(f"Saved {path}")
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
