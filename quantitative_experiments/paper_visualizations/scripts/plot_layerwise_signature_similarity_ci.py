#!/usr/bin/env python3
"""Plot layerwise aligned and control similarity with question-bootstrap CIs.

The bootstrap unit is a parallel-question ID. All language versions of a
question therefore receive the same bootstrap weight. Within each condition,
the statistic first averages over the ten unordered language pairs. Each
aggregate bootstrap replicate then averages the condition-level curves; the
confidence intervals are computed from the resulting aggregate distributions.
The aligned-minus-control difference and its paired interval are also exported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parent
QUANT_ROOT = PAPER_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(PAPER_DIR / ".matplotlib"))

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


DEFAULT_REPORTS = [
    QUANT_ROOT / "llama_outputs/belebele/multilingual_head_patch/test_baselines_layerwise/baseline_layerwise_report.json",
    QUANT_ROOT / "llama_outputs/mmluprox/multilingual_head_patch/test_baselines_layerwise/baseline_layerwise_report.json",
    QUANT_ROOT / "qwen_outputs/belebele/multilingual_head_patch/test_baselines_layerwise/baseline_layerwise_report.json",
    QUANT_ROOT / "qwen_outputs/mmluprox/multilingual_head_patch/test_baselines_layerwise/baseline_layerwise_report.json",
]

EPS = 1.0e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot layerwise aligned similarity and the mean cross-language control "
            "with question-resampled confidence intervals."
        )
    )
    parser.add_argument(
        "--report",
        action="append",
        default=[],
        help=(
            "Path to a baseline_layerwise_report.json file. Can be passed "
            "multiple times. Defaults to the four main paper conditions."
        ),
    )
    parser.add_argument(
        "--metric",
        default="pct_var_mean_incorrect",
        help="Importance metric used to construct the layerwise signatures.",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=10_000,
        help="Number of parallel-question bootstrap resamples.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
        help="Base seed for the question bootstrap.",
    )
    parser.add_argument(
        "--bootstrap-block-size",
        type=int,
        default=512,
        help="Number of bootstrap replicates evaluated per block.",
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=0.95,
        help="Confidence level for the percentile interval.",
    )
    parser.add_argument(
        "--relative-depth-points",
        type=int,
        default=256,
        help="Number of points used to interpolate curves to relative depth.",
    )
    parser.add_argument(
        "--outdir",
        default=str(PAPER_DIR / "figures"),
        help="Output directory for the figure, curve CSV, and metadata JSON.",
    )
    parser.add_argument(
        "--stem",
        default="layerwise_signature_similarity_gap_ci",
        help="Output filename stem.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png"],
        help="Figure formats to save, e.g. pdf png svg.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="DPI for raster outputs.")
    parser.add_argument("--width", type=float, default=10.0, help="Figure width in inches.")
    parser.add_argument("--height", type=float, default=4.5, help="Figure height in inches.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.bootstrap_resamples < 2:
        raise ValueError("--bootstrap-resamples must be at least 2.")
    if args.bootstrap_block_size < 1:
        raise ValueError("--bootstrap-block-size must be positive.")
    if not 0.0 < args.ci_level < 1.0:
        raise ValueError("--ci-level must be between 0 and 1.")
    if args.relative_depth_points < 2:
        raise ValueError("--relative-depth-points must be at least 2.")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_input_dir(report_path: Path, value: str) -> Path:
    input_dir = Path(value).expanduser()
    if input_dir.is_absolute():
        return input_dir

    candidates = [QUANT_ROOT / input_dir, report_path.parent / input_dir]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not resolve input_dir '{value}' from {report_path}. "
        f"Checked: {', '.join(str(path) for path in candidates)}"
    )


def load_balanced_layer_tensor(
    report_path: Path,
    metric: str,
) -> tuple[np.ndarray, list[str], list[str], Path]:
    report = load_json(report_path)
    input_dir = resolve_input_dir(report_path, str(report["input_dir"]))
    samples = load_json(input_dir / "samples.json")
    metadata = load_json(input_dir / "metadata.json")
    languages = [str(language) for language in report.get("languages", [])]
    if len(languages) < 2:
        raise ValueError(f"Expected at least two report languages in {report_path}.")

    index: dict[tuple[str, str], int] = {}
    all_parallel_ids: set[str] = set()
    for sample in samples:
        parallel_id = str(sample["parallel_id"])
        language = str(sample["language"])
        key = (parallel_id, language)
        if key in index:
            raise ValueError(f"Duplicate sample {key} in {input_dir / 'samples.json'}.")
        index[key] = int(sample["sample_index"])
        all_parallel_ids.add(parallel_id)

    parallel_ids = sorted(
        parallel_id
        for parallel_id in all_parallel_ids
        if all((parallel_id, language) in index for language in languages)
    )
    if len(parallel_ids) != len(all_parallel_ids):
        raise ValueError(
            f"Question-cluster resampling requires a balanced language-by-question design, "
            f"but {input_dir} contains {len(all_parallel_ids) - len(parallel_ids)} incomplete questions."
        )

    with np.load(input_dir / "importance_tensors.npz") as archive:
        flat_key = f"flat_{metric}"
        if metric in archive:
            raw = np.asarray(archive[metric], dtype=np.float32)
        elif flat_key in archive:
            raw = np.asarray(archive[flat_key], dtype=np.float32)
        else:
            available = ", ".join(sorted(archive.files))
            raise KeyError(f"Metric '{metric}' not found in {input_dir}. Available: {available}")

    n_layers = int(metadata["n_layers"])
    n_groups = int(metadata["n_patch_units"])
    if raw.ndim == 2:
        if raw.shape[1] != n_layers * n_groups:
            raise ValueError(
                f"Cannot reshape metric '{metric}' with shape {raw.shape} "
                f"to ({n_layers}, {n_groups}) in {input_dir}."
            )
        raw = raw.reshape(raw.shape[0], n_layers, n_groups)
    elif raw.ndim != 3 or raw.shape[1:] != (n_layers, n_groups):
        raise ValueError(
            f"Expected metric '{metric}' to have shape [samples, {n_layers}, {n_groups}], "
            f"got {raw.shape} in {input_dir}."
        )

    tensor = np.stack(
        [raw[[index[(parallel_id, language)] for parallel_id in parallel_ids]] for language in languages],
        axis=0,
    )
    return tensor, languages, parallel_ids, input_dir


def rankdata_last_axis(values: np.ndarray) -> np.ndarray:
    """Return tie-aware, 1-based average ranks along the last axis."""
    original_shape = values.shape
    flat = np.asarray(values).reshape(-1, original_shape[-1])
    width = flat.shape[1]
    order = np.argsort(flat, axis=1, kind="mergesort")
    ranks = np.empty_like(flat, dtype=np.float32)
    ordinal = np.broadcast_to(np.arange(1, width + 1, dtype=np.float32), flat.shape)
    np.put_along_axis(ranks, order, ordinal, axis=1)

    for row_index in range(flat.shape[0]):
        sorted_values = flat[row_index, order[row_index]]
        starts = np.concatenate(
            [np.array([0], dtype=np.int64), np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
        )
        ends = np.concatenate([starts[1:], np.array([width], dtype=np.int64)])
        for start, end in zip(starts, ends):
            if end - start > 1:
                average_rank = 0.5 * (float(start) + float(end) - 1.0) + 1.0
                ranks[row_index, order[row_index, start:end]] = average_rank
    return ranks.reshape(original_shape)


def normalized_ranks(tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ranks = rankdata_last_axis(tensor)
    ranks -= ranks.mean(axis=-1, keepdims=True)
    norms = np.linalg.norm(ranks, axis=-1, keepdims=True)
    valid = norms[..., 0] > EPS
    normalized = np.divide(ranks, norms, out=np.zeros_like(ranks), where=norms > EPS)
    return normalized, valid


def condition_question_contributions(
    tensor: np.ndarray,
    languages: list[str],
) -> dict[str, np.ndarray]:
    """Return per-question aligned, control, and gap curves for one condition.

    For a language pair, the cross-language control is the exact expectation
    over a uniformly permuted question-identity mapping. A question's control
    contribution is the average of its row and column means in the pairwise
    similarity matrix. Averaging these contributions recovers that exact
    expected control without treating question-pair cells as independent.
    """
    normalized, valid = normalized_ranks(tensor)
    n_languages, n_questions, n_layers, _ = normalized.shape
    if n_languages != len(languages):
        raise ValueError("Language labels do not match the tensor's language axis.")
    if not np.all(valid):
        invalid_count = int(np.size(valid) - np.count_nonzero(valid))
        raise ValueError(
            f"Found {invalid_count} degenerate layerwise signatures. The paper reports use "
            "degenerate_spearman='skip'; handling unequal per-layer denominators requires "
            "an explicit estimand and is intentionally not inferred here."
        )

    aligned_sum = np.zeros((n_questions, n_layers), dtype=np.float64)
    control_sum = np.zeros_like(aligned_sum)
    language_pairs = list(combinations(range(n_languages), 2))
    for first, second in language_pairs:
        first_ranks = normalized[first]
        second_ranks = normalized[second]
        aligned_sum += np.einsum("qlg,qlg->ql", first_ranks, second_ranks, optimize=True)

        mean_first = first_ranks.mean(axis=0)
        mean_second = second_ranks.mean(axis=0)
        row_mean = np.einsum("qlg,lg->ql", first_ranks, mean_second, optimize=True)
        column_mean = np.einsum("qlg,lg->ql", second_ranks, mean_first, optimize=True)
        control_sum += 0.5 * (row_mean + column_mean)

    pair_count = float(len(language_pairs))
    aligned = aligned_sum / pair_count
    control = control_sum / pair_count
    return {
        "aligned": aligned,
        "control": control,
        "gap": aligned - control,
    }


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def interpolation_matrix(n_layers: int, target_x: np.ndarray) -> np.ndarray:
    source_x = np.linspace(0.0, 1.0, n_layers)
    weights = np.zeros((n_layers, target_x.size), dtype=np.float64)
    for column, x_value in enumerate(target_x):
        upper = int(np.searchsorted(source_x, x_value, side="right"))
        if upper == 0:
            weights[0, column] = 1.0
        elif upper >= n_layers:
            weights[-1, column] = 1.0
        else:
            lower = upper - 1
            fraction = (x_value - source_x[lower]) / (source_x[upper] - source_x[lower])
            weights[lower, column] = 1.0 - fraction
            weights[upper, column] = fraction
    return weights


def bootstrap_condition(
    contributions: dict[str, np.ndarray],
    target_x: np.ndarray,
    n_resamples: int,
    seed: int,
    block_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    n_questions, n_layers = contributions["gap"].shape
    weights = interpolation_matrix(n_layers, target_x)
    points = {
        key: values.mean(axis=0) @ weights
        for key, values in contributions.items()
    }
    samples = {
        key: np.empty((n_resamples, target_x.size), dtype=np.float32)
        for key in contributions
    }

    rng = np.random.default_rng(seed)
    probabilities = np.full(n_questions, 1.0 / float(n_questions), dtype=np.float64)
    native_contributions = {
        key: np.asarray(contributions[key], dtype=np.float32)
        for key in ("aligned", "control")
    }
    for start in range(0, n_resamples, block_size):
        stop = min(start + block_size, n_resamples)
        counts = rng.multinomial(
            n_questions,
            probabilities,
            size=stop - start,
        ).astype(np.float32)
        counts /= float(n_questions)
        for key, values in native_contributions.items():
            native_samples = counts @ values
            samples[key][start:stop] = native_samples @ weights
        samples["gap"][start:stop] = (
            samples["aligned"][start:stop] - samples["control"][start:stop]
        )
    return points, samples


def percentile_interval(samples: np.ndarray, ci_level: float) -> tuple[np.ndarray, np.ndarray]:
    alpha = 1.0 - ci_level
    low, high = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0)
    return np.asarray(low), np.asarray(high)


def aggregate_conditions(
    condition_points: list[dict[str, np.ndarray]],
    condition_samples: list[dict[str, np.ndarray]],
    target_x: np.ndarray,
    ci_level: float,
) -> dict[str, np.ndarray]:
    curves: dict[str, np.ndarray] = {"relative_depth": target_x}
    for key in ("aligned", "control", "gap"):
        point = np.stack([condition[key] for condition in condition_points], axis=0).mean(axis=0)
        bootstrap = np.stack([condition[key] for condition in condition_samples], axis=0).mean(axis=0)
        low, high = percentile_interval(bootstrap, ci_level)
        curves[key] = point
        curves[f"{key}_ci_low"] = low
        curves[f"{key}_ci_high"] = high
    return curves


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 24,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def plot_curve(
    curves: dict[str, np.ndarray],
    ci_level: float,
    outdir: Path,
    stem: str,
    formats: list[str],
    dpi: int,
    size: tuple[float, float],
) -> None:
    configure_matplotlib()
    confidence = 100.0 * ci_level
    confidence_label = f"{confidence:g}% confidence intervals"

    fig, ax = plt.subplots(figsize=size)
    aligned_line, = ax.plot(
        curves["relative_depth"],
        curves["aligned"],
        color="#0072B2",
        linewidth=2.2,
        label="Aligned questions",
    )
    ax.fill_between(
        curves["relative_depth"],
        curves["aligned_ci_low"],
        curves["aligned_ci_high"],
        color="#0072B2",
        alpha=0.18,
        linewidth=0.0,
    )
    control_line, = ax.plot(
        curves["relative_depth"],
        curves["control"],
        color="#D55E00",
        linewidth=2.0,
        label="Shuffled cross-language control",
    )
    ax.fill_between(
        curves["relative_depth"],
        curves["control_ci_low"],
        curves["control_ci_high"],
        color="#D55E00",
        alpha=0.18,
        linewidth=0.0,
    )
    ax.axhline(0.0, color="#555555", linewidth=0.9, linestyle="--", alpha=0.8)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Relative layer depth")
    ax.set_ylabel("Average Spearman\ncorrelation")
    interval_patch = Patch(
        facecolor="#777777",
        edgecolor="none",
        alpha=0.18,
        label=confidence_label,
    )
    ax.legend(
        handles=[aligned_line, control_line, interval_patch],
        frameon=False,
        loc="best",
    )

    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(outdir / f"{stem}.{fmt}", dpi=dpi)
    plt.close(fig)


def write_curve_csv(curves: dict[str, np.ndarray], out_path: Path) -> None:
    fieldnames = [
        "relative_depth",
        "aligned_similarity",
        "aligned_ci_low",
        "aligned_ci_high",
        "mean_cross_language_control",
        "control_ci_low",
        "control_ci_high",
        "aligned_minus_control",
        "gap_ci_low",
        "gap_ci_high",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(curves["relative_depth"].size):
            writer.writerow(
                {
                    "relative_depth": curves["relative_depth"][index],
                    "aligned_similarity": curves["aligned"][index],
                    "aligned_ci_low": curves["aligned_ci_low"][index],
                    "aligned_ci_high": curves["aligned_ci_high"][index],
                    "mean_cross_language_control": curves["control"][index],
                    "control_ci_low": curves["control_ci_low"][index],
                    "control_ci_high": curves["control_ci_high"][index],
                    "aligned_minus_control": curves["gap"][index],
                    "gap_ci_low": curves["gap_ci_low"][index],
                    "gap_ci_high": curves["gap_ci_high"][index],
                }
            )


def write_metadata(
    out_path: Path,
    args: argparse.Namespace,
    report_paths: list[Path],
    conditions: list[dict[str, Any]],
) -> None:
    payload = {
        "estimand": "aligned similarity minus the mean cross-language control",
        "figure_display": (
            "aligned questions and the shuffled cross-language control are shown as separate lines; "
            "each shaded band is its question-bootstrap confidence interval"
        ),
        "similarity": "layerwise Spearman correlation of attention-group usage signatures",
        "cross_language_control": (
            "exact expectation over a uniformly permuted question-identity mapping "
            "within each language pair"
        ),
        "within_condition_aggregation": (
            "each question is averaged equally over all unordered language pairs"
        ),
        "across_condition_aggregation": (
            "equal-weight mean of condition estimates within each bootstrap replicate"
        ),
        "uncertainty": {
            "method": "nonparametric percentile bootstrap",
            "unit": "parallel-question ID",
            "dependence_preserved": "all five language versions are kept together",
            "question_level_control_contribution": (
                "average of the question's row and column means in each "
                "cross-language similarity matrix, then averaged over language pairs"
            ),
            "resamples": args.bootstrap_resamples,
            "confidence_level": args.ci_level,
            "seed": args.bootstrap_seed,
            "bounds_aggregation": (
                "interval percentiles are computed after aggregating bootstrap replicates; "
                "condition-level interval bounds are not averaged"
            ),
        },
        "metric": args.metric,
        "relative_depth_points": args.relative_depth_points,
        "reports": [str(path) for path in report_paths],
        "conditions": conditions,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    validate_args(args)
    report_paths = [Path(path).expanduser().resolve() for path in args.report] if args.report else DEFAULT_REPORTS
    target_x = np.linspace(0.0, 1.0, args.relative_depth_points)

    condition_points: list[dict[str, np.ndarray]] = []
    condition_samples: list[dict[str, np.ndarray]] = []
    condition_metadata: list[dict[str, Any]] = []
    for condition_index, report_path in enumerate(report_paths):
        tensor, languages, parallel_ids, input_dir = load_balanced_layer_tensor(report_path, args.metric)
        contributions = condition_question_contributions(tensor, languages)
        points, samples = bootstrap_condition(
            contributions=contributions,
            target_x=target_x,
            n_resamples=args.bootstrap_resamples,
            seed=stable_seed(args.bootstrap_seed, condition_index),
            block_size=args.bootstrap_block_size,
        )
        condition_points.append(points)
        condition_samples.append(samples)
        condition_metadata.append(
            {
                "report": str(report_path),
                "input_dir": str(input_dir),
                "n_questions": len(parallel_ids),
                "n_languages": len(languages),
                "n_language_pairs": len(list(combinations(languages, 2))),
                "n_layers": int(tensor.shape[2]),
                "n_attention_groups_per_layer": int(tensor.shape[3]),
            }
        )
        print(
            f"Computed {report_path}: {len(parallel_ids)} questions, "
            f"{len(languages)} languages, {tensor.shape[2]} layers"
        )

    curves = aggregate_conditions(condition_points, condition_samples, target_x, args.ci_level)
    outdir = Path(args.outdir).expanduser()
    plot_curve(
        curves=curves,
        ci_level=args.ci_level,
        outdir=outdir,
        stem=args.stem,
        formats=args.formats,
        dpi=args.dpi,
        size=(args.width, args.height),
    )
    write_curve_csv(curves, outdir / f"{args.stem}_curve.csv")
    write_metadata(
        outdir / f"{args.stem}_metadata.json",
        args,
        report_paths,
        condition_metadata,
    )

    for fmt in args.formats:
        print(f"Saved {outdir / f'{args.stem}.{fmt}'}")
    print(f"Saved {outdir / f'{args.stem}_curve.csv'}")
    print(f"Saved {outdir / f'{args.stem}_metadata.json'}")


if __name__ == "__main__":
    main()
