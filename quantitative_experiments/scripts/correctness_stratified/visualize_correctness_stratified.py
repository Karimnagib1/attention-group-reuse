#!/usr/bin/env python3
"""Create publication-style plots for correctness-stratified analysis outputs."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANALYSIS_ROOT = (
    REPOSITORY_ROOT
    / "results"
    / "multilingual_head_patch"
    / "correctness_stratified"
    / "analysis"
)
REQUIRED_FILES = (
    "similarity_by_k.csv",
    "similarity_by_pair_status.csv",
    "ordinal_trends.csv",
    "matched_subset_results.csv",
)
METRICS = ("spearman", "jaccard_at_10")
METRIC_LABELS = {
    "spearman": "Spearman correlation",
    "jaccard_at_10": "Jaccard@10",
}
CONDITION_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
STATUS_STYLES = {
    "correct_correct": ("Both correct", "#0072B2", "o"),
    "correct_incorrect": ("One correct", "#D55E00", "s"),
    "incorrect_incorrect": ("Both incorrect", "#7A7A7A", "^"),
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize correctness-stratified analysis CSV files."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--condition",
        help=(
            "Condition directory name under <analysis-root>/by_condition, "
            "for example qwen_belebele."
        ),
    )
    source.add_argument(
        "--input-dir",
        help="Exact combined- or single-condition analysis output directory.",
    )
    parser.add_argument(
        "--analysis-root",
        default=str(DEFAULT_ANALYSIS_ROOT),
        help=f"Analysis root used with --condition (default: {DEFAULT_ANALYSIS_ROOT}).",
    )
    parser.add_argument(
        "--output-dir",
        help="Figure directory (default: <input-dir>/viz).",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png", "pdf"),
        help="One or more output formats (default: png pdf).",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Raster output DPI (default: 300).")
    return parser.parse_args(argv)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing analysis result: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _float(row: Mapping[str, Any], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _int(row: Mapping[str, Any], key: str) -> int:
    return int(float(row[key]))


def _finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def _condition_ids(rows: Iterable[Mapping[str, Any]]) -> List[str]:
    return list(dict.fromkeys(str(row["condition_id"]) for row in rows))


def _condition_label(
    condition_id: str, rows: Sequence[Mapping[str, Any]], *, compact: bool = False
) -> str:
    row = next(row for row in rows if str(row["condition_id"]) == condition_id)
    model = str(row.get("model", condition_id))
    dataset = str(row.get("dataset", ""))
    separator = " / " if compact else " | "
    return separator.join(item for item in (model, dataset) if item)


def _standalone_metric_label(
    rows: Sequence[Mapping[str, Any]], metric: str
) -> str:
    is_average = any(
        str(row.get("aggregation_method", "")).startswith("equal_weight_mean")
        or str(row.get("interval_method", "")).startswith("equal_weight_mean")
        for row in rows
    )
    prefix = "Average " if is_average else ""
    return f"{prefix}{METRIC_LABELS[metric]}"


def _interval_legend_handle(rows: Sequence[Mapping[str, Any]]) -> Patch:
    is_descriptive_aggregate = any(
        str(row.get("interval_method", "")).startswith("equal_weight_mean")
        for row in rows
    )
    label = (
        "Shading: mean condition-level 95% interval bounds"
        if is_descriptive_aggregate
        else "Shading: 95% bootstrap confidence interval"
    )
    return Patch(facecolor="#7A7A7A", edgecolor="none", alpha=0.18, label=label)


def _errorbar_legend_handle(rows: Sequence[Mapping[str, Any]]) -> Line2D:
    is_descriptive_aggregate = any(
        str(row.get("interval_method", "")).startswith("equal_weight_mean")
        for row in rows
    )
    label = (
        "Error bars: mean condition-level 95% interval bounds"
        if is_descriptive_aggregate
        else "Error bars: 95% bootstrap confidence interval"
    )
    return Line2D(
        [0],
        [0],
        color="#7A7A7A",
        marker="|",
        markersize=18,
        markeredgewidth=2,
        linestyle="none",
        label=label,
    )


def _metric_rows(
    rows: Sequence[Mapping[str, Any]],
    condition_id: str,
    metric: str,
    *,
    scope: Optional[str] = None,
) -> List[Mapping[str, Any]]:
    selected = [
        row
        for row in rows
        if str(row["condition_id"]) == condition_id
        and str(row["metric"]) == metric
        and (scope is None or str(row.get("scope")) == scope)
    ]
    return sorted(selected, key=lambda row: _int(row, "k_correct"))


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "axes.labelsize": 24,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.75,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for extension in formats:
        path = output_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=dpi if extension == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _line_with_interval(
    ax: plt.Axes,
    rows: Sequence[Mapping[str, Any]],
    *,
    estimate: str,
    low: str,
    high: str,
    label: str,
    color: str,
    marker: str,
    linestyle: str = "-",
    alpha: float = 0.13,
) -> None:
    points = [
        (
            _int(row, "k_correct"),
            _float(row, estimate),
            _float(row, low),
            _float(row, high),
        )
        for row in rows
    ]
    points = [point for point in points if _finite(point[1], point[2], point[3])]
    if not points:
        return
    x = np.asarray([point[0] for point in points], dtype=np.float64)
    y = np.asarray([point[1] for point in points], dtype=np.float64)
    lower = np.asarray([point[2] for point in points], dtype=np.float64)
    upper = np.asarray([point[3] for point in points], dtype=np.float64)
    ax.plot(
        x,
        y,
        color=color,
        marker=marker,
        markersize=5,
        linestyle=linestyle,
        label=label,
    )
    ax.fill_between(x, lower, upper, color=color, alpha=alpha, linewidth=0)


def _plot_aligned_vs_null_axis(
    ax: plt.Axes,
    similarity_rows: Sequence[Mapping[str, Any]],
    condition_id: str,
    metric: str,
    *,
    show_title: bool = True,
) -> None:
    rows = _metric_rows(
        similarity_rows, condition_id, metric, scope="aggregate"
    )
    _line_with_interval(
        ax,
        rows,
        estimate="aligned_estimate",
        low="aligned_ci_low",
        high="aligned_ci_high",
        label="Same question",
        color="#0072B2",
        marker="o",
    )
    _line_with_interval(
        ax,
        rows,
        estimate="null_exact_estimate",
        low="null_ci_low",
        high="null_ci_high",
        label="Shuffled question",
        color="#6F6F6F",
        marker="s",
        linestyle="--",
    )
    if show_title:
        ax.set_title(
            f"{METRIC_LABELS[metric]}\n"
            f"{_condition_label(condition_id, similarity_rows)}"
        )
    ax.set_xticks(range(6))
    ax.set_xlim(-0.15, 5.15)


def plot_aligned_vs_null(
    similarity_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    conditions = _condition_ids(similarity_rows)
    fig, axes = plt.subplots(
        len(conditions),
        len(METRICS),
        figsize=(16.0, 4.8 * len(conditions)),
        squeeze=False,
        sharex=True,
    )
    for row_index, condition_id in enumerate(conditions):
        for column_index, metric in enumerate(METRICS):
            ax = axes[row_index, column_index]
            _plot_aligned_vs_null_axis(
                ax, similarity_rows, condition_id, metric
            )
            if column_index == 0:
                ax.set_ylabel("Similarity")
            if row_index == len(conditions) - 1:
                ax.set_xlabel("Number of correctly answered languages (K)")
    handles = [
        Line2D([0], [0], color="#0072B2", marker="o", label="Same question"),
        Line2D(
            [0],
            [0],
            color="#6F6F6F",
            marker="s",
            linestyle="--",
            label="Shuffled question",
        ),
        _interval_legend_handle(similarity_rows),
    ]
    fig.suptitle("Usage-signature similarity by multilingual correctness")
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.04),
        labelspacing=0.25,
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0, 0.085, 1, 0.95))
    written = _save_figure(
        fig, output_dir, "aligned_vs_null_by_k", formats, dpi
    )

    for metric in METRICS:
        standalone_fig, standalone_axes = plt.subplots(
            len(conditions),
            1,
            figsize=(10.5, 7.5 * len(conditions)),
            squeeze=False,
            sharex=True,
        )
        for row_index, condition_id in enumerate(conditions):
            standalone_ax = standalone_axes[row_index, 0]
            _plot_aligned_vs_null_axis(
                standalone_ax,
                similarity_rows,
                condition_id,
                metric,
                show_title=False,
            )
            standalone_ax.set_ylabel(
                _standalone_metric_label(similarity_rows, metric)
            )
            if row_index == len(conditions) - 1:
                standalone_ax.set_xlabel(
                    "Number of correctly answered languages (K)"
                )
        standalone_fig.legend(
            handles=handles,
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.5, 0.06),
            labelspacing=0.25,
            borderaxespad=0.0,
        )
        standalone_fig.tight_layout(rect=(0, 0.14, 1, 1))
        written.extend(
            _save_figure(
                standalone_fig,
                output_dir,
                f"aligned_vs_null_{metric}",
                formats,
                dpi,
            )
        )
    return written


def _trend_lookup(
    trend_rows: Sequence[Mapping[str, Any]], condition_id: str, metric: str
) -> Optional[Mapping[str, Any]]:
    return next(
        (
            row
            for row in trend_rows
            if str(row["condition_id"]) == condition_id
            and str(row["metric"]) == metric
            and str(row["outcome"]) == "gap"
        ),
        None,
    )


def _p_value_text(value: float) -> str:
    if not math.isfinite(value):
        return "not estimated"
    if value < 0.001:
        return "< 0.001"
    return f"= {value:.3f}"


def plot_gap(
    similarity_rows: Sequence[Mapping[str, Any]],
    trend_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    conditions = _condition_ids(similarity_rows)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(16.0, 6.5), squeeze=False)
    for metric_index, metric in enumerate(METRICS):
        ax = axes[0, metric_index]
        for condition_index, condition_id in enumerate(conditions):
            color = CONDITION_COLORS[condition_index % len(CONDITION_COLORS)]
            rows = _metric_rows(
                similarity_rows, condition_id, metric, scope="aggregate"
            )
            _line_with_interval(
                ax,
                rows,
                estimate="gap_estimate",
                low="gap_ci_low",
                high="gap_ci_high",
                label=_condition_label(
                    condition_id, similarity_rows, compact=True
                ),
                color=color,
                marker=("o", "s", "^", "D")[condition_index % 4],
            )
            if len(conditions) == 1:
                trend = _trend_lookup(trend_rows, condition_id, metric)
                if trend is not None:
                    slope = _float(trend, "slope_per_additional_correct_language")
                    p_holm = _float(trend, "permutation_p_holm")
                    annotation = f"Slope = {slope:+.4f} per language"
                    if math.isfinite(p_holm):
                        annotation += f"\nHolm p {_p_value_text(p_holm)}"
                    ax.text(
                        0.96,
                        0.06,
                        annotation,
                        transform=ax.transAxes,
                        ha="right",
                        va="bottom",
                        fontsize=15,
                        bbox={
                            "facecolor": "white",
                            "edgecolor": "#CCCCCC",
                            "boxstyle": "round,pad=0.35",
                            "alpha": 0.9,
                        },
                    )
        ax.axhline(0.0, color="#333333", linewidth=1.0, linestyle=":")
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel("Number of correctly answered languages (K)")
        ax.set_xticks(range(6))
        ax.set_xlim(-0.15, 5.15)
        if metric_index == 0:
            ax.set_ylabel("Same-question minus shuffled similarity")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        interval_handle = _interval_legend_handle(similarity_rows)
        fig.legend(
            [*handles, interval_handle],
            [*labels, interval_handle.get_label()],
            loc="lower center",
            ncol=min(4, len(handles) + 1),
            bbox_to_anchor=(0.5, 0.04),
            labelspacing=0.25,
            borderaxespad=0.0,
        )
    fig.suptitle("Question-specific cross-lingual similarity gap")
    fig.tight_layout(rect=(0, 0.09, 1, 0.92))
    return _save_figure(fig, output_dir, "gap_by_k", formats, dpi)


def _plot_pair_status_axis(
    ax: plt.Axes,
    status_rows: Sequence[Mapping[str, Any]],
    condition_id: str,
    metric: str,
    *,
    show_title: bool = True,
) -> None:
    for status, (label, color, marker) in STATUS_STYLES.items():
        rows = [
            row
            for row in _metric_rows(status_rows, condition_id, metric)
            if str(row.get("pair_status")) == status
        ]
        _line_with_interval(
            ax,
            rows,
            estimate="aligned_estimate",
            low="aligned_ci_low",
            high="aligned_ci_high",
            label=label,
            color=color,
            marker=marker,
        )
    if show_title:
        ax.set_title(
            f"{METRIC_LABELS[metric]}\n"
            f"{_condition_label(condition_id, status_rows)}"
        )
    ax.set_xticks(range(6))
    ax.set_xlim(-0.15, 5.15)


def plot_pair_status(
    status_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    conditions = _condition_ids(status_rows)
    fig, axes = plt.subplots(
        len(conditions),
        len(METRICS),
        figsize=(16.0, 4.8 * len(conditions)),
        squeeze=False,
        sharex=True,
    )
    for row_index, condition_id in enumerate(conditions):
        for column_index, metric in enumerate(METRICS):
            ax = axes[row_index, column_index]
            _plot_pair_status_axis(ax, status_rows, condition_id, metric)
            if column_index == 0:
                ax.set_ylabel("Same-question similarity")
            if row_index == len(conditions) - 1:
                ax.set_xlabel("Number of correctly answered languages (K)")
    handles = [
        Line2D([0], [0], color=color, marker=marker, label=label)
        for label, color, marker in STATUS_STYLES.values()
    ]
    handles.append(_interval_legend_handle(status_rows))
    fig.suptitle("Similarity by language-pair correctness")
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.04),
        labelspacing=0.25,
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.95))
    written = _save_figure(fig, output_dir, "pair_status_by_k", formats, dpi)

    for metric in METRICS:
        standalone_fig, standalone_axes = plt.subplots(
            len(conditions),
            1,
            figsize=(10.5, 7.5 * len(conditions)),
            squeeze=False,
            sharex=True,
        )
        for row_index, condition_id in enumerate(conditions):
            standalone_ax = standalone_axes[row_index, 0]
            _plot_pair_status_axis(
                standalone_ax,
                status_rows,
                condition_id,
                metric,
                show_title=False,
            )
            standalone_ax.set_ylabel(
                _standalone_metric_label(status_rows, metric)
            )
            if row_index == len(conditions) - 1:
                standalone_ax.set_xlabel(
                    "Number of correctly answered languages (K)"
                )
        standalone_fig.legend(
            handles=handles,
            loc="lower center",
            ncol=2,
            bbox_to_anchor=(0.5, 0.06),
            labelspacing=0.25,
            borderaxespad=0.0,
        )
        standalone_fig.tight_layout(rect=(0, 0.14, 1, 1))
        written.extend(
            _save_figure(
                standalone_fig,
                output_dir,
                f"pair_status_{metric}",
                formats,
                dpi,
            )
        )
    return written


def _plot_matched_subset_axis(
    ax: plt.Axes,
    matched_rows: Sequence[Mapping[str, Any]],
    conditions: Sequence[str],
    offsets: np.ndarray,
    match_type: str,
    match_label: str,
    metric: str,
    *,
    show_title: bool = True,
) -> None:
    for condition_index, condition_id in enumerate(conditions):
        color = CONDITION_COLORS[condition_index % len(CONDITION_COLORS)]
        rows = [
            row
            for row in matched_rows
            if str(row["condition_id"]) == condition_id
            and str(row["metric"]) == metric
            and str(row["match_type"]) == match_type
        ]
        points = []
        for row in rows:
            estimate = _float(row, "partial_minus_all_correct")
            low = _float(row, "difference_ci_low")
            high = _float(row, "difference_ci_high")
            if _finite(estimate, low, high):
                points.append((_int(row, "k_correct"), estimate, low, high))
        points.sort()
        if not points:
            continue
        x = np.asarray([point[0] for point in points], dtype=np.float64)
        y = np.asarray([point[1] for point in points], dtype=np.float64)
        low = np.asarray([point[2] for point in points], dtype=np.float64)
        high = np.asarray([point[3] for point in points], dtype=np.float64)
        ax.errorbar(
            x + offsets[condition_index],
            y,
            yerr=np.vstack((y - low, high - y)),
            color=color,
            marker=("o", "s", "^", "D")[condition_index % 4],
            markersize=8,
            linestyle="none",
            capsize=4,
            elinewidth=1.5,
            label=_condition_label(condition_id, matched_rows, compact=True),
        )
    ax.axhline(0.0, color="#333333", linewidth=1.0, linestyle=":")
    if show_title:
        ax.set_title(f"{match_label}\n{METRIC_LABELS[metric]}")
    ax.set_xticks((2, 3, 4))
    ax.set_xlim(1.65, 4.35)


def plot_matched_subsets(
    matched_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    conditions = _condition_ids(matched_rows)
    confidence_margin_keys = {
        str(row.get("confidence_margin_key"))
        for row in matched_rows
        if str(row.get("match_type")) == "confidence"
        and row.get("confidence_margin_key")
    }
    confidence_label = (
        "Strongest-incorrect confidence matching"
        if confidence_margin_keys == {"base_margin_max"}
        else "Confidence matching"
    )
    labels = (
        ("confidence", confidence_label),
        ("difficulty", "Difficulty matching"),
    )
    available_match_types = {str(row.get("match_type")) for row in matched_rows}
    match_types: Tuple[Tuple[str, str], ...] = tuple(
        item for item in labels if item[0] in available_match_types
    )
    if not match_types:
        raise ValueError("No supported matching types found in matched-subset results.")
    fig, axes = plt.subplots(
        len(match_types),
        len(METRICS),
        figsize=(16.0, 5.5 * len(match_types) + 0.5),
        squeeze=False,
        sharex=True,
    )
    offsets = (
        np.linspace(-0.18, 0.18, len(conditions))
        if len(conditions) > 1
        else np.asarray([0.0])
    )
    for row_index, (match_type, match_label) in enumerate(match_types):
        for column_index, metric in enumerate(METRICS):
            ax = axes[row_index, column_index]
            _plot_matched_subset_axis(
                ax,
                matched_rows,
                conditions,
                offsets,
                match_type,
                match_label,
                metric,
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        interval_handle = _errorbar_legend_handle(matched_rows)
        fig.legend(
            [*handles, interval_handle],
            [*labels, interval_handle.get_label()],
            loc="lower center",
            ncol=min(4, len(handles) + 1),
            bbox_to_anchor=(0.5, 0.035),
            labelspacing=0.25,
            borderaxespad=0.0,
        )
    fig.suptitle("Matched partially correct questions")
    fig.supxlabel("Number of correctly answered languages (K)", y=0.085)
    fig.supylabel("Partial correct - matched all correct", x=0.015)
    fig.tight_layout(rect=(0.055, 0.14, 1, 0.94))
    written = _save_figure(fig, output_dir, "matched_subsets", formats, dpi)

    for match_type, match_label in match_types:
        for metric in METRICS:
            standalone_fig, standalone_ax = plt.subplots(figsize=(10.5, 7.5))
            _plot_matched_subset_axis(
                standalone_ax,
                matched_rows,
                conditions,
                offsets,
                match_type,
                match_label,
                metric,
                show_title=False,
            )
            standalone_ax.set_ylabel("Partial correct - matched all correct")
            standalone_ax.set_xlabel("Number of correctly answered languages (K)")
            handles, labels = standalone_ax.get_legend_handles_labels()
            if handles:
                interval_handle = _errorbar_legend_handle(matched_rows)
                standalone_fig.legend(
                    [*handles, interval_handle],
                    [*labels, interval_handle.get_label()],
                    loc="lower center",
                    ncol=1,
                    bbox_to_anchor=(0.5, 0.04),
                    labelspacing=0.25,
                    borderaxespad=0.0,
                )
            standalone_fig.tight_layout(rect=(0, 0.135, 1, 1))
            written.extend(
                _save_figure(
                    standalone_fig,
                    output_dir,
                    f"matched_subsets_{match_type}_{metric}",
                    formats,
                    dpi,
                )
            )
    return written


def visualize(
    input_dir: Path,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    if dpi <= 0:
        raise ValueError("DPI must be positive.")
    unsupported = sorted(set(formats) - {"png", "pdf", "svg"})
    if unsupported:
        raise ValueError(f"Unsupported output formats: {', '.join(unsupported)}")
    for filename in REQUIRED_FILES:
        if not (input_dir / filename).is_file():
            raise FileNotFoundError(f"Missing analysis result: {input_dir / filename}")

    similarity_rows = _read_csv(input_dir / "similarity_by_k.csv")
    status_rows = _read_csv(input_dir / "similarity_by_pair_status.csv")
    trend_rows = _read_csv(input_dir / "ordinal_trends.csv")
    matched_rows = _read_csv(input_dir / "matched_subset_results.csv")
    return visualize_rows(
        similarity_rows=similarity_rows,
        status_rows=status_rows,
        trend_rows=trend_rows,
        matched_rows=matched_rows,
        output_dir=output_dir,
        formats=formats,
        dpi=dpi,
    )


def visualize_rows(
    similarity_rows: Sequence[Mapping[str, Any]],
    status_rows: Sequence[Mapping[str, Any]],
    trend_rows: Sequence[Mapping[str, Any]],
    matched_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    """Render figures from one or more conditions' in-memory result rows."""
    if dpi <= 0:
        raise ValueError("DPI must be positive.")
    unsupported = sorted(set(formats) - {"png", "pdf", "svg"})
    if unsupported:
        raise ValueError(f"Unsupported output formats: {', '.join(unsupported)}")
    if not similarity_rows:
        raise ValueError("No similarity rows supplied.")
    if not status_rows:
        raise ValueError("No pair-status rows supplied.")
    if not matched_rows:
        raise ValueError("No matched rows supplied.")

    _set_style()
    written: List[Path] = []
    written.extend(
        plot_aligned_vs_null(similarity_rows, output_dir, formats, dpi)
    )
    written.extend(
        plot_gap(similarity_rows, trend_rows, output_dir, formats, dpi)
    )
    written.extend(plot_pair_status(status_rows, output_dir, formats, dpi))
    written.extend(plot_matched_subsets(matched_rows, output_dir, formats, dpi))
    return written


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    try:
        if args.input_dir:
            input_dir = Path(args.input_dir).resolve()
        else:
            input_dir = (
                Path(args.analysis_root).resolve()
                / "by_condition"
                / str(args.condition)
            )
        output_dir = (
            Path(args.output_dir).resolve()
            if args.output_dir
            else input_dir / "viz"
        )
        written = visualize(
            input_dir=input_dir,
            output_dir=output_dir,
            formats=tuple(args.formats),
            dpi=int(args.dpi),
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error

    print(f"Read analysis results from {input_dir}")
    print(f"Saved {len(written)} visualization files to {output_dir}")
    for path in written:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
