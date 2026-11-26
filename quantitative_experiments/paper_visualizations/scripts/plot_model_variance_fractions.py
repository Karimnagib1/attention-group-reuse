#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parent
QUANT_ROOT = PAPER_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(PAPER_DIR / ".matplotlib"))

import matplotlib.pyplot as plt


DEFAULT_MODEL_REPORTS = {
    "qwen3_8b": {
        "label": "Qwen3-8B",
        "reports": [
            QUANT_ROOT / "qwen_outputs/belebele/multilingual_head_patch/test/variance_decomposition_continuous/pct_var_mean_incorrect/layer_component_distribution.json",
            QUANT_ROOT / "qwen_outputs/mmluprox/multilingual_head_patch/test/variance_decomposition_continuous/pct_var_mean_incorrect/layer_component_distribution.json",
        ],
    },
    "llama_3_1_8b": {
        "label": "Llama-3.1-8B",
        "reports": [
            QUANT_ROOT / "llama_outputs/belebele/multilingual_head_patch/test/variance_decomposition_continuous/pct_var_mean_incorrect/layer_component_distribution.json",
            QUANT_ROOT / "llama_outputs/mmluprox/multilingual_head_patch/test/variance_decomposition_continuous/pct_var_mean_incorrect/layer_component_distribution.json",
        ],
    },
}

COMPONENTS = [
    ("f_lang", "Language", "#e76f51"),
    ("f_question", "Question", "#2a9d8f"),
    ("f_interaction", "Interaction", "#457b9d"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create no-title model-specific layerwise variance-fraction plots."
    )
    parser.add_argument(
        "--outdir",
        default=str(PAPER_DIR / "figures"),
        help="Output directory for figures and CSV files.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png"],
        help="Figure formats to save, e.g. pdf png svg.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="DPI for raster outputs.")
    parser.add_argument("--width", type=float, default=10.8, help="Figure width in inches.")
    parser.add_argument("--height", type=float, default=5.3, help="Figure height in inches.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_component_curves(path: Path) -> dict[str, np.ndarray]:
    payload = load_json(path)
    rows = payload["per_layer"]
    return {
        "layer": np.array([int(row["layer"]) for row in rows], dtype=np.int64),
        "f_lang": np.array([float(row["f_lang"]) for row in rows], dtype=np.float64),
        "f_question": np.array([float(row["f_question"]) for row in rows], dtype=np.float64),
        "f_interaction": np.array([float(row["f_interaction"]) for row in rows], dtype=np.float64),
    }


def average_reports(paths: list[Path]) -> dict[str, np.ndarray]:
    curves = [load_component_curves(path) for path in paths]
    layers = curves[0]["layer"]
    for path, curve in zip(paths[1:], curves[1:]):
        if not np.array_equal(curve["layer"], layers):
            raise ValueError(f"Layer indices do not match for {path}.")

    out = {"layer": layers}
    for key, _, _ in COMPONENTS:
        out[key] = np.stack([curve[key] for curve in curves], axis=0).mean(axis=0)
    return out


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 24,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 24,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def plot_model(curves: dict[str, np.ndarray], outdir: Path, stem: str, formats: list[str], dpi: int, size: tuple[float, float]) -> None:
    configure_matplotlib()

    fig, ax = plt.subplots(figsize=size)
    bottom = np.zeros_like(curves["layer"], dtype=np.float64)
    bar_width = 0.85
    for key, label, color in COMPONENTS:
        ax.bar(
            curves["layer"],
            curves[key],
            bottom=bottom,
            color=color,
            width=bar_width,
            label=label,
        )
        bottom += curves[key]

    ax.set_xlim(float(curves["layer"][0]) - bar_width / 2.0, float(curves["layer"][-1]) + bar_width / 2.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Variance fraction")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.96, bottom=0.30)

    outdir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(outdir / f"{stem}.{fmt}", dpi=dpi)
    plt.close(fig)


def write_csv(curves: dict[str, np.ndarray], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["layer", "language", "question", "interaction"])
        for layer, lang, question, interaction in zip(
            curves["layer"],
            curves["f_lang"],
            curves["f_question"],
            curves["f_interaction"],
        ):
            writer.writerow([int(layer), lang, question, interaction])


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)

    for model_key, spec in DEFAULT_MODEL_REPORTS.items():
        curves = average_reports([Path(path) for path in spec["reports"]])
        stem = f"variance_fraction_by_layer_{model_key}"
        plot_model(
            curves=curves,
            outdir=outdir,
            stem=stem,
            formats=args.formats,
            dpi=args.dpi,
            size=(args.width, args.height),
        )
        write_csv(curves, outdir / f"{stem}_curve.csv")
        print(f"Saved {spec['label']}: {outdir / f'{stem}.pdf'}")


if __name__ == "__main__":
    main()
