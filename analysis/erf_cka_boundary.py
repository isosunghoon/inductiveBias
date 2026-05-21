"""Boundary matching analysis between block-wise ERD and same-model CKA.

This standalone module consumes precomputed ERD and CKA matrices and tests
whether both matrices jump at the same layer transitions.
"""

import sys
import os
import argparse
import csv
import glob
import math
import re
from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from scipy.stats import pearsonr
except ImportError:
    pearsonr = None

from analysis._model_compat import _MODEL_DISPLAY_NAMES, display_name


MODEL_KEYS = [
    "convformer",
    "convformer_w5",
    "denseformer",
    "identity",
    "localvit",
    "localvit_w5",
    "vit",
]

LOCALITY_LABEL = {
    "convformer": "local",
    "convformer_w5": "local",
    "localvit": "local",
    "localvit_w5": "local",
    "denseformer": "global",
    "vit": "global",
    "identity": "global",
}

LAYER_LABELS = ["patch_embed"] + [f"block_{i}" for i in range(12)]
TRANSITION_LABELS = [
    f"{LAYER_LABELS[i]}→{LAYER_LABELS[i + 1]}" for i in range(len(LAYER_LABELS) - 1)
]

LOCAL_COLOR = "#2f6fdb"
GLOBAL_COLOR = "#c94c4c"
BASELINE_COLOR = "#7a7a7a"
TIMESTAMP_RE = re.compile(r"-(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")


@dataclass(frozen=True)
class BoundaryResult:
    """Result bundle for one model's ERD/CKA boundary matching analysis."""

    model: str
    locality: str
    pearson_r: float
    p_value: float
    erd_curve: np.ndarray
    cka_curve: np.ndarray
    erd_path: str
    cka_path: str


def _model_display_name(model_key: str) -> str:
    """Return display name, with w5 variants derived from the base model."""
    if model_key in _MODEL_DISPLAY_NAMES:
        return display_name(model_key)
    if model_key.endswith("_w5"):
        return f"{display_name(model_key[:-3])} w5"
    return display_name(model_key)


def _style_color(model_key: str) -> str:
    """Return the plot color for a model/locality group."""
    if model_key == "identity":
        return BASELINE_COLOR
    return LOCAL_COLOR if LOCALITY_LABEL[model_key] == "local" else GLOBAL_COLOR


def _latest_file(directory: str, pattern: str) -> str:
    """Return the newest file matching pattern by filename timestamp."""
    matches = glob.glob(os.path.join(directory, pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched {os.path.join(directory, pattern)}")
    return max(matches, key=_file_recency_key)


def _file_recency_key(path: str) -> tuple[str, float]:
    """Return sortable timestamp key, with mtime as a stable fallback."""
    match = TIMESTAMP_RE.search(os.path.basename(path))
    timestamp = match.group(1) if match else ""
    return timestamp, os.path.getmtime(path)


def _has_model_files(directory: str, suffix: str) -> bool:
    """Return whether a directory contains one file for every required model."""
    return all(glob.glob(os.path.join(directory, f"{key}-*{suffix}")) for key in MODEL_KEYS)


def resolve_erd_dir(erd_dir: str) -> str:
    """Resolve ERD directory, falling back to sibling output variants."""
    if _has_model_files(erd_dir, "_erf_layers.npy"):
        return erd_dir

    parent = os.path.dirname(erd_dir)
    candidates = [
        path for path in glob.glob(os.path.join(parent, "*"))
        if os.path.isdir(path) and _has_model_files(path, "_erf_layers.npy")
    ]
    if not candidates:
        return erd_dir
    return max(candidates, key=os.path.getmtime)


def find_erd_file(erd_dir: str, model_key: str) -> str:
    """Find the newest ERD matrix for a model."""
    return _latest_file(erd_dir, f"{model_key}-*_erf_layers.npy")


def find_cka_file(cka_dir: str, model_key: str) -> str:
    """Find the newest same-model CKA matrix for a model."""
    return _latest_file(cka_dir, f"{model_key}-*_vs_{model_key}-*_cka.txt")


def load_erd_matrix(path: str) -> np.ndarray:
    """Load a saved ERD matrix and align it to CKA's 13 analysis layers."""
    mat = np.load(path)
    if mat.shape == (14, 14):
        mat = mat[1:, 1:]
    if mat.shape != (13, 13):
        raise ValueError(f"Expected ERD shape (13, 13) or (14, 14), got {mat.shape}: {path}")
    return mat.astype(np.float64, copy=False)


def load_cka_matrix(path: str) -> np.ndarray:
    """Load a tab-separated CKA text matrix with metadata/header rows."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 14 or parts[0] not in LAYER_LABELS:
                continue
            rows.append([float(v) for v in parts[1:]])
    mat = np.asarray(rows, dtype=np.float64)
    if mat.shape != (13, 13):
        raise ValueError(f"Expected CKA shape (13, 13), got {mat.shape}: {path}")
    return mat


def boundary_curve(mat: np.ndarray) -> np.ndarray:
    """Compute row-difference boundary scores for a square layer matrix."""
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {mat.shape}")
    diffs = mat[1:] - mat[:-1]
    return np.linalg.norm(diffs, axis=1)


def safe_pearsonr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return Pearson r/p, using NaN when variance or SciPy is unavailable."""
    if x.shape != y.shape:
        raise ValueError(f"Curve shapes differ: {x.shape} vs {y.shape}")
    if len(x) != 12:
        raise ValueError(f"Expected boundary curve length 12, got {len(x)}")
    if np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
        return float("nan"), float("nan")
    if pearsonr is None:
        return float(np.corrcoef(x, y)[0, 1]), float("nan")
    result = pearsonr(x, y)
    return float(result[0]), float(result[1])


def analyze_model(erd_dir: str, cka_dir: str, model_key: str) -> BoundaryResult:
    """Load matrices, compute boundary curves, and correlate them for one model."""
    erd_path = find_erd_file(erd_dir, model_key)
    cka_path = find_cka_file(cka_dir, model_key)
    erd_curve = boundary_curve(load_erd_matrix(erd_path))
    cka_curve = boundary_curve(load_cka_matrix(cka_path))
    r, p = safe_pearsonr(erd_curve, cka_curve)
    return BoundaryResult(
        model=model_key,
        locality=LOCALITY_LABEL[model_key],
        pearson_r=r,
        p_value=p,
        erd_curve=erd_curve,
        cka_curve=cka_curve,
        erd_path=erd_path,
        cka_path=cka_path,
    )


def _format_stat(value: float) -> str:
    """Format a statistic for plot titles."""
    if math.isnan(value):
        return "nan"
    return f"{value:.3f}"


def _scatter_boundary(ax: plt.Axes, result: BoundaryResult) -> None:
    """Draw ERD-vs-CKA boundary scatter plot on an axis."""
    color = _style_color(result.model)
    ax.scatter(result.erd_curve, result.cka_curve, s=55, color=color, alpha=0.85)
    ax.set_xlabel("ERD jump magnitude", fontsize=12)
    ax.set_ylabel("CKA jump magnitude", fontsize=12)
    ax.set_title(
        f"r={_format_stat(result.pearson_r)}, p={_format_stat(result.p_value)}",
        fontsize=13,
    )
    ax.grid(True, alpha=0.25)


def plot_boundary_match_full(result: BoundaryResult) -> plt.Figure:
    """Create a three-panel boundary matching figure for one model."""
    x = np.arange(len(result.erd_curve))
    color = _style_color(result.model)
    fig, axes = plt.subplots(3, 1, figsize=(9, 10))
    title = f"{_model_display_name(result.model)} | ERD-CKA Boundary Match"
    fig.suptitle(title, fontsize=18)

    axes[0].plot(x, result.erd_curve, marker="o", color=color, linewidth=2)
    axes[0].set_ylabel("ERD jump magnitude", fontsize=12)
    axes[0].set_title("ERD boundary curve", fontsize=13)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(x, result.cka_curve, marker="o", color=color, linewidth=2)
    axes[1].set_ylabel("CKA jump magnitude", fontsize=12)
    axes[1].set_title("CKA boundary curve", fontsize=13)
    axes[1].grid(True, alpha=0.25)

    _scatter_boundary(axes[2], result)
    for ax in axes[:2]:
        ax.set_xticks(x)
        ax.set_xticklabels(TRANSITION_LABELS, rotation=45, ha="right", fontsize=9)
        ax.set_xlabel("Layer transition", fontsize=12)
    plt.tight_layout()
    return fig


def plot_boundary_match_compact(result: BoundaryResult) -> plt.Figure:
    """Create a compact scatter-only boundary matching figure for one model."""
    fig, ax = plt.subplots(figsize=(5, 5))
    _scatter_boundary(ax, result)
    ax.set_title(
        f"{_model_display_name(result.model)} | r={_format_stat(result.pearson_r)}, "
        f"p={_format_stat(result.p_value)}",
        fontsize=14,
    )
    plt.tight_layout()
    return fig


def save_summary_csv(results: Iterable[BoundaryResult], path: str) -> None:
    """Save model-level Pearson results to a CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "locality", "pearson_r", "p_value", "n"])
        for result in results:
            writer.writerow([
                result.model,
                result.locality,
                result.pearson_r,
                result.p_value,
                len(result.erd_curve),
            ])


def plot_summary_bar(results: list[BoundaryResult]) -> plt.Figure:
    """Create a bar chart summarizing Pearson r across models."""
    names = [_model_display_name(r.model) for r in results]
    values = [r.pearson_r for r in results]
    colors = [_style_color(r.model) for r in results]
    fig_w = max(7, len(results) * 0.9)
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))

    ax.bar(np.arange(len(results)), values, color=colors)
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel("Pearson r", fontsize=13)
    ax.set_title("ERD-CKA Boundary Matching", fontsize=14)
    ax.set_xticks(np.arange(len(results)))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    return fig


def run_boundary_analysis(args: argparse.Namespace) -> list[BoundaryResult]:
    """Run H1 boundary matching analysis and write all outputs."""
    os.makedirs(args.output_dir, exist_ok=True)
    erd_dir = resolve_erd_dir(args.erd_dir)
    if erd_dir != args.erd_dir:
        print(f"[boundary] ERD dir not complete; using {erd_dir}", flush=True)
    results = [analyze_model(erd_dir, args.cka_dir, key) for key in MODEL_KEYS]

    for result in results:
        plot_fn = plot_boundary_match_full
        if args.viz_mode == "compact":
            plot_fn = plot_boundary_match_compact
        fig = plot_fn(result)
        save_path = os.path.join(args.output_dir, f"{result.model}_boundary.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    save_summary_csv(results, os.path.join(args.output_dir, "summary.csv"))
    fig = plot_summary_bar(results)
    plt.savefig(os.path.join(args.output_dir, "summary_bar.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return results


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for standalone execution."""
    parser = argparse.ArgumentParser(description="ERD-CKA boundary matching analysis")
    parser.add_argument("--erd_dir", default="analysis_output/erf_layers/model_resized")
    parser.add_argument("--cka_dir", default="analysis_output/cka/model_resized")
    parser.add_argument("--output_dir", default="analysis_output/boundary_match")
    parser.add_argument("--viz_mode", choices=["full", "compact"], default="full")
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()
    results = run_boundary_analysis(args)
    for result in results:
        print(
            f"{result.model}: r={_format_stat(result.pearson_r)}, "
            f"p={_format_stat(result.p_value)}, n={len(result.erd_curve)}"
        )


if __name__ == "__main__":
    main()
