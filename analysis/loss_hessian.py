"""
Pipeline-compatible loss landscape analysis (Hessian eigenvalues).

Usage
-----
    from analysis.pipeline import run_pipeline
    from analysis.loss_landscape_fn import analyze_loss_landscape

    run_pipeline(
        project_path="output/base_model_exp",
        analysis_fns={"loss_landscape": analyze_loss_landscape},
        batch_size=16,
        ratio=0.05,
    )

Outputs per model
-----------------
    {model_name}_loss_landscape_eigenvalues.npy   – shape (steps, top_n)
    {model_name}_loss_landscape_histogram.png     – eigenvalue frequency histogram
"""

from __future__ import annotations

import warnings
import torch
import numpy as np
import matplotlib.pyplot as plt

from pyhessian import hessian
from tqdm import tqdm

from utils.dataset import get_dataloader, make_subset_loader

warnings.filterwarnings(
    "ignore",
    message="Using backward\\(\\) with create_graph=True will create a reference cycle.*",
    category=UserWarning,
)


def _compute_eigenvalues(args, model, loader, mixup_fn, top_n: int = 5) -> np.ndarray:
    """Run pyhessian over each batch in loader and collect top-n eigenvalues.

    Returns
    -------
    np.ndarray of shape (steps, top_n)
    """
    model.train()
    res = []
    pbar = tqdm(loader, total=len(loader), desc="Hessian", dynamic_ncols=True)

    for step, batch in enumerate(pbar):
        model.zero_grad(set_to_none=True)

        x, y = batch
        x = x.to(args.device, non_blocking=True)
        y = y.to(args.device, non_blocking=True)

        if mixup_fn is not None:
            x, y = mixup_fn(x, y)

        def criterion(logits, targets):
            with torch.amp.autocast("cuda", enabled=args.fp16):
                return torch.nn.functional.cross_entropy(
                    logits, targets, label_smoothing=args.label_smoothing
                )

        hessian_comp = hessian(
            model, criterion, data=(x, y), cuda=(args.device == "cuda")
        )
        eigenvalues, _ = hessian_comp.eigenvalues(top_n=top_n)
        res.append(eigenvalues)
        model.zero_grad(set_to_none=True)
        pbar.set_postfix({"step": step + 1, "top1": float(eigenvalues[0])})

    return np.asarray(res)  # (steps, top_n)


def _make_histogram_fig(eigvals: np.ndarray, model_name: str) -> plt.Figure:
    """Build and return a histogram figure for the eigenvalue distribution.

    Parameters
    ----------
    eigvals : np.ndarray of shape (steps, top_n)
    model_name : str  – used as the plot title/label
    """
    flat = eigvals.flatten()
    bins = np.linspace(flat.min(), flat.max(), 51)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(flat, bins=bins, label=model_name, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.set_title(f"Hessian Eigenvalue Distribution — {model_name}")
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    return fig


def analyze_loss_landscape(
    args,
    model,
    batch_size: int = 16,
    ratio: float = 0.05,
    top_n: int = 5,
    **kwargs,
):
    """Pipeline-compatible Hessian eigenvalue analysis.

    Parameters
    ----------
    args        : parsed config (from pipeline._load_args)
    model       : loaded model (from pipeline.build_model)
    batch_size  : mini-batch size used for Hessian computation
    ratio       : fraction of training set to use as subset
    top_n       : number of top eigenvalues to compute per batch

    Returns
    -------
    list[AnalysisResult]
        - suffix="loss_landscape_eigenvalues", fmt=".npy"  – shape (steps, top_n)
        - suffix="loss_landscape_histogram",   fmt=".png"  – histogram figure
    """
    from analysis.pipeline import AnalysisResult

    # Unwrap torch.compile — pyhessian needs double backward which aot_autograd doesn't support
    model = getattr(model, "_orig_mod", model)

    # Override batch size; keep fp16 flag if already set
    args.train_batch_size = batch_size
    if not hasattr(args, "fp16"):
        args.fp16 = True

    train_loader, _, mixup_fn = get_dataloader(args)
    loader = make_subset_loader(args, train_loader, ratio=ratio)

    eigvals = _compute_eigenvalues(args, model, loader, mixup_fn, top_n=top_n)

    model_name = str(getattr(args, "output_path", "model")).rstrip("/").split("/")[-1]
    fig = _make_histogram_fig(eigvals, model_name)

    return [
        AnalysisResult("loss_landscape_eigenvalues", eigvals, ".npy"),
        AnalysisResult("loss_landscape_histogram", fig, ".png"),
    ]
