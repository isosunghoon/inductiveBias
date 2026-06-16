"""
Pipeline-compatible Hessian spectrum analysis.

This module estimates per-minibatch minimum and maximum Hessian eigenvalues
with Lanczos iterations using Hessian-vector products. It does not materialize
the full Hessian matrix.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from analysis.pipeline import AnalysisResult
from utils.dataset import get_dataloader


@contextmanager
def _sdp_math_backend(device: str):
    """Force MATH SDPA so Hessian-vector products can double-backward through attention."""
    if not str(device).startswith("cuda") or not torch.backends.cuda.is_built():
        yield
        return

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            yield
    except (ImportError, AttributeError, TypeError):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_mem_efficient=False,
            enable_math=True,
        ):
            yield


def _trainable_params(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def _flatten_tensors(tensors: Iterable[torch.Tensor | None], params: list[torch.nn.Parameter]) -> torch.Tensor:
    flat = []
    for tensor, param in zip(tensors, params):
        if tensor is None:
            flat.append(torch.zeros(param.numel(), device=param.device, dtype=param.dtype))
        else:
            flat.append(tensor.contiguous().view(-1))
    return torch.cat(flat)


def _hessian_vector_product(
    loss: torch.Tensor,
    params: list[torch.nn.Parameter],
    vector: torch.Tensor,
) -> torch.Tensor:
    grads = torch.autograd.grad(
        loss,
        params,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )
    flat_grad = _flatten_tensors(grads, params)
    grad_vector_product = torch.dot(flat_grad, vector)
    hvp = torch.autograd.grad(
        grad_vector_product,
        params,
        retain_graph=True,
        allow_unused=True,
    )
    return _flatten_tensors(hvp, params).detach()


def _lanczos_extreme_eigenvalues(
    loss: torch.Tensor,
    params: list[torch.nn.Parameter],
    steps: int,
    eps: float = 1e-8,
) -> tuple[float, float]:
    device = loss.device
    dtype = params[0].dtype
    dim = sum(p.numel() for p in params)

    q = torch.randn(dim, device=device, dtype=dtype)
    q = q / q.norm().clamp_min(eps)
    q_prev = torch.zeros_like(q)
    beta_prev = torch.zeros((), device=device, dtype=dtype)

    alphas = []
    betas = []

    for _ in range(steps):
        z = _hessian_vector_product(loss, params, q)
        alpha = torch.dot(q, z)
        z = z - alpha * q - beta_prev * q_prev

        beta = z.norm()
        alphas.append(alpha.detach())

        if beta.item() < eps:
            break

        betas.append(beta.detach())
        q_prev = q
        q = z / beta
        beta_prev = beta

    if not alphas:
        return math.nan, math.nan

    diag = torch.stack(alphas)
    n = diag.numel()
    tri = torch.diag(diag)
    if n > 1:
        off_diag = torch.stack(betas[: n - 1])
        tri = tri + torch.diag(off_diag, diagonal=1) + torch.diag(off_diag, diagonal=-1)

    eigvals = torch.linalg.eigvalsh(tri.float()).detach().cpu().numpy()
    return float(eigvals.max()), float(eigvals.min())


def _batch_loss(args, model, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    use_amp = bool(getattr(args, "fp16", False)) and str(args.device).startswith("cuda")
    with torch.amp.autocast("cuda", enabled=use_amp):
        logits = model(x)
        return torch.nn.functional.cross_entropy(
            logits,
            y,
            label_smoothing=getattr(args, "label_smoothing", 0.0),
        )


def _trim_percentile(values: np.ndarray, trim_percent: float) -> tuple[np.ndarray, tuple[float, float] | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return finite, None
    if trim_percent <= 0.0:
        return finite, None

    trim_percent = min(trim_percent, 0.49)
    lo = float(np.quantile(finite, trim_percent))
    hi = float(np.quantile(finite, 1.0 - trim_percent))
    trimmed = finite[(finite >= lo) & (finite <= hi)]
    if trimmed.size == 0:
        return finite, None
    return trimmed, (lo, hi)


def _make_density_subplots(
    ev_pairs: np.ndarray,
    model_name: str,
    trim_percent: float = 0.10,
) -> plt.Figure:
    max_ev = ev_pairs[:, 0]
    min_ev = ev_pairs[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    trim_pct_label = int(round(trim_percent * 100))

    for ax, values, title, color in [
        (axes[0], max_ev, "Max Hessian eigenvalue", "#2563eb"),
        (axes[1], min_ev, "Min Hessian eigenvalue", "#dc2626"),
    ]:
        finite = values[np.isfinite(values)]
        trimmed, bounds = _trim_percentile(values, trim_percent=trim_percent)
        if finite.size == 0:
            ax.text(0.5, 0.5, "No finite values", ha="center", va="center")
        else:
            plot_values = trimmed if trimmed.size else finite
            bins = min(40, max(8, int(np.sqrt(plot_values.size) * 2)))
            ax.hist(
                plot_values,
                bins=bins,
                density=True,
                alpha=0.78,
                color=color,
                edgecolor="black",
                linewidth=0.4,
            )
            ax.axvline(float(np.mean(plot_values)), color="black", linestyle="--", linewidth=1.2, label="mean")
            if bounds is not None:
                low, high = bounds
                ax.axvline(low, color=color, linestyle=":", linewidth=1.0, alpha=0.8)
                ax.axvline(high, color=color, linestyle=":", linewidth=1.0, alpha=0.8)
                ax.text(
                    0.02,
                    0.96,
                    f"trimmed {trim_pct_label}% tails\nkept {plot_values.size}/{finite.size}",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=9,
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
                )
            else:
                ax.text(
                    0.02,
                    0.96,
                    f"kept {plot_values.size}/{finite.size}",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=9,
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
                )
            ax.legend(frameon=False)
        ax.set_title(title)
        ax.set_xlabel("Eigenvalue")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.25)

    fig.suptitle(f"Hessian Spectrum Extremes - {model_name}")
    fig.tight_layout()
    return fig


def analyze_hessian_spectrum(
    args,
    model,
    batch_size: int = 16,
    lanczos_steps: int = 30,
    num_batches: int | None = None,
    trim_percent: float = 0.10,
    **kwargs,
) -> list[AnalysisResult]:
    """Estimate minibatch-wise min/max Hessian eigenvalues.

    Returns
    -------
    list[AnalysisResult]
        - hessian_spectrum_extreme_eigenvalues.npy:
          ndarray with shape (num_batches, 2), columns [max_ev, min_ev]
        - hessian_spectrum_density.png:
          side-by-side density histograms for max_ev and min_ev
          after optionally trimming the upper/lower percentile tails.
    """
    del kwargs

    model = getattr(model, "_orig_mod", model)
    model.eval()
    model.zero_grad(set_to_none=True)

    args.train_batch_size = batch_size
    args.fp16 = False

    train_loader, _, mixup_fn = get_dataloader(args)
    params = _trainable_params(model)
    if not params:
        raise RuntimeError("No trainable parameters found for Hessian spectrum analysis.")

    max_steps = len(train_loader) if num_batches is None else min(len(train_loader), int(num_batches))
    if max_steps <= 0:
        raise ValueError(f"num_batches must select at least one batch, got {num_batches!r}.")

    ev_pairs = []

    pbar = tqdm(train_loader, total=max_steps, desc="Hessian spectrum", dynamic_ncols=True)
    for step, (x, y) in enumerate(pbar):
        if step >= max_steps:
            break

        model.zero_grad(set_to_none=True)
        x = x.to(args.device, non_blocking=True)
        y = y.to(args.device, non_blocking=True)
        if mixup_fn is not None:
            x, y = mixup_fn(x, y)

        with _sdp_math_backend(args.device):
            loss = _batch_loss(args, model, x, y)
            max_ev, min_ev = _lanczos_extreme_eigenvalues(loss, params, steps=lanczos_steps)
        ev_pairs.append([max_ev, min_ev])
        pbar.set_postfix({"max_ev": f"{max_ev:.3g}", "min_ev": f"{min_ev:.3g}"})

        del loss
        model.zero_grad(set_to_none=True)
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    ev_pairs = np.asarray(ev_pairs, dtype=np.float64)
    model_name = str(getattr(args, "output_path", "model")).rstrip("/").split("/")[-1]
    fig = _make_density_subplots(ev_pairs, model_name, trim_percent=trim_percent)

    return [
        AnalysisResult("hessian_spectrum_extreme_eigenvalues", ev_pairs, ".npy"),
        AnalysisResult("hessian_spectrum_density", fig, ".png"),
    ]
