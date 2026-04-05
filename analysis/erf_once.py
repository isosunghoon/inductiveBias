"""
Pipeline-compatible ERF analysis function.

Computes the Effective Receptive Field (ERF) and produces both a raw numpy
array and a heatmap figure in one pass.

Reference:
    Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs
    https://arxiv.org/abs/2203.06717

Usage in run_analysis.py::

    from analysis.erf_once import analyze_erf

    ANALYSIS_FNS = {"erf": analyze_erf}
"""

import types

import matplotlib.pyplot as plt
import numpy as np
import torch

from timm.utils import AverageMeter
from tqdm import tqdm

from utils.dataset import get_dataloader, make_subset_loader
from analysis.pipeline import AnalysisResult


# ---------------------------------------------------------------------------
# Defaults (can be overridden via kwargs in run_pipeline)
# ---------------------------------------------------------------------------

_RATIO      = 0.05   # subset ratio of train set when NUM_IMAGES is None
_BATCH_SIZE = 1      # ERF requires batch_size=1 for per-image gradients
_NUM_IMAGES = 1000   # stop after this many images (None = use full subset)


# ---------------------------------------------------------------------------
# ERF computation
# ---------------------------------------------------------------------------

def _erf_forward(self, x):
    """Patched forward: returns feature map [B, C, H, W] without GAP or head."""
    x = self.forward_embeddings(x)
    x = self.forward_tokens(x)
    x = self.norm(x)
    return x


def _get_input_grad(model, samples):
    outputs = model(samples)
    h, w = outputs.size(2), outputs.size(3)
    central_point = torch.nn.functional.relu(
        outputs[:, :, h // 2, w // 2]
    ).sum()
    grad = torch.autograd.grad(central_point, samples)[0]
    grad = torch.nn.functional.relu(grad)
    return grad.sum((0, 1)).cpu().numpy()


def _compute_erf_array(args, model, ratio, num_images):
    orig_batch_size = args.train_batch_size
    args.train_batch_size = _BATCH_SIZE
    train_loader, _, _ = get_dataloader(args)
    sample_loader = make_subset_loader(args, train_loader, ratio=ratio)
    args.train_batch_size = orig_batch_size  # restore so other analyses are unaffected

    total = min(num_images, len(sample_loader.dataset)) if num_images else len(sample_loader.dataset)
    print(f"[ERF] accumulating over up to {total} images")

    meter = AverageMeter()
    for samples, _ in tqdm(sample_loader, total=total, desc="Computing ERF"):
        if num_images and meter.count >= num_images:
            break
        samples = samples.to(args.device, non_blocking=True)
        samples.requires_grad = True
        grad_map = _get_input_grad(model, samples)
        if np.isnan(np.sum(grad_map)):
            print("[ERF] got NaN, skipping image")
            continue
        meter.update(grad_map)

    return meter.avg


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _make_erf_figure(erf_array, title="ERF"):
    data = np.log10(erf_array + 1)
    data = data / (np.max(data) + 1e-8)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(data, cmap="inferno", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Pipeline-compatible entry point
# ---------------------------------------------------------------------------

def analyze_erf(
    args,
    model,
    ratio: float = _RATIO,
    num_images: int = _NUM_IMAGES,
    **kwargs,
) -> list[AnalysisResult]:
    """
    Compute ERF and return both the raw array and a heatmap figure.

    Extra kwargs accepted
    --------------------
    ratio      : float  – fraction of train set to sample (default 0.05)
    num_images : int    – max images to accumulate (default 1000)
    """
    # Unwrap torch.compile if present
    raw_model = getattr(model, "_orig_mod", model)

    # Patch forward temporarily (restored in finally so other analyses are unaffected)
    original_forward = raw_model.forward
    raw_model.forward = types.MethodType(_erf_forward, raw_model)

    try:
        raw_model.eval()
        erf_array = _compute_erf_array(args, raw_model, ratio=ratio, num_images=num_images)
    finally:
        raw_model.forward = original_forward

    fig = _make_erf_figure(erf_array)

    return [
        AnalysisResult("erf", erf_array, ".npy"),
        AnalysisResult("erf", fig,       ".png"),
    ]
