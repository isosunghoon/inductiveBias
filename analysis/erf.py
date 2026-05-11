# A script to visualize the ERF.
# Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs (https://arxiv.org/abs/2203.06717)
# Github source: https://github.com/DingXiaoH/RepLKNet-pytorch
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import os
import sys
import json
import argparse
import re
import numpy as np
import types
import torch
from timm.utils import AverageMeter
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch import optim as optim

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.config import parse_args, _apply_yaml, resolve_runtime_device
from utils.dataset import get_dataloader, make_subset_loader
from utils.build_model import build_model


# ============================================================================
# Module-level helpers
# ============================================================================

def erf_forward(self, x):
    """
    Remove Gap and return x with spatial information (H, W)
    """
    x = self.forward_embeddings(x)
    x = self.forward_tokens(x)
    x = self.norm(x)

    return x


def choose_anchor_points(h_out=8, w_out=8, mode="random", num_anchors=8, custom_x_values=None, custom_y_values=None):
    if mode == "center":
        return [(h_out//2, w_out//2)]

    if mode == "random":
        all_points = [(y,x) for y in range(h_out) for x in range(w_out)]
        num_anchors = min(num_anchors, len(all_points))
        idx = np.random.choice(len(all_points), size=num_anchors, replace=False)
        return [all_points[i] for i in idx]

    if mode == "all":
        return [(y,x) for y in range(h_out) for x in range(w_out)]

    if mode == "custom":
        return [(y,x) for y,x in zip(custom_y_values, custom_x_values)]


# ============================================================================
# Main analysis class
# ============================================================================

class ERFAnalysis:
    """ERF analysis driver.

    Holds shared config (distance_metric, patch_size). Static helpers do the
    distance/weight bookkeeping; instance methods drive a full per-image
    accumulation pass for one variant of ERF (currently: input → output).

    Module-level plot functions reuse the static helpers via ERFAnalysis._foo(...).
    """

    def __init__(self, distance_metric: str = "taxi", patch_size: int = 4):
        self.distance_metric = distance_metric
        self.patch_size = patch_size

    # ------------------------------------------------------------------
    # Shared static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _erf_to_patch_map(erf_map, patch_size):
        """
        Return normalized patch_map
        Sum pixel-level erf_map into patches and normalize so all patch weights sum to 1.
        Have to run this no matter patch_size is 1 or not - normalization
        """
        h, w = erf_map.shape
        H, W = h // patch_size, w // patch_size
        patch_map = np.zeros((H, W), dtype=np.float64)
        for i in range(h):
            for j in range(w):
                patch_map[i // patch_size, j // patch_size] += erf_map[i, j]
        total = patch_map.sum()
        if total > 0:
            patch_map /= total
        return patch_map

    @staticmethod
    def _compute_weight_per_dist(erf_map, anchor, distance_metric, patch_size):
        """
        Return (unique_dists, weight_per_dist) arrays for a single anchor.
        weight_per_dist is the sum of patch weights at each distance, so it sums to 1.
        """
        py, px = anchor
        token_weights = ERFAnalysis._erf_to_patch_map(erf_map, patch_size)
        H, W = token_weights.shape

        yy, xx = np.indices((H, W), dtype=np.float64)
        if distance_metric == "taxi":
            dist = np.abs(yy - py) + np.abs(xx - px)
        else:
            dist = np.sqrt((yy - py) ** 2 + (xx - px) ** 2)

        dist_flat, weight_flat = dist.flatten(), token_weights.flatten()
        unique_dists = np.unique(dist_flat)
        weight_per_dist = np.array([weight_flat[dist_flat == d].sum() for d in unique_dists])
        return unique_dists, weight_per_dist

    @staticmethod
    def _compute_erf_dist_std(all_dists, avg_weights):
        """
        Compute std of the distance r.v. under the ERF distribution.
        avg_weights is treated as a prob mass function over all_dists.
        Returns (mean_dist, std_dist).
        """
        w = avg_weights / avg_weights.sum() if avg_weights.sum() > 0 else avg_weights
        mean_d = (all_dists * w).sum()
        std_d = np.sqrt(np.maximum((all_dists ** 2 * w).sum() - mean_d ** 2, 0.0))
        return mean_d, std_d

    @staticmethod
    def _compute_per_anchor_dist_stats(avg_maps, distance_metric, patch_size):
        """
        For each anchor, compute (mean, std) of distance under that anchor's own ERF distribution.
        ERD is then defined as the average of per-anchor means: each anchor patch has a different
        geometric reach, so we treat ERF(Y_ij, ·) as a separate distribution per anchor and aggregate
        only after taking the per-anchor expectation/std.
        Returns (per_anchor_means, per_anchor_stds) of shape (n_anchors,).
        """
        means, stds = [], []
        for anchor, erf_map in avg_maps.items():
            unique_dists, weight_per_dist = ERFAnalysis._compute_weight_per_dist(
                erf_map, anchor, distance_metric, patch_size)
            m, s = ERFAnalysis._compute_erf_dist_std(unique_dists, weight_per_dist)
            means.append(m)
            stds.append(s)
        return np.array(means), np.array(stds)

    @staticmethod
    def _compute_sum_weight_per_dist(avg_maps, distance_metric, patch_size):
        """Accumulate weight-per-distance across all anchors. Returns (all_dists, avg_weights)."""
        dist_to_weights = {}
        for anchor, erf_map in avg_maps.items():
            unique_dists, weight_per_dist = ERFAnalysis._compute_weight_per_dist(
                erf_map, anchor, distance_metric, patch_size)
            for d, w in zip(unique_dists, weight_per_dist):
                dist_to_weights.setdefault(d, []).append(w)
        all_dists = np.array(sorted(dist_to_weights.keys()))
        sum_weights = np.array([np.sum(dist_to_weights[d]) for d in all_dists])
        sum_weights = sum_weights / sum_weights.sum()
        return all_dists, sum_weights

    @staticmethod
    def compute_long_range_metric(erf_map, anchor, distance_metric="taxi", patch_size=4):
        """
        Given erf_map and distance metric type,
        return the average distance from each point
        """
        py, px = anchor
        token_weights = ERFAnalysis._erf_to_patch_map(erf_map, patch_size)
        H, W = token_weights.shape

        yy, xx = np.indices((H, W), dtype=np.float64)
        if distance_metric == "taxi":
            dist = np.abs(yy - py) + np.abs(xx - px)
        if distance_metric == "euclidean":
            dist = np.sqrt((yy - py)**2 + (xx - px)**2)

        return float((token_weights * dist).sum())

    # ------------------------------------------------------------------
    # Variant 1: input → output ERF (gradient w.r.t. raw input image)
    # ------------------------------------------------------------------

    def _get_input_grad_per_anchors(self, model, samples, anchors):
        """
        Given model, samples, anchors -> return dictionary of {anchor: grad map}
        """
        outputs = model(samples)
        out_size = outputs.size()           # outputs.shape = ([1, 192, 8, 8]) for Cifar-100
        _, _, h_out, w_out = outputs.shape

        anchor_to_maps = {}

        for i, (y, x) in enumerate(anchors):
            target = torch.nn.functional.relu(outputs[:, :, y, x]).sum()
            grad = torch.autograd.grad(target, samples, retain_graph=True)
            grad = grad[0]
            grad = torch.nn.functional.relu(grad)
            aggregated = grad.sum((0,1))
            anchor_to_maps[(y,x)] = aggregated.detach().cpu().numpy()

        return anchor_to_maps

    def analyze_input(self, args, model, num_images=100, ratio=1.0,
                      anchor_mode="center", num_anchors=4,
                      average=True,
                      custom_x_values=[], custom_y_values=[], **kwargs):
        """Pipeline-compatible ERF analysis (variant 1: input → output). Returns list[AnalysisResult]."""
        from analysis.pipeline import AnalysisResult

        np.random.seed(args.seed)
        model = getattr(model, "_orig_mod", model)
        model.forward = types.MethodType(erf_forward, model)
        model.cuda().eval()

        args.num_images = num_images
        args.ratio = ratio
        args.train_batch_size = 1

        train_loader, _, _ = get_dataloader(args)
        sample_loader = make_subset_loader(args, train_loader, ratio=ratio)
        total = min(num_images, len(sample_loader.dataset))
        print(f"ERF: accumulating over up to {num_images} images")

        first_batch = next(iter(sample_loader))
        first_samples = first_batch[0].cuda(non_blocking=True)
        first_samples.requires_grad_(True)
        with torch.enable_grad():
            test_out = model(first_samples)
        _, _, h_out, w_out = test_out.shape
        _, _, h_in, w_in = first_samples.shape

        self.patch_size = h_in // h_out

        anchors = choose_anchor_points(h_out, w_out, anchor_mode, num_anchors,
                                       custom_x_values, custom_y_values)
        print(f"Selected anchors: {anchors}")

        accum = {anchor: np.zeros((h_in, w_in), dtype=np.float64) for anchor in anchors}
        per_image_dists = {anchor: [] for anchor in anchors}
        count = 0
        for samples, _ in tqdm(sample_loader, total=total, desc="Computing ERF"):
            if count >= num_images:
                break
            samples = samples.cuda(non_blocking=True)
            samples.requires_grad = True
            anchor_to_maps = self._get_input_grad_per_anchors(model, samples, anchors)
            for anchor in anchors:
                single_map = anchor_to_maps[anchor]
                if np.isnan(np.sum(single_map)):
                    print("got NAN, skip")
                    continue
                accum[anchor] += single_map
                per_image_dists[anchor].append(
                    self.compute_long_range_metric(single_map, anchor,
                                                   self.distance_metric, self.patch_size)
                )
            count += samples.shape[0]

        avg_maps, metrics = {}, []
        for anchor in anchors:
            avg_map = accum[anchor] / count
            avg_maps[anchor] = avg_map
            d_arr = np.array(per_image_dists[anchor])
            mean_d = float(d_arr.mean())
            se_d = float(d_arr.std() / np.sqrt(len(d_arr)))
            metrics.append({
                "y": anchor[0], "x": anchor[1],
                "long_range_metric": mean_d,
                "se": se_d,
            })

        per_anchor_means, per_anchor_stds = self._compute_per_anchor_dist_stats(
            avg_maps, self.distance_metric, self.patch_size)
        overall_dis = float(per_anchor_means.mean())
        overall_intrinsic_std = float(per_anchor_stds.mean())
        overall_se = np.mean([m["se"] for m in metrics])
        print(f"overall long range metric: {overall_dis:.4f} ± {overall_se:.4f} tokens (SE), σ={overall_intrinsic_std:.4f}")

        results = []
        '''
        # Skip this part cause we don't need to store every erf maps per anchor, in general.
        results.append(
            AnalysisResult(f"erf_anchor_{a[0]}_{a[1]}", m, ".npy")
            for a, m in avg_maps.items()

        '''
        token_mixer_name = args.model
        if token_mixer_name == "convformer":
            token_mixer_name = "Convformer"
        if token_mixer_name == "localvit":
            token_mixer_name = "local ViT"
        if token_mixer_name == "denseformer":
            token_mixer_name = "MLP mixer"
        if token_mixer_name == "vit":
            token_mixer_name = "ViT"

        results.append(AnalysisResult(
            "weight_per_distance" + ("" if average else "_by_anchor"),
            _make_weight_per_dis_fig(avg_maps, token_mixer_name, average,
                                     self.distance_metric, self.patch_size, se_d=overall_se),
            ".png"
        ))

        all_dists, avg_weights = self._compute_sum_weight_per_dist(
            avg_maps, self.distance_metric, self.patch_size)
        results.append(AnalysisResult("weight_per_distance_data", np.array([
            all_dists,
            avg_weights,
            np.full_like(all_dists, overall_se, dtype=np.float64),
            np.full_like(all_dists, overall_dis, dtype=np.float64),
            np.full_like(all_dists, overall_intrinsic_std, dtype=np.float64),
        ]), ".npy"))
        print(f"weight_per_distance_data saved")

        if average:
            results.append(AnalysisResult(f"metrics_{overall_dis}", json.dumps(metrics, indent=2), ".txt"))

        if anchor_mode == "custom":
            results.extend(make_erf_heatmap(avg_maps, token_mixer_name, self.patch_size))

        return results

    # ------------------------------------------------------------------
    # Variant 2: layer-to-layer ERF (gradient of layer b w.r.t. layer a, a < b)
    # ------------------------------------------------------------------

    def _get_all_layer_grads_per_anchors(self, model, samples, layers, anchors):
        """
        Single forward + per-anchor backward. Returns {anchor: {(a, b): np.ndarray}}.

        activations[0] is the raw input image (samples); activations[i+1] is the
        output of layers[i]. So a=0 yields a pixel-grid grad of shape (h_in, w_in)
        and a>=1 yields a patch-grid grad of shape (H, W).

        One forward populates all hooks; then for each anchor and each target b we
        run torch.autograd.grad against inputs=[X_0, ..., X_{b-1}]. retain_graph=True
        is required because the same graph is traversed multiple times (once per
        anchor × b).
        """
        activations = {}
        hooks = []

        def make_hook(idx):
            def hook(module, inp, output):
                activations[idx] = output
            return hook

        try:
            for i, layer in enumerate(layers):
                hooks.append(layer.register_forward_hook(make_hook(i + 1)))

            activations[0] = samples
            with torch.enable_grad():
                _ = model(samples)

            L = len(layers) + 1
            anchor_to_pairs = {}

            for (y, x) in anchors:
                pair_to_grad = {}
                for b in range(1, L):
                    X_b = activations[b]
                    target = torch.nn.functional.relu(X_b[:, :, y, x]).sum()
                    inputs = [activations[a] for a in range(b)]
                    grads = torch.autograd.grad(target, inputs=inputs, retain_graph=True)
                    for a, g in enumerate(grads):
                        aggregated = torch.nn.functional.relu(g).sum(dim=(0, 1))
                        pair_to_grad[(a, b)] = aggregated.detach().cpu().numpy()
                anchor_to_pairs[(y, x)] = pair_to_grad

            return anchor_to_pairs
        finally:
            for h in hooks:
                h.remove()

    def analyze_layers(self, args, model, num_images=50,
                       anchor_mode="center", num_anchors=4,
                       custom_x_values=[], custom_y_values=[],
                       batch_size=16, **kwargs):
        """Pipeline-compatible block-wise ERF analysis (variant 2).

        For each layer pair (a, b) with a < b, computes the ERD of layer b's output
        w.r.t. layer a's activation. Returns an (L, L) symmetric ERD matrix, where
        L = 3 + n_blocks (index 0 = raw input image, 1 = patch_embed,
        2..L-2 = blocks, L-1 = final norm).
        """
        from analysis.pipeline import AnalysisResult

        np.random.seed(args.seed)
        model = getattr(model, "_orig_mod", model)
        model.forward = types.MethodType(erf_forward, model)
        model.cuda().eval()

        layers = [model.patch_embed] + list(model.blocks) + [model.norm]
        L = len(layers) + 1  # +1 for raw input image at index 0

        args.num_images = num_images
        args.train_batch_size = batch_size

        train_loader, _, _ = get_dataloader(args)
        sample_loader = make_subset_loader(args, train_loader, ratio=1.0)
        total = min(num_images, len(sample_loader.dataset))
        n_batches = (total + batch_size - 1) // batch_size
        print(f"ERF (block-wise): L={L} layers, batch_size={batch_size}, accumulating over up to {num_images} images")

        # Probe forward to determine block-resolution spatial grid (H, W) and
        # input-pixel grid (h_in, w_in). model_patch_size = h_in // H is the
        # actual patch size (used to aggregate the a=0 pixel grad to the patch
        # grid so its ERD is in the same units as a>=1 pairs).
        first_batch = next(iter(sample_loader))
        first_samples = first_batch[0].cuda(non_blocking=True)
        first_samples.requires_grad_(True)
        with torch.enable_grad():
            probe_out = model(first_samples)
        _, _, H, W = probe_out.shape
        _, _, h_in, w_in = first_samples.shape
        model_patch_size = h_in // H

        anchors = choose_anchor_points(H, W, anchor_mode, num_anchors,
                                       custom_x_values, custom_y_values)
        print(f"Selected anchors: {anchors}")

        # a=0 grads are on the pixel grid (h_in, w_in); a>=1 are on the patch
        # grid (H, W). Allocate accumulators with the right shape per a.
        accum = {}
        for a in range(L):
            for b in range(a + 1, L):
                for anchor in anchors:
                    if a == 0:
                        accum[(a, b, anchor)] = np.zeros((h_in, w_in), dtype=np.float64)
                    else:
                        accum[(a, b, anchor)] = np.zeros((H, W), dtype=np.float64)

        count = 0
        for samples, _ in tqdm(sample_loader, total=n_batches, desc="Computing layer ERF"):
            if count >= num_images:
                break
            samples = samples.cuda(non_blocking=True)
            samples.requires_grad = True

            per_anchor_pairs = self._get_all_layer_grads_per_anchors(model, samples, layers, anchors)

            skip = any(
                np.isnan(np.sum(grad_map))
                for pair_to_grad in per_anchor_pairs.values()
                for grad_map in pair_to_grad.values()
            )
            if skip:
                print("got NAN, skip")
                continue

            for anchor, pair_to_grad in per_anchor_pairs.items():
                for (a, b), grad_map in pair_to_grad.items():
                    accum[(a, b, anchor)] += grad_map
            count += samples.shape[0]

        erd_matrix = np.zeros((L, L), dtype=np.float64)
        for a in range(L):
            for b in range(a + 1, L):
                # a=0: aggregate pixel grad to patch grid via _erf_to_patch_map.
                # a>=1: grad is already on the patch grid, so patch_size=1.
                ps = model_patch_size if a == 0 else 1
                anchor_erds = []
                for anchor in anchors:
                    avg_map = accum[(a, b, anchor)] / count
                    erd = self.compute_long_range_metric(
                        avg_map, anchor, self.distance_metric, patch_size=ps)
                    anchor_erds.append(erd)
                mean_erd = float(np.mean(anchor_erds))
                erd_matrix[a, b] = mean_erd
                erd_matrix[b, a] = mean_erd

        token_mixer_name = args.model
        if token_mixer_name == "convformer":
            token_mixer_name = "Convformer"
        if token_mixer_name == "localvit":
            token_mixer_name = "local ViT"
        if token_mixer_name == "denseformer":
            token_mixer_name = "MLP mixer"
        if token_mixer_name == "vit":
            token_mixer_name = "ViT"

        fig = _make_layer_erd_heatmap(erd_matrix, token_mixer_name)

        return [
            AnalysisResult("erf_layers", erd_matrix, ".npy"),
            AnalysisResult("erf_layers", fig, ".png"),
        ]

# ============================================================================
# Module-level plot functions (stateless — call ERFAnalysis static helpers)
# ============================================================================

def _layer_labels(n):
    """['input', 'patch_embed', 'block_0', ..., 'final_norm'] for the raw-image-included matrix.

    Note: deliberately diverges from cka._layer_labels — CKA does not treat the
    raw image as a layer, so its label list has no 'input' entry.
    """
    if n <= 2:
        return ["input", "patch_embed"][:n]
    return ["input", "patch_embed"] + [f"block_{i}" for i in range(n - 3)] + ["final_norm"]


def _make_layer_erd_heatmap(erd_matrix: np.ndarray, model_name: str) -> plt.Figure:
    """Heatmap of the (L, L) block-wise ERD matrix."""
    L = erd_matrix.shape[0]
    fig_w = max(6, L * 0.7)
    fig_h = max(5, L * 0.7)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmax = float(erd_matrix.max()) if erd_matrix.max() > 0 else 1.0
    im = ax.imshow(erd_matrix, vmin=0.0, vmax=vmax, cmap="viridis", aspect="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("ERD", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    labels = _layer_labels(L)
    ax.set_xticks(range(L))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=12)
    ax.set_yticks(range(L))
    ax.set_yticklabels(labels, fontsize=12)

    ax.set_xlabel("anchor layer (b)", fontsize=14)
    ax.set_ylabel("input layer (a)", fontsize=14)
    ax.set_title(f"{model_name} | Block-wise ERD", fontsize=18)

    threshold = vmax * 0.6
    for i in range(L):
        for j in range(L):
            ax.text(j, i, f"{erd_matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=6,
                    color="white" if erd_matrix[i, j] < threshold else "black")

    plt.tight_layout()
    return fig


def _make_weight_per_dis_fig(avg_maps, token_mixer_name:str="", average=True, distance_metric="taxi", patch_size=4, se_d=None):
    """
    Build and return the weight-per-distance figure (does not save).
    average=True  -> average weight-per-distance across all anchors, one line.
    average=False -> one line per anchor.
    se_d: standard error of the mean distance metric across images (σ_images / √n).
          If provided, the shaded band shows mean_d ± se_d. Falls back to intrinsic ERF std if None.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    if average:
        all_dists, avg_weights = ERFAnalysis._compute_sum_weight_per_dist(
            avg_maps, distance_metric, patch_size)
        per_anchor_means, per_anchor_stds = ERFAnalysis._compute_per_anchor_dist_stats(
            avg_maps, distance_metric, patch_size)
        mean_d = float(per_anchor_means.mean())
        intrinsic_std = float(per_anchor_stds.mean())
        band = se_d if se_d is not None else intrinsic_std
        label = f"μ={mean_d:.2f}, SE={band:.4f}" if se_d is not None else f"μ={mean_d:.2f}, σ={band:.2f}"
        line, = ax.plot(all_dists, avg_weights, marker="o")
        ax.axvline(mean_d, color=line.get_color(), linestyle="--", alpha=0.7, label=label)
        ax.axvspan(mean_d - band, mean_d + band, alpha=0.12, color=line.get_color())
        ax.legend(fontsize=10)
        # ax.set_title(f"{token_mixer_name} | Average Weight per Distance")
    else:
        for anchor, erf_map in avg_maps.items():
            unique_dists, weight_per_dist = ERFAnalysis._compute_weight_per_dist(
                erf_map, anchor, distance_metric, patch_size)
            ax.plot(unique_dists, weight_per_dist, marker="o", label=f"anchor {anchor}")
        ax.legend(fontsize=10)
        # ax.set_title(f"{token_mixer_name} | Weight per Distance From Anchor")

    ax.set_xlabel(f"Distance ({distance_metric})", fontsize=13)
    ax.set_ylabel("Average weight", fontsize=13)
    ax.tick_params(labelsize=11)
    return fig

def make_erf_heatmap(avg_maps, token_mixer_name: str, patch_size: int =4):
    """
    Build one ERF heatmap figure per anchor at token resolution and return a list of AnalysisResult objects.
    Each patch value is the sum of pixels within that patch, normalized so all patches sum to 1.
    Intended for use when anchor_mode == "custom".
    """
    from analysis.pipeline import AnalysisResult

    results = []
    for anchor, erf in avg_maps.items():
        token_map = ERFAnalysis._erf_to_patch_map(erf, patch_size)

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(
            token_map,
            cmap="inferno",
            norm=mcolors.PowerNorm(gamma=0.5, vmin=0.0, vmax=0.05)
        )
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=14)

        H, W = token_map.shape
        ax.set_xticks(range(W))
        ax.set_yticks(range(H))
        ax.tick_params(labelsize=14)
        ax.set_title(f"{token_mixer_name}", fontsize=20)
        plt.tight_layout()

        results.append(AnalysisResult(f"erf_anchor_{anchor[0]}_{anchor[1]}", fig, ".png"))
        results.append(AnalysisResult(f"token_map_anchor_{anchor[0]}_{anchor[1]}", token_map, ".npy"))
    return results

def make_combined_weight_per_dist_plot(npy_files: dict, distance_metric="taxi"):
    """
    Load weight-per-distance .npy files for multiple models and overlay them on one plot.
    Each .npy must have shape (2+, n): row 0 = distances, row 1 = avg weights.
    Optional rows:
      row 2 = SE repeated across distances
      row 3 = ERD/mean distance repeated across distances
      row 4 = intrinsic std repeated across distances
    Matches the format of _make_weight_per_dis_fig (average=True).

    Parameters
    ----------
    npy_files : dict[str, str]
        Mapping of display name -> path to weight_per_distance_data.npy
    distance_metric : str
        Label used for the x-axis (e.g. "taxi", "euclidean")
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for model_name, path in npy_files.items():
        data = np.load(path)
        all_dists, avg_weights = data[0], data[1]
        if data.shape[0] >= 5:
            se_d = float(data[2][0])
            mean_d = float(data[3][0])
            intrinsic_std = float(data[4][0])
        elif data.shape[0] >= 3:
            se_d = float(data[2][0])
            mean_d, intrinsic_std = ERFAnalysis._compute_erf_dist_std(all_dists, avg_weights)
        else:
            se_d = None
            mean_d, intrinsic_std = ERFAnalysis._compute_erf_dist_std(all_dists, avg_weights)
        label = f"{model_name} (μ={mean_d:.2f}, SE={se_d:.3f})" if se_d is not None else f"{model_name} (μ={mean_d:.2f}, σ={intrinsic_std:.2f})"
        band = se_d if se_d is not None else intrinsic_std
        line, = ax.plot(all_dists, avg_weights, marker="o", label=label)
        ax.axvline(mean_d, color=line.get_color(), linestyle="--", alpha=0.7)
        ax.axvspan(mean_d - band, mean_d + band, alpha=0.1, color=line.get_color())
    ax.set_title("Total Weight per Distance and ERD by Token Mixer", fontsize=14)
    ax.set_xlabel(f"Distance ({distance_metric})", fontsize=13)
    ax.set_ylabel("Total weight (Normalized)", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    return fig


def _read_weight_per_dist_summary(path: str):
    """Return distance curve and ERD summary fields from a weight_per_distance_data.npy."""
    data = np.load(path)
    if data.ndim != 2 or data.shape[0] < 2:
        raise ValueError(f"{path} must have shape (2+, n), got {data.shape}")

    all_dists, avg_weights = data[0], data[1]
    if data.shape[0] >= 5:
        se_d = float(data[2][0])
        mean_d = float(data[3][0])
        intrinsic_std = float(data[4][0])
    elif data.shape[0] >= 3:
        se_d = float(data[2][0])
        mean_d, intrinsic_std = ERFAnalysis._compute_erf_dist_std(all_dists, avg_weights)
    else:
        se_d = None
        mean_d, intrinsic_std = ERFAnalysis._compute_erf_dist_std(all_dists, avg_weights)

    return all_dists, avg_weights, mean_d, se_d, intrinsic_std


def _display_name_from_weight_file(path: str) -> str:
    """Convert timestamped output filenames to stable labels."""
    stem = os.path.basename(path).replace("_weight_per_distance_data.npy", "")
    stem = re.sub(r"-\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", "", stem)
    labels = {
        "vit": "ViT",
        "localvit": "local ViT",
        "localvit_w5": "local ViT w5",
        "localvit_w7": "local ViT w7",
        "denseformer": "MLP mixer",
        "convformer": "Convformer",
        "convformer_w5": "Convformer w5",
        "convformer_w7": "Convformer w7",
        #"identity": "Identity",
    }
    return labels.get(stem, stem)


def discover_weight_per_distance_files(npy_dir: str, exclude=None) -> dict:
    """
    Find *_weight_per_distance_data.npy files in npy_dir.
    Returns an ordered mapping of display name -> path.
    """
    exclude = [item.lower() for item in (exclude or [])]
    files = sorted(
        os.path.join(npy_dir, name)
        for name in os.listdir(npy_dir)
        if name.endswith("_weight_per_distance_data.npy")
    )
    if not files:
        raise FileNotFoundError(f"No *_weight_per_distance_data.npy files found in {npy_dir}")

    preferred_order = [
        # "Identity",
        "local ViT",
        "local ViT w5",
        "local ViT w7",
        "Convformer",
        "Convformer w5",
        "Convformer w7",
        "MLP mixer",
        "ViT",
    ]
    name_to_path = {}
    for path in files:
        display_name = _display_name_from_weight_file(path)
        searchable = f"{display_name} {os.path.basename(path)}".lower()
        if any(pattern in searchable for pattern in exclude):
            continue
        name_to_path[display_name] = path
    if not name_to_path:
        raise FileNotFoundError(
            f"No weight_per_distance_data.npy files left in {npy_dir} after excludes: {exclude}"
        )
    ordered = {
        name: name_to_path[name]
        for name in preferred_order
        if name in name_to_path
    }
    ordered.update({name: path for name, path in name_to_path.items() if name not in ordered})
    return ordered


def make_erd_summary_plot(npy_files: dict):
    """Build a bar plot of ERD/mean distance for several weight_per_distance_data.npy files."""
    names, means, errors = [], [], []
    for model_name, path in npy_files.items():
        _, _, mean_d, se_d, intrinsic_std = _read_weight_per_dist_summary(path)
        names.append(model_name)
        means.append(mean_d)
        errors.append(se_d if se_d is not None else intrinsic_std)

    fig_w = max(7, len(names) * 0.9)
    fig, ax = plt.subplots(figsize=(fig_w, 4))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=errors, capsize=4, color="#4C78A8", alpha=0.9)
    ax.set_ylabel("ERD (mean distance)", fontsize=13)
    ax.set_title("ERD by Token Mixer", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=10)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    return fig


def save_erd_summary(npy_files: dict, save_path: str):
    """Write a small CSV-style ERD summary next to the generated plots."""
    rows = ["model,erd,se,intrinsic_std,path"]
    for model_name, path in npy_files.items():
        _, _, mean_d, se_d, intrinsic_std = _read_weight_per_dist_summary(path)
        se_text = "" if se_d is None else f"{se_d:.10f}"
        rows.append(
            f"{model_name},{mean_d:.10f},{se_text},{intrinsic_std:.10f},{path}"
        )
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")


# ============================================================================
# Pipeline-compatible wrapper
# ============================================================================

def analyze_erf(args, model, distance_metric: str = "taxi", patch_size: int = 4, **kwargs):
    """Pipeline entry point. Wraps ERFAnalysis.analyze_input so run_analysis.py keeps working."""
    return ERFAnalysis(distance_metric=distance_metric, patch_size=patch_size).analyze_input(
        args, model, **kwargs
    )


def analyze_erf_layers(args, model, distance_metric: str = "taxi", **kwargs):
    """Pipeline entry point for block-wise ERF (variant 2). patch_size=1 because
    block outputs are already at patch resolution — _erf_to_patch_map is a no-op."""
    return ERFAnalysis(distance_metric=distance_metric, patch_size=1).analyze_layers(
        args, model, **kwargs
    )


# ============================================================================
# Standalone arg parsing / main
# ============================================================================

def _get_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--output_path", type=str, default=None)
    pre_parser.add_argument("--config_path", type=str, default=None)
    pre_parser.add_argument("--num_images", type=int, default = 100)
    pre_parser.add_argument("--ratio", type=float, default = 1)         # The ratio of len(sample dataset) when num_images is not small
    pre_parser.add_argument("--train_batch_size", type=int, default = 1)
    pre_parser.add_argument("--anchor_mode", type=str, default="center")
    pre_parser.add_argument("--num_anchors", type=int, default=4)
    pre_parser.add_argument("--distance_metric", type=str, default="taxi")
    pre_parser.add_argument("--custom_x_values", type=int, nargs="+", default=None)
    pre_parser.add_argument("--custom_y_values", type=int, nargs="+", default=None)
    pre_args, remaining = pre_parser.parse_known_args()

    args = parse_args(remaining)

    if pre_args.config_path is not None:
        _apply_yaml(args, pre_args.config_path)

    args.output_path = pre_args.output_path
    args.num_images = pre_args.num_images
    args.ratio = pre_args.ratio
    args.train_batch_size = pre_args.train_batch_size
    args.anchor_mode = pre_args.anchor_mode
    args.num_anchors = pre_args.num_anchors
    args.distance_metric = pre_args.distance_metric
    args.custom_x_values = pre_args.custom_x_values
    args.custom_y_values = pre_args.custom_y_values

    resolve_runtime_device(args)
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot saved ERF weight-per-distance .npy files.")
    parser.add_argument(
        "--npy_dir",
        type=str,
        default="analysis_output/erf/model_resized",
        help="Directory containing *_weight_per_distance_data.npy files.",
    )
    parser.add_argument("--distance_metric", type=str, default="taxi")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Case-insensitive substrings to exclude from display names or filenames.",
    )
    args = parser.parse_args()

    npy_files = discover_weight_per_distance_files(args.npy_dir, exclude=args.exclude)
    weight_save_path = os.path.join(args.npy_dir, "combined_weight_per_distance.png")
    erd_save_path = os.path.join(args.npy_dir, "combined_erd.png")
    summary_save_path = os.path.join(args.npy_dir, "combined_erd_summary.csv")

    fig = make_combined_weight_per_dist_plot(npy_files, distance_metric=args.distance_metric)
    fig.savefig(weight_save_path, bbox_inches="tight", dpi=args.dpi)
    plt.close(fig)

    fig = make_erd_summary_plot(npy_files)
    fig.savefig(erd_save_path, bbox_inches="tight", dpi=args.dpi)
    plt.close(fig)

    save_erd_summary(npy_files, summary_save_path)

    print(f"Loaded {len(npy_files)} npy files from {args.npy_dir}")
    print(f"Saved combined weight-per-distance plot to {weight_save_path}")
    print(f"Saved ERD plot to {erd_save_path}")
    print(f"Saved ERD summary to {summary_save_path}")
