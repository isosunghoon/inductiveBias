# A script to visualize the ERF.
# Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs (https://arxiv.org/abs/2203.06717)
# Github source: https://github.com/DingXiaoH/RepLKNet-pytorch
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import os
import json
import argparse
import numpy as np
import types
import torch
from timm.utils import AverageMeter
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# from erf.resnet_for_erf import resnet101, resnet152
# from erf.replknet_for_erf import RepLKNetForERF
from torch import optim as optim

from utils.config import parse_args, _apply_yaml
from utils.dataset import get_dataloader, make_subset_loader
from utils.build_model import build_model

# 새로운 forward 함수 정의 (ERF 측정용: GAP 제거 버전)
def erf_forward(self, x):
    """
    ERF 측정을 위해 GAP와 Head를 제거하고 
    최종 피처 맵(B, C, H, W)을 반환하는 함수
    """
    x = self.forward_embeddings(x)
    x = self.forward_tokens(x)
    x = self.norm(x)
    
    # 여기서 GAP(x.mean([-2, -1]))를 수행하지 않고 
    # 공간 정보(H, W)가 살아있는 텐서를 그대로 반환합니다.
    return x

def get_input_grad_per_anchors(model, samples, anchors):
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

def _erf_to_patch_map(erf_map, patch_size):
    """Sum pixel-level erf_map into patches and normalize so all patch weights sum to 1."""
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

def compute_long_range_metric(erf_map, anchor, distance_metric="taxi", patch_size=4):
    """
    Given erf_map and distance metric type,
    return the average distance from each point
    """
    py, px = anchor
    token_weights = _erf_to_patch_map(erf_map, patch_size)
    H, W = token_weights.shape

    yy, xx = np.indices((H, W), dtype=np.float64)
    if distance_metric == "taxi":
        dist = np.abs(yy - py) + np.abs(xx - px)
    if distance_metric == "euclidean":
        dist = np.sqrt((yy - py)**2 + (xx - px)**2)

    return float((token_weights * dist).sum())

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
    return args

def _compute_weight_per_dist(erf_map, anchor, distance_metric, patch_size):
    """Return (unique_dists, weight_per_dist) arrays for a single anchor.
    weight_per_dist is the sum of patch weights at each distance, so it sums to 1.
    """
    py, px = anchor
    token_weights = _erf_to_patch_map(erf_map, patch_size)
    H, W = token_weights.shape

    yy, xx = np.indices((H, W), dtype=np.float64)
    if distance_metric == "taxi":
        dist = np.abs(yy - py) + np.abs(xx - px)
    else:
        dist = np.sqrt((yy - py) ** 2 + (xx - px) ** 2)

    dist_flat, weight_flat = dist.flatten(), token_weights.flatten()
    unique_dists = np.unique(dist_flat)
    weight_per_dist = np.array([weight_flat[dist_flat == d].mean() for d in unique_dists])
    return unique_dists, weight_per_dist

def _compute_avg_weight_per_dist(avg_maps, distance_metric, patch_size):
    """Average weight-per-distance across all anchors. Returns (all_dists, avg_weights)."""
    dist_to_weights = {}
    for anchor, erf_map in avg_maps.items():
        unique_dists, weight_per_dist = _compute_weight_per_dist(erf_map, anchor, distance_metric, patch_size)
        for d, w in zip(unique_dists, weight_per_dist):
            dist_to_weights.setdefault(d, []).append(w)
    all_dists = np.array(sorted(dist_to_weights.keys()))
    avg_weights = np.array([np.mean(dist_to_weights[d]) for d in all_dists])
    return all_dists, avg_weights


def _make_weight_per_dis_fig(avg_maps, token_mixer_name:str="", average=True, distance_metric="taxi", patch_size=4):
    """
    Build and return the weight-per-distance figure (does not save).
    average=True  -> average weight-per-distance across all anchors, one line.
    average=False -> one line per anchor.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    if average:
        all_dists, avg_weights = _compute_avg_weight_per_dist(avg_maps, distance_metric, patch_size)
        ax.plot(all_dists, avg_weights, marker="o")
        ax.set_title(f"{token_mixer_name} | Average Weight per Distance")
    else:
        for anchor, erf_map in avg_maps.items():
            unique_dists, weight_per_dist = _compute_weight_per_dist(erf_map, anchor, distance_metric, patch_size)
            ax.plot(unique_dists, weight_per_dist, marker="o", label=f"anchor {anchor}")
        ax.legend(fontsize=7)
        ax.set_title(f"{token_mixer_name} | Weight per Distance From Anchor")

    ax.set_xlabel(f"Distance ({distance_metric})")
    ax.set_ylabel("Total weight")
    return fig

def make_individual_plots(avg_maps, token_mixer_name: str, patch_size=4):
    """
    Build one ERF heatmap figure per anchor at token resolution and return a list of AnalysisResult objects.
    Each patch value is the sum of pixels within that patch, normalized so all patches sum to 1.
    Intended for use when anchor_mode == "custom".
    """
    from analysis.pipeline import AnalysisResult

    results = []
    for anchor, erf in avg_maps.items():
        token_map = _erf_to_patch_map(erf, patch_size)

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(
            token_map,
            cmap="inferno",
            norm=mcolors.PowerNorm(gamma=0.5, vmin=0.0, vmax=0.06)
        )     
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{token_mixer_name} | anchor: {anchor}")
        ax.axis("off")

        results.append(AnalysisResult(f"erf_anchor_{anchor[0]}_{anchor[1]}", fig, ".png"))
        results.append(AnalysisResult(f"token_map_anchor_{anchor[0]}_{anchor[1]}", token_map, ".npy"))
    return results

def analyze_erf(args, model, num_images=100, ratio=1.0,
                anchor_mode="center", num_anchors=4,
                distance_metric="taxi", patch_size=4, average=True, 
                custom_x_values=[], custom_y_values=[], **kwargs):
    """Pipeline-compatible ERF analysis. Returns list[AnalysisResult]."""
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

    anchors = choose_anchor_points(h_out, w_out, anchor_mode, num_anchors, custom_x_values, custom_y_values)
    print(f"Selected anchors: {anchors}")

    accum = {anchor: np.zeros((h_in, w_in), dtype=np.float64) for anchor in anchors}
    count = 0
    for samples, _ in tqdm(sample_loader, total=total, desc="Computing ERF"):
        if count >= num_images:
            break
        samples = samples.cuda(non_blocking=True)
        samples.requires_grad = True
        anchor_to_maps = get_input_grad_per_anchors(model, samples, anchors)
        for anchor in anchors:
            single_map = anchor_to_maps[anchor]
            if np.isnan(np.sum(single_map)):
                print("got NAN, skip")
            accum[anchor] += single_map
        count += samples.shape[0]

    avg_maps, metrics = {}, []
    for anchor in anchors:
        avg_map = accum[anchor] / count
        avg_maps[anchor] = avg_map
        metrics.append({
            "y": anchor[0], "x": anchor[1],
            "long_range_metric": compute_long_range_metric(avg_map, anchor, distance_metric, patch_size),
        })

    overall_dis = sum(m["long_range_metric"] for m in metrics) / len(metrics)
    print(f"overall long range metric: {overall_dis} tokens")

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
        token_mixer_name = "MLPMixer"
    if token_mixer_name == "vit":
        token_mixer_name = "ViT"

    results.append(AnalysisResult("weight_per_distance"+("" if average else "_by_anchor"), _make_weight_per_dis_fig(avg_maps, token_mixer_name, average, distance_metric, patch_size), ".png")) 

    all_dists, avg_weights = _compute_avg_weight_per_dist(avg_maps, distance_metric, patch_size)
    results.append(AnalysisResult("weight_per_distance_data", np.array([all_dists, avg_weights]), ".npy"))
    print(f"weight_per_distance_data saved")

    if average:
        results.append(AnalysisResult(f"metrics_{overall_dis}", json.dumps(metrics, indent=2), ".txt"))
    
    if anchor_mode == "custom":
        results.extend(make_individual_plots(avg_maps, token_mixer_name, patch_size))

    return results

def make_combined_weight_per_dist_plot(npy_files: dict, distance_metric="taxi"):
    """
    Load weight-per-distance .npy files for multiple models and overlay them on one plot.
    Each .npy must have shape (2, n): row 0 = distances, row 1 = avg weights.
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
        ax.plot(all_dists, avg_weights, marker="o", label=model_name)
    ax.set_title("Avereaged Weight per Distance")
    ax.set_xlabel(f"Distance ({distance_metric})")
    ax.set_ylabel("Total weight")
    # ax.set_ylim(bottom=0)
    ax.legend()
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    token_mixers = {
        "Convformer":"convformer",
        "MLPMixer": "denseformer",
        "local ViT": "localvit", 
        "ViT": "vit"
        }
    NPY_FILES = {
        k: f"analysis_output/erf/final1/{w}_weight_per_distance_data.npy" for k,w in token_mixers.items()
    }
    DISTANCE_METRIC = "taxi"
    SAVE_PATH = "analysis_output/erf/final1/combined_weight_per_distance.png"

    fig = make_combined_weight_per_dist_plot(NPY_FILES, distance_metric=DISTANCE_METRIC)
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    fig.savefig(SAVE_PATH, bbox_inches="tight", dpi=220)
    plt.close(fig)
    print(f"Saved combined plot to {SAVE_PATH}")