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

ck = 0 # debugging

def get_input_grad_per_anchors(model, samples, anchors):
    """
    Given model, samples, anchors -> return dictionary of {anchor: grad map}
    """
    global ck
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
    
    if mode == "custom":
        all_points = [(y,x) for y,x in zip(custom_y_values, custom_x_values)]
        return all_points

def compute_long_range_metric(erf_map, anchor, distance_metric="taxi", patch_size=4):
    global ck
    """
    Given erf_map and distance metric type,
    return the average distance from each point
    """
    h, w = erf_map.shape
    py, px = anchor
    H, W = h//patch_size, w//patch_size
    
    token_weights = np.zeros((h//patch_size, w//patch_size))
    
    for i in range(h):
        for j in range(w):
            token_weights[i//patch_size, j//patch_size] += erf_map[i,j]

    total = token_weights.sum()
    token_weights = token_weights / total

    yy, xx = np.indices((H,W), dtype=np.float64)

    if distance_metric == "taxi":
        dist = np.abs(yy-py) + np.abs(xx-px)
    if distance_metric == "euclidean":
        dist = np.sqrt((yy-py)**2 + (xx-px)**2)

    if ck==0:
        print(f"H: {H}, W: {W}, py:{py}, px:{px}")
        ck+=1

    return float((token_weights*dist).sum())


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

def make_erf(output_path=None):
    args = _get_args()
    np.random.seed(args.seed)

    model = build_model(args)
    model = getattr(model, "_orig_mod", model)  # unwrap torch.compile if present
    model.forward = types.MethodType(erf_forward, model)

    model.cuda()
    model.eval()

    train_loader, _, _ = get_dataloader(args)
    sample_loader = make_subset_loader(args, train_loader, ratio=args.ratio)
    max_images = args.num_images
    print(f"ERF: accumulating over up to {max_images} images (num_images={args.num_images}, ratio={args.ratio})")

    total = min(args.num_images, len(sample_loader.dataset))

    first_batch = next(iter(sample_loader))
    first_samples = first_batch[0].cuda(non_blocking=True)
    first_samples.requires_grad_(True)

    with torch.enable_grad():
        test_out = model(first_samples)
    _, _, h_out, w_out = test_out.shape

    print(args.custom_x_values)

    anchors = choose_anchor_points(h_out, w_out, args.anchor_mode, args.num_anchors, args.custom_x_values, args.custom_y_values)
    
    print(f"Selected anchors: {anchors}")

    _, _, h_in, w_in = first_samples.shape

    accum = {anchor: np.zeros((h_in, w_in), dtype=np.float64) for anchor in anchors}
    count = 0

    for samples, _ in tqdm(sample_loader, total=total, desc="Computing ERF"):
        if count >= args.num_images:
            break
        samples = samples.cuda(non_blocking=True)
        samples.requires_grad = True
        anchor_to_maps = get_input_grad_per_anchors(model, samples, anchors)

        bsz = samples.shape[0]

        for anchor in anchors:
            single_map = anchor_to_maps[anchor]
            if np.isnan(np.sum(single_map)):
                print(f"got NAN, skip image")
            accum[anchor] += single_map
            
        count += bsz

    avg_maps = {}
    metrics = []

    for anchor in anchors:
        avg_map = accum[anchor] / count

        avg_maps[anchor] = avg_map

        metrics.append({
            "y": anchor[0],
            "x": anchor[1],
            "long_range_metric": compute_long_range_metric(avg_map, anchor, args.distance_metric),
        })

    overall_dis = 0
    for metric in metrics:
        overall_dis += metric["long_range_metric"]

    overall_dis = overall_dis / len(metrics)

    print(f"overall long range metric: {overall_dis} tokens")
    run_name = os.path.basename(os.path.normpath(args.output_path))
    save_dir = os.path.join("analysis_output", "erf")
    os.makedirs(save_dir, exist_ok=True)

    npz_dict = {}
    for anchor, avg_map in avg_maps.items():
        npz_dict[f"anchor_{anchor[0]}_{anchor[1]}"] = avg_map
    np.savez_compressed(os.path.join(save_dir, "anchor_to_maps.npz"), **npz_dict)

    with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return save_dir, avg_maps, metrics

def normalize_map(erf):
    erf = erf.astype(np.float64)
    erf = erf - erf.min()
    denom = erf.max() + 1e-8
    return erf / denom

def save_individual_plots(save_dir, avg_maps):
    for anchor, erf in avg_maps.items():
        erf_vis = normalize_map(erf)

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(erf_vis, cmap="inferno", norm=mcolors.PowerNorm(gamma=0.4))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"Anchor {anchor}")
        ax.axis("off")

        plot_path = os.path.join(save_dir, f"erf_anchor_{anchor[0]}_{anchor[1]}.png")
        fig.savefig(plot_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

def _make_weight_per_dis_fig(avg_maps, distance_metric="taxi", patch_size=4):
    """Build and return the weight-per-distance figure (does not save)."""
    fig, ax = plt.subplots(figsize=(7, 4))

    for anchor, erf_map in avg_maps.items():
        h, w = erf_map.shape
        H, W = h // patch_size, w // patch_size
        py, px = anchor

        token_weights = np.zeros((H, W))
        for i in range(h):
            for j in range(w):
                token_weights[i // patch_size, j // patch_size] += erf_map[i, j]
        total = token_weights.sum()
        if total > 0:
            token_weights /= total

        yy, xx = np.indices((H, W), dtype=np.float64)
        if distance_metric == "taxi":
            dist = np.abs(yy - py) + np.abs(xx - px)
        else:
            dist = np.sqrt((yy - py) ** 2 + (xx - px) ** 2)

        dist_flat, weight_flat = dist.flatten(), token_weights.flatten()
        unique_dists = np.unique(dist_flat)
        weight_per_dist = [weight_flat[dist_flat == d].sum() for d in unique_dists]
        ax.plot(unique_dists, weight_per_dist, marker="o", label=f"anchor {anchor}")

    ax.set_xlabel(f"Distance ({distance_metric})")
    ax.set_ylabel("Total weight")
    ax.set_title("Weight per distance from anchor")
    ax.legend(fontsize=7)
    return fig


def save_weight_per_dis_plot(save_dir, avg_maps, distance_metric="taxi", patch_size=4):
    fig = _make_weight_per_dis_fig(avg_maps, distance_metric, patch_size)
    plot_path = os.path.join(save_dir, "weight_per_distance.png")
    fig.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved weight-per-distance plot to {plot_path}")

def analyze_erf(args, model, num_images=100, ratio=1.0,
                anchor_mode="center", num_anchors=4,
                distance_metric="taxi", patch_size=4, **kwargs):
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

    anchors = choose_anchor_points(h_out, w_out, anchor_mode, num_anchors)
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

    results = [
        AnalysisResult(f"erf_anchor_{a[0]}_{a[1]}", m, ".npy")
        for a, m in avg_maps.items()
    ]
    results.append(AnalysisResult("metrics", json.dumps(metrics, indent=2), ".txt"))
    results.append(AnalysisResult("weight_per_distance", _make_weight_per_dis_fig(avg_maps, distance_metric, patch_size), ".png"))
    return results


if __name__ == '__main__':
    save_dir, avg_maps, metrics = make_erf()
    # save_individual_plots(save_dir, avg_maps)
    save_weight_per_dis_plot(save_dir, avg_maps)