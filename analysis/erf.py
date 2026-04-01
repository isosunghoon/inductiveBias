# A script to visualize the ERF.
# Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs (https://arxiv.org/abs/2203.06717)
# Github source: https://github.com/DingXiaoH/RepLKNet-pytorch
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import os
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

from utils.config import parse_args
from utils.dataset import get_dataloader, make_subset_loader
from utils.build_model import build_model

RATIO = 0.05  # subset ratio of train set when NUM_IMAGES is None
BATCH_SIZE = 1
NUM_IMAGES = 1000  # if None: use full subset from ratio; else: stop after this many images

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


def get_input_grad(model, samples):
    outputs = model(samples)
    out_size = outputs.size()
    central_point = torch.nn.functional.relu(outputs[:, :, out_size[2] // 2, out_size[3] // 2]).sum()
    grad = torch.autograd.grad(central_point, samples)
    grad = grad[0]
    grad = torch.nn.functional.relu(grad)
    aggregated = grad.sum((0, 1))
    grad_map = aggregated.cpu().numpy()
    return grad_map

def _get_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--save_path", type=str, default=None)
    pre_parser.add_argument("--output_path", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    if pre_args.save_path is not None:
        args = parse_args([
            "--base_config", os.path.join(pre_args.save_path, "base.yaml"),
            "--config", os.path.join(pre_args.save_path, "config.yaml"),
            "--output_path", pre_args.save_path,
        ])
    else:
        args = parse_args()
        if pre_args.output_path is not None:
            args.output_path = pre_args.output_path
    return args

def make_erf(output_path=None):
    if output_path is not None:
        # programmatic call: load config directly from the run folder
        args = parse_args([
            "--base_config", os.path.join(output_path, "base.yaml"),
            "--config", os.path.join(output_path, "config.yaml"),
            "--output_path", output_path,
        ])
    else:
        args = _get_args()

    args.train_batch_size = BATCH_SIZE

    model = build_model(args)
    model = getattr(model, "_orig_mod", model)  # unwrap torch.compile if present
    model.forward = types.MethodType(erf_forward, model)

    model.cuda()
    model.eval()

    train_loader, _, _ = get_dataloader(args)
    sample_loader = make_subset_loader(args, train_loader, ratio=RATIO)
    max_images = NUM_IMAGES if NUM_IMAGES is not None else len(sample_loader.dataset)
    print(f"ERF: accumulating over up to {max_images} images (NUM_IMAGES={NUM_IMAGES}, ratio={RATIO})")

    meter = AverageMeter()

    if NUM_IMAGES is not None:
        total = min(NUM_IMAGES, len(sample_loader.dataset))
    else:
        total = len(sample_loader.dataset)

    for samples, _ in tqdm(sample_loader, total=total, desc="Computing ERF"):
        if NUM_IMAGES is not None and meter.count >= NUM_IMAGES:
            break
        samples = samples.cuda(non_blocking=True)
        samples.requires_grad = True
        contribution_scores = get_input_grad(model, samples)

        if np.isnan(np.sum(contribution_scores)):
            print("got NAN, skip image")
            continue
        meter.update(contribution_scores)

    run_name = os.path.basename(os.path.normpath(args.output_path))
    save_dir = os.path.join("analysis", "erf_results")
    os.makedirs(save_dir, exist_ok=True)
    erf_npy_path = os.path.join(save_dir, f"{run_name}.npy")
    np.save(erf_npy_path, meter.avg)
    print(f"Saved ERF matrix to {erf_npy_path} (avg over {meter.count} images)")
    return erf_npy_path


def create_graph(npy_path):
    """Load an erf.npy file and save a heatmap plot next to it."""
    erf = np.load(npy_path)

    # Normalize to [0, 1] for visualization
    erf = erf - erf.min()
    erf = erf / (erf.max() + 1e-8)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(erf, cmap="inferno", norm=mcolors.PowerNorm(gamma=0.4))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(os.path.splitext(os.path.basename(npy_path))[0])
    ax.axis("off")

    plot_path = os.path.splitext(npy_path)[0] + ".png"
    fig.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved ERF plot to {plot_path}")
    return plot_path


if __name__ == '__main__':
    npy_path = make_erf()
    create_graph(npy_path)