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
    pre_parser.add_argument("--output_path", type=str, default=None)
    pre_parser.add_argument("--config_path", type=str, default=None)
    pre_parser.add_argument("--num_images", type=int, default = 100)
    pre_parser.add_argument("--ratio", type=float, default = 1)         # The ratio of len(sample dataset) when num_images is not small
    pre_parser.add_argument("--train_batch_size", type=int, default = 1)
    pre_args, remaining = pre_parser.parse_known_args()

    args = parse_args(remaining)

    if pre_args.config_path is not None:
        _apply_yaml(args, pre_args.config_path)

    args.output_path = pre_args.output_path
    args.num_images = pre_args.num_images
    args.ratio = pre_args.ratio
    args.train_batch_size = pre_args.train_batch_size
    return args

def make_erf(output_path=None):
    args = _get_args()


    model = build_model(args)
    model = getattr(model, "_orig_mod", model)  # unwrap torch.compile if present
    model.forward = types.MethodType(erf_forward, model)

    model.cuda()
    model.eval()

    train_loader, _, _ = get_dataloader(args)
    sample_loader = make_subset_loader(args, train_loader, ratio=args.ratio)
    max_images = args.num_images
    print(f"ERF: accumulating over up to {max_images} images (num_images={args.num_images}, ratio={args.ratio})")

    meter = AverageMeter()

    total = min(args.num_images, len(sample_loader.dataset))

    for samples, _ in tqdm(sample_loader, total=total, desc="Computing ERF"):
        if meter.count >= args.num_images:
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