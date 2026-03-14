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
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
# from erf.resnet_for_erf import resnet101, resnet152
# from erf.replknet_for_erf import RepLKNetForERF
from torch import optim as optim

from utils.config import parse_args
from utils.dataset import get_dataloader, make_subset_loader
from utils.build_model import build_model

RATIO = 0.05  # subset ratio of train set when NUM_IMAGES is None
BATCH_SIZE = 1
NUM_IMAGES = None  # if None: use full subset from ratio; else: stop after this many images

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

def main():
    args = parse_args()
    args.train_batch_size = BATCH_SIZE

    model = build_model(args)
    model = getattr(model, "_orig_mod", model)  # unwrap torch.compile if present
    model.forward = types.MethodType(erf_forward, model)

    model.cuda()
    model.eval()

    train_loader, test_loader, mixup_fn = get_dataloader(args)
    sample_loader = make_subset_loader(args, train_loader, ratio=RATIO)
    max_images = NUM_IMAGES if NUM_IMAGES is not None else len(sample_loader.dataset)
    print(f"ERF: accumulating over up to {max_images} images (NUM_IMAGES={NUM_IMAGES}, ratio={RATIO})")

    optimizer = optim.SGD(model.parameters(), lr=0, weight_decay=0)

    meter = AverageMeter()
    optimizer.zero_grad()

    for _, (samples, _) in enumerate(sample_loader):
        if NUM_IMAGES is not None and meter.count >= NUM_IMAGES:
            break
        samples = samples.cuda(non_blocking=True)
        samples.requires_grad = True
        optimizer.zero_grad()
        contribution_scores = get_input_grad(model, samples)

        if np.isnan(np.sum(contribution_scores)):
            print("got NAN, skip image")
            continue
        meter.update(contribution_scores)

    save_path = os.path.join(args.output_path, "erf.npy")
    os.makedirs(args.output_path, exist_ok=True)
    np.save(save_path, meter.avg)
    print(f"Saved ERF matrix to {save_path} (avg over {meter.count} images)")


if __name__ == '__main__':
    main()