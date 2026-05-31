#!/usr/bin/env python3
"""Select ImageNet-100 classes by ResNet-50 per-class F1 on the validation split."""

import argparse
import json
from pathlib import Path

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.dataset import (
    IMAGENET100_NUM_CLASSES,
    default_imagenet100_manifest_path,
    imagenet_root,
    prepare_imagenet,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_val_loader(data_path: str, batch_size: int, num_workers: int) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    valset = datasets.ImageFolder(imagenet_root(data_path) / "val", transform=transform)
    kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = 4
    return DataLoader(valset, **kwargs)


@torch.no_grad()
def compute_per_class_f1(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    num_classes: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    tp = torch.zeros(num_classes, dtype=torch.int64)
    fp = torch.zeros(num_classes, dtype=torch.int64)
    fn = torch.zeros(num_classes, dtype=torch.int64)

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        preds = model(images).argmax(dim=1)

        for cls in range(num_classes):
            cls_mask = targets == cls
            pred_mask = preds == cls
            tp[cls] += (pred_mask & cls_mask).sum().cpu()
            fp[cls] += (pred_mask & ~cls_mask).sum().cpu()
            fn[cls] += (~pred_mask & cls_mask).sum().cpu()

    eps = 1e-12
    precision = tp.float() / (tp + fp + eps)
    recall = tp.float() / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    support = tp + fn
    return f1, support


def save_imagenet100_manifest(
    path: str | Path,
    class_indices: list[int],
    class_names: list[str],
    per_class_f1: dict[int, float],
    *,
    selection_model: str,
    split: str = "val",
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "selection_model": selection_model,
        "selection_metric": "f1",
        "split": split,
        "num_classes": len(class_indices),
        "class_indices": class_indices,
        "class_names": class_names,
        "per_class_f1": {str(k): float(v) for k, v in per_class_f1.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def select_top_classes(f1: torch.Tensor, support: torch.Tensor, k: int) -> list[int]:
    ranked = sorted(
        range(len(f1)),
        key=lambda i: (-f1[i].item(), -support[i].item(), i),
    )
    return ranked[:k]


def main():
    parser = argparse.ArgumentParser(
        description="Select top ImageNet classes by ResNet-50 per-class F1"
    )
    parser.add_argument("--data_path", type=str, default="./data", help="Dataset root containing imagenet/")
    parser.add_argument("--model", type=str, default="resnet50", help="timm model name")
    parser.add_argument("--num_classes", type=int, default=IMAGENET100_NUM_CLASSES)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Manifest output path (default: {data_path}/imagenet100_resnet50_f1.json)",
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("[data] preparing ImageNet validation split (~6.3 GiB download if missing) ...")
    prepare_imagenet(args.data_path, download_train=False, download_val=True)

    print(f"[env] device={device}, model={args.model}")
    loader = build_val_loader(args.data_path, args.batch_size, args.num_workers)
    dataset = loader.dataset
    num_imagenet_classes = len(dataset.classes)
    print(f"[data] val samples={len(dataset)}, classes={num_imagenet_classes}")

    model = timm.create_model(args.model, pretrained=True, num_classes=num_imagenet_classes)
    model.to(device)
    model.eval()

    print("[eval] computing per-class F1 ...")
    f1, support = compute_per_class_f1(model, loader, device, num_classes=num_imagenet_classes)

    top_indices = select_top_classes(f1, support, args.num_classes)
    class_names = [dataset.classes[i] for i in top_indices]
    per_class_f1 = {idx: float(f1[idx]) for idx in top_indices}

    output_path = Path(args.output) if args.output else default_imagenet100_manifest_path(args.data_path)
    save_imagenet100_manifest(
        output_path,
        class_indices=top_indices,
        class_names=class_names,
        per_class_f1=per_class_f1,
        selection_model=args.model,
        split="val",
    )

    print(f"[result] saved manifest: {output_path}")
    print("[result] top-10 classes:")
    for rank, idx in enumerate(top_indices[:10], start=1):
        print(
            f"  {rank:02d}. idx={idx:4d}  name={dataset.classes[idx]:15s}  "
            f"f1={f1[idx].item():.4f}  support={support[idx].item()}"
        )


if __name__ == "__main__":
    main()
