import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import timm
from timm.models.vision_transformer import _load_weights as load_jax_npz_weights


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(img_size: int, use_imagenet_norm: bool):
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if use_imagenet_norm else (CIFAR100_MEAN, CIFAR100_STD)
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_loader(data_path: str, train: bool, img_size: int, batch_size: int, num_workers: int, use_imagenet_norm: bool):
    tfm = build_transforms(img_size, use_imagenet_norm)
    ds = datasets.CIFAR100(root=data_path, train=train, download=True, transform=tfm)
    kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = 4
    return DataLoader(ds, **kwargs)


def build_model(model_name: str, num_classes: int):
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    return model


def load_checkpoint(model: nn.Module, ckpt_path: str):
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".npz":
        # Google Research ViT(JAX) checkpoints
        load_jax_npz_weights(model, str(path))
        return

    if suffix in (".pth", ".pt", ".bin"):
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict) and "state_dict" in obj:
            state_dict = obj["state_dict"]
        else:
            state_dict = obj

        # torch.compile prefix cleanup
        prefix = "_orig_mod."
        if any(k.startswith(prefix) for k in state_dict.keys()):
            state_dict = {k.removeprefix(prefix): v for k, v in state_dict.items()}

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[load] missing keys: {len(missing)}")
        print(f"[load] unexpected keys: {len(unexpected)}")
        if missing:
            print(f"[load] missing (first 10): {missing[:10]}")
        if unexpected:
            print(f"[load] unexpected (first 10): {unexpected[:10]}")
        return

    raise ValueError(f"Unsupported checkpoint extension: {suffix}. Use .npz/.pth/.pt/.bin")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    correct, total = 0, 0
    for x, y in tqdm(loader, leave=False, dynamic_ncols=True):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(description="Evaluate ViT checkpoint on CIFAR-100")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.npz/.pth/.pt/.bin)")
    parser.add_argument("--model", type=str, default="vit_base_patch16_224", help="timm model name")
    parser.add_argument("--num_classes", type=int, default=100, help="Number of classes (CIFAR-100=100)")
    parser.add_argument("--img_size", type=int, default=224, help="Input resolution")
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_train", action="store_true", help="Also evaluate train split")
    parser.add_argument(
        "--norm",
        type=str,
        default="imagenet",
        choices=["imagenet", "cifar100"],
        help="Input normalization",
    )
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda:0, cpu")
    args = parser.parse_args()

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    use_imagenet_norm = args.norm == "imagenet"

    print(f"[env] device={device}")
    print(f"[env] model={args.model}, num_classes={args.num_classes}, img_size={args.img_size}")
    print(f"[env] norm={args.norm}")

    model = build_model(args.model, args.num_classes)
    load_checkpoint(model, args.checkpoint)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params:,}")

    test_loader = build_loader(
        data_path=args.data_path,
        train=False,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_imagenet_norm=use_imagenet_norm,
    )
    test_acc = evaluate(model, test_loader, device)
    print(f"[result] test acc : {test_acc:.2f}%")

    if args.eval_train:
        train_loader = build_loader(
            data_path=args.data_path,
            train=True,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_imagenet_norm=use_imagenet_norm,
        )
        train_acc = evaluate(model, train_loader, device)
        print(f"[result] train acc: {train_acc:.2f}%")


if __name__ == "__main__":
    main()