"""
pretrained_val.py — CIFAR-100 pretrained baseline evaluation

Sources:
  - torch.hub (chenyaofo/pytorch-cifar-models): ResNet, VGG, MobileNet, etc.
      native 32×32, CIFAR-100 normalization
  - timm HuggingFace Hub: ViT variants fine-tuned on CIFAR-100
      224×224, ImageNet normalization

Usage:
  # All default models
  python pretrained_val.py

  # Specific models only
  python pretrained_val.py --models cifar100_resnet56 cifar100_vgg16_bn

  # Skip train-set evaluation (faster)
  python pretrained_val.py --no_train
"""

import argparse
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Normalization constants
# ---------------------------------------------------------------------------

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# Model registry
# Each entry: (display_name, source, hub_name, img_size, norm)
#   source  : 'hub'  → torch.hub chenyaofo/pytorch-cifar-models
#             'timm' → timm.create_model with hf_hub: prefix
#   norm    : 'cifar' | 'imagenet'
# ---------------------------------------------------------------------------

MODEL_REGISTRY = [
    # ── ResNet (chenyaofo, 32×32) ──────────────────────────────────────────
    ("resnet20",        "hub",  "cifar100_resnet20",   32,  "cifar"),
    ("resnet32",        "hub",  "cifar100_resnet32",   32,  "cifar"),
    ("resnet44",        "hub",  "cifar100_resnet44",   32,  "cifar"),
    ("resnet56",        "hub",  "cifar100_resnet56",   32,  "cifar"),
    # ── VGG-BN (chenyaofo, 32×32) ──────────────────────────────────────────
    ("vgg11_bn",        "hub",  "cifar100_vgg11_bn",   32,  "cifar"),
    ("vgg13_bn",        "hub",  "cifar100_vgg13_bn",   32,  "cifar"),
    ("vgg16_bn",        "hub",  "cifar100_vgg16_bn",   32,  "cifar"),
    ("vgg19_bn",        "hub",  "cifar100_vgg19_bn",   32,  "cifar"),
    # ── ViT (timm HF Hub, 224×224) ─────────────────────────────────────────
    ("vit_tiny_p16",    "timm", "hf_hub:edadaltocg/vit_tiny_patch16_224_cifar100",  224, "imagenet"),
    ("vit_small_p16",   "timm", "hf_hub:edadaltocg/vit_small_patch16_224_cifar100", 224, "imagenet"),
]

# Short alias → registry entry (for --models CLI filter)
_ALIAS = {entry[0]: entry for entry in MODEL_REGISTRY}
# Also allow raw hub names like "cifar100_resnet56"
_HUB_NAME = {entry[2]: entry for entry in MODEL_REGISTRY}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _make_loader(data_path, train, img_size, norm, batch_size, num_workers):
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if norm == "imagenet" else (CIFAR100_MEAN, CIFAR100_STD)
    t = []
    if img_size != 32:
        t.append(transforms.Resize((img_size, img_size)))
    t += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    dataset = datasets.CIFAR100(
        root=data_path, train=train, download=True,
        transform=transforms.Compose(t),
    )
    kw = dict(num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    if num_workers > 0:
        kw["prefetch_factor"] = 4
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, **kw)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_hub(hub_name):
    return torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        hub_name,
        pretrained=True,
        verbose=False,
        trust_repo=True,
    )


def _load_timm(hub_name):
    import timm
    return timm.create_model(hub_name, pretrained=True)


def load_model(source, hub_name):
    if source == "hub":
        return _load_hub(hub_name)
    elif source == "timm":
        return _load_timm(hub_name)
    else:
        raise ValueError(f"Unknown source: {source}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in tqdm(loader, leave=False, dynamic_ncols=True):
        x, y = x.to(device), y.to(device)
        out = model(x)
        # timm models return logits directly; some may return tuples
        if isinstance(out, (tuple, list)):
            out = out[0]
        pred = out.argmax(1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve_models(names):
    """Resolve CLI model names to registry entries."""
    resolved = []
    for name in names:
        if name in _ALIAS:
            resolved.append(_ALIAS[name])
        elif name in _HUB_NAME:
            resolved.append(_HUB_NAME[name])
        else:
            print(f"  [warn] unknown model '{name}', skipping")
    return resolved


def _print_table(rows, show_train):
    name_w = max(len(r[0]) for r in rows) + 2
    if show_train:
        header = f"{'Model':{name_w}s} | {'Train Acc':>10s} | {'Test Acc':>10s}"
    else:
        header = f"{'Model':{name_w}s} | {'Test Acc':>10s}"
    sep = "-" * len(header)
    print("\n" + "=" * len(header))
    print(header)
    print(sep)
    for name, train_acc, test_acc in rows:
        if test_acc is None:
            fail = f"{'ERROR':>10s}"
            print(f"{name:{name_w}s} | {fail}" + (f" | {fail}" if show_train else ""))
        elif show_train:
            tr = f"{train_acc:9.2f}%" if train_acc is not None else f"{'N/A':>10s}"
            print(f"{name:{name_w}s} | {tr} | {test_acc:9.2f}%")
        else:
            print(f"{name:{name_w}s} | {test_acc:9.2f}%")
    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser(description="Evaluate pretrained CIFAR-100 baselines")
    parser.add_argument("--data_path",    type=str, default="./data")
    parser.add_argument("--batch_size",   type=int, default=256)
    parser.add_argument("--num_workers",  type=int, default=4)
    parser.add_argument("--no_train",     action="store_true", help="Skip train-set evaluation")
    parser.add_argument(
        "--models", type=str, nargs="+", default=None,
        help=(
            "Models to evaluate (by short name or hub name). "
            "Defaults to all registry entries. "
            f"Available: {', '.join(e[0] for e in MODEL_REGISTRY)}"
        ),
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    entries = _resolve_models(args.models) if args.models else MODEL_REGISTRY

    results = []
    for display_name, source, hub_name, img_size, norm in entries:
        print(f"\n[{display_name}] loading from {source} ({hub_name}) ...")
        try:
            model = load_model(source, hub_name).to(device)
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append((display_name, None, None))
            continue

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  params: {n_params:,}  |  img_size: {img_size}  |  norm: {norm}")

        test_loader = _make_loader(args.data_path, train=False,
                                   img_size=img_size, norm=norm,
                                   batch_size=args.batch_size,
                                   num_workers=args.num_workers)
        test_acc = evaluate(model, test_loader, device)
        print(f"  test  acc: {test_acc:.2f}%")

        train_acc = None
        if not args.no_train:
            train_loader = _make_loader(args.data_path, train=True,
                                        img_size=img_size, norm=norm,
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers)
            train_acc = evaluate(model, train_loader, device)
            print(f"  train acc: {train_acc:.2f}%")

        results.append((display_name, train_acc, test_acc))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    _print_table(results, show_train=not args.no_train)


if __name__ == "__main__":
    main()
