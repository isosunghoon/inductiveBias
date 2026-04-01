"""
pretrained_val.py — CIFAR-100 pretrained baseline evaluation (edadaltocg)

All models from edadaltocg, consistent training (300 epochs, SGD, CosineAnnealingLR):
  - resnet18/34/50   : ~79-81% test acc  (torchvision arch + CIFAR stem)
  - vgg16_bn         : timm features + single FC head
  - densenet121      : torchvision DenseNet, k=12, C0=24, CIFAR stem
  - vit_base_p16_224 : ViT-Base/16, ImageNet-21k → CIFAR-100 ft, ~93%

Usage:
  python pretrained_val.py                      # all models
  python pretrained_val.py --models resnet50 vit_base
  python pretrained_val.py --no_train           # test set only
"""

import argparse
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader
from tqdm import tqdm

import timm
import torchvision.models as tv_models
from torchvision.models.densenet import DenseNet as TvDenseNet
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

VIT_BASE_URL = (
    "https://huggingface.co/edadaltocg/vit_base_patch16_224_in21k_ft_cifar100"
    "/resolve/main/pytorch_model.bin"
)

# ---------------------------------------------------------------------------
# Model registry  (display_name, hf_repo, loader, img_size, mean, std)
# ---------------------------------------------------------------------------

MODEL_REGISTRY = [
    ("resnet18",    "edadaltocg/resnet18_cifar100",    "tv_resnet18",   32,  CIFAR100_MEAN, CIFAR100_STD),
    ("resnet34",    "edadaltocg/resnet34_cifar100",    "tv_resnet34",   32,  CIFAR100_MEAN, CIFAR100_STD),
    ("resnet50",    "edadaltocg/resnet50_cifar100",    "tv_resnet50",   32,  CIFAR100_MEAN, CIFAR100_STD),
    ("vgg16_bn",    "edadaltocg/vgg16_bn_cifar100",    "cifar_vgg16bn", 32,  CIFAR100_MEAN, CIFAR100_STD),
    ("densenet121", "edadaltocg/densenet121_cifar100", "cifar_dn121",   32,  CIFAR100_MEAN, CIFAR100_STD),
    ("vit_base",    None,                              "url_vit",       224, IMAGENET_MEAN, IMAGENET_STD),
]

_ALIAS = {e[0]: e for e in MODEL_REGISTRY}


# ---------------------------------------------------------------------------
# Architecture helpers
# ---------------------------------------------------------------------------

def _patch_resnet_cifar(model):
    """Replace 7×7/2 stem + maxpool with 3×3/1 + Identity for 32×32 input."""
    in_c  = model.conv1.in_channels
    out_c = model.conv1.out_channels
    model.conv1  = nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


class _VGGCIFARHead(nn.Module):
    """Global-avg-pool → single Linear, matching edadaltocg's head.fc keys."""
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean([-2, -1])   # global avg pool
        return self.fc(x)


class _VGGCIFARModel(nn.Module):
    """timm VGG16-BN features + lightweight head (no pre_logits FC blocks)."""
    def __init__(self, num_classes: int = 100):
        super().__init__()
        base = timm.create_model("vgg16_bn", pretrained=False, num_classes=0)
        self.features = base.features          # preserves features.* key names
        self.head = _VGGCIFARHead(512, num_classes)  # matches head.fc.* keys

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def _make_cifar_densenet121(num_classes: int = 100) -> nn.Module:
    """
    DenseNet-121 with CIFAR settings used by edadaltocg:
      growth_rate=12, num_init_features=24, block_config=(6,12,24,16)
      stem: Conv2d(3, 24, 3, stride=1, padding=1)  — no max-pool
    Uses torchvision DenseNet so classifier.* keys match.
    """
    model = TvDenseNet(
        growth_rate=12,
        block_config=(6, 12, 24, 16),
        num_init_features=24,
        num_classes=num_classes,
    )
    # Replace 7×7/2 stem with 3×3/1
    model.features.conv0 = nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, bias=False)
    # Remove initial max-pool
    model.features.pool0 = nn.Identity()
    return model


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_from_hf(model: nn.Module, hf_repo: str) -> nn.Module:
    path = hf_hub_download(hf_repo, "pytorch_model.bin")
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model


def load_model(_display_name: str, hf_repo, loader: str) -> nn.Module:
    if loader == "tv_resnet18":
        model = _patch_resnet_cifar(tv_models.resnet18(num_classes=100))
        return _load_from_hf(model, hf_repo)

    if loader == "tv_resnet34":
        model = _patch_resnet_cifar(tv_models.resnet34(num_classes=100))
        return _load_from_hf(model, hf_repo)

    if loader == "tv_resnet50":
        model = _patch_resnet_cifar(tv_models.resnet50(num_classes=100))
        return _load_from_hf(model, hf_repo)

    if loader == "cifar_vgg16bn":
        model = _VGGCIFARModel(num_classes=100)
        return _load_from_hf(model, hf_repo)

    if loader == "cifar_dn121":
        model = _make_cifar_densenet121(num_classes=100)
        return _load_from_hf(model, hf_repo)

    if loader == "url_vit":
        base = timm.create_model("vit_base_patch16_224.orig_in21k_ft_in1k", pretrained=False)
        base.head = nn.Linear(base.head.in_features, 100)
        state_dict = torch.hub.load_state_dict_from_url(
            VIT_BASE_URL,
            map_location="cpu",
            file_name="vit_base_patch16_224_in21k_ft_cifar100.pth",
        )
        base.load_state_dict(state_dict)
        return base

    raise ValueError(f"Unknown loader: {loader}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _make_loader(data_path, train, img_size, mean, std, batch_size, num_workers):
    t = []
    if img_size != 32:
        t.append(Resize((img_size, img_size)))
    t += [ToTensor(), Normalize(mean, std)]
    dataset = datasets.CIFAR100(
        root=data_path, train=train, download=True,
        transform=Compose(t),
    )
    kw = dict(num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    if num_workers > 0:
        kw["prefetch_factor"] = 4
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, **kw)


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
        if isinstance(out, (tuple, list)):
            out = out[0]
        correct += out.argmax(1).eq(y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

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
            row = f"{name:{name_w}s} | {fail}" + (f" | {fail}" if show_train else "")
        elif show_train:
            tr = f"{train_acc:9.2f}%" if train_acc is not None else f"{'N/A':>10s}"
            row = f"{name:{name_w}s} | {tr} | {test_acc:9.2f}%"
        else:
            row = f"{name:{name_w}s} | {test_acc:9.2f}%"
        print(row)
    print("=" * len(header))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate pretrained CIFAR-100 baselines (edadaltocg)")
    parser.add_argument("--data_path",   type=str, default="./data")
    parser.add_argument("--batch_size",  type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_train",    action="store_true", help="Skip train-set evaluation")
    parser.add_argument(
        "--models", type=str, nargs="+", default=None,
        help=f"Models to run. Available: {', '.join(e[0] for e in MODEL_REGISTRY)}",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.models:
        entries = []
        for name in args.models:
            if name in _ALIAS:
                entries.append(_ALIAS[name])
            else:
                print(f"  [warn] unknown model '{name}', skipping")
    else:
        entries = MODEL_REGISTRY

    results = []
    for display_name, hf_repo, loader, img_size, mean, std in entries:
        print(f"\n[{display_name}] loading ...")
        try:
            model = load_model(display_name, hf_repo, loader).to(device)
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append((display_name, None, None))
            continue

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  params: {n_params:,}  |  input: {img_size}×{img_size}")

        test_loader = _make_loader(
            args.data_path, train=False,
            img_size=img_size, mean=mean, std=std,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
        test_acc = evaluate(model, test_loader, device)
        print(f"  test  acc: {test_acc:.2f}%")

        train_acc = None
        if not args.no_train:
            train_loader = _make_loader(
                args.data_path, train=True,
                img_size=img_size, mean=mean, std=std,
                batch_size=args.batch_size, num_workers=args.num_workers,
            )
            train_acc = evaluate(model, train_loader, device)
            print(f"  train acc: {train_acc:.2f}%")

        results.append((display_name, train_acc, test_acc))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    _print_table(results, show_train=not args.no_train)


if __name__ == "__main__":
    main()
