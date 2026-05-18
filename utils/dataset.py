import random
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# Google Research ViT (.npz) checkpoints assume (pixel - 127.5) / 127.5 → [-1, 1]
INCEPTION_MEAN = (0.5, 0.5, 0.5)
INCEPTION_STD  = (0.5, 0.5, 0.5)

NORM_TABLE = {
    "cifar100": (CIFAR100_MEAN, CIFAR100_STD),
    "imagenet": (IMAGENET_MEAN, IMAGENET_STD),
    "inception": (INCEPTION_MEAN, INCEPTION_STD),
}

TINY_IMAGENET_NATIVE_SIZE = 64
TINY_IMAGENET_NUM_CLASSES = 200


# ---------------------------------------------------------------------------
# Repeated Augmentation Sampler
# ---------------------------------------------------------------------------

class RASampler(torch.utils.data.Sampler):
    """
    에폭마다 각 샘플을 num_repeats번 반복해 미니배치 내에
    동일 이미지의 다른 증강 버전이 포함되도록 함.

    인덱스 순서: 셔플 후 [i0,i0,i0, i1,i1,i1, ...] (num_repeats=3)
    → 배치 내 연속된 num_repeats개가 같은 이미지, 각각 다른 랜덤 증강 적용.

    Reference: https://github.com/facebookresearch/deit/blob/main/samplers.py
    """

    def __init__(self, dataset, num_repeats: int, shuffle: bool = True, batch_size: int = 256):
        self.n = len(dataset)
        self.num_repeats = num_repeats
        self.shuffle = shuffle
        self.batch_size = batch_size
        # 에폭 길이를 데이터셋 크기(N)와 동일하게 유지하여 학습 시간 관리
        self.num_selected_samples = self.n

    def __iter__(self):
        if self.shuffle:
            base_indices = torch.randperm(self.n).tolist()
        else:
            base_indices = list(range(self.n))

        # 배치 기반 구성 (Batch-aware construction):
        # - 동일 이미지의 반복(repeats)을 같은 배치 안에 최대한 팩킹
        # - batch_size가 num_repeats로 나누어떨어지지 않는 경우도 고려함
        out = []
        ptr = 0

        def next_base_idx():
            nonlocal ptr
            if ptr >= self.n:
                ptr = 0
                if self.shuffle:
                    random.shuffle(base_indices)
            idx = base_indices[ptr]
            ptr += 1
            return idx

        while len(out) < self.num_selected_samples:
            remaining = self.num_selected_samples - len(out)
            cur_bs = min(self.batch_size, remaining)
            
            full_groups = cur_bs // self.num_repeats
            remainder = cur_bs % self.num_repeats

            batch = []
            # 같은 이미지를 num_repeats만큼 반복해서 배치에 채움
            for _ in range(full_groups):
                idx = next_base_idx()
                batch.extend([idx] * self.num_repeats)

            # 배치의 남은 공간(remainder) 처리
            if remainder > 0:
                idx = next_base_idx()
                batch.extend([idx] * remainder)

            out.extend(batch)

        return iter(out[:self.num_selected_samples])

    def __len__(self) -> int:
        return self.num_selected_samples


# ---------------------------------------------------------------------------
# Mixup / CutMix
# ---------------------------------------------------------------------------

def _rand_bbox(h: int, w: int, lam: float):
    """CutMix용 랜덤 사각형 (x1, y1, x2, y2) 반환."""
    cut_ratio = (1.0 - lam) ** 0.5
    cut_h, cut_w = int(h * cut_ratio), int(w * cut_ratio)
    cx, cy = random.randint(0, w), random.randint(0, h)
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, w)
    y2 = min(cy + cut_h // 2, h)
    return x1, y1, x2, y2


class MixupCutmix:
    """배치 단위 Mixup / CutMix. 소프트 라벨 [B, num_classes] 반환.

    - mixup_alpha > 0, cutmix_alpha = 0  → 항상 Mixup
    - mixup_alpha = 0, cutmix_alpha > 0  → 항상 CutMix
    - 둘 다 > 0                          → switch_prob 확률로 CutMix, 나머지 Mixup

    PyTorch 2.x F.cross_entropy는 float 2D 타겟(소프트 라벨)을 직접 지원.
    Reference: https://github.com/facebookresearch/deit
    """

    def __init__(self, num_classes: int, mixup_alpha: float, cutmix_alpha: float, switch_prob: float):
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.switch_prob = switch_prob

    def _one_hot(self, y: torch.Tensor, B: int) -> torch.Tensor:
        return torch.zeros(B, self.num_classes, device=y.device).scatter_(1, y.unsqueeze(1), 1.0)

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        B, _, H, W = x.shape
        index = torch.randperm(B, device=x.device)

        use_cutmix = (self.cutmix_alpha > 0) and (
            self.mixup_alpha <= 0 or random.random() < self.switch_prob
        )

        if use_cutmix:
            lam = float(np.random.beta(self.cutmix_alpha, self.cutmix_alpha))
            x1, y1, x2, y2 = _rand_bbox(H, W, lam)
            x = x.clone()
            x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
            lam = 1.0 - (x2 - x1) * (y2 - y1) / (H * W)  # 실제 면적 비율로 재계산
        else:
            lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
            x = lam * x + (1.0 - lam) * x[index]

        y_mixed = lam * self._one_hot(y, B) + (1.0 - lam) * self._one_hot(y[index], B)
        return x, y_mixed


# ---------------------------------------------------------------------------
# Tiny ImageNet layout (ImageFolder-compatible)
# ---------------------------------------------------------------------------

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def tiny_imagenet_root(data_path: str) -> Path:
    return Path(data_path) / "tiny-imagenet-200"


def download_tiny_imagenet(data_path: str) -> Path:
    """Download and extract Tiny ImageNet-200 if not present under data_path."""
    data_dir = Path(data_path)
    data_dir.mkdir(parents=True, exist_ok=True)
    root = tiny_imagenet_root(data_path)
    if root.is_dir():
        return root

    zip_path = data_dir / "tiny-imagenet-200.zip"
    if not zip_path.is_file():
        print(f"Downloading Tiny ImageNet to {zip_path} ...")
        urlretrieve(TINY_IMAGENET_URL, zip_path)

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)
    return root


def _tiny_imagenet_train_prepared(train_dir: Path) -> bool:
    class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        return False
    sample = class_dirs[0]
    return not (sample / "images").is_dir()


def _tiny_imagenet_val_prepared(val_dir: Path) -> bool:
    if not val_dir.is_dir():
        return False
    return not (val_dir / "images").is_dir()


def prepare_tiny_imagenet(data_path: str) -> Path:
    """
    Reorganize the official Tiny ImageNet-200 zip into ImageFolder layout.

    Expected raw layout under {data_path}/tiny-imagenet-200/:
      train/<wnid>/images/*.JPEG
      val/images/*.JPEG + val/val_annotations.txt

    After preparation:
      train/<wnid>/*.JPEG
      val/<wnid>/*.JPEG
    """
    root = tiny_imagenet_root(data_path)
    if not root.is_dir():
        download_tiny_imagenet(data_path)

    train_dir = root / "train"
    val_dir = root / "val"

    if train_dir.is_dir() and not _tiny_imagenet_train_prepared(train_dir):
        for class_dir in sorted(train_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            images_dir = class_dir / "images"
            if not images_dir.is_dir():
                continue
            for img_path in images_dir.iterdir():
                if img_path.is_file():
                    img_path.rename(class_dir / img_path.name)
            for txt_path in class_dir.glob("*.txt"):
                txt_path.unlink()
            images_dir.rmdir()

    if val_dir.is_dir() and not _tiny_imagenet_val_prepared(val_dir):
        anno_file = val_dir / "val_annotations.txt"
        images_dir = val_dir / "images"
        if not anno_file.is_file():
            raise FileNotFoundError(f"Missing validation annotations: {anno_file}")
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Missing validation images directory: {images_dir}")

        with open(anno_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                file_name, class_id = line.split()[:2]
                dest_dir = val_dir / class_id
                dest_dir.mkdir(parents=True, exist_ok=True)
                src = images_dir / file_name
                if src.is_file():
                    src.rename(dest_dir / file_name)

        images_dir.rmdir()

    return root


def _build_transforms(args, native_size: int):
    level = getattr(args, "augment", "none")  # 'none' | 'weak' | 'strong'
    norm_mean, norm_std = NORM_TABLE[getattr(args, "norm_type", "cifar100")]

    train_transforms = []
    test_transforms = []

    if args.img_size != native_size:
        train_transforms.append(transforms.Resize((args.img_size, args.img_size)))
        test_transforms.append(transforms.Resize((args.img_size, args.img_size)))

    if level in ("weak", "strong"):
        train_transforms.extend([
            transforms.RandomCrop(args.img_size, padding=4),
            transforms.RandomHorizontalFlip(),
        ])

    if level == "strong":
        train_transforms.append(transforms.RandAugment(num_ops=2, magnitude=9))

    train_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    if level == "strong":
        train_transforms.append(
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value="random")
        )

    test_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    return transforms.Compose(train_transforms), transforms.Compose(test_transforms), level


def _build_loaders(args, trainset, testset, level):
    common_loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        common_loader_kwargs["prefetch_factor"] = 4

    if level == "strong":
        sampler = RASampler(
            trainset,
            num_repeats=3,
            shuffle=True,
            batch_size=args.train_batch_size,
        )
        train_loader = DataLoader(
            trainset,
            batch_size=args.train_batch_size,
            sampler=sampler,
            **common_loader_kwargs,
        )
    else:
        train_loader = DataLoader(
            trainset,
            batch_size=args.train_batch_size,
            shuffle=True,
            **common_loader_kwargs,
        )

    test_loader = DataLoader(
        testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        **common_loader_kwargs,
    )

    mixup_fn = None
    if level == "strong":
        mixup_fn = MixupCutmix(
            num_classes=args.num_classes,
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            switch_prob=0.5,
        )

    return train_loader, test_loader, mixup_fn


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloader(args):
    """
    Returns:
        train_loader, test_loader, mixup_fn
        mixup_fn: MixupCutmix 인스턴스 (strong 레벨) 또는 None

    Supported datasets:
      - cifar100
      - tiny-imagenet (or tiny_imagenet): expects {data_path}/tiny-imagenet-200/
    """
    dataset = args.dataset.replace("_", "-")

    if dataset == "cifar100":
        transform_train, transform_test, level = _build_transforms(args, native_size=32)
        trainset = datasets.CIFAR100(
            root=args.data_path, train=True, download=True, transform=transform_train,
        )
        testset = datasets.CIFAR100(
            root=args.data_path, train=False, download=True, transform=transform_test,
        )
        return _build_loaders(args, trainset, testset, level)

    if dataset == "tiny-imagenet":
        if args.num_classes != TINY_IMAGENET_NUM_CLASSES:
            raise ValueError(
                f"tiny-imagenet requires num_classes={TINY_IMAGENET_NUM_CLASSES}, "
                f"got {args.num_classes}"
            )

        root = prepare_tiny_imagenet(args.data_path)
        transform_train, transform_test, level = _build_transforms(
            args, native_size=TINY_IMAGENET_NATIVE_SIZE,
        )
        trainset = datasets.ImageFolder(root / "train", transform=transform_train)
        testset = datasets.ImageFolder(root / "val", transform=transform_test)
        return _build_loaders(args, trainset, testset, level)

    raise ValueError(f"Unknown dataset: {args.dataset}")


def make_subset_loader(args, train_loader, ratio):
    """
    Build a DataLoader over a random subset of the training set.
    Uses RASampler when args.augment == 'strong', else shuffle=True.
    batch_size: if None, uses args.train_batch_size (e.g. use 1 for ERF so each step = one image).
    """
    dataset = train_loader.dataset
    n = max(1, int(len(dataset) * ratio))
    indices = np.random.choice(len(dataset), n, replace=False)
    subset = Subset(dataset, indices)

    bs = args.train_batch_size
    common_loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        common_loader_kwargs["prefetch_factor"] = 4

    if getattr(args, "augment", "none") == "strong":
        sampler = RASampler(subset, num_repeats=3, shuffle=True, batch_size=bs,)
        return DataLoader(
            subset,
            batch_size=bs,
            sampler=sampler,
            **common_loader_kwargs,
        )
    return DataLoader(
        subset,
        batch_size=bs,
        shuffle=True,
        **common_loader_kwargs,
    )

# 굳이 이정도까지? 
# def get_dataloader_with_subset(args, ratio):
#     """
#     Convenience: get_dataloader(args) + make_subset_loader on train.
#     Returns:
#         train_loader, test_loader, mixup_fn, subset_loader
#     """
#     train_loader, test_loader, mixup_fn = get_dataloader(args)
#     subset_loader = make_subset_loader(args, train_loader, ratio)
#     return train_loader, test_loader, mixup_fn, subset_loader
