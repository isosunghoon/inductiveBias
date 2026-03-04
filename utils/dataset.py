import random

import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)


# ---------------------------------------------------------------------------
# Repeated Augmentation Sampler
# ---------------------------------------------------------------------------

class RASampler(torch.utils.data.Sampler):
    """에폭마다 각 샘플을 num_repeats번 반복해 미니배치 내에
    동일 이미지의 다른 증강 버전이 포함되도록 함.

    인덱스 순서: 셔플 후 [i0,i0,i0, i1,i1,i1, ...] (num_repeats=3)
    → 배치 내 연속된 num_repeats개가 같은 이미지, 각각 다른 랜덤 증강 적용.

    Reference: https://github.com/facebookresearch/deit/blob/main/samplers.py
    """

    def __init__(self, dataset, num_repeats: int, shuffle: bool = True):
        self.n = len(dataset)
        self.num_repeats = num_repeats
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.n).tolist()
        else:
            indices = list(range(self.n))
        repeated = [idx for idx in indices for _ in range(self.num_repeats)]
        return iter(repeated)

    def __len__(self) -> int:
        return self.n * self.num_repeats


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
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloader(args):
    """
    Returns:
        train_loader, test_loader, mixup_fn
        mixup_fn: MixupCutmix 인스턴스 (strong 레벨) 또는 None
    """
    if args.dataset == 'cifar100':
        level = getattr(args, 'augment', 'none')  # 'none' | 'weak' | 'strong'

        train_transforms = []
        test_transforms = []

        if args.img_size != 32:
            train_transforms.append(transforms.Resize((args.img_size, args.img_size)))
            test_transforms.append(transforms.Resize((args.img_size, args.img_size)))

        # weak & strong 공통: 기본 공간 증강
        if level in ('weak', 'strong'):
            train_transforms.extend([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ])

        # strong 전용: RandAugment (PIL 단계, ToTensor 이전)
        if level == 'strong':
            train_transforms.append(
                transforms.RandAugment(num_ops=2, magnitude=9)
            )

        train_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])

        # strong 전용: RandomErasing (ToTensor 이후 텐서 단계)
        if level == 'strong':
            train_transforms.append(
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')
            )

        test_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])

        transform_train = transforms.Compose(train_transforms)
        transform_test = transforms.Compose(test_transforms)

        trainset = datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=transform_train)
        testset  = datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=transform_test)

        common_loader_kwargs = {
            "num_workers": args.num_workers,
            "pin_memory": True,
            "persistent_workers": args.num_workers > 0,
        }
        if args.num_workers > 0:
            common_loader_kwargs["prefetch_factor"] = 4

        # strong 전용: Repeated Augmentation 샘플러 (num_repeats=3)
        if level == 'strong':
            sampler = RASampler(trainset, num_repeats=3, shuffle=True)
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

        # strong 전용: Mixup(alpha=0.8) + CutMix(alpha=1.0), switch_prob=0.5
        mixup_fn = None
        if level == 'strong':
            mixup_fn = MixupCutmix(
                num_classes=args.num_classes,
                mixup_alpha=0.8,
                cutmix_alpha=1.0,
                switch_prob=0.5,
            )

        return train_loader, test_loader, mixup_fn
