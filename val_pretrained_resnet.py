from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import get_model, get_model_weights
from tqdm import tqdm

from train import set_seed, validate

# -------------------------------------------------
# Change only these values when you want a new run.
# -------------------------------------------------
MODEL_NAME = "resnet18"
WEIGHTS_NAME = "DEFAULT"
DATA_PATH = "./data"
NUM_CLASSES = 100
TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 256
NUM_WORKERS = 4
SEED = 67
FP16 = True
EPOCHS = 10
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 1e-4
FREEZE_BACKBONE = True


def build_args(device: str) -> SimpleNamespace:
    return SimpleNamespace(
        device=device,
        fp16=FP16 and device == "cuda",
    )


def build_model(device: str):
    weights_enum = get_model_weights(MODEL_NAME)
    weights = weights_enum.DEFAULT if WEIGHTS_NAME == "DEFAULT" else getattr(weights_enum, WEIGHTS_NAME)
    model = get_model(MODEL_NAME, weights=weights)

    if not hasattr(model, "fc"):
        raise ValueError(f"Model does not expose an fc head: {MODEL_NAME}")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)

    if FREEZE_BACKBONE:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False

    model = model.to(device)
    return torch.compile(model), weights


def build_dataloaders(weights):
    eval_preprocess = weights.transforms()
    crop_size = eval_preprocess.crop_size[0] if isinstance(eval_preprocess.crop_size, (tuple, list)) else eval_preprocess.crop_size
    resize_size = eval_preprocess.resize_size[0] if isinstance(eval_preprocess.resize_size, (tuple, list)) else eval_preprocess.resize_size
    mean = eval_preprocess.mean
    std = eval_preprocess.std

    train_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    trainset = datasets.CIFAR100(
        root=DATA_PATH,
        train=True,
        download=True,
        transform=train_transform,
    )
    testset = datasets.CIFAR100(
        root=DATA_PATH,
        train=False,
        download=True,
        transform=test_transform,
    )

    common_kwargs = {
        "num_workers": NUM_WORKERS,
        "pin_memory": True,
        "persistent_workers": NUM_WORKERS > 0,
    }
    if NUM_WORKERS > 0:
        common_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(
        trainset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        **common_kwargs,
    )
    test_loader = DataLoader(
        testset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        **common_kwargs,
    )
    return train_loader, test_loader


def train_head(args, model, train_loader):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda', enabled=args.fp16)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        progress = tqdm(train_loader, desc=f"Train {epoch}/{EPOCHS}", dynamic_ncols=True)

        for x, y in progress:
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=args.fp16):
                logits = model(x)
                loss = F.cross_entropy(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * y.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            progress.set_postfix(loss=f"{running_loss / total:.4f}", acc=f"{100.0 * correct / total:.2f}%")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(SEED)

    args = build_args(device)
    model, weights = build_model(device)
    train_loader, test_loader = build_dataloaders(weights)

    print(f"[Run] model: {MODEL_NAME}")
    print(f"[Run] weights: {WEIGHTS_NAME}")
    print(f"[Run] device: {device}")
    print(f"[Run] freeze_backbone: {FREEZE_BACKBONE}")
    print(f"[Run] epochs: {EPOCHS}")

    train_head(args, model, train_loader)
    val_acc = validate(args, model, test_loader)
    print(f"[Val] acc: {val_acc:.2f}%")


if __name__ == "__main__":
    main()
