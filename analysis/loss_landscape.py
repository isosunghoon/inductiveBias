import os
import warnings
import torch
import numpy as np
import wandb

from utils.config import parse_args
from utils.dataset import get_dataloader, RASampler
from train import setup, set_seed

from torch.utils.data import DataLoader, Subset
from pyhessian import hessian
from tqdm import tqdm

BATCH_SIZE = 16
RATIO = 0.05
ENABLE_FP16 = True

warnings.filterwarnings(
    "ignore",
    message="Using backward\\(\\) with create_graph=True will create a reference cycle.*",
    category=UserWarning,
)

def load_checkpoint(model, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

def make_subset(args, train_loader, ratio):
    dataset = train_loader.dataset
    n = max(1, int(len(dataset) * ratio))
    indices = np.random.choice(len(dataset), n, replace=False)
    subset = Subset(dataset, indices)

    common_loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        common_loader_kwargs["prefetch_factor"] = 4

    if args.augment == 'strong':
        sampler = RASampler(subset, num_repeats=3, shuffle=True)
        return DataLoader(
            subset,
            batch_size=args.train_batch_size,
            sampler=sampler,
            **common_loader_kwargs,
        )
    else:
        return DataLoader(
            subset,
            batch_size=args.train_batch_size,
            shuffle=True,
            **common_loader_kwargs,
        )

def calc_loss_landscape(args, model, loader, mixup_fn):
    model.train()
    res = []
    pbar = tqdm(loader, total=len(loader), desc="Hessian", dynamic_ncols=True)
    for step, batch in enumerate(pbar):
        model.zero_grad(set_to_none=True)

        x, y = batch
        x = x.to(args.device, non_blocking=True)
        y = y.to(args.device, non_blocking=True)
    
        if mixup_fn is not None:
            x, y = mixup_fn(x, y)

        def criterion(logits, y):
            with torch.amp.autocast('cuda', enabled=args.fp16):
                return torch.nn.functional.cross_entropy(logits, y, label_smoothing=args.label_smoothing)

        hessian_comp = hessian(model, criterion, data=(x, y), cuda=(args.device == "cuda"))
        eigenvalues, _ = hessian_comp.eigenvalues(top_n=5)
        res.append(eigenvalues)
        model.zero_grad(set_to_none=True)
        pbar.set_postfix({"step": step + 1, "top1": float(eigenvalues[0])})

    return res

def main():
    # 0. parse args
    args = parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    args.fp16 = ENABLE_FP16
    args.train_batch_size = BATCH_SIZE

    # 1. prepare model
    wandb.init(mode="disabled")
    model = setup(args)
    ckpt_path = os.path.join(args.output_path, "best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    load_checkpoint(model, ckpt_path, args.device)

    # 2. prepare dataset
    train_loader, test_loader, mixup_fn = get_dataloader(args)
    loader = make_subset(args, train_loader, ratio = RATIO)

    # 3. calculate eigenvalues
    evs = calc_loss_landscape(args, model, loader, mixup_fn)

    # 4. save eigenvalues
    os.makedirs(args.output_path, exist_ok=True)
    save_path = os.path.join(args.output_path, "loss_landscape_eigenvalues.npy")
    np.save(save_path, np.asarray(evs))
    print(f"Saved eigenvalues to {save_path}")

if __name__ == "__main__":
    main()
