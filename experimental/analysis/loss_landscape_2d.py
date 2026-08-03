import os
import warnings
import torch
import numpy as np

from utils.config import parse_args
from utils.dataset import get_dataloader, make_subset_loader
from utils.build_model import build_model
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

        hessian_comp = hessian(model, criterion, data=(x, y), cuda=args.device.startswith("cuda"))
        eigenvalues, _ = hessian_comp.eigenvalues(top_n=5)
        res.append(eigenvalues)
        model.zero_grad(set_to_none=True)
        pbar.set_postfix({"step": step + 1, "top1": float(eigenvalues[0])})

    return res

def main():
    args = parse_args()
    args.fp16 = ENABLE_FP16
    args.train_batch_size = BATCH_SIZE

    # 1. prepare model (build_model: setup + load best.pt; unwrap compile for pyhessian)
    model = build_model(args)

    # 2. prepare dataset (train/test/mixup + subset loader for Hessian)
    train_loader, test_loader, mixup_fn = get_dataloader(args)
    loader = make_subset_loader(args, train_loader, ratio=RATIO)

    # 3. calculate eigenvalues
    evs = calc_loss_landscape(args, model, loader, mixup_fn)

    # 4. save eigenvalues
    os.makedirs(args.output_path, exist_ok=True)
    save_path = os.path.join(args.output_path, "loss_landscape_eigenvalues.npy")
    np.save(save_path, np.asarray(evs))
    print(f"Saved eigenvalues to {save_path}")

if __name__ == "__main__":
    main()
