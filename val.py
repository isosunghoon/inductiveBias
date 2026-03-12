import os
import torch
import wandb

from utils.config import parse_args
from utils.dataset import get_dataloader
from train import setup, validate, set_seed


def _load_checkpoint(model, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

def main():
    args = parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    wandb.init(mode="disabled")
    model = setup(args)

    ckpt_path = os.path.join(args.output_path, "best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    _load_checkpoint(model, ckpt_path, args.device)

    _, test_loader, mixup_fn = get_dataloader(args)
    val_acc = validate(args, model, test_loader)

    print(f"[Val] checkpoint: {ckpt_path}")
    print(f"[Val] acc: {val_acc:.2f}%")


if __name__ == "__main__":
    main()
