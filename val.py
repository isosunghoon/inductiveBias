import os
import argparse
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
    # --ckpt_path는 config와 독립적으로 처리
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--ckpt_path", type=str, default=None,
                            help="직접 체크포인트 경로 지정 (없으면 output_path/best.pt 사용)")
    pre_args, remaining = pre_parser.parse_known_args()

    args = parse_args(remaining)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    # ckpt_path 결정
    if pre_args.ckpt_path is not None:
        ckpt_path = pre_args.ckpt_path
    else:
        ckpt_path = os.path.join(args.output_path, "best.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    args.no_wandb = True
    wandb.init(mode="disabled")
    model = setup(args)

    _load_checkpoint(model, ckpt_path, args.device)

    # augment='none'으로 고정: train set도 증강 없이 평가
    args.augment = 'none'
    train_loader, test_loader, _ = get_dataloader(args)

    train_acc = validate(args, model, train_loader)
    val_acc = validate(args, model, test_loader)

    print(f"[Val] checkpoint : {ckpt_path}")
    print(f"[Val] train acc  : {train_acc:.2f}%")
    print(f"[Val] test acc   : {val_acc:.2f}%")


if __name__ == "__main__":
    main()
