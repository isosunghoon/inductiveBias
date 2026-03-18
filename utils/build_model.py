"""
Model build/load utilities. Use build_model() to get a configured model with checkpoint loaded.
"""
import os
import torch
import wandb

from train import setup, set_seed


def _normalize_state_dict(state_dict):
    """
    체크포인트가 torch.compile()로 저장된 경우 키에 '_orig_mod.' 접두사가 붙어 있어
    현재 모델(state_dict)과 맞지 않을 수 있음. 접두사를 제거한 state_dict를 반환.
    """
    prefix = "_orig_mod."
    if not any(k.startswith(prefix) for k in state_dict):
        return state_dict
    return {k.removeprefix(prefix): v for k, v in state_dict.items()}


def load_checkpoint(model, ckpt_path, device):
    """Load state_dict from checkpoint into model. Handles both raw state_dict and dict with 'state_dict' key."""
    checkpoint = torch.load(ckpt_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    state_dict = _normalize_state_dict(state_dict)
    target = getattr(model, "_orig_mod", model)
    target.load_state_dict(state_dict, strict=True) 


def build_model(args, ckpt_name="best.pt"):
    """
    Build model from args and load checkpoint from args.output_path / ckpt_name.
    Flow: set device/seed -> wandb disabled -> setup(args) -> load checkpoint.

    Args:
        args: Parsed args (e.g. from parse_args()). Must have output_path, seed, and setup-compatible fields.
        ckpt_name: Checkpoint filename under args.output_path (default "best.pt"). Path is always args.output_path / ckpt_name.

    Returns:
        model: Model on args.device with checkpoint loaded.
    """
    args.device = getattr(args, "device", None) or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    wandb.init(mode="disabled")
    model = setup(args)
    path = os.path.join(args.output_path, ckpt_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    load_checkpoint(model, path, args.device)

    return model
