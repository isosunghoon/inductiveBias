import os
import warnings

import torch
import wandb

from utils.config import get_parser, _apply_yaml
from train import set_seed, setup


MODEL_CONFIGS = {
    "convformer1": "output/base_model_exp/convformer-2026-03-14_05-12-40/config.yaml",
    "vit": "output/base_model_exp/convformer-2026-03-14_06-59-45/config.yaml",
}


def build_args_from_yaml(base_cfg: str, model_cfg: str):
    parser = get_parser()
    # CLI 인자 없이 기본값만 가진 args 생성
    args = parser.parse_args([])

    if base_cfg is not None:
        _apply_yaml(args, base_cfg)
    if model_cfg is not None:
        _apply_yaml(args, model_cfg)

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    return args


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    return total, trainable, non_trainable


def print_param_table(rows):
    header = f"{'Model':20s} | {'Total':>15s} | {'Trainable':>15s} | {'Non-train':>15s}"
    print(header)
    print("-" * len(header))
    for name, total, trainable, non_trainable in rows:
        print(f"{name:20s} | {total:15,d} | {trainable:15,d} | {non_trainable:15,d}")


def main():
    base_cfg = "config/base.yaml"

    # wandb는 disabled 모드로 한 번만 초기화 (setup에서 define_metric 사용)
    wandb.init(mode="disabled")

    rows = []
    for model_name, cfg_path in MODEL_CONFIGS.items():
        args = build_args_from_yaml(base_cfg, cfg_path)
        # model 이름이 yaml에서 지정돼 있지 않으면 fallback
        if getattr(args, "model", None) in (None, ""):
            args.model = model_name

        model = setup(args)
        total, trainable, non_trainable = count_parameters(model)
        rows.append((model_name, total, trainable, non_trainable))

    print_param_table(rows)


if __name__ == "__main__":
    main()
