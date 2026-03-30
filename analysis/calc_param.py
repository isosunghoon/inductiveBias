import os
import warnings

import torch
import wandb

from utils.config import get_parser, _apply_yaml
from train import set_seed, setup

OUTPUT_ROOT = "output"
BASE_CFG = "config/base.yaml"


def discover_experiments(output_root: str):
    """output/{project}/{model}-{timestamp}/ 구조를 탐색해 (name, config_cfg) 목록 반환."""
    experiments = []
    for project in sorted(os.listdir(output_root)):
        project_dir = os.path.join(output_root, project)
        if not os.path.isdir(project_dir):
            continue
        for run in sorted(os.listdir(project_dir)):
            run_dir = os.path.join(project_dir, run)
            config_cfg = os.path.join(run_dir, "config.yaml")
            if os.path.isfile(config_cfg):
                name = f"{project}/{run}"
                experiments.append((name, config_cfg))
    return experiments


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
    name_w = max(len(r[0]) for r in rows) + 2
    header = f"{'Model':{name_w}s} | {'Total':>15s} | {'Trainable':>15s} | {'Non-train':>15s}"
    print(header)
    print("-" * len(header))
    for name, total, trainable, non_trainable in rows:
        print(f"{name:{name_w}s} | {total:15,d} | {trainable:15,d} | {non_trainable:15,d}")


def main():
    # wandb는 disabled 모드로 한 번만 초기화 (setup에서 define_metric 사용)
    wandb.init(mode="disabled")

    experiments = discover_experiments(OUTPUT_ROOT)
    if not experiments:
        print(f"No experiments found under '{OUTPUT_ROOT}/'")
        return

    rows = []
    for name, cfg_path in experiments:
        args = build_args_from_yaml(BASE_CFG, cfg_path)
        model = setup(args)
        total, trainable, non_trainable = count_parameters(model)
        rows.append((name, total, trainable, non_trainable))

    print_param_table(rows)


if __name__ == "__main__":
    main()
