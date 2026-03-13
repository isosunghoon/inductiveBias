import torch
import torch.nn as nn
import os
import wandb
import argparse
import copy

from utils.config import get_parser, _apply_yaml
from utils.dataset import get_dataloader
from train import setup, set_seed

def linear_cka(X, Y, epsilon=1e-8): 
    """
    X: n*d1, Y:n*d2가 들어오면 X와 Y의 CKA 값을 반환함
    """

    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    hsic = torch.norm(Y.T@X, p="fro")**2
    norm_x = torch.norm(X.T@X, p="fro")
    norm_y = torch.norm(Y.T@Y, p="fro")

    return hsic / (norm_x*norm_y+epsilon)

def flatten_features(feat):
    return feat.reshape(feat.size(0), -1)

class ActivationCatcher:
    def __init__(self):
        self.outputs = {}

    def hook(self, name):
        def fn(module, input, output):
            self.outputs[name] = output.detach()
        return fn
    
@torch.no_grad()
def compare_cka(args, model_1, layers_1, model_2, layers_2, dataloader):
    model_1.eval().to(args.device)
    model_2.eval().to(args.device)

    catcher_1 = ActivationCatcher()
    catcher_2 = ActivationCatcher()

    hooks = []

    for i, layer in enumerate(layers_1):
        hooks.append(layer.register_forward_hook(catcher_1.outputs(f"feat{i}")))
    for i, layer in enumerate(layers_2):
        hooks.append(layer.register_forward_hook(catcher_2.outputs(f"feat{i}")))
        
    # Per-layer feature storage
    feats_1 = {i: [] for i in range(layers_1)}
    feats_2 = {j: [] for j in range(layers_2)}

    for i, batch in enumerate(dataloader):
        x, _ = batch
        x = x.to(args.device)

        catcher_1.outputs.clear()

        # model_1를 실행하면서 자동으로 hook이 연산됨
        _ = model_1(x)
        for i in range(len(layers_1)):
            f1 = flatten_features(catcher_1.outputs[f"feat{i}"]).cpu()
            feats_1[i].append(f1)

        catcher_2.outputs_clear()
        _ = model_2(x)
        for j in range(len(layers_2)):
            f2 = flatten_features(catcher_2.outputs[f"feat{j}"]).cpu()
            feats_2[i].append(f2)
        
    for h in hooks:
        h.remove()

    feats_1 = {i: torch.cat(feats_1[i], dim=0) for i in range(layers_1)}
    feats_2 = {i: torch.cat(feats_2[i], dim=0) for i in range(layers_2)}

    cka_matrix = torch.empty(len(layers_1), len(layers_2), dtype=torch.float32)

    for i in range(len(layers_1)):
        X = feats_1[i]
        for j in range(len(layers_2)):
            Y = feats_2[j]
            cka_matrix[i, j] = linear_cka(X, Y)

    return cka_matrix

# Defined local _parse_args and _load_checkpoint for two models
def _parse_args():
    # Parse only config paths first so we know which YAML files to load.
    config_path_parser = argparse.ArgumentParser(add_help=False)
    config_path_parser.add_argument("--base_config", type=str, default="./config/base.yaml")
    config_path_parser.add_argument("--config1", type=str, default=None)
    config_path_parser.add_argument("--config2", type=str, default=None)
    config_paths, _ = config_path_parser.parse_known_args()

    parser = get_parser()

    # Start from argparse defaults.
    defaults = parser.parse_args([])

    # Create args1
    args1 = copy.deepcopy(defaults)
    if config_paths.base_config is not None:
        _apply_yaml(args1, config_paths.base_config)
    if config_paths.config1 is not None:
        _apply_yaml(args1, config_paths.config1)

    # Create args2
    args2 = copy.deepcopy(defaults)
    if config_paths.base_config is not None:
        _apply_yaml(args1, config_paths.base_config)
    if config_paths.config2 is not None:
        _apply_yaml(defaults, config_paths.config2)

    return args1, args2

def _load_checkpoint(model, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)


def main():
    args1, args2 = _parse_args()
    wandb.init(mode="disabled")

    args1.device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args1.seed)
    
    model1 = setup(args1)
    ckpt_path1 = os.path.join(args1.output_path, "best.pt")
    if not os.path.exists(ckpt_path1):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path1}")
    _load_checkpoint(model1, ckpt_path1, args1.device)

    args2.device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args2.seed)
    
    model2 = setup(args2)
    ckpt_path2 = os.path.join(args2.output_path, "best.pt")
    if not os.path.exists(ckpt_path2):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path2}")
    _load_checkpoint(model2, ckpt_path2, args2.device)

    _, test_loader, mixup_fn1 = get_dataloader(args1)

    """
    -> print each model's layers name
    print(f"model1: layers name")
    for name, module in model1.named_modules():
        print(name)

    print(f"model2: layers name")
    for name, module in model2.named_modules():
        print(name)
    """

    layers1 = []
    layers2 = []
    # cka_matrix = compare_cka(model1, layers1, model2, layers2, test_loader)


        

if __name__ == "__main__":
    main()