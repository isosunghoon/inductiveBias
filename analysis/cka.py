import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import wandb
import argparse
import copy
import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
        hooks.append(layer.register_forward_hook(catcher_1.hook(f"feat{i}")))
    for i, layer in enumerate(layers_2):
        hooks.append(layer.register_forward_hook(catcher_2.hook(f"feat{i}")))
        
    # Per-layer feature storage
    feats_1 = {i: [] for i in range(len(layers_1))}
    feats_2 = {j: [] for j in range(len(layers_2))}

    for i, batch in enumerate(dataloader):
        x, _ = batch
        x = x.to(args.device)

        # model_1를 실행하면서 자동으로 hook이 연산됨
        catcher_1.outputs.clear()
        _ = model_1(x)
        for j in range(len(layers_1)):
            f1 = flatten_features(catcher_1.outputs[f"feat{i}"]).cpu()
            feats_1[j].append(f1)

        catcher_2.outputs.clear()
        _ = model_2(x)
        for j in range(len(layers_2)):
            f2 = flatten_features(catcher_2.outputs[f"feat{i}"]).cpu()
            feats_2[j].append(f2)
        
    for h in hooks:
        h.remove()

    feats_1 = {i: torch.cat(feats_1[i], dim=0) for i in range(len(layers_1))}
    feats_2 = {i: torch.cat(feats_2[i], dim=0) for i in range(len(layers_2))}

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
        _apply_yaml(args2, config_paths.base_config)
    if config_paths.config2 is not None:
        _apply_yaml(args2, config_paths.config2)

    return args1, args2

def _load_checkpoint(model, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)


def _layer_labels(n):
    """Generate tick labels: ['patch_embed', 'block_0', 'block_1', ...]"""
    return ["patch_embed"] + [f"block_{i}" for i in range(n - 1)]


def save_cka_results(cka_matrix, model1_name, model2_name):
    out_dir = os.path.join(os.path.dirname(__file__), "cka_results")
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    stem = f"{model1_name}_vs_{model2_name}_{timestamp}"

    # --- .txt ---
    txt_path = os.path.join(out_dir, f"{stem}.txt")
    mat = cka_matrix.numpy()
    with open(txt_path, "w") as f:
        f.write(f"CKA matrix  ({model1_name} vs {model2_name})\n")
        f.write(f"shape: {mat.shape[0]} x {mat.shape[1]}  "
                f"(rows=model1 layers, cols=model2 layers)\n\n")
        header = "\t".join(_layer_labels(mat.shape[1]))
        f.write(f"\t{header}\n")
        for i, row_label in enumerate(_layer_labels(mat.shape[0])):
            vals = "\t".join(f"{v:.4f}" for v in mat[i])
            f.write(f"{row_label}\t{vals}\n")
    print(f"[CKA] txt  saved → {txt_path}")

    # --- heatmap ---
    n_rows, n_cols = mat.shape
    fig_w = max(6, n_cols * 0.7)
    fig_h = max(5, n_rows * 0.7)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="magma", aspect="auto")
    plt.colorbar(im, ax=ax, label="CKA similarity")

    col_labels = _layer_labels(n_cols)
    row_labels = _layer_labels(n_rows)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8)

    ax.set_xlabel(f"Model 2: {model2_name}", fontsize=10)
    ax.set_ylabel(f"Model 1: {model1_name}", fontsize=10)
    ax.set_title(f"Linear CKA  ({model1_name} vs {model2_name})", fontsize=11)

    # annotate cells
    for i in range(n_rows):
        for j in range(n_cols):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if mat[i, j] < 0.6 else "black")

    plt.tight_layout()
    png_path = os.path.join(out_dir, f"{stem}.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"[CKA] plot saved → {png_path}")


def main():
    args1, args2 = _parse_args()
    wandb.init(mode="disabled")

    args1.device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args1.seed)
    
    model1 = setup(args1)
    ckpt_path1 = os.path.join(args1.output_path, "base_model_exp/mlpmixer-2026-03-13_02-56-40/best.pt")
    if not os.path.exists(ckpt_path1):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path1}")
    _load_checkpoint(model1, ckpt_path1, args1.device)

    args2.device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args2.seed)
    
    model2 = setup(args2)
    ckpt_path2 = os.path.join(args2.output_path, "base_model_exp/vit-2026-03-12_15-39-12/best.pt")
    if not os.path.exists(ckpt_path2):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path2}")
    _load_checkpoint(model2, ckpt_path2, args2.device)

    _, test_loader, _ = get_dataloader(args1)

    """
    # print each model's layers name: 
    print(f"model1: layers name")
    for name, module in model1.named_modules():
        print(name)

    print(f"model2: layers name")
    for name, module in model2.named_modules():
        print(name)
    """

    # patch_embed + all MetaFormerBlocks — covers full representation trajectory.
    # Change the layers part if you want to make different experiments
    layers1 = [model1.patch_embed] + list(model1.blocks)
    layers2 = [model2.patch_embed] + list(model2.blocks)
    cka_matrix = compare_cka(args1, model1, layers1, model2, layers2, test_loader)

    print("CKA matrix:")
    print(cka_matrix)
    print("shape:", cka_matrix.shape)

    save_cka_results(cka_matrix.cpu(), args1.model, args2.model)


if __name__ == "__main__":
    main()