import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import wandb
import argparse
import copy
import datetime
import itertools
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
    
    n, d1 = X.shape
    _, d2 = Y.shape

    # Choose the faster algorithm between feature and sample CKA 
    if d1*d2+d1*d1+d2*d2 < n*(d1+d2):
        hsic = torch.norm(Y.T@X, p="fro")**2
        norm_x = torch.norm(X.T@X, p="fro")
        norm_y = torch.norm(Y.T@Y, p="fro")
        return hsic / (norm_x*norm_y+epsilon)
    else:
        K = X @ X.T           
        L = Y @ Y.T          

        hsic = torch.sum(K * L)
        norm_k = torch.sqrt(torch.sum(K * K).clamp_min(epsilon))
        norm_l = torch.sqrt(torch.sum(L * L).clamp_min(epsilon))
        return hsic / (norm_k*norm_l+epsilon)

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
def compare_cka(args, model_1, layers_1, model_2, layers_2, dataloader, max_samples=1024):
    model_1.eval().to(args.device)
    model_2.eval().to(args.device)

    catcher_1 = ActivationCatcher()
    catcher_2 = ActivationCatcher()

    hooks = []

    for i, layer in enumerate(layers_1):
        hooks.append(layer.register_forward_hook(catcher_1.hook(f"feat{i}")))
    for i, layer in enumerate(layers_2):
        hooks.append(layer.register_forward_hook(catcher_2.hook(f"feat{i}")))
    
    max_batches = max_samples // dataloader.batch_size
    total_batches = min(max_batches, len(dataloader))

    # Per-layer feature storage
    feats_1 = {i: [] for i in range(len(layers_1))}
    feats_2 = {j: [] for j in range(len(layers_2))}

    print(f"[CKA] collecting features: 0/{total_batches} batches", flush=True)
    for i, batch in enumerate(dataloader):
        if i>=max_batches:
            break
        x, _ = batch
        x = x.to(args.device)

        # model_1를 실행하면서 자동으로 hook이 연산됨
        catcher_1.outputs.clear()
        _ = model_1(x)
        for j in range(len(layers_1)):
            f1 = flatten_features(catcher_1.outputs[f"feat{j}"]).cpu()
            feats_1[j].append(f1)

        catcher_2.outputs.clear()
        _ = model_2(x)
        for j in range(len(layers_2)):
            f2 = flatten_features(catcher_2.outputs[f"feat{j}"]).cpu()
            feats_2[j].append(f2)

        print(f"[CKA] collecting features: {i+1}/{total_batches} batches", flush=True)
        
    for h in hooks:
        h.remove()

    feats_1 = {i: torch.cat(feats_1[i], dim=0) for i in range(len(layers_1))}
    feats_2 = {i: torch.cat(feats_2[i], dim=0) for i in range(len(layers_2))}

    cka_matrix = torch.empty(len(layers_1), len(layers_2), dtype=torch.float32)

    print(f"[CKA] computing CKA matrix ({len(layers_1)}x{len(layers_2)})", flush=True)
    for i in range(len(layers_1)):
        X = feats_1[i]
        for j in range(len(layers_2)):
            Y = feats_2[j]
            cka_matrix[i, j] = linear_cka(X, Y)
        print(f"[CKA] matrix row {i+1}/{len(layers_1)} done", flush=True)

    return cka_matrix

# Returns list of (args, ckpt_path) for every model passed on the command line
def _parse_args():
    config_path_parser = argparse.ArgumentParser(add_help=False)
    config_path_parser.add_argument("--base_config", type=str, default="./config/base.yaml")
    config_path_parser.add_argument("--configs", type=str, nargs="*", default=[],
                                    help="per-model config yamls (same order as --ckpts)")
    config_path_parser.add_argument("--ckpts", type=str, nargs="+", required=True,
                                    help="one or more checkpoint paths")
    config_paths, _ = config_path_parser.parse_known_args()

    if config_paths.configs and len(config_paths.configs) != len(config_paths.ckpts):
        raise ValueError(f"--configs length ({len(config_paths.configs)}) must match --ckpts length ({len(config_paths.ckpts)})")

    parser = get_parser()
    defaults = parser.parse_args([])

    model_configs = []
    for idx, ckpt in enumerate(config_paths.ckpts):
        args = copy.deepcopy(defaults)
        if config_paths.base_config is not None:
            _apply_yaml(args, config_paths.base_config)
        if idx < len(config_paths.configs):
            _apply_yaml(args, config_paths.configs[idx])
        model_configs.append((args, ckpt))

    return model_configs

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


def _cka_txt(mat, model1_name, model2_name) -> str:
    """Build the CKA matrix as a tab-separated string."""
    lines = [
        f"CKA matrix  ({model1_name} vs {model2_name})",
        f"shape: {mat.shape[0]} x {mat.shape[1]}  (rows=model1 layers, cols=model2 layers)\n",
        "\t" + "\t".join(_layer_labels(mat.shape[1])),
    ]
    for i, row_label in enumerate(_layer_labels(mat.shape[0])):
        vals = "\t".join(f"{v:.4f}" for v in mat[i])
        lines.append(f"{row_label}\t{vals}")
    return "\n".join(lines) + "\n"


def _cka_figure(mat, model1_name, model2_name):
    """Build and return a CKA heatmap Figure."""
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

    for i in range(n_rows):
        for j in range(n_cols):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if mat[i, j] < 0.6 else "black")

    plt.tight_layout()
    return fig


def save_cka_results(cka_matrix, model1_name, model2_name):
    out_dir = os.path.join(os.path.dirname(__file__), "cka_results")
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    stem = f"{model1_name}_vs_{model2_name}_{timestamp}"
    mat = cka_matrix.numpy()

    txt_path = os.path.join(out_dir, f"{stem}.txt")
    with open(txt_path, "w") as f:
        f.write(_cka_txt(mat, model1_name, model2_name))
    print(f"[CKA] txt  saved → {txt_path}")

    fig = _cka_figure(mat, model1_name, model2_name)
    png_path = os.path.join(out_dir, f"{stem}.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"[CKA] plot saved → {png_path}")


def analyze_cka(args1, model1, args2, model2, max_samples: int = 1024, **kwargs) -> list:
    """Pipeline-compatible CKA analysis for a pair of models.

    Signature matches the n_models=2 convention in run_pipeline:
        fn(args1, model1, args2, model2, **kwargs) -> list[AnalysisResult]

    Layers are derived automatically from each model:
        [patch_embed] + list(blocks)

    Parameters
    ----------
    args1, args2  : parsed configs (from pipeline._load_args)
    model1, model2: loaded models (from pipeline.build_model)
    max_samples   : max number of test samples used for CKA (default 1024)

    Returns
    -------
    list[AnalysisResult]
        - suffix="cka", fmt=".txt"  – tab-separated CKA matrix
        - suffix="cka", fmt=".png"  – heatmap figure
    """
    from analysis.pipeline import AnalysisResult

    layers1 = [model1.patch_embed] + list(model1.blocks)
    layers2 = [model2.patch_embed] + list(model2.blocks)

    model1_name = getattr(args1, "model", "model1")
    model2_name = getattr(args2, "model", "model2")

    print(f"[CKA] {model1_name} vs {model2_name}", flush=True)

    _, test_loader, _ = get_dataloader(args1)
    cka_matrix = compare_cka(args1, model1, layers1, model2, layers2, test_loader,
                              max_samples=max_samples)

    mat = cka_matrix.cpu().numpy()
    return [
        AnalysisResult("cka", _cka_txt(mat, model1_name, model2_name), ".txt"),
        AnalysisResult("cka", _cka_figure(mat, model1_name, model2_name),  ".png"),
    ]


def main():
    model_configs = _parse_args()
    wandb.init(mode="disabled")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load all models
    models = []
    for args, ckpt_path in model_configs:
        args.device = device
        set_seed(args.seed)
        model = setup(args)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        _load_checkpoint(model, ckpt_path, device)
        models.append((args, model))

    # Use the first model's args for the dataloader (dataset config should be shared)
    _, test_loader, _ = get_dataloader(model_configs[0][0])

    n = len(models)
    pairs = list(itertools.combinations(range(n), 2))
    print(f"[CKA] {n} models → {len(pairs)} pairs", flush=True)

    for pair_idx, (i, j) in enumerate(pairs):
        args_i, model_i = models[i]
        args_j, model_j = models[j]

        print(f"\n[CKA] pair {pair_idx+1}/{len(pairs)}: {args_i.model} vs {args_j.model}", flush=True)

        # patch_embed + all MetaFormerBlocks — covers full representation trajectory.
        # Change the layers part to try different layers
        layers_i = [model_i.patch_embed] + list(model_i.blocks)
        layers_j = [model_j.patch_embed] + list(model_j.blocks)
        cka_matrix = compare_cka(args_i, model_i, layers_i, model_j, layers_j, test_loader)

        print("CKA matrix:")
        print(cka_matrix)
        print("shape:", cka_matrix.shape)

        save_cka_results(cka_matrix.cpu(), args_i.model, args_j.model)


if __name__ == "__main__":
    main()