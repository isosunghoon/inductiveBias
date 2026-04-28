import os
import argparse
import torch
import wandb
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from utils.config import parse_args, resolve_runtime_device
from utils.dataset import get_dataloader
from train import setup, set_seed


def _load_checkpoint(model, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    prefix = "_orig_mod."
    if any(k.startswith(prefix) for k in state_dict):
        state_dict = {k.removeprefix(prefix): v for k, v in state_dict.items()}

    target = getattr(model, "_orig_mod", model)
    target.load_state_dict(state_dict, strict=True)


def grid_distance_bins(H, W, anchor_x, anchor_y, max_d: int=10, device: torch.device=None):
    """
    Given the shape [H, W] of the image and the anchor_y and anchor_x,
    return the distance_bins - a list of length (max_d+1)
    Measure the distance using Manhattan Distance
    TODO Improve efficiency of the for loop; don't need iteration of max_d+1
    """
    xx, yy = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    dist = (xx - anchor_x).abs() + (yy - anchor_y).abs()
    bins = []

    for d in range(max_d + 1):
        mask = (dist == d).reshape(-1)
        bins.append(torch.nonzero(mask, as_tuple=False).squeeze(1))
    return bins


def occlude_patches(x: torch.Tensor, patch_ids: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
    """
    Given x: [B,C,H,W] & patch_ids -> return occluded copy
    TODO improve efficiency of for loop? get px, py as index list and use parallel computing
    """
    B, C, H, W = x.shape
    out = x.clone()
    for pid in patch_ids.tolist():
        px, py = pid // W, pid % W
        out[:, :, px, py] = fill

    return out


@torch.no_grad()
def calculate_delta_given_anchor(args, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, anchor_row: int = 0, anchor_col: int = 0) -> dict[int, torch.Tensor]:
    """
    Compute true-class logit drops for each distance bin from one anchor.

    Returns:
        A dict mapping distance -> tensor of shape [B], where each value is the
        logit drop for that distance bin.
    """
    logits = model(x)
    true_logit = logits.gather(1, y.view(-1, 1)).squeeze(1)

    _, _, height, width = x.shape
    bins = grid_distance_bins(height, width, anchor_row, anchor_col, device=x.device)

    deltas: dict[int, torch.Tensor] = {}
    for distance, ids in enumerate(bins):
        if ids.numel() < k_per_bin:
            continue
        if distance==0:
            print(f"ids.numel: {ids.numel()}")
        sample_size = min(args.k_per_bin, ids.numel())
        perm = torch.randperm(ids.numel(), device=ids.device)[:sample_size]
        selected_ids = ids[perm]

        x_occ = occlude_patches(x, selected_ids, fill=fill)
        logits_occ = model(x_occ)
        true_logit_occ = logits_occ.gather(1, y.view(-1, 1)).squeeze(1)
        deltas[distance] = true_logit - true_logit_occ

    return deltas


@torch.no_grad()
def calculate_mean_delta(args, model: torch.nn.Module, test_loader, ) -> np.ndarray:
    """
    Return the delta values after aggregating all of them
    """
    model.eval()
    distance_to_values: dict[int, list[torch.Tensor]] = defaultdict(list)

    for batch_idx, (x, y) in enumerate(
        tqdm(test_loader, desc="Distance-conditioned Occlusion Sensitivity")
    ):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        x = x.to(args.device)
        y = y.to(args.device)
        _, _, height, width = x.shape
        print(f"batch_idx: {batch_idx}, max_batches:{args.max_batches}, max_d:{args.max_d}")
        for anchor_row in range(0, height, args.anchor_stride):
            for anchor_col in range(0, width, args.anchor_stride):
                delta_dict = calculate_delta_given_anchor(
                    args, model=model, x=x, y=y, anchor_row=anchor_row, anchor_col=anchor_col
                    )
                for distance, delta in delta_dict.items():
                    distance_to_values[distance].append(delta.detach().cpu())

    if not distance_to_values:
        raise RuntimeError("No deltas were collected. Check the dataloader and anchor settings.")

    mean_delta = []
    for distance in sorted(distance_to_values):
        values = torch.cat(distance_to_values[distance], dim=0)
        mean_delta.append(values.mean())

    return torch.stack(mean_delta).numpy()    

def create_delta_per_distance_chart(mean_delta: np.ndarray, save_path: str = None):
    distances = np.arange(len(mean_delta))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(distances, mean_delta, marker="o")
    ax.set_xlabel("Manhattan Distance from Center")
    ax.set_ylabel("Mean Logit Drop (true class)")
    ax.set_title("Distance-conditioned Occlusion Sensitivity")
    ax.grid(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
        print(f"Chart saved to {save_path}")
    plt.show()
    return fig

def _get_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--output_path", type=str, required=True)
    pre_parser.add_argument("--fill", type=float, default=0.0)
    pre_parser.add_argument("--k_per_bin", type=int, default=4)
    pre_parser.add_argument("--max_batches", type=int, default=16)
    pre_parser.add_argument("--max_d", type=int, default=10)
    pre_parser.add_argument("--anchor_stride", type=int, default=1)
    pre_args, _ = pre_parser.parse_known_args()

    config_yaml = os.path.join(pre_args.output_path, "config.yaml")
    if not os.path.exists(config_yaml):
        raise FileNotFoundError(f"Saved config not found: {config_yaml}")

    args = parse_args([
        "--config", config_yaml,
        "--output_path", pre_args.output_path,
    ])   

    args.fill = pre_args.fill
    args.k_per_bin = pre_args.k_per_bin
    args.max_batches = pre_args.max_batches
    args.max_d = pre_args.max_d
    args.anchor_stride = pre_args.anchor_stride

    resolve_runtime_device(args)

    return args

def main():
    args = _get_args()
    set_seed(args.seed)
    
    wandb.init(mode="disabled")
    model = setup(args)

    ckpt_path = os.path.join(args.output_path, "best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    _load_checkpoint(model, ckpt_path, args.device)

    _, test_loader, _ = get_dataloader(args)

    mean_delta = calculate_mean_delta(args, model, test_loader)

    save_path = os.path.join("./analysis/dis_occ_results/", f"dis_occ_{str(args.model).replace('/', '_')}.png")
    fig = create_delta_per_distance_chart(mean_delta, save_path=save_path)

if __name__ == "__main__":
    main()