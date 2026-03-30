import os
import argparse
import torch
import wandb
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils.config import parse_args
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


def grid_distance_bins(H, W, anchor_y, anchor_x):
    """
    Given the shape [H, W] of the image and the anchor_y and anchor_x,
    return the distance_bins
    Measure the distance using Manhattan Distance
    TODO Improve efficiency of the for loop; don't need iteration of max_d+1
    """
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    dist = (yy - anchor_y).abs() + (xx - anchor_x).abs()
    bins = []
    max_d = int(dist.max().item())

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
        py, px = pid // W, pid % W
        out[:, :, py, px] = fill

    return out


@torch.no_grad()
def calculate_delta_given_archor(model, x, y, anchor_x, anchor_y, k_per_bin: int = 4, fill: float = 0.0):
    """
    Given model and x,y from batch,
    Return delta [B, num_bins] = logit drop for true class per distance bin
    TODO Choose multiple anchor_x, anchor_y and repeat
    If we have a bin that has a size less than k_per_bins, then what?
    """
    model.eval()
    logits = model(x)

    B, C, H, W = x.shape

    true_logit = logits.gather(1, y.view(-1, 1)).squeeze(1)


    bins = grid_distance_bins(H, W, anchor_x, anchor_y)

    deltas = []

    for ids in bins:
        if ids.numel() == 0:
            continue
        sel = ids[torch.randperm(ids.numel(), device=ids.device)[:min(k_per_bin, ids.numel())]].cpu()
        x_occ = occlude_patches(x, sel, fill=fill)

        logits_occ = model(x_occ)
        true_logit_occ = logits_occ.gather(1, y.view(-1, 1)).squeeze(1)
        deltas.append((true_logit - true_logit_occ).unsqueeze(1))

    return torch.cat(deltas, dim=1)  # [B, num_bins_kept]


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


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--output_path", type=str, required=True)
    pre_args, _ = pre_parser.parse_known_args()

    config_yaml = os.path.join(pre_args.output_path, "config.yaml")
    if not os.path.exists(config_yaml):
        raise FileNotFoundError(f"Saved config not found: {config_yaml}")

    args = parse_args(["--config", config_yaml, "--output_path", pre_args.output_path])
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    
    wandb.init(mode="disabled")
    model = setup(args)

    ckpt_path = os.path.join(args.output_path, "best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    _load_checkpoint(model, ckpt_path, args.device)

    _, test_loader, _ = get_dataloader(args)

    
    all_deltas = []
    for x, y in tqdm(test_loader, desc="Distance-conditioned occlusion"):
        x = x.to(args.device)
        y = y.to(args.device)
        delta = distance_conditioned_occlusion(model, x, y)  # [B, num_bins]
        all_deltas.append(delta.cpu())

    all_deltas = torch.cat(all_deltas, dim=0)  # [N, num_bins]
    mean_delta = all_deltas.mean(dim=0).numpy()  # [num_bins]

    save_path = os.path.join("./analysis/dis_occ_results/", f"dis_occ_{str(args.model).replace('/', '_')}.png")
    fig = create_delta_per_distance_chart(mean_delta, save_path=save_path)
    return fig


if __name__ == "__main__":
    main()