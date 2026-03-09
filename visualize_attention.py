"""Attention map visualization for MetaFormer models.

Loads a saved checkpoint and visualizes per-block attention maps for a
handful of CIFAR-100 test images.  Works with any attention-based token
mixer (Attention, ConvAttention).  Identity mixers produce no map and are
skipped gracefully.

Usage:
    python visualize_attention.py \\
        --config    config/vit.yaml \\
        --checkpoint output/vit/best.pt \\
        --num_samples 4 \\
        --output_dir  ./attn_vis
"""

import argparse
import os
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from torchvision import transforms
from torchvision.datasets import CIFAR100

import models.norm_layers as NL
import models.token_mixers as TM
from models.metaformer import MetaFormer

# CIFAR-100 normalisation constants
_MEAN = (0.5071, 0.4867, 0.4408)
_STD  = (0.2675, 0.2565, 0.2761)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(cfg: dict) -> MetaFormer:
    """Reconstruct MetaFormer from a config dict loaded from YAML."""
    norm_str = cfg.get("norm_layer", "identity")
    if norm_str == "identity":
        norm_layer = nn.Identity
    elif norm_str == "layernorm":
        norm_layer = NL.LayerNorm
    else:
        raise ValueError(f"Unknown norm_layer: {norm_str!r}")

    act_str = cfg.get("act_layer", "GELU")
    act_layer = nn.GELU if act_str == "GELU" else nn.ReLU

    model_type = cfg.get("model", "vit")
    head_dim    = cfg.get("attn_head_dim", 32)
    qkv_bias    = cfg.get("attn_qkv_bias", False)
    attn_drop   = cfg.get("attn_drop", 0.0)
    proj_drop   = cfg.get("attn_proj_drop", 0.0)
    window_size = cfg.get("window_size", 5)

    if model_type == "identity":
        token_mixer = nn.Identity
    elif model_type == "vit":
        token_mixer = partial(
            TM.Attention,
            head_dim=head_dim, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=proj_drop,
        )
    elif model_type == "local_vit":
        token_mixer = partial(
            TM.ConvAttention,
            head_dim=head_dim, window_size=window_size,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type!r}")

    return MetaFormer(
        depth=cfg.get("depth", 12),
        embed_dim=cfg.get("embed_dim", 384),
        token_mixer=token_mixer,
        mlp_ratio=cfg.get("mlp_ratio", 4.0),
        norm_layer=norm_layer,
        act_layer=act_layer,
        num_classes=cfg.get("num_classes", 100),
        patch_size=cfg.get("patch_size", 16),
        img_size=cfg.get("img_size", 224),
        add_pos_emb=cfg.get("add_pos_emb", True),
        drop_rate=cfg.get("drop_rate", 0.0),
        drop_path_rate=cfg.get("drop_path", 0.0),
        use_layer_scale=cfg.get("use_layer_scale", True),
        layer_scale_init_value=cfg.get("layer_scale_init_value", 1e-5),
    )


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_samples(data_path: str, num_samples: int, img_size: int):
    """Return (images [N,C,H,W], labels [N], class_names list)."""
    tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    dataset = CIFAR100(root=data_path, train=False, download=True, transform=tf)
    images = torch.stack([dataset[i][0] for i in range(num_samples)])
    labels = [dataset[i][1] for i in range(num_samples)]
    return images, labels, dataset.classes


def denormalize(tensor):
    """Normalized tensor → HWC float32 in [0, 1] for matplotlib."""
    mean = torch.tensor(_MEAN).view(3, 1, 1)
    std  = torch.tensor(_STD).view(3, 1, 1)
    img  = (tensor.cpu() * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _attn_to_heatmap(attn_block, sample_idx: int, grid_size: int, patch_size: int):
    """Convert a single block's attn tensor to a (img_size, img_size) heatmap.

    attn_block: [B, num_heads, N, N]
    Strategy: average over heads, then average over query positions → per-patch
    'importance' score, upsample to pixel space via nearest-neighbour repeat.
    """
    attn = attn_block[sample_idx]          # [num_heads, N, N]
    attn = attn.float().cpu()
    attn_mean = attn.mean(0)               # [N, N]  (avg over heads)
    patch_score = attn_mean.mean(0)        # [N]     (avg attention received per patch)
    patch_score = patch_score.reshape(grid_size, grid_size).numpy()
    lo, hi = patch_score.min(), patch_score.max()
    patch_score = (patch_score - lo) / (hi - lo + 1e-8)
    # Upsample to image resolution
    return np.kron(patch_score, np.ones((patch_size, patch_size)))


def visualize_and_save(images, labels, class_names, attn_maps, img_size, patch_size, output_dir, cfg_name):
    """Save one figure per sample showing original image + per-block attention overlays."""
    os.makedirs(output_dir, exist_ok=True)

    grid_size = img_size // patch_size
    valid_blocks = [(i, a) for i, a in enumerate(attn_maps) if a is not None]

    if not valid_blocks:
        print("[visualize] No attention maps found — model may use a non-attention token mixer.")
        return

    B = images.shape[0]
    for s in range(B):
        img_np     = denormalize(images[s])
        label_name = class_names[labels[s]]

        ncols = len(valid_blocks) + 1
        fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))

        # Column 0: original image
        axes[0].imshow(img_np)
        axes[0].set_title(f"Input\n({label_name})")
        axes[0].axis("off")

        for col, (block_idx, attn) in enumerate(valid_blocks, start=1):
            heatmap = _attn_to_heatmap(attn, s, grid_size, patch_size)
            axes[col].imshow(img_np)
            axes[col].imshow(heatmap, cmap="jet", alpha=0.5, vmin=0, vmax=1)
            axes[col].set_title(f"Block {block_idx}")
            axes[col].axis("off")

        fig.suptitle(f"{cfg_name} — sample {s} ({label_name})", fontsize=11)
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"sample_{s:03d}_{label_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Attention map visualization for MetaFormer")
    parser.add_argument("--config",      type=str, required=True,
                        help="Path to training config YAML (e.g. config/vit.yaml)")
    parser.add_argument("--checkpoint",  type=str, required=True,
                        help="Path to saved model state-dict (.pt)")
    parser.add_argument("--num_samples", type=int, default=4,
                        help="Number of CIFAR-100 test images to visualize")
    parser.add_argument("--output_dir",  type=str, default="./attn_vis",
                        help="Directory where PNG figures are saved")
    parser.add_argument("--data_path",   type=str, default=None,
                        help="Override data root from config (default: use config value)")
    parser.add_argument("--device",      type=str, default=None,
                        help="Compute device (cpu / cuda). Auto-detected if not set.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device    = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_path = args.data_path or cfg.get("data_path", "./data")
    img_size  = cfg.get("img_size", 32)
    patch_size = cfg.get("patch_size", 2)
    cfg_name  = os.path.splitext(os.path.basename(args.config))[0]

    # Build model and load weights
    print(f"Building model from: {args.config}")
    model = build_model(cfg)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt)
    model.to(device).eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Load test samples
    print(f"Loading {args.num_samples} CIFAR-100 test samples...")
    images, labels, class_names = load_samples(data_path, args.num_samples, img_size)
    images = images.to(device)

    # Inference — return_attn=True collects maps from every block
    print("Running inference with attention map extraction...")
    with torch.no_grad():
        logits, attn_maps = model(images, return_attn=True)

    preds = logits.argmax(dim=1).cpu().tolist()
    for i, (pred, label) in enumerate(zip(preds, labels)):
        status = "OK" if pred == label else "WRONG"
        print(f"  [{status}] sample {i}: label={class_names[label]}, pred={class_names[pred]}")

    # Visualize
    print(f"\nSaving visualizations to: {args.output_dir}")
    visualize_and_save(images.cpu(), labels, class_names, attn_maps,
                       img_size, patch_size, args.output_dir, cfg_name)
    print("Done.")


if __name__ == "__main__":
    main()
