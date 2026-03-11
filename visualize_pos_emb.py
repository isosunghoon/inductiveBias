"""
Position Embedding Cosine Similarity Heatmap
Visualizes pairwise cosine similarity between patch position embeddings,
following the method in "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2021).

Usage:
    python visualize_pos_emb.py --checkpoint ./output/vit/best.pt --config config/vit.yaml
    python visualize_pos_emb.py --checkpoint ./output/vit/best.pt  # uses base.yaml defaults
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from functools import partial

import models.token_mixers as TM
import models.norm_layers as NL
from models.metaformer import MetaFormer
from utils.config import get_parser, _apply_yaml


def build_model(args):
    if args.norm_layer == "identity":
        norm_layer = nn.Identity
    elif args.norm_layer == "layernorm":
        norm_layer = NL.LayerNorm
    else:
        norm_layer = nn.Identity

    act_layer = nn.GELU

    if args.model == "identity":
        token_mixer = nn.Identity
    elif args.model == "vit":
        token_mixer = partial(TM.Attention, head_dim=args.attn_head_dim, qkv_bias=args.attn_qkv_bias,
                              attn_drop=args.attn_drop, proj_drop=args.attn_proj_drop)
    elif args.model == "localvit":
        token_mixer = partial(TM.ConvAttention, head_dim=args.attn_head_dim, window_size=args.window_size,
                              qkv_bias=args.attn_qkv_bias, attn_drop=args.attn_drop, proj_drop=args.attn_proj_drop)
    elif args.model == "mlpmixer":
        token_mixer = partial(TM.MLPMixer, img_size=args.img_size, patch_size=args.patch_size,
                              expansion_factor=args.expansion_factor, mixer_drop=args.mixer_drop)
    elif args.model == "poolformer":
        token_mixer = partial(TM.PoolFormer, pool_size=args.pool_size, stride=args.stride)
    else:
        token_mixer = nn.Identity

    model = MetaFormer(
        depth=args.depth,
        embed_dim=args.embed_dim,
        token_mixer=token_mixer,
        mlp_ratio=args.mlp_ratio,
        norm_layer=norm_layer,
        act_layer=act_layer,
        num_classes=args.num_classes,
        patch_size=args.patch_size,
        img_size=args.img_size,
        add_pos_emb=args.add_pos_emb,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path,
        use_layer_scale=args.use_layer_scale,
        layer_scale_init_value=args.layer_scale_init_value,
    )
    return model


def plot_pos_emb_similarity(pos_embed, grid_h, grid_w, save_path=None):
    """
    pos_embed: numpy array of shape [N, C] where N = grid_h * grid_w
    """
    # Normalize embeddings
    norms = np.linalg.norm(pos_embed, axis=1, keepdims=True)
    emb_norm = pos_embed / (norms + 1e-8)

    # Pairwise cosine similarity [N, N]
    sim = emb_norm @ emb_norm.T  # [N, N]

    # sim[i] -> similarity of patch i to all N patches, reshape to [grid_h, grid_w]
    N = grid_h * grid_w

    fig = plt.figure(figsize=(grid_w * 1.5 + 1.5, grid_h * 1.5 + 1.0))
    fig.patch.set_facecolor("#d0d0d0")

    # GridSpec: grid_h rows x (grid_w + 1) cols, last col for colorbar
    gs = gridspec.GridSpec(
        grid_h, grid_w + 1,
        width_ratios=[1] * grid_w + [0.15],
        hspace=0.05,
        wspace=0.05,
        left=0.08, right=0.92, top=0.90, bottom=0.08,
    )

    im = None
    for idx in range(N):
        row = idx // grid_w
        col = idx % grid_w
        ax = fig.add_subplot(gs[row, col])

        # Similarity map of patch idx to all patches
        sim_map = sim[idx].reshape(grid_h, grid_w)
        im = ax.imshow(sim_map, cmap="viridis", vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])

        # Row labels (left side, centered per row)
        if col == 0:
            ax.set_ylabel(str(row + 1), fontsize=10, rotation=0, labelpad=8, va="center")

        # Column labels (bottom, only last row)
        if row == grid_h - 1:
            ax.set_xlabel(str(col + 1), fontsize=10)

    # Colorbar in the last column spanning all rows
    cbar_ax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(["-1", "0", "1"])
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("Cosine similarity", fontsize=10, labelpad=6)

    # Axis labels
    fig.text(0.50, 0.02, "Input patch column", ha="center", fontsize=11)
    fig.text(0.02, 0.50, "Input patch row", va="center", rotation="vertical", fontsize=11)
    fig.suptitle("Position embedding similarity", fontsize=14, y=0.96)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved to {save_path}")
    else:
        plt.show()


def main():
    # Parse model config args (reuse project's parser)
    parser = get_parser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.pt). If not given, uses random init.")
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save the figure (e.g. pos_emb_sim.png). If not given, shows interactively.")
    args = parser.parse_args()

    # Apply configs
    if args.base_config:
        _apply_yaml(args, args.base_config)
    if args.config:
        _apply_yaml(args, args.config)

    if not args.add_pos_emb:
        print("Warning: add_pos_emb is False for this config. The model has no positional embedding to visualize.")
        return

    # Build model
    model = build_model(args)
    model.eval()

    # Load checkpoint if provided
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        # Support both raw state_dict and {"model": state_dict} wrappers
        state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint provided — visualizing randomly initialized position embeddings.")

    # Extract pos_embed: shape [1, C, H, W]
    pos_embed = model.pos_emb.pos_embed  # nn.Parameter
    _, C, H, W = pos_embed.shape
    print(f"pos_embed shape: {pos_embed.shape}  (grid: {H}x{W}, dim: {C})")

    # [1, C, H, W] -> [H*W, C]
    emb = pos_embed.detach().squeeze(0)           # [C, H, W]
    emb = emb.permute(1, 2, 0).reshape(-1, C)     # [H*W, C]
    emb_np = emb.numpy()

    plot_pos_emb_similarity(emb_np, H, W, save_path=args.save)


if __name__ == "__main__":
    main()
