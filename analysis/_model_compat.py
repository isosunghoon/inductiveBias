"""Compatibility helpers for analyses over MetaFormer-style and ViT-style models."""

import math
import types

import numpy as np
import torch
from torch import nn


_MODEL_DISPLAY_NAMES = {
    "convformer": "Convformer",
    "localvit": "local ViT",
    "denseformer": "MLP mixer",
    "vit": "ViT",
    "pretrained_vit": "Pretrained ViT",
}


def _unwrap_compile(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def _is_gapvit(model: nn.Module) -> bool:
    try:
        from models.pretrained_vit import GAPViT
    except ImportError:
        return False
    return isinstance(model, GAPViT)


def _seq_to_grid(t: torch.Tensor) -> torch.Tensor:
    """Convert a square token sequence of shape (B, N, C) to (B, C, H, H)."""
    assert t.ndim == 3, f"expected a 3D tensor, got shape {tuple(t.shape)}"
    b, n, c = t.shape
    h = int(math.sqrt(n))
    assert h * h == n, f"token count must be a square, got N={n}"
    return t.transpose(1, 2).contiguous().reshape(b, c, h, h)


def _metaformer_erf_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = self.forward_embeddings(x)
    x = self.forward_tokens(x)
    x = self.norm(x)
    return x


def _gapvit_erf_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = self.backbone.forward_features(x)
    return _seq_to_grid(x)


def patch_erf_forward(model: nn.Module) -> nn.Module:
    """Patch a model forward to return a 4D feature map for input-to-output ERF."""
    raw_model = _unwrap_compile(model)
    if _is_gapvit(raw_model):
        raw_model.forward = types.MethodType(_gapvit_erf_forward, raw_model)
    else:
        raw_model.forward = types.MethodType(_metaformer_erf_forward, raw_model)
    return raw_model


def get_analysis_layers(model: nn.Module) -> list[nn.Module]:
    """Return hookable analysis layers as [patch_embed, block_0, block_1, ...]."""
    raw_model = _unwrap_compile(model)
    if _is_gapvit(raw_model):
        return [raw_model.backbone.patch_embed] + list(raw_model.backbone.blocks)
    return [raw_model.patch_embed] + list(raw_model.blocks)


def pick_anchor_target(activation: torch.Tensor, anchor: tuple[int, int]) -> torch.Tensor:
    """Build a scalar ReLU-sum target at anchor (y, x) from 3D or 4D activations."""
    y, x = anchor
    if activation.ndim == 4:
        return torch.nn.functional.relu(activation[:, :, y, x]).sum()
    if activation.ndim == 3:
        _, n, _ = activation.shape
        h = int(math.sqrt(n))
        assert h * h == n, f"token count must be a square, got N={n}"
        token_idx = y * h + x
        return torch.nn.functional.relu(activation[:, token_idx, :]).sum()
    raise ValueError(f"expected a 3D or 4D activation, got shape {tuple(activation.shape)}")


def aggregate_grad_to_2d(grad: torch.Tensor) -> np.ndarray:
    """Aggregate a 3D token or 4D feature gradient into a nonnegative 2D map."""
    if grad.ndim == 3:
        grad = _seq_to_grid(grad)
    if grad.ndim == 4:
        aggregated = torch.nn.functional.relu(grad).sum(dim=(0, 1))
        return aggregated.detach().cpu().numpy()
    raise ValueError(f"expected a 3D or 4D gradient, got shape {tuple(grad.shape)}")


def display_name(model_arg_name: str) -> str:
    """Return a stable display name for a model CLI name."""
    return _MODEL_DISPLAY_NAMES.get(model_arg_name, model_arg_name)
