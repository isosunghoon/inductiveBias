"""
Model definitions: MetaFormer and registered model variants.
"""
from functools import partial
import torch.nn as nn

from timm.models.registry import register_model

from .metaformer import (
    MetaFormer,
    AddPositionEmb,
    LayerNormChannel,
    GroupNorm,
    _cfg,
)
from .token_mixers import Pooling, Attention, SpatialFc, ConvAttention2d


# ViT-style single-stage model factories (224x224, patch_size=16 -> 14x14 grid)


@register_model
def metaformer_id_s12(**kwargs):
    """12 blocks, Identity token mixer."""
    return MetaFormer(
        depth=12, embed_dim=768,
        token_mixer=nn.Identity, mlp_ratio=4.,
        norm_layer=GroupNorm, patch_size=16,
        **kwargs)


@register_model
def metaformer_pppa_s12_224(**kwargs):
    """12 blocks, Attention + pos_emb (ViT-like)."""
    return MetaFormer(
        depth=12, embed_dim=768,
        token_mixer=Attention, mlp_ratio=4.,
        norm_layer=LayerNormChannel, patch_size=16,
        **kwargs)


@register_model
def metaformer_pooling_s12(**kwargs):
    """12 blocks, Pooling token mixer."""
    return MetaFormer(
        depth=12, embed_dim=768,
        token_mixer=Pooling, mlp_ratio=4.,
        norm_layer=GroupNorm, patch_size=16,
        **kwargs)


@register_model
def metaformer_spatialfc_s12_224(**kwargs):
    """12 blocks, SpatialFc token mixer (14x14)."""
    return MetaFormer(
        depth=12, embed_dim=768,
        token_mixer=partial(SpatialFc, spatial_shape=[14, 14]),
        mlp_ratio=4., norm_layer=GroupNorm, patch_size=16,
        **kwargs)


# ---------------------------------------------------------------------------
# CIFAR-100 models (32x32 input, patch_size=4 -> 8x8 grid, num_classes=100)
# Usage: --model metaformer_cifar100_pooling_s6 --num-classes 100 (or omit; default 100)
# ---------------------------------------------------------------------------

# Default kwargs for all CIFAR-100 variants
CIFAR100_KW = dict(
    num_classes=100,
    img_size=32,
    patch_size=4,
)

# Names of CIFAR-100 models (for train.py to set num_classes when needed)
CIFAR100_MODEL_NAMES = [
    "metaformer_cifar100_id_s6",
    "metaformer_cifar100_pooling_s6",
    "metaformer_cifar100_attention_s6",
    "metaformer_cifar100_spatialfc_s6",
    "metaformer_cifar100_convit_s6",
]


@register_model
def metaformer_cifar100_id_s6(**kwargs):
    """CIFAR-100: 6 blocks, Identity token mixer."""
    merged = {**CIFAR100_KW, **kwargs}
    return MetaFormer(
        depth=6, embed_dim=384,
        token_mixer=nn.Identity, mlp_ratio=4.,
        norm_layer=GroupNorm, **merged)


@register_model
def metaformer_cifar100_pooling_s6(**kwargs):
    """CIFAR-100: 6 blocks, Pooling token mixer."""
    merged = {**CIFAR100_KW, **kwargs}
    return MetaFormer(
        depth=6, embed_dim=384,
        token_mixer=Pooling, mlp_ratio=4.,
        norm_layer=GroupNorm, **merged)


@register_model
def metaformer_cifar100_attention_s6(**kwargs):
    """CIFAR-100: 6 blocks, Attention token mixer (8x8 pos_emb)."""
    merged = {**CIFAR100_KW, **kwargs}
    return MetaFormer(
        depth=6, embed_dim=384,
        token_mixer=Attention, mlp_ratio=4.,
        norm_layer=LayerNormChannel,
        add_pos_emb=partial(AddPositionEmb, dim=384, spatial_shape=[8, 8]),
        **merged)


@register_model
def metaformer_cifar100_spatialfc_s6(**kwargs):
    """CIFAR-100: 6 blocks, SpatialFc token mixer (8x8)."""
    merged = {**CIFAR100_KW, **kwargs}
    return MetaFormer(
        depth=6, embed_dim=384,
        token_mixer=partial(SpatialFc, spatial_shape=[8, 8]),
        mlp_ratio=4., norm_layer=GroupNorm, **merged)


@register_model
def metaformer_cifar100_convit_s6(**kwargs):
    """CIFAR-100: 6 blocks, ConvAttention2d token mixer (local 2D conv attention, 8x8 grid)."""
    merged = {**CIFAR100_KW, **kwargs}
    return MetaFormer(
        depth=6, embed_dim=384,
        token_mixer=partial(ConvAttention2d, heads=6, dim_head=64, kernel_size=3, k=1),
        mlp_ratio=4., norm_layer=LayerNormChannel,
        add_pos_emb=partial(AddPositionEmb, dim=384, spatial_shape=[8, 8]),
        **merged)
