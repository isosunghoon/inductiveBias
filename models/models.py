"""
Model definitions: MetaFormer and registered model variants.
"""
from functools import partial
import torch.nn as nn

from timm.models.registry import register_model

from .metaformer import (
    MetaFormer,
    LayerNormChannel,
    GroupNorm,
    _cfg,
)
from .token_mixers import Pooling, Attention, SpatialFc


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
