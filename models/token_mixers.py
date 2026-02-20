"""
Token mixers for MetaFormer (Pooling, Attention, SpatialFc, ConvAttention2d).
"""
from typing import Sequence
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer.
    """
    def __init__(self, pool_size=3, **kwargs):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class Attention(nn.Module):
    """
    Attention module that can take tensor with [B, N, C] or [B, C, H, W] as input.
    """
    def __init__(self, dim, head_dim=32, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % head_dim == 0, 'dim should be divisible by head_dim'
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        shape = x.shape
        if len(shape) == 4:
            B, C, H, W = shape
            N = H * W
            x = torch.flatten(x, start_dim=2).transpose(-2, -1)  # (B, N, C)
        else:
            B, N, C = shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if len(shape) == 4:
            x = x.transpose(-2, -1).reshape(B, C, H, W)

        return x


class SpatialFc(nn.Module):
    """
    SpatialFc module that take features with shape of (B, C, *) as input.
    """
    def __init__(self, spatial_shape=(14, 14), **kwargs):
        super().__init__()
        if isinstance(spatial_shape, int):
            spatial_shape = [spatial_shape]
        assert isinstance(spatial_shape, Sequence), \
            f'"spatial_shape" must be a sequence or int, got {type(spatial_shape)}.'
        N = reduce(lambda x, y: x * y, spatial_shape)
        self.fc = nn.Linear(N, N, bias=False)

    def forward(self, x):
        shape = x.shape
        x = torch.flatten(x, start_dim=2)
        x = self.fc(x)
        x = x.reshape(*shape)
        return x


class ConvAttention2d(nn.Module):
    """
    2D Convolutional self-attention (spatially local attention).
    Each position attends only to a kernel_size x kernel_size neighborhood.
    Ref: "Convolutional self-attention networks" (Yang et al., NAACL 2019).
    Compatible with MetaFormer: input/output shape (B, C, H, W).
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0,
                 kernel_size=3, k=1, **kwargs):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        padding = kernel_size // 2

        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_kv = nn.Conv2d(dim, inner_dim * 2, k, stride=k, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )
        self.kernel_size = kernel_size
        self.padding = padding
        self.w = kernel_size ** 2

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.to_q(x)   # (B, inner_dim, H, W)
        kv = self.to_kv(x)
        k, v = kv.chunk(2, dim=1)

        # q: (B, inner_dim, H, W) -> (B, heads, H*W, dim_head)
        q = q.flatten(2).transpose(1, 2).reshape(B, H * W, self.heads, self.dim_head).permute(0, 2, 1, 3)
        # (B, heads, H*W, w) for attention with w neighbors
        q = q.unsqueeze(3).expand(B, self.heads, H * W, self.w, self.dim_head)

        # Unfold k, v: (B, inner_dim, H, W) -> (B, inner_dim*w, H*W)
        k = F.unfold(k, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        v = F.unfold(v, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        # (B, heads, H*W, w, dim_head)
        k = k.reshape(B, self.heads, self.dim_head, self.w, H * W).permute(0, 1, 4, 3, 2)
        v = v.reshape(B, self.heads, self.dim_head, self.w, H * W).permute(0, 1, 4, 3, 2)

        dots = einsum('b h n w d, b h n w d -> b h n w', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = einsum('b h n w, b h n w d -> b h n d', attn, v)
        out = out.transpose(1, 2).reshape(B, H * W, -1).transpose(1, 2).reshape(B, -1, H, W)
        return self.to_out(out)
