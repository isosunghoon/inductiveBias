import torch
import torch.nn as nn
import torch.nn.functional as F

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

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)x  
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


class ConvAttention(nn.Module):
    def __init__(self, dim, head_dim = 32, window_size = 5, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % head_dim == 0, 'dim should be divisible by head_dim'
        assert window_size % 2 == 1, "Window size must be odd"

        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim ** -0.5
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def make_local_mask(self, H: int, W: int, window_size: int, device=None):
        """
        Implementing 2d CSAN
        Returns bool mask of shape (N, N) where N=H*W
        window_size must be odd: center window around diagonal
        """
        half = window_size//2
        ys = torch.arange(H, device = device)
        xs = torch.arange(W, device = device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij') # [H, W], [H, W]

        coords = torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=1)
        dy = (coords[:, None, 0] - coords[None, :, 0]).abs()
        dx = (coords[:, None, 1] - coords[None, :, 1]).abs()

        # create a square window. Could use taxi distance insteaed
        allowed = (dy <= half) & (dx <= half)
        return allowed

    def forward(self, x):
        shape = x.shape
        if len(shape) == 4:
            B, C, H, W = shape
            N = H * W
            x = torch.flatten(x, start_dim=2).transpose(-2, -1)
        else:
            raise ValueError("convAttention expects [B, C, H, W] input for 2d local attention")
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q*self.scale) @ k.transpose(-2, -1) 

        # apply masking right before softmax operation
        allowed = self.make_local_mask(H, W, self.window_size, x.device)
        attn = attn.masked_fill(~allowed[None, None, :, :], float("-inf")) 

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1,2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.transpose(-2, -1).reshape(B, C, H, W)

        return x