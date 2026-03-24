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

        self._cached_mask = None
        self._cached_hw = None

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

        # create a square window. Could use taxi distance instead
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
        if self._cached_mask is None or self._cached_hw != (H, W) or self._cached_mask.device != x.device:
            self._cached_mask = self.make_local_mask(H, W, self.window_size, x.device)
            self._cached_hw = (H, W)
        attn = attn.masked_fill(~self._cached_mask[None, None, :, :], float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1,2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.transpose(-2, -1).reshape(B, C, H, W)

        return x

# MLP-Mixer Implementation
class DenseFormer(nn.Module):
    def __init__(self, dim, img_size=32, patch_size=2, expansion_factor=2, mixer_drop=0.5):
        # num_patches = N
        # dim = C
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        num_features = (img_size // patch_size) ** 2
        self.dim = dim
        self.expansion_factor = expansion_factor
        num_hidden = int(num_features * expansion_factor)
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.dropout1 = nn.Dropout(mixer_drop)
        self.fc2 = nn.Linear(num_hidden, num_features)
        self.dropout2 = nn.Dropout(mixer_drop)

    def forward(self, x):
        shape = x.shape
        if len(shape) == 4:
            B, C, H, W = shape
            x = torch.flatten(x, start_dim=2).transpose(-2, -1)
        else:
            B, N, C = shape
        x = x.transpose(-2, -1)
        x = self.dropout1(F.gelu(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        x = x.transpose(-2, -1)
        if len(shape) == 4:
            x = x.transpose(-2, -1).reshape(B, C, H, W)
        return x

# poolFormer Implementation
class PoolFormer(nn.Module):
    def __init__(self, dim, pool_size=3, stride=1):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=stride, padding=pool_size//2, count_include_pad=False)
    
    def forward(self, x):
        return self.pool(x) - x

# implement convformer
class ConvFormer(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, groups=192):
        super().__init__()
        if dim % groups != 0:
            raise ValueError(f"ConvFormer requires dim ({dim}) to be divisible by groups ({groups}).")
        self.conv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups,
        )

    def forward(self, x):
        return self.conv(x)

# # resmlp Implementation
# class AffineTransform(nn.Module):
#     def __init__(self, num_features):
#         super().__init__()
#         self.alpha = nn.Parameter(torch.ones(num_features))
#         self.beta = nn.Parameter(torch.zeros(num_features))

#     def forward(self, x):
#         return self.alpha.view(1,1,-1)*x + self.beta.view(1,1,-1)

# class CommunicationLayer(nn.Module):
#     def __init__(self, num_features, num_patches):
#         super().__init__()
#         self.aff1 = AffineTransform(num_features)
#         self.fc1 = nn.Linear(num_patches, num_patches)
#         self.aff2 = AffineTransform(num_features)
    
#     def forward(self, x):
#         residual = x
#         x = self.aff1(x)
#         x = self.fc1(x.transpose(1,2)).transpose(1,2)
#         x = self.aff2(x)
#         out = x + residual
#         return out

# class FeedForward(nn.Module):
#     def __init__(self, num_features, expansion_factor):
#         super().__init__()
#         num_hidden = num_features * expansion_factor
#         self.aff1 = AffineTransform(num_features)
#         self.fc1 = nn.Linear(num_features, num_hidden)
#         self.fc2 = nn.Linear(num_hidden, num_features)
#         self.aff2 = AffineTransform(num_features)

#     def forward(self, x):
#         x = self.aff1(x)
#         residual = x
#         x = self.fc1(x)
#         x = F.gelu(x)
#         x = self.fc2(x)
#         x = self.aff2(x)
#         out = x + residual
#         return out

# class ResMLP(nn.Module):
#     def __init__(self, dim, img_size=32, patch_size=4, expansion_factor=4):
#         super().__init__()
#         assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
#         num_patches = img_size // patch_size
#         num_patches = num_patches*num_patches
#         num_features = dim

#         self.cl = CommunicationLayer(num_features, num_patches)
#         self.ff = FeedForward(num_features, expansion_factor)

#     def forward(self, x):
#         B, C, H, W = x.shape
#         N = H * W
#         x = torch.flatten(x, start_dim=2).transpose(-2, -1)  # (B, N, C)

#         x = self.cl(x)
#         x = self.ff(x)

#         x = x.transpose(-2, -1).reshape(B, C, H, W)
#         return x

# class MLPMixer(nn.Module):
#     def __init__(self, dim, img_size=32, patch_size=2, expansion_factor=2, mixer_drop=0.5):
#         # num_patches = N
#         # dim = C
#         super().__init__()
#         assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
#         num_patches = (img_size // patch_size) ** 2
#         self.dim = dim
#         self.expansion_factor = expansion_factor
#         self.mixer_drop = mixer_drop
#         self.token_mixer = TokenMixer(num_patches, dim, self.expansion_factor, self.mixer_drop)
#         self.channel_mixer = ChannelMixer(num_patches, dim, self.expansion_factor, self.mixer_drop)

#     def forward(self, x):
#         shape = x.shape
#         if len(shape) == 4:
#             B, C, H, W = shape
#             N = H * W
#             x = torch.flatten(x, start_dim=2).transpose(-2, -1)
#         else:
#             B, N, C = shape
#         x = self.token_mixer(x)
#         x = self.channel_mixer(x)

#         if len(shape) == 4:
#             x = x.transpose(-2, -1).reshape(B, C, H, W)
#         return x

# class ChannelMixer(nn.Module):
#     def __init__(self, num_patches, num_features, expansion_factor, mixer_drop):
#         super().__init__()
#         self.norm = nn.LayerNorm(num_features)
#         self.mlp = MLP(num_features, expansion_factor, mixer_drop)
    
#     def forward(self, x): 
#         # x.shape = (B, N, C), 서로 다른 channel을 섞음: C->C
#         residual = x
#         x = self.norm(x)
#         x = self.mlp(x)
#         out = x + residual
#         return out