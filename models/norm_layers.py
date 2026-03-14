import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    Normalizes over C for each spatial position independently.
    """
    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x


class BatchNorm(nn.Module):
    """
    BatchNorm for [B, C, H, W] tensors.
    Thin wrapper around nn.BatchNorm2d.
    Normalizes over (B, H, W) per channel — batch-dependent, spatial-invariant.
    """
    def __init__(self, num_channels, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_channels, eps=eps, momentum=momentum)

    def forward(self, x):
        return self.bn(x)


class GroupNorm(nn.Module):
    """
    GroupNorm for [B, C, H, W] tensors.
    Thin wrapper around nn.GroupNorm.
    Normalizes over (C // num_groups, H, W) per group — batch-independent.
    num_channels must be divisible by num_groups (default 32).
    """
    def __init__(self, num_channels, num_groups=32, eps=1e-5):
        super().__init__()
        assert num_channels % num_groups == 0, (
            f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})"
        )
        self.gn = nn.GroupNorm(num_groups, num_channels, eps=eps)

    def forward(self, x):
        return self.gn(x)