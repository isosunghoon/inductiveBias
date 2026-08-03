"""
Token mixers explored during the study but not included in the final report.

Import them the same way as the reported mixers in models/token_mixers.py; they take
`dim=` and a [B, C, H, W] tensor. To use one, add a branch for it in `setup()` in
train.py (see experimental/README.md).
"""

import torch.nn as nn


class PoolFormer(nn.Module):
    def __init__(self, dim, pool_size=3, stride=1):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=stride, padding=pool_size//2, count_include_pad=False)
    
    def forward(self, x):
        return self.pool(x) - x
