import torch.nn as nn


class Mlp(nn.Module):
    """
    MLP channel mixer with 1x1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = int(dim * mlp_ratio)
        self.fc1 = nn.Conv2d(dim, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, dim, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SwiGLU(nn.Module):
    """
    SwiGLU channel mixer with 1x1 convolutions (NCHW).
    Formula: down_proj( SiLU(gate_proj(x)) * up_proj(x) )
    SiLU is fixed (standard for SwiGLU); no configurable act_layer.
    """
    def __init__(self, dim, mlp_ratio=8 / 3, drop=0.):
        super().__init__()
        hidden_features = int(dim * mlp_ratio)
        self.fc1 = nn.Conv2d(dim, hidden_features, 1)
        self.fc2 = nn.Conv2d(dim, hidden_features, 1)
        self.act = nn.SiLU()
        self.fc3 = nn.Conv2d(hidden_features, dim, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        gate = self.act(self.fc1(x))
        up = self.fc2(x)
        x = gate * up
        x = self.drop(x)
        x = self.fc3(x)
        x = self.drop(x)
        return x