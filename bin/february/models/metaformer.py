    # Copyright 2021 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
MetaFormer implementation with single-stage ViT-like structure
"""
from typing import Sequence
import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.layers import to_2tuple

# try:
#     from mmseg.models.builder import BACKBONES as seg_BACKBONES
#     from mmseg.utils import get_root_logger
#     from mmcv.runner import _load_checkpoint
#     has_mmseg = True
# except ImportError:
#     print("If for semantic segmentation, please install mmsegmentation first")
#     has_mmseg = False

# try:
#     from mmdet.models.builder import BACKBONES as det_BACKBONES
#     from mmdet.utils import get_root_logger
#     from mmcv.runner import _load_checkpoint
#     has_mmdet = True
# except ImportError:
#     print("If for detection, please install mmdetection first")
#     has_mmdet = False


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 
        'classifier': 'head',
        **kwargs
    }

class PatchEmbed(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv. 
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """
    def __init__(self, patch_size=16,
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, 
                              stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
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
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x 

class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)

class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, 
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AddPositionEmb(nn.Module):
    """Module to add position embedding to input features
    """
    def __init__(
        self, dim=384, spatial_shape=[14, 14],
        ):
        super().__init__()
        if isinstance(spatial_shape, int):
            spatial_shape = [spatial_shape]
        assert isinstance(spatial_shape, Sequence), \
            f'"spatial_shape" must by a sequence or int, ' \
            f'get {type(spatial_shape)} instead.'
        if len(spatial_shape) == 1:
            embed_shape = list(spatial_shape) + [dim]
        else:
            embed_shape = [dim] + list(spatial_shape)
        self.pos_embed = nn.Parameter(torch.zeros(1, *embed_shape))
    def forward(self, x):
        return x+self.pos_embed


class MetaFormerBlock(nn.Module):
    """
    Implementation of one MetaFormer block.
    --dim: embedding dim
    --token_mixer: token mixer module
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth, 
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale, 
        refer to https://arxiv.org/abs/2103.17239
    """
    def __init__(self, dim, 
                 token_mixer=nn.Identity, 
                 mlp_ratio=4., 
                 act_layer=nn.GELU, norm_layer=LayerNormChannel, 
                 drop=0., drop_path=0., 
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(dim=dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)


    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def basic_blocks(dim, depth, token_mixer=nn.Identity, 
                 mlp_ratio=4., 
                 act_layer=nn.GELU, norm_layer=LayerNormChannel, 
                 drop_rate=.0, drop_path_rate=0., 
                 use_layer_scale=True, layer_scale_init_value=1e-5):
    """
    Generate MetaFormer blocks for single-stage (ViT-like).
    """
    blocks = []
    for block_idx in range(depth):
        block_dpr = drop_path_rate * block_idx / max(depth - 1, 1)
        blocks.append(MetaFormerBlock(
            dim, token_mixer=token_mixer, mlp_ratio=mlp_ratio, 
            act_layer=act_layer, norm_layer=norm_layer, 
            drop=drop_rate, drop_path=block_dpr, 
            use_layer_scale=use_layer_scale, 
            layer_scale_init_value=layer_scale_init_value, 
            ))
    return nn.Sequential(*blocks)


class MetaFormer(nn.Module):
    """
    MetaFormer with ViT-like single-stage structure.
    - Single patch embedding at the beginning.
    - Positional embedding right after patch embed (once). Default True; set add_pos_emb=False to disable. Uses img_size (default 224) to set grid for pos emb.
    - No stages / no downsampling; all blocks use same embed_dim.
    """
    def __init__(self, depth=12, embed_dim=768, 
                 token_mixer=nn.Identity, mlp_ratio=4., 
                 norm_layer=LayerNormChannel, act_layer=nn.GELU, 
                 num_classes=1000,
                 patch_size=16, img_size=224,
                 add_pos_emb=True, 
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5, 
                 **kwargs):

        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # ViT-style: single patch embedding at the beginning
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, 
            in_chans=3, embed_dim=embed_dim, norm_layer=norm_layer)
        # Positional embedding only once, right after patch embed (grid = img_size // patch_size)
        if add_pos_emb is True:
            grid_size = img_size // patch_size
            self.pos_emb = AddPositionEmb(dim=embed_dim, spatial_shape=[grid_size, grid_size])
        elif add_pos_emb is None or add_pos_emb is False:
            self.pos_emb = None
        else:
            self.pos_emb = add_pos_emb() if not isinstance(add_pos_emb, nn.Module) else add_pos_emb

        # Single-stage blocks (no downsampling)
        self.blocks = basic_blocks(
            dim=embed_dim, depth=depth,
            token_mixer=token_mixer, mlp_ratio=mlp_ratio,
            act_layer=act_layer, norm_layer=norm_layer,
            drop_rate=drop_rate, drop_path_rate=drop_path_rate,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(
            embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self.cls_init_weights)

    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        if self.pos_emb is not None:
            x = self.pos_emb(x)
        return x

    def forward_tokens(self, x):
        return self.blocks(x)

    def forward(self, x):
        x = self.forward_embeddings(x)
        x = self.forward_tokens(x)
        x = self.norm(x)
        cls_out = self.head(x.mean([-2, -1]))
        return cls_out