import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import _load_weights as load_jax_npz_weights


class GAPViT(nn.Module):
    """
    timm ViT backbone loaded from Google JAX (.npz) weights, switched to GAP head.

    The .npz checkpoints are CLS-token-style; we load them into a standard timm ViT,
    then drop the CLS token and its positional-embedding slot, and replace the
    classifier with a fresh head that consumes mean-pooled patch tokens.
    """

    def __init__(self, npz_path, base_model="vit_base_patch16_224",
                 num_classes=100, drop_path_rate=0.0):
        super().__init__()

        # Build the backbone with num_classes matching the npz head shape so
        # _load_weights does not error on the head/kernel & head/bias tensors.
        # The original head is discarded below.
        backbone = timm.create_model(
            base_model,
            pretrained=True,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
        )
        # load_jax_npz_weights(backbone, str(npz_path))

        # Drop CLS token and align pos_embed to patch positions only.
        # timm's _pos_embed skips the CLS prepend when cls_token is None,
        # so the (now N-token) pos_embed lines up with patch features.
        with torch.no_grad():
            patch_pos = backbone.pos_embed[:, 1:, :].detach().clone().contiguous()
        backbone.cls_token = None
        backbone.pos_embed = nn.Parameter(patch_pos)
        backbone.head = nn.Identity()

        self.backbone = backbone
        self.head = nn.Linear(backbone.embed_dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        gap = feats.mean(dim=1)
        return self.head(gap)


def build_pretrained_vit(args):
    if not args.pretrained_npz:
        raise ValueError("pretrained_npz must be set (path to a Google JAX .npz checkpoint)")
    return GAPViT(
        npz_path=args.pretrained_npz,
        base_model=args.pretrained_base_model,
        num_classes=args.num_classes,
        drop_path_rate=args.drop_path,
    )
