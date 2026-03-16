import argparse
import yaml
import os

def get_parser():
    parser = argparse.ArgumentParser()
    # config
    parser.add_argument("--base_config", type=str, default="./config/base.yaml",
                        help="path to base config yaml (shared defaults, e.g. config/base.yaml)")
    parser.add_argument("--config", type=str, default=None,
                        help="path to model-specific config yaml (e.g. config/model_vit.yaml)")

    # 기본
    parser.add_argument('--seed', type=int, default=67, help="random seed for initialization") #six-seven
    parser.add_argument("--dataset",  default="cifar100", help="dataset for training")
    parser.add_argument("--model", default="identity", help="model type")

    # model 구조 관련
    parser.add_argument("--norm_layer", default="identity", help="normalization layer")
    parser.add_argument("--act_layer", default="GELU", help="activation layer")

    parser.add_argument("--depth", type=int, default=12, help="number of MetaFormer blocks")
    parser.add_argument("--embed_dim", type=int, default=384, help="embedding dimension")
    parser.add_argument("--mlp_ratio", type=float, default=4.0, help="MLP hidden dimension ratio")
    parser.add_argument("--patch_size", type=int, default=2, help="patch size")
    parser.add_argument("--img_size", type=int, default=32, help="input image size")

    parser.add_argument("--drop_rate", type=float, default=0.0, help="dropout rate")
    parser.add_argument("--drop_path", type=float, default=0.0, help="drop path rate")
    parser.add_argument("--add_pos_emb", action="store_true", help="add positional embedding")
    parser.add_argument("--use_layer_scale", action="store_true", help="use layer scale")
    parser.add_argument("--layer_scale_init_value", type=float, default=1e-5, help="layer scale initial value")
    
    # train 하이퍼파라미터
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=128, help="batch size for training")
    parser.add_argument("--test_batch_size", type=int, default=256, help="batch size for evaluation")

    parser.add_argument("--optimizer", type=str, default="sgd", help="optimizer type")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="optimizer learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay (L2 regularization)")
    parser.add_argument("--decay_type", default="cosine", help="lr decay type (cosine or linear)")
    parser.add_argument("--warmup_epochs", type=int, default=1, help="number of warmup epochs")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="max gradient norm for clipping")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="label smoothing factor for cross entropy (0.0 = disabled)")
    parser.add_argument("--augment", type=str, default="none",
                        choices=["none", "weak", "strong"],
                        help="augmentation level: none | weak (crop+flip) | strong (weak + RandAugment + RandomErasing + Mixup + CutMix + RepeatedAug)")

    # validation 관련
    parser.add_argument("--eval_interval", type=int, default=5, help="run validation every N epochs")
    parser.add_argument("--save_best", action="store_true", help="save best model by val acc")
    parser.add_argument("--output_path", type=str, default="./output", help="output path")

    # 기타
    parser.add_argument("--num_classes", type=int, default=100, help="number of classes")
    parser.add_argument("--num_workers", type=int, default=4, help="number of data loading workers")
    parser.add_argument("--log_interval", type=int, default=50, help="steps between logging/progress updates")
    parser.add_argument("--fp16", action="store_true", help="enable mixed precision training (AMP fp16)")
    parser.add_argument("--data_path", type=str, default="./data", help="path to dataset root directory")
    parser.add_argument("--no_wandb", action="store_true", help="disable Weights & Biases logging")
    parser.add_argument("--project", type=str, default="exp1", help="W&B project name")
    parser.add_argument("--run_name", type=str, default="XXXXX", help="W&B run name")

    # token-mixer specific
    # attention
    parser.add_argument("--attn_head_dim", type=int, default=32, help="per-head dimension for Attention mixer")
    parser.add_argument("--window_size", type=int, default=5, help="local attention window size (odd)")
    parser.add_argument("--attn_qkv_bias", action="store_true", help="use bias in qkv projection")
    parser.add_argument("--attn_drop", type=float, default=0.0, help="attention dropout rate")
    parser.add_argument("--attn_proj_drop", type=float, default=0.0, help="output projection dropout rate")
    
    # MLP-Mixer
    parser.add_argument("--expansion_factor", type=int, default=2, help="hidden dimension expansion rate")
    parser.add_argument("--mixer_drop", type=float, default=0.5, help="mlp mixer layer drop rate")

    # PoolFormer
    parser.add_argument("--pool_size", type=int, default=3, help="pooling size for PoolFormer")
    parser.add_argument("--stride", type=int, default=1, help="stride for PoolFormer")

    # ConvFormer
    parser.add_argument("--kernel_size", type=int, default=3, help="kernel size for ConvFormer depthwise convolution")
    # parser.add_argument("--stride", type=int, default=1, help="stride for ConvFormer depthwise convolution")
    return parser

    
def _apply_yaml(args, path):
    """Load a yaml file and apply its values onto args. Raises on unknown keys."""
    assert os.path.exists(path), f"Config not found: {path}"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k, v in cfg.items():
        if not hasattr(args, k):
            raise ValueError(f"Unknown config key '{k}' in {path}")
        old_val = getattr(args, k)
        if old_val is not None:
            v = type(old_val)(v)
        setattr(args, k, v)


def parse_args():
    # Parse only config paths first so we know which YAML files to load.
    config_path_parser = argparse.ArgumentParser(add_help=False)
    config_path_parser.add_argument("--base_config", type=str, default="./config/base.yaml")
    config_path_parser.add_argument("--config", type=str, default=None)
    config_paths, _ = config_path_parser.parse_known_args()

    parser = get_parser()

    # Start from argparse defaults.
    defaults = parser.parse_args([])

    # Apply YAML defaults (base first, then model-specific override).
    if config_paths.base_config is not None:
        _apply_yaml(defaults, config_paths.base_config)
    if config_paths.config is not None:
        _apply_yaml(defaults, config_paths.config)

    # Re-parse real CLI last so CLI has highest priority.
    parser.set_defaults(**vars(defaults))
    args = parser.parse_args()
    return args
