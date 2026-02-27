import argparse
import yaml
import os

def get_parser():
    parser = argparse.ArgumentParser()
    # config
    parser.add_argument("--config", type=str, default=None, help="path to config file")

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
    parser.add_argument("--use_layer_scale", action="store_true", default=True, help="use layer scale")
    parser.add_argument("--layer_scale_init_value", type=float, default=1e-5, help="layer scale initial value")
    
    # train 하이퍼파라미터
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=128, help="batch size for training")
    parser.add_argument("--test_batch_size", type=int, default=256, help="batch size for evaluation")

    parser.add_argument("--learning_rate", type=float, default=0.1, help="optimizer learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay (L2 regularization)")
    parser.add_argument("--decay_type", default="cosine", help="lr decay type (cosine or linear)")
    parser.add_argument("--warmup_epochs", type=int, default=1, help="number of warmup epochs")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="max gradient norm for clipping")
    
    # validation 관련
    parser.add_argument("--eval_interval", type=int, default=5, help="run validation every N epochs")
    parser.add_argument("--save_best", action="store_true", help="save best model by val acc")
    parser.add_argument("--output_path", type=str, default="./out", help="output path")

    # 기타
    parser.add_argument("--num_classes", type=int, default=100, help="number of classes")
    parser.add_argument("--num_workers", type=int, default=4, help="number of data loading workers")
    parser.add_argument("--log_interval", type=int, default=50, help="steps between logging/progress updates")
    parser.add_argument("--fp16", action="store_true", help="enable mixed precision training (AMP fp16)")
    parser.add_argument("--data_path", type=str, default="./data", help="path to dataset root directory")

    # token-mixer specific
    # attention
    parser.add_argument("--attn_head_dim", type=int, default=32, help="per-head dimension for Attention mixer")
    parser.add_argument("--attn_qkv_bias", action="store_true", help="use bias in qkv projection")
    parser.add_argument("--attn_drop", type=float, default=0.0, help="attention dropout rate")
    parser.add_argument("--attn_proj_drop", type=float, default=0.0, help="output projection dropout rate")

    return parser

    
def parse_args():
    parser = get_parser()
    args = parser.parse_args()

    if args.config is not None:
        assert os.path.exists(args.config), f"Config not found: {args.config}"
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)

    if args.config is not None:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)

        for k, v in cfg.items():
            if not hasattr(args, k):
                raise ValueError(f"Unknown config key: {k}")

            old_val = getattr(args, k)
            if old_val is not None:
                v = type(old_val)(v)

            setattr(args, k, v)

    return args