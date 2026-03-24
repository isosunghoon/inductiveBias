import torch
import random
import numpy as np
import torch.nn as nn
import yaml

from models.metaformer import MetaFormer
import models.token_mixers as TM
import models.channel_mixers as CM
import models.norm_layers as NL
import models.resnet as RN

from utils.config import parse_args
from utils.dataset import get_dataloader

from tqdm import tqdm
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from functools import partial

import os
import warnings
import datetime

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torch.optim.lr_scheduler"
)

import wandb

def set_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

def setup(args):
    if not args.no_wandb:
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")

    if args.is_metaformer:
        if args.norm_layer == 'identity':
            args.norm_layer = nn.Identity
        elif args.norm_layer == 'layernorm':
            args.norm_layer = NL.LayerNorm
        elif args.norm_layer == 'batchnorm':
            args.norm_layer = NL.BatchNorm
        elif args.norm_layer == 'groupnorm':
            args.norm_layer = NL.GroupNorm
        elif args.norm_layer == 'rmsnorm':
            args.norm_layer = NL.RMSNorm

        if args.act_layer == 'GELU':
            args.act_layer = nn.GELU

        if args.model == "identity":
            args.token_mixer = nn.Identity
        elif args.model == "vit":
            args.token_mixer = partial(TM.Attention, head_dim=args.attn_head_dim, qkv_bias=args.attn_qkv_bias,
                                attn_drop=args.attn_drop, proj_drop=args.attn_proj_drop,)
        elif args.model == "localvit":
            args.token_mixer = partial(TM.ConvAttention, head_dim=args.attn_head_dim, window_size=args.window_size, 
                                qkv_bias=args.attn_qkv_bias, attn_drop=args.attn_drop, proj_drop=args.attn_proj_drop,)
        elif args.model == "mlpmixer":
            args.token_mixer = partial(TM.MLPMixer, img_size=args.img_size, patch_size=args.patch_size,
                                        expansion_factor=args.expansion_factor, mixer_drop=args.mixer_drop,)                               
        elif args.model == "poolformer":
            args.token_mixer = partial(TM.PoolFormer, pool_size=args.pool_size, stride=args.stride)
        elif args.model == "convformer":
            args.token_mixer = partial(TM.ConvFormer, kernel_size=args.kernel_size, stride=args.stride, groups=args.conv_groups)
        elif args.model == "convformer2":
            args.token_mixer = TM.ConvFormer2
        elif args.model == "denseformer":
            args.token_mixer = partial(TM.DenseFormer, img_size=args.img_size, patch_size=args.patch_size,
                                        expansion_factor=args.expansion_factor, mixer_drop=args.mixer_drop,)    
        elif args.model == "resmlp":
            args.token_mixer = partial(TM.ResMLP, img_size=args.img_size, patch_size=args.patch_size,
                                        expansion_factor=args.expansion_factor)

        # Channel Mixers
        if args.channel_mixer == "mlp":
            args.channel_mixer = partial(CM.Mlp, mlp_ratio=args.mlp_ratio, act_layer=args.act_layer, drop=args.drop_rate)
        elif args.channel_mixer == "swiglu":
            args.channel_mixer = partial(CM.SwiGLU, mlp_ratio=args.mlp_ratio, drop=args.drop_rate)

        model = MetaFormer(depth=args.depth, embed_dim=args.embed_dim, token_mixer=args.token_mixer,
                    channel_mixer=args.channel_mixer,
                    norm_layer=args.norm_layer, num_classes=args.num_classes, patch_size=args.patch_size,
                    img_size=args.img_size, add_pos_emb=args.add_pos_emb, drop_path_rate=args.drop_path,
                    use_layer_scale=args.use_layer_scale, layer_scale_init_value=args.layer_scale_init_value)
    else:
        if args.model =='resnet18':
            model = RN.build_resnet(args)

    model.to(args.device)
    return torch.compile(model)

def train(args, model, run=None):
    # prepare dataset
    train_loader, test_loader, mixup_fn = get_dataloader(args)

    # prepare optimizer & scheduler
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,)
    
    if args.decay_type == "cosine":
        warmup_scheduler = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=args.warmup_epochs)
        main_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[args.warmup_epochs])
    else:
        scheduler = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=args.epochs)

    scaler = None
    if args.fp16:
        scaler = torch.amp.GradScaler('cuda')

    model.zero_grad()
    best_acc = 0
    global_step = 0

    # train loop
    for epoch in range(1, args.epochs+1):
        model.train()
        epoch_iterator = tqdm(train_loader, desc="Training (X / X epochs) (loss=X.X)",
                                bar_format="{l_bar}{r_bar}", dynamic_ncols=True)

        running_loss = torch.zeros((), device=args.device)
        for step, batch in enumerate(epoch_iterator):
            x, y = batch
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)

            if mixup_fn is not None:
                x, y = mixup_fn(x, y)

            with torch.amp.autocast('cuda', enabled=args.fp16):
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits, y, label_smoothing=args.label_smoothing)

            running_loss += loss.detach()
            global_step += 1

            if args.fp16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            optimizer.zero_grad()

            if step % args.log_interval == 0:
                avg_loss = (running_loss / (step + 1)).item()
                running_loss.zero_()
                epoch_iterator.set_description(f"Training ({epoch} / {args.epochs} epochs) (loss={avg_loss:.5f})")

        scheduler.step()

        # Save checkpoint right after warmup ends
        if epoch == args.warmup_epochs:
            os.makedirs(args.output_path, exist_ok=True)
            warmup_ckpt_path = os.path.join(args.output_path, "epoch_warmup_end.pt")
            torch.save(model.state_dict(), warmup_ckpt_path)
            print(f"[Checkpoint] warmup-end model saved to {warmup_ckpt_path}")

        # Save model at every n*100 epochs (100, 200, 300, ...)
        if epoch % 100 == 0:
            os.makedirs(args.output_path, exist_ok=True)
            epoch_ckpt_path = os.path.join(args.output_path, f"epoch_{epoch}.pt")
            torch.save(model.state_dict(), epoch_ckpt_path)
            print(f"[Checkpoint] epoch-{epoch} model saved to {epoch_ckpt_path}")

        current_lr = scheduler.get_last_lr()[0]
        if run is not None:
            run.log({"train/lr": current_lr, "epoch": epoch})
            run.log({"train/loss": (running_loss / len(train_loader)).item(), "epoch": epoch})

        if epoch % args.eval_interval == 0:
            val_acc = validate(args, model, test_loader)
            print(f"[Eval] epoch {epoch}/{args.epochs} | val_acc: {val_acc:.2f}%")

            if val_acc > best_acc:
                best_acc = val_acc
                if args.save_best:
                    os.makedirs(args.output_path, exist_ok=True)
                    save_path = os.path.join(args.output_path, 'best.pt')
                    torch.save(model.state_dict(), save_path)
                    print(f"[Eval] best model saved to {save_path} (acc={best_acc:.2f}%)")

            if run is not None:
                run.log({"val/acc": val_acc, "epoch": epoch})
                run.log({"val/best_acc": best_acc, "epoch": epoch})

            model.train()

def validate(args, model, data_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=args.fp16):
                logits = model(x)

            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = 100.0 * correct / total
    return acc

def build_wandb_config(args):
    """Build a serializable config dict from args (before setup() mutates class fields)."""
    exclude = {"config", "device"}
    return {k: v for k, v in vars(args).items() if k not in exclude}

def main():
    args = parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    # Build structured output directory: output/{project}/{model}-{start_time}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    project_dir = os.path.join("output", args.project)
    run_dir_name = f"{args.model}-{timestamp}"
    run_dir = os.path.join(project_dir, run_dir_name)
    os.makedirs(run_dir, exist_ok=True)

    # All checkpoints (best, warmup, periodic) will be saved under this directory
    args.output_path = run_dir

    # Save the fully resolved training configuration (after YAML + CLI) as a single config.yaml
    # Exclude non-serializable / runtime-only fields like device.
    config_path = os.path.join(run_dir, "config.yaml")
    config_to_save = {k: v for k, v in vars(args).items() if k != "device"}
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_to_save, f, sort_keys=False, allow_unicode=True)
    print(f"[Config] Full training config saved to {config_path}")

    if args.run_name == 'XXXXX':
        run_name = f"{args.model}"
    else:
        run_name = f"{args.run_name}"

    if args.no_wandb:
        run = None
        model = setup(args)
        train(args, model, run=None)
    else:
        wandb_config = build_wandb_config(args)
        with wandb.init(entity="snu-inductive-bias", project=args.project, name=run_name, config=wandb_config,) as run:
            model = setup(args)
            train(args, model, run=run)

if __name__ == "__main__":
    main()
