import torch
import random
import numpy as np
import torch.nn as nn

from models.metaformer import MetaFormer
import models.token_mixers as TM
import models.norm_layers as NL

from utils.config import parse_args
from utils.dataset import get_dataloader

from tqdm import tqdm
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from functools import partial

import os
import warnings
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
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")

    if args.norm_layer == 'identity':
        args.norm_layer = nn.Identity
    elif args.norm_layer == 'layernorm':
        args.norm_layer = NL.LayerNorm

    if args.act_layer == 'GELU':
        args.act_layer = nn.GELU

    if args.model == "identity":
        args.token_mixer = nn.Identity
    elif args.model == "vit":
        args.token_mixer = partial(TM.Attention, head_dim=args.attn_head_dim, qkv_bias=args.attn_qkv_bias,
                            attn_drop=args.attn_drop, proj_drop=args.attn_proj_drop,)
    elif args.model == "local_vit":
        args.token_mixer = partial(TM.ConvAttention, head_dim=args.attn_head_dim, window_size=args.window_size, 
                            qkv_bias=args.attn_qkv_bias, attn_drop=args.attn_drop, proj_drop=args.attn_proj_drop,)
    elif args.model == "MLP-Mixer":
        args.token_mixer = partial(TM.MLPMixer, img_size=args.img_size, patch_size=args.patch_size,
                                    expansion_factor=args.expansion_factor, mixer_drop=args.mixer_drop,)                               

    model = MetaFormer(depth=args.depth, embed_dim=args.embed_dim, token_mixer=args.token_mixer, mlp_ratio=args.mlp_ratio,
                 norm_layer=args.norm_layer, act_layer=args.act_layer, num_classes=args.num_classes, patch_size=args.patch_size, img_size=args.img_size, add_pos_emb=args.add_pos_emb, drop_rate=args.drop_rate, drop_path_rate = args.drop_path,
                 use_layer_scale=args.use_layer_scale, layer_scale_init_value=args.layer_scale_init_value)
    model.to(args.device)

    return model

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

    run_name = f"{args.model}_p{args.patch_size}"

    if args.no_wandb:
        run = None
        model = setup(args)
        train(args, model, run=None)
    else:
        wandb_config = build_wandb_config(args)
        with wandb.init(entity="snu-inductive-bias", project="exp1", name=run_name, config=wandb_config,) as run:
            model = setup(args)
            train(args, model, run=run)

if __name__ == "__main__":
    main()
