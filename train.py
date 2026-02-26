import torch
import random
import numpy as np
import torch.nn as nn
from models.metaformer import MetaFormer
import models.token_mixers as TM
from utils.config import parse_args
from utils.dataset import get_dataloader
from tqdm import tqdm
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from functools import partial

import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torch.optim.lr_scheduler"
)

def set_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

def setup(args):
    if args.norm_layer == 'identity':
        args.norm_layer = nn.Identity

    if args.act_layer == 'GELU':
        args.act_layer = nn.GELU

    if args.model == "identity":
        args.token_mixer = nn.Identity
    elif args.model == "vit":
        args.token_mixer = partial(TM.Attention, head_dim=args.attn_head_dim, qkv_bias=args.attn_qkv_bias,
                            attn_drop=args.attn_drop, proj_drop=args.attn_proj_drop,)

    model = MetaFormer(depth=args.depth, embed_dim=args.embed_dim, token_mixer=args.token_mixer, mlp_ratio=args.mlp_ratio, 
                 norm_layer=args.norm_layer, act_layer=args.act_layer, num_classes=args.num_classes, patch_size=args.patch_size, img_size=args.img_size, add_pos_emb=args.add_pos_emb, drop_rate=args.drop_rate, drop_path_rate = args.drop_path,
                 use_layer_scale=args.use_layer_scale, layer_scale_init_value=args.layer_scale_init_value)
    model.to(args.device)

    return model

def train(args, model):
    # prepare dataset
    train_loader, test_loader = get_dataloader(args)

    # prepare optimizer & schedular
    # TODO
    # use other optimizer (e.g. Adam)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    
    if args.decay_type == "cosine":
        warmup_scheduler = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=args.warmup_epochs)
        main_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[args.warmup_epochs])
    else:
        scheduler = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=args.epochs)

    scaler = None
    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()

    model.zero_grad()
    best_acc = 0

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
            
            with torch.amp.autocast('cuda', enabled=args.fp16):
                logits = model(x) 
                loss = torch.nn.functional.cross_entropy(logits, y)
            
            running_loss += loss.detach()

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
                avg_loss = (running_loss / step).item()
                running_loss.zero_()
                epoch_iterator.set_description(
                    f"Training ({epoch} / {args.epochs} epochs) (loss={avg_loss:.5f})"
                )
        
        scheduler.step()

        if epoch % args.eval_interval == 0:
            val_acc = validate(args, model, test_loader)
            print(f"[Eval] epoch {epoch}/{args.epochs} | val_acc: {val_acc:.2f}%")

            if val_acc > best_acc:
                best_acc = val_acc
                if args.save_best:
                    save_path = args.output_path+'/best.pt'
                    torch.save(model.state_dict(), save_path)
                    print(f"[Eval] best model saved to {save_path} (acc={best_acc:.2f}%)")

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

def main():
    args = parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    model = setup(args)
    train(args, model)


if __name__ == "__main__":
    main()