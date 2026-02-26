import torch
import random
import numpy as np
import torch.nn as nn
from models.metaformer import MetaFormer
from utils.config import parse_args
from utils.dataset import get_dataloader
from utils.AverageMeter import AverageMeter
from tqdm import tqdm
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
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
    losses = AverageMeter()

    # train loop
    for epoch in range(1, args.epochs+1):
        model.train()
        epoch_iterator = tqdm(train_loader, desc="Training (X / X epochs) (loss=X.X)",
                                bar_format="{l_bar}{r_bar}", dynamic_ncols=True)
    
        for _, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            with torch.amp.autocast('cuda', enabled=args.fp16):
                logits = model(x) 
                loss = torch.nn.functional.cross_entropy(logits, y)
            
            losses.update(loss.item())

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

            scheduler.step()
            optimizer.zero_grad()
            
            epoch_iterator.set_description(
                "Training (%d / %d epochs) (loss=%2.5f)" % (epoch, args.epochs, losses.val)
            )
        
        losses.reset()

        # TODO
        # add validation
            # if epoch % args.eval_step == 0:
            #     accuracy = valid(args, model, writer, test_loader, global_step)
            #         if best_acc < accuracy:
            #             save_model(args, model)
            #             best_acc = accuracy
            #         model.train()

            #     if global_step % t_total == 0:
            #         break

def main():
    args = parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    model = setup(args)
    train(args, model)


if __name__ == "__main__":
    main()