import torch
import random
import numpy as np
import torch.nn as nn
import yaml

from models.metaformer import MetaFormer
import models.token_mixers as TM
import models.channel_mixers as CM
import models.norm_layers as NL
import models.pretrained_vit as PV

from utils.config import parse_args, resolve_runtime_device
from utils.dataset import build_gpu_post_transform, build_train_eval_loader, get_dataloader
from utils.distributed import (
    all_reduce_sum,
    broadcast_object,
    cleanup_distributed,
    init_distributed,
    is_main_process,
    unwrap_model,
)

from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from functools import partial



import glob
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

def amp_dtype(args):
    """autocast dtype for --fp16. bfloat16 needs no loss scaling."""
    return torch.bfloat16 if getattr(args, "amp_dtype", "float16") == "bfloat16" else torch.float16


def to_device(args, x, training: bool = False):
    """Move a batch to the training device, in NHWC when channels_last is on.

    With --gpu_transform the workers hand over uint8, so the float conversion,
    normalization and RandomErasing happen here, on device, in one batched pass.
    """
    post = getattr(args, "_gpu_post", None)
    if post is not None and x.dtype == torch.uint8:
        return post(x.to(args.device, non_blocking=True), training=training)
    if getattr(args, "channels_last", False):
        return x.to(args.device, non_blocking=True, memory_format=torch.channels_last)
    return x.to(args.device, non_blocking=True)


def setup(args):
    if not args.no_wandb and is_main_process():
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

        # Token mixers. The four reported mixers plus the identity (no-mixing) baseline;
        # see README for the report-name <-> CLI-name mapping.
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
        elif args.model == "convformer":
            args.token_mixer = partial(TM.ConvFormer, kernel_size=args.kernel_size, stride=args.stride, groups=args.conv_groups)
        else:
            raise ValueError(
                f"Unsupported MetaFormer token mixer: {args.model!r}. "
                f"Choose one of: identity, vit, localvit, mlpmixer, convformer."
            )

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
        # The only non-MetaFormer model in the report is the pretrained ViT reference
        # model. The ResNet baseline lives in experimental/models/resnet.py.
        if args.model == 'pretrained_vit':
            model = PV.build_pretrained_vit(args)
        else:
            raise ValueError(f"Unsupported non-MetaFormer model: {args.model}")

    model.to(args.device)
    if getattr(args, "channels_last", False):
        model = model.to(memory_format=torch.channels_last)

    # DDP first, then torch.compile: compiling the wrapped module lets DDPOptimizer split
    # the graph at the gradient-bucket boundaries so communication overlaps with backward.
    if getattr(args, "distributed", False):
        device_ids = [args.local_rank] if torch.cuda.is_available() else None
        model = DistributedDataParallel(model, device_ids=device_ids)

    if getattr(args, "compile", True):
        model = torch.compile(model)
    return model


def save_weights(args, model, path):
    """
    Save a plain weight-only checkpoint (rank 0 only).

    Always saved from the unwrapped module, so the keys carry neither the DDP `module.`
    nor the torch.compile `_orig_mod.` prefix and the analysis pipeline can load them
    regardless of how the run was launched. The write is atomic so that a job killed
    mid-save cannot leave a truncated checkpoint behind.
    """
    if not is_main_process():
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    torch.save(unwrap_model(model).state_dict(), tmp_path)
    os.replace(tmp_path, path)


def save_training_state(args, model, optimizer, scheduler, scaler, epoch, best_acc, best_train_acc):
    """Save everything needed to resume (rank 0 only) to output_path/last.pt."""
    if not is_main_process():
        return
    path = os.path.join(args.output_path, "last.pt")
    os.makedirs(args.output_path, exist_ok=True)
    state = {
        # 'state_dict' 키로 저장해 두면 val.py / analysis의 load_checkpoint가 그대로 읽는다.
        "state_dict": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_acc": best_acc,
        "best_train_acc": best_train_acc,
    }
    tmp_path = f"{path}.tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def load_training_state(args, path, model, optimizer, scheduler, scaler):
    """Restore a run from a last.pt written by save_training_state()."""
    checkpoint = torch.load(path, map_location=args.device, weights_only=False)
    unwrap_model(model).load_state_dict(checkpoint["state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    epoch = int(checkpoint["epoch"])
    if is_main_process():
        print(f"[Resume] restored from {path}; continuing at epoch {epoch + 1}")
    return epoch, float(checkpoint.get("best_acc", 0.0)), float(checkpoint.get("best_train_acc", 0.0))


def resolve_resume_path(args, run_name):
    """
    Resolve --resume into a concrete last.pt path.

    'auto' picks the most recent last.pt under output/{project}/{run_name}-*, which is what
    a requeued cluster job needs: the same command line then continues the previous run
    instead of starting a new timestamped one.
    """
    if not args.resume:
        return None

    if args.resume != "auto":
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"--resume checkpoint not found: {args.resume}")
        return args.resume

    pattern = os.path.join("output", args.project, f"{run_name}-*", "last.pt")
    candidates = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not candidates:
        print(f"[Resume] --resume auto: no previous checkpoint matching {pattern}; starting fresh")
        return None
    return candidates[-1]

def train(args, model, run=None, resume_path=None):
    # prepare dataset
    args._gpu_post = build_gpu_post_transform(args)
    train_loader, test_loader, mixup_fn = get_dataloader(args)
    train_eval_loader = build_train_eval_loader(args, train_loader, args.train_eval_ratio)

    # prepare optimizer & scheduler
    # The reported runs use AdamW; SGD is kept for the ResNet-style baselines.
    # The SAM optimizer was explored but is not part of the report — see experimental/.
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer!r}. Choose 'adamw' or 'sgd'.")

    if args.decay_type == "cosine":
        warmup_scheduler = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=args.warmup_epochs)
        # max(..., 1): epochs == warmup_epochs 인 짧은 스모크 런에서 T_max=0 으로 죽지 않게
        main_scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs - args.warmup_epochs, 1))
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[args.warmup_epochs])
    else:
        scheduler = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=args.epochs)

    # bfloat16 autocast has fp32 dynamic range, so it needs no loss scaling.
    scaler = None
    if args.fp16 and amp_dtype(args) == torch.float16:
        scaler = torch.amp.GradScaler('cuda')

    model.zero_grad()
    best_acc = 0
    best_train_acc = 0.0
    global_step = 0
    start_epoch = 0

    if resume_path is not None:
        start_epoch, best_acc, best_train_acc = load_training_state(
            args, resume_path, model, optimizer, scheduler, scaler,
        )

    # train loop
    for epoch in range(start_epoch + 1, args.epochs+1):
        model.train()
        # 분산 시 에폭마다 셔플 순서를 바꾸려면 sampler에 epoch을 알려줘야 한다.
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        epoch_iterator = tqdm(train_loader, desc="Training (X / X epochs) (loss=X.X)",
                                bar_format="{l_bar}{r_bar}", dynamic_ncols=True,
                                disable=not is_main_process())

        running_loss = torch.zeros((), device=args.device)   # 진행률 표시용 (log_interval마다 리셋)
        epoch_loss = torch.zeros((), device=args.device)     # 에폭 전체 평균 loss용
        interval_steps = 0
        epoch_steps = 0
        for step, batch in enumerate(epoch_iterator):
            x, y = batch
            x = to_device(args, x, training=True)
            y = y.to(args.device, non_blocking=True)

            if mixup_fn is not None:
                x, y = mixup_fn(x, y)

            with torch.amp.autocast('cuda', enabled=args.fp16, dtype=amp_dtype(args)):
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits, y, label_smoothing=args.label_smoothing)

            running_loss += loss.detach()
            epoch_loss += loss.detach()
            interval_steps += 1
            epoch_steps += 1
            global_step += 1

            if scaler is not None:
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
                avg_loss = (running_loss / max(interval_steps, 1)).item()
                running_loss.zero_()
                interval_steps = 0
                epoch_iterator.set_description(f"Training ({epoch} / {args.epochs} epochs) (loss={avg_loss:.5f})")

        scheduler.step()

        # Save checkpoint right after warmup ends
        if epoch == args.warmup_epochs:
            warmup_ckpt_path = os.path.join(args.output_path, "epoch_warmup_end.pt")
            save_weights(args, model, warmup_ckpt_path)
            if is_main_process():
                print(f"[Checkpoint] warmup-end model saved to {warmup_ckpt_path}")

        # Save model every ckpt_interval epochs (default 100: 100, 200, 300, ...)
        if args.ckpt_interval > 0 and epoch % args.ckpt_interval == 0:
            epoch_ckpt_path = os.path.join(args.output_path, f"epoch_{epoch}.pt")
            save_weights(args, model, epoch_ckpt_path)
            if is_main_process():
                print(f"[Checkpoint] epoch-{epoch} model saved to {epoch_ckpt_path}")

        current_lr = scheduler.get_last_lr()[0]
        # 에폭 평균 loss는 모든 rank에 걸쳐 평균낸다.
        mean_loss = all_reduce_sum(epoch_loss.item(), args.device) / max(
            epoch_steps * getattr(args, "world_size", 1), 1
        )
        if run is not None:
            run.log({"train/lr": current_lr, "epoch": epoch})
            run.log({"train/loss": mean_loss, "epoch": epoch})

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            train_acc = validate(args, model, train_eval_loader)
            val_acc = validate(args, model, test_loader)
            if is_main_process():
                print(f"[Eval] epoch {epoch}/{args.epochs} | train_acc: {train_acc:.2f}% | val_acc: {val_acc:.2f}%)")
            if train_acc > best_train_acc:
                best_train_acc = train_acc
            if val_acc > best_acc:
                best_acc = val_acc
                if args.save_best:
                    save_path = os.path.join(args.output_path, 'best.pt')
                    save_weights(args, model, save_path)
                    if is_main_process():
                        print(f"[Eval] best model saved to {save_path} (acc={best_acc:.2f}%)")

            if run is not None:
                run.log({"train/acc": train_acc, "epoch": epoch})
                run.log({"train/best_acc": best_train_acc, "epoch": epoch})
                run.log({"val/acc": val_acc, "epoch": epoch})
                run.log({"val/best_acc": best_acc, "epoch": epoch})

            model.train()

        # 매 에폭 resume용 상태 저장 (노드 장애 / requeue 대비)
        save_training_state(
            args, model, optimizer, scheduler, scaler, epoch, best_acc, best_train_acc,
        )

def validate(args, model, data_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in data_loader:
            x = to_device(args, x)
            y = y.to(args.device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=args.fp16, dtype=amp_dtype(args)):
                logits = model(x)

            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    # 분산 시 rank마다 서로 다른 샤드를 봤으므로 개수를 합친 뒤 정확도를 계산한다
    # (rank별 정확도를 평균내면 샤드 크기가 다를 때 값이 틀어진다).
    correct = all_reduce_sum(correct, args.device)
    total = all_reduce_sum(total, args.device)
    return 100.0 * correct / max(total, 1.0)

# Attributes that only describe *this process* (device, DDP rank, ...). They are set at
# runtime, so writing them into config.yaml would make that file unusable as an input
# config later on (_apply_yaml rejects keys the parser does not know).
_RUNTIME_ONLY_KEYS = {"device", "distributed", "rank", "world_size", "local_rank"}


def build_wandb_config(args):
    """Build a serializable config dict from args (before setup() mutates class fields)."""
    exclude = {"config", "device"}
    # 밑줄로 시작하는 키는 런타임 객체(예: _gpu_post)라 직렬화 대상이 아니다.
    return {k: v for k, v in vars(args).items()
            if k not in exclude and not k.startswith("_")}

def main():
    args = parse_args()
    init_distributed(args)
    resolve_runtime_device(args)
    # rank마다 다른 시드로 증강을 다양화한다. 모델 파라미터는 DDP 생성 시 rank 0에서
    # 브로드캐스트되므로 초기화가 갈라지지 않는다.
    set_seed(args.seed + args.rank)
    torch.backends.cudnn.benchmark = True

    if args.run_name == 'XXXXX':
        run_name = f"{args.model}"
    else:
        run_name = f"{args.run_name}"

    # Build structured output directory: output/{project}/{model}-{start_time}
    # Resuming reuses the previous run directory instead of opening a new one, and the
    # directory is decided on rank 0 and broadcast so every rank agrees on the timestamp.
    run_dir = None
    resume_path = None
    if is_main_process():
        resume_path = resolve_resume_path(args, run_name)
        if resume_path is not None:
            run_dir = os.path.dirname(os.path.abspath(resume_path))
        else:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_dir = os.path.join("output", args.project, f"{run_name}-{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
    run_dir, resume_path = broadcast_object((run_dir, resume_path))

    # All checkpoints (best, warmup, periodic, last) will be saved under this directory
    args.output_path = run_dir

    # Save the fully resolved training configuration (after YAML + CLI) as a single config.yaml.
    # Save only device_type (runtime device string is reconstructed at execution time).
    if is_main_process():
        config_path = os.path.join(run_dir, "config.yaml")
        config_to_save = {k: v for k, v in vars(args).items()
                          if k not in _RUNTIME_ONLY_KEYS and not k.startswith("_")}
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_to_save, f, sort_keys=False, allow_unicode=True)
        print(f"[Config] Full training config saved to {config_path}")
        if args.distributed:
            print(f"[DDP] world_size={args.world_size}, per-GPU batch={args.train_batch_size}, "
                  f"global batch={args.train_batch_size * args.world_size}")

    try:
        # W&B는 rank 0에서만 기록한다. run id를 run 디렉터리 이름으로 고정해 두면
        # resume 시 같은 W&B run에 이어서 로깅된다.
        if args.no_wandb or not is_main_process():
            model = setup(args)
            train(args, model, run=None, resume_path=resume_path)
        else:
            wandb_config = build_wandb_config(args)
            with wandb.init(entity=args.wandb_entity, project=args.project, name=run_name,
                            id=os.path.basename(run_dir), resume="allow",
                            config=wandb_config,) as run:
                model = setup(args)
                train(args, model, run=run, resume_path=resume_path)
    finally:
        cleanup_distributed()

if __name__ == "__main__":
    main()
