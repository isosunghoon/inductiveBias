"""
Helpers for single-process and torchrun-launched (DDP) runs.

Every entry point here is safe to call when distributed training is *not* in use: the
process then behaves as a world of size 1, so the single-GPU code path that produced the
reported CIFAR-100 / Tiny ImageNet runs is unchanged.

Launch with:
    torchrun --nproc_per_node=8 train.py ...
"""

import os
from contextlib import contextmanager
from datetime import timedelta

import torch
import torch.distributed as dist


DEFAULT_TIMEOUT = timedelta(minutes=30)


def init_distributed(args) -> bool:
    """
    Read the torchrun environment and initialise the process group.

    Sets args.distributed / args.rank / args.world_size / args.local_rank in both the
    distributed and the single-process case, and returns whether DDP is active.
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size <= 1 or not dist.is_available():
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return False

    args.rank = int(os.environ["RANK"])
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.world_size = world_size

    if torch.cuda.is_available():
        backend = "nccl"
        torch.cuda.set_device(args.local_rank)
    else:
        backend = "gloo"

    dist.init_process_group(backend=backend, timeout=DEFAULT_TIMEOUT)
    args.distributed = True
    dist.barrier()
    return True


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier():
    if is_distributed():
        dist.barrier()


@contextmanager
def main_process_first():
    """
    Run the block on rank 0 first, then on every other rank.

    Used for one-off filesystem work (dataset download / extraction) that must happen
    exactly once but that every rank has to see finished before it continues.
    """
    if not is_distributed():
        yield
        return

    if not is_main_process():
        dist.barrier()
    try:
        yield
    finally:
        if is_main_process():
            dist.barrier()


def broadcast_object(obj, src: int = 0):
    """Broadcast a picklable object from `src` to every rank (returns it on all ranks)."""
    if not is_distributed():
        return obj
    holder = [obj]
    dist.broadcast_object_list(holder, src=src)
    return holder[0]


def all_reduce_sum(value, device) -> float:
    """Sum a python scalar across ranks and return it as a float."""
    tensor = torch.tensor([float(value)], device=device)
    if is_distributed():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Strip DistributedDataParallel (`.module`) and torch.compile (`._orig_mod`) wrappers.

    Checkpoints are always saved from the unwrapped module so that the state_dict keys
    stay identical to a plain single-GPU run and the analysis pipeline can load them.
    """
    seen = set()
    while True:
        if id(model) in seen:
            return model
        seen.add(id(model))
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        else:
            return model
