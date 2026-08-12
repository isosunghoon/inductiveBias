#!/usr/bin/env bash
# ImageNet-1k sweep — same five token mixers as run.sh / run_tinyimagenet.sh.
#
# Multi-GPU only: ImageNet on a single GPU takes days per model. Each command launches
# one torchrun job over all GPUs of the node; `--device_type` must NOT be passed, since
# torchrun assigns each rank its GPU through LOCAL_RANK.
#
# Before running, stage the dataset (it is not downloadable anonymously):
#   ./data/imagenet/train/<wnid>/*.JPEG
#   ./data/imagenet/val/<wnid>/*.JPEG
#
# NOTE: the per-mixer configs under config/ were tuned for CIFAR-100 and carry
# `epochs: 1000` (and, for mlpmixer, its own lr / warmup). `--config` overrides
# `--base_config`, so the ImageNet schedule is re-stated on the command line below.
#
# `--resume auto` continues the newest run of the same name under
# output/{project}/{run_name}-*, so a preempted or requeued job can be resubmitted with
# exactly the same command line.

set -u

NPROC=${NPROC:-8}          # GPUs per node
EPOCHS=${EPOCHS:-300}
WARMUP=${WARMUP:-20}

run () {
  local cfg=$1 name=$2
  torchrun --standalone --nproc_per_node="${NPROC}" train.py \
    --base_config ./config/base_imagenet.yaml \
    --config "${cfg}" \
    --run_name "${name}" \
    --epochs "${EPOCHS}" \
    --warmup_epochs "${WARMUP}" \
    --resume auto
}

# ===== Reported token mixers =====
run config/vit.yaml         vit
run config/localvit.yaml    localvit
run config/convformer.yaml  convformer
run config/mlpmixer.yaml    mlpmixer
run config/identity.yaml    identity
