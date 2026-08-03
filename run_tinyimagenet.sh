#!/usr/bin/env bash
# Tiny ImageNet experiment sweep — the runs reported in Section 4 / Table 2.
#
# NOTE: every command is prefixed with `tsp` (task-spooler), which queues the jobs so
# they run one at a time on a shared GPU box. Install it with `apt install task-spooler`
# (the binary is `tsp` on Debian/Ubuntu, `ts` on some distros), or simply drop the
# `tsp` prefix to run a single job in the foreground.
#
# `--base_config` switches the shared defaults to Tiny ImageNet (200 classes,
# ImageNet normalization); `--config` then picks the token mixer.

# ===== Reported token mixers (Table 2) =====
tsp python train.py --base_config ./config/base_tinyimagenet.yaml --config config/vit.yaml --run_name vit --device_type 0 --train_batch_size 1024
tsp python train.py --base_config ./config/base_tinyimagenet.yaml --config config/localvit.yaml --run_name localvit --device_type 0 --train_batch_size 1024
tsp python train.py --base_config ./config/base_tinyimagenet.yaml --config config/convformer.yaml --run_name convformer --device_type 0 --train_batch_size 1024
tsp python train.py --base_config ./config/base_tinyimagenet.yaml --config config/mlpmixer.yaml --run_name mlpmixer --device_type 0 --train_batch_size 1024
tsp python train.py --base_config ./config/base_tinyimagenet.yaml --config config/identity.yaml --run_name identity --device_type 0 --train_batch_size 1024

# ===== Reference model (ImageNet-21k pretrained ViT-B/16) =====
tsp python train.py --base_config ./config/base_tinyimagenet.yaml --config config/pretrained_vit.yaml --run_name pretrained_vit --device_type 0 --train_batch_size 256

# ===== Patch-size / capacity ablations =====
tsp python train.py --base_config ./config/base_tinyimagenet.yaml --config config/vit.yaml --run_name vit_64_p4 --device_type 0 --train_batch_size 512 --patch_size 4
tsp python train.py --base_config ./config/base_tinyimagenet.yaml --config config/vit.yaml --run_name vit_64_p8_lr1e-3 --device_type 0 --train_batch_size 1024 --learning_rate 0.001
tsp python train.py --base_config ./config/base_tinyimagenet.yaml --config config/vit.yaml --run_name vit_small --device_type 0 --train_batch_size 256 --embed_dim 384
