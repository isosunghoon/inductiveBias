#!/usr/bin/env bash
# CIFAR-100 experiment sweep — the runs reported in Section 4 / Table 2.
#
# NOTE: every command is prefixed with `tsp` (task-spooler), which queues the jobs so
# they run one at a time on a shared GPU box. Install it with `apt install task-spooler`
# (the binary is `tsp` on Debian/Ubuntu, `ts` on some distros), or simply drop the
# `tsp` prefix to run a single job in the foreground:
#
#     python train.py --config config/vit.yaml --run_name vit --device_type 0
#
# `--device_type` is the CUDA device index and must always be given explicitly.

# ===== Reported token mixers (Table 2) =====
tsp python train.py --config config/vit.yaml --run_name vit --device_type 0 --train_batch_size 1024
tsp python train.py --config config/localvit.yaml --run_name localvit --device_type 0 --train_batch_size 1024
tsp python train.py --config config/convformer.yaml --run_name convformer --device_type 0 --train_batch_size 1024
tsp python train.py --config config/mlpmixer.yaml --run_name mlpmixer --device_type 0 --train_batch_size 1024
tsp python train.py --config config/identity.yaml --run_name identity --device_type 0 --train_batch_size 1024

# ===== Reference model (ImageNet-21k pretrained ViT-B/16) =====
tsp python train.py --config config/pretrained_vit.yaml --run_name pretrained_vit --device_type 0 --train_batch_size 256

# ===== Receptive-field ablations (window / kernel size) =====
tsp python train.py --config config/localvit.yaml --run_name localvit_w5 --device_type 0 --train_batch_size 1024 --window_size 5
tsp python train.py --config config/convformer.yaml --run_name convformer_w5 --device_type 0 --train_batch_size 1024 --kernel_size 5
tsp python train.py --config config/localvit.yaml --run_name localvit_w7 --device_type 0 --train_batch_size 1024 --window_size 7
tsp python train.py --config config/convformer.yaml --run_name convformer_w7 --device_type 0 --train_batch_size 1024 --kernel_size 7
