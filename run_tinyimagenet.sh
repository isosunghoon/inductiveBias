tsp python train.py --base_config ./config/base_tinyimagenet.yaml --config config/vit.yaml --run_name vit --device_type 0 --train_batch_size 1024
tsp python train.py --base_config ./config/base_tinyimagenet.yaml --config config/localvit.yaml --run_name localvit --device_type 0 --train_batch_size 1024
tsp python train.py --base_config ./config/base_tinyimagenet.yaml --config config/convformer.yaml --run_name convformer --device_type 0 --train_batch_size 1024
tsp python train.py --base_config ./config/base_tinyimagenet.yaml --config config/denseformer.yaml --run_name denseformer --device_type 0 --train_batch_size 1024
tsp python train.py --base_config ./config/base_tinyimagenet.yaml --config config/identity.yaml --run_name identity --device_type 0 --train_batch_size 1024

tsp python train.py --base_config ./config/base_tinyimagenet.yaml --config config/vit.yaml --run_name vit_64_p4 --device_type 0 --train_batch_size 512 --patch_size 4
tsp python train.py --base_config ./config/base_tinyimagenet.yaml --config config/vit.yaml --run_name vit_64_p8_lr1e-3 --device_type 0 --train_batch_size 1024 --learning_rate 0.001
tsp python train.py --base_config ./config/base_tinyimagenet.yaml --config config/pretrained_vit.yaml --run_name pretrained_vit --device_type 0 --train_batch_size 256
tsp python train.py --base_config ./config/base_tinyimagenet.yaml --config config/vit.yaml --run_name vit_small --device_type 0 --train_batch_size 256 --embed_dim 384