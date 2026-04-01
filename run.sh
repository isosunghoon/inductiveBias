tsp python train.py --config config/vit.yaml --run_name vit_d12_e192
tsp python train.py --config config/vit.yaml --depth 6 --run_name vit_d6_e192
tsp python train.py --config config/vit.yaml --depth 6 --embed_dim 128 --run_name vit_d6_e128
tsp python train.py --config config/vit.yaml --depth 3 --run_name vit_d3_e192
tsp python train.py --config config/vit.yaml --depth 6 --embed_dim 128 --attn_head_dim 32 --run_name vit_d6_e128_32
tsp python train.py --config config/vit.yaml --depth 6 --embed_dim 64 --attn_head_dim 32 --run_name vit_d6_e64_32
tsp python train.py --config config/vit.yaml --patch_size 2 --depth 6 --embed_dim 128 --run_name vit_p2_d6_e128
tsp python train.py --config config/vit.yaml --depth 6 --embed_dim 128 --drop_path 0.2 --weight_decay 0.1 --run_name vit_d6_e128_regularization
tsp python train.py --config config/vit.yaml --depth 6 --embed_dim 128 --attn_head_dim 32 --drop_path 0.2 --weight_decay 0.1 --drop_rate 0.1 --attn_drop 0.1 --attn_proj_drop 0.1 --epochs 1500 --run_name vit_d6_e128_32_reg_strong
tsp python train.py --config config/vit.yaml --depth 6 --embed_dim 128 --attn_head_dim 32 --augment weak --epochs 1500 --run_name vit_d6_e128_32_aug_weak
tsp python train.py --config config/vit.yaml --embed_dim 384 --epochs 3000 --run_name vit_d12_e384

1. base
2. depth만 6으로 낮춘것 epoch 1000으로 다시
3. depth 6 + dim 128
4. depth만 3으로 낮춘것
5. depth 6 dim 128 + att.dim 32
6. depth 6 dim 64 + att.dim 32
7. depth 3 + dim 96 + att.dim 32 (이미 돌아가는 중)
8. depth 6 + dim 128 + p.size 2
9. depth 6 + dim 128 + drop_path 0.2 + weight decay 0.1 
10. depth 6 + dim 128 + att.dim 32 + drop_path 0.2 + weight decay 0.1 + drop rate 3개 다 0.1 + 1500 epochs
11. depth 6 + dim 128 + att.dim 32 + weak augmentation