python train.py --config config/localvit.yaml --drop_path 0.2 --run_name localvit_droppath0.2
python train.py --config config/localvit.yaml --optimizer sam --run_name localvit_sam
python train.py --config config/localvit.yaml --weight_decay 0.1 --run_name localvit_weight0.1
python train.py --config config/localvit.yaml --learning_rate 0.001 --epochs 500 --run_name localvit_lr1e-3
python train.py --config config/localvit.yaml --layer_norm groupnorm --run_name localvit_groupnorm
python train.py --config config/localvit.yaml --seed 42 --run_name localvit_seed42