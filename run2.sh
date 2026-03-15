python train.py --config config/denseformer.yaml --mixer_drop 0.1
python train.py --config config/convformer.yaml --kernel_size 5 --run_name convformer_w5
python train.py --config config/convformer.yaml --kernel_size 3 --run_name convformer_w3
python train.py --config config/convformer2.yaml --run_name convformer2