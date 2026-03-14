python train.py --config config/vit.yaml
python train.py --config config/localvit.yaml --window_size 5 --run_name localvit_w5
python train.py --config config/localvit.yaml --window_size 3 --run_name localvit_w3
python train.py --config config/poolformer.yaml --pool_size 5 --run_name poolformer_p5
python train.py --config config/poolformer.yaml --pool_size 3 --run_name poolformer_p3
python train.py --config config/mlpmixer.yaml --run_name mlpmixer_md5
python train.py --config config/mlpmixer.yaml --mixer_drop 0.1 --run_name mlpmixer_md1
python train.py --config config/mlpmixer.yaml --mixer_drop 0