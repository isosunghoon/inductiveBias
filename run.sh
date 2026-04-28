tsp python train.py --config config/vit.yaml --run_name vit --device_type 0 --train_batch_size 1024
tsp python train.py --config config/localvit.yaml --run_name localvit --device_type 0 --train_batch_size 1024
tsp python train.py --config config/convformer.yaml --run_name convformer --device_type 0 --train_batch_size 1024
tsp python train.py --config config/denseformer.yaml --run_name denseformer --device_type 0 --train_batch_size 1024
tsp python train.py --config config/identity.yaml --run_name identity --device_type 0 --train_batch_size 1024
