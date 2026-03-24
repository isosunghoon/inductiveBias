python train.py --config config/vit.yaml --channel_mixer mlp --norm_layer layernorm --run_name vit_mlp_layernorm
python train.py --config config/vit.yaml --channel_mixer mlp --norm_layer rmsnorm --run_name vit_mlp_rmsnorm
python train.py --config config/vit.yaml --channel_mixer swiglu --norm_layer layernorm --run_name vit_swiglu_layernorm
python train.py --config config/vit.yaml --channel_mixer swiglu --norm_layer rmsnorm --run_name vit_swiglu_rmsnorm
