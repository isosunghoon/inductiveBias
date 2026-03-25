python train.py --config config/localvit.yaml --channel_mixer swiglu --norm_layer rmsnorm --run_name localvit_swiglu_rmsnorm
python train.py --config config/localvit.yaml --channel_mixer mlp --norm_layer layernorm --run_name localvit_mlp_layernorm
python train.py --config config/localvit.yaml --channel_mixer mlp --norm_layer rmsnorm --run_name localvit_mlp_rmsnorm
python train.py --config config/localvit.yaml --channel_mixer swiglu --norm_layer layernorm --run_name localvit_swiglu_layernorm
