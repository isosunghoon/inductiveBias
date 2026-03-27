set -e

# ── Configurable inputs ──────────────────────────────────────────────────────
BASE_CONFIG="${BASE_CONFIG:-./config/base.yaml}"

# Space-separated list of per-model config yamls (same order as CKPTS)
CONFIGS=(
    # "./config/mlpmixer.yaml"
    "./config/vit.yaml"
    "./config/vit.yaml"
    #"./config/poolformer.yaml"
    #"./config/localvit.yaml"
)

# Space-separated list of checkpoint paths (same order as CONFIGS)
CKPTS=(
    # "./output/base_model_exp/mlpmixer-2026-03-13_05-47-26/best.pt"
    "./output/base_model_exp/vit-2026-03-12_15-39-12/best.pt"
    "./output/base_model_exp/vit-2026-03-12_15-39-12/best.pt"
    #"./output/base_model_exp/poolformer-2026-03-12_23-11-35/best.pt"
    #"./output/base_model_exp/localvit-2026-03-12_18-08-26/best.pt"
)

# ── Run ──────────────────────────────────────────────────────────────────────
python analysis/cka.py \
    --base_config "$BASE_CONFIG" \
    --configs "${CONFIGS[@]}" \
    --ckpts   "${CKPTS[@]}"
