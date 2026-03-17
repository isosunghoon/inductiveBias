set -e

# ── Configurable inputs ──────────────────────────────────────────────────────
BASE_CONFIG="${BASE_CONFIG:-./config/base.yaml}"
CONFIG1="${CONFIG1:-./config/mlpmixer.yaml}"
CONFIG2="${CONFIG2:-./config/vit.yaml}"
CKPT1="${CKPT1:./output/base_model_exp/mlpmixer-2026-03-13_05-47-26/best.pt}"
CKPT2="${CKPT2:./output/base_model_exp/vit-2026-03-12_15-39-12/best.pt}"

# ── Run ──────────────────────────────────────────────────────────────────────
python analysis/cka.py \
    --base_config  "$BASE_CONFIG"  \
    --config1      "$CONFIG1"      \
    --config2      "$CONFIG2"      \
    --ckpt1        "$CKPT1"        \
    --ckpt2        "$CKPT2"
