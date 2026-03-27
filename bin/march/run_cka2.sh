set -e

BASE_CONFIG="${BASE_CONFIG:-./config/base.yaml}"

# ── mlpmixer ─────────────────────────────────────────────────────────────────
python analysis/cka.py \
    --base_config "$BASE_CONFIG" \
    --configs ./config/mlpmixer.yaml ./config/mlpmixer.yaml \
    --ckpts \
        ./output/base_model_exp/mlpmixer-2026-03-13_05-47-26/best.pt \
        ./output/base_model_exp/mlpmixer-2026-03-13_05-47-26/best.pt

# ── vit ───────────────────────────────────────────────────────────────────────
python analysis/cka.py \
    --base_config "$BASE_CONFIG" \
    --configs ./config/vit.yaml ./config/vit.yaml \
    --ckpts \
        ./output/base_model_exp/vit-2026-03-12_15-39-12/best.pt \
        ./output/base_model_exp/vit-2026-03-12_15-39-12/best.pt

# ── localvit ──────────────────────────────────────────────────────────────────
python analysis/cka.py \
    --base_config "$BASE_CONFIG" \
    --configs ./config/localvit.yaml ./config/localvit.yaml \
    --ckpts \
        ./output/base_model_exp/localvit-2026-03-12_18-08-26/best.pt \
        ./output/base_model_exp/localvit-2026-03-12_18-08-26/best.pt

# ── poolformer ────────────────────────────────────────────────────────────────
python analysis/cka.py \
    --base_config "$BASE_CONFIG" \
    --configs ./config/poolformer.yaml ./config/poolformer.yaml \
    --ckpts \
        ./output/base_model_exp/poolformer-2026-03-12_23-11-35/best.pt \
        ./output/base_model_exp/poolformer-2026-03-12_23-11-35/best.pt
