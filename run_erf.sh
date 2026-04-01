set -e

PATHS=(
    "output/for_analysis/identity-2026-03-25_13-05-47"
    "output/search_sunghoon/denseformer-2026-03-24_02-11-37"
    "output/model_scaling_exp/vit_d12_e192-2026-03-27_17-07-18"
)


for path in "${PATHS[@]}"; do
    echo "=========================================="
    echo "Running ERF for: $path"
    echo "=========================================="
    PYTHONPATH=/workspace/inductiveBias python analysis/erf.py --save_path "$path"
    echo "Done: $path"
    echo ""
done

echo "All ERF runs complete."
