set -e

PATHS=(
    "output/for_analysis/identity-2026-03-25_13-05-47"
    "output/for_analysis/localvit-2026-03-25_04-22-24"
    #"output/for_analysis/vit-2026-03-24_07-47-38"
    #"output/for_analysis/convformer-2026-03-16_10-13-46"
    #"output/for_analysis/denseformer-2026-03-24_02-11-37"
)


for path in "${PATHS[@]}"; do
    echo "=========================================="
    echo "Running ERF for: $path"
    echo "=========================================="
    PYTHONPATH=/workspace/inductiveBias python analysis/erf.py \
        --output_path "$path" \
        --num_images 100 \
        --anchor_mode random \
        --num_anchors 3 \
        --distance_metric taxi \
        --config "$path/config.yaml"
    echo "Done: $path"
    echo ""
done

echo "All ERF runs complete."
