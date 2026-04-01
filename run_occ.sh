set -e

PYTHONPATH=/workspace/inductiveBias python analysis/dis_occ_sensitivity.py \
  --config config/bin/denseformer.yaml \
  --output_path output/search_sunghoon/denseformer-2026-03-24_02-11-37 \
  --max_batches 5 \
  --max_d 20 \
  --anchor_stride 4 \
