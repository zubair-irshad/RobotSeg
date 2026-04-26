#!/usr/bin/env bash
# Full OXE pipeline: segmentation → robust per-episode PnP → reprojection viz.
#
# Usage:
#   bash run_oxe_pipeline.sh                # everything under data/oxe_subset
#   bash run_oxe_pipeline.sh bridge taco_play
#   IMAGE_ROOT=... SAVE_ROOT=... bash run_oxe_pipeline.sh
#   PIPELINE_SKIP_SEG=1 bash run_oxe_pipeline.sh   # only PnP step

set -euo pipefail

IMAGE_ROOT="${IMAGE_ROOT:-$HOME/RobotSeg/data/oxe_subset}"
SAVE_ROOT="${SAVE_ROOT:-$HOME/RobotSeg/data/oxe_subset_seg}"
CATEGORIES="${CATEGORIES:-arm,gripper}"

cd "$(dirname "$0")"

if [[ $# -gt 0 ]]; then
  DATASETS=("$@")
else
  mapfile -t DATASETS < <(
    find "$IMAGE_ROOT" -mindepth 1 -maxdepth 1 -type d \
      -not -name '_frames_view' -not -name '.*' -printf '%f\n' | sort
  )
fi

# 1. Segmentation (arm + gripper masks + sigmoid prob maps).
if [[ "${PIPELINE_SKIP_SEG:-0}" != "1" ]]; then
  for d in "${DATASETS[@]}"; do
    src="$IMAGE_ROOT/$d"
    dst="$SAVE_ROOT/$d"
    [[ -d "$src" ]] || { echo "[skip] $src"; continue; }
    echo "==> seg: $d"
    python inference_auto_qual.py \
      --image_root "$src" \
      --save_root  "$dst" \
      --categories "$CATEGORIES" \
      --save_overlay --save_prob
  done
fi

# 2. Robust PnP per episode + scene-level bad-flagging.
echo "==> pnp"
python ../tools/pnp_oxe.py \
  --oxe_root  "$IMAGE_ROOT" \
  --mask_root "$SAVE_ROOT" \
  --datasets "${DATASETS[@]}" \
  --viz

echo "Done."
