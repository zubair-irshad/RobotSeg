#!/usr/bin/env bash
# Run RobotSeg on every OXE dataset already downloaded under data/oxe_subset.
# Produces per-category masks (arm in 000/, gripper in 001/) and a combined
# overlay (arm=green, gripper=red, EE centroid marked) under <seq>/combined/.
#
# Usage:
#   bash run_oxe_seg.sh                # all datasets in default root
#   bash run_oxe_seg.sh ds1 ds2        # only listed datasets
#   IMAGE_ROOT=... SAVE_ROOT=... bash run_oxe_seg.sh

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

if [[ ${#DATASETS[@]} -eq 0 ]]; then
  echo "No datasets found under $IMAGE_ROOT" >&2
  exit 1
fi

echo "Datasets: ${DATASETS[*]}"
echo "Categories: $CATEGORIES"
echo "Save root: $SAVE_ROOT"
echo

for d in "${DATASETS[@]}"; do
  src="$IMAGE_ROOT/$d"
  dst="$SAVE_ROOT/$d"
  if [[ ! -d "$src" ]]; then
    echo "[skip] $src (not a directory)"
    continue
  fi
  # Skip top-level non-dataset metadata files like dataset_map.json.
  if [[ -z "$(find "$src" -mindepth 1 -maxdepth 2 -type d -print -quit)" ]]; then
    echo "[skip] $d (no sequence subdirs)"
    continue
  fi
  echo "==> $d"
  python inference_auto_qual.py \
    --image_root "$src" \
    --save_root  "$dst" \
    --categories "$CATEGORIES" \
    --save_overlay
done

echo "Done."
