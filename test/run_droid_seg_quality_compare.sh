#!/usr/bin/env bash
# Generate side-by-side DROID RobotSeg outputs with separate folder names.
#
# Output layout:
#   <SAVE_ROOT_BASE>/native_raw/droid/<episode>/{000,001,...}
#   <SAVE_ROOT_BASE>/native_guided/droid/<episode>/{000,001,...}
#   <SAVE_ROOT_BASE>/current_pipeline/droid/<episode>/{000,001,...}
#
# Usage:
#   bash run_droid_seg_quality_compare.sh
#   IMAGE_ROOT=... SAVE_ROOT_BASE=... bash run_droid_seg_quality_compare.sh

set -euo pipefail

IMAGE_ROOT="${IMAGE_ROOT:-$HOME/RobotSeg/data/oxe_subset}"
SAVE_ROOT_BASE="${SAVE_ROOT_BASE:-$HOME/RobotSeg/data/oxe_subset_seg_compare}"
CATEGORIES="${CATEGORIES:-arm,gripper}"
DATASET="${DATASET:-droid}"

src="$IMAGE_ROOT/$DATASET"
[[ -d "$src" ]] || { echo "Missing DROID image root: $src" >&2; exit 1; }

cd "$(dirname "$0")"

echo "Input: $src"
echo "Output base: $SAVE_ROOT_BASE"
echo "Categories: $CATEGORIES"
echo

echo "==> native_raw: author API path, native RLDS frames, no guided filter, no arm subtraction"
python inference_auto_qual.py \
  --image_root "$src" \
  --save_root "$SAVE_ROOT_BASE/native_raw/$DATASET" \
  --categories "$CATEGORIES" \
  --save_overlay --save_prob \
  --infer_max_side 0 \
  --no_offload_to_cpu \
  --no_subtract_arm_from_gripper \
  --overwrite

echo "==> native_guided: native RLDS frames + author guided_refine_mask"
python inference_auto_qual.py \
  --image_root "$src" \
  --save_root "$SAVE_ROOT_BASE/native_guided/$DATASET" \
  --categories "$CATEGORIES" \
  --save_overlay --save_prob \
  --guided_filter \
  --infer_max_side 0 \
  --no_offload_to_cpu \
  --no_subtract_arm_from_gripper \
  --overwrite

echo "==> current_pipeline: native RLDS frames + current OXE defaults"
python inference_auto_qual.py \
  --image_root "$src" \
  --save_root "$SAVE_ROOT_BASE/current_pipeline/$DATASET" \
  --categories "$CATEGORIES" \
  --save_overlay --save_prob \
  --infer_max_side 0 \
  --overwrite

echo "Done."
