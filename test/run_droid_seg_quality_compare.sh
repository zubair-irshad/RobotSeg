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
#   MAX_FRAMES_PER_SEQ=32 bash run_droid_seg_quality_compare.sh
#   GPU_ONLY=0 bash run_droid_seg_quality_compare.sh   # allow CPU offload fallback

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON="${PYTHON:-python}"
IMAGE_ROOT="${IMAGE_ROOT:-$HOME/RobotSeg/data/oxe_subset}"
SAVE_ROOT_BASE="${SAVE_ROOT_BASE:-$HOME/RobotSeg/data/oxe_subset_seg_compare}"
CATEGORIES="${CATEGORIES:-arm,gripper}"
DATASET="${DATASET:-droid}"
GPU_ONLY="${GPU_ONLY:-1}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
MAX_FRAMES_PER_SEQ="${MAX_FRAMES_PER_SEQ:-32}"

src="$IMAGE_ROOT/$DATASET"
[[ -d "$src" ]] || { echo "Missing DROID image root: $src" >&2; exit 1; }

cd "$(dirname "$0")"

echo "Input: $src"
echo "Output base: $SAVE_ROOT_BASE"
echo "Categories: $CATEGORIES"
echo "GPU only: $GPU_ONLY"
echo "Frame stride: $FRAME_STRIDE"
echo "Max frames per episode: $MAX_FRAMES_PER_SEQ"
if [[ "$GPU_ONLY" == "1" ]]; then
  echo "Note: GPU-only mode matches the author scripts but can OOM on long episodes unless the GPU is mostly free."
fi
echo

if "$PYTHON" - <<'PY'
import cv2
raise SystemExit(0 if hasattr(getattr(cv2, "ximgproc", None), "guidedFilter") else 1)
PY
then
  HAS_GUIDED_FILTER=1
else
  HAS_GUIDED_FILTER=0
fi

OFFLOAD_ARGS=()
if [[ "$GPU_ONLY" == "1" ]]; then
  OFFLOAD_ARGS=(--no_offload_to_cpu)
fi

echo "==> native_raw: author API path, native RLDS frames, no guided filter, no arm subtraction"
"$PYTHON" inference_auto_qual.py \
  --image_root "$src" \
  --save_root "$SAVE_ROOT_BASE/native_raw/$DATASET" \
  --categories "$CATEGORIES" \
  --save_overlay --save_prob \
  --infer_max_side 0 \
  --frame_stride "$FRAME_STRIDE" \
  --max_frames_per_seq "$MAX_FRAMES_PER_SEQ" \
  "${OFFLOAD_ARGS[@]}" \
  --no_subtract_arm_from_gripper \
  --overwrite

if [[ "$HAS_GUIDED_FILTER" == "1" ]]; then
  echo "==> native_guided: native RLDS frames + author guided_refine_mask"
  "$PYTHON" inference_auto_qual.py \
    --image_root "$src" \
    --save_root "$SAVE_ROOT_BASE/native_guided/$DATASET" \
    --categories "$CATEGORIES" \
    --save_overlay --save_prob \
    --guided_filter \
    --infer_max_side 0 \
    --frame_stride "$FRAME_STRIDE" \
    --max_frames_per_seq "$MAX_FRAMES_PER_SEQ" \
    "${OFFLOAD_ARGS[@]}" \
    --no_subtract_arm_from_gripper \
    --overwrite
else
  echo "==> native_guided: skipped; cv2.ximgproc.guidedFilter is unavailable"
fi

echo "==> current_pipeline: native RLDS frames + current OXE defaults"
"$PYTHON" inference_auto_qual.py \
  --image_root "$src" \
  --save_root "$SAVE_ROOT_BASE/current_pipeline/$DATASET" \
  --categories "$CATEGORIES" \
  --save_overlay --save_prob \
  --infer_max_side 0 \
  --frame_stride "$FRAME_STRIDE" \
  --max_frames_per_seq "$MAX_FRAMES_PER_SEQ" \
  "${OFFLOAD_ARGS[@]}" \
  --overwrite

echo "Done."
