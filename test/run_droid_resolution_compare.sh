#!/usr/bin/env bash
# Compare RobotSeg masks on low-res DROID RLDS frames vs raw DROID MP4 frames.
#
# This downloads only the matching raw MP4 camera files when DOWNLOAD_RAW=1,
# extracts the same sampled trajectory rows, then runs the current RobotSeg
# mask pipeline into separate save roots.
#
# Usage:
#   DOWNLOAD_RAW=1 bash test/run_droid_resolution_compare.sh
#   MAX_FRAMES_PER_SEQ=16 CUDA_VISIBLE_DEVICES=1 bash test/run_droid_resolution_compare.sh

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OXE_ROOT="${OXE_ROOT:-$HOME/RobotSeg/data/oxe_subset}"
RAW_FRAME_ROOT="${RAW_FRAME_ROOT:-$HOME/RobotSeg/data/oxe_subset_droid_raw_mp4}"
RAW_MP4_ROOT="${RAW_MP4_ROOT:-$HOME/RobotSeg/data/droid_raw_mp4}"
SAVE_ROOT_BASE="${SAVE_ROOT_BASE:-$HOME/RobotSeg/data/oxe_subset_seg_resolution_compare}"
CACHE_DIR="${CACHE_DIR:-$HOME/RobotSeg/data/droid_hf_calib}"
DATASET="${DATASET:-droid}"
CAMERA="${CAMERA:-exterior_image_1_left}"
CATEGORIES="${CATEGORIES:-arm,gripper}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
MAX_FRAMES_PER_SEQ="${MAX_FRAMES_PER_SEQ:-32}"
DOWNLOAD_RAW="${DOWNLOAD_RAW:-0}"
GPU_ONLY="${GPU_ONLY:-1}"

DOWNLOAD_ARGS=()
if [[ "$DOWNLOAD_RAW" == "1" ]]; then
  DOWNLOAD_ARGS=(--download)
fi

OFFLOAD_ARGS=()
if [[ "$GPU_ONLY" == "1" ]]; then
  OFFLOAD_ARGS=(--no_offload_to_cpu)
fi

echo "==> extract matching raw-MP4 frames"
"$PYTHON" "$REPO_ROOT/tools/extract_droid_raw_mp4_frames.py" \
  --oxe_root "$OXE_ROOT" \
  --dataset "$DATASET" \
  --out_root "$RAW_FRAME_ROOT" \
  --raw_root "$RAW_MP4_ROOT" \
  --cache_dir "$CACHE_DIR" \
  --camera "$CAMERA" \
  --frame_stride "$FRAME_STRIDE" \
  --max_frames_per_seq "$MAX_FRAMES_PER_SEQ" \
  "${DOWNLOAD_ARGS[@]}"

RAW_SUMMARY="$RAW_FRAME_ROOT/$DATASET/raw_mp4_extract_summary.json"
RAW_OK_COUNT="$("$PYTHON" - "$RAW_SUMMARY" <<'PY'
import json, sys
path = sys.argv[1]
with open(path) as f:
    data = json.load(f)
print(sum(1 for r in data.get("results", []) if r.get("status") == "ok" and r.get("num_saved", 0) > 0))
PY
)"
if [[ "$RAW_OK_COUNT" == "0" ]]; then
  echo "No raw MP4 frames were extracted. Stopping before RobotSeg." >&2
  echo "Inspect: $RAW_SUMMARY" >&2
  exit 1
fi

cd "$REPO_ROOT/test"

echo
echo "==> RobotSeg on RLDS frames"
"$PYTHON" inference_auto_qual.py \
  --image_root "$OXE_ROOT/$DATASET" \
  --save_root "$SAVE_ROOT_BASE/rlds/current_pipeline/$DATASET" \
  --categories "$CATEGORIES" \
  --save_overlay --save_prob \
  --infer_max_side 0 \
  --frame_stride "$FRAME_STRIDE" \
  --max_frames_per_seq "$MAX_FRAMES_PER_SEQ" \
  "${OFFLOAD_ARGS[@]}" \
  --overwrite

echo
echo "==> RobotSeg on raw-MP4 frames"
"$PYTHON" inference_auto_qual.py \
  --image_root "$RAW_FRAME_ROOT/$DATASET" \
  --save_root "$SAVE_ROOT_BASE/raw_mp4/current_pipeline/$DATASET" \
  --categories "$CATEGORIES" \
  --save_overlay --save_prob \
  --infer_max_side 0 \
  --frame_stride 1 \
  --max_frames_per_seq 0 \
  "${OFFLOAD_ARGS[@]}" \
  --overwrite

echo
echo "Done."
echo "RLDS masks:    $SAVE_ROOT_BASE/rlds/current_pipeline/$DATASET"
echo "Raw MP4 masks: $SAVE_ROOT_BASE/raw_mp4/current_pipeline/$DATASET"
