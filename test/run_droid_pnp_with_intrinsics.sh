#!/usr/bin/env bash
# Build scaled DROID camera intrinsics and run PnP without HFOV fallback.
#
# Usage from repo root:
#   bash test/run_droid_pnp_with_intrinsics.sh
#
# Usage from test/:
#   bash run_droid_pnp_with_intrinsics.sh

set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OXE_ROOT="${OXE_ROOT:-$HOME/RobotSeg/data/oxe_subset}"
MASK_ROOT="${MASK_ROOT:-$HOME/RobotSeg/data/oxe_subset_seg_compare/current_pipeline}"
DATASET="${DATASET:-droid}"
CAMERA="${CAMERA:-exterior_image_1_left}"
PNP_POINT_SOURCE="${PNP_POINT_SOURCE:-all_finger_midpoint}"
K_JSON="${K_JSON:-$HOME/RobotSeg/data/droid_hf_intrinsics_scaled.json}"
CACHE_DIR="${CACHE_DIR:-$HOME/RobotSeg/data/droid_hf_calib}"
RUN_URDF_VIZ="${RUN_URDF_VIZ:-0}"
RUN_PNP_AUDIT="${RUN_PNP_AUDIT:-1}"
URDF_PATH="${URDF_PATH:-$REPO_ROOT/data/urdfs/python-example-droid-dataset/franka_description/panda.urdf}"
URDF_IMAGE_SOURCE="${URDF_IMAGE_SOURCE:-combined}"
URDF_MAX_FRAMES="${URDF_MAX_FRAMES:-32}"
URDF_FRAME_STRIDE="${URDF_FRAME_STRIDE:-1}"
URDF_BACKEND="${URDF_BACKEND:-simple}"
MULTIVIEW_EXTRINSICS_JSON="${MULTIVIEW_EXTRINSICS_JSON:-$REPO_ROOT/pnp_cam2base_multiview.json}"
MULTIVIEW_EXTRINSICS_MODE="${MULTIVIEW_EXTRINSICS_MODE:-rpy_cam2base}"

echo "==> build DROID intrinsics: camera=$CAMERA"
"$PYTHON" "$REPO_ROOT/tools/droid_hf_intrinsics.py" \
  --oxe_root "$OXE_ROOT" \
  --dataset "$DATASET" \
  --camera "$CAMERA" \
  --cache_dir "$CACHE_DIR" \
  --out_json "$K_JSON"

echo
echo "==> pnp with DROID intrinsics only: point_source=$PNP_POINT_SOURCE"
"$PYTHON" "$REPO_ROOT/tools/pnp_oxe.py" \
  --oxe_root "$OXE_ROOT" \
  --mask_root "$MASK_ROOT" \
  --datasets "$DATASET" \
  --K_json "$K_JSON" \
  --pnp_point_source "$PNP_POINT_SOURCE" \
  --urdf_path "$URDF_PATH" \
  --urdf_backend "$URDF_BACKEND" \
  --require_known_intrinsics \
  --print_intrinsics \
  --viz

if [[ "$RUN_PNP_AUDIT" == "1" ]]; then
  echo
  echo "==> exact PnP audit panels"
  AUDIT_EXTRINSICS_ARGS=()
  if [[ -f "$MULTIVIEW_EXTRINSICS_JSON" ]]; then
    AUDIT_EXTRINSICS_ARGS=(
      --extrinsics_json "$MULTIVIEW_EXTRINSICS_JSON"
      --extrinsics_sixd_mode "$MULTIVIEW_EXTRINSICS_MODE"
    )
  else
    echo "[warn] multiview extrinsics JSON not found: $MULTIVIEW_EXTRINSICS_JSON"
  fi
  "$PYTHON" "$REPO_ROOT/tools/viz_pnp_audit_panel.py" \
    --oxe_root "$OXE_ROOT" \
    --mask_root "$MASK_ROOT" \
    --dataset "$DATASET" \
    --urdf_path "$URDF_PATH" \
    --urdf_backend "$URDF_BACKEND" \
    "${AUDIT_EXTRINSICS_ARGS[@]}" \
    --skip_bad_pnp
fi

if [[ "$RUN_URDF_VIZ" == "1" ]]; then
  echo
  echo "==> URDF overlay visualization"
  "$PYTHON" "$REPO_ROOT/tools/viz_cam2base_urdf.py" \
    --oxe_root "$OXE_ROOT" \
    --mask_root "$MASK_ROOT" \
    --dataset "$DATASET" \
    --urdf_path "$URDF_PATH" \
    --urdf_backend "$URDF_BACKEND" \
    --image_source "$URDF_IMAGE_SOURCE" \
    --allow_partial_trajectory \
    --frame_stride "$URDF_FRAME_STRIDE" \
    --max_frames "$URDF_MAX_FRAMES" \
    --skip_bad_pnp \
    --draw_status
fi
