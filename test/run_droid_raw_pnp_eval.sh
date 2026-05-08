#!/usr/bin/env bash
# Run PnP on raw-MP4 DROID RobotSeg masks and compare to multiview GT.
# No silhouette/differentiable refinement is run here.

set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OXE_ROOT="${OXE_ROOT:-$HOME/RobotSeg/data/oxe_subset_droid_raw_mp4}"
MASK_ROOT="${MASK_ROOT:-$HOME/RobotSeg/data/oxe_subset_seg_resolution_compare/raw_mp4/current_pipeline}"
LOWRES_OXE_ROOT="${LOWRES_OXE_ROOT:-$HOME/RobotSeg/data/oxe_subset}"
LOWRES_MASK_ROOT="${LOWRES_MASK_ROOT:-$HOME/RobotSeg/data/oxe_subset_seg_resolution_compare/rlds/current_pipeline}"
DATASET="${DATASET:-droid}"
CAMERA="${CAMERA:-exterior_image_1_left}"
PNP_POINT_SOURCE="${PNP_POINT_SOURCE:-all_finger_midpoint}"
K_JSON="${K_JSON:-$HOME/RobotSeg/data/droid_hf_intrinsics_scaled_raw_mp4.json}"
LOWRES_K_JSON="${LOWRES_K_JSON:-$HOME/RobotSeg/data/droid_hf_intrinsics_scaled_rlds.json}"
CACHE_DIR="${CACHE_DIR:-$HOME/RobotSeg/data/droid_hf_calib}"
URDF_PATH="${URDF_PATH:-$REPO_ROOT/data/urdfs/python-example-droid-dataset/franka_description/panda.urdf}"
URDF_BACKEND="${URDF_BACKEND:-simple}"
MULTIVIEW_EXTRINSICS_JSON="${MULTIVIEW_EXTRINSICS_JSON:-$REPO_ROOT/pnp_cam2base_multiview.json}"
MULTIVIEW_EXTRINSICS_MODE="${MULTIVIEW_EXTRINSICS_MODE:-rpy_cam2base}"
PNP_JSON_NAME="${PNP_JSON_NAME:-pnp.json}"
RUN_LOWRES_PNP="${RUN_LOWRES_PNP:-1}"

echo "==> build DROID intrinsics at raw-MP4 frame resolution"
"$PYTHON" "$REPO_ROOT/tools/droid_hf_intrinsics.py" \
  --oxe_root "$OXE_ROOT" \
  --dataset "$DATASET" \
  --camera "$CAMERA" \
  --cache_dir "$CACHE_DIR" \
  --out_json "$K_JSON"

echo
echo "==> PnP on raw-MP4 masks: point_source=$PNP_POINT_SOURCE"
"$PYTHON" "$REPO_ROOT/tools/pnp_oxe.py" \
  --oxe_root "$OXE_ROOT" \
  --mask_root "$MASK_ROOT" \
  --datasets "$DATASET" \
  --K_json "$K_JSON" \
  --require_known_intrinsics \
  --pnp_point_source "$PNP_POINT_SOURCE" \
  --urdf_path "$URDF_PATH" \
  --urdf_backend "$URDF_BACKEND" \
  --print_intrinsics \
  --pnp_json_name "$PNP_JSON_NAME" \
  --viz

if [[ -f "$MULTIVIEW_EXTRINSICS_JSON" ]]; then
  echo
  echo "==> compare raw-MP4 PnP to multiview"
  "$PYTHON" "$REPO_ROOT/tools/compare_cam2base_extrinsics.py" \
    --oxe_root "$OXE_ROOT" \
    --mask_root "$MASK_ROOT" \
    --dataset "$DATASET" \
    --pnp_json_name "$PNP_JSON_NAME" \
    --reference_json "$MULTIVIEW_EXTRINSICS_JSON" \
    --sixd_mode "$MULTIVIEW_EXTRINSICS_MODE"

  echo
  echo "==> render raw-MP4 PnP audit panels"
  "$PYTHON" "$REPO_ROOT/tools/viz_pnp_audit_panel.py" \
    --oxe_root "$OXE_ROOT" \
    --mask_root "$MASK_ROOT" \
    --dataset "$DATASET" \
    --urdf_path "$URDF_PATH" \
    --urdf_backend "$URDF_BACKEND" \
    --extrinsics_json "$MULTIVIEW_EXTRINSICS_JSON" \
    --extrinsics_sixd_mode "$MULTIVIEW_EXTRINSICS_MODE" \
    --pnp_json_name "$PNP_JSON_NAME" \
    --out_dir_name pnp_audit_raw_mp4 \
    --skip_bad_pnp
fi

if [[ "$RUN_LOWRES_PNP" == "1" && -d "$LOWRES_MASK_ROOT/$DATASET" ]]; then
  echo
  echo "==> build DROID intrinsics at low-res RLDS frame resolution"
  "$PYTHON" "$REPO_ROOT/tools/droid_hf_intrinsics.py" \
    --oxe_root "$LOWRES_OXE_ROOT" \
    --dataset "$DATASET" \
    --camera "$CAMERA" \
    --cache_dir "$CACHE_DIR" \
    --out_json "$LOWRES_K_JSON"

  echo
  echo "==> PnP on low-res RLDS masks: point_source=$PNP_POINT_SOURCE"
  "$PYTHON" "$REPO_ROOT/tools/pnp_oxe.py" \
    --oxe_root "$LOWRES_OXE_ROOT" \
    --mask_root "$LOWRES_MASK_ROOT" \
    --datasets "$DATASET" \
    --K_json "$LOWRES_K_JSON" \
    --require_known_intrinsics \
    --pnp_point_source "$PNP_POINT_SOURCE" \
    --urdf_path "$URDF_PATH" \
    --urdf_backend "$URDF_BACKEND" \
    --print_intrinsics \
    --pnp_json_name "$PNP_JSON_NAME" \
    --viz
fi

if [[ -f "$MULTIVIEW_EXTRINSICS_JSON" && -d "$LOWRES_MASK_ROOT/$DATASET" ]]; then
  echo
  echo "==> compare low-res RLDS PnP to multiview"
  "$PYTHON" "$REPO_ROOT/tools/compare_cam2base_extrinsics.py" \
    --oxe_root "$LOWRES_OXE_ROOT" \
    --mask_root "$LOWRES_MASK_ROOT" \
    --dataset "$DATASET" \
    --pnp_json_name "$PNP_JSON_NAME" \
    --reference_json "$MULTIVIEW_EXTRINSICS_JSON" \
    --sixd_mode "$MULTIVIEW_EXTRINSICS_MODE"
fi

echo
echo "Raw comparison JSON: $MASK_ROOT/$DATASET/cam2base_compare_${PNP_JSON_NAME%.json}.json"
echo "Raw audit panels:    $MASK_ROOT/$DATASET/episode_XXXX/pnp_audit_raw_mp4/"
