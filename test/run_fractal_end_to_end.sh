#!/usr/bin/env bash
# End-to-end Fractal / Google Everyday Robot demo:
# download RLDS frames at native saved resolution, run RobotSeg without
# spatial pre-downsampling, estimate intrinsics with MoGe, solve TCP PnP,
# and visualize projected TCP against RobotSeg gripper centroids.

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASET="${DATASET:-fractal20220817_data}"
OXE_ROOT="${OXE_ROOT:-$HOME/RobotSeg/data/oxe_subset_fractal}"
MASK_ROOT="${MASK_ROOT:-$HOME/RobotSeg/data/oxe_subset_seg_fractal/native/current_pipeline}"
CATEGORIES="${CATEGORIES:-arm,gripper}"
NUM_EPISODES="${NUM_EPISODES:-5}"
DOWNLOAD="${DOWNLOAD:-1}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
MAX_FRAMES_PER_SEQ="${MAX_FRAMES_PER_SEQ:-64}"
INFER_MAX_SIDE="${INFER_MAX_SIDE:-0}"
MOGE_DEVICE="${MOGE_DEVICE:-cuda}"
PNP_JSON_NAME="${PNP_JSON_NAME:-pnp_moge.json}"
SEG_EXTRA_ARGS_STR="${SEG_EXTRA_ARGS_STR:---no_require_gripper_near_arm}"
PNP_EXTRA_ARGS_STR="${PNP_EXTRA_ARGS_STR:---no_use_observed_flag --no_use_mask_stage_accept --no_reject_fragmented --min_conf 0.0 --min_conf_max 0.0 --min_gripper_area_px 8 --min_post_subtract_area_ratio 0.0}"

read -r -a SEG_EXTRA_ARGS <<< "$SEG_EXTRA_ARGS_STR"
read -r -a PNP_EXTRA_ARGS <<< "$PNP_EXTRA_ARGS_STR"

if [[ "$DOWNLOAD" == "1" ]]; then
  echo "==> download Fractal RLDS subset: image key is native RLDS observation['image']"
  "$PYTHON" "$REPO_ROOT/tools/download_oxe_subset.py" \
    --out_dir "$OXE_ROOT" \
    --datasets "$DATASET" \
    --num_episodes "$NUM_EPISODES" \
    --frame_stride "$FRAME_STRIDE"
fi

echo
echo "==> inspect saved frame resolution"
"$PYTHON" - "$OXE_ROOT/$DATASET" <<'PY'
import sys
from pathlib import Path
from PIL import Image
root = Path(sys.argv[1])
for ep in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("episode_")):
    imgs = sorted((ep / "frames").glob("*.jpg"))
    if not imgs:
        print(f"{ep.name}: no frames")
        continue
    im = Image.open(imgs[0])
    print(f"{ep.name}: {im.width}x{im.height} frames={len(imgs)}")
PY

echo
echo "==> RobotSeg at native spatial resolution: infer_max_side=$INFER_MAX_SIDE"
echo "    Fractal mask gates: $SEG_EXTRA_ARGS_STR"
(cd "$REPO_ROOT/test" && \
  "$PYTHON" inference_auto_qual.py \
    --image_root "$OXE_ROOT/$DATASET" \
    --save_root "$MASK_ROOT/$DATASET" \
    --categories "$CATEGORIES" \
    --save_overlay --save_prob \
    --infer_max_side "$INFER_MAX_SIDE" \
    --frame_stride 1 \
    --max_frames_per_seq "$MAX_FRAMES_PER_SEQ" \
    --no_offload_to_cpu \
    "${SEG_EXTRA_ARGS[@]}" \
    --overwrite)

echo
echo "==> PnP with Fractal TCP state + MoGe intrinsics"
echo "    Fractal PnP gates: $PNP_EXTRA_ARGS_STR"
"$PYTHON" "$REPO_ROOT/tools/pnp_oxe.py" \
  --oxe_root "$OXE_ROOT" \
  --mask_root "$MASK_ROOT" \
  --datasets "$DATASET" \
  --pnp_point_source eef_xyz \
  --moge_intrinsics \
  --moge_device "$MOGE_DEVICE" \
  --no_hfov_fallback \
  --print_intrinsics \
  --pnp_json_name "$PNP_JSON_NAME" \
  "${PNP_EXTRA_ARGS[@]}" \
  --viz

echo
echo "==> TCP projection panels: projected Fractal base_pose_tool_reached vs gripper centroid"
"$PYTHON" "$REPO_ROOT/tools/viz_eef_projection.py" \
  --oxe_root "$OXE_ROOT" \
  --mask_root "$MASK_ROOT" \
  --dataset "$DATASET" \
  --pnp_json_name "$PNP_JSON_NAME" \
  --frames_from_pnp kept \
  --frame_stride 1 \
  --max_frames 0 \
  --candidate_layout panels \
  --draw_candidates eef_xyz \
  --no_urdf \
  --out_dir_name tcp_pnp_candidate_compare

echo
echo "==> exact PnP audit panels without URDF body rendering"
"$PYTHON" "$REPO_ROOT/tools/viz_pnp_audit_panel.py" \
  --oxe_root "$OXE_ROOT" \
  --mask_root "$MASK_ROOT" \
  --dataset "$DATASET" \
  --pnp_json_name "$PNP_JSON_NAME" \
  --out_dir_name pnp_audit_fractal \
  --no_urdf \
  --skip_bad_pnp

echo
echo "Done."
echo "Masks:       $MASK_ROOT/$DATASET/episode_XXXX/{000,001,combined}/"
echo "PnP JSON:    $MASK_ROOT/$DATASET/episode_XXXX/$PNP_JSON_NAME"
echo "TCP panels:  $MASK_ROOT/$DATASET/episode_XXXX/tcp_pnp_candidate_compare/"
echo "PnP audit:   $MASK_ROOT/$DATASET/episode_XXXX/pnp_audit_fractal/"
