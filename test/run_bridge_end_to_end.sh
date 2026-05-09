#!/usr/bin/env bash
# Bridge/WidowX end-to-end demo: OXE download, RobotSeg, TCP PnP, optional
# MuJoCo audit if a trossen_wx250s qpos/IK export is available.

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASET="${DATASET:-bridge}"
OXE_ROOT="${OXE_ROOT:-$HOME/RobotSeg/data/oxe_subset_bridge}"
MASK_ROOT="${MASK_ROOT:-$HOME/RobotSeg/data/oxe_subset_seg_bridge/native/current_pipeline}"
CATEGORIES="${CATEGORIES:-arm,gripper}"
NUM_EPISODES="${NUM_EPISODES:-5}"
DOWNLOAD="${DOWNLOAD:-1}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
MAX_FRAMES_PER_SEQ="${MAX_FRAMES_PER_SEQ:-64}"
INFER_MAX_SIDE="${INFER_MAX_SIDE:-0}"
BRIDGE_CAMERA_FOV="${BRIDGE_CAMERA_FOV:-45.0}"
PNP_JSON_NAME="${PNP_JSON_NAME:-pnp_fovy${BRIDGE_CAMERA_FOV}.json}"
PNP_EXTRA_ARGS_STR="${PNP_EXTRA_ARGS_STR:---no_use_observed_flag --no_use_mask_stage_accept --no_reject_fragmented --min_conf 0.0 --min_conf_max 0.0 --min_gripper_area_px 8 --min_post_subtract_area_ratio 0.0}"
SEG_EXTRA_ARGS_STR="${SEG_EXTRA_ARGS_STR:---no_require_gripper_near_arm --relax_gripper_observed}"
RUN_MUJOCO_RENDER="${RUN_MUJOCO_RENDER:-0}"
RUN_VIEWPOINT_CHECK="${RUN_VIEWPOINT_CHECK:-0}"
AUGE_ROOT="${AUGE_ROOT:-$HOME/AugE-Toolkit}"
BRIDGE_MUJOCO_XML_PATH="${BRIDGE_MUJOCO_XML_PATH:-$AUGE_ROOT/robot_xml/trossen_wx250s/scene.xml}"
BRIDGE_MUJOCO_QPOS_KEY="${BRIDGE_MUJOCO_QPOS_KEY:-bridge_all_qpos}"
BRIDGE_MUJOCO_GEOM_GROUPS="${BRIDGE_MUJOCO_GEOM_GROUPS:-0,1,2,3,4,5}"
BRIDGE_VIEWPOINTS_JSON="${BRIDGE_VIEWPOINTS_JSON:-}"

read -r -a PNP_EXTRA_ARGS <<< "$PNP_EXTRA_ARGS_STR"
read -r -a SEG_EXTRA_ARGS <<< "$SEG_EXTRA_ARGS_STR"

if [[ "$DOWNLOAD" == "1" ]]; then
  echo "==> download Bridge OXE subset"
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
echo "==> PnP with Bridge TCP state + viewpoint fovy"
"$PYTHON" "$REPO_ROOT/tools/pnp_oxe.py" \
  --oxe_root "$OXE_ROOT" \
  --mask_root "$MASK_ROOT" \
  --datasets "$DATASET" \
  --pnp_point_source eef_xyz \
  --hfov_deg "$BRIDGE_CAMERA_FOV" \
  --fov_mode vertical \
  --print_intrinsics \
  --pnp_json_name "$PNP_JSON_NAME" \
  "${PNP_EXTRA_ARGS[@]}" \
  --viz

echo
echo "==> exact PnP audit panels"
AUDIT_ARGS=(--no_urdf)
if [[ "$RUN_MUJOCO_RENDER" == "1" && -f "$BRIDGE_MUJOCO_XML_PATH" ]]; then
  AUDIT_ARGS=(
    --mujoco_xml_path "$BRIDGE_MUJOCO_XML_PATH"
    --mujoco_qpos_key "$BRIDGE_MUJOCO_QPOS_KEY"
    --mujoco_geom_groups "$BRIDGE_MUJOCO_GEOM_GROUPS"
  )
else
  echo "    MuJoCo rendering disabled. Bridge RLDS has TCP state but not arm qpos; provide $BRIDGE_MUJOCO_QPOS_KEY via IK to render trossen_wx250s."
fi
"$PYTHON" "$REPO_ROOT/tools/viz_pnp_audit_panel.py" \
  --oxe_root "$OXE_ROOT" \
  --mask_root "$MASK_ROOT" \
  --dataset "$DATASET" \
  --pnp_json_name "$PNP_JSON_NAME" \
  --out_dir_name pnp_audit_bridge \
  "${AUDIT_ARGS[@]}" \
  --skip_bad_pnp

if [[ "$RUN_VIEWPOINT_CHECK" == "1" ]]; then
  if [[ -z "$BRIDGE_VIEWPOINTS_JSON" || ! -f "$BRIDGE_VIEWPOINTS_JSON" || ! -f "$BRIDGE_MUJOCO_XML_PATH" ]]; then
    echo "[warn] set BRIDGE_VIEWPOINTS_JSON and BRIDGE_MUJOCO_XML_PATH to render Bridge viewpoint checks" >&2
  else
    echo
    echo "==> MuJoCo free-camera viewpoint sanity check"
    "$PYTHON" "$REPO_ROOT/tools/viz_fractal_mujoco_viewpoints.py" \
      --oxe_root "$OXE_ROOT" \
      --mask_root "$MASK_ROOT" \
      --dataset "$DATASET" \
      --pnp_json_name "$PNP_JSON_NAME" \
      --mujoco_xml_path "$BRIDGE_MUJOCO_XML_PATH" \
      --viewpoints_json "$BRIDGE_VIEWPOINTS_JSON" \
      --target_mask_dirs 000 001
  fi
fi

echo
echo "==> PnP reprojection summary"
"$PYTHON" "$REPO_ROOT/tools/summarize_pnp_errors.py" \
  --mask_root "$MASK_ROOT" \
  --dataset "$DATASET" \
  --pnp_json_name "$PNP_JSON_NAME"

echo
echo "Done."
echo "Masks:    $MASK_ROOT/$DATASET/episode_XXXX/{000,001,combined}/"
echo "PnP JSON: $MASK_ROOT/$DATASET/episode_XXXX/$PNP_JSON_NAME"
echo "Audit:    $MASK_ROOT/$DATASET/episode_XXXX/pnp_audit_bridge/"
