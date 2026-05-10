#!/usr/bin/env bash
# Berkeley Autolab UR5 end-to-end demo: OXE download, RobotSeg, TCP PnP,
# optional MuJoCo UR5 silhouette audit/refinement.

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASET="${DATASET:-berkeley_autolab_ur5}"
OXE_ROOT="${OXE_ROOT:-$HOME/RobotSeg/data/oxe_subset_ur5}"
MASK_ROOT="${MASK_ROOT:-$HOME/RobotSeg/data/oxe_subset_seg_ur5/native/current_pipeline}"
CATEGORIES="${CATEGORIES:-arm,gripper}"
NUM_EPISODES="${NUM_EPISODES:-5}"
DOWNLOAD="${DOWNLOAD:-1}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
MAX_FRAMES_PER_SEQ="${MAX_FRAMES_PER_SEQ:-64}"
INFER_MAX_SIDE="${INFER_MAX_SIDE:-0}"
UR5_CAMERA_FOV="${UR5_CAMERA_FOV:-57.82240163683314}"
PNP_JSON_NAME="${PNP_JSON_NAME:-pnp_fovy${UR5_CAMERA_FOV}.json}"
REFINED_PNP_JSON_NAME="${REFINED_PNP_JSON_NAME:-pnp_fovy${UR5_CAMERA_FOV}_silhouette_refined.json}"
PNP_EXTRA_ARGS_STR="${PNP_EXTRA_ARGS_STR:---no_use_observed_flag --no_use_mask_stage_accept --no_reject_fragmented --min_conf 0.0 --min_conf_max 0.0 --min_gripper_area_px 8 --min_post_subtract_area_ratio 0.0}"
SEG_EXTRA_ARGS_STR="${SEG_EXTRA_ARGS_STR:---no_require_gripper_near_arm --relax_gripper_observed}"
REPAIR_UR5_QPOS="${REPAIR_UR5_QPOS:-1}"
UR5_SOLVE_TOOL_OFFSET="${UR5_SOLVE_TOOL_OFFSET:-1}"
UR5_TOOL_OFFSET_BOUND="${UR5_TOOL_OFFSET_BOUND:-0.30}"
RUN_MUJOCO_RENDER="${RUN_MUJOCO_RENDER:-1}"
RUN_SILHOUETTE_REFINE="${RUN_SILHOUETTE_REFINE:-0}"
RUN_VIEWPOINT_INIT_REFINE="${RUN_VIEWPOINT_INIT_REFINE:-0}"
AUGE_ROOT="${AUGE_ROOT:-$HOME/AugE-Toolkit}"
UR5_MUJOCO_XML_PATH="${UR5_MUJOCO_XML_PATH:-$AUGE_ROOT/robot_xml/universal_robots_ur5e/scene.xml}"
UR5_MUJOCO_QPOS_KEY="${UR5_MUJOCO_QPOS_KEY:-joint_position}"
UR5_MUJOCO_GEOM_GROUPS="${UR5_MUJOCO_GEOM_GROUPS:-2}"
UR5_MUJOCO_GEOM_NAME_INCLUDE="${UR5_MUJOCO_GEOM_NAME_INCLUDE:-}"
UR5_MUJOCO_GEOM_NAME_EXCLUDE="${UR5_MUJOCO_GEOM_NAME_EXCLUDE:-floor|table|desk|wall|world|scene|camera|light|object|prop|box|bin|tray|cloth|pad|plane}"
UR5_VIEWPOINTS_JSON="${UR5_VIEWPOINTS_JSON:-$REPO_ROOT/data/viewpoints/berkeley_autolab_ur5_viewpoints.json}"
VIEWPOINT_REFINED_PNP_JSON_NAME="${VIEWPOINT_REFINED_PNP_JSON_NAME:-pnp_fovy${UR5_CAMERA_FOV}_viewpoint_silhouette_refined.json}"
SIL_REFINE_FRAME_STRIDE="${SIL_REFINE_FRAME_STRIDE:-4}"
SIL_REFINE_MAX_FRAMES="${SIL_REFINE_MAX_FRAMES:-24}"
SIL_REFINE_MAX_EVALS="${SIL_REFINE_MAX_EVALS:-240}"
SIL_REFINE_EXTRA_ARGS_STR="${SIL_REFINE_EXTRA_ARGS_STR:---w_render_to_target 1.0 --w_precision 1.0 --w_area 0.35}"

read -r -a PNP_EXTRA_ARGS <<< "$PNP_EXTRA_ARGS_STR"
read -r -a SEG_EXTRA_ARGS <<< "$SEG_EXTRA_ARGS_STR"
read -r -a SIL_REFINE_EXTRA_ARGS <<< "$SIL_REFINE_EXTRA_ARGS_STR"
MUJOCO_NAME_FILTER_ARGS=()
if [[ -n "$UR5_MUJOCO_GEOM_NAME_INCLUDE" ]]; then
  MUJOCO_NAME_FILTER_ARGS+=(--mujoco_geom_name_include "$UR5_MUJOCO_GEOM_NAME_INCLUDE")
fi
if [[ -n "$UR5_MUJOCO_GEOM_NAME_EXCLUDE" ]]; then
  MUJOCO_NAME_FILTER_ARGS+=(--mujoco_geom_name_exclude "$UR5_MUJOCO_GEOM_NAME_EXCLUDE")
fi

if [[ "$DOWNLOAD" == "1" ]]; then
  echo "==> download UR5 OXE subset"
  "$PYTHON" "$REPO_ROOT/tools/download_oxe_subset.py" \
    --out_dir "$OXE_ROOT" \
    --datasets "$DATASET" \
    --num_episodes "$NUM_EPISODES" \
    --frame_stride "$FRAME_STRIDE"
fi

if [[ "$REPAIR_UR5_QPOS" == "1" ]]; then
  echo
  echo "==> repair UR5 MuJoCo qpos to match AugE convention"
  "$PYTHON" "$REPO_ROOT/tools/repair_ur5_mujoco_qpos.py" \
    --oxe_root "$OXE_ROOT" \
    --dataset "$DATASET"
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
echo "==> PnP with UR5 TCP state + viewpoint fovy"
PNP_OFFSET_ARGS=()
if [[ "$UR5_SOLVE_TOOL_OFFSET" == "1" ]]; then
  echo "    UR5 PnP: --solve_tool_offset --tool_offset_bound $UR5_TOOL_OFFSET_BOUND"
  PNP_OFFSET_ARGS=(--solve_tool_offset --tool_offset_bound "$UR5_TOOL_OFFSET_BOUND")
fi
"$PYTHON" "$REPO_ROOT/tools/pnp_oxe.py" \
  --oxe_root "$OXE_ROOT" \
  --mask_root "$MASK_ROOT" \
  --datasets "$DATASET" \
  --pnp_point_source eef_xyz \
  --hfov_deg "$UR5_CAMERA_FOV" \
  --fov_mode vertical \
  --print_intrinsics \
  --pnp_json_name "$PNP_JSON_NAME" \
  "${PNP_OFFSET_ARGS[@]}" \
  "${PNP_EXTRA_ARGS[@]}" \
  --viz

VIS_PNP_JSON_NAME="$PNP_JSON_NAME"
if [[ "$RUN_SILHOUETTE_REFINE" == "1" ]]; then
  if [[ ! -f "$UR5_MUJOCO_XML_PATH" ]]; then
    echo "[warn] missing UR5 MuJoCo XML: $UR5_MUJOCO_XML_PATH" >&2
  else
    echo
    echo "==> refine UR5 PnP cam2base against MuJoCo silhouettes"
    "$PYTHON" "$REPO_ROOT/tools/refine_cam2base_silhouette.py" \
      --oxe_root "$OXE_ROOT" \
      --mask_root "$MASK_ROOT" \
      --dataset "$DATASET" \
      --pnp_json_name "$PNP_JSON_NAME" \
      --out_pnp_json_name "$REFINED_PNP_JSON_NAME" \
      --mujoco_xml_path "$UR5_MUJOCO_XML_PATH" \
      --mujoco_qpos_key "$UR5_MUJOCO_QPOS_KEY" \
      --mujoco_geom_groups "$UR5_MUJOCO_GEOM_GROUPS" \
      "${MUJOCO_NAME_FILTER_ARGS[@]}" \
      --mask_dirs 000 001 \
      --frame_stride "$SIL_REFINE_FRAME_STRIDE" \
      --max_frames "$SIL_REFINE_MAX_FRAMES" \
      --max_evals "$SIL_REFINE_MAX_EVALS" \
      "${SIL_REFINE_EXTRA_ARGS[@]}" \
      --viz_dir_name silhouette_refined_viz
    VIS_PNP_JSON_NAME="$REFINED_PNP_JSON_NAME"

    if [[ "$RUN_VIEWPOINT_INIT_REFINE" == "1" ]]; then
      echo
      echo "==> refine UR5 from provided viewpoint initialization"
      "$PYTHON" "$REPO_ROOT/tools/refine_cam2base_silhouette.py" \
        --oxe_root "$OXE_ROOT" \
        --mask_root "$MASK_ROOT" \
        --dataset "$DATASET" \
        --pnp_json_name "$PNP_JSON_NAME" \
        --out_pnp_json_name "$VIEWPOINT_REFINED_PNP_JSON_NAME" \
        --init_pose_source best_viewpoint \
        --viewpoints_json "$UR5_VIEWPOINTS_JSON" \
        --mujoco_xml_path "$UR5_MUJOCO_XML_PATH" \
        --mujoco_qpos_key "$UR5_MUJOCO_QPOS_KEY" \
        --mujoco_geom_groups "$UR5_MUJOCO_GEOM_GROUPS" \
        "${MUJOCO_NAME_FILTER_ARGS[@]}" \
        --mask_dirs 000 001 \
        --frame_stride "$SIL_REFINE_FRAME_STRIDE" \
        --max_frames "$SIL_REFINE_MAX_FRAMES" \
        --max_evals "$SIL_REFINE_MAX_EVALS" \
        "${SIL_REFINE_EXTRA_ARGS[@]}" \
        --viz_dir_name silhouette_refined_from_viewpoint_viz
      VIS_PNP_JSON_NAME="$VIEWPOINT_REFINED_PNP_JSON_NAME"
    fi
  fi
fi

echo
echo "==> exact PnP audit panels"
AUDIT_ARGS=(--no_urdf)
if [[ "$RUN_MUJOCO_RENDER" == "1" && -f "$UR5_MUJOCO_XML_PATH" ]]; then
  AUDIT_ARGS=(
    --mujoco_xml_path "$UR5_MUJOCO_XML_PATH"
    --mujoco_qpos_key "$UR5_MUJOCO_QPOS_KEY"
    --mujoco_geom_groups "$UR5_MUJOCO_GEOM_GROUPS"
    "${MUJOCO_NAME_FILTER_ARGS[@]}"
  )
fi
"$PYTHON" "$REPO_ROOT/tools/viz_pnp_audit_panel.py" \
  --oxe_root "$OXE_ROOT" \
  --mask_root "$MASK_ROOT" \
  --dataset "$DATASET" \
  --pnp_json_name "$VIS_PNP_JSON_NAME" \
  --out_dir_name pnp_audit_ur5 \
  "${AUDIT_ARGS[@]}" \
  --skip_bad_pnp

echo
echo "==> PnP reprojection summary: base $PNP_JSON_NAME"
"$PYTHON" "$REPO_ROOT/tools/summarize_pnp_errors.py" \
  --mask_root "$MASK_ROOT" \
  --dataset "$DATASET" \
  --pnp_json_name "$PNP_JSON_NAME"

if [[ "$VIS_PNP_JSON_NAME" != "$PNP_JSON_NAME" ]]; then
  echo
  echo "==> PnP reprojection summary: refined $VIS_PNP_JSON_NAME"
  "$PYTHON" "$REPO_ROOT/tools/summarize_pnp_errors.py" \
    --mask_root "$MASK_ROOT" \
    --dataset "$DATASET" \
    --pnp_json_name "$VIS_PNP_JSON_NAME"
  echo
  echo "==> Silhouette IoU improvement summary"
  "$PYTHON" "$REPO_ROOT/tools/summarize_silhouette_refine.py" \
    --mask_root "$MASK_ROOT" \
    --dataset "$DATASET" \
    --pnp_json_names "$VIS_PNP_JSON_NAME" \
    --labels ur5-pnp-init
fi

if [[ "$RUN_VIEWPOINT_INIT_REFINE" == "1" ]]; then
  echo
  echo "==> PnP reprojection summary: UR5 viewpoint refined $VIEWPOINT_REFINED_PNP_JSON_NAME"
  "$PYTHON" "$REPO_ROOT/tools/summarize_pnp_errors.py" \
    --mask_root "$MASK_ROOT" \
    --dataset "$DATASET" \
    --pnp_json_name "$VIEWPOINT_REFINED_PNP_JSON_NAME"
  echo
  echo "==> Silhouette IoU improvement summary: UR5 viewpoint init"
  "$PYTHON" "$REPO_ROOT/tools/summarize_silhouette_refine.py" \
    --mask_root "$MASK_ROOT" \
    --dataset "$DATASET" \
    --pnp_json_names "$VIEWPOINT_REFINED_PNP_JSON_NAME" \
    --labels ur5-view-init
fi

echo
echo "Done."
echo "Masks:     $MASK_ROOT/$DATASET/episode_XXXX/{000,001,combined}/"
echo "PnP JSON:  $MASK_ROOT/$DATASET/episode_XXXX/$PNP_JSON_NAME"
echo "Audit:     $MASK_ROOT/$DATASET/episode_XXXX/pnp_audit_ur5/"
echo "Viewpoint refined PnP: $MASK_ROOT/$DATASET/episode_XXXX/$VIEWPOINT_REFINED_PNP_JSON_NAME (when RUN_VIEWPOINT_INIT_REFINE=1)"
