#!/usr/bin/env bash
# End-to-end Fractal / Google Everyday Robot demo:
# download RLDS frames at native saved resolution, run RobotSeg without
# spatial pre-downsampling, use AugE/MuJoCo camera_fov, solve TCP PnP,
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
USE_MOGE_INTRINSICS="${USE_MOGE_INTRINSICS:-0}"
FRACTAL_CAMERA_FOV="${FRACTAL_CAMERA_FOV:-57}"
PNP_JSON_NAME="${PNP_JSON_NAME:-pnp_fovy${FRACTAL_CAMERA_FOV}.json}"
SEG_EXTRA_ARGS_STR="${SEG_EXTRA_ARGS_STR:---no_require_gripper_near_arm --relax_gripper_observed}"
PNP_EXTRA_ARGS_STR="${PNP_EXTRA_ARGS_STR:---no_use_observed_flag --no_use_mask_stage_accept --no_reject_fragmented --min_conf 0.0 --min_conf_max 0.0 --min_gripper_area_px 8 --min_post_subtract_area_ratio 0.0}"
RUN_URDF_IK="${RUN_URDF_IK:-0}"
RUN_AUGE_IK="${RUN_AUGE_IK:-0}"
RUN_VIEWPOINT_CHECK="${RUN_VIEWPOINT_CHECK:-0}"
RUN_SILHOUETTE_REFINE="${RUN_SILHOUETTE_REFINE:-0}"
RUN_VIEWPOINT_INIT_REFINE="${RUN_VIEWPOINT_INIT_REFINE:-0}"
REFINED_PNP_JSON_NAME="${REFINED_PNP_JSON_NAME:-pnp_fovy${FRACTAL_CAMERA_FOV}_silhouette_refined.json}"
VIEWPOINT_REFINED_PNP_JSON_NAME="${VIEWPOINT_REFINED_PNP_JSON_NAME:-pnp_fovy${FRACTAL_CAMERA_FOV}_viewpoint_silhouette_refined.json}"
SIL_REFINE_FRAME_STRIDE="${SIL_REFINE_FRAME_STRIDE:-4}"
SIL_REFINE_MAX_FRAMES="${SIL_REFINE_MAX_FRAMES:-24}"
SIL_REFINE_MAX_EVALS="${SIL_REFINE_MAX_EVALS:-240}"
SIL_REFINE_ROT_STEP_DEG="${SIL_REFINE_ROT_STEP_DEG:-8.0}"
SIL_REFINE_TRANS_STEP_M="${SIL_REFINE_TRANS_STEP_M:-0.04}"
AUGE_ROOT="${AUGE_ROOT:-$HOME/AugE-Toolkit}"
GOOGLE_URDF_PATH="${GOOGLE_URDF_PATH:-$HOME/RobotSeg/data/urdfs/google_robot/google_robot_description/urdf/google_robot.urdf}"
GOOGLE_URDF_BACKEND="${GOOGLE_URDF_BACKEND:-yourdfpy}"
GOOGLE_MUJOCO_XML_PATH="${GOOGLE_MUJOCO_XML_PATH:-$AUGE_ROOT/robot_xml/google_robot/scene.xml}"
GOOGLE_EE_LINK="${GOOGLE_EE_LINK:-}"
GOOGLE_JOINT_NAMES_STR="${GOOGLE_JOINT_NAMES_STR:-joint_torso joint_shoulder joint_bicep joint_elbow joint_forearm joint_wrist joint_gripper}"
GOOGLE_GRIPPER_JOINT_NAMES_STR="${GOOGLE_GRIPPER_JOINT_NAMES_STR:-joint_finger_right joint_finger_left}"

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

if [[ "$RUN_AUGE_IK" == "1" ]]; then
  echo
  echo "==> solve Google Robot joints from Fractal TCP with AugE MuJoCo IK"
  "$PYTHON" "$REPO_ROOT/tools/export_fractal_google_robot_ik.py" \
    --oxe_root "$OXE_ROOT" \
    --dataset "$DATASET" \
    --auge_root "$AUGE_ROOT"
fi

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
echo "==> PnP with Fractal TCP state + AugE/MuJoCo camera_fov"
echo "    Fractal PnP gates: $PNP_EXTRA_ARGS_STR"
PNP_INTRINSIC_ARGS=(
  --hfov_deg "$FRACTAL_CAMERA_FOV"
  --fov_mode vertical
)
if [[ "$USE_MOGE_INTRINSICS" == "1" ]]; then
  PNP_INTRINSIC_ARGS=(
    --moge_intrinsics
    --moge_device "$MOGE_DEVICE"
    --no_hfov_fallback
  )
fi
"$PYTHON" "$REPO_ROOT/tools/pnp_oxe.py" \
  --oxe_root "$OXE_ROOT" \
  --mask_root "$MASK_ROOT" \
  --datasets "$DATASET" \
  --pnp_point_source eef_xyz \
  "${PNP_INTRINSIC_ARGS[@]}" \
  --print_intrinsics \
  --pnp_json_name "$PNP_JSON_NAME" \
  "${PNP_EXTRA_ARGS[@]}" \
  --viz

VIS_PNP_JSON_NAME="$PNP_JSON_NAME"
if [[ "$RUN_SILHOUETTE_REFINE" == "1" ]]; then
  if [[ "$RUN_AUGE_IK" != "1" || ! -f "$GOOGLE_MUJOCO_XML_PATH" ]]; then
    echo "[warn] silhouette refinement needs RUN_AUGE_IK=1 and GOOGLE_MUJOCO_XML_PATH=$GOOGLE_MUJOCO_XML_PATH" >&2
  else
    echo
    echo "==> refine PnP cam2base against MuJoCo Google Robot body silhouettes"
    "$PYTHON" "$REPO_ROOT/tools/refine_cam2base_silhouette.py" \
      --oxe_root "$OXE_ROOT" \
      --mask_root "$MASK_ROOT" \
      --dataset "$DATASET" \
      --pnp_json_name "$PNP_JSON_NAME" \
      --out_pnp_json_name "$REFINED_PNP_JSON_NAME" \
      --mujoco_xml_path "$GOOGLE_MUJOCO_XML_PATH" \
      --mask_dirs 000 001 \
      --frame_stride "$SIL_REFINE_FRAME_STRIDE" \
      --max_frames "$SIL_REFINE_MAX_FRAMES" \
      --max_evals "$SIL_REFINE_MAX_EVALS" \
      --rot_step_deg "$SIL_REFINE_ROT_STEP_DEG" \
      --trans_step_m "$SIL_REFINE_TRANS_STEP_M" \
      --viz_dir_name silhouette_refined_viz
    VIS_PNP_JSON_NAME="$REFINED_PNP_JSON_NAME"

    if [[ "$RUN_VIEWPOINT_INIT_REFINE" == "1" ]]; then
      echo
      echo "==> refine from best AugE viewpoint IoU initialization"
      "$PYTHON" "$REPO_ROOT/tools/refine_cam2base_silhouette.py" \
        --oxe_root "$OXE_ROOT" \
        --mask_root "$MASK_ROOT" \
        --dataset "$DATASET" \
        --pnp_json_name "$PNP_JSON_NAME" \
        --out_pnp_json_name "$VIEWPOINT_REFINED_PNP_JSON_NAME" \
        --init_pose_source best_viewpoint \
        --mujoco_xml_path "$GOOGLE_MUJOCO_XML_PATH" \
        --mask_dirs 000 001 \
        --frame_stride "$SIL_REFINE_FRAME_STRIDE" \
        --max_frames "$SIL_REFINE_MAX_FRAMES" \
        --max_evals "$SIL_REFINE_MAX_EVALS" \
        --rot_step_deg "$SIL_REFINE_ROT_STEP_DEG" \
        --trans_step_m "$SIL_REFINE_TRANS_STEP_M" \
        --viz_dir_name silhouette_refined_from_viewpoint_viz
    fi
  fi
fi

echo
echo "==> TCP projection panels: projected Fractal observation['state'][:3] vs gripper centroid"
"$PYTHON" "$REPO_ROOT/tools/viz_eef_projection.py" \
  --oxe_root "$OXE_ROOT" \
  --mask_root "$MASK_ROOT" \
  --dataset "$DATASET" \
  --pnp_json_name "$VIS_PNP_JSON_NAME" \
  --frames_from_pnp kept \
  --frame_stride 1 \
  --max_frames 0 \
  --candidate_layout panels \
  --draw_candidates eef_xyz \
  --no_urdf \
  --out_dir_name tcp_pnp_candidate_compare

echo
echo "==> exact PnP audit panels"
AUDIT_URDF_ARGS=(--no_urdf)
if [[ "$RUN_AUGE_IK" == "1" && -f "$GOOGLE_MUJOCO_XML_PATH" ]]; then
  read -r -a GOOGLE_JOINT_NAMES <<< "$GOOGLE_JOINT_NAMES_STR"
  read -r -a GOOGLE_GRIPPER_JOINT_NAMES <<< "$GOOGLE_GRIPPER_JOINT_NAMES_STR"
  AUDIT_URDF_ARGS=(
    --mujoco_xml_path "$GOOGLE_MUJOCO_XML_PATH"
    --arm_joint_names "${GOOGLE_JOINT_NAMES[@]}"
    --gripper_joint_names "${GOOGLE_GRIPPER_JOINT_NAMES[@]}"
    --gripper_open_rad 0.333
    --gripper_closed_rad 1.0
  )
else
  echo "    robot rendering disabled. Set RUN_AUGE_IK=1 and provide GOOGLE_MUJOCO_XML_PATH to render the articulated Google Robot."
fi
"$PYTHON" "$REPO_ROOT/tools/viz_pnp_audit_panel.py" \
  --oxe_root "$OXE_ROOT" \
  --mask_root "$MASK_ROOT" \
  --dataset "$DATASET" \
  --pnp_json_name "$VIS_PNP_JSON_NAME" \
  --out_dir_name pnp_audit_fractal \
  "${AUDIT_URDF_ARGS[@]}" \
  --skip_bad_pnp

if [[ "$RUN_VIEWPOINT_CHECK" == "1" ]]; then
  if [[ ! -f "$GOOGLE_MUJOCO_XML_PATH" ]]; then
    echo "[warn] Google Robot MuJoCo XML not found: $GOOGLE_MUJOCO_XML_PATH" >&2
  else
    echo
    echo "==> MuJoCo free-camera viewpoint sanity check"
    "$PYTHON" "$REPO_ROOT/tools/viz_fractal_mujoco_viewpoints.py" \
      --oxe_root "$OXE_ROOT" \
      --mask_root "$MASK_ROOT" \
      --dataset "$DATASET" \
      --pnp_json_name "$VIS_PNP_JSON_NAME" \
      --mujoco_xml_path "$GOOGLE_MUJOCO_XML_PATH"
  fi
fi

if [[ "$RUN_URDF_IK" == "1" ]]; then
  if [[ ! -f "$GOOGLE_URDF_PATH" ]]; then
    echo "[warn] Google Robot URDF not found: $GOOGLE_URDF_PATH" >&2
    echo "       Run: bash tools/setup_google_robot_urdf.sh" >&2
  else
    echo
    echo "==> Google Robot URDF overlay via TCP IK"
    IK_ARGS=()
    if [[ -n "$GOOGLE_EE_LINK" ]]; then
      IK_ARGS+=(--ee_link "$GOOGLE_EE_LINK")
    fi
    if [[ -n "$GOOGLE_JOINT_NAMES_STR" ]]; then
      read -r -a GOOGLE_JOINT_NAMES <<< "$GOOGLE_JOINT_NAMES_STR"
      IK_ARGS+=(--joint_names "${GOOGLE_JOINT_NAMES[@]}")
    fi
    "$PYTHON" "$REPO_ROOT/tools/viz_urdf_from_tcp_ik.py" \
      --oxe_root "$OXE_ROOT" \
      --mask_root "$MASK_ROOT" \
      --dataset "$DATASET" \
      --pnp_json_name "$VIS_PNP_JSON_NAME" \
      --urdf_path "$GOOGLE_URDF_PATH" \
      --urdf_backend "$GOOGLE_URDF_BACKEND" \
      --skip_bad_pnp \
      "${IK_ARGS[@]}"
  fi
fi

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
fi

if [[ "$RUN_VIEWPOINT_INIT_REFINE" == "1" ]]; then
  echo
  echo "==> PnP reprojection summary: best-viewpoint refined $VIEWPOINT_REFINED_PNP_JSON_NAME"
  "$PYTHON" "$REPO_ROOT/tools/summarize_pnp_errors.py" \
    --mask_root "$MASK_ROOT" \
    --dataset "$DATASET" \
    --pnp_json_name "$VIEWPOINT_REFINED_PNP_JSON_NAME"
fi

if [[ "$RUN_SILHOUETTE_REFINE" == "1" ]]; then
  echo
  echo "==> Silhouette IoU improvement summary"
  SIL_SUMMARY_NAMES=("$REFINED_PNP_JSON_NAME")
  SIL_SUMMARY_LABELS=("pnp-init")
  if [[ "$RUN_VIEWPOINT_INIT_REFINE" == "1" ]]; then
    SIL_SUMMARY_NAMES+=("$VIEWPOINT_REFINED_PNP_JSON_NAME")
    SIL_SUMMARY_LABELS+=("view-init")
  fi
  "$PYTHON" "$REPO_ROOT/tools/summarize_silhouette_refine.py" \
    --mask_root "$MASK_ROOT" \
    --dataset "$DATASET" \
    --pnp_json_names "${SIL_SUMMARY_NAMES[@]}" \
    --labels "${SIL_SUMMARY_LABELS[@]}"
fi

echo
echo "Done."
echo "Masks:       $MASK_ROOT/$DATASET/episode_XXXX/{000,001,combined}/"
echo "PnP JSON:    $MASK_ROOT/$DATASET/episode_XXXX/$PNP_JSON_NAME"
echo "Refined PnP: $MASK_ROOT/$DATASET/episode_XXXX/$REFINED_PNP_JSON_NAME (when RUN_SILHOUETTE_REFINE=1)"
echo "Viewpoint refined PnP: $MASK_ROOT/$DATASET/episode_XXXX/$VIEWPOINT_REFINED_PNP_JSON_NAME (when RUN_VIEWPOINT_INIT_REFINE=1)"
echo "TCP panels:  $MASK_ROOT/$DATASET/episode_XXXX/tcp_pnp_candidate_compare/"
echo "PnP audit:   $MASK_ROOT/$DATASET/episode_XXXX/pnp_audit_fractal/"
echo "Viewpoints:  $MASK_ROOT/$DATASET/episode_XXXX/mujoco_viewpoint_check/ (when RUN_VIEWPOINT_CHECK=1)"
echo "URDF IK:     $MASK_ROOT/$DATASET/episode_XXXX/urdf_tcp_ik_overlay/ (when RUN_URDF_IK=1)"
echo "Silhouette refinement settings: frame_stride=$SIL_REFINE_FRAME_STRIDE max_frames=$SIL_REFINE_MAX_FRAMES max_evals=$SIL_REFINE_MAX_EVALS"
