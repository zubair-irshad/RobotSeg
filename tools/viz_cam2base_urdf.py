"""Validate cam2base by overlaying a URDF-rendered robot silhouette.

Reads pnp.json (written by tools/pnp_oxe.py) for one or more episodes and,
for each frame with valid joint state, renders the URDF silhouette through
the estimated cam2base and overlays it on the original image. Saves
side-by-side comparisons to <episode>/cam2base_viz/.

Usage (Franka / taco_play):
  python tools/viz_cam2base_urdf.py \
    --oxe_root  ~/RobotSeg/data/oxe_subset \
    --mask_root ~/RobotSeg/data/oxe_subset_seg \
    --dataset taco_play \
    --urdf /path/to/panda.urdf \
    --mesh_dir /path/to/franka_description/meshes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import sys

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from urdf_robot_masker import URDFRobotMasker  # noqa: E402


# Per-dataset: how to dig joint angles + a normalized gripper position
# (0 = open, 1 = closed) out of trajectory["state"].
#
# gripper_kind:
#   "width_m"        : state[idx] is finger_width in meters (Franka).
#                      Mapped to 0..1 via [gripper_open_rad, gripper_closed_rad].
#   "binary_closed"  : state[idx] in {0,1}, 1 == closed.
JOINT_DIMS = {
    "taco_play": {"arm": (7, 14), "gripper_idx": 6,
                  "gripper_kind": "width_m"},
    "berkeley_autolab_ur5": {"arm": (0, 6), "gripper_idx": 6,
                              "gripper_kind": "binary_closed"},
}

# Default URDF joint names per embodiment, in the same order the dataset
# stores joint angles. Override at the CLI with --arm_joint_names if your
# URDF uses different names.
ARM_JOINT_NAMES = {
    "taco_play": [
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
        "panda_joint5", "panda_joint6", "panda_joint7",
    ],
    "berkeley_autolab_ur5": [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    ],
}

# Default URDF gripper joint name per embodiment. Robotiq 2F-85 (used on
# Berkeley AutoLab UR5) drives the whole gripper from 'finger_joint' or
# 'robotiq_85_left_knuckle_joint' depending on the URDF source.
GRIPPER_JOINT_NAMES = {
    "taco_play": "finger_joint",
    "berkeley_autolab_ur5": "finger_joint",
}


def _overlay(image_bgr, mask_bool, color=(0, 255, 255), alpha=0.5):
    out = image_bgr.copy()
    sel = mask_bool
    if not np.any(sel):
        return out
    out[sel] = ((1 - alpha) * image_bgr[sel] +
                alpha * np.array(color, dtype=np.uint8)).astype(np.uint8)
    cv2.drawContours(out,
                     cv2.findContours(mask_bool.astype(np.uint8),
                                      cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)[0],
                     -1, color, 1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--oxe_root", type=Path, required=True)
    p.add_argument("--mask_root", type=Path, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--urdf", type=Path, required=True)
    p.add_argument("--mesh_dir", type=Path, default=None)
    p.add_argument("--episodes", nargs="+", default=None,
                   help="Subset of episode_XXXX names. Default: all OK ones.")
    p.add_argument("--max_frames_per_ep", type=int, default=20)
    p.add_argument("--downsample", type=int, default=2)
    p.add_argument("--dilate_px", type=int, default=2)
    p.add_argument("--gripper_open_rad", type=float, default=0.0)
    p.add_argument("--gripper_closed_rad", type=float, default=0.7)
    p.add_argument("--arm_joint_names", default=None,
                   help="Comma-separated URDF joint names; default per dataset "
                        "(see ARM_JOINT_NAMES).")
    p.add_argument("--gripper_joint_name", default=None,
                   help="URDF joint that drives the gripper open/close. Default "
                        "per dataset (see GRIPPER_JOINT_NAMES).")
    args = p.parse_args()

    if args.dataset not in JOINT_DIMS:
        raise SystemExit(
            f"{args.dataset}: no joint-dim slice known. Add to JOINT_DIMS.")
    spec = JOINT_DIMS[args.dataset]
    arm_lo, arm_hi = spec["arm"]
    grip_idx = spec.get("gripper_idx")
    grip_kind = spec.get("gripper_kind", "width_m")

    arm_names = (args.arm_joint_names.split(",") if args.arm_joint_names
                 else ARM_JOINT_NAMES.get(args.dataset))
    gripper_joint_name = (args.gripper_joint_name or
                          GRIPPER_JOINT_NAMES.get(args.dataset, "finger_joint"))

    masker = URDFRobotMasker(
        urdf_path=args.urdf, mesh_dir=args.mesh_dir,
        downsample=args.downsample, dilate_px=args.dilate_px,
        arm_joint_names=arm_names,
        gripper_joint_name=gripper_joint_name,
        gripper_open_rad=args.gripper_open_rad,
        gripper_closed_rad=args.gripper_closed_rad,
    )

    ds_oxe = args.oxe_root / args.dataset
    ds_seg = args.mask_root / args.dataset

    summary_path = ds_seg / "pnp_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"{summary_path} missing — run pnp_oxe.py first.")
    summary = json.loads(summary_path.read_text())

    if args.episodes:
        targets = args.episodes
    else:
        targets = [r["episode"] for r in summary["results"]
                   if r.get("status") == "ok"]

    for ep in targets:
        pnp = json.loads((ds_seg / ep / "pnp.json").read_text())
        T_cam2base = np.asarray(pnp["T_cam2base"], dtype=np.float64)
        K = pnp["K"]
        traj = np.load(ds_oxe / ep / "trajectory.npz")
        state = traj["state"]
        if state.shape[1] < arm_hi:
            print(f"[skip] {ep}: state too short for arm dims")
            continue

        frames_dir = ds_oxe / ep / "frames"
        out_dir = ds_seg / ep / "cam2base_viz"
        out_dir.mkdir(exist_ok=True)

        # Prefer inlier frames as the qualitative check.
        stems = pnp.get("inlier_stems", [])
        if not stems:
            stems = sorted(p.stem for p in frames_dir.glob("*.jpg"))
        stems = stems[:args.max_frames_per_ep]

        for stem in stems:
            img = cv2.imread(str(frames_dir / f"{stem}.jpg"))
            if img is None:
                continue
            idx = int(stem)
            if idx >= len(state):
                continue
            q_arm = np.asarray(state[idx, arm_lo:arm_hi], dtype=float)
            if grip_idx is None:
                grip = 0.0
            else:
                raw = float(state[idx, grip_idx])
                if grip_kind == "binary_closed":
                    grip = 1.0 if raw >= 0.5 else 0.0
                elif grip_kind == "width_m":
                    # Map measured width (m) into 0=open .. 1=closed using
                    # the per-call open/closed radian span as proxy bounds.
                    grip = float(np.clip(1.0 - raw / 0.085, 0.0, 1.0))
                else:
                    grip = float(np.clip(raw, 0.0, 1.0))
            mask = masker.render(
                K=K, T_cam2base=T_cam2base,
                joint_positions=q_arm, gripper_position=grip,
                image_hw=(img.shape[0], img.shape[1]),
            )
            viz = _overlay(img, mask)
            cv2.imwrite(str(out_dir / f"{stem}.jpg"), viz)

        print(f"[ok] {ep}: wrote {out_dir}")


if __name__ == "__main__":
    main()
