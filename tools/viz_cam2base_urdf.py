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
try:
    from mjcf_robot_masker import MuJoCoRobotMasker  # noqa: E402
except Exception:  # mujoco not installed yet
    MuJoCoRobotMasker = None


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
    # Bridge state[:, :6] is (x, y, z, roll, pitch, yaw) — 6-DoF EE POSE,
    # NOT joint angles. Only meaningful with --zero_pose / --base_only.
    # state[6] is the gripper position (~0..1).
    "bridge": {"arm": (0, 6), "gripper_idx": 6,
               "gripper_kind": "binary_closed",
               "no_joint_angles": True},
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
    p.add_argument("--urdf", type=Path, default=None,
                   help="URDF path (yourdfpy backend). Required unless --mjcf is used.")
    p.add_argument("--mjcf", type=Path, default=None,
                   help="MuJoCo MJCF path (e.g. AugE's "
                        "robot_xml/universal_robots_ur5e/ur5e.xml). When set, "
                        "uses MuJoCo for kinematics + mesh assets and ignores "
                        "URDF-specific flags (joint_signs, base_offset, etc.).")
    p.add_argument("--mesh_dir", type=Path, default=None)
    p.add_argument("--arm_dof", type=int, default=6,
                   help="Number of arm DOFs to push into MJCF qpos[:n].")
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
    p.add_argument("--wrap_revolute", action="store_true", default=True,
                   help="Wrap arm joint angles into [-pi, pi] before applying "
                        "to the URDF. Necessary for URDFs whose joints are "
                        "limited to that range (e.g. CogRob's "
                        "ur5e_2f85_joint_limited_robot.urdf).")
    p.add_argument("--no_wrap_revolute", dest="wrap_revolute",
                   action="store_false")
    # URDF-root vs dataset-base frame offset. For UR URDFs (incl. CogRob
    # ur5e_2f85), the URDF root 'base_link' is 180° around Z relative to
    # the real-robot 'base' frame the dataset publishes EE pose in.
    p.add_argument("--base_offset_rpy_deg", nargs=3, type=float,
                   default=[0.0, 0.0, 180.0],
                   metavar=("R", "P", "Y"),
                   help="Roll/pitch/yaw (deg) of URDF root expressed in the "
                        "dataset's base frame. Default 0 0 180 for UR URDFs. "
                        "Use 0 0 0 for Franka URDFs (rerun-io/franka_description).")
    p.add_argument("--base_offset_xyz", nargs=3, type=float,
                   default=[0.0, 0.0, 0.0],
                   metavar=("X", "Y", "Z"),
                   help="Translation (m) of URDF root in dataset base frame.")
    p.add_argument("--debug_markers", action="store_true", default=True,
                   help="Overlay BASE (URDF root projection) and EE (eef_xyz "
                        "projection) markers. Use to diagnose whether the "
                        "silhouette mismatch is from T_cam2base or from "
                        "joint angles / URDF internals.")
    p.add_argument("--no_debug_markers", dest="debug_markers", action="store_false")
    p.add_argument("--ik_from_eef_pose", action="store_true",
                   help="When the dataset has EE pose but no joint angles "
                        "(Bridge), solve IK each frame from "
                        "trajectory.npz['eef_xyz'] + ['eef_rot'] to get joints, "
                        "then render. MJCF backend only.")
    p.add_argument("--ik_ee_body", default=None,
                   help="MJCF body name to drive in IK (e.g. 'ee_gripper_link', "
                        "'gripper_link', 'wx250s/ee_gripper_link'). If unset, "
                        "the first body whose name contains 'ee_' is used.")
    p.add_argument("--ik_pos_tol", type=float, default=2e-3)
    p.add_argument("--ik_rot_tol", type=float, default=2e-2)
    p.add_argument("--ik_max_iter", type=int, default=80)
    p.add_argument("--ik_ee_pos_offset_xyz", nargs=3, type=float,
                   default=[0.0, 0.0, 0.0], metavar=("X", "Y", "Z"),
                   help="Constant 3-vector in the proprio-EE frame added to "
                        "the IK target position. Use to compensate for the "
                        "(MJCF EE body) ↔ (proprio eef_xyz reference point) "
                        "offset. E.g. WidowX MJCF's ee_gripper_link is often "
                        "~+5cm along tool z from the jaw tip.")
    p.add_argument("--ik_ee_R_offset_rpy_deg", nargs=3, type=float,
                   default=[0.0, 0.0, 0.0], metavar=("R", "P", "Y"),
                   help="Constant rotation (deg, RPY in proprio-EE frame) "
                        "applied to the IK target orientation. Use when the "
                        "MJCF EE body's axes differ from proprio eef_rot's.")
    p.add_argument("--ik_use_tool_offset", action="store_true", default=True,
                   help="Add the per-episode tool_offset from pnp.json to the "
                        "IK target so the MJCF gripper body lands where "
                        "cam2base was actually fitted (the visible tip), not "
                        "where eef_xyz alone would put the wrist. On by default.")
    p.add_argument("--no_ik_use_tool_offset", dest="ik_use_tool_offset",
                   action="store_false")

    p.add_argument("--base_only", action="store_true",
                   help="Render ONLY the URDF base link (joint-angle-independent "
                        "cam2base sanity check). Implies --zero_pose. URDF backend "
                        "only — for MJCF, use --zero_pose plus --debug_markers.")
    p.add_argument("--base_link_names", default=None,
                   help="Comma-separated link-name substrings to keep when "
                        "--base_only is set. Default: the URDF's declared base_link.")
    p.add_argument("--zero_pose", action="store_true",
                   help="Render the URDF at q=0 ignoring the dataset's joint "
                        "angles. Useful to see whether T_cam2base alone places "
                        "the robot correctly when the rest pose is known.")
    p.add_argument("--joint_signs", nargs="+", type=float, default=None,
                   metavar="S",
                   help="Per-joint sign multiplier applied to q_dataset before "
                        "feeding the URDF. Length must match arm DOF. "
                        "Common UR fixes: '1 -1 -1 -1 1 1' or '1 -1 -1 1 1 1'.")
    p.add_argument("--joint_offsets_rad", nargs="+", type=float, default=None,
                   metavar="O",
                   help="Per-joint additive offset (rad) added AFTER signs. "
                        "ros-industrial UR5 sometimes needs offsets near "
                        "+/- pi/2 on shoulder_lift.")
    args = p.parse_args()

    if args.dataset not in JOINT_DIMS:
        raise SystemExit(
            f"{args.dataset}: no joint-dim slice known. Add to JOINT_DIMS.")
    spec = JOINT_DIMS[args.dataset]
    arm_lo, arm_hi = spec["arm"]
    grip_idx = spec.get("gripper_idx")
    grip_kind = spec.get("gripper_kind", "width_m")
    if (spec.get("no_joint_angles") and
            not (args.zero_pose or args.base_only or args.ik_from_eef_pose)):
        raise SystemExit(
            f"{args.dataset}: trajectory.npz has no joint angles "
            f"(state[:6] is EE POSE, not joints). Re-run with one of: "
            f"--zero_pose, --base_only, or --ik_from_eef_pose (solves IK "
            f"each frame from eef_xyz+eef_rot; MJCF backend only).")

    arm_names = (args.arm_joint_names.split(",") if args.arm_joint_names
                 else ARM_JOINT_NAMES.get(args.dataset))
    gripper_joint_name = (args.gripper_joint_name or
                          GRIPPER_JOINT_NAMES.get(args.dataset, "finger_joint"))

    if args.mjcf is not None:
        if MuJoCoRobotMasker is None:
            raise SystemExit("mujoco is not installed. `pip install mujoco`.")
        masker = MuJoCoRobotMasker(
            mjcf_path=args.mjcf, arm_dof=args.arm_dof,
            downsample=args.downsample, dilate_px=args.dilate_px,
        )
        # MJCF backends use the real-controller joint convention; no need
        # for the URDF base-link / base flip, no per-joint signs/offsets,
        # no joint-limit clamping (MuJoCo respects URDF-style limits but
        # the Menagerie MJCF uses ±2π for all UR joints).
        using_mjcf = True
    else:
        if args.urdf is None:
            raise SystemExit("Pass --urdf <path> (or --mjcf <path>).")
        masker = URDFRobotMasker(
            urdf_path=args.urdf, mesh_dir=args.mesh_dir,
            downsample=args.downsample, dilate_px=args.dilate_px,
            arm_joint_names=arm_names,
            gripper_joint_name=gripper_joint_name,
            gripper_open_rad=args.gripper_open_rad,
            gripper_closed_rad=args.gripper_closed_rad,
        )
        using_mjcf = False

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

    # If the URDF root differs from the dataset's base frame by a fixed
    # rigid offset (very common for UR URDFs, where 'base_link' is rotated
    # 180° around Z relative to the real-robot 'base'), pre-multiply that
    # correction onto T_cam2base. The user passes the rotation in degrees.
    rx, ry, rz = (np.deg2rad(a) for a in args.base_offset_rpy_deg)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    R_off = Rz @ Ry @ Rx
    T_root_in_base = np.eye(4)
    T_root_in_base[:3, :3] = R_off
    T_root_in_base[:3, 3] = args.base_offset_xyz

    # Resolve IK target body once if we're in IK mode.
    ik_body = None
    if args.ik_from_eef_pose:
        if not using_mjcf:
            raise SystemExit("--ik_from_eef_pose requires the MJCF backend (--mjcf).")
        if args.ik_ee_body:
            ik_body = args.ik_ee_body
        else:
            import mujoco as _mj
            for i in range(masker.model.nbody):
                name = _mj.mj_id2name(masker.model, _mj.mjtObj.mjOBJ_BODY, i) or ""
                if "ee_" in name or name.endswith("gripper_link"):
                    ik_body = name
                    break
            if ik_body is None:
                raise SystemExit(
                    "Couldn't auto-detect IK EE body. Pass --ik_ee_body explicitly. "
                    f"Available bodies: "
                    f"{[_mj.mj_id2name(masker.model, _mj.mjtObj.mjOBJ_BODY, i) for i in range(masker.model.nbody)]}"
                )
        print(f"[ik] target body: {ik_body!r}")
        # Pre-compute the constant rotation offset in matrix form.
        rx_, ry_, rz_ = (np.deg2rad(a) for a in args.ik_ee_R_offset_rpy_deg)
        cx_, sx_ = np.cos(rx_), np.sin(rx_)
        cy_, sy_ = np.cos(ry_), np.sin(ry_)
        cz_, sz_ = np.cos(rz_), np.sin(rz_)
        Rx_ = np.array([[1, 0, 0], [0, cx_, -sx_], [0, sx_, cx_]])
        Ry_ = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]])
        Rz_ = np.array([[cz_, -sz_, 0], [sz_, cz_, 0], [0, 0, 1]])
        R_ee_offset = Rz_ @ Ry_ @ Rx_
        pos_ee_offset = np.asarray(args.ik_ee_pos_offset_xyz, dtype=np.float64)

    def _eef_rot_to_R(rot, fmt):
        rot = np.asarray(rot, dtype=np.float64)
        fmt = (fmt or "").lower().strip()
        from scipy.spatial.transform import Rotation
        if fmt in ("quat_xyzw", "xyzw"):
            return Rotation.from_quat(rot).as_matrix()
        if fmt in ("quat_wxyz", "wxyz"):
            return Rotation.from_quat(np.concatenate([rot[..., 1:], rot[..., :1]], -1)).as_matrix()
        if fmt in ("rotvec", "axis_angle", "axisangle"):
            return Rotation.from_rotvec(rot).as_matrix()
        if fmt in ("matrix", "rotmat"):
            return rot.reshape(-1, 3, 3)
        if fmt.startswith("euler"):
            seq = fmt.split("_", 1)[1] if "_" in fmt else "xyz"
            return Rotation.from_euler(seq, rot).as_matrix()
        raise ValueError(f"Unknown eef_rot_format: {fmt!r}")

    for ep in targets:
        pnp = json.loads((ds_seg / ep / "pnp.json").read_text())
        T_cam2base = np.asarray(pnp["T_cam2base"], dtype=np.float64)
        # Re-express the camera pose in the URDF root frame. If the URDF
        # root (e.g. base_link) is related to the dataset's base frame by
        #   T_root_in_base  (pose of root expressed in base),
        # then the same camera, expressed in root, is
        #   T_cam_in_root = inv(T_root_in_base) @ T_cam_in_base.
        # This is a LEFT-multiplication; a right-multiplication would
        # rotate the camera's local axes instead of changing the frame
        # the verts live in.
        # MJCF backend uses the dataset's base frame directly — no flip
        # needed. URDF backend requires the root-vs-base correction.
        if using_mjcf:
            T_cam2base_for_render = T_cam2base
        else:
            T_cam2base_for_render = np.linalg.inv(T_root_in_base) @ T_cam2base
        K = pnp["K"]
        traj = np.load(ds_oxe / ep / "trajectory.npz")
        state = traj["state"]
        if state.shape[1] < arm_hi:
            print(f"[skip] {ep}: state too short for arm dims")
            continue

        # Pre-decode eef_xyz / eef_rot for IK mode.
        ee_xyz_seq = ee_R_seq = None
        ep_tool_offset = np.zeros(3)
        if args.ik_from_eef_pose:
            ee_xyz_seq = np.asarray(traj["eef_xyz"], dtype=np.float64)
            fmt = (str(traj["eef_rot_format"])
                   if "eef_rot_format" in traj.files else "rotvec")
            ee_R_seq = _eef_rot_to_R(traj["eef_rot"], fmt)
            if args.ik_use_tool_offset and "tool_offset" in pnp:
                ep_tool_offset = np.asarray(pnp["tool_offset"], dtype=np.float64)
                print(f"  [ik] using episode tool_offset = "
                      f"[{ep_tool_offset[0]:+.3f},{ep_tool_offset[1]:+.3f},"
                      f"{ep_tool_offset[2]:+.3f}]m")
        q_prev = None  # warm-start IK from previous frame's solution

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
            # MJCF path: feed dataset joints straight through. The
            # Menagerie MJCF allows ±2π and uses the real-controller
            # joint convention, so no wrap / sign / offset hacks needed.
            if not using_mjcf:
                # URDF path: many community URDFs limit revolute joints
                # to [-π, π] and use a different convention than the
                # real UR controller, so wrap + apply sign/offset.
                if args.wrap_revolute:
                    q_arm = ((q_arm + np.pi) % (2 * np.pi)) - np.pi
                if args.joint_signs is not None:
                    signs = np.asarray(args.joint_signs, dtype=float)
                    if len(signs) >= len(q_arm):
                        q_arm = signs[:len(q_arm)] * q_arm
                if args.joint_offsets_rad is not None:
                    offs = np.asarray(args.joint_offsets_rad, dtype=float)
                    if len(offs) >= len(q_arm):
                        q_arm = q_arm + offs[:len(q_arm)]
                if args.wrap_revolute:
                    q_arm = ((q_arm + np.pi) % (2 * np.pi)) - np.pi
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
            if args.zero_pose or args.base_only:
                q_arm = np.zeros_like(q_arm)
                grip = 0.0
            if args.ik_from_eef_pose and ee_xyz_seq is not None:
                if idx < len(ee_xyz_seq):
                    R_eef = ee_R_seq[idx]
                    # IK target position: where the MJCF EE body must land so
                    # that the rendered arm matches what cam2base was fitted
                    # to. Compose:
                    #   tip_in_base   = eef_xyz + R_eef · tool_offset_episode
                    #   target_in_base = tip_in_base + R_eef · pos_ee_offset
                    target_pos = (ee_xyz_seq[idx]
                                  + R_eef @ (ep_tool_offset + pos_ee_offset))
                    target_R = R_eef @ R_ee_offset
                    q_arm, ik_info = masker.solve_ik(
                        target_pos=target_pos,
                        target_R=target_R,
                        ee_body_name=ik_body,
                        q_init=q_prev,
                        gripper_position=grip,
                        max_iter=args.ik_max_iter,
                        tol_pos=args.ik_pos_tol,
                        tol_rot=args.ik_rot_tol,
                    )
                    q_prev = q_arm.copy()
                    if ik_info["pos_err"] > 5 * args.ik_pos_tol:
                        print(f"  [ik] {stem}: pos_err={ik_info['pos_err']:.4f}m "
                              f"rot_err={ik_info['rot_err']:.3f} "
                              f"iters={ik_info['iters']}")
            link_filter = None
            if args.base_only and not using_mjcf:
                if args.base_link_names:
                    link_filter = [s.strip() for s in args.base_link_names.split(",")]
                else:
                    bl = masker.robot.base_link
                    # Many community URDFs use 'world' or 'base_footprint' as
                    # the URDF root with no visuals attached. Fall back to
                    # 'base_link'/'base' substrings, which are nearly
                    # universal for the first link that actually has meshes.
                    if bl in ("world", "base_footprint", "map"):
                        link_filter = ["base_link", "base"]
                    else:
                        link_filter = [bl]
            render_kwargs = dict(
                K=K, T_cam2base=T_cam2base_for_render,
                joint_positions=q_arm, gripper_position=grip,
                image_hw=(img.shape[0], img.shape[1]),
            )
            if link_filter is not None:
                render_kwargs["link_filter"] = link_filter
            mask = masker.render(**render_kwargs)
            viz = _overlay(img, mask)

            # Debug overlays: project the URDF root origin (where the
            # masker thinks the robot base is) and the dataset's EE xyz
            # (which we know lands at the actual gripper). Read both off
            # the same K and T_cam2base used for rendering.
            if args.debug_markers:
                K_mat = np.array([[K["fx"], 0, K["cx"]],
                                  [0, K["fy"], K["cy"]],
                                  [0, 0, 1.0]], dtype=np.float64)
                T_base2cam_render = np.linalg.inv(T_cam2base_for_render)

                def _proj(p_root):
                    p_cam = T_base2cam_render @ np.array([*p_root, 1.0])
                    if p_cam[2] <= 0.05:
                        return None
                    u = K_mat[0, 0] * p_cam[0] / p_cam[2] + K_mat[0, 2]
                    v = K_mat[1, 1] * p_cam[1] / p_cam[2] + K_mat[1, 2]
                    return int(round(u)), int(round(v))

                # 🔵 URDF root origin (= robot base in URDF's frame)
                base_uv = _proj([0.0, 0.0, 0.0])
                if base_uv is not None:
                    cv2.drawMarker(viz, base_uv, (255, 0, 0),
                                   cv2.MARKER_CROSS, 32, 3)
                    cv2.putText(viz, "BASE", (base_uv[0]+8, base_uv[1]-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # 🟢 EE position from eef_xyz (we trust this since it lands
                # at the actual gripper via T_cam2base directly). For URDF
                # mode the renderer's T is rotated, so we pre-rotate the
                # 3D point too; for MJCF mode the renderer's T is unchanged
                # so we project eef_xyz directly through T_base2cam_real.
                if state.shape[1] >= 10:
                    ee_in_base = np.asarray(state[idx, 7:10], dtype=float)
                    if using_mjcf:
                        T_base2cam_real = np.linalg.inv(T_cam2base)
                        p_cam = T_base2cam_real @ np.array([*ee_in_base, 1.0])
                        if p_cam[2] > 0.05:
                            u = K_mat[0, 0] * p_cam[0] / p_cam[2] + K_mat[0, 2]
                            v = K_mat[1, 1] * p_cam[1] / p_cam[2] + K_mat[1, 2]
                            ee_uv = (int(round(u)), int(round(v)))
                        else:
                            ee_uv = None
                    else:
                        ee_in_root = (np.linalg.inv(T_root_in_base)
                                      @ np.array([*ee_in_base, 1.0]))[:3]
                        ee_uv = _proj(ee_in_root)
                    if ee_uv is not None:
                        cv2.drawMarker(viz, ee_uv, (0, 255, 0),
                                       cv2.MARKER_TILTED_CROSS, 32, 3)
                        cv2.putText(viz, "EE", (ee_uv[0]+8, ee_uv[1]-8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 🟣 (MJCF only) Project the MJCF's computed wrist EE
                # position. If this lands at the same pixel as the green
                # EE marker, then MJCF FK is consistent with the dataset.
                # If it lands somewhere else, it's a joint-convention or
                # frame mismatch between dataset and MJCF.
                if using_mjcf:
                    try:
                        import mujoco as _mj
                        link_name = "wrist_3_link"
                        bid = _mj.mj_name2id(masker.model,
                                             _mj.mjtObj.mjOBJ_BODY, link_name)
                        if bid >= 0:
                            ee_mjcf = np.asarray(masker.data.xpos[bid], dtype=float)
                            T_base2cam_real = np.linalg.inv(T_cam2base)
                            p = T_base2cam_real @ np.array([*ee_mjcf, 1.0])
                            if p[2] > 0.05:
                                u = K_mat[0, 0] * p[0] / p[2] + K_mat[0, 2]
                                v = K_mat[1, 1] * p[1] / p[2] + K_mat[1, 2]
                                uv = (int(round(u)), int(round(v)))
                                cv2.drawMarker(viz, uv, (255, 0, 255),
                                               cv2.MARKER_DIAMOND, 32, 3)
                                cv2.putText(viz, "MJCF_EE",
                                            (uv[0]+8, uv[1]-8),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                            (255, 0, 255), 2)
                                # Also print the per-frame 3D delta:
                                if idx == int(stems[0]):
                                    delta = ee_mjcf - ee_in_base
                                    print(f"[mjcf_check] wrist_3 - eef_xyz = "
                                          f"[{delta[0]:+.3f}, {delta[1]:+.3f}, "
                                          f"{delta[2]:+.3f}] m  (should be small "
                                          f"if conventions match)")
                    except Exception as e:
                        print(f"[mjcf_check] skipped: {e}")
            cv2.imwrite(str(out_dir / f"{stem}.jpg"), viz)

        print(f"[ok] {ep}: wrote {out_dir}")


if __name__ == "__main__":
    main()
