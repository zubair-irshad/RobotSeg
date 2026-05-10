"""Repair existing Berkeley Autolab UR5 trajectory.npz proprio fields.

Older downloads in this repo saved only the first six UR5 arm joints under
trajectory['joint_position']. AugE replays the same RLDS data with a 14D qpos:
six arm joints plus eight synthetic gripper joints selected from
robot_state[13]. AugE's joint-angle path does not use the logged
robot_state[6:13] TCP for rendering; it renders the named MuJoCo joints and
uses the MuJoCo attachment_site as the replay TCP. This script rewrites
trajectory.npz files in-place using the saved trajectory['state'] array, so
DOWNLOAD=0 runs can match either the official logged TCP or AugE's rendered TCP:

    [joint0..joint5, x, y, z, qx, qy, qz, qw, gripper_is_closed, action_blocked]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


UR5_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "right_driver_joint",
    "right_coupler_joint",
    "right_spring_link_joint",
    "right_follower_joint",
    "left_driver_joint",
    "left_coupler_joint",
    "left_spring_link_joint",
    "left_follower_joint",
]


def _ur5_qpos_from_state(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    closed = state[:, 13] > 0.5
    qpos = np.zeros((state.shape[0], 14), dtype=np.float32)
    qpos[:, :6] = state[:, :6]
    closed_vals = np.asarray(
        [1.0, 0.025, 0.80, -0.80, 1.0, 0.0252768, 0.80, -0.80],
        dtype=np.float32,
    )
    qpos[closed, 6:] = closed_vals
    return qpos


def _same_array(a: np.ndarray | None, b: np.ndarray) -> bool:
    return a is not None and a.shape == b.shape and np.allclose(a, b)


def _quat_xyzw_from_matrix(Rm: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    return R.from_matrix(np.asarray(Rm, dtype=np.float64).reshape(3, 3)).as_quat()


def _T_from_body(data, body_id: int) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    if body_id <= 0:
        return T
    T[:3, :3] = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(data.xpos[body_id], dtype=np.float64)
    return T


def _auge_attachment_site_fk(state: np.ndarray, qpos: np.ndarray, xml_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco  # type: ignore

    model = mujoco.MjModel.from_xml_path(str(xml_path.expanduser()))
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    if site_id < 0:
        raise RuntimeError(f"{xml_path} has no MuJoCo site named attachment_site")
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ur5e/base")
    if base_body_id < 0:
        raise RuntimeError(f"{xml_path} has no MuJoCo body named ur5e/base")

    qpos_addrs = []
    for name in UR5_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"{xml_path} has no MuJoCo joint named {name}")
        qpos_addrs.append(int(model.jnt_qposadr[jid]))

    xyz_base = np.zeros((state.shape[0], 3), dtype=np.float32)
    xyz_world = np.zeros((state.shape[0], 3), dtype=np.float32)
    quat_base = np.zeros((state.shape[0], 4), dtype=np.float32)
    for i in range(state.shape[0]):
        data.qpos[:] = 0.0
        for col, addr in enumerate(qpos_addrs):
            if addr < model.nq and col < qpos.shape[1]:
                data.qpos[addr] = float(qpos[i, col])
        mujoco.mj_forward(model, data)
        T_world_base = _T_from_body(data, base_body_id)
        T_base_world = np.linalg.inv(T_world_base)
        p_world = np.asarray(data.site_xpos[site_id], dtype=np.float64)
        R_world_site = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
        R_base_site = T_base_world[:3, :3] @ R_world_site
        xyz_world[i] = p_world.astype(np.float32)
        xyz_base[i] = (T_base_world @ np.r_[p_world, 1.0])[:3].astype(np.float32)
        quat_base[i] = _quat_xyzw_from_matrix(R_base_site)
    return xyz_base, quat_base, xyz_world


def repair(path: Path, dry_run: bool = False, tcp_source: str = "logged",
           mujoco_xml_path: Path | None = None) -> str:
    with np.load(path, allow_pickle=True) as data:
        if "state" not in data.files:
            return "skip-no-state"
        state = np.asarray(data["state"], dtype=np.float32)
        if state.ndim != 2 or state.shape[1] < 15:
            return f"skip-bad-state-shape-{state.shape}"
        qpos = _ur5_qpos_from_state(state)
        logged_eef_xyz = state[:, 6:9].astype(np.float32)
        logged_eef_rot = state[:, 9:13].astype(np.float32)
        gripper = state[:, 13:14].astype(np.float32)
        auge_xyz = None
        auge_quat = None
        if tcp_source == "auge_fk":
            if mujoco_xml_path is None:
                raise ValueError("--tcp_source auge_fk requires --mujoco_xml_path")
            auge_xyz, auge_quat, auge_xyz_world = _auge_attachment_site_fk(state, qpos, mujoco_xml_path)
            eef_xyz = auge_xyz
            eef_rot = auge_quat
            eef_source = "AugE MuJoCo attachment_site FK in ur5e/base frame"
        elif tcp_source == "logged":
            auge_xyz_world = None
            eef_xyz = logged_eef_xyz
            eef_rot = logged_eef_rot
            eef_source = "Berkeley logged robot_state[6:13]"
        else:
            raise ValueError(f"unknown tcp_source: {tcp_source}")

        old_qpos = np.asarray(data["joint_position"], dtype=np.float32) if "joint_position" in data.files else None
        old_xyz = np.asarray(data["eef_xyz"], dtype=np.float32) if "eef_xyz" in data.files else None
        old_rot = np.asarray(data["eef_rot"], dtype=np.float32) if "eef_rot" in data.files else None
        old_gripper = np.asarray(data["gripper"], dtype=np.float32) if "gripper" in data.files else None
        old_fmt = str(data["eef_rot_format"]) if "eef_rot_format" in data.files else None

        changed = []
        if not _same_array(old_qpos, qpos):
            changed.append("joint_position")
        if not _same_array(old_xyz, eef_xyz):
            changed.append("eef_xyz")
        if not _same_array(old_rot, eef_rot) or old_fmt != "quat_xyzw":
            changed.append("eef_rot")
        if not _same_array(old_gripper, gripper):
            changed.append("gripper")
        old_source = str(data["eef_xyz_source"]) if "eef_xyz_source" in data.files else None
        if old_source != eef_source:
            changed.append("eef_xyz_source")
        old_logged = np.asarray(data["logged_eef_xyz"], dtype=np.float32) if "logged_eef_xyz" in data.files else None
        if not _same_array(old_logged, logged_eef_xyz):
            changed.append("logged_eef_xyz")
        if auge_xyz is not None:
            old_auge = np.asarray(data["auge_attachment_site_xyz"], dtype=np.float32) if "auge_attachment_site_xyz" in data.files else None
            if not _same_array(old_auge, auge_xyz):
                changed.append("auge_attachment_site_xyz")
            old_auge_world = np.asarray(data["auge_attachment_site_xyz_world"], dtype=np.float32) if "auge_attachment_site_xyz_world" in data.files else None
            if not _same_array(old_auge_world, auge_xyz_world):
                changed.append("auge_attachment_site_xyz_world")
        if not changed:
            return "ok-already"

        arrays = {name: data[name] for name in data.files}
        arrays["joint_position"] = qpos
        arrays["eef_xyz"] = eef_xyz
        arrays["eef_rot"] = eef_rot
        arrays["eef_rot_format"] = np.array("quat_xyzw")
        arrays["gripper"] = gripper
        arrays["logged_eef_xyz"] = logged_eef_xyz
        arrays["logged_eef_rot"] = logged_eef_rot
        arrays["eef_xyz_source"] = np.array(eef_source)
        arrays["ur5_mujoco_joint_names"] = np.asarray(UR5_JOINT_NAMES)
        if auge_xyz is not None:
            arrays["auge_attachment_site_xyz"] = auge_xyz
            arrays["auge_attachment_site_xyz_world"] = auge_xyz_world
            arrays["auge_attachment_site_quat_xyzw"] = auge_quat
    if not dry_run:
        np.savez_compressed(path, **arrays)
    return "updated-" + ",".join(changed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oxe_root", type=Path, required=True)
    parser.add_argument("--dataset", default="berkeley_autolab_ur5")
    parser.add_argument("--tcp_source", choices=["logged", "auge_fk"], default="logged",
                        help="Which TCP should be written to trajectory['eef_xyz']. "
                             "'auge_fk' matches AugE's rendered attachment_site.")
    parser.add_argument("--mujoco_xml_path", type=Path, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    root = args.oxe_root / args.dataset
    paths = sorted(root.glob("episode_*/trajectory.npz"))
    if not paths:
        print(f"[skip] no trajectory.npz files under {root}")
        return
    counts: dict[str, int] = {}
    for path in paths:
        status = repair(
            path,
            dry_run=args.dry_run,
            tcp_source=args.tcp_source,
            mujoco_xml_path=args.mujoco_xml_path,
        )
        counts[status] = counts.get(status, 0) + 1
        print(f"[{path.parent.name}] {status}")
    print("summary:", counts)


if __name__ == "__main__":
    main()
