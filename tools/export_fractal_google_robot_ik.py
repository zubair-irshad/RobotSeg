#!/usr/bin/env python3
"""Export Google Robot joints for local Fractal/OXE episodes via AugE MuJoCo IK.

AugE does not read joint positions from the original Fractal RLDS. It reads the
logged TCP pose, solves IK in MuJoCo/Mink for the target robot, then stores the
resulting qpos. This script applies the same idea to the local
tools/download_oxe_subset.py layout and writes `joint_position` into each
episode's trajectory.npz so our URDF projection/silhouette tools can render an
articulated Google Robot instead of only a TCP marker.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


GOOGLE_ARM_JOINTS = 7


def _load_aug_e(auge_root: Path):
    auge_root = auge_root.expanduser().resolve()
    if not auge_root.exists():
        raise FileNotFoundError(f"AugE root does not exist: {auge_root}")
    sys.path.insert(0, str(auge_root))

    import mujoco  # type: ignore
    import mink  # type: ignore
    from loop_rate_limiters import RateLimiter  # type: ignore
    from core.gripper_utils import GRIPPER_JOINT_IDS, GRIPPER_RANGES  # type: ignore
    from core.utils import XML_PATH  # type: ignore

    xml_path = Path(XML_PATH["google_robot"])
    if not xml_path.is_absolute():
        xml_path = auge_root / xml_path
    return mujoco, mink, RateLimiter, GRIPPER_JOINT_IDS, GRIPPER_RANGES, xml_path


def _as_str(x) -> str | None:
    if x is None:
        return None
    arr = np.asarray(x)
    if arr.shape == ():
        val = arr.item()
        if isinstance(val, bytes):
            return val.decode("utf-8")
        return str(val)
    return str(arr)


def _rot_to_quat_wxyz(traj: dict[str, np.ndarray], n: int) -> np.ndarray:
    if "eef_rot" not in traj:
        return np.repeat(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64), n, axis=0)
    rot = np.asarray(traj["eef_rot"], dtype=np.float64)
    if rot.ndim == 1:
        rot = rot[None, :]
    rot = rot[:n]
    fmt = _as_str(traj.get("eef_rot_format"))
    if fmt == "quat_wxyz":
        quat_wxyz = rot[:, :4]
    elif fmt == "quat_xyzw":
        quat_wxyz = np.roll(rot[:, :4], 1, axis=1)
    elif fmt == "euler_xyz":
        quat_xyzw = R.from_euler("xyz", rot[:, :3]).as_quat()
        quat_wxyz = np.roll(quat_xyzw, 1, axis=1)
    else:
        if rot.shape[1] >= 4:
            quat_wxyz = rot[:, :4]
        elif rot.shape[1] >= 3:
            quat_xyzw = R.from_euler("xyz", rot[:, :3]).as_quat()
            quat_wxyz = np.roll(quat_xyzw, 1, axis=1)
        else:
            quat_wxyz = np.repeat(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64), len(rot), axis=0)

    norm = np.linalg.norm(quat_wxyz, axis=1, keepdims=True)
    norm[norm < 1e-9] = 1.0
    return quat_wxyz / norm


def _open_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _write_npz(path: Path, data: dict[str, np.ndarray]) -> None:
    tmp = path.with_suffix(".npz.tmp")
    with tmp.open("wb") as f:
        np.savez(f, **data)
    tmp.replace(path)


def _set_gripper(model, data, gripper_ranges, gripper_value: float) -> None:
    lo = float(gripper_ranges["google_robot"]["open"])
    hi = float(gripper_ranges["google_robot"]["close"])
    value = float(np.clip(gripper_value, min(lo, hi), max(lo, hi)))
    for actuator in gripper_ranges["google_robot"]["actuators"]:
        data.ctrl[model.actuator(actuator).id] = value


def solve_episode(
    traj_path: Path,
    *,
    auge_root: Path,
    max_iters: int,
    position_cost: float,
    orientation_cost: float,
    damping: float,
    update_trajectory: bool,
) -> dict:
    mujoco, mink, RateLimiter, gripper_joint_ids, gripper_ranges, xml_path = _load_aug_e(auge_root)
    traj = _open_npz(traj_path)
    if "eef_xyz" not in traj:
        return {"episode": traj_path.parent.name, "status": "skip-missing-eef_xyz"}

    xyz = np.asarray(traj["eef_xyz"], dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        return {"episode": traj_path.parent.name, "status": "skip-bad-eef_xyz", "shape": list(xyz.shape)}
    n = len(xyz)
    quat_wxyz = _rot_to_quat_wxyz(traj, n)
    gripper = np.asarray(traj.get("gripper", np.ones((n, 1), dtype=np.float32)), dtype=np.float64).reshape(-1)
    if len(gripper) < n:
        gripper = np.pad(gripper, (0, n - len(gripper)), mode="edge")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    configuration = mink.Configuration(model)
    end_effector_task = mink.FrameTask(
        frame_name="attachment_site",
        frame_type="site",
        position_cost=position_cost,
        orientation_cost=orientation_cost,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model=model, cost=1e-2)
    tasks = [end_effector_task, posture_task]

    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    configuration.update(data.qpos)
    posture_task.set_target_from_configuration(configuration)
    mujoco.mj_forward(model, data)
    rate = RateLimiter(frequency=500.0, warn=False)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")

    joint_positions = []
    all_qpos = []
    replay_positions = []
    target_positions = []
    errors = []

    non_gripper_joint_ids = [i for i in range(model.njnt) if i not in gripper_joint_ids["google_robot"]]
    for i in range(n):
        _set_gripper(model, data, gripper_ranges, float(gripper[i]))
        for _ in range(50):
            mujoco.mj_step(model, data)

        rotation = mink.SO3(wxyz=quat_wxyz[i])
        target = mink.SE3.from_rotation_and_translation(rotation, xyz[i, :3])
        end_effector_task.set_target(target)

        for _ in range(max_iters):
            vel = mink.solve_ik(configuration, tasks, rate.dt, "quadprog", damping)
            configuration.integrate_inplace(vel, rate.dt)
            err = end_effector_task.compute_error(configuration)
            if np.linalg.norm(err[:3]) <= 1e-5 and np.linalg.norm(err[3:]) <= 1e-5:
                break

        data.qpos[non_gripper_joint_ids] = configuration.q[non_gripper_joint_ids]
        mujoco.mj_forward(model, data)
        replay = data.site_xpos[site_id].copy()
        err_m = float(np.linalg.norm(replay - xyz[i, :3]))
        joint_positions.append(data.qpos.copy()[:GOOGLE_ARM_JOINTS])
        all_qpos.append(data.qpos.copy())
        replay_positions.append(replay)
        target_positions.append(xyz[i, :3].copy())
        errors.append(err_m)

    joint_positions_arr = np.asarray(joint_positions, dtype=np.float32)
    info = {
        "google_robot_joint_position": joint_positions_arr,
        "google_robot_all_qpos": np.asarray(all_qpos, dtype=np.float32),
        "google_robot_target_positions": np.asarray(target_positions, dtype=np.float32),
        "google_robot_replay_positions": np.asarray(replay_positions, dtype=np.float32),
        "google_robot_ik_error_m": np.asarray(errors, dtype=np.float32),
        "google_robot_ik_source": np.array("AugE MuJoCo/Mink attachment_site IK"),
    }
    np.savez(traj_path.parent / "google_robot_ik_info.npz", **info)

    if update_trajectory:
        traj["joint_position"] = joint_positions_arr
        traj["google_robot_all_qpos"] = info["google_robot_all_qpos"]
        traj["google_robot_ik_error_m"] = info["google_robot_ik_error_m"]
        _write_npz(traj_path, traj)

    errors_arr = np.asarray(errors)
    return {
        "episode": traj_path.parent.name,
        "status": "ok",
        "frames": int(n),
        "mean_error_m": float(errors_arr.mean()) if n else None,
        "median_error_m": float(np.median(errors_arr)) if n else None,
        "max_error_m": float(errors_arr.max()) if n else None,
        "updated_trajectory": bool(update_trajectory),
        "out": str(traj_path.parent / "google_robot_ik_info.npz"),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--oxe_root", type=Path, required=True)
    p.add_argument("--dataset", default="fractal20220817_data")
    p.add_argument("--auge_root", type=Path, default=Path("~/AugE-Toolkit"))
    p.add_argument("--max_iters", type=int, default=100)
    p.add_argument("--position_cost", type=float, default=10.0)
    p.add_argument("--orientation_cost", type=float, default=1.0)
    p.add_argument("--damping", type=float, default=1e-3)
    p.add_argument("--no_update_trajectory", action="store_true")
    args = p.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    ds_root = args.oxe_root.expanduser() / args.dataset
    episodes = sorted(p for p in ds_root.iterdir() if p.is_dir() and p.name.startswith("episode_"))
    for ep in episodes:
        traj_path = ep / "trajectory.npz"
        if not traj_path.exists():
            print(f"[{args.dataset}/{ep.name}] skip-missing-trajectory")
            continue
        result = solve_episode(
            traj_path,
            auge_root=args.auge_root,
            max_iters=args.max_iters,
            position_cost=args.position_cost,
            orientation_cost=args.orientation_cost,
            damping=args.damping,
            update_trajectory=not args.no_update_trajectory,
        )
        if result["status"] == "ok":
            print(
                f"[{args.dataset}/{result['episode']}] ok "
                f"frames={result['frames']} "
                f"median_err={result['median_error_m']:.6f}m "
                f"max_err={result['max_error_m']:.6f}m "
                f"updated={result['updated_trajectory']}"
            )
        else:
            print(f"[{args.dataset}/{result['episode']}] {result['status']}")


if __name__ == "__main__":
    main()
