"""Repair existing Berkeley Autolab UR5 trajectory.npz proprio fields.

Older downloads in this repo saved only the first six UR5 arm joints under
trajectory['joint_position']. AugE replays the same RLDS data with a 14D qpos:
six arm joints plus eight synthetic gripper joints selected from
robot_state[13]. Earlier downloads also used an off-by-one TCP slice. This
script rewrites trajectory.npz files in-place using the saved trajectory['state']
array, so DOWNLOAD=0 runs use the official Berkeley UR5 layout:

    [joint0..joint5, x, y, z, qx, qy, qz, qw, gripper_is_closed, action_blocked]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


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


def repair(path: Path, dry_run: bool = False) -> str:
    with np.load(path, allow_pickle=True) as data:
        if "state" not in data.files:
            return "skip-no-state"
        state = np.asarray(data["state"], dtype=np.float32)
        if state.ndim != 2 or state.shape[1] < 15:
            return f"skip-bad-state-shape-{state.shape}"
        qpos = _ur5_qpos_from_state(state)
        eef_xyz = state[:, 6:9].astype(np.float32)
        eef_rot = state[:, 9:13].astype(np.float32)
        gripper = state[:, 13:14].astype(np.float32)

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
        if not changed:
            return "ok-already"

        arrays = {name: data[name] for name in data.files}
        arrays["joint_position"] = qpos
        arrays["eef_xyz"] = eef_xyz
        arrays["eef_rot"] = eef_rot
        arrays["eef_rot_format"] = np.array("quat_xyzw")
        arrays["gripper"] = gripper
    if not dry_run:
        np.savez_compressed(path, **arrays)
    return "updated-" + ",".join(changed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oxe_root", type=Path, required=True)
    parser.add_argument("--dataset", default="berkeley_autolab_ur5")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    root = args.oxe_root / args.dataset
    paths = sorted(root.glob("episode_*/trajectory.npz"))
    if not paths:
        print(f"[skip] no trajectory.npz files under {root}")
        return
    counts: dict[str, int] = {}
    for path in paths:
        status = repair(path, dry_run=args.dry_run)
        counts[status] = counts.get(status, 0) + 1
        print(f"[{path.parent.name}] {status}")
    print("summary:", counts)


if __name__ == "__main__":
    main()
