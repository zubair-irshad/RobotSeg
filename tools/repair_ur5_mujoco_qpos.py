"""Repair existing Berkeley Autolab UR5 trajectory.npz MuJoCo qpos.

Older downloads in this repo saved only the first six UR5 arm joints under
trajectory['joint_position']. AugE replays the same RLDS data with a 14D qpos:
six arm joints plus eight synthetic gripper joints selected from
robot_state[-2]. This script rewrites trajectory.npz files in-place using the
saved trajectory['state'] array, so DOWNLOAD=0 runs can still render UR5 with
the same qpos convention as AugE.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _ur5_qpos_from_state(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    closed = state[:, -2] > 0.5
    qpos = np.zeros((state.shape[0], 14), dtype=np.float32)
    qpos[:, :6] = state[:, :6]
    closed_vals = np.asarray(
        [1.0, 0.025, 0.80, -0.80, 1.0, 0.0252768, 0.80, -0.80],
        dtype=np.float32,
    )
    qpos[closed, 6:] = closed_vals
    return qpos


def repair(path: Path, dry_run: bool = False) -> str:
    with np.load(path, allow_pickle=True) as data:
        if "state" not in data.files:
            return "skip-no-state"
        state = np.asarray(data["state"], dtype=np.float32)
        if state.ndim != 2 or state.shape[1] < 8:
            return f"skip-bad-state-shape-{state.shape}"
        qpos = _ur5_qpos_from_state(state)
        old = np.asarray(data["joint_position"], dtype=np.float32) if "joint_position" in data.files else None
        if old is not None and old.shape == qpos.shape and np.allclose(old, qpos):
            return "ok-already"
        arrays = {name: data[name] for name in data.files}
        arrays["joint_position"] = qpos
    if not dry_run:
        np.savez_compressed(path, **arrays)
    if old is None:
        return f"updated-created-{qpos.shape[1]}d"
    return f"updated-{old.shape}->{qpos.shape}"


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
