"""Diagnose DROID PnP vs URDF FK alignment without changing calibration.

This script compares two projections against the RobotSeg gripper centroid:

1. PnP input point: trajectory["eef_xyz"] projected with pnp.json K/T.
2. URDF FK link origins from trajectory["joint_position"], projected with the
   exact same pnp.json K/T.

If (1) is low error but every URDF FK candidate is high error, then the PnP
calibration is internally consistent for DROID Cartesian state, but the URDF
base/link convention does not match the joint_position/cartesian_position
convention used by the dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from urdf_robot_masker import URDFRobotMasker  # noqa: E402


DEFAULT_URDF = (
    Path("data")
    / "urdfs"
    / "python-example-droid-dataset"
    / "franka_description"
    / "panda.urdf"
)


def _invert_se3(T: np.ndarray) -> np.ndarray:
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -Ti[:3, :3] @ T[:3, 3]
    return Ti


def _project(points_base: np.ndarray, T_cam2base: np.ndarray, K: dict) -> np.ndarray:
    T_base2cam = _invert_se3(T_cam2base)
    pts = np.asarray(points_base, dtype=np.float64).reshape(-1, 3)
    pts_h = np.c_[pts, np.ones(len(pts))]
    cam = (T_base2cam @ pts_h.T).T[:, :3]
    z = cam[:, 2]
    out = np.full((len(pts), 2), np.nan, dtype=np.float64)
    good = z > 0.05
    out[good, 0] = float(K["fx"]) * cam[good, 0] / z[good] + float(K["cx"])
    out[good, 1] = float(K["fy"]) * cam[good, 1] / z[good] + float(K["cy"])
    return out


def _centroid(entry) -> list[float] | None:
    if entry is None:
        return None
    if isinstance(entry, dict):
        return entry.get("centroid")
    return entry


def _load_joints(traj: np.lib.npyio.NpzFile) -> np.ndarray:
    for key in ("joint_position", "joint_positions", "joints", "q", "arm_joints"):
        if key in traj.files:
            arr = np.asarray(traj[key], dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] >= 7:
                return arr[:, :7]
    raise ValueError("trajectory.npz has no joint_position-like array")


def _load_gripper(traj: np.lib.npyio.NpzFile, n: int) -> np.ndarray:
    for key in ("gripper", "gripper_position", "finger_joint"):
        if key in traj.files:
            arr = np.asarray(traj[key], dtype=np.float64).reshape(-1)
            if len(arr) == n:
                return arr
    return np.zeros(n, dtype=np.float64)


def _load_eef(traj: np.lib.npyio.NpzFile, n: int) -> np.ndarray | None:
    if "eef_xyz" in traj.files:
        arr = np.asarray(traj["eef_xyz"], dtype=np.float64)
        if arr.ndim == 2 and arr.shape[0] == n and arr.shape[1] >= 3:
            return arr[:, :3]
    if "state" in traj.files:
        arr = np.asarray(traj["state"], dtype=np.float64)
        if arr.ndim == 2 and arr.shape[0] == n and arr.shape[1] >= 3:
            return arr[:, :3]
    return None


def _stats(errors: list[float]) -> dict:
    arr = np.asarray(errors, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"n": 0}
    return {
        "n": int(len(arr)),
        "mean_px": float(arr.mean()),
        "median_px": float(np.median(arr)),
        "p90_px": float(np.percentile(arr, 90)),
        "max_px": float(arr.max()),
    }


def _frame_idx(stem: str, ordinal: int, n: int) -> int | None:
    try:
        idx = int(stem)
    except ValueError:
        idx = ordinal
    if 0 <= idx < n:
        return idx
    if 0 <= ordinal < n:
        return ordinal
    return None


def process_episode(args: argparse.Namespace, masker: URDFRobotMasker, ep_name: str) -> dict:
    ep_oxe = args.oxe_root / args.dataset / ep_name
    ep_seg = args.mask_root / args.dataset / ep_name
    pnp = json.loads((ep_seg / args.pnp_json_name).read_text())
    K = pnp["K"]
    T_cam2base = np.asarray(pnp["T_cam2base"], dtype=np.float64)
    centroids = json.loads((ep_seg / "001" / "centroids.json").read_text())

    with np.load(ep_oxe / "trajectory.npz", allow_pickle=True) as traj:
        joints = _load_joints(traj)
        gripper = _load_gripper(traj, len(joints))
        eef = _load_eef(traj, len(joints))

    stems = sorted(centroids.keys())
    observed = []
    for ordinal, stem in enumerate(stems):
        idx = _frame_idx(stem, ordinal, len(joints))
        c = _centroid(centroids[stem])
        if idx is None or c is None:
            continue
        observed.append((stem, idx, np.asarray(c, dtype=np.float64)))

    out = {
        "episode": ep_name,
        "K_source": K.get("source", "unknown"),
        "pnp_rmse_px": pnp.get("rmse_px"),
        "pnp_inliers": pnp.get("num_inliers"),
        "num_centroids": len(stems),
        "num_compared": len(observed),
        "num_joint_positions": int(len(joints)),
    }

    if eef is not None:
        errs = []
        for _, idx, c in observed:
            uv = _project(eef[idx], T_cam2base, K)[0]
            errs.append(float(np.linalg.norm(uv - c)))
        out["pnp_eef_xyz"] = _stats(errs)

    for link in args.links:
        errs = []
        for _, idx, c in observed:
            cfg = masker._cfg(joints[idx], float(gripper[idx]))
            link_T = masker.robot.link_transforms(cfg).get(link)
            if link_T is None:
                continue
            uv = _project(link_T[:3, 3], T_cam2base, K)[0]
            errs.append(float(np.linalg.norm(uv - c)))
        out[f"urdf_{link}"] = _stats(errs)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oxe_root", type=Path, default=Path("data/oxe_subset"))
    parser.add_argument("--mask_root", type=Path, default=Path("data/oxe_subset_seg"))
    parser.add_argument("--dataset", default="droid")
    parser.add_argument("--episodes", nargs="+", default=None)
    parser.add_argument("--pnp_json_name", default="pnp.json")
    parser.add_argument("--urdf_path", type=Path, default=DEFAULT_URDF)
    parser.add_argument(
        "--links",
        nargs="+",
        default=[
            "panda_link7",
            "panda_link8",
            "robotiq_85_base_link",
            "left_outer_finger",
            "right_outer_finger",
        ],
    )
    args = parser.parse_args()

    ds_seg = args.mask_root / args.dataset
    episodes = args.episodes or sorted(
        p.name for p in ds_seg.iterdir() if p.is_dir() and p.name.startswith("episode_")
    )
    masker = URDFRobotMasker(args.urdf_path, downsample=4, dilate_px=0, verbose=False)

    results = []
    for ep in episodes:
        result = process_episode(args, masker, ep)
        results.append(result)
        print(f"\n[{args.dataset}/{ep}] K={result['K_source']} pnp_rmse={result.get('pnp_rmse_px')}")
        for key, val in result.items():
            if not isinstance(val, dict) or "median_px" not in val:
                continue
            print(
                f"  {key:24s} n={val['n']:4d} "
                f"median={val['median_px']:7.2f}px mean={val['mean_px']:7.2f}px "
                f"p90={val['p90_px']:7.2f}px"
            )

    out_path = ds_seg / "urdf_alignment_diagnostics.json"
    out_path.write_text(json.dumps({"dataset": args.dataset, "results": results}, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
