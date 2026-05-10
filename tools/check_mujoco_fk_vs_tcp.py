#!/usr/bin/env python3
"""Compare logged OXE TCP xyz against MuJoCo FK sites/bodies.

This checks the common failure mode where dataset TCP is expressed in the
robot base frame, while MuJoCo renders in a scene/world frame with the robot
base placed at a non-identity transform.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def _name(mujoco, model, obj, idx: int) -> str:
    return mujoco.mj_id2name(model, obj, int(idx)) or ""


def _all_names(mujoco, model, obj, n: int) -> list[str]:
    return [_name(mujoco, model, obj, i) for i in range(n)]


def _find_base_body(mujoco, model, requested: str | None) -> int:
    names = _all_names(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    if requested and requested != "auto":
        if requested in names:
            return names.index(requested)
        raise ValueError(f"base body {requested!r} not found. Bodies: {names}")
    for target in ("base", "base_link", "ur5_base", "ur5e_base", "world"):
        if target in names:
            return names.index(target)
    for i, name in enumerate(names):
        if "base" in name.lower():
            return i
    return 0


def _body_T_world(data, body_id: int) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    if body_id == 0:
        return T
    T[:3, :3] = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(data.xpos[body_id], dtype=np.float64)
    return T


def _load_qpos(traj, key: str) -> np.ndarray:
    if key in traj.files:
        arr = np.asarray(traj[key], dtype=np.float64)
    elif "joint_position" in traj.files:
        arr = np.asarray(traj["joint_position"], dtype=np.float64)
    else:
        arr = np.asarray(traj["state"], dtype=np.float64)[:, :6]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _load_tcp(traj) -> np.ndarray:
    if "eef_xyz" in traj.files:
        return np.asarray(traj["eef_xyz"], dtype=np.float64)
    return np.asarray(traj["state"], dtype=np.float64)[:, 6:9]


def _candidate_names(names: list[str], requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    keep = []
    needles = ("tcp", "tool", "ee", "eef", "end", "gripper", "finger", "wrist", "flange")
    for name in names:
        low = name.lower()
        if any(k in low for k in needles):
            keep.append(name)
    return keep or [n for n in names if n]


def _stats(err: list[float]) -> dict[str, float]:
    arr = np.asarray(err, dtype=np.float64)
    return {
        "n": int(arr.size),
        "median_m": float(np.median(arr)),
        "mean_m": float(arr.mean()),
        "p90_m": float(np.percentile(arr, 90)),
        "max_m": float(arr.max()),
    }


def process_episode(args, mujoco, model, ep_dir: Path) -> dict:
    traj_path = ep_dir / "trajectory.npz"
    with np.load(traj_path, allow_pickle=True) as traj:
        tcp = _load_tcp(traj)
        qpos = _load_qpos(traj, args.qpos_key)

    n = min(len(tcp), len(qpos))
    indices = list(range(0, n, max(1, args.frame_stride)))
    if args.max_frames > 0:
        indices = indices[:args.max_frames]

    data = mujoco.MjData(model)
    base_body_id = _find_base_body(mujoco, model, args.base_body)
    base_body_name = _name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, base_body_id)

    body_names = _all_names(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    site_names = _all_names(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, model.nsite)
    body_candidates = _candidate_names(body_names, args.body_names)
    site_candidates = _candidate_names(site_names, args.site_names)

    rows: list[dict] = []
    for kind, names in (("site", site_candidates), ("body", body_candidates)):
        for name in names:
            if not name:
                continue
            obj = mujoco.mjtObj.mjOBJ_SITE if kind == "site" else mujoco.mjtObj.mjOBJ_BODY
            obj_id = mujoco.mj_name2id(model, obj, name)
            if obj_id < 0:
                continue
            err_world = []
            err_base = []
            offsets_base = []
            for idx in indices:
                data.qpos[: min(model.nq, qpos.shape[1])] = qpos[idx, : min(model.nq, qpos.shape[1])]
                mujoco.mj_forward(model, data)
                T_world_base = _body_T_world(data, base_body_id)
                T_base_world = np.linalg.inv(T_world_base)
                p_world = (
                    np.asarray(data.site_xpos[obj_id], dtype=np.float64)
                    if kind == "site"
                    else np.asarray(data.xpos[obj_id], dtype=np.float64)
                )
                p_base = (T_base_world @ np.r_[p_world, 1.0])[:3]
                err_world.append(float(np.linalg.norm(p_world - tcp[idx])))
                err_base.append(float(np.linalg.norm(p_base - tcp[idx])))
                offsets_base.append((p_base - tcp[idx]).astype(float))
            row = {
                "kind": kind,
                "name": name,
                "world": _stats(err_world),
                "base": _stats(err_base),
                "median_base_offset_xyz_m": np.median(np.asarray(offsets_base), axis=0).astype(float).tolist(),
            }
            rows.append(row)

    rows.sort(key=lambda r: r["base"]["median_m"])
    return {
        "episode": ep_dir.name,
        "num_frames": len(indices),
        "base_body": base_body_name,
        "best": rows[0] if rows else None,
        "rows": rows,
        "body_names": body_names if args.print_names else None,
        "site_names": site_names if args.print_names else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oxe_root", type=Path, required=True)
    parser.add_argument("--dataset", default="berkeley_autolab_ur5")
    parser.add_argument("--mujoco_xml_path", type=Path, required=True)
    parser.add_argument("--qpos_key", default="joint_position")
    parser.add_argument("--base_body", default="auto")
    parser.add_argument("--episodes", nargs="+", default=None)
    parser.add_argument("--site_names", nargs="+", default=None)
    parser.add_argument("--body_names", nargs="+", default=None)
    parser.add_argument("--frame_stride", type=int, default=5)
    parser.add_argument("--max_frames", type=int, default=64)
    parser.add_argument("--print_names", action="store_true")
    parser.add_argument("--out_json", type=Path, default=None)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco  # type: ignore

    model = mujoco.MjModel.from_xml_path(str(args.mujoco_xml_path.expanduser()))
    ds_root = args.oxe_root / args.dataset
    episodes = args.episodes or sorted(
        p.name for p in ds_root.iterdir()
        if p.is_dir() and p.name.startswith("episode_")
    )
    results = []
    for ep in episodes:
        result = process_episode(args, mujoco, model, ds_root / ep)
        results.append(result)
        best = result.get("best")
        if best is None:
            print(f"[{args.dataset}/{ep}] no candidates")
            continue
        print(
            f"[{args.dataset}/{ep}] base_body={result['base_body']} "
            f"best={best['kind']}:{best['name']} "
            f"base_med={100*best['base']['median_m']:.2f}cm "
            f"world_med={100*best['world']['median_m']:.2f}cm "
            f"offset={np.round(best['median_base_offset_xyz_m'], 4).tolist()}"
        )
        for row in result["rows"][: min(10, len(result["rows"]))]:
            print(
                f"  {row['kind']:<4} {row['name']:<32} "
                f"base_med={100*row['base']['median_m']:7.2f}cm "
                f"world_med={100*row['world']['median_m']:7.2f}cm "
                f"off={np.round(row['median_base_offset_xyz_m'], 4).tolist()}"
            )

    out = args.out_json or (ds_root / "mujoco_fk_vs_tcp.json")
    out.write_text(json.dumps({"dataset": args.dataset, "results": results}, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
