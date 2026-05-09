"""Render a URDF by solving per-frame IK to match TCP poses.

Use this for datasets like Fractal that publish TCP pose but not measured
robot joint positions. This reconstructs a plausible joint configuration from
the TCP; it is not equivalent to rendering logged joints.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from cam2base_json import invert_se3  # noqa: E402
from pnp_oxe import _eef_rot_to_R  # noqa: E402
from urdf_robot_masker import URDFRobotMasker  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _project(point_base: np.ndarray, K: dict[str, float], T_cam2base: np.ndarray) -> np.ndarray:
    T_base2cam = invert_se3(np.asarray(T_cam2base, dtype=np.float64))
    p = np.r_[np.asarray(point_base, dtype=np.float64).reshape(3), 1.0]
    c = (T_base2cam @ p)[:3]
    if c[2] <= 0.05:
        return np.array([np.nan, np.nan], dtype=np.float64)
    return np.array([
        float(K["fx"]) * c[0] / c[2] + float(K["cx"]),
        float(K["fy"]) * c[1] / c[2] + float(K["cy"]),
    ])


def _centroid(entry: Any) -> np.ndarray | None:
    if isinstance(entry, dict):
        val = entry.get("centroid")
    else:
        val = entry
    if val is None:
        return None
    arr = np.asarray(val, dtype=np.float64).reshape(-1)
    return arr[:2] if arr.size >= 2 else None


def _link_names(masker: URDFRobotMasker) -> list[str]:
    robot = masker.robot
    if hasattr(robot, "links"):
        return sorted(robot.links)
    inner = getattr(robot, "robot", None)
    if inner is not None and hasattr(inner, "link_map"):
        return sorted(inner.link_map.keys())
    return []


def _joint_names(masker: URDFRobotMasker) -> list[str]:
    robot = masker.robot
    if hasattr(robot, "joints"):
        return [
            j.name for j in robot.joints
            if j.joint_type not in {"fixed"} and j.mimic is None
        ]
    inner = getattr(robot, "robot", None)
    joint_map = getattr(inner, "joint_map", None) if inner is not None else None
    if joint_map is None:
        joint_map = getattr(inner, "_joint_map", {}) if inner is not None else {}
    names = []
    for name, joint in joint_map.items():
        typ = getattr(joint, "type", getattr(joint, "joint_type", ""))
        if typ != "fixed":
            names.append(str(name))
    return sorted(names)


def _auto_ee_link(links: list[str]) -> str:
    prefs = ("tool", "tcp", "gripper", "ee", "wrist", "finger")
    scored = []
    for link in links:
        low = link.lower()
        score = max((10 - i for i, p in enumerate(prefs) if p in low), default=0)
        scored.append((score, link))
    scored.sort()
    return scored[-1][1] if scored else ""


def _cfg(names: list[str], q: np.ndarray) -> dict[str, float]:
    return {name: float(q[i]) for i, name in enumerate(names)}


def _rot_err(R_a: np.ndarray, R_b: np.ndarray) -> np.ndarray:
    R = np.asarray(R_a, dtype=np.float64).T @ np.asarray(R_b, dtype=np.float64)
    rvec, _ = cv2.Rodrigues(R)
    return rvec.reshape(3)


def _solve_ik(
    masker: URDFRobotMasker,
    joint_names: list[str],
    ee_link: str,
    target_pos: np.ndarray,
    target_R: np.ndarray | None,
    q0: np.ndarray,
    orientation_weight: float,
    smooth_weight: float,
    max_nfev: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        from scipy.optimize import least_squares
    except ImportError as exc:
        raise RuntimeError("scipy is required for IK: pip install scipy") from exc

    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    q0 = np.asarray(q0, dtype=np.float64).reshape(len(joint_names))

    def residual(q: np.ndarray) -> np.ndarray:
        cfg = _cfg(joint_names, q)
        link_T = masker.robot.link_transforms(cfg).get(ee_link)
        if link_T is None:
            return np.ones(3, dtype=np.float64) * 1e3
        res = [10.0 * (link_T[:3, 3] - target_pos)]
        if target_R is not None and orientation_weight > 0:
            res.append(float(orientation_weight) * _rot_err(link_T[:3, :3], target_R))
        if smooth_weight > 0:
            res.append(float(smooth_weight) * (q - q0))
        return np.concatenate(res)

    sol = least_squares(residual, q0, max_nfev=max_nfev)
    q = sol.x.astype(np.float64)
    link_T = masker.robot.link_transforms(_cfg(joint_names, q)).get(ee_link)
    pos_err = float(np.linalg.norm(link_T[:3, 3] - target_pos)) if link_T is not None else float("nan")
    return q, {
        "success": bool(sol.success),
        "cost": float(sol.cost),
        "nfev": int(sol.nfev),
        "position_error_m": pos_err,
    }


def _load_rotations(traj: np.lib.npyio.NpzFile) -> np.ndarray | None:
    if "eef_rot" not in traj.files:
        return None
    rot = np.asarray(traj["eef_rot"], dtype=np.float64)
    fmt = "euler_xyz"
    if "eef_rot_format" in traj.files:
        raw = traj["eef_rot_format"]
        fmt = str(raw.tolist() if hasattr(raw, "tolist") else raw)
    try:
        return _eef_rot_to_R(rot, fmt)
    except Exception:
        return None


def process_episode(args: argparse.Namespace, ep: str, masker: URDFRobotMasker,
                    joint_names: list[str], ee_link: str) -> dict[str, Any]:
    ep_oxe = args.oxe_root / args.dataset / ep
    ep_seg = args.mask_root / args.dataset / ep
    pnp_path = ep_seg / args.pnp_json_name
    if not pnp_path.exists():
        return {"episode": ep, "status": "skip-missing-pnp"}
    pnp = _read_json(pnp_path)
    if pnp.get("status") != "ok" and args.skip_bad_pnp:
        return {"episode": ep, "status": f"skip-pnp-{pnp.get('status')}"}
    if "T_cam2base" not in pnp or "K" not in pnp:
        return {"episode": ep, "status": "skip-no-pose"}

    K = pnp["K"]
    T_cam2base = np.asarray(pnp["T_cam2base"], dtype=np.float64)
    centroids = _read_json(ep_seg / "001" / "centroids.json")
    stems = pnp.get("inlier_stems") or pnp.get("kept_stems") or sorted(centroids.keys())
    if args.max_frames > 0:
        stems = stems[:args.max_frames]

    with np.load(ep_oxe / "trajectory.npz", allow_pickle=True) as traj:
        if "eef_xyz" in traj.files:
            xyz = np.asarray(traj["eef_xyz"], dtype=np.float64)
        else:
            xyz = np.asarray(traj["state"], dtype=np.float64)[:, :3]
        R_seq = _load_rotations(traj)

    out_dir = ep_seg / args.out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    q = np.zeros(len(joint_names), dtype=np.float64)
    records = []
    saved = 0
    for stem in stems:
        stem = str(stem)
        if not stem.isdigit():
            continue
        idx = int(stem)
        if idx >= len(xyz):
            continue
        img = cv2.imread(str(ep_oxe / "frames" / f"{stem}.jpg"))
        if img is None:
            continue
        target_R = R_seq[idx] if R_seq is not None and idx < len(R_seq) else None
        q, ik = _solve_ik(
            masker,
            joint_names,
            ee_link,
            xyz[idx],
            target_R,
            q,
            args.orientation_weight,
            args.smooth_weight,
            args.max_ik_evals,
        )
        overlay, _ = masker.overlay_cfg(
            img,
            K,
            T_cam2base,
            _cfg(joint_names, q),
            color_bgr=(255, 170, 40),
            outline_bgr=(255, 225, 120),
            outline_px=1,
            alpha=0.60,
            include_link_prefixes=tuple(args.include_link_prefixes) if args.include_link_prefixes else None,
            exclude_link_prefixes=tuple(args.exclude_link_prefixes) if args.exclude_link_prefixes else None,
        )
        c = _centroid(centroids.get(stem))
        uv = _project(xyz[idx], K, T_cam2base)
        if c is not None:
            cv2.drawMarker(overlay, tuple(np.rint(c).astype(int)), (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
        if np.all(np.isfinite(uv)):
            cv2.circle(overlay, tuple(np.rint(uv).astype(int)), 7, (0, 220, 0), 2)
            if c is not None:
                cv2.line(overlay, tuple(np.rint(c).astype(int)), tuple(np.rint(uv).astype(int)), (0, 220, 0), 2)
        cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(
            overlay,
            f"URDF IK {ee_link} pos_err={ik['position_error_m']*100:.1f}cm",
            (6, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(out_dir / f"{stem}.jpg"), overlay)
        records.append({"stem": stem, "q": q.tolist(), **ik})
        saved += 1

    (out_dir / "ik_records.json").write_text(json.dumps({
        "episode": ep,
        "joint_names": joint_names,
        "ee_link": ee_link,
        "records": records,
    }, indent=2))
    return {
        "episode": ep,
        "status": "ok" if saved else "bad-no-frames",
        "saved": saved,
        "out_dir": str(out_dir),
        "median_ik_pos_err_cm": float(np.median([r["position_error_m"] for r in records]) * 100.0) if records else None,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--oxe_root", type=Path, required=True)
    p.add_argument("--mask_root", type=Path, required=True)
    p.add_argument("--dataset", default="fractal20220817_data")
    p.add_argument("--episodes", nargs="+", default=None)
    p.add_argument("--pnp_json_name", default="pnp_moge.json")
    p.add_argument("--urdf_path", type=Path, required=True)
    p.add_argument("--mesh_dir", type=Path, default=None)
    p.add_argument("--urdf_backend", choices=["yourdfpy", "simple", "auto"], default="yourdfpy")
    p.add_argument("--joint_names", nargs="+", default=None)
    p.add_argument("--ee_link", default=None)
    p.add_argument("--print_model_info", action="store_true")
    p.add_argument("--out_dir_name", default="urdf_tcp_ik_overlay")
    p.add_argument("--max_frames", type=int, default=24)
    p.add_argument("--max_ik_evals", type=int, default=80)
    p.add_argument("--orientation_weight", type=float, default=0.0)
    p.add_argument("--smooth_weight", type=float, default=0.02)
    p.add_argument("--include_link_prefixes", nargs="+", default=None)
    p.add_argument("--exclude_link_prefixes", nargs="+", default=None)
    p.add_argument("--skip_bad_pnp", action="store_true")
    args = p.parse_args()

    masker = URDFRobotMasker(
        args.urdf_path,
        mesh_dir=args.mesh_dir,
        backend=args.urdf_backend,
        downsample=2,
        dilate_px=2,
        verbose=True,
    )
    links = _link_names(masker)
    joints = _joint_names(masker)
    if args.print_model_info:
        print("Links:")
        for link in links:
            print(f"  {link}")
        print("Movable joints:")
        for joint in joints:
            print(f"  {joint}")
    joint_names = args.joint_names or joints
    ee_link = args.ee_link or _auto_ee_link(links)
    if not joint_names or not ee_link:
        raise SystemExit("No joint names or end-effector link found. Use --print_model_info, --joint_names, and --ee_link.")
    print(f"Using ee_link={ee_link}")
    print(f"Using joint_names={joint_names}")

    ds_seg = args.mask_root / args.dataset
    episodes = args.episodes or sorted(
        p.name for p in ds_seg.iterdir() if p.is_dir() and p.name.startswith("episode_")
    )
    results = []
    for ep in episodes:
        result = process_episode(args, ep, masker, joint_names, ee_link)
        results.append(result)
        print(
            f"[{args.dataset}/{ep}] {result['status']} saved={result.get('saved', 0)} "
            f"ik_med={result.get('median_ik_pos_err_cm')} out={result.get('out_dir', '')}"
        )
    out = ds_seg / "urdf_tcp_ik_summary.json"
    out.write_text(json.dumps({"dataset": args.dataset, "results": results}, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
