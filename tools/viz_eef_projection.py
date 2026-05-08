"""Visualize 3D EEF/TCP candidates projected with a chosen cam2base extrinsic."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from cam2base_json import SIXD_MODES, episode_lookup_keys, find_T_cam2base, invert_se3, load_json  # noqa: E402
from urdf_robot_masker import URDFRobotMasker  # noqa: E402
from viz_cam2base_urdf import DEFAULT_URDF  # noqa: E402


def _load_pnp(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _serial_from_pnp(pnp: dict[str, Any]) -> str | None:
    K = pnp.get("K", {})
    source = K.get("source") if isinstance(K, dict) else None
    if isinstance(source, str):
        m = re.search(r"serial=([0-9]+)", source)
        if m:
            return m.group(1)
    return None


def _episode_id_from_pnp(pnp: dict[str, Any]) -> str | None:
    K = pnp.get("K", {})
    source = K.get("source") if isinstance(K, dict) else None
    if isinstance(source, str):
        m = re.search(r"episode_id=([^ ]+)", source)
        if m:
            return m.group(1)
    return None


def _load_T_cam2base(dataset: str, ep_name: str, args: argparse.Namespace,
                     pnp: dict[str, Any]) -> tuple[np.ndarray, str]:
    if args.extrinsics_json is None:
        return np.asarray(pnp["T_cam2base"], dtype=np.float64), args.pnp_json_name
    data = load_json(args.extrinsics_json)
    keys = episode_lookup_keys(args.oxe_root, dataset, ep_name)
    raw_id = _episode_id_from_pnp(pnp)
    if raw_id and raw_id not in keys:
        keys += [raw_id, f"{dataset}/{raw_id}"]
    preferred = []
    serial = args.camera_serial or _serial_from_pnp(pnp)
    if serial:
        preferred.append(serial)
    T, source = find_T_cam2base(
        data,
        keys,
        preferred_fields=preferred,
        sixd_mode=args.extrinsics_sixd_mode,
    )
    if T is None:
        raise KeyError(f"could not find {dataset}/{ep_name} in {args.extrinsics_json}; keys={keys}")
    return T, f"{args.extrinsics_json}:{source}"


def _project(points_base: np.ndarray, T_cam2base: np.ndarray,
             K: dict[str, float]) -> np.ndarray:
    T_base2cam = invert_se3(np.asarray(T_cam2base, dtype=np.float64))
    pts = np.asarray(points_base, dtype=np.float64).reshape(-1, 3)
    pts_h = np.c_[pts, np.ones(len(pts))]
    cam = (T_base2cam @ pts_h.T).T[:, :3]
    z = cam[:, 2]
    out = np.full((len(pts), 2), np.nan, dtype=np.float64)
    good = z > 0.05
    out[good, 0] = float(K["fx"]) * cam[good, 0] / z[good] + float(K["cx"])
    out[good, 1] = float(K["fy"]) * cam[good, 1] / z[good] + float(K["cy"])
    return out


def _centroid(entry: Any) -> np.ndarray | None:
    if entry is None:
        return None
    if isinstance(entry, dict):
        val = entry.get("centroid")
    else:
        val = entry
    if val is None:
        return None
    arr = np.asarray(val, dtype=np.float64).reshape(-1)
    return arr[:2] if arr.size >= 2 else None


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
            if len(arr) >= n:
                return arr[:n]
    return np.zeros(n, dtype=np.float64)


def _load_eef(traj: np.lib.npyio.NpzFile, n: int) -> np.ndarray | None:
    if "eef_xyz" in traj.files:
        arr = np.asarray(traj["eef_xyz"], dtype=np.float64)
        if arr.ndim == 2 and arr.shape[0] >= n and arr.shape[1] >= 3:
            return arr[:n, :3]
    if "state" in traj.files:
        arr = np.asarray(traj["state"], dtype=np.float64)
        if arr.ndim == 2 and arr.shape[0] >= n and arr.shape[1] >= 3:
            return arr[:n, :3]
    return None


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


def _image_paths(ep_oxe: Path, ep_seg: Path, source: str) -> list[Path]:
    roots = []
    if source in {"auto", "raw"}:
        roots.append(ep_oxe / "frames")
    if source in {"auto", "combined"}:
        roots.append(ep_seg / "combined")
    for root in roots:
        if root.is_dir():
            paths = sorted(p for p in root.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            if paths:
                return paths
    raise FileNotFoundError(f"no images found; checked {roots}")


def _stats(errors: list[float]) -> dict[str, float]:
    arr = np.asarray(errors, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"n": 0}
    return {
        "n": int(len(arr)),
        "median_px": float(np.median(arr)),
        "mean_px": float(arr.mean()),
        "p90_px": float(np.percentile(arr, 90)),
        "max_px": float(arr.max()),
    }


def _draw_point(img: np.ndarray, uv: np.ndarray, color: tuple[int, int, int],
                label: str, marker: int, args: argparse.Namespace) -> None:
    if not np.all(np.isfinite(uv)):
        return
    x, y = int(round(float(uv[0]))), int(round(float(uv[1])))
    cv2.drawMarker(
        img, (x, y), color, marker, int(args.marker_size),
        int(args.marker_thickness), cv2.LINE_AA,
    )
    if label:
        cv2.putText(
            img, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
            float(args.label_scale), color, 1, cv2.LINE_AA,
        )


def _draw_legend(img: np.ndarray, items: list[tuple[str, tuple[int, int, int]]],
                 args: argparse.Namespace) -> None:
    if not args.legend or not items:
        return
    x0, y0 = 8, 16
    line_h = max(11, int(20 * float(args.label_scale)))
    for i, (name, color) in enumerate(items):
        y = y0 + i * line_h
        cv2.circle(img, (x0, y - 4), 3, color, -1, cv2.LINE_AA)
        cv2.putText(
            img, name, (x0 + 9, y), cv2.FONT_HERSHEY_SIMPLEX,
            float(args.label_scale), color, 1, cv2.LINE_AA,
        )


_CANDIDATE_HELP = {
    "eef_xyz": "DROID cartesian_position[:3] / proprio EEF point",
    "panda_link7": "Franka link7 origin before fixed flange",
    "panda_link8": "Franka flange frame after 0.107 m offset",
    "robotiq_85_base_link": "Robotiq gripper base frame",
    "left_outer_finger": "left outer finger link origin",
    "right_outer_finger": "right outer finger link origin",
    "left_inner_finger": "left inner finger link origin",
    "right_inner_finger": "right inner finger link origin",
    "finger_midpoint": "midpoint of left/right outer finger origins",
    "inner_finger_midpoint": "midpoint of left/right inner finger origins",
    "all_finger_midpoint": "mean of all available finger link origins",
}


def _candidate_help(name: str) -> str:
    return _CANDIDATE_HELP.get(name, "URDF link origin")


def _put_small_text(img: np.ndarray, text: str, xy: tuple[int, int],
                    color: tuple[int, int, int], scale: float) -> None:
    x, y = xy
    cv2.putText(
        img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
        (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
        color, 1, cv2.LINE_AA,
    )


def _draw_centroid(img: np.ndarray, c: np.ndarray, args: argparse.Namespace) -> None:
    cv2.drawMarker(
        img, tuple(np.rint(c).astype(int)), (0, 255, 255),
        cv2.MARKER_CROSS, int(args.marker_size) + 4,
        int(args.marker_thickness), cv2.LINE_AA,
    )


def _draw_candidate_panel(
    image: np.ndarray,
    c: np.ndarray,
    name: str,
    uv: np.ndarray,
    color: tuple[int, int, int],
    marker: int,
    args: argparse.Namespace,
) -> np.ndarray:
    panel = image.copy()
    _draw_centroid(panel, c, args)
    dist_text = "not visible"
    if np.all(np.isfinite(uv)):
        err = float(np.linalg.norm(uv - c))
        dist_text = f"{err:.1f}px from seg centroid"
        _draw_point(panel, uv, color, "", marker, args)
        pt0 = tuple(np.rint(c).astype(int))
        pt1 = tuple(np.rint(uv).astype(int))
        cv2.line(panel, pt0, pt1, color, 1, cv2.LINE_AA)
    _put_small_text(panel, name, (7, 14), color, float(args.panel_label_scale))
    if args.panel_descriptions:
        _put_small_text(
            panel, _candidate_help(name), (7, 29), color,
            float(args.panel_label_scale) * 0.82,
        )
    _put_small_text(
        panel, dist_text, (7, panel.shape[0] - 8), color,
        float(args.panel_label_scale) * 0.9,
    )
    return panel


def _make_candidate_panels(
    image: np.ndarray,
    c: np.ndarray,
    draw_uvs: dict[str, np.ndarray],
    palette: dict[str, tuple[tuple[int, int, int], int]],
    args: argparse.Namespace,
) -> np.ndarray:
    panels = []
    base = image.copy()
    _draw_centroid(base, c, args)
    _put_small_text(base, "segmentation centroid", (7, 14), (0, 255, 255),
                    float(args.panel_label_scale))
    panels.append(base)
    for name, uv in draw_uvs.items():
        color, marker = palette.get(name, ((255, 255, 255), cv2.MARKER_CROSS))
        panels.append(_draw_candidate_panel(image, c, name, uv, color, marker, args))

    cols = max(1, int(args.panel_cols))
    h, w = image.shape[:2]
    rows = int(np.ceil(len(panels) / cols))
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, panel in enumerate(panels):
        r, col = divmod(i, cols)
        canvas[r * h:(r + 1) * h, col * w:(col + 1) * w] = panel
    return canvas


def process_episode(args: argparse.Namespace, ep_name: str,
                    masker: URDFRobotMasker | None) -> dict[str, Any]:
    ep_oxe = args.oxe_root / args.dataset / ep_name
    ep_seg = args.mask_root / args.dataset / ep_name
    pnp = _load_pnp(ep_seg / args.pnp_json_name)
    K = pnp["K"]
    T_cam2base, T_source = _load_T_cam2base(args.dataset, ep_name, args, pnp)
    centroids = json.loads((ep_seg / "001" / "centroids.json").read_text())
    frames = _image_paths(ep_oxe, ep_seg, args.image_source)

    with np.load(ep_oxe / "trajectory.npz", allow_pickle=True) as traj:
        joints = _load_joints(traj)
        gripper = _load_gripper(traj, len(joints))
        eef = _load_eef(traj, len(joints))

    out_dir = ep_seg / args.out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates: dict[str, list[float]] = {}
    saved = 0
    selected = frames[:: max(1, args.frame_stride)]
    if args.max_frames > 0:
        selected = selected[:args.max_frames]
    for ordinal, img_path in enumerate(frames):
        if img_path not in selected:
            continue
        idx = _frame_idx(img_path.stem, ordinal, len(joints))
        if idx is None:
            continue
        c = _centroid(centroids.get(img_path.stem))
        if c is None:
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        points: dict[str, np.ndarray] = {}
        if eef is not None:
            points["eef_xyz"] = eef[idx]
        if masker is not None:
            cfg = masker._cfg(joints[idx], float(gripper[idx]))
            link_T = masker.robot.link_transforms(cfg)
            for link in args.links:
                if link in link_T:
                    points[link] = link_T[link][:3, 3]
            if (
                "left_outer_finger" in link_T
                and "right_outer_finger" in link_T
            ):
                points["finger_midpoint"] = 0.5 * (
                    link_T["left_outer_finger"][:3, 3]
                    + link_T["right_outer_finger"][:3, 3]
                )
            if (
                "left_inner_finger" in link_T
                and "right_inner_finger" in link_T
            ):
                points["inner_finger_midpoint"] = 0.5 * (
                    link_T["left_inner_finger"][:3, 3]
                    + link_T["right_inner_finger"][:3, 3]
                )
            finger_links = [
                name for name in (
                    "left_outer_finger",
                    "right_outer_finger",
                    "left_inner_finger",
                    "right_inner_finger",
                )
                if name in link_T
            ]
            if finger_links:
                points["all_finger_midpoint"] = np.stack(
                    [link_T[name][:3, 3] for name in finger_links], axis=0
                ).mean(axis=0)
        if args.draw_candidates:
            allowed = set(args.draw_candidates)
            points = {k: v for k, v in points.items() if k in allowed}

        uvs = {name: _project(pt, T_cam2base, K)[0] for name, pt in points.items()}
        for name, uv in uvs.items():
            if np.all(np.isfinite(uv)):
                candidates.setdefault(name, []).append(float(np.linalg.norm(uv - c)))

        palette = {
            "eef_xyz": ((0, 0, 255), cv2.MARKER_DIAMOND),
            "panda_link7": ((255, 0, 0), cv2.MARKER_TILTED_CROSS),
            "panda_link8": ((0, 180, 255), cv2.MARKER_STAR),
            "robotiq_85_base_link": ((255, 0, 255), cv2.MARKER_TRIANGLE_UP),
            "left_outer_finger": ((0, 255, 0), cv2.MARKER_SQUARE),
            "right_outer_finger": ((255, 255, 0), cv2.MARKER_SQUARE),
            "finger_midpoint": ((0, 140, 255), cv2.MARKER_STAR),
            "left_inner_finger": ((80, 255, 80), cv2.MARKER_TRIANGLE_UP),
            "right_inner_finger": ((255, 255, 80), cv2.MARKER_TRIANGLE_UP),
            "inner_finger_midpoint": ((0, 110, 220), cv2.MARKER_STAR),
            "all_finger_midpoint": ((0, 80, 200), cv2.MARKER_STAR),
        }
        draw_uvs = dict(uvs)
        if args.draw_best_only and draw_uvs:
            valid = {
                name: float(np.linalg.norm(uv - c))
                for name, uv in draw_uvs.items()
                if np.all(np.isfinite(uv))
            }
            if valid:
                best = min(valid, key=valid.get)
                draw_uvs = {best: draw_uvs[best]}
        if args.candidate_layout == "panels":
            viz = _make_candidate_panels(image, c, draw_uvs, palette, args)
            cv2.imwrite(str(out_dir / f"{img_path.stem}.jpg"), viz)
        elif args.candidate_layout == "files":
            stem_dir = out_dir / img_path.stem
            stem_dir.mkdir(parents=True, exist_ok=True)
            base = image.copy()
            _draw_centroid(base, c, args)
            cv2.imwrite(str(stem_dir / "seg_centroid.jpg"), base)
            for name, uv in draw_uvs.items():
                color, marker = palette.get(name, ((255, 255, 255), cv2.MARKER_CROSS))
                viz = _draw_candidate_panel(image, c, name, uv, color, marker, args)
                cv2.imwrite(str(stem_dir / f"{name}.jpg"), viz)
        else:
            _draw_centroid(image, c, args)
            legend_items = [("seg centroid", (0, 255, 255))]
            for name, uv in draw_uvs.items():
                color, marker = palette.get(name, ((255, 255, 255), cv2.MARKER_CROSS))
                label = name if args.label_mode == "all" else ""
                if args.label_mode == "best" and args.draw_best_only:
                    label = name
                _draw_point(image, uv, color, label, marker, args)
                legend_items.append((name, color))
            _draw_legend(image, legend_items, args)
            cv2.imwrite(str(out_dir / f"{img_path.stem}.jpg"), image)
        saved += 1

    stats = {name: _stats(errs) for name, errs in candidates.items()}
    return {
        "episode": ep_name,
        "status": "ok" if saved else "bad-no-viz",
        "T_source": T_source,
        "K_source": K.get("source", "unknown"),
        "saved": saved,
        "stats": stats,
        "out_dir": str(out_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oxe_root", type=Path, default=Path("data/oxe_subset"))
    parser.add_argument("--mask_root", type=Path, default=Path("data/oxe_subset_seg"))
    parser.add_argument("--dataset", default="droid")
    parser.add_argument("--episodes", nargs="+", default=None)
    parser.add_argument("--pnp_json_name", default="pnp.json")
    parser.add_argument("--extrinsics_json", type=Path, default=None)
    parser.add_argument("--extrinsics_sixd_mode", choices=SIXD_MODES, default="rpy_cam2base")
    parser.add_argument("--camera_serial", default=None)
    parser.add_argument("--urdf_path", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--urdf_backend", choices=["simple", "yourdfpy", "auto"], default="simple")
    parser.add_argument("--image_source", choices=["auto", "raw", "combined"], default="auto")
    parser.add_argument("--out_dir_name", default="eef_projection_viz")
    parser.add_argument("--frame_stride", type=int, default=10)
    parser.add_argument("--max_frames", type=int, default=40)
    parser.add_argument("--marker_size", type=int, default=13)
    parser.add_argument("--marker_thickness", type=int, default=2)
    parser.add_argument("--label_scale", type=float, default=0.32)
    parser.add_argument("--label_mode", choices=["none", "best", "all"], default="none")
    parser.add_argument("--legend", action="store_true")
    parser.add_argument("--draw_best_only", action="store_true")
    parser.add_argument(
        "--candidate_layout",
        choices=["overlay", "panels", "files"],
        default="overlay",
        help="overlay draws all candidates on one frame; panels/files draw one candidate per view.",
    )
    parser.add_argument("--panel_cols", type=int, default=3)
    parser.add_argument("--panel_label_scale", type=float, default=0.34)
    parser.add_argument("--panel_descriptions", action="store_true", default=True)
    parser.add_argument(
        "--no_panel_descriptions",
        dest="panel_descriptions",
        action="store_false",
    )
    parser.add_argument(
        "--draw_candidates",
        nargs="+",
        default=None,
        help="Optional subset to draw, e.g. eef_xyz left_outer_finger right_outer_finger.",
    )
    parser.add_argument(
        "--links",
        nargs="+",
        default=[
            "panda_link7",
            "panda_link8",
            "robotiq_85_base_link",
            "left_outer_finger",
            "right_outer_finger",
            "left_inner_finger",
            "right_inner_finger",
        ],
    )
    args = parser.parse_args()

    ds_seg = args.mask_root / args.dataset
    episodes = args.episodes or sorted(
        p.name for p in ds_seg.iterdir() if p.is_dir() and p.name.startswith("episode_")
    )
    masker = URDFRobotMasker(args.urdf_path, backend=args.urdf_backend, downsample=4,
                             dilate_px=0, verbose=False)

    results = []
    for ep in episodes:
        result = process_episode(args, ep, masker)
        results.append(result)
        print(f"\n[{args.dataset}/{ep}] {result['status']} T={result['T_source']}")
        for name, stat in result.get("stats", {}).items():
            print(
                f"  {name:22s} n={stat.get('n', 0):4d} "
                f"median={stat.get('median_px', float('nan')):7.2f}px "
                f"mean={stat.get('mean_px', float('nan')):7.2f}px "
                f"p90={stat.get('p90_px', float('nan')):7.2f}px"
            )
        print(f"  out={result.get('out_dir')}")

    out_path = ds_seg / f"eef_projection_stats_{Path(args.pnp_json_name).stem}.json"
    out_path.write_text(json.dumps({"dataset": args.dataset, "results": results}, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
