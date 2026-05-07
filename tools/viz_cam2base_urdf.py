"""Render a Franka/Robotiq URDF overlay using per-episode PnP extrinsics.

Expected inputs match the OXE branch pipeline:

    <oxe_root>/droid/episode_0000/frames/*.jpg
    <oxe_root>/droid/episode_0000/trajectory.npz
    <mask_root>/droid/episode_0000/pnp.json

Outputs are written to:

    <mask_root>/droid/episode_0000/urdf_viz/*.jpg
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


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_K(ep_seg: Path, pnp: dict, image_bgr: np.ndarray) -> dict[str, float]:
    K = dict(pnp.get("K") or {})
    if not K:
        moge = ep_seg / "moge_K.json"
        if moge.exists():
            K = {
                k: v
                for k, v in _load_json(moge).items()
                if k in {"fx", "fy", "cx", "cy", "width", "height", "source"}
            }
    if not all(k in K for k in ("fx", "fy", "cx", "cy")):
        raise ValueError(f"No usable K in {ep_seg / 'pnp.json'} or {ep_seg / 'moge_K.json'}")
    H, W = image_bgr.shape[:2]
    K["width"] = int(K.get("width", W))
    K["height"] = int(K.get("height", H))
    return {k: float(v) if k in {"fx", "fy", "cx", "cy"} else v for k, v in K.items()}


def _parse_bgr(text: str) -> tuple[int, int, int]:
    vals = tuple(int(x.strip()) for x in text.split(","))
    if len(vals) != 3 or any(v < 0 or v > 255 for v in vals):
        raise argparse.ArgumentTypeError("expected B,G,R values in 0..255, e.g. 255,210,80")
    return vals


def _load_joint_array(traj: np.lib.npyio.NpzFile, key: str | None) -> np.ndarray:
    candidates = []
    if key:
        candidates.append(key)
    candidates += [
        "joint_position",
        "joint_positions",
        "joints",
        "q",
        "arm_joints",
    ]
    for name in candidates:
        if name in traj.files:
            arr = np.asarray(traj[name], dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[-1] >= 7:
                return arr[..., :7].reshape(arr.shape[0], -1)

    if "state" in traj.files:
        state = np.asarray(traj["state"], dtype=np.float64)
        if state.ndim == 2 and state.shape[1] >= 7:
            return state[:, :7]

    raise ValueError(
        "trajectory.npz does not contain Franka joint positions. "
        "Re-run tools/download_oxe_subset.py on this oxe branch so DROID "
        "episodes save observation['joint_position'] as trajectory['joint_position']."
    )


def _load_gripper_array(traj: np.lib.npyio.NpzFile, n: int, key: str | None) -> np.ndarray:
    candidates = []
    if key:
        candidates.append(key)
    candidates += ["gripper", "gripper_position", "finger_joint"]
    for name in candidates:
        if name in traj.files:
            arr = np.asarray(traj[name], dtype=np.float64).reshape(-1)
            if len(arr) == n:
                return arr
    return np.zeros(n, dtype=np.float64)


def _frame_index_for_stem(stem: str, ordinal: int, n: int) -> int | None:
    try:
        idx = int(stem)
    except ValueError:
        idx = ordinal
    if 0 <= idx < n:
        return idx
    if 0 <= ordinal < n:
        return ordinal
    return None


def _select_frames(paths: list[Path], stride: int, max_frames: int) -> list[Path]:
    out = paths[:: max(1, stride)]
    if max_frames > 0:
        out = out[:max_frames]
    return out


def _write_preview(image_paths: list[Path], out_path: Path, tile_w: int = 320, cols: int = 4) -> None:
    if not image_paths:
        return
    tiles = []
    for path in image_paths[:16]:
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = tile_w / float(w)
        tile = cv2.resize(img, (tile_w, max(1, int(round(h * scale)))))
        tiles.append(tile)
    if not tiles:
        return
    tile_h = max(t.shape[0] for t in tiles)
    rows = []
    for i in range(0, len(tiles), cols):
        row_tiles = tiles[i : i + cols]
        padded = []
        for tile in row_tiles:
            if tile.shape[0] < tile_h:
                pad = np.zeros((tile_h - tile.shape[0], tile.shape[1], 3), dtype=np.uint8)
                tile = np.vstack([tile, pad])
            padded.append(tile)
        while len(padded) < cols:
            padded.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
        rows.append(np.hstack(padded))
    cv2.imwrite(str(out_path), np.vstack(rows))


def _image_paths(ep_oxe: Path, ep_seg: Path, source: str) -> list[Path]:
    roots: list[Path] = []
    if source in {"auto", "raw"}:
        roots.append(ep_oxe / "frames")
    if source in {"auto", "combined"}:
        roots.append(ep_seg / "combined")
    for root in roots:
        if root.is_dir():
            paths = sorted(
                p for p in root.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            if paths:
                return paths
    raise FileNotFoundError(f"No images found for {ep_seg.name}; checked {roots}")


def process_episode(
    dataset: str,
    ep_name: str,
    args: argparse.Namespace,
    masker: URDFRobotMasker,
) -> dict:
    ep_oxe = args.oxe_root / dataset / ep_name
    ep_seg = args.mask_root / dataset / ep_name
    pnp_path = ep_seg / args.pnp_json_name
    traj_path = ep_oxe / "trajectory.npz"
    if not pnp_path.exists():
        return {"episode": ep_name, "status": "skip-missing-pnp"}
    if not traj_path.exists():
        return {"episode": ep_name, "status": "skip-missing-trajectory"}

    pnp = _load_json(pnp_path)
    if args.skip_bad_pnp and pnp.get("status") != "ok":
        return {"episode": ep_name, "status": f"skip-pnp-{pnp.get('status', 'unknown')}"}
    T_cam2base = np.asarray(pnp["T_cam2base"], dtype=np.float64)

    with np.load(traj_path, allow_pickle=True) as traj:
        joints = _load_joint_array(traj, args.joint_key)
        gripper = _load_gripper_array(traj, len(joints), args.gripper_key)

    frames = _image_paths(ep_oxe, ep_seg, args.image_source)
    if len(joints) != len(frames) and not args.allow_partial_trajectory:
        return {
            "episode": ep_name,
            "status": "bad-frame-trajectory-count-mismatch",
            "num_frames": len(frames),
            "num_joint_positions": int(len(joints)),
            "hint": (
                "Regenerate data/oxe_subset with the same frame_stride used "
                "for segmentation/PnP, or pass --allow_partial_trajectory for debugging."
            ),
        }
    selected = _select_frames(frames, args.frame_stride, args.max_frames)
    if not selected:
        return {"episode": ep_name, "status": "skip-no-selected-frames"}

    out_dir = ep_seg / args.out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    rendered = 0
    skipped = 0
    first_img = cv2.imread(str(selected[0]))
    if first_img is None:
        return {"episode": ep_name, "status": "skip-first-image-unreadable"}
    K = _load_K(ep_seg, pnp, first_img)

    color = _parse_bgr(args.color_bgr)
    outline = _parse_bgr(args.outline_bgr)
    for ordinal, img_path in enumerate(frames):
        if img_path not in selected:
            continue
        idx = _frame_index_for_stem(img_path.stem, ordinal, len(joints))
        if idx is None:
            skipped += 1
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            skipped += 1
            continue
        overlay, mask = masker.overlay(
            image,
            K,
            T_cam2base,
            joints[idx],
            gripper_position=float(gripper[idx]),
            color_bgr=color,
            outline_bgr=outline,
            outline_px=int(args.outline_px),
            alpha=float(args.alpha),
        )
        if args.draw_status:
            cv2.putText(
                overlay,
                f"{dataset}/{ep_name} frame={img_path.stem} pnp={pnp.get('status', '?')} "
                f"mask={int(mask.sum())}px",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        out_path = out_dir / f"{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), overlay)
        saved.append(out_path)
        rendered += 1

    _write_preview(saved, out_dir / "_preview.jpg")
    return {
        "episode": ep_name,
        "status": "ok" if rendered else "bad-no-rendered-frames",
        "pnp_status": pnp.get("status"),
        "K_source": K.get("source", "unknown"),
        "rmse_px": pnp.get("rmse_px"),
        "num_inliers": pnp.get("num_inliers"),
        "num_frames": len(frames),
        "num_joint_positions": int(len(joints)),
        "rendered": rendered,
        "skipped": skipped,
        "out_dir": str(out_dir),
        "preview": str(out_dir / "_preview.jpg"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oxe_root", type=Path, default=Path("data/oxe_subset"))
    parser.add_argument("--mask_root", type=Path, default=Path("data/oxe_subset_seg"))
    parser.add_argument("--dataset", default="droid")
    parser.add_argument("--episodes", nargs="+", default=None)
    parser.add_argument("--urdf_path", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--mesh_dir", type=Path, default=None)
    parser.add_argument(
        "--pnp_json_name",
        default="pnp.json",
        help="Per-episode PnP JSON filename, e.g. pnp.json or pnp_rlds.json.",
    )
    parser.add_argument("--out_dir_name", default="urdf_viz")
    parser.add_argument("--image_source", choices=["auto", "raw", "combined"], default="auto")
    parser.add_argument("--joint_key", default=None)
    parser.add_argument("--gripper_key", default=None)
    parser.add_argument(
        "--allow_partial_trajectory",
        action="store_true",
        help="Render frames whose numeric stem can be matched to a joint row "
             "even when frame and trajectory counts differ. Debug only.",
    )
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--dilate_px", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument(
        "--color_bgr",
        default="255,195,52",
        help="Translucent overlay color as OpenCV B,G,R. Default is #34C3FF.",
    )
    parser.add_argument(
        "--outline_bgr",
        default="255,215,92",
        help="Optional outline color as OpenCV B,G,R. Used only when --outline_px > 0.",
    )
    parser.add_argument(
        "--outline_px",
        type=int,
        default=0,
        help="Contour thickness in pixels. Default 0 disables edges.",
    )
    parser.add_argument("--gripper_open_rad", type=float, default=0.0)
    parser.add_argument("--gripper_closed_rad", type=float, default=0.7)
    parser.add_argument("--skip_bad_pnp", action="store_true")
    parser.add_argument("--draw_status", action="store_true")
    args = parser.parse_args()

    if not args.urdf_path.exists():
        raise SystemExit(
            f"Missing URDF: {args.urdf_path}\n"
            "Clone it with: python tools/download_robot_urdfs.py"
        )

    ds_seg = args.mask_root / args.dataset
    if args.episodes:
        episodes = args.episodes
    else:
        episodes = sorted(p.name for p in ds_seg.iterdir() if p.is_dir() and p.name.startswith("episode_"))

    masker = URDFRobotMasker(
        args.urdf_path,
        mesh_dir=args.mesh_dir,
        downsample=args.downsample,
        dilate_px=args.dilate_px,
        gripper_open_rad=args.gripper_open_rad,
        gripper_closed_rad=args.gripper_closed_rad,
        verbose=True,
    )

    results = []
    for ep in episodes:
        result = process_episode(args.dataset, ep, args, masker)
        results.append(result)
        rmse = result.get("rmse_px")
        rmse_s = f"{rmse:.2f}px" if isinstance(rmse, (int, float)) else "?"
        print(
            f"[{args.dataset}/{ep}] {result['status']} "
            f"rendered={result.get('rendered', 0)} "
            f"frames={result.get('num_frames', '?')} "
            f"joints={result.get('num_joint_positions', '?')} "
            f"K={result.get('K_source', '?')} rmse={rmse_s} "
            f"inliers={result.get('num_inliers', '?')} "
            f"out={result.get('out_dir', '')}"
        )

    summary_path = ds_seg / "urdf_viz_summary.json"
    summary_path.write_text(json.dumps({"dataset": args.dataset, "results": results}, indent=2))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
