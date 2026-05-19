"""Smoke test for tools/mujoco_ur5_renderer.py.

Loads the per-episode PnP result (T_cam2base + K) and the trajectory.npz
(state or 14D joint_position), then writes an overlay video that registers the
rendered UR5e onto the real camera frames.

Example:
  python tools/test_ur5_pnp_render.py \
      --pnp        data/oxe_subset_seg_ur5/native/current_pipeline/berkeley_autolab_ur5/episode_0000/pnp_fovy57.82240163683314.json \
      --trajectory data/oxe_subset_ur5/berkeley_autolab_ur5/episode_0000/trajectory.npz \
      --images     data/oxe_subset_ur5/berkeley_autolab_ur5/episode_0000/images/ \
      --out        /tmp/ur5_pnp_overlay_0000.mp4 \
      --side-by-side
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import imageio.v3 as iio

from mujoco_ur5_renderer import MuJoCoUR5Renderer, UR5_CLOSED_GRIPPER  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = REPO_ROOT / "third_party" / "AugE-Toolkit" / "robot_xml" / "universal_robots_ur5e" / "scene.xml"


def load_image_frames(p: Path) -> np.ndarray:
    if p.is_dir():
        files = sorted(q for q in p.iterdir() if q.suffix.lower() in (".png", ".jpg", ".jpeg"))
        if not files:
            raise FileNotFoundError(f"no frames under {p}")
        return np.stack([iio.imread(q) for q in files], axis=0)
    return np.asarray(iio.imread(p))


def load_qpos_and_gripper(traj: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (qpos14, gripper_closed_bool) per frame. AugE convention."""
    with np.load(traj, allow_pickle=True) as f:
        if "joint_position" in f.files and f["joint_position"].shape[-1] == 14:
            qpos = np.asarray(f["joint_position"], dtype=np.float32)
            if "state" in f.files:
                gc = np.asarray(f["state"][:, 13], dtype=np.float32) > 0.5
            else:
                # Fall back to inspecting the qpos itself.
                gc = np.linalg.norm(qpos[:, 6:] - UR5_CLOSED_GRIPPER, axis=1) < 1e-3
            return qpos, gc
        if "state" not in f.files:
            raise KeyError(f"{traj} has neither joint_position nor state")
        state = np.asarray(f["state"], dtype=np.float32)
        qpos  = np.zeros((state.shape[0], 14), dtype=np.float32)
        qpos[:, :6] = state[:, :6]
        gc = state[:, 13] > 0.5
        qpos[gc, 6:] = UR5_CLOSED_GRIPPER
        return qpos, gc


def load_pnp(p: Path) -> tuple[dict, np.ndarray, tuple[int, int]]:
    j = json.loads(p.read_text())
    K = j["K"]
    K = {"fx": float(K["fx"]), "fy": float(K["fy"]),
         "cx": float(K["cx"]), "cy": float(K["cy"])}
    T = np.asarray(j["T_cam2base"], dtype=np.float64).reshape(4, 4)
    W, H = j.get("image_size", [None, None])
    if W is None or H is None:
        W, H = 640, 480
    return K, T, (int(H), int(W))


def colorize_robot(mask: np.ndarray, bg: np.ndarray,
                   color=(40, 170, 255), alpha: float = 0.65) -> np.ndarray:
    out = bg.copy()
    if mask.any():
        m3 = mask[..., None]
        tint = np.asarray(color, dtype=np.float32)
        out = np.where(
            m3,
            (alpha * tint + (1 - alpha) * bg.astype(np.float32)).astype(np.uint8),
            out,
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pnp",        type=Path, required=True, help="per-episode pnp_*.json")
    ap.add_argument("--trajectory", type=Path, required=True, help="trajectory.npz with state or 14D joint_position")
    ap.add_argument("--images",     type=Path, required=True, help="frames dir or .mp4")
    ap.add_argument("--out",        type=Path, required=True, help="output .mp4")
    ap.add_argument("--xml",        type=Path, default=DEFAULT_XML)
    ap.add_argument("--base-body",  type=str,  default="ur5e/base")
    ap.add_argument("--geom-groups", type=int, nargs="+", default=[2])
    ap.add_argument("--side-by-side", action="store_true",
                    help="emit [image | render-mask | overlay] panel")
    ap.add_argument("--fps",   type=int,   default=5)
    ap.add_argument("--alpha", type=float, default=0.65)
    ap.add_argument("--limit", type=int,   default=0, help="max frames (0 = all)")
    args = ap.parse_args()

    K, T_cam2base, image_hw = load_pnp(args.pnp)
    H, W = image_hw
    print(f"[test] K fx={K['fx']:.2f} fy={K['fy']:.2f} cx={K['cx']:.2f} cy={K['cy']:.2f}  HxW={H}x{W}")
    print(f"[test] T_cam2base t={np.round(T_cam2base[:3, 3], 4).tolist()}")

    qpos, gripper = load_qpos_and_gripper(args.trajectory)
    frames = load_image_frames(args.images)
    if frames.shape[1:3] != (H, W):
        print(f"[test] WARN: image size {frames.shape[1:3]} != pnp image_size {(H, W)}; "
              f"using image-derived ({frames.shape[1]}x{frames.shape[2]}) so fovy stays consistent")
        H, W = frames.shape[1:3]

    n = min(len(qpos), len(frames))
    if args.limit:
        n = min(n, args.limit)
    print(f"[test] rendering {n} frames")

    renderer = MuJoCoUR5Renderer(
        xml_path=args.xml,
        verbose=True,
        geom_groups=tuple(args.geom_groups),
        base_body=args.base_body,
    )

    out_frames = []
    for i in range(n):
        mask = renderer.render(K=K, T_cam2base=T_cam2base,
                               qpos=qpos[i], gripper_closed=bool(gripper[i]),
                               image_hw=(H, W))
        overlay = colorize_robot(mask, frames[i], alpha=args.alpha)
        if args.side_by_side:
            mask_rgb = (np.stack([mask] * 3, axis=-1) * 255).astype(np.uint8)
            panel = np.concatenate([frames[i], mask_rgb, overlay], axis=1)
            out_frames.append(panel)
        else:
            out_frames.append(overlay)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.out, np.stack(out_frames), fps=args.fps)
    print(f"[test] wrote {args.out}")
    renderer.close()


if __name__ == "__main__":
    main()
