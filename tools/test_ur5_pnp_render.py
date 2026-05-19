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
import math
from pathlib import Path

import numpy as np
import imageio.v3 as iio

from mujoco_ur5_renderer import MuJoCoUR5Renderer, UR5_CLOSED_GRIPPER  # noqa: E402


# AugE config["berkeley_autolab_ur5"]["viewpoints"][0], pasted verbatim.
AUGE_VIEWPOINT_BERKELEY_UR5 = {
    "lookat":     [0.24398564, 0.23394822, 0.25454247],
    "distance":   0.36241041268887836,
    "azimuth":    140.5502567979282 - 180.0,
    "elevation": -42.125,
    "camera_fov": 57.82240163683314,
}


def auge_viewpoint_to_pnp_inputs(viewpoint: dict, H: int, W: int,
                                 ) -> tuple[dict, np.ndarray]:
    """Convert an AugE MjvCamera-style viewpoint to (K_dict, T_cam2base_OpenCV).

    Returns the same shapes the MuJoCoUR5Renderer expects from PnP, so the
    renderer code path is identical for both inputs. Assumes the robot's base
    body sits at the world origin (true for AugE's UR5e scene.xml — confirmed
    by renderer's T_world_base_t=[0,0,0]).
    """
    az = math.radians(float(viewpoint["azimuth"]))
    el = math.radians(float(viewpoint["elevation"]))
    d  = float(viewpoint["distance"])
    lookat = np.asarray(viewpoint["lookat"], dtype=np.float64)

    # MuJoCo free-camera convention (engine_vis_init / mjv_moveCamera):
    #   cam_pos = lookat + distance * (cos(az)*cos(el), sin(az)*cos(el), sin(el))
    #   forward = -(cos(az)*cos(el), sin(az)*cos(el), sin(el))   # what the camera looks at
    backward = np.array([math.cos(el) * math.cos(az),
                         math.cos(el) * math.sin(az),
                         math.sin(el)], dtype=np.float64)
    cam_pos = lookat + d * backward
    forward = -backward

    # MuJoCo/GL camera local axes (right, up, -forward) expressed in world.
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up); right /= np.linalg.norm(right)
    up    = np.cross(right, forward);    up    /= np.linalg.norm(up)
    R_cam2world_mj = np.column_stack([right, up, -forward])

    # Renderer expects OpenCV cam2base; it re-applies diag(1,-1,-1) to map
    # back to MuJoCo. So pre-bake the inverse here.
    R_cam2world_cv = R_cam2world_mj @ np.diag([1.0, -1.0, -1.0])
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_cam2world_cv
    T[:3, 3]  = cam_pos

    # MuJoCo uses vertical FOV with square pixels; recover K from fovy.
    fovy = float(viewpoint["camera_fov"])
    fy = 0.5 * H / math.tan(math.radians(fovy) * 0.5)
    K = {"fx": float(fy), "fy": float(fy),
         "cx": float(W) / 2.0, "cy": float(H) / 2.0}
    return K, T


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
    ap.add_argument("--also-auge", action="store_true",
                    help="also render the AugE viewpoint and stack it underneath the PnP panel")
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

    K_auge = T_auge = None
    if args.also_auge:
        K_auge, T_auge = auge_viewpoint_to_pnp_inputs(
            AUGE_VIEWPOINT_BERKELEY_UR5, H=H, W=W,
        )
        print(f"[test] AugE K fx=fy={K_auge['fx']:.2f} cx={K_auge['cx']:.1f} cy={K_auge['cy']:.1f}")
        print(f"[test] AugE T_cam2base t={np.round(T_auge[:3, 3], 4).tolist()}")

    def _render_panel(K_in, T_in, bg, qp, gc, tag: str) -> np.ndarray:
        mask = renderer.render(K=K_in, T_cam2base=T_in,
                               qpos=qp, gripper_closed=gc, image_hw=(H, W))
        overlay = colorize_robot(mask, bg, alpha=args.alpha)
        if args.side_by_side:
            mask_rgb = (np.stack([mask] * 3, axis=-1) * 255).astype(np.uint8)
            row = np.concatenate([bg, mask_rgb, overlay], axis=1)
        else:
            row = overlay
        # tiny label burn-in (top-left)
        import cv2  # local import to avoid hard dep elsewhere
        cv2.putText(row, tag, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return row

    out_frames = []
    for i in range(n):
        pnp_row = _render_panel(K, T_cam2base, frames[i], qpos[i],
                                bool(gripper[i]), "PnP")
        if args.also_auge:
            auge_row = _render_panel(K_auge, T_auge, frames[i], qpos[i],
                                     bool(gripper[i]), "AugE viewpoint")
            out_frames.append(np.concatenate([pnp_row, auge_row], axis=0))
        else:
            out_frames.append(pnp_row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    stacked = np.stack(out_frames)
    wrote = False
    for kwargs in (
        {"fps": args.fps, "codec": "libx264", "pixel_format": "yuv420p"},
        {"fps": args.fps, "codec": "mpeg4"},
        {"fps": args.fps, "plugin": "FFMPEG"},
    ):
        try:
            iio.imwrite(args.out, stacked, **kwargs)
            print(f"[test] wrote {args.out} with {kwargs}")
            wrote = True
            break
        except Exception as e:
            print(f"[test] writer {kwargs} failed: {e}")
    if not wrote:
        # last resort: dump per-frame PNGs next to the requested path
        dump_dir = args.out.with_suffix("")
        dump_dir.mkdir(parents=True, exist_ok=True)
        for i, fr in enumerate(stacked):
            iio.imwrite(dump_dir / f"frame_{i:04d}.png", fr)
        print(f"[test] no video writer worked; wrote PNG frames to {dump_dir}/")
    renderer.close()


if __name__ == "__main__":
    main()
