"""Render UR5e on top of berkeley_autolab_ur5 frames using the AugE-Toolkit
viewpoint verbatim. This is the *initial* (non-optimized) overlay used as the
starting point for differentiable refinement.

Pipeline mirrors AugE-Toolkit core/RobotReplay.py + processing_utils.process_berkeley_ur5:

  qpos        = [state[:6], 8 gripper joints]
                gripper joints are [1.0, 0.025, 0.80, -0.80, 1.0, 0.0252768, 0.80, -0.80]
                when state[13] > 0.5 else zeros.
  XML         = third_party/AugE-Toolkit/robot_xml/universal_robots_ur5e/scene.xml
  camera      = MjvCamera with lookat / distance / azimuth / elevation,
                model.vis.global_.fovy = camera_fov (deg, vertical).
  viewpoint   = config["berkeley_autolab_ur5"]["viewpoints"][0]
                (rotation_values has no entry for (ur5e, berkeley_autolab_ur5),
                 so AugE's rotate_cam_orientation / rotate_cam_lookat_about_z
                 / translate_position are identities here.)

Usage:
  python tools/render_ur5_auge_viewpoint.py \
      --trajectory data/.../episode_0000/trajectory.npz \
      --images     data/.../episode_0000/images/         \
      --out        /tmp/ur5_auge_overlay_0000.mp4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco                         # noqa: E402
import imageio.v3 as iio              # noqa: E402
import cv2                            # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
AUGE_XML  = REPO_ROOT / "third_party" / "AugE-Toolkit" / "robot_xml" / "universal_robots_ur5e" / "scene.xml"

# AugE viewpoint for berkeley_autolab_ur5 (config.py), pasted verbatim.
AUGE_VIEWPOINT = {
    "lookat":     [0.24398564, 0.23394822, 0.25454247],
    "distance":   0.36241041268887836,
    "azimuth":    140.5502567979282 - 180.0,   # AugE stores it pre-subtracted
    "elevation": -42.125,
    "camera_fov": 57.82240163683314,           # MuJoCo fovy (vertical, deg)
}

# AugE process_berkeley_ur5 gripper-closed qpos (last 8 dims of the 14D qpos).
UR5_CLOSED_GRIPPER = np.asarray(
    [1.0, 0.025, 0.80, -0.80, 1.0, 0.0252768, 0.80, -0.80], dtype=np.float32
)

UR5_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    "right_driver_joint", "right_coupler_joint",
    "right_spring_link_joint", "right_follower_joint",
    "left_driver_joint", "left_coupler_joint",
    "left_spring_link_joint", "left_follower_joint",
]


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def load_qpos_from_state(state: np.ndarray) -> np.ndarray:
    """Replicate AugE process_berkeley_ur5: 14D qpos per frame."""
    state = np.asarray(state, dtype=np.float32)
    qpos = np.zeros((state.shape[0], 14), dtype=np.float32)
    qpos[:, :6] = state[:, :6]
    closed = state[:, 13] > 0.5
    qpos[closed, 6:] = UR5_CLOSED_GRIPPER
    return qpos


def load_trajectory_qpos(traj_path: Path) -> np.ndarray:
    with np.load(traj_path, allow_pickle=True) as f:
        if "joint_position" in f.files and f["joint_position"].shape[-1] == 14:
            return np.asarray(f["joint_position"], dtype=np.float32)
        if "state" not in f.files:
            raise KeyError(f"{traj_path} has neither 14D joint_position nor state")
        return load_qpos_from_state(f["state"])


def load_image_frames(images_arg: Path) -> np.ndarray:
    """Accepts a directory of PNG/JPG frames or a video file."""
    p = images_arg
    if p.is_dir():
        files = sorted([q for q in p.iterdir()
                        if q.suffix.lower() in (".png", ".jpg", ".jpeg")])
        if not files:
            raise FileNotFoundError(f"no image frames under {p}")
        return np.stack([iio.imread(q) for q in files], axis=0)
    return np.asarray(iio.imread(p))


# ---------------------------------------------------------------------------
# renderer
# ---------------------------------------------------------------------------

class AugeUR5Renderer:
    """Offscreen MuJoCo renderer wired up exactly like AugE-Toolkit."""

    def __init__(self, width: int, height: int, viewpoint: dict = AUGE_VIEWPOINT,
                 xml_path: Path = AUGE_XML):
        if not xml_path.exists():
            raise FileNotFoundError(xml_path)

        # AugE loads relative to its repo root because XML uses relative includes.
        cwd_save = os.getcwd()
        os.chdir(xml_path.parents[2])      # third_party/AugE-Toolkit
        try:
            self.model = mujoco.MjModel.from_xml_path(
                str(xml_path.relative_to(xml_path.parents[2]))
            )
        finally:
            os.chdir(cwd_save)

        self.data     = mujoco.MjData(self.model)
        self.width    = width
        self.height   = height

        # Match AugE: set fovy *before* creating the Renderer (it's read on init).
        self.model.vis.global_.fovy = float(viewpoint["camera_fov"])

        self.renderer = mujoco.Renderer(self.model, width=width, height=height)

        # MjvCamera free-camera with AugE's lookat/distance/azimuth/elevation.
        self.cam            = mujoco.MjvCamera()
        self.cam.type       = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:]  = np.asarray(viewpoint["lookat"], dtype=np.float64)
        self.cam.distance   = float(viewpoint["distance"])
        self.cam.azimuth    = float(viewpoint["azimuth"])
        self.cam.elevation  = float(viewpoint["elevation"])

        # Resolve qpos addresses for the 14 AugE-named joints.
        self._qpos_addrs = []
        for name in UR5_JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"joint {name!r} missing from {xml_path}")
            self._qpos_addrs.append(int(self.model.jnt_qposadr[jid]))

    def render(self, qpos_14: np.ndarray) -> np.ndarray:
        self.data.qpos[:] = 0.0
        for col, addr in enumerate(self._qpos_addrs):
            if col < qpos_14.shape[0] and addr < self.model.nq:
                self.data.qpos[addr] = float(qpos_14[col])
        mujoco.mj_forward(self.model, self.data)

        self.renderer.update_scene(self.data, camera=self.cam)
        return self.renderer.render()       # uint8 HxWx3 RGB

    def render_with_depth(self, qpos_14: np.ndarray):
        rgb = self.render(qpos_14)
        self.renderer.enable_depth_rendering()
        try:
            self.renderer.update_scene(self.data, camera=self.cam)
            depth = self.renderer.render()
        finally:
            self.renderer.disable_depth_rendering()
        return rgb, depth


# ---------------------------------------------------------------------------
# overlay helpers
# ---------------------------------------------------------------------------

def robot_mask_from_depth(depth: np.ndarray) -> np.ndarray:
    """MuJoCo depth is far for empty pixels — robot is anything closer."""
    finite = np.isfinite(depth)
    if not finite.any():
        return np.zeros_like(depth, dtype=bool)
    far = depth[finite].max()
    return finite & (depth < far - 1e-4)


def overlay_robot(bg_rgb: np.ndarray, rob_rgb: np.ndarray,
                  mask: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    out = bg_rgb.copy()
    m3 = mask[..., None]
    out = np.where(m3, (alpha * rob_rgb + (1 - alpha) * bg_rgb).astype(np.uint8), out)
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", type=Path, required=True,
                    help="path to trajectory.npz (must contain 'state' or 14D 'joint_position')")
    ap.add_argument("--images", type=Path, required=True,
                    help="directory of frames or an .mp4 of the source video")
    ap.add_argument("--out",  type=Path, required=True,
                    help="output mp4 path for the overlay video")
    ap.add_argument("--side-by-side", action="store_true",
                    help="write [image | render | overlay] panel instead of plain overlay")
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.65)
    args = ap.parse_args()

    qpos    = load_trajectory_qpos(args.trajectory)
    frames  = load_image_frames(args.images)
    n       = min(len(qpos), len(frames))
    if n == 0:
        raise RuntimeError("no overlapping frames between trajectory and images")
    H, W    = frames.shape[1:3]
    print(f"[render] {n} frames at {W}x{H}, qpos={qpos.shape}")

    renderer = AugeUR5Renderer(width=W, height=H)
    out_frames = []
    for i in range(n):
        rob, depth = renderer.render_with_depth(qpos[i])
        mask       = robot_mask_from_depth(depth)
        ov         = overlay_robot(frames[i], rob, mask, alpha=args.alpha)
        if args.side_by_side:
            panel = np.concatenate([frames[i], rob, ov], axis=1)
            out_frames.append(panel)
        else:
            out_frames.append(ov)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.out, np.stack(out_frames), fps=args.fps)
    print(f"[render] wrote {args.out}")


if __name__ == "__main__":
    main()
