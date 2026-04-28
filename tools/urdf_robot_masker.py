"""URDF-based robot silhouette rasterizer.

Loads a URDF via yourdfpy, applies joint angles, projects the visual
mesh into a pinhole camera, and rasterizes the silhouette via
cv2.fillPoly. Useful for visually validating an estimated cam2base by
overlaying the rendered robot silhouette on the original image.

Adapted from the user's snippet — designed for Franka + Robotiq 2f-85 but
generalizable: pass any URDF whose first 7 actuated joints correspond to
the arm. Override _URDF_ARM_JOINT_NAMES if you use a different robot.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np


# Default arm joint names (Franka panda). Override at the call-site if you
# use a different robot.
_URDF_ARM_JOINT_NAMES = [
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7",
]


def invert_se3(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def project(points_cam: np.ndarray, K: dict) -> np.ndarray:
    """points_cam: (N,3) in camera frame. Returns (N,2) pixel coords."""
    z = np.maximum(points_cam[:, 2], 1e-6)
    x = points_cam[:, 0] / z
    y = points_cam[:, 1] / z
    u = K["fx"] * x + K["cx"]
    v = K["fy"] * y + K["cy"]
    return np.stack([u, v], axis=1)


class URDFRobotMasker:
    """Render a URDF-defined robot's silhouette from a given camera pose.

    Parameters
    ----------
    urdf_path : path to a URDF that yourdfpy can load.
    mesh_dir  : optional override for the URDF's mesh search root.
    downsample: rasterize at 1/s resolution then nearest-upsample (s>=1).
    dilate_px : grow the mask by N pixels after rasterization.
    arm_joint_names : ordered list mapping joint_positions[i] -> URDF joint name.
    gripper_joint_name : URDF joint that drives the gripper open/close.
    gripper_open_rad / gripper_closed_rad : URDF joint range corresponding to
        gripper_position 0 (open) and 1 (closed).
    """

    def __init__(self, urdf_path: str | Path,
                 mesh_dir: str | Path | None = None,
                 downsample: int = 2,
                 dilate_px: int = 3,
                 arm_joint_names: list[str] | None = None,
                 gripper_joint_name: str = "finger_joint",
                 gripper_open_rad: float = 0.0,
                 gripper_closed_rad: float = 0.7,
                 verbose: bool = True):
        import yourdfpy
        kwargs = dict(build_scene_graph=True, load_meshes=True,
                      build_collision_scene_graph=False,
                      load_collision_meshes=False)
        if mesh_dir is not None:
            mesh_dir = str(mesh_dir)
            kwargs["mesh_dir"] = mesh_dir
            # Resolve `package://<pkg>/...` URIs to <mesh_dir>/... so we
            # don't need a working ROS install to load community URDFs
            # (e.g. ros-industrial's universal_robot, automaticaddison's
            # ur_robotiq).
            def _filename_handler(fname: str) -> str:
                if fname.startswith("package://"):
                    rest = fname[len("package://"):]
                    parts = rest.split("/", 1)
                    if len(parts) == 2:
                        # Try a few resolution strategies in order:
                        candidates = [
                            os.path.join(mesh_dir, parts[1]),               # mesh_dir/<after-pkg>
                            os.path.join(mesh_dir, rest),                   # mesh_dir/<pkg>/<...>
                            os.path.join(os.path.dirname(mesh_dir), rest),  # parent/<pkg>/<...>
                        ]
                        for c in candidates:
                            if os.path.exists(c):
                                return c
                        return candidates[0]
                return fname
            kwargs["filename_handler"] = _filename_handler
        self.robot = yourdfpy.URDF.load(str(urdf_path), **kwargs)
        self.downsample = max(1, int(downsample))
        self.dilate_px = int(dilate_px)
        self.arm_joint_names = arm_joint_names or _URDF_ARM_JOINT_NAMES
        self.gripper_joint_name = gripper_joint_name
        self._grip_span = (float(gripper_open_rad), float(gripper_closed_rad))
        if verbose:
            print(f"[URDFRobotMasker] actuated joints: "
                  f"{list(self.robot.actuated_joint_names)}")
            print(f"[URDFRobotMasker] base frame: {self.robot.base_link}")

    def _apply_cfg(self, q_arm: np.ndarray, gripper_position: float) -> None:
        cfg: dict[str, float] = {}
        for i, name in enumerate(self.arm_joint_names):
            if i >= len(q_arm):
                break
            if name in self.robot.joint_map:
                cfg[name] = float(q_arm[i])
        if (self.gripper_joint_name and
                self.gripper_joint_name in self.robot.joint_map):
            g = float(np.clip(gripper_position, 0.0, 1.0))
            lo, hi = self._grip_span
            cfg[self.gripper_joint_name] = lo + g * (hi - lo)
        if cfg:
            self.robot.update_cfg(cfg)

    def render(self, K: dict, T_cam2base: np.ndarray,
               joint_positions: np.ndarray,
               gripper_position: float = 0.0,
               image_hw: tuple[int, int] | None = None) -> np.ndarray:
        H = int(image_hw[0] if image_hw is not None else K["height"])
        W = int(image_hw[1] if image_hw is not None else K["width"])
        s = self.downsample
        Hs, Ws = H // s, W // s
        Ks = {"fx": K["fx"] / s, "fy": K["fy"] / s,
              "cx": K["cx"] / s, "cy": K["cy"] / s}

        q_arm = np.asarray(joint_positions, dtype=float).flatten()
        self._apply_cfg(q_arm, gripper_position)

        combined = self.robot.scene.dump(concatenate=True)
        verts = np.asarray(combined.vertices, dtype=float)
        faces = np.asarray(combined.faces, dtype=np.int32)
        if len(verts) == 0 or len(faces) == 0:
            return np.zeros((H, W), dtype=bool)

        T_base2cam = invert_se3(np.asarray(T_cam2base, dtype=float))
        vh = np.hstack([verts, np.ones((len(verts), 1))])
        vc = (T_base2cam @ vh.T).T[:, :3]
        z = vc[:, 2]
        proj = project(vc, Ks)

        keep = ((z[faces[:, 0]] > 0.05)
                & (z[faces[:, 1]] > 0.05)
                & (z[faces[:, 2]] > 0.05))
        tri = faces[keep]

        mask = np.zeros((Hs, Ws), dtype=np.uint8)
        if len(tri) > 0:
            poly = proj[:, :2][tri].astype(np.int32)
            cv2.fillPoly(mask, poly, color=1)

        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        if self.dilate_px > 0:
            k = 2 * self.dilate_px + 1
            mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
        return mask.astype(bool)
