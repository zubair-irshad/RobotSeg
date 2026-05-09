#!/usr/bin/env python3
"""MuJoCo renderer for Google Robot audit overlays.

This uses the same AugE / MuJoCo Menagerie XML as the IK exporter, avoiding the
lossy MJCF->URDF conversion path. It injects a fixed camera per PnP estimate and
renders the current MuJoCo qpos into an image-sized mask.
"""

from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


def _invert_se3(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ T[:3, 3]
    return out


def _camera_quat_wxyz_from_cv_cam2base(T_cam2base: np.ndarray) -> np.ndarray:
    # OpenCV camera: +x right, +y down, +z forward.
    # MuJoCo/OpenGL camera: +x right, +y up, -z forward.
    R_cv_cam2base = np.asarray(T_cam2base, dtype=np.float64)[:3, :3]
    R_cv_to_mj = np.diag([1.0, -1.0, -1.0])
    R_mj_cam2base = R_cv_cam2base @ R_cv_to_mj
    q_xyzw = R.from_matrix(R_mj_cam2base).as_quat()
    return np.roll(q_xyzw, 1)


def _fovy_from_K(K: dict[str, float], H: int) -> float:
    fy = float(K["fy"])
    return float(np.degrees(2.0 * np.arctan(0.5 * H / fy)))


def _xml_with_camera(xml_path: Path, T_cam2base: np.ndarray, K: dict[str, float],
                     H: int, W: int) -> Path:
    del W
    root = ET.parse(xml_path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"MuJoCo XML has no worldbody: {xml_path}")
    cam_pos = np.asarray(T_cam2base, dtype=np.float64)[:3, 3]
    cam_quat = _camera_quat_wxyz_from_cv_cam2base(T_cam2base)
    fovy = _fovy_from_K(K, H)
    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": "pnp_cam",
            "pos": " ".join(f"{v:.9g}" for v in cam_pos),
            "quat": " ".join(f"{v:.9g}" for v in cam_quat),
            "fovy": f"{fovy:.9g}",
        },
    )
    # Keep the temporary XML beside scene.xml. AugE's scene.xml includes
    # robot.xml with a relative path, and robot.xml resolves assets/ relative
    # to that same directory. A /tmp XML would break those includes.
    tmp = tempfile.NamedTemporaryFile(
        prefix=".google_robot_pnp_cam_",
        suffix=".xml",
        dir=str(xml_path.parent),
        delete=False,
    )
    tmp.close()
    ET.ElementTree(root).write(tmp.name)
    return Path(tmp.name)


class MuJoCoGoogleRobotRenderer:
    def __init__(self, xml_path: str | Path, verbose: bool = True,
                 postprocess: str = "none"):
        os.environ.setdefault("MUJOCO_GL", "egl")
        import mujoco  # type: ignore

        self.mujoco = mujoco
        self.xml_path = Path(xml_path).expanduser()
        if not self.xml_path.exists():
            raise FileNotFoundError(f"missing Google Robot MuJoCo XML: {self.xml_path}")
        self.verbose = verbose
        self._cache_key = None
        self._tmp_xml: Path | None = None
        self.model = None
        self.data = None
        self.renderer = None
        self.postprocess = postprocess

    def _postprocess_mask(self, mask: np.ndarray, mode: str | None = None) -> np.ndarray:
        mode = self.postprocess if mode is None else mode
        if mode == "none":
            return mask
        if mode == "auge":
            # Matches AugE core.utils.get_overlay_img:
            #   cv2.flip(mask, 1), then cv2.rotate(..., ROTATE_180)
            # Only use this with low-level mjr_readPixels-style buffers.
            # mujoco.Renderer.render() already returns top-left image order.
            return cv2.rotate(cv2.flip(mask.astype(np.uint8), 1), cv2.ROTATE_180) > 0
        if mode == "flip_x":
            return cv2.flip(mask.astype(np.uint8), 1) > 0
        if mode == "flip_y":
            return cv2.flip(mask.astype(np.uint8), 0) > 0
        if mode == "rot180":
            return cv2.rotate(mask.astype(np.uint8), cv2.ROTATE_180) > 0
        raise ValueError(f"unknown MuJoCo postprocess: {mode}")

    def _ensure_renderer(self, K: dict[str, float], T_cam2base: np.ndarray,
                         image_hw: tuple[int, int]):
        H, W = image_hw
        key = (
            int(H),
            int(W),
            tuple(np.round(np.asarray(T_cam2base, dtype=np.float64).reshape(-1), 8)),
            round(float(K["fy"]), 8),
        )
        if key == self._cache_key:
            return
        if self.renderer is not None:
            self.renderer.close()
        if self._tmp_xml is not None:
            try:
                self._tmp_xml.unlink()
            except FileNotFoundError:
                pass
        self._tmp_xml = _xml_with_camera(self.xml_path, T_cam2base, K, H, W)
        self.model = self.mujoco.MjModel.from_xml_path(str(self._tmp_xml))
        self.data = self.mujoco.MjData(self.model)
        self.renderer = self.mujoco.Renderer(self.model, height=H, width=W)
        self.scene_option = self.mujoco.MjvOption()
        # Google Robot visual geoms are group 2 in the MuJoCo Menagerie XML.
        # Rendering only that group avoids floor/world geoms entering the mask.
        self.scene_option.geomgroup[:] = 0
        if len(self.scene_option.geomgroup) > 2:
            self.scene_option.geomgroup[2] = 1
        self._cache_key = key
        if self.verbose:
            print(f"[mujoco_render] loaded {self.xml_path} camera=pnp_cam size={W}x{H}")

    def render(self, K: dict[str, float], T_cam2base: np.ndarray,
               qpos: np.ndarray, gripper_position: float = 0.0,
               image_hw: tuple[int, int] | None = None,
               postprocess: str | None = None) -> np.ndarray:
        if image_hw is None:
            image_hw = (int(K["height"]), int(K["width"]))
        self._ensure_renderer(K, T_cam2base, image_hw)
        assert self.model is not None and self.data is not None and self.renderer is not None

        q = np.asarray(qpos, dtype=np.float64).reshape(-1)
        n = min(len(q), self.model.nq)
        self.data.qpos[:n] = q[:n]
        if len(q) < self.model.nq and self.model.nq >= 9:
            g = float(np.clip(gripper_position, 0.0, 1.0))
            self.data.qpos[7:9] = 0.333 + g * (1.0 - 0.333)
        self.mujoco.mj_forward(self.model, self.data)

        # Primary path: render only the robot visual geom group into an empty
        # scene and threshold non-background RGB. This is more reliable across
        # MuJoCo versions than depth, whose background value can appear finite.
        self.renderer.update_scene(self.data, camera="pnp_cam", scene_option=self.scene_option)
        rgb = self.renderer.render()
        mask = np.any(rgb > 12, axis=2)
        area = int(mask.sum())
        if 0 < area < int(0.65 * mask.size):
            return self._postprocess_mask(mask, postprocess)

        try:
            self.renderer.update_scene(self.data, camera="pnp_cam", scene_option=self.scene_option)
            self.renderer.enable_segmentation_rendering()
            seg = self.renderer.render()
            self.renderer.disable_segmentation_rendering()
            if seg.ndim == 3:
                if np.issubdtype(seg.dtype, np.signedinteger):
                    mask = seg[..., 0] >= 0
                else:
                    mask = seg[..., 0] > 0
            else:
                mask = seg >= 0 if np.issubdtype(seg.dtype, np.signedinteger) else seg > 0
            # If segmentation semantics differ across MuJoCo versions and mark
            # most of the frame as foreground, fall back to depth.
            if 0 < int(mask.sum()) < int(0.65 * mask.size):
                return self._postprocess_mask(mask, postprocess)
        except Exception:
            try:
                self.renderer.disable_segmentation_rendering()
            except Exception:
                pass

        try:
            self.renderer.update_scene(self.data, camera="pnp_cam", scene_option=self.scene_option)
            self.renderer.enable_depth_rendering()
            depth = self.renderer.render()
            self.renderer.disable_depth_rendering()
            mask = np.isfinite(depth) & (depth > 0.0) & (depth < 0.99)
            if 0 < int(mask.sum()) < int(0.65 * mask.size):
                return self._postprocess_mask(mask, postprocess)
        except Exception:
            try:
                self.renderer.disable_depth_rendering()
            except Exception:
                pass

        return np.zeros(image_hw, dtype=bool)

    def overlay(self, image: np.ndarray, K: dict[str, float], T_cam2base: np.ndarray,
                joint_positions: np.ndarray, gripper_position: float = 0.0,
                color_bgr=(255, 170, 40), outline_bgr=(255, 225, 120),
                outline_px: int = 1, alpha: float = 0.60):
        mask = self.render(
            K=K,
            T_cam2base=T_cam2base,
            qpos=joint_positions,
            gripper_position=gripper_position,
            image_hw=image.shape[:2],
        )
        out = image.copy().astype(np.float32)
        if np.any(mask):
            out[mask] = (1.0 - alpha) * out[mask] + alpha * np.asarray(color_bgr, dtype=np.float32)
            if outline_px > 0:
                u8 = mask.astype(np.uint8) * 255
                edge = cv2.morphologyEx(
                    u8,
                    cv2.MORPH_GRADIENT,
                    np.ones((2 * outline_px + 1, 2 * outline_px + 1), np.uint8),
                ) > 0
                out[edge] = np.asarray(outline_bgr, dtype=np.float32)
        return out.astype(np.uint8), mask

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
        if self._tmp_xml is not None:
            try:
                self._tmp_xml.unlink()
            except FileNotFoundError:
                pass
