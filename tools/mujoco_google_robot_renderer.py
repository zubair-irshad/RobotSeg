#!/usr/bin/env python3
"""MuJoCo renderer for Google Robot audit overlays.

This uses the same AugE / MuJoCo Menagerie XML as the IK exporter, avoiding the
lossy MJCF->URDF conversion path. It injects a fixed camera per PnP estimate and
renders the current MuJoCo qpos into an image-sized mask.
"""

from __future__ import annotations

import os
import re
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
                 postprocess: str = "none", geom_groups: tuple[int, ...] = (2,),
                 geom_name_include: str | None = None,
                 geom_name_exclude: str | None = None):
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
        self.geom_groups = tuple(int(g) for g in geom_groups)
        self.geom_name_include = geom_name_include
        self.geom_name_exclude = geom_name_exclude
        self._robot_geom_ids: set[int] | None = None

    def _geom_name(self, geom_id: int) -> str:
        assert self.model is not None
        name = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
        return name or ""

    def _allowed_geom_ids(self) -> set[int]:
        assert self.model is not None
        include_re = re.compile(self.geom_name_include) if self.geom_name_include else None
        exclude_re = re.compile(self.geom_name_exclude) if self.geom_name_exclude else None
        allowed: set[int] = set()
        for gid in range(int(self.model.ngeom)):
            group = int(self.model.geom_group[gid])
            if self.geom_groups and group not in self.geom_groups:
                continue
            name = self._geom_name(gid)
            if include_re is not None and not include_re.search(name):
                continue
            if exclude_re is not None and exclude_re.search(name):
                continue
            allowed.add(gid)
        return allowed

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
        # Other robot XMLs can override this with --mujoco_geom_groups.
        self.scene_option.geomgroup[:] = 0
        for group in self.geom_groups:
            if 0 <= group < len(self.scene_option.geomgroup):
                self.scene_option.geomgroup[group] = 1
        self._robot_geom_ids = self._allowed_geom_ids()
        self._cache_key = key
        if self.verbose:
            preview = []
            for gid in sorted(self._robot_geom_ids)[:8]:
                preview.append(self._geom_name(gid) or f"geom_{gid}")
            print(
                f"[mujoco_render] loaded {self.xml_path} camera=pnp_cam "
                f"size={W}x{H} nq={self.model.nq} ngeom={self.model.ngeom} "
                f"geom_groups={self.geom_groups} selected_geoms={len(self._robot_geom_ids)} "
                f"preview={preview}"
            )

    def _segmentation_mask(self, seg: np.ndarray) -> np.ndarray | None:
        if self._robot_geom_ids is None or not self._robot_geom_ids:
            return None
        if seg.ndim != 3:
            return None
        ids = seg[..., 0]
        # MuJoCo Renderer segmentation commonly returns geom/object ids in the
        # first channel. Depending on version, ids can be either 0-based with
        # -1 as background or 1-based with 0 as background. Try both and keep
        # the one with a plausible foreground area.
        candidates = [
            np.isin(ids, list(self._robot_geom_ids)),
            np.isin(ids - 1, list(self._robot_geom_ids)),
        ]
        for mask in candidates:
            area = int(mask.sum())
            if 0 < area < int(0.65 * mask.size):
                return mask
        return None

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

        try:
            self.renderer.update_scene(self.data, camera="pnp_cam", scene_option=self.scene_option)
            self.renderer.enable_segmentation_rendering()
            seg = self.renderer.render()
            self.renderer.disable_segmentation_rendering()
            mask = self._segmentation_mask(seg)
            if mask is not None:
                return self._postprocess_mask(mask, postprocess)
        except Exception:
            try:
                self.renderer.disable_segmentation_rendering()
            except Exception:
                pass

        # Fallback: render selected geom groups and threshold non-background
        # RGB. This can include same-group scene props, so segmentation above
        # is preferred whenever MuJoCo exposes usable geom ids.
        self.renderer.update_scene(self.data, camera="pnp_cam", scene_option=self.scene_option)
        rgb = self.renderer.render()
        mask = np.any(rgb > 12, axis=2)
        area = int(mask.sum())
        if 0 < area < int(0.65 * mask.size):
            return self._postprocess_mask(mask, postprocess)

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
