"""MuJoCo renderer for UR5e overlays — image-aligned, not viewpoint-aligned.

Mirrors tools/mujoco_google_robot_renderer.py one-to-one, swapping only the
robot-specific bits (XML, base body, joint convention, gripper handling). Use
this when you need a render that registers pixel-for-pixel with the real
camera (e.g. as the starting point for differentiable silhouette refinement).

Camera comes from PnP: T_cam2base (OpenCV convention) + intrinsics K. The
constant AugE viewpoint dict is NOT used here — that path renders from a
synthetic viewpoint that was never intended to match the original camera.
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


# AugE process_berkeley_ur5 gripper-closed qpos (last 8 of the 14D qpos).
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
# helpers shared with the Google Robot renderer (kept verbatim where possible)
# ---------------------------------------------------------------------------

def _body_T_world(mujoco, model, data, body_id: int) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    if body_id <= 0:
        return T
    T[:3, :3] = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
    T[:3, 3]  = np.asarray(data.xpos[body_id], dtype=np.float64)
    return T


def _camera_quat_wxyz_from_cv_cam2base(T_cam2base: np.ndarray) -> np.ndarray:
    # OpenCV cam: +x right, +y down, +z forward.
    # MuJoCo/GL cam: +x right, +y up,  -z forward.
    R_cv = np.asarray(T_cam2base, dtype=np.float64)[:3, :3]
    R_mj = R_cv @ np.diag([1.0, -1.0, -1.0])
    q_xyzw = R.from_matrix(R_mj).as_quat()
    return np.roll(q_xyzw, 1)


def _fovy_from_K(K: dict[str, float], H: int) -> float:
    fy = float(K["fy"])
    return float(np.degrees(2.0 * np.arctan(0.5 * H / fy)))


def _xml_with_camera(xml_path: Path, T_cam2world: np.ndarray, K: dict[str, float],
                     H: int, W: int) -> Path:
    del W
    root = ET.parse(xml_path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"MuJoCo XML has no worldbody: {xml_path}")
    cam_pos  = np.asarray(T_cam2world, dtype=np.float64)[:3, 3]
    cam_quat = _camera_quat_wxyz_from_cv_cam2base(T_cam2world)
    fovy     = _fovy_from_K(K, H)
    ET.SubElement(
        worldbody, "camera",
        {
            "name": "pnp_cam",
            "pos":  " ".join(f"{v:.9g}" for v in cam_pos),
            "quat": " ".join(f"{v:.9g}" for v in cam_quat),
            "fovy": f"{fovy:.9g}",
        },
    )
    # Keep the temp XML beside scene.xml so the relative <include> + assets/
    # paths in AugE's scene resolve correctly. Same trick the Google Robot
    # renderer uses — DO NOT move this to /tmp.
    tmp = tempfile.NamedTemporaryFile(
        prefix=".ur5_pnp_cam_",
        suffix=".xml",
        dir=str(xml_path.parent),
        delete=False,
    )
    tmp.close()
    ET.ElementTree(root).write(tmp.name)
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# renderer
# ---------------------------------------------------------------------------

class MuJoCoUR5Renderer:
    def __init__(
        self,
        xml_path: str | Path,
        verbose: bool = True,
        postprocess: str = "none",
        geom_groups: tuple[int, ...] = (2,),
        geom_name_include: str | None = None,
        geom_name_exclude: str | None = None,
        base_body: str | None = "ur5e/base",
    ):
        os.environ.setdefault("MUJOCO_GL", "egl")
        import mujoco  # type: ignore

        self.mujoco = mujoco
        self.xml_path = Path(xml_path).expanduser()
        if not self.xml_path.exists():
            raise FileNotFoundError(f"missing UR5 MuJoCo XML: {self.xml_path}")
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
        self.base_body = base_body
        self._robot_geom_ids: set[int] | None = None
        self._qpos_addrs: list[int] | None = None
        self._T_world_base = self._load_world_base_transform()

    # ----- body / geom plumbing (same logic as Google Robot renderer) -----

    def _body_names(self, model) -> list[str]:
        return [
            self.mujoco.mj_id2name(model, self.mujoco.mjtObj.mjOBJ_BODY, i) or ""
            for i in range(int(model.nbody))
        ]

    def _resolve_base_body(self, model, requested: str | None) -> int:
        if not requested:
            return 0
        names = self._body_names(model)
        if requested != "auto":
            if requested in names:
                return names.index(requested)
            raise ValueError(f"base body {requested!r} not found. Bodies: {names}")
        for target in ("ur5e/base", "ur5_base", "ur5e_base", "base_link", "base", "world"):
            if target in names:
                return names.index(target)
        for i, name in enumerate(names):
            if "base" in name.lower():
                return i
        return 0

    def _load_world_base_transform(self) -> np.ndarray:
        if not self.base_body:
            return np.eye(4, dtype=np.float64)
        model = self.mujoco.MjModel.from_xml_path(str(self.xml_path))
        data  = self.mujoco.MjData(model)
        self.mujoco.mj_forward(model, data)
        bid = self._resolve_base_body(model, self.base_body)
        T   = _body_T_world(self.mujoco, model, data, bid)
        if self.verbose:
            names = self._body_names(model)
            print(f"[ur5_render] base_body={names[bid] if bid < len(names) else bid} "
                  f"T_world_base_t={np.round(T[:3, 3], 5).tolist()}")
        return T

    def _geom_name(self, gid: int) -> str:
        assert self.model is not None
        return self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or ""

    def _allowed_geom_ids(self) -> set[int]:
        assert self.model is not None
        inc = re.compile(self.geom_name_include) if self.geom_name_include else None
        exc = re.compile(self.geom_name_exclude) if self.geom_name_exclude else None
        out: set[int] = set()
        for gid in range(int(self.model.ngeom)):
            g = int(self.model.geom_group[gid])
            if self.geom_groups and g not in self.geom_groups:
                continue
            n = self._geom_name(gid)
            if inc and not inc.search(n): continue
            if exc and exc.search(n):     continue
            out.add(gid)
        return out

    def _postprocess_mask(self, mask: np.ndarray, mode: str | None = None) -> np.ndarray:
        mode = self.postprocess if mode is None else mode
        if mode == "none":   return mask
        if mode == "flip_x": return cv2.flip(mask.astype(np.uint8), 1) > 0
        if mode == "flip_y": return cv2.flip(mask.astype(np.uint8), 0) > 0
        if mode == "rot180": return cv2.rotate(mask.astype(np.uint8), cv2.ROTATE_180) > 0
        raise ValueError(f"unknown postprocess: {mode}")

    # ----- per-call renderer (re)build -----

    def _ensure_renderer(self, K: dict[str, float], T_cam2base: np.ndarray,
                         image_hw: tuple[int, int]):
        H, W = image_hw
        key = (
            int(H), int(W),
            tuple(np.round(np.asarray(T_cam2base, dtype=np.float64).reshape(-1), 8)),
            round(float(K["fy"]), 8),
        )
        if key == self._cache_key:
            return
        if self.renderer is not None:
            self.renderer.close()
        if self._tmp_xml is not None:
            try: self._tmp_xml.unlink()
            except FileNotFoundError: pass

        T_cam2world  = self._T_world_base @ np.asarray(T_cam2base, dtype=np.float64)
        self._tmp_xml = _xml_with_camera(self.xml_path, T_cam2world, K, H, W)
        self.model    = self.mujoco.MjModel.from_xml_path(str(self._tmp_xml))
        self.data     = self.mujoco.MjData(self.model)
        self.renderer = self.mujoco.Renderer(self.model, height=H, width=W)
        self.scene_option = self.mujoco.MjvOption()
        self.scene_option.geomgroup[:] = 0
        for g in self.geom_groups:
            if 0 <= g < len(self.scene_option.geomgroup):
                self.scene_option.geomgroup[g] = 1
        self._robot_geom_ids = self._allowed_geom_ids()

        # Resolve AugE-style joint -> qpos addrs once per model rebuild.
        self._qpos_addrs = []
        for name in UR5_JOINT_NAMES:
            jid = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                self._qpos_addrs.append(int(self.model.jnt_qposadr[jid]))
        self._cache_key = key

        if self.verbose:
            preview = [self._geom_name(g) or f"geom_{g}"
                       for g in sorted(self._robot_geom_ids)[:8]]
            print(f"[ur5_render] loaded {self.xml_path} cam=pnp_cam {W}x{H} "
                  f"nq={self.model.nq} ngeom={self.model.ngeom} "
                  f"groups={self.geom_groups} sel={len(self._robot_geom_ids)} "
                  f"preview={preview}")

    def _segmentation_mask(self, seg: np.ndarray) -> np.ndarray | None:
        if not self._robot_geom_ids or seg.ndim != 3:
            return None
        ids = seg[..., 0]
        for cand in (np.isin(ids, list(self._robot_geom_ids)),
                     np.isin(ids - 1, list(self._robot_geom_ids))):
            area = int(cand.sum())
            if 0 < area < int(0.65 * cand.size):
                return cand
        return None

    # ----- public render API -----

    def _set_qpos_auge(self, state6_or_qpos14: np.ndarray, gripper_closed: bool):
        """Match AugE process_berkeley_ur5: 6 arm joints + 8 synthetic gripper."""
        assert self.model is not None and self.data is not None
        self.data.qpos[:] = 0.0
        q = np.asarray(state6_or_qpos14, dtype=np.float64).reshape(-1)
        if self._qpos_addrs and len(self._qpos_addrs) >= 6:
            for col in range(min(6, len(q))):
                self.data.qpos[self._qpos_addrs[col]] = float(q[col])
            if gripper_closed and len(self._qpos_addrs) >= 14:
                for col, val in enumerate(UR5_CLOSED_GRIPPER):
                    self.data.qpos[self._qpos_addrs[6 + col]] = float(val)
            if len(q) >= 14:
                # caller supplied a full 14D qpos — trust it over our default
                for col in range(6, min(14, len(q))):
                    self.data.qpos[self._qpos_addrs[col]] = float(q[col])

    def render(self, K: dict[str, float], T_cam2base: np.ndarray,
               qpos: np.ndarray, gripper_closed: bool = False,
               image_hw: tuple[int, int] | None = None,
               postprocess: str | None = None) -> np.ndarray:
        if image_hw is None:
            image_hw = (int(K["height"]), int(K["width"]))
        self._ensure_renderer(K, T_cam2base, image_hw)
        self._set_qpos_auge(qpos, gripper_closed)
        self.mujoco.mj_forward(self.model, self.data)

        try:
            self.renderer.update_scene(self.data, camera="pnp_cam",
                                       scene_option=self.scene_option)
            self.renderer.enable_segmentation_rendering()
            seg = self.renderer.render()
            self.renderer.disable_segmentation_rendering()
            mask = self._segmentation_mask(seg)
            if mask is not None:
                return self._postprocess_mask(mask, postprocess)
        except Exception:
            try: self.renderer.disable_segmentation_rendering()
            except Exception: pass

        self.renderer.update_scene(self.data, camera="pnp_cam",
                                   scene_option=self.scene_option)
        rgb = self.renderer.render()
        mask = np.any(rgb > 12, axis=2)
        if 0 < int(mask.sum()) < int(0.65 * mask.size):
            return self._postprocess_mask(mask, postprocess)

        try:
            self.renderer.update_scene(self.data, camera="pnp_cam",
                                       scene_option=self.scene_option)
            self.renderer.enable_depth_rendering()
            depth = self.renderer.render()
            self.renderer.disable_depth_rendering()
            mask = np.isfinite(depth) & (depth > 0.0) & (depth < 0.99)
            if 0 < int(mask.sum()) < int(0.65 * mask.size):
                return self._postprocess_mask(mask, postprocess)
        except Exception:
            try: self.renderer.disable_depth_rendering()
            except Exception: pass

        return np.zeros(image_hw, dtype=bool)

    def overlay(self, image: np.ndarray, K: dict[str, float], T_cam2base: np.ndarray,
                qpos: np.ndarray, gripper_closed: bool = False,
                color_bgr=(255, 170, 40), outline_bgr=(255, 225, 120),
                outline_px: int = 1, alpha: float = 0.60):
        mask = self.render(K=K, T_cam2base=T_cam2base, qpos=qpos,
                           gripper_closed=gripper_closed, image_hw=image.shape[:2])
        out = image.copy().astype(np.float32)
        if np.any(mask):
            out[mask] = (1.0 - alpha) * out[mask] + alpha * np.asarray(color_bgr, dtype=np.float32)
            if outline_px > 0:
                u8 = mask.astype(np.uint8) * 255
                edge = cv2.morphologyEx(
                    u8, cv2.MORPH_GRADIENT,
                    np.ones((2 * outline_px + 1, 2 * outline_px + 1), np.uint8),
                ) > 0
                out[edge] = np.asarray(outline_bgr, dtype=np.float32)
        return out.astype(np.uint8), mask

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
        if self._tmp_xml is not None:
            try: self._tmp_xml.unlink()
            except FileNotFoundError: pass
