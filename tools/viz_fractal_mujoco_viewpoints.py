#!/usr/bin/env python3
"""Render Google Robot from AugE Fractal free-camera viewpoints.

This is a renderer sanity check that does not use the PnP extrinsic. It sets the
MuJoCo free camera with the Fractal viewpoints used by OXE-AugE, renders the
current Google Robot qpos, and scores each rendered silhouette against the saved
RobotSeg body mask.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np


FRACTAL_VIEWPOINTS = [
    {"lookat": [0.705072111870345, 0.013975478534784026, 0.6680251974054469], "distance": 0.7979868858827744, "azimuth": -2.4443359375, "elevation": -40.873046875, "camera_fov": 57},
    {"lookat": [0.7048949088644558, 0.02749863352837792, 0.6741001984235468], "distance": 0.792821361526563, "azimuth": 0.587890625, "elevation": -39.203125, "camera_fov": 57},
    {"lookat": [0.7052842721455245, -0.016368015238348046, 0.6339465158322075], "distance": 0.7954041237046687, "azimuth": -1.3896484375, "elevation": -42.630859375, "camera_fov": 57},
    {"lookat": [0.7024345226082027, -0.0028704309768615534, 0.6727576453286863], "distance": 0.7407394917247053, "azimuth": -1.51904296875, "elevation": -39.7607421875, "camera_fov": 57},
    {"lookat": [0.7053134200731075, 0.0005384909683360795, 0.7008562941856086], "distance": 0.7414789834494107, "azimuth": -2.7958984375, "elevation": -39.818359375, "camera_fov": 57},
    {"lookat": [0.70533450282253, 0.003704349859609352, 0.5796376127928865], "distance": 0.8599994388268911, "azimuth": -0.113037109375, "elevation": -45.603515625, "camera_fov": 57},
    {"lookat": [0.6997674313001118, -0.020530853148594195, 0.6614478955343539], "distance": 0.7841997517837511, "azimuth": -9.605224609375, "elevation": -41.736328125, "camera_fov": 57},
    {"lookat": [0.7002655265592601, -0.010816362621324512, 0.5573805217762815], "distance": 0.85, "azimuth": 2.109375, "elevation": -49.21875, "camera_fov": 57},
    {"lookat": [0.6979446018967236, -0.005879223835473823, 0.6729901255903193], "distance": 0.7799865335834213, "azimuth": -1.167724609375, "elevation": -43.494140625, "camera_fov": 57},
    {"lookat": [0.69789022826663, -0.0025710913446387344, 0.6399018883029551], "distance": 0.7799865335834213, "azimuth": 0.941650390625, "elevation": -44.197265625, "camera_fov": 57},
    {"lookat": [0.6979308146711577, -0.007549449747350338, 0.5637507722956394], "distance": 0.7799865335834213, "azimuth": -0.464599609375, "elevation": -47.009765625, "camera_fov": 57},
    {"lookat": [0.698585150958781, 0.025545750606431557, 0.6878787317455327], "distance": 0.7799865335834213, "azimuth": -1.167724609375, "elevation": -43.494140625, "camera_fov": 57},
    {"lookat": [0.6977380589887675, 0.012347976290985843, 0.7292335562647291], "distance": 0.7799865335834213, "azimuth": -5.034912109375, "elevation": -41.736328125, "camera_fov": 57},
    {"lookat": [0.6981461138647066, 0.004006839260568541, 0.6030605443963426], "distance": 0.8326134820570754, "azimuth": -1.167724609375, "elevation": -43.494140625, "camera_fov": 57},
    {"lookat": [0.6986173140485316, -0.02036985412694068, 0.6962989431759237], "distance": 0.7841997517837511, "azimuth": -1.870849609375, "elevation": -38.923828125, "camera_fov": 57},
    {"lookat": [0.6984640079625198, -0.02172122539325369, 0.6912207821428602], "distance": 0.7624446283289074, "azimuth": -1.519287109375, "elevation": -39.978515625, "camera_fov": 57},
    {"lookat": [0.6472670402681677, -0.0326971252495982, 0.9442527585861564], "distance": 0.40898720590602416, "azimuth": -10.308349609375, "elevation": -72.125, "camera_fov": 57},
    {"lookat": [0.6558801257686773, -0.001365699335457367, 0.6433031522284819], "distance": 0.8144390731619926, "azimuth": -15.230224609375, "elevation": -60.017578125, "camera_fov": 57},
    {"lookat": [0.6608054839958618, 0.016701091684091805, 0.7963179328599144], "distance": 0.8082898446741533, "azimuth": -21.558349609375, "elevation": -49.119140625, "camera_fov": 57},
    {"lookat": [0.6436299634139799, -0.042107918018819, 0.7080959925655009], "distance": 0.6828806963949621, "azimuth": -8.550537109375, "elevation": -42.791015625, "camera_fov": 57},
]


def _read_mask(path: Path | None, shape_hw: tuple[int, int]) -> np.ndarray:
    if path is None or not path.exists():
        return np.zeros(shape_hw, dtype=bool)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros(shape_hw, dtype=bool)
    if img.shape[:2] != shape_hw:
        img = cv2.resize(img, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return img > 127


def _mask_path(ep_seg: Path, root: str, stem: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg"):
        path = ep_seg / root / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def _load_viewpoints(path: Path | None) -> list[dict]:
    if path is None:
        return FRACTAL_VIEWPOINTS
    data = json.loads(path.expanduser().read_text())
    if isinstance(data, dict):
        for key in ("viewpoints", "bridge_viewpoints", "fractal20220817_data", "berkeley_autolab_ur5"):
            if key in data:
                data = data[key]
                if isinstance(data, dict) and "viewpoints" in data:
                    data = data["viewpoints"]
                break
    if not isinstance(data, list):
        raise ValueError(f"viewpoints JSON must be a list or contain a viewpoints list: {path}")
    out = []
    for item in data:
        v = dict(item)
        if "camera_fov" not in v and "fovy" in v:
            v["camera_fov"] = v["fovy"]
        out.append(v)
    return out


def _draw_header(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _overlay(image: np.ndarray, mask: np.ndarray, color_bgr=(255, 170, 40)) -> np.ndarray:
    out = image.copy().astype(np.float32)
    if np.any(mask):
        out[mask] = 0.40 * out[mask] + 0.60 * np.asarray(color_bgr, dtype=np.float32)
        edge = cv2.morphologyEx(
            mask.astype(np.uint8) * 255,
            cv2.MORPH_GRADIENT,
            np.ones((3, 3), np.uint8),
        ) > 0
        out[edge] = np.asarray((255, 225, 120), dtype=np.float32)
    return out.astype(np.uint8)


def _metrics(target: np.ndarray, render: np.ndarray) -> dict[str, float]:
    inter = float(np.logical_and(target, render).sum())
    union = float(np.logical_or(target, render).sum())
    target_area = float(target.sum())
    render_area = float(render.sum())
    return {
        "iou": inter / union if union > 0 else 0.0,
        "target_coverage": inter / target_area if target_area > 0 else 0.0,
        "render_precision": inter / render_area if render_area > 0 else 0.0,
        "target_area": target_area,
        "render_area": render_area,
    }


def _parse_groups(text: str) -> tuple[int, ...]:
    vals = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            vals.append(int(part))
    return tuple(vals) or (2,)


class FreeCameraGoogleRobotRenderer:
    def __init__(self, xml_path: Path, height: int, width: int,
                 geom_groups: tuple[int, ...] = (2,),
                 geom_name_include: str | None = None,
                 geom_name_exclude: str | None = None,
                 base_body: str | None = None):
        os.environ.setdefault("MUJOCO_GL", "egl")
        import mujoco  # type: ignore

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.scene_option = mujoco.MjvOption()
        self.scene_option.geomgroup[:] = 0
        self.geom_groups = tuple(int(g) for g in geom_groups)
        for group in self.geom_groups:
            if 0 <= group < len(self.scene_option.geomgroup):
                self.scene_option.geomgroup[group] = 1
        self.geom_name_include = geom_name_include
        self.geom_name_exclude = geom_name_exclude
        self._robot_geom_ids = self._allowed_geom_ids()
        self.base_body = base_body
        self._base_body_id = self._resolve_base_body(base_body)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE

    def _body_names(self) -> list[str]:
        return [
            self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_BODY, i) or ""
            for i in range(int(self.model.nbody))
        ]

    def _resolve_base_body(self, requested: str | None) -> int:
        if requested is None or requested == "":
            return 0
        names = self._body_names()
        if requested != "auto":
            if requested in names:
                return names.index(requested)
            raise ValueError(f"base body {requested!r} not found. Bodies: {names}")
        for target in ("base", "base_link", "ur5_base", "ur5e_base", "world"):
            if target in names:
                return names.index(target)
        for i, name in enumerate(names):
            if "base" in name.lower():
                return i
        return 0

    def _T_world_base(self) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        if self._base_body_id <= 0:
            return T
        T[:3, :3] = np.asarray(self.data.xmat[self._base_body_id], dtype=np.float64).reshape(3, 3)
        T[:3, 3] = np.asarray(self.data.xpos[self._base_body_id], dtype=np.float64)
        return T

    def _geom_name(self, geom_id: int) -> str:
        name = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
        return name or ""

    def _allowed_geom_ids(self) -> set[int]:
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

    def _segmentation_mask(self, seg: np.ndarray) -> np.ndarray | None:
        if seg.ndim != 3 or not self._robot_geom_ids:
            return None
        ids = seg[..., 0]
        for mask in (
            np.isin(ids, list(self._robot_geom_ids)),
            np.isin(ids - 1, list(self._robot_geom_ids)),
        ):
            area = int(mask.sum())
            if 0 < area < int(0.75 * mask.size):
                return mask
        return None

    def close(self) -> None:
        self.renderer.close()

    def _set_qpos(self, qpos: np.ndarray) -> None:
        q = np.asarray(qpos, dtype=np.float64).reshape(-1)
        n = min(len(q), self.model.nq)
        self.data.qpos[:n] = q[:n]
        self.mujoco.mj_forward(self.model, self.data)

    def _set_viewpoint(self, viewpoint: dict) -> None:
        self.camera.lookat[:] = np.asarray(viewpoint["lookat"], dtype=np.float64)
        self.camera.distance = float(viewpoint["distance"])
        self.camera.azimuth = float(viewpoint["azimuth"])
        self.camera.elevation = float(viewpoint["elevation"])
        self.model.vis.global_.fovy = float(viewpoint.get("camera_fov", 57.0))

    def render_mask(self, qpos: np.ndarray, viewpoint: dict) -> np.ndarray:
        self._set_qpos(qpos)
        self._set_viewpoint(viewpoint)
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.scene_option)
        try:
            self.renderer.enable_segmentation_rendering()
            seg = self.renderer.render()
            self.renderer.disable_segmentation_rendering()
            mask = self._segmentation_mask(seg)
            if mask is not None:
                return mask
        except Exception:
            try:
                self.renderer.disable_segmentation_rendering()
            except Exception:
                pass

        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.scene_option)
        rgb = self.renderer.render()
        mask = np.any(rgb > 12, axis=2)
        if 0 < int(mask.sum()) < int(0.75 * mask.size):
            return mask
        return np.zeros(rgb.shape[:2], dtype=bool)

    def T_cam2base_from_viewpoint(self, qpos: np.ndarray, viewpoint: dict) -> np.ndarray:
        """Return OpenCV cam2base for the current MuJoCo free-camera viewpoint."""
        self._set_qpos(qpos)
        self._set_viewpoint(viewpoint)
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.scene_option)
        gl_cam = self.renderer.scene.camera[0]
        pos = np.asarray(gl_cam.pos, dtype=np.float64).reshape(3)
        forward = np.asarray(gl_cam.forward, dtype=np.float64).reshape(3)
        up = np.asarray(gl_cam.up, dtype=np.float64).reshape(3)

        forward = forward / max(1e-12, np.linalg.norm(forward))
        up = up / max(1e-12, np.linalg.norm(up))
        right = np.cross(forward, up)
        right = right / max(1e-12, np.linalg.norm(right))
        up = np.cross(-forward, right)
        up = up / max(1e-12, np.linalg.norm(up))

        # MuJoCo/OpenGL camera columns are +x right, +y up, +z backward.
        R_mj_cam2base = np.column_stack([right, up, -forward])
        R_cv_cam2base = R_mj_cam2base @ np.diag([1.0, -1.0, -1.0])
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_cv_cam2base
        T[:3, 3] = pos
        return np.linalg.inv(self._T_world_base()) @ T


def _load_qpos(traj, key: str | None = None) -> np.ndarray:
    candidates = []
    if key:
        candidates.append(key)
    candidates += ["google_robot_all_qpos", "all_qpos", "joint_position", "joint_positions", "joints", "q"]
    for name in candidates:
        if name in traj.files:
            qpos = np.asarray(traj[name], dtype=np.float64)
            break
    else:
        raise RuntimeError(f"trajectory.npz has no MuJoCo qpos. Tried: {candidates}")
    if qpos.ndim == 1:
        qpos = qpos.reshape(1, -1)
    return qpos


def _choose_stems(args: argparse.Namespace, ep_seg: Path) -> list[str]:
    if args.frames:
        return [f"{int(s):05d}" if str(s).isdigit() else str(s) for s in args.frames]
    pnp_path = ep_seg / args.pnp_json_name
    if pnp_path.exists():
        pnp = json.loads(pnp_path.read_text())
        stems = pnp.get("inlier_stems") or pnp.get("kept_stems") or []
        if stems:
            return [str(s) for s in stems[: args.max_frames]]
    stems = sorted(p.stem for p in (ep_seg / "000").glob("*.png"))
    return stems[: args.max_frames]


def process_episode(args: argparse.Namespace, ep_name: str) -> dict:
    ep_oxe = args.oxe_root / args.dataset / ep_name
    ep_seg = args.mask_root / args.dataset / ep_name
    traj_path = ep_oxe / "trajectory.npz"
    if not traj_path.exists():
        return {"episode": ep_name, "status": "skip-missing-trajectory"}
    stems = _choose_stems(args, ep_seg)
    if not stems:
        return {"episode": ep_name, "status": "skip-no-frames"}

    out_root = ep_seg / args.out_dir_name
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    with np.load(traj_path, allow_pickle=True) as traj:
        qpos = _load_qpos(traj, args.mujoco_qpos_key)
        for stem in stems:
            img = cv2.imread(str(ep_oxe / "frames" / f"{stem}.jpg"))
            if img is None:
                rows.append({"episode": ep_name, "stem": stem, "status": "skip-missing-image"})
                continue
            idx = int(stem)
            if idx >= len(qpos):
                rows.append({"episode": ep_name, "stem": stem, "status": "skip-qpos-range"})
                continue
            H, W = img.shape[:2]
            target = np.zeros((H, W), dtype=bool)
            for mask_dir in args.target_mask_dirs:
                target |= _read_mask(_mask_path(ep_seg, mask_dir, stem), (H, W))
            renderer = FreeCameraGoogleRobotRenderer(
                args.mujoco_xml_path,
                H,
                W,
                geom_groups=_parse_groups(args.mujoco_geom_groups),
                geom_name_include=args.mujoco_geom_name_include,
                geom_name_exclude=args.mujoco_geom_name_exclude,
                base_body=args.mujoco_base_body,
            )
            viewpoints = _load_viewpoints(args.viewpoints_json)
            frame_dir = out_root / f"frame_{stem}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            frame_rows = []
            tiles = [_draw_header(img, f"raw {stem}")]
            for vidx, viewpoint in enumerate(viewpoints):
                mask = renderer.render_mask(qpos[idx], viewpoint)
                rec = _metrics(target, mask)
                rec.update({"episode": ep_name, "stem": stem, "viewpoint": vidx, "status": "ok"})
                frame_rows.append(rec)
                over = _overlay(img, mask)
                label = f"v{vidx:02d} iou={rec['iou']:.3f} cov={rec['target_coverage']:.2f} prec={rec['render_precision']:.2f}"
                cv2.imwrite(str(frame_dir / f"view_{vidx:02d}.jpg"), _draw_header(over, label))
            renderer.close()

            frame_rows.sort(key=lambda r: (r["iou"], r["target_coverage"], r["render_precision"]), reverse=True)
            for rec in frame_rows[: args.top_k]:
                tile = cv2.imread(str(frame_dir / f"view_{int(rec['viewpoint']):02d}.jpg"))
                if tile is not None:
                    tiles.append(tile)
            if tiles:
                sheet = np.hstack(tiles)
                cv2.imwrite(str(frame_dir / "_top_viewpoints.jpg"), sheet)
            rows.extend(frame_rows)
            best = frame_rows[0] if frame_rows else None
            if best is not None:
                print(
                    f"[{args.dataset}/{ep_name} {stem}] best_view={best['viewpoint']:02d} "
                    f"iou={best['iou']:.3f} cov={best['target_coverage']:.3f} "
                    f"prec={best['render_precision']:.3f} out={frame_dir}"
                )

    (out_root / "viewpoint_scores.json").write_text(json.dumps(rows, indent=2))
    return {"episode": ep_name, "status": "ok", "frames": len(stems), "out": str(out_root)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--oxe_root", type=Path, required=True)
    p.add_argument("--mask_root", type=Path, required=True)
    p.add_argument("--dataset", default="fractal20220817_data")
    p.add_argument("--episodes", nargs="+", default=None)
    p.add_argument("--frames", nargs="+", default=None)
    p.add_argument("--pnp_json_name", default="pnp_fovy57.json")
    p.add_argument("--mujoco_xml_path", type=Path, required=True)
    p.add_argument("--mujoco_qpos_key", default=None)
    p.add_argument("--mujoco_geom_groups", default="2")
    p.add_argument("--mujoco_geom_name_include", default=None)
    p.add_argument("--mujoco_geom_name_exclude", default=None)
    p.add_argument("--mujoco_base_body", default=None)
    p.add_argument("--viewpoints_json", type=Path, default=None)
    p.add_argument("--target_mask_dirs", nargs="+", default=["000"])
    p.add_argument("--out_dir_name", default="mujoco_viewpoint_check")
    p.add_argument("--max_frames", type=int, default=4)
    p.add_argument("--top_k", type=int, default=5)
    args = p.parse_args()

    ds_seg = args.mask_root / args.dataset
    episodes = args.episodes or sorted(
        pth.name for pth in ds_seg.iterdir()
        if pth.is_dir() and pth.name.startswith("episode_")
    )
    summary = [process_episode(args, ep) for ep in episodes]
    (ds_seg / "mujoco_viewpoint_check_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
