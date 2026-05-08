"""URDF silhouette rendering utilities for camera/extrinsic QA.

The preferred backend uses ``yourdfpy``/``trimesh`` to flatten the articulated
visual scene, matching the DROID Franka rendering path. A lightweight standard
library STL backend is kept as a fallback for environments without those
dependencies.
"""

from __future__ import annotations

import math
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


FRANKA_ARM_JOINT_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))


def _floats(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=np.float64)
    vals = [float(x) for x in text.split()]
    return np.asarray(vals, dtype=np.float64)


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    rx, ry, rz = rpy
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rx_m = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry_m = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz_m = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rz_m @ ry_m @ rx_m


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    n = float(np.linalg.norm(axis))
    if n < 1e-12:
        return np.eye(3)
    x, y, z = axis / n
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ],
        dtype=np.float64,
    )


def _origin_matrix(origin: ET.Element | None) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    if origin is None:
        return T
    T[:3, 3] = _floats(origin.attrib.get("xyz"), (0.0, 0.0, 0.0))[:3]
    T[:3, :3] = _rpy_matrix(_floats(origin.attrib.get("rpy"), (0.0, 0.0, 0.0))[:3])
    return T


def _apply_transform(vertices: np.ndarray, T: np.ndarray) -> np.ndarray:
    vh = np.concatenate(
        [vertices.astype(np.float64), np.ones((len(vertices), 1), dtype=np.float64)],
        axis=1,
    )
    return (T @ vh.T).T[:, :3]


def _invert_se3(T: np.ndarray) -> np.ndarray:
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -Ti[:3, :3] @ T[:3, 3]
    return Ti


def _project(pts_cam: np.ndarray, K: dict[str, float]) -> np.ndarray:
    z = pts_cam[:, 2]
    u = K["fx"] * pts_cam[:, 0] / z + K["cx"]
    v = K["fy"] * pts_cam[:, 1] / z + K["cy"]
    return np.stack([u, v, z], axis=-1)


def _load_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = path.read_bytes()
    if len(data) >= 84:
        n_tri = struct.unpack_from("<I", data, 80)[0]
        expected = 84 + n_tri * 50
        if expected == len(data):
            dtype = np.dtype(
                [
                    ("normal", "<f4", (3,)),
                    ("vertices", "<f4", (3, 3)),
                    ("attr", "<u2"),
                ]
            )
            arr = np.frombuffer(data, dtype=dtype, count=n_tri, offset=84)
            vertices = arr["vertices"].reshape(-1, 3).astype(np.float64)
            faces = np.arange(len(vertices), dtype=np.int32).reshape(-1, 3)
            return vertices, faces

    vertices: list[list[float]] = []
    for line in data.decode("utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if len(vertices) % 3:
        raise ValueError(f"ASCII STL has a non-triangular vertex count: {path}")
    verts = np.asarray(vertices, dtype=np.float64)
    faces = np.arange(len(verts), dtype=np.int32).reshape(-1, 3)
    return verts, faces


@dataclass(frozen=True)
class _MeshChunk:
    vertices_link: np.ndarray
    faces: np.ndarray


@dataclass(frozen=True)
class _Mimic:
    joint: str
    multiplier: float = 1.0
    offset: float = 0.0


@dataclass(frozen=True)
class _Joint:
    name: str
    parent: str
    child: str
    joint_type: str
    origin: np.ndarray
    axis: np.ndarray
    mimic: _Mimic | None = None


def _link_allowed(
    link: str,
    include_prefixes: tuple[str, ...] | None,
    exclude_prefixes: tuple[str, ...] | None,
) -> bool:
    if include_prefixes and not any(link.startswith(prefix) for prefix in include_prefixes):
        return False
    if exclude_prefixes and any(link.startswith(prefix) for prefix in exclude_prefixes):
        return False
    return True


class _SimpleURDF:
    def __init__(
        self,
        urdf_path: Path,
        mesh_dir: Path | None = None,
        geometry: str = "collision",
        verbose: bool = True,
    ):
        self.urdf_path = urdf_path
        self.mesh_dir = mesh_dir
        self.geometry = geometry
        self.verbose = verbose
        self.links: set[str] = set()
        self.joints: list[_Joint] = []
        self.children: dict[str, list[_Joint]] = {}
        self.meshes_by_link: dict[str, list[_MeshChunk]] = {}
        self.root_link = ""
        self._load()

    def _load(self) -> None:
        root = ET.parse(self.urdf_path).getroot()
        self.links = {elem.attrib["name"] for elem in root.findall("link")}
        for elem in root.findall("joint"):
            parent = elem.find("parent")
            child = elem.find("child")
            if parent is None or child is None:
                continue
            axis_elem = elem.find("axis")
            mimic_elem = elem.find("mimic")
            mimic = None
            if mimic_elem is not None:
                mimic = _Mimic(
                    joint=mimic_elem.attrib["joint"],
                    multiplier=float(mimic_elem.attrib.get("multiplier", 1.0)),
                    offset=float(mimic_elem.attrib.get("offset", 0.0)),
                )
            joint = _Joint(
                name=elem.attrib["name"],
                parent=parent.attrib["link"],
                child=child.attrib["link"],
                joint_type=elem.attrib.get("type", "fixed"),
                origin=_origin_matrix(elem.find("origin")),
                axis=_floats(axis_elem.attrib.get("xyz") if axis_elem is not None else None, (0.0, 0.0, 1.0))[:3],
                mimic=mimic,
            )
            self.joints.append(joint)
            self.children.setdefault(joint.parent, []).append(joint)

        child_links = {j.child for j in self.joints}
        roots = sorted(self.links - child_links)
        self.root_link = roots[0] if roots else "panda_link0"

        for link_elem in root.findall("link"):
            link_name = link_elem.attrib["name"]
            chunks = self._load_link_meshes(link_elem)
            if chunks:
                self.meshes_by_link[link_name] = chunks

        if self.verbose:
            n_meshes = sum(len(v) for v in self.meshes_by_link.values())
            n_faces = sum(int(c.faces.shape[0]) for v in self.meshes_by_link.values() for c in v)
            print(
                f"[urdf_render] loaded {self.urdf_path} root={self.root_link} "
                f"geometry={self.geometry} links={len(self.meshes_by_link)} "
                f"meshes={n_meshes} faces={n_faces}"
            )

    def _load_link_meshes(self, link_elem: ET.Element) -> list[_MeshChunk]:
        chunks: list[_MeshChunk] = []
        groups = link_elem.findall(self.geometry)
        if not groups and self.geometry == "collision":
            groups = link_elem.findall("visual")
        if not groups and self.geometry == "visual":
            groups = link_elem.findall("collision")

        for group in groups:
            mesh_elem = group.find("geometry/mesh")
            if mesh_elem is None:
                continue
            filename = mesh_elem.attrib.get("filename")
            if not filename:
                continue
            mesh_path = self._resolve_mesh_path(filename)
            if mesh_path.suffix.lower() not in {".stl"}:
                if self.verbose:
                    print(f"[urdf_render] skip non-STL mesh in lightweight loader: {mesh_path}")
                continue
            vertices, faces = _load_stl(mesh_path)
            scale = _floats(mesh_elem.attrib.get("scale"), (1.0, 1.0, 1.0))[:3]
            vertices = vertices * scale.reshape(1, 3)
            vertices = _apply_transform(vertices, _origin_matrix(group.find("origin")))
            chunks.append(_MeshChunk(vertices_link=vertices, faces=faces))
        return chunks

    def _resolve_mesh_path(self, filename: str) -> Path:
        raw = filename
        if raw.startswith("package://"):
            parts = Path(raw[len("package://") :]).parts
            package, rest = parts[0], Path(*parts[1:])
            roots = [p for p in (self.mesh_dir, self.urdf_path.parent.parent, self.urdf_path.parent) if p is not None]
            for root in roots:
                cand = root / package / rest
                if cand.exists():
                    return cand
            raise FileNotFoundError(f"could not resolve package mesh {filename}")

        path = Path(raw)
        candidates: list[Path] = []
        if path.is_absolute():
            candidates.append(path)
        else:
            if self.mesh_dir is not None:
                candidates.append(self.mesh_dir / path)
            candidates.append(self.urdf_path.parent / path)
            candidates.append(self.urdf_path.parent.parent / path)
        for cand in candidates:
            if cand.exists():
                return cand
        raise FileNotFoundError(f"could not resolve mesh {filename}; tried {candidates}")

    def link_transforms(self, cfg: dict[str, float]) -> dict[str, np.ndarray]:
        out = {self.root_link: np.eye(4, dtype=np.float64)}
        stack = [self.root_link]
        while stack:
            parent = stack.pop()
            parent_T = out[parent]
            for joint in self.children.get(parent, []):
                q = float(cfg.get(joint.name, 0.0))
                if joint.mimic is not None:
                    q = (
                        float(cfg.get(joint.mimic.joint, 0.0)) * joint.mimic.multiplier
                        + joint.mimic.offset
                    )
                motion = np.eye(4, dtype=np.float64)
                if joint.joint_type in {"revolute", "continuous"}:
                    motion[:3, :3] = _axis_angle(joint.axis, q)
                elif joint.joint_type == "prismatic":
                    motion[:3, 3] = joint.axis * q
                out[joint.child] = parent_T @ joint.origin @ motion
                stack.append(joint.child)
        return out

    def combined_mesh(
        self,
        cfg: dict[str, float],
        include_link_prefixes: tuple[str, ...] | None = None,
        exclude_link_prefixes: tuple[str, ...] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        link_T = self.link_transforms(cfg)
        vertices_all: list[np.ndarray] = []
        faces_all: list[np.ndarray] = []
        offset = 0
        for link, chunks in self.meshes_by_link.items():
            if not _link_allowed(link, include_link_prefixes, exclude_link_prefixes):
                continue
            T = link_T.get(link)
            if T is None:
                continue
            for chunk in chunks:
                vertices = _apply_transform(chunk.vertices_link, T)
                vertices_all.append(vertices)
                faces_all.append(chunk.faces + offset)
                offset += len(vertices)
        if not vertices_all:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32)
        return np.vstack(vertices_all), np.vstack(faces_all).astype(np.int32)


class _YourdfpyURDF:
    def __init__(
        self,
        urdf_path: Path,
        mesh_dir: Path | None = None,
        verbose: bool = True,
    ):
        import yourdfpy

        kwargs = dict(
            build_scene_graph=True,
            load_meshes=True,
            build_collision_scene_graph=False,
            load_collision_meshes=False,
        )
        if mesh_dir is not None:
            kwargs["mesh_dir"] = str(mesh_dir)
        self.urdf_path = urdf_path
        self.mesh_dir = mesh_dir
        self.robot = yourdfpy.URDF.load(str(urdf_path), **kwargs)
        if verbose:
            combined = self.robot.scene.dump(concatenate=True)
            n_vertices = int(len(getattr(combined, "vertices", [])))
            n_faces = int(len(getattr(combined, "faces", [])))
            print(
                f"[urdf_render] loaded {urdf_path} backend=yourdfpy "
                f"root={self.robot.base_link} visual_vertices={n_vertices} "
                f"visual_faces={n_faces}"
            )

    def combined_mesh(
        self,
        cfg: dict[str, float],
        include_link_prefixes: tuple[str, ...] | None = None,
        exclude_link_prefixes: tuple[str, ...] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if cfg:
            self.robot.update_cfg(cfg)
        if include_link_prefixes or exclude_link_prefixes:
            return self._combined_mesh_by_link(
                cfg, include_link_prefixes, exclude_link_prefixes
            )
        combined = self.robot.scene.dump(concatenate=True)
        vertices = np.asarray(combined.vertices, dtype=np.float64)
        faces = np.asarray(combined.faces, dtype=np.int32)
        if len(vertices) == 0 or len(faces) == 0:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32)
        return vertices, faces

    def _combined_mesh_by_link(
        self,
        cfg: dict[str, float],
        include_link_prefixes: tuple[str, ...] | None,
        exclude_link_prefixes: tuple[str, ...] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        link_T = self.link_transforms(cfg)
        vertices_all: list[np.ndarray] = []
        faces_all: list[np.ndarray] = []
        offset = 0
        for link, link_obj in self.robot.link_map.items():
            if not _link_allowed(link, include_link_prefixes, exclude_link_prefixes):
                continue
            T_link = link_T.get(link)
            if T_link is None:
                continue
            visuals = getattr(link_obj, "visuals", []) or []
            for visual in visuals:
                T_visual = getattr(visual, "origin", None)
                if T_visual is None:
                    T_visual = np.eye(4, dtype=np.float64)
                T = T_link @ np.asarray(T_visual, dtype=np.float64)
                geometry = getattr(visual, "geometry", None)
                mesh_obj = getattr(geometry, "mesh", None)
                meshes = getattr(mesh_obj, "meshes", None)
                if meshes is None and mesh_obj is not None and hasattr(mesh_obj, "vertices"):
                    meshes = [mesh_obj]
                if not meshes:
                    continue
                for mesh in meshes:
                    vertices = np.asarray(getattr(mesh, "vertices", []), dtype=np.float64)
                    faces = np.asarray(getattr(mesh, "faces", []), dtype=np.int32)
                    if len(vertices) == 0 or len(faces) == 0:
                        continue
                    vertices_all.append(_apply_transform(vertices, T))
                    faces_all.append(faces + offset)
                    offset += len(vertices)
        if not vertices_all:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32)
        return np.vstack(vertices_all), np.vstack(faces_all).astype(np.int32)

    def link_transforms(self, cfg: dict[str, float]) -> dict[str, np.ndarray]:
        if cfg:
            self.robot.update_cfg(cfg)
        out: dict[str, np.ndarray] = {}
        for link in self.robot.link_map:
            try:
                mat = self.robot.get_transform(link)
            except Exception:
                continue
            out[link] = np.asarray(mat, dtype=np.float64)
        return out


class URDFRobotMasker:
    """Render a Franka + Robotiq URDF silhouette into a camera image."""

    def __init__(
        self,
        urdf_path: str | Path,
        mesh_dir: str | Path | None = None,
        downsample: int = 2,
        dilate_px: int = 3,
        geometry: str = "collision",
        backend: str = "simple",
        gripper_joint_name: str = "finger_joint",
        gripper_open_rad: float = 0.0,
        gripper_closed_rad: float = 0.7,
        verbose: bool = True,
    ):
        self.urdf_path = Path(urdf_path)
        if mesh_dir is None:
            mesh_dir = _default_mesh_dir(self.urdf_path)
        self.mesh_dir = Path(mesh_dir) if mesh_dir is not None else None
        backend = backend.lower()
        if backend not in {"simple", "yourdfpy", "auto"}:
            raise ValueError("backend must be one of: simple, yourdfpy, auto")
        self.backend = backend
        if backend in {"yourdfpy", "auto"}:
            try:
                self.robot = _YourdfpyURDF(
                    self.urdf_path,
                    mesh_dir=self.mesh_dir,
                    verbose=verbose,
                )
                self.backend = "yourdfpy"
            except Exception:
                if backend == "yourdfpy":
                    raise
                if verbose:
                    print("[urdf_render] yourdfpy backend unavailable; falling back to simple STL loader")
                self.robot = _SimpleURDF(
                    self.urdf_path,
                    mesh_dir=self.mesh_dir,
                    geometry=geometry,
                    verbose=verbose,
                )
                self.backend = "simple"
        else:
            self.robot = _SimpleURDF(
                self.urdf_path,
                mesh_dir=self.mesh_dir,
                geometry=geometry,
                verbose=verbose,
            )
        self.downsample = max(1, int(downsample))
        self.dilate_px = int(dilate_px)
        self.gripper_joint_name = gripper_joint_name
        self._grip_span = (float(gripper_open_rad), float(gripper_closed_rad))

    def _cfg(self, joint_positions: np.ndarray, gripper_position: float) -> dict[str, float]:
        q_arm = np.asarray(joint_positions, dtype=np.float64).reshape(-1)
        cfg: dict[str, float] = {}
        for i, name in enumerate(FRANKA_ARM_JOINT_NAMES):
            if i >= len(q_arm):
                break
            cfg[name] = float(q_arm[i])
        if self.gripper_joint_name:
            g = float(np.clip(gripper_position, 0.0, 1.0))
            lo, hi = self._grip_span
            cfg[self.gripper_joint_name] = lo + g * (hi - lo)
        return cfg

    def render(
        self,
        K: dict[str, float],
        T_cam2base: np.ndarray,
        joint_positions: np.ndarray,
        gripper_position: float = 0.0,
        image_hw: tuple[int, int] | None = None,
        include_link_prefixes: tuple[str, ...] | None = None,
        exclude_link_prefixes: tuple[str, ...] | None = None,
    ) -> np.ndarray:
        H = int(image_hw[0] if image_hw is not None else K["height"])
        W = int(image_hw[1] if image_hw is not None else K["width"])
        s = self.downsample
        Hs = max(1, H // s)
        Ws = max(1, W // s)
        Ks = {
            "fx": float(K["fx"]) / s,
            "fy": float(K["fy"]) / s,
            "cx": float(K["cx"]) / s,
            "cy": float(K["cy"]) / s,
        }

        vertices, faces = self.robot.combined_mesh(
            self._cfg(joint_positions, gripper_position),
            include_link_prefixes=include_link_prefixes,
            exclude_link_prefixes=exclude_link_prefixes,
        )
        if len(vertices) == 0 or len(faces) == 0:
            return np.zeros((H, W), dtype=bool)

        T_base2cam = _invert_se3(np.asarray(T_cam2base, dtype=np.float64))
        verts_cam = _apply_transform(vertices, T_base2cam)
        z = verts_cam[:, 2]
        keep = (
            (z[faces[:, 0]] > 0.05)
            & (z[faces[:, 1]] > 0.05)
            & (z[faces[:, 2]] > 0.05)
        )
        tri = faces[keep]
        mask = np.zeros((Hs, Ws), dtype=np.uint8)
        if len(tri):
            proj = _project(verts_cam, Ks)
            polys = proj[:, :2][tri].astype(np.int32)
            cv2.fillPoly(mask, polys, color=1)

        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        if self.dilate_px > 0:
            k = 2 * self.dilate_px + 1
            mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
        return mask.astype(bool)

    def overlay(
        self,
        image_bgr: np.ndarray,
        K: dict[str, float],
        T_cam2base: np.ndarray,
        joint_positions: np.ndarray,
        gripper_position: float = 0.0,
        color_bgr: tuple[int, int, int] = (229, 132, 11),
        outline_bgr: tuple[int, int, int] = (245, 165, 35),
        outline_px: int = 0,
        alpha: float = 0.60,
    ) -> tuple[np.ndarray, np.ndarray]:
        mask = self.render(
            K,
            T_cam2base,
            joint_positions,
            gripper_position=gripper_position,
            image_hw=image_bgr.shape[:2],
        )
        out = image_bgr.copy()
        if np.any(mask):
            color = np.asarray(color_bgr, dtype=np.float32)
            sel = mask
            out_f = out.astype(np.float32)
            out_f[sel] = (1.0 - alpha) * out_f[sel] + alpha * color
            out = np.clip(out_f, 0, 255).astype(np.uint8)
            if outline_px > 0:
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(
                    out, contours, -1, outline_bgr, int(outline_px),
                    lineType=cv2.LINE_AA,
                )
        return out, mask


def _default_mesh_dir(urdf_path: Path) -> Path | None:
    if urdf_path.parent.name == "franka_description":
        parent = urdf_path.parent.parent
        if (parent / "robotiq_arg85_description").exists():
            return parent
    return urdf_path.parent


def segment_robot_urdf(
    image: np.ndarray,
    K: dict[str, float],
    T_cam2base: np.ndarray,
    joint_positions: np.ndarray,
    gripper_position: float,
    urdf_path: str | Path,
    mesh_dir: str | Path | None = None,
    downsample: int = 2,
    dilate_px: int = 3,
    gripper_open_rad: float = 0.0,
    gripper_closed_rad: float = 0.7,
    backend: str = "simple",
) -> np.ndarray:
    masker = URDFRobotMasker(
        urdf_path,
        mesh_dir=mesh_dir,
        downsample=downsample,
        dilate_px=dilate_px,
        backend=backend,
        gripper_open_rad=gripper_open_rad,
        gripper_closed_rad=gripper_closed_rad,
    )
    return masker.render(
        K,
        T_cam2base,
        joint_positions,
        gripper_position=gripper_position,
        image_hw=image.shape[:2],
    )
