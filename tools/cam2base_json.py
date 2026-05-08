"""Helpers for reading camera-to-base extrinsics from loose JSON formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def invert_se3(T: np.ndarray) -> np.ndarray:
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -Ti[:3, :3] @ T[:3, 3]
    return Ti


def _axis_angle_matrix(rotvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = rotvec / theta
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    C = 1.0 - c
    return np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ],
        dtype=np.float64,
    )


def _matrix_from_value(value: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.shape == (4, 4):
        return arr
    if arr.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :] = arr
        return out
    if arr.shape == (16,):
        return arr.reshape(4, 4)
    if arr.shape == (12,):
        out = np.eye(4, dtype=np.float64)
        out[:3, :] = arr.reshape(3, 4)
        return out
    if arr.shape == (6,):
        # DROID pnp_cam2base_multiview.json stores [x, y, z, rx, ry, rz],
        # where r* is an axis-angle rotation vector for T_cam2base.
        out = np.eye(4, dtype=np.float64)
        out[:3, 3] = arr[:3]
        out[:3, :3] = _axis_angle_matrix(arr[3:])
        return out
    return None


def _extract_T_cam2base(
    record: Any,
    preferred_fields: list[str] | None = None,
) -> tuple[np.ndarray | None, str | None]:
    preferred_fields = preferred_fields or []
    if isinstance(record, dict):
        for key in (
            "T_cam2base",
            "cam2base",
            "camera_to_base",
            "T_camera_to_base",
            "T_cam_to_base",
        ):
            if key in record:
                T = _matrix_from_value(record[key])
                if T is not None:
                    return T, key
        for key in (
            "T_base2cam",
            "base2cam",
            "base_to_camera",
            "T_base_to_camera",
            "T_base_to_cam",
        ):
            if key in record:
                T = _matrix_from_value(record[key])
                if T is not None:
                    return invert_se3(T), f"invert({key})"
        for key in preferred_fields:
            if key in record:
                T = _matrix_from_value(record[key])
                if T is not None:
                    return T, key
        for key, value in record.items():
            if key in {"relative_path", "path", "episode", "episode_id", "dataset"}:
                continue
            T = _matrix_from_value(value)
            if T is not None:
                return T, key
    T = _matrix_from_value(record)
    if T is not None:
        return T, "matrix"
    return None, None


def _record_episode_keys(record: dict[str, Any]) -> set[str]:
    keys = set()
    for field in (
        "episode",
        "episode_id",
        "local_episode",
        "episode_name",
        "droid_episode_id",
        "raw_episode_id",
        "key",
        "id",
    ):
        val = record.get(field)
        if isinstance(val, str) and val:
            keys.add(val)
    ds = record.get("dataset")
    ep = record.get("episode") or record.get("episode_id") or record.get("local_episode")
    if isinstance(ds, str) and isinstance(ep, str):
        keys.add(f"{ds}/{ep}")
    return keys


def _load_episode_id_map(oxe_root: Path, dataset: str) -> dict[str, str]:
    path = oxe_root / dataset / "episode_id_map.json"
    if not path.exists():
        return {}
    try:
        data = load_json(path)
    except json.JSONDecodeError:
        return {}
    if isinstance(data, dict) and isinstance(data.get(dataset), dict):
        data = data[dataset]
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if isinstance(v, str)}


def episode_lookup_keys(oxe_root: Path, dataset: str, episode: str) -> list[str]:
    keys = [f"{dataset}/{episode}", episode]
    raw_id = _load_episode_id_map(oxe_root, dataset).get(episode)
    if raw_id:
        keys += [raw_id, f"{dataset}/{raw_id}"]
    out = []
    seen = set()
    for key in keys:
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _candidate_records(data: Any, keys: list[str]) -> list[tuple[Any, str]]:
    out: list[tuple[Any, str]] = []
    if isinstance(data, dict):
        for key in keys:
            if key in data:
                out.append((data[key], key))
        for container_key in ("results", "episodes", "data", "extrinsics", "cam2base"):
            val = data.get(container_key)
            if isinstance(val, (dict, list)):
                out += _candidate_records(val, keys)
        for key in keys:
            ds, _, ep = key.partition("/")
            if ep and isinstance(data.get(ds), dict):
                nested = data[ds]
                if ep in nested:
                    out.append((nested[ep], f"{ds}/{ep}"))
                if key in nested:
                    out.append((nested[key], f"{ds}/{key}"))
        rec_keys = _record_episode_keys(data)
        if rec_keys.intersection(keys):
            out.append((data, ",".join(sorted(rec_keys.intersection(keys)))))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict):
                rec_keys = _record_episode_keys(item)
                if rec_keys.intersection(keys):
                    out.append((item, f"list[{i}]"))
            elif str(i) in keys:
                out.append((item, f"list[{i}]"))
    return out


def find_T_cam2base(
    data: Any,
    keys: list[str],
    preferred_fields: list[str] | None = None,
) -> tuple[np.ndarray | None, str | None]:
    for record, source in _candidate_records(data, keys):
        T, field = _extract_T_cam2base(record, preferred_fields=preferred_fields)
        if T is not None:
            return T, f"{source}:{field}"
    T, field = _extract_T_cam2base(data, preferred_fields=preferred_fields)
    if T is not None and len(keys) <= 1:
        return T, field
    return None, None


def rotation_angle_deg(R_delta: np.ndarray) -> float:
    cos_theta = (float(np.trace(R_delta)) - 1.0) * 0.5
    cos_theta = min(1.0, max(-1.0, cos_theta))
    return float(np.degrees(np.arccos(cos_theta)))


def compare_T(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    delta = invert_se3(a) @ b
    return {
        "translation_delta_m": float(np.linalg.norm(a[:3, 3] - b[:3, 3])),
        "rotation_delta_deg": rotation_angle_deg(delta[:3, :3]),
        "dx_m": float(b[0, 3] - a[0, 3]),
        "dy_m": float(b[1, 3] - a[1, 3]),
        "dz_m": float(b[2, 3] - a[2, 3]),
    }
