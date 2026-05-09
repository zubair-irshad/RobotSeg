"""Per-dataset proprio (EEF pose) extractors for OXE.

Inspired by otter's oxe_standardization_transforms.py — every dataset has
its own field names, units, and rotation conventions; rather than try to
infer dims from a schema string, we hardcode an explicit extractor per
dataset that returns a unified record:

    {
        "xyz":   (T, 3) float32,   # EE position in robot base frame
        "rot":   (T, 3) or (T, 4),  # rotation, format below
        "rot_format": "euler_xyz" | "quat_wxyz" | "quat_xyzw" | None,
        "gripper": (T, 1) float32 | None,
    }

Use `extract_episode_proprio(steps, extractor)` to pull this from a list
of TFDS steps. `nested_get` traverses 'a/b'-style nested keys.

Verified per-dataset slices match real data via tools/inspect_oxe_state.py;
units are documented per entry. The translation from solvePnP will be in
the same units as the dataset's xyz (meters for everything except UCSD
pick-and-place).
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np


@dataclass
class EEFExtractor:
    state_key: str  # supports nested 'a/b' lookup
    xyz_slice: tuple[int, int] = (0, 3)
    rot_slice: tuple[int, int] | None = None
    rot_format: str | None = None  # "euler_xyz" / "quat_wxyz" / "quat_xyzw"
    # Gripper can either be an extra slice on the same `state_key` array, or
    # a separate observation key (e.g. fractal's 'gripper_closed'). At most one.
    gripper_slice: tuple[int, int] | None = None
    gripper_obs_key: str | None = None
    # Some datasets store the state field as ZLIB-compressed bytes
    # (kuka's `clip_function_input/base_pose_tool_reached`).
    decompress_zlib: bool = False
    decoded_shape: tuple[int, ...] | None = None
    decoded_dtype: str = "float32"
    units_note: str = "meters"


# ------ verified by tools/inspect_oxe_state.py ------
EXTRACTORS: dict[str, EEFExtractor] = {
    "taco_play": EEFExtractor(
        state_key="robot_obs",
        xyz_slice=(0, 3), rot_slice=(3, 6), rot_format="euler_xyz",
        gripper_slice=(6, 7),
        units_note="meters; gripper_width in m",
    ),
    "bridge": EEFExtractor(
        state_key="state",
        xyz_slice=(0, 3), rot_slice=(3, 6), rot_format="euler_xyz",
        gripper_slice=(6, 7),
        units_note="meters",
    ),
    "berkeley_autolab_ur5": EEFExtractor(
        state_key="robot_state",
        xyz_slice=(7, 10), rot_slice=(10, 14), rot_format="quat_wxyz",
        gripper_slice=(6, 7),
        units_note="meters; gripper_is_closed at [6]",
    ),
    "fractal20220817_data": EEFExtractor(
        state_key="state",
        xyz_slice=(0, 3), rot_slice=(3, 6), rot_format="euler_xyz",
        gripper_slice=(6, 7),
        units_note="meters; state[0:3] is TCP xyz, state[3:6] is TCP rotation",
    ),
    "kuka": EEFExtractor(
        state_key="clip_function_input/base_pose_tool_reached",
        xyz_slice=(0, 3), rot_slice=(3, 7), rot_format="quat_wxyz",
        gripper_obs_key="gripper_closed",
        decompress_zlib=True, decoded_shape=(7,), decoded_dtype="float32",
        units_note="meters; ZLIB-compressed in TFDS shards",
    ),
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": EEFExtractor(
        state_key="state",
        xyz_slice=(0, 3), rot_slice=(3, 6), rot_format="euler_xyz",
        gripper_slice=(6, 7),
        units_note="NOT meters (likely scaled); ranges ~2–4 in dim 0",
    ),
    "droid": EEFExtractor(
        state_key="cartesian_position",
        xyz_slice=(0, 3), rot_slice=(3, 6), rot_format="euler_xyz",
        gripper_obs_key="gripper_position",
        units_note="meters",
    ),
}


def nested_get(obs, key):
    """Look up a possibly-nested key like 'a/b/c'."""
    if key in obs:
        return obs[key]
    cur = obs
    for part in key.split("/"):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _read_state(obs, ext: EEFExtractor) -> np.ndarray | None:
    raw = nested_get(obs, ext.state_key)
    if raw is None:
        return None
    if ext.decompress_zlib:
        arr_bytes = np.asarray(raw).tobytes()
        try:
            decoded = zlib.decompress(arr_bytes)
        except zlib.error:
            return None
        arr = np.frombuffer(decoded, dtype=ext.decoded_dtype).copy()
        if ext.decoded_shape is not None:
            arr = arr.reshape(ext.decoded_shape)
        return arr.ravel().astype(np.float32)
    return np.asarray(raw, dtype=np.float32).ravel()


def extract_step_proprio(step_obs, ext: EEFExtractor) -> dict | None:
    """Pull (xyz, rot, gripper) from one TFDS step's observation."""
    arr = _read_state(step_obs, ext)
    if arr is None or arr.size < ext.xyz_slice[1]:
        return None
    out = {"xyz": arr[ext.xyz_slice[0]:ext.xyz_slice[1]].astype(np.float32)}
    if ext.rot_slice is not None and arr.size >= ext.rot_slice[1]:
        out["rot"] = arr[ext.rot_slice[0]:ext.rot_slice[1]].astype(np.float32)
        out["rot_format"] = ext.rot_format
    if ext.gripper_slice is not None and arr.size >= ext.gripper_slice[1]:
        out["gripper"] = arr[ext.gripper_slice[0]:ext.gripper_slice[1]].astype(np.float32)
    elif ext.gripper_obs_key is not None:
        g = nested_get(step_obs, ext.gripper_obs_key)
        if g is not None:
            out["gripper"] = np.asarray(g, dtype=np.float32).ravel()[:1]
    return out


def extract_episode_proprio(step_obs_list, ext: EEFExtractor) -> dict:
    """Stack proprio over an episode. Returns standardized dict-of-arrays.

    Returned keys:
        xyz:   (T, 3)
        rot:   (T, R) or absent
        rot_format: str or None
        gripper:  (T, 1) or absent
    """
    rows = [extract_step_proprio(o, ext) for o in step_obs_list]
    rows = [r for r in rows if r is not None]
    if not rows:
        return {}
    out = {"xyz": np.stack([r["xyz"] for r in rows], axis=0)}
    if "rot" in rows[0]:
        out["rot"] = np.stack([r["rot"] for r in rows], axis=0)
        out["rot_format"] = rows[0]["rot_format"]
    if "gripper" in rows[0]:
        out["gripper"] = np.stack([r["gripper"] for r in rows], axis=0)
    return out
