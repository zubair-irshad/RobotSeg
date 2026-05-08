"""Robust per-episode cam2base estimation from RobotSeg gripper centroids.

Pipeline (per OXE episode):
  1. Load 3D end-effector positions from <oxe_root>/<dataset>/<episode>/trajectory.npz
     using the state-schema dims encoded in tools/oxe_registry.py.
  2. Load 2D gripper centroids + per-frame confidence from
     <mask_root>/<dataset>/<episode>/001/centroids.json
     (the new format written by test/inference_auto_qual.py — also reads
     legacy [cx, cy] entries with conf=1.0).
  3. Reject frames where the prediction is unreliable (low confidence,
     tiny / huge mask, multiple connected components, mask touching the
     image border, NaN state, etc.).
  4. Solve cam2base via cv2.solvePnPRansac on (3D, 2D) and refine on
     the inlier set with cv2.solvePnPRefineLM.
  5. Flag a scene as bad if too few inliers, high reprojection RMSE,
     or low inlier fraction.

Outputs (per episode):
  <mask_root>/<dataset>/<episode>/pnp.json   # T_cam2base, inliers, rmse, status
  <mask_root>/<dataset>/<episode>/pnp_viz/   # reprojection overlays (--viz)

Outputs (per dataset):
  <mask_root>/<dataset>/pnp_summary.json     # one row per episode + bad-scene list

Usage:
  python tools/pnp_oxe.py \
    --oxe_root  ~/RobotSeg/data/oxe_subset \
    --mask_root ~/RobotSeg/data/oxe_subset_seg \
    --viz

  # only certain datasets:
  python tools/pnp_oxe.py --datasets bridge taco_play --viz

Intrinsics:
  Most OXE datasets ship without camera intrinsics. By default we assume a
  pinhole with horizontal FOV --hfov_deg (default 60°) and principal point
  at the image center. Override per dataset via --K_json:
      {"bridge": {"fx":..., "fy":..., "cx":..., "cy":...}, ...}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from oxe_registry import OXE_DATASETS  # noqa: E402
from urdf_robot_masker import URDFRobotMasker  # noqa: E402


DEFAULT_URDF = (
    Path("data")
    / "urdfs"
    / "python-example-droid-dataset"
    / "franka_description"
    / "panda.urdf"
)


# Where in trajectory["state"] the EE xyz lives, parsed from the registry schema.
EE_XYZ_DIMS: dict[str, tuple[int, int]] = {
    "taco_play": (0, 3),
    "berkeley_autolab_ur5": (7, 10),
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": (0, 3),
    "bridge": (0, 3),
    # fractal: registry sets state_key='base_pose_tool_reached'.
    "fractal20220817_data": (0, 3),
    # kuka: registry sets state_key='clip_function_input/base_pose_tool_reached'.
    "kuka": (0, 3),
    "droid": (0, 3),
    # NOT included: cmu_stretch (state[1]≡0 → coplanar EE → degenerate
    # PnP); roboturk (no proprio state); fanuc_manipulation_v2 (GCS path
    # broken); aloha_mobile (GCS path broken).
}


def _entry_to_record(entry):
    """Parse a centroids.json value into a dict with the fields the filter
    uses. Handles both the new dict format and the legacy [cx,cy] list."""
    if entry is None:
        return {"centroid": None, "conf_mean": 0.0, "conf_max": 0.0,
                "area_frac": 0.0, "n_components": 0,
                "arm_overlap_frac": 1.0, "observed": False}
    if isinstance(entry, dict):
        return {
            "centroid": entry.get("centroid"),
            "conf_mean": float(entry.get("conf_mean", 1.0)),
            "conf_max":  float(entry.get("conf_max", 1.0)),
            "area_frac": float(entry.get("area_frac", 0.0)),
            "n_components": int(entry.get("n_components", 1)),
            "arm_overlap_frac": float(entry.get("arm_overlap_frac", 0.0)),
            "pre_subtract_area_frac": float(
                entry.get("pre_subtract_area_frac", entry.get("area_frac", 0.0))
            ),
            "observed": entry.get("observed", None),
        }
    # Legacy list form (no quality info; assume good).
    return {"centroid": entry, "conf_mean": 1.0, "conf_max": 1.0,
            "area_frac": 0.0, "n_components": 1,
            "arm_overlap_frac": 0.0, "pre_subtract_area_frac": 0.0,
            "observed": True}


def _mask_path(root: Path, stem: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg"):
        path = root / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def _read_binary_mask(path: Path, shape_hw: tuple[int, int]) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = img > 127
    if mask.shape[:2] != shape_hw:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (shape_hw[1], shape_hw[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return mask


def _load_masks_for_stem(ep_dir_seg: Path, stem: str, roots: list[str],
                         shape_hw: tuple[int, int]) -> np.ndarray | None:
    out = np.zeros(shape_hw, dtype=bool)
    found = False
    for root_name in roots:
        path = _mask_path(ep_dir_seg / root_name, stem)
        if path is None:
            continue
        mask = _read_binary_mask(path, shape_hw)
        if mask is None:
            continue
        out |= mask
        found = True
    return out if found else None


def _parse_mask_roots(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _robot_proximity_ok(ep_dir_seg: Path, stem: str, centroid: list[float],
                        W: int, H: int, args) -> tuple[bool, str | None]:
    if not args.require_gripper_near_robot and not args.require_gripper_inside_robot:
        return True, None
    roots = _parse_mask_roots(args.robot_mask_roots)
    robot = _load_masks_for_stem(ep_dir_seg, stem, roots, (H, W))
    if robot is None:
        if args.skip_if_robot_mask_missing:
            return False, "missing-robot-mask"
        return True, None
    if int(robot.sum()) < args.min_robot_mask_area_px:
        return False, f"robot-mask-small({int(robot.sum())})"
    cx, cy = int(round(float(centroid[0]))), int(round(float(centroid[1])))
    if not (0 <= cx < W and 0 <= cy < H):
        return False, "centroid-outside-image"

    if args.require_gripper_inside_robot:
        radius = max(0, int(args.gripper_robot_disk_radius_px))
        disk = np.zeros((H, W), dtype=np.uint8)
        cv2.circle(disk, (cx, cy), radius, 1, thickness=-1)
        sel = disk.astype(bool)
        cover = float(robot[sel].mean()) if np.any(sel) else float(robot[cy, cx])
        if cover < args.min_gripper_robot_disk_frac:
            return False, f"outside-robot-mask({cover:.2f})"

    if robot[cy, cx] or not args.require_gripper_near_robot:
        return True, None
    inv = (~robot).astype(np.uint8)
    dt = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    dist = float(dt[cy, cx])
    if dist > args.max_gripper_robot_dist_px:
        return False, f"far-from-robot({dist:.1f}px)"
    return True, None


def _K_from_camera_json(cam_json: dict, W: int, H: int) -> dict | None:
    """Pick the most plausible 3x3 K from a saved camera.json (written by
    download_oxe_subset.py). Returns None if nothing usable found.

    Heuristic: prefer a (3,3) tensor with positive diagonal and cx/cy
    inside the image. Accept (9,) flattened too.
    """
    raw = cam_json.get("raw", []) if isinstance(cam_json, dict) else []
    best = None
    for entry in raw:
        data = entry.get("data")
        try:
            arr = np.asarray(data, dtype=np.float64)
        except Exception:
            continue
        if arr.size == 9:
            K = arr.reshape(3, 3)
        elif arr.shape == (4,) and "intrinsic" in entry.get("path", "").lower():
            # (fx, fy, cx, cy) packed
            fx, fy, cx, cy = arr.tolist()
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        else:
            continue
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        if fx <= 0 or fy <= 0:
            continue
        if not (0 <= cx <= W and 0 <= cy <= H):
            continue
        cand = {"fx": float(fx), "fy": float(fy),
                "cx": float(cx), "cy": float(cy),
                "width": W, "height": H,
                "source": entry.get("path", "rlds")}
        # prefer entries whose cx/cy are near image center
        score = -(abs(cx - 0.5 * W) + abs(cy - 0.5 * H))
        if best is None or score > best[0]:
            best = (score, cand)
    return best[1] if best else None


def _normalize_if_needed(K: np.ndarray, W: int, H: int) -> tuple[np.ndarray, str]:
    """Scale normalized intrinsics to pixels when they look like [0..1] K."""
    K = np.asarray(K, dtype=np.float64).copy()
    normalized = (
        0 < K[0, 0] <= 5.0 and 0 < K[1, 1] <= 5.0
        and 0 <= K[0, 2] <= 1.5 and 0 <= K[1, 2] <= 1.5
    )
    if normalized:
        K[0, 0] *= W
        K[1, 1] *= H
        K[0, 2] *= W
        K[1, 2] *= H
        return K, ":normalized"
    return K, ""


def _K_candidate_from_array(arr: np.ndarray, path: str, W: int, H: int) -> dict | None:
    """Parse a plausible K from one trajectory.npz array."""
    try:
        arr = np.asarray(arr)
    except Exception:
        return None
    if arr.dtype.kind not in "fiu":
        return None
    try:
        data = arr.astype(np.float64)
    except Exception:
        return None
    if not np.all(np.isfinite(data)):
        data = data[np.isfinite(data)]
        if data.size == 0:
            return None

    K = None
    if data.size == 9:
        K = data.reshape(3, 3)
    elif data.ndim >= 2 and data.shape[-2:] == (3, 3):
        mats = data.reshape(-1, 3, 3)
        mats = mats[np.all(np.isfinite(mats), axis=(1, 2))]
        if len(mats):
            K = np.median(mats, axis=0)
    elif data.shape[-1:] == (4,):
        rows = data.reshape(-1, 4)
        rows = rows[np.all(np.isfinite(rows), axis=1)]
        if len(rows):
            fx, fy, cx, cy = np.median(rows, axis=0).tolist()
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]],
                         dtype=np.float64)

    if K is None:
        return None
    K, suffix = _normalize_if_needed(K, W, H)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    if fx <= 0 or fy <= 0:
        return None
    if not (0 <= cx <= W and 0 <= cy <= H):
        return None
    if fx < 0.1 * W or fy < 0.1 * H or fx > 20 * W or fy > 20 * H:
        return None

    name = path.lower()
    hint_bonus = 1000.0 if any(
        h in name for h in (
            "intrinsic", "camera_matrix", "camera_k", "cam_k",
            "exterior_image_1_left", "image_1_left", "left",
        )
    ) else 0.0
    center_score = -(abs(cx - 0.5 * W) + abs(cy - 0.5 * H))
    aspect_score = -abs((fx / max(fy, 1e-9)) - (W / max(H, 1)))
    return {
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "width": W, "height": H,
        "source": f"trajectory.npz:{path}{suffix}",
        "_score": float(hint_bonus + center_score + 20.0 * aspect_score),
    }


def _K_from_trajectory_npz(traj, W: int, H: int) -> tuple[dict | None, list[dict]]:
    """Scan trajectory.npz for stored camera intrinsics.

    Handles direct 3x3 K, flattened 9-vector K, packed [fx,fy,cx,cy], and
    per-frame stacks of those. Returns (best, candidates).
    """
    candidates = []
    for key in getattr(traj, "files", []):
        try:
            cand = _K_candidate_from_array(traj[key], key, W, H)
        except Exception:
            cand = None
        if cand is not None:
            candidates.append(cand)
    candidates.sort(key=lambda c: c.get("_score", 0.0), reverse=True)
    public = []
    for cand in candidates:
        clean = dict(cand)
        clean.pop("_score", None)
        public.append(clean)
    return (public[0] if public else None), public


def _fmt_K(K: dict | None) -> str:
    if not K:
        return "none"
    return (
        f"{K.get('source', '?')} "
        f"fx={float(K['fx']):.2f} fy={float(K['fy']):.2f} "
        f"cx={float(K['cx']):.2f} cy={float(K['cy']):.2f}"
    )


def _valid_K_dict(K: dict | None, W: int, H: int) -> bool:
    if not isinstance(K, dict):
        return False
    try:
        fx = float(K["fx"])
        fy = float(K["fy"])
        cx = float(K["cx"])
        cy = float(K["cy"])
    except (KeyError, TypeError, ValueError):
        return False
    if not all(np.isfinite(v) for v in (fx, fy, cx, cy)):
        return False
    if fx <= 0 or fy <= 0:
        return False
    if not (0 <= cx <= W and 0 <= cy <= H):
        return False
    if fx < 0.1 * W or fy < 0.1 * H or fx > 20 * W or fy > 20 * H:
        return False
    return True


def _default_K(W: int, H: int, hfov_deg: float) -> dict:
    fx = 0.5 * W / math.tan(0.5 * math.radians(hfov_deg))
    return {"fx": fx, "fy": fx, "cx": 0.5 * W, "cy": 0.5 * H,
            "width": W, "height": H}


def _K_to_mat(K: dict) -> np.ndarray:
    return np.array([[K["fx"], 0, K["cx"]],
                     [0, K["fy"], K["cy"]],
                     [0, 0, 1.0]], dtype=np.float64)


def _lookup_K_override(K_overrides: dict, ds_name: str, ep_name: str) -> dict | None:
    """Accept dataset-wide or per-episode K override JSONs.

    Supported forms:
      {"droid": {"fx":..., "fy":..., "cx":..., "cy":...}}
      {"droid": {"episode_0000": {"fx":...}}}
      {"droid/episode_0000": {"fx":...}}
      {"episode_0000": {"fx":...}}
    """
    if not K_overrides:
        return None
    for key in (f"{ds_name}/{ep_name}", ep_name):
        val = K_overrides.get(key)
        if isinstance(val, dict) and all(k in val for k in ("fx", "fy", "cx", "cy")):
            return val

    ds_val = K_overrides.get(ds_name)
    if isinstance(ds_val, dict):
        if all(k in ds_val for k in ("fx", "fy", "cx", "cy")):
            return ds_val
        ep_val = ds_val.get(ep_name)
        if isinstance(ep_val, dict) and all(k in ep_val for k in ("fx", "fy", "cx", "cy")):
            return ep_val
    return None


def _read_image_size(seq_frames_dir: Path) -> tuple[int, int] | None:
    for name in sorted(os.listdir(seq_frames_dir)):
        if name.lower().endswith((".jpg", ".png")):
            img = cv2.imread(str(seq_frames_dir / name))
            if img is not None:
                return img.shape[1], img.shape[0]  # W, H
    return None


def _eef_rot_to_R(rot, fmt):
    """Convert OXE eef_rot of any common format to (T,3,3) R_wrist->base matrices."""
    rot = np.asarray(rot, dtype=np.float64)
    fmt = (fmt or "").lower().strip()
    from scipy.spatial.transform import Rotation
    if fmt in ("quat_xyzw", "xyzw", "quaternion_xyzw"):
        return Rotation.from_quat(rot).as_matrix()
    if fmt in ("quat_wxyz", "wxyz", "quaternion_wxyz"):
        # scipy expects (x,y,z,w); roll w to last.
        return Rotation.from_quat(np.concatenate([rot[..., 1:], rot[..., :1]], -1)).as_matrix()
    if fmt in ("rotvec", "axis_angle", "axisangle", "axangle"):
        return Rotation.from_rotvec(rot).as_matrix()
    if fmt in ("matrix", "rotmat", "matrix_3x3"):
        return rot.reshape(-1, 3, 3)
    if fmt.startswith("euler"):
        seq = fmt.split("_", 1)[1] if "_" in fmt else "xyz"
        return Rotation.from_euler(seq, rot).as_matrix()
    if fmt in ("6d", "rot6d"):
        a1, a2 = rot[..., :3], rot[..., 3:]
        b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
        b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
        b2 /= np.linalg.norm(b2, axis=-1, keepdims=True)
        b3 = np.cross(b1, b2)
        return np.stack([b1, b2, b3], axis=-1)
    raise ValueError(f"Unknown eef_rot_format: {fmt!r}")


def _load_joint_array(traj) -> np.ndarray | None:
    for key in ("joint_position", "joint_positions", "joints", "q", "arm_joints"):
        if key in getattr(traj, "files", []):
            arr = np.asarray(traj[key], dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.ndim == 2 and arr.shape[1] >= 7:
                return arr[:, :7]
    return None


def _load_gripper_array(traj, n: int) -> np.ndarray:
    for key in ("gripper", "gripper_position", "finger_joint"):
        if key in getattr(traj, "files", []):
            arr = np.asarray(traj[key], dtype=np.float64).reshape(-1)
            if len(arr) >= n:
                return arr[:n]
    return np.zeros(n, dtype=np.float64)


def _make_urdf_masker(args) -> URDFRobotMasker:
    if not args.urdf_path.exists():
        raise FileNotFoundError(
            f"missing URDF for --pnp_point_source={args.pnp_point_source}: {args.urdf_path}"
        )
    return URDFRobotMasker(
        args.urdf_path,
        mesh_dir=args.mesh_dir,
        backend=args.urdf_backend,
        downsample=4,
        dilate_px=0,
        verbose=False,
    )


def _fk_point_sequence(traj, n: int, args) -> tuple[np.ndarray | None, str]:
    source = args.pnp_point_source
    if source == "eef_xyz":
        return None, "eef_xyz"

    joints = _load_joint_array(traj)
    if joints is None:
        return None, "missing-joint_position"
    n = min(n, len(joints))
    gripper = _load_gripper_array(traj, n)
    masker = _make_urdf_masker(args)

    if source == "finger_midpoint":
        links = ("left_outer_finger", "right_outer_finger")
    elif source == "inner_finger_midpoint":
        links = ("left_inner_finger", "right_inner_finger")
    elif source == "all_finger_midpoint":
        links = (
            "left_outer_finger",
            "right_outer_finger",
            "left_inner_finger",
            "right_inner_finger",
        )
    elif source.startswith("urdf_midpoint:"):
        links = tuple(x.strip() for x in source.split(":", 1)[1].split(",") if x.strip())
        if len(links) != 2:
            return None, f"bad-point-source({source})"
    elif source.startswith("urdf_link:"):
        links = (source.split(":", 1)[1].strip(),)
    else:
        aliases = {
            "panda_link7": ("panda_link7",),
            "panda_link8": ("panda_link8",),
            "robotiq_base": ("robotiq_85_base_link",),
            "left_outer_finger": ("left_outer_finger",),
            "right_outer_finger": ("right_outer_finger",),
            "left_inner_finger": ("left_inner_finger",),
            "right_inner_finger": ("right_inner_finger",),
        }
        links = aliases.get(source)
        if links is None:
            return None, f"unknown-point-source({source})"

    pts = []
    for i in range(n):
        cfg = masker._cfg(joints[i], float(gripper[i]))
        link_T = masker.robot.link_transforms(cfg)
        missing = [link for link in links if link not in link_T]
        if missing:
            return None, f"missing-urdf-link({','.join(missing)})"
        origins = np.stack([link_T[link][:3, 3] for link in links], axis=0)
        pts.append(origins.mean(axis=0))
    return np.asarray(pts, dtype=np.float64), source


def _filter_frames(centroids_json, ee_xyz_seq, W, H, args, ee_R_seq=None,
                   ep_dir_seg: Path | None = None):
    """Return (pts3d Nx3, pts2d Nx2, kept_stems, rejected[, R_wrist Nx3x3]) post-filter."""
    img_area = float(W * H)
    min_area_eff = max(args.min_area, args.min_area_px / img_area)

    pts3d, pts2d, kept, rej, Rs = [], [], [], [], []
    for stem, entry in centroids_json.items():
        idx = int(stem)
        if idx >= len(ee_xyz_seq):
            rej.append((stem, "no-traj-state"))
            continue
        xyz = ee_xyz_seq[idx]
        if not np.all(np.isfinite(xyz)):
            rej.append((stem, "nan-state"))
            continue
        Rw = ee_R_seq[idx] if ee_R_seq is not None else None
        if Rw is not None and not np.all(np.isfinite(Rw)):
            rej.append((stem, "nan-rot"))
            continue

        rec = _entry_to_record(entry)
        c = rec["centroid"]
        if c is None:
            rej.append((stem, "no-centroid"))
            continue
        # Hard gate: if the seg pipeline already flagged the gripper as
        # unobserved (e.g. 16th frame in your bridge episode), drop it.
        if args.use_observed_flag and rec["observed"] is False:
            rej.append((stem, "not-observed"))
            continue
        if rec["conf_mean"] < args.min_conf:
            rej.append((stem, f"low-conf({rec['conf_mean']:.2f})"))
            continue
        if rec["conf_max"] < args.min_conf_max:
            rej.append((stem, f"low-conf-max({rec['conf_max']:.2f})"))
            continue
        if rec["area_frac"] < min_area_eff:
            rej.append((stem, f"area-small({rec['area_frac']:.4f})"))
            continue
        area_px = rec["area_frac"] * img_area
        if area_px < args.min_gripper_area_px:
            rej.append((stem, f"area-px-small({area_px:.0f})"))
            continue
        if rec["pre_subtract_area_frac"] > 0:
            residual_frac = rec["area_frac"] / rec["pre_subtract_area_frac"]
            if residual_frac < args.min_post_subtract_area_ratio:
                rej.append((stem, f"post-subtract-small({residual_frac:.3f})"))
                continue
        if rec["area_frac"] > args.max_area:
            rej.append((stem, f"area-large({rec['area_frac']:.4f})"))
            continue
        if rec["arm_overlap_frac"] > args.max_arm_overlap:
            rej.append((stem, f"arm-overlap({rec['arm_overlap_frac']:.2f})"))
            continue
        if rec["n_components"] > 1 and args.reject_fragmented:
            rej.append((stem, f"fragmented({rec['n_components']})"))
            continue
        cx, cy = c
        if (cx < args.edge_margin or cy < args.edge_margin or
                cx > W - args.edge_margin or cy > H - args.edge_margin):
            rej.append((stem, "edge"))
            continue
        if ep_dir_seg is not None:
            ok, reason = _robot_proximity_ok(ep_dir_seg, stem, c, W, H, args)
            if not ok:
                rej.append((stem, reason or "far-from-robot"))
                continue

        pts3d.append(xyz)
        pts2d.append([cx, cy])
        kept.append(stem)
        if Rw is not None:
            Rs.append(Rw)

    pts3d = np.asarray(pts3d, dtype=np.float64)
    pts2d = np.asarray(pts2d, dtype=np.float64)
    R_arr = np.asarray(Rs, dtype=np.float64) if Rs else None
    return pts3d, pts2d, kept, rej, R_arr


def _solve_pnp_with_K_estimation(pts3d, pts2d, W, H, args):
    """Jointly estimate fx (=fy), rvec, tvec for a fixed-camera trajectory.

    Principal point fixed at image center; only the focal length is free
    alongside the 6-DoF extrinsic. Initialised from a HFOV-guess PnP. RANSAC
    is run at the initial focal first to reject 2D outliers, then SciPy LM
    optimises (fx, rvec, tvec) on the inlier set. Returns the same dict as
    _solve_pnp plus the estimated K."""
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return None  # caller will fall back to fixed K

    if len(pts3d) < max(5, args.min_inliers):
        return None
    cx, cy = 0.5 * W, 0.5 * H
    fx0 = 0.5 * W / math.tan(0.5 * math.radians(args.hfov_deg))

    K0 = np.array([[fx0, 0, cx], [0, fx0, cy], [0, 0, 1.0]], dtype=np.float64)
    seed = _solve_pnp(pts3d, pts2d, K0, args)
    if seed is None:
        return None
    inlier_idx = np.array(seed["inlier_idx"], dtype=int)
    if len(inlier_idx) < max(5, args.min_inliers):
        return None
    rvec0 = np.asarray(seed["rvec"], dtype=np.float64).reshape(3)
    tvec0 = np.asarray(seed["tvec"], dtype=np.float64).reshape(3)

    p3 = pts3d[inlier_idx].astype(np.float64)
    p2 = pts2d[inlier_idx].astype(np.float64)

    def residuals(params):
        fx = params[0]
        rvec = params[1:4].reshape(3, 1)
        tvec = params[4:7].reshape(3, 1)
        K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1.0]], dtype=np.float64)
        proj, _ = cv2.projectPoints(p3, rvec, tvec, K, np.zeros(5))
        return (proj.reshape(-1, 2) - p2).flatten()

    # Normalize the rvec seed to canonical [-pi, pi] magnitude so it doesn't
    # blow past the bounds (cv2's RANSAC can produce angles > 2π).
    ang = float(np.linalg.norm(rvec0))
    if ang > 1e-9:
        ang_wrapped = ((ang + np.pi) % (2 * np.pi)) - np.pi
        rvec0 = rvec0 * (ang_wrapped / ang)

    # Soft bounds keep fx in a sane range so we don't drift into degeneracy.
    fx_lo = 0.5 * W / math.tan(0.5 * math.radians(120.0))
    fx_hi = 0.5 * W / math.tan(0.5 * math.radians(15.0))
    # Allow large translation ranges for datasets in non-meter units (e.g.
    # ucsd_pick_and_place is in scaled units; tvec can be O(10²)).
    t_abs = float(np.abs(tvec0).max())
    t_bound = max(100.0, 5.0 * t_abs)
    lo = np.array([fx_lo,  -np.pi, -np.pi, -np.pi,  -t_bound, -t_bound, -t_bound])
    hi = np.array([fx_hi,   np.pi,  np.pi,  np.pi,   t_bound,  t_bound,  t_bound])

    x0 = np.concatenate([[fx0], rvec0, tvec0])
    # Final safety clip — avoids "Initial guess is outside of provided bounds"
    # from any remaining numerical drift in fx0 / rvec0 / tvec0.
    eps = 1e-6
    x0 = np.minimum(np.maximum(x0, lo + eps), hi - eps)
    bounds = (lo, hi)

    sol = least_squares(residuals, x0, bounds=bounds, method="trf",
                        loss="huber", f_scale=2.0, max_nfev=200)
    fx_est = float(sol.x[0])
    rvec = sol.x[1:4].reshape(3, 1)
    tvec = sol.x[4:7].reshape(3, 1)

    K_est = {"fx": fx_est, "fy": fx_est, "cx": cx, "cy": cy,
             "width": W, "height": H, "source": "estimated"}
    K_mat = _K_to_mat(K_est)

    proj, _ = cv2.projectPoints(p3, rvec, tvec, K_mat, np.zeros(5))
    err = np.linalg.norm(proj.reshape(-1, 2) - p2, axis=1)
    rmse = float(np.sqrt((err ** 2).mean()))

    R, _ = cv2.Rodrigues(rvec)
    T_base2cam = np.eye(4)
    T_base2cam[:3, :3] = R
    T_base2cam[:3, 3] = tvec.flatten()
    T_cam2base = np.linalg.inv(T_base2cam)

    return {
        "rvec": rvec.flatten().tolist(),
        "tvec": tvec.flatten().tolist(),
        "T_cam2base": T_cam2base.tolist(),
        "T_base2cam": T_base2cam.tolist(),
        "inlier_idx": inlier_idx.astype(int).tolist(),
        "rmse_px": rmse,
        "max_err_px": float(err.max()),
        "K_estimated": K_est,
    }


def _solve_pnp_with_tool_offset(pts3d, R_wrist, pts2d, K_mat, args):
    """Jointly fit (rvec, tvec, offset_tool[3]) so that the tip in base frame is
       tip_base(t) = pts3d[t] + R_wrist[t] @ offset_tool
    This compensates for the case where pts3d is the WRIST and the observed
    centroid is the gripper TIP — a constant 3-vector in the tool frame whose
    image-space projection rotates with the gripper. Same K throughout.
    """
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return None
    if pts3d.shape[0] < max(6, args.min_inliers):
        return None
    if R_wrist is None or R_wrist.shape[0] != pts3d.shape[0]:
        return None

    # Seed extrinsic from a regular wrist-only PnP (offset = 0).
    seed = _solve_pnp(pts3d, pts2d, K_mat, args)
    if seed is None:
        return None
    rvec0 = np.asarray(seed["rvec"], dtype=np.float64).reshape(3)
    tvec0 = np.asarray(seed["tvec"], dtype=np.float64).reshape(3)

    # Wrap rvec into [-pi, pi] so it doesn't blow past bounds during LM.
    ang = float(np.linalg.norm(rvec0))
    if ang > 1e-9:
        rvec0 = rvec0 * ((((ang + np.pi) % (2 * np.pi)) - np.pi) / ang)

    inliers = np.array(seed["inlier_idx"], dtype=int)
    p3 = pts3d[inliers].astype(np.float64)
    Rw = R_wrist[inliers].astype(np.float64)
    p2 = pts2d[inliers].astype(np.float64)

    K = K_mat.astype(np.float64)
    dist = np.zeros(5)

    def residuals(params):
        rvec = params[0:3].reshape(3, 1)
        tvec = params[3:6].reshape(3, 1)
        offset = params[6:9]
        # tip_base = wrist_base + R_wrist @ offset
        tip = p3 + (Rw @ offset)
        proj, _ = cv2.projectPoints(tip, rvec, tvec, K, dist)
        return (proj.reshape(-1, 2) - p2).flatten()

    t_abs = float(np.abs(tvec0).max())
    t_bound = max(100.0, 5.0 * t_abs)
    off_bound = float(args.tool_offset_bound)
    lo = np.array([-np.pi]*3 + [-t_bound]*3 + [-off_bound]*3)
    hi = np.array([ np.pi]*3 + [ t_bound]*3 + [ off_bound]*3)
    x0 = np.concatenate([rvec0, tvec0, np.zeros(3)])
    eps = 1e-6
    x0 = np.minimum(np.maximum(x0, lo + eps), hi - eps)

    sol = least_squares(residuals, x0, bounds=(lo, hi), method="trf",
                        loss="huber", f_scale=2.0, max_nfev=300)
    rvec = sol.x[0:3].reshape(3, 1)
    tvec = sol.x[3:6].reshape(3, 1)
    offset_tool = sol.x[6:9].astype(np.float64)

    # Re-evaluate inliers on FULL kept set under the optimised model.
    tip_all = pts3d + (R_wrist @ offset_tool)
    proj_all, _ = cv2.projectPoints(tip_all.astype(np.float64),
                                    rvec, tvec, K, dist)
    err_all = np.linalg.norm(proj_all.reshape(-1, 2) - pts2d, axis=1)
    refined_inliers = np.where(err_all <= float(args.reproj_thresh))[0]
    if len(refined_inliers) < args.min_inliers:
        refined_inliers = inliers  # keep seed inliers

    # Final LM refine on the cleaner inlier set.
    p3 = pts3d[refined_inliers]
    Rw = R_wrist[refined_inliers]
    p2 = pts2d[refined_inliers]
    sol2 = least_squares(residuals, sol.x, bounds=(lo, hi), method="trf",
                         loss="huber", f_scale=2.0, max_nfev=200)
    rvec = sol2.x[0:3].reshape(3, 1)
    tvec = sol2.x[3:6].reshape(3, 1)
    offset_tool = sol2.x[6:9].astype(np.float64)

    tip_in = pts3d[refined_inliers] + (R_wrist[refined_inliers] @ offset_tool)
    proj, _ = cv2.projectPoints(tip_in.astype(np.float64), rvec, tvec, K, dist)
    err = np.linalg.norm(proj.reshape(-1, 2) - pts2d[refined_inliers], axis=1)
    rmse = float(np.sqrt((err ** 2).mean()))

    R, _ = cv2.Rodrigues(rvec)
    T_base2cam = np.eye(4)
    T_base2cam[:3, :3] = R
    T_base2cam[:3, 3] = tvec.flatten()
    T_cam2base = np.linalg.inv(T_base2cam)

    return {
        "rvec": rvec.flatten().tolist(),
        "tvec": tvec.flatten().tolist(),
        "T_cam2base": T_cam2base.tolist(),
        "T_base2cam": T_base2cam.tolist(),
        "inlier_idx": refined_inliers.astype(int).tolist(),
        "rmse_px": rmse,
        "max_err_px": float(err.max()) if err.size else 0.0,
        "tool_offset": offset_tool.tolist(),
        "tool_offset_norm": float(np.linalg.norm(offset_tool)),
    }


def _solve_pnp(pts3d, pts2d, K_mat, args):
    """Returns dict with rvec, tvec, T_cam2base, inlier_idx (into pts3d), rmse."""
    if len(pts3d) < max(4, args.min_inliers):
        return None
    dist = np.zeros(5)
    # EPNP for >=5 points, ITERATIVE for 4 (which needs at least 4 coplanar/3D pts).
    flags = cv2.SOLVEPNP_EPNP if len(pts3d) >= 5 else cv2.SOLVEPNP_ITERATIVE
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d.astype(np.float64), pts2d.astype(np.float64),
        K_mat, dist,
        iterationsCount=2000,
        reprojectionError=float(args.reproj_thresh),
        confidence=0.999,
        flags=flags,
    )
    if not ok or inliers is None or len(inliers) < args.min_inliers:
        return None
    inlier_idx = inliers.flatten()

    # LM refine on inliers
    rvec, tvec = cv2.solvePnPRefineLM(
        pts3d[inlier_idx], pts2d[inlier_idx], K_mat, dist, rvec, tvec,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-8),
    )

    # Second-pass outlier rejection: reproject ALL points with the
    # refined extrinsic and keep only those within reproj_thresh again.
    # This catches outliers that slipped past the looser EPNP-init solve.
    proj_all, _ = cv2.projectPoints(pts3d, rvec, tvec, K_mat, dist)
    err_all = np.linalg.norm(proj_all.reshape(-1, 2) - pts2d, axis=1)
    refined_inliers = np.where(err_all <= float(args.reproj_thresh))[0]
    if len(refined_inliers) >= args.min_inliers:
        inlier_idx = refined_inliers
        # Re-refine on the cleaner inlier set.
        rvec, tvec = cv2.solvePnPRefineLM(
            pts3d[inlier_idx], pts2d[inlier_idx], K_mat, dist, rvec, tvec,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-8),
        )

    R, _ = cv2.Rodrigues(rvec)
    # cv2 PnP returns the transform mapping points-in-world -> points-in-camera.
    # Here "world" = robot base. So [R|t] maps base -> cam, i.e. T_base2cam.
    T_base2cam = np.eye(4)
    T_base2cam[:3, :3] = R
    T_base2cam[:3, 3] = tvec.flatten()
    T_cam2base = np.linalg.inv(T_base2cam)

    proj, _ = cv2.projectPoints(pts3d[inlier_idx], rvec, tvec, K_mat, dist)
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(proj - pts2d[inlier_idx], axis=1)
    rmse = float(np.sqrt((err ** 2).mean()))

    return {
        "rvec": rvec.flatten().tolist(),
        "tvec": tvec.flatten().tolist(),
        "T_cam2base": T_cam2base.tolist(),
        "T_base2cam": T_base2cam.tolist(),
        "inlier_idx": inlier_idx.astype(int).tolist(),
        "rmse_px": rmse,
        "max_err_px": float(err.max()),
    }


def _viz_reproj(image_bgr, pts2d_kept, pts3d_kept, inlier_set,
                ee_centroid, K_mat, rvec, tvec):
    """Per-frame reprojection visualization."""
    viz = image_bgr.copy()
    proj, _ = cv2.projectPoints(
        pts3d_kept, rvec, tvec, K_mat, np.zeros(5)
    )
    proj = proj.reshape(-1, 2)
    for i, ((u, v), (cx, cy)) in enumerate(zip(proj, pts2d_kept)):
        col = (0, 200, 0) if i in inlier_set else (0, 0, 200)
        cv2.circle(viz, (int(round(cx)), int(round(cy))), 3, col, -1)
        cv2.circle(viz, (int(round(u)), int(round(v))), 5, col, 1)
        cv2.line(viz, (int(round(cx)), int(round(cy))),
                 (int(round(u)), int(round(v))), col, 1)
    return viz


def _viz_one_frame(image_bgr, c2d, c3d_proj, is_inlier):
    viz = image_bgr.copy()
    col = (0, 200, 0) if is_inlier else (0, 0, 200)
    if c2d is not None:
        cv2.drawMarker(viz, (int(round(c2d[0])), int(round(c2d[1]))),
                       (0, 255, 255), cv2.MARKER_CROSS, 22, 2)
    if c3d_proj is not None:
        cv2.circle(viz, (int(round(c3d_proj[0])), int(round(c3d_proj[1]))),
                   6, col, 2)
    return viz


def _project_pts(pts3d, rvec, tvec, K_mat):
    proj, _ = cv2.projectPoints(np.asarray(pts3d, dtype=np.float64),
                                rvec, tvec, K_mat, np.zeros(5))
    return proj.reshape(-1, 2)


def process_episode(ds_name, ep_dir_oxe: Path, ep_dir_seg: Path,
                    K_override: dict | None, args) -> dict:
    if ds_name not in EE_XYZ_DIMS:
        return {"status": "skip-no-ee-dim", "episode": ep_dir_seg.name}

    traj_npz = ep_dir_oxe / "trajectory.npz"
    centroids_path = ep_dir_seg / "001" / "centroids.json"
    frames_dir = ep_dir_oxe / "frames"
    if not traj_npz.exists() or not centroids_path.exists() or not frames_dir.exists():
        return {"status": "skip-missing", "episode": ep_dir_seg.name}

    traj = np.load(traj_npz)
    # Prefer the standardized 'eef_xyz' field (written by the new
    # oxe_extractors path); fall back to the raw 'state' slice for old
    # trajectory.npz files.
    if "eef_xyz" in traj.files:
        ee = np.asarray(traj["eef_xyz"], dtype=np.float64)
    elif "state" in traj.files:
        state = np.asarray(traj["state"], dtype=np.float64)
        a, b = EE_XYZ_DIMS[ds_name]
        if state.shape[1] < b:
            return {"status": "skip-state-too-short", "episode": ep_dir_seg.name}
        ee = state[:, a:b]
    else:
        return {"status": "skip-no-state", "episode": ep_dir_seg.name}

    fk_points, point_source_status = _fk_point_sequence(traj, len(ee), args)
    if fk_points is not None:
        ee = fk_points
    elif args.pnp_point_source != "eef_xyz":
        return {
            "status": "skip-bad-point-source",
            "episode": ep_dir_seg.name,
            "dataset": ds_name,
            "point_source": args.pnp_point_source,
            "reason": point_source_status,
        }

    # Per-frame R_wrist->base (only needed for --solve_tool_offset).
    ee_R = None
    if args.solve_tool_offset and args.pnp_point_source != "eef_xyz":
        print(
            f"  [warn] {ep_dir_seg.name}: --solve_tool_offset is ignored for "
            f"--pnp_point_source={args.pnp_point_source}"
        )
    elif args.solve_tool_offset and "eef_rot" in traj.files:
        try:
            fmt = (str(traj["eef_rot_format"])
                   if "eef_rot_format" in traj.files else "quat_xyzw")
            ee_R = _eef_rot_to_R(traj["eef_rot"], fmt)
        except Exception as e:
            print(f"  [warn] {ep_dir_seg.name}: eef_rot decode failed ({e}); "
                  f"disabling tool-offset solve")
            ee_R = None

    centroids = json.loads(centroids_path.read_text())

    img_size = _read_image_size(frames_dir)
    if img_size is None:
        return {"status": "skip-no-image", "episode": ep_dir_seg.name}
    W, H = img_size

    traj_K, traj_K_candidates = _K_from_trajectory_npz(traj, W, H)

    # Resolution order for K:
    #   --K_json override
    # > trajectory.npz intrinsics (--trajectory_intrinsics)
    # > per-episode camera.json (saved by download_oxe_subset.py from RLDS)
    # > MoGe-2 robust-median estimate over N sampled frames (--moge_intrinsics)
    # > HFOV guess (default).
    K = None
    selected_from = None
    if K_override is not None:
        if _valid_K_dict(K_override, W, H):
            K = dict(K_override)
            K.setdefault("width", W)
            K.setdefault("height", H)
            K.setdefault("source", "K_json")
            selected_from = "K_json"
        else:
            print(
                f"  [warn] {ds_name}/{ep_dir_seg.name}: ignoring invalid K_json "
                f"{_fmt_K(K_override)}"
            )
    if K is None and args.trajectory_intrinsics and traj_K is not None:
        K = dict(traj_K)
        selected_from = "trajectory"
    if args.require_trajectory_intrinsics and traj_K is None:
        if args.print_intrinsics:
            print(f"  [K/{ds_name}/{ep_dir_seg.name}] trajectory=none")
        return {
            "status": "skip-no-trajectory-intrinsics",
            "episode": ep_dir_seg.name,
            "dataset": ds_name,
            "image_size": [W, H],
            "trajectory_K_candidates": traj_K_candidates,
        }
    camera_K = None
    cam_path = ep_dir_oxe / "camera.json"
    if cam_path.exists():
        try:
            cam_json = json.loads(cam_path.read_text())
            camera_K = _K_from_camera_json(cam_json, W, H)
        except Exception:
            camera_K = None
    if K is None and camera_K is not None:
        K = camera_K
        selected_from = "camera_json"
    K_known = K is not None
    if args.require_known_intrinsics and not K_known:
        if args.print_intrinsics:
            print(
                f"  [K/{ds_name}/{ep_dir_seg.name}] "
                f"trajectory={_fmt_K(traj_K)} camera={_fmt_K(camera_K)} selected=none"
            )
        return {
            "status": "skip-no-known-intrinsics",
            "episode": ep_dir_seg.name,
            "dataset": ds_name,
            "image_size": [W, H],
            "camera_json_exists": (ep_dir_oxe / "camera.json").exists(),
            "trajectory_K_candidates": traj_K_candidates,
        }

    K_moge = None
    moge_cache_K = None
    moge_cache = ep_dir_seg / "moge_K.json"
    if moge_cache.exists():
        try:
            moge_cache_K = json.loads(moge_cache.read_text())
        except Exception:
            moge_cache_K = None
    if K is None and args.moge_intrinsics:
        if moge_cache_K is not None and not args.moge_recompute:
            K_moge = moge_cache_K
        if K_moge is None:
            from moge_intrinsics import estimate_K_for_frames
            K_moge = estimate_K_for_frames(
                frames_dir,
                num_samples=args.moge_num_samples,
                device=args.moge_device,
                repo_id=args.moge_repo_id,
                verbose=args.moge_verbose,
            )
            if K_moge is not None:
                moge_cache.write_text(json.dumps(K_moge, indent=2))
        if K_moge is not None:
            K = {k: K_moge[k] for k in ("fx", "fy", "cx", "cy", "width", "height")}
            K["source"] = K_moge.get("source", "moge-2")
            selected_from = "moge"

    if K is None:
        K = _default_K(W, H, args.hfov_deg)
        K["source"] = f"hfov={args.hfov_deg}deg"
        selected_from = "hfov"
    if args.print_intrinsics:
        print(
            f"  [K/{ds_name}/{ep_dir_seg.name}] "
            f"trajectory={_fmt_K(traj_K)} | "
            f"camera={_fmt_K(camera_K)} | "
            f"moge_cache={_fmt_K(moge_cache_K)} | "
            f"selected({selected_from})={_fmt_K(K)}"
        )
    K_mat = _K_to_mat(K)

    pts3d, pts2d, kept_stems, rejected, R_kept = _filter_frames(
        centroids, ee, W, H, args, ee_R_seq=ee_R, ep_dir_seg=ep_dir_seg
    )

    result = {
        "episode": ep_dir_seg.name,
        "dataset": ds_name,
        "image_size": [W, H],
        "K": K,
        "K_known": K_known,
        "K_selected_from": selected_from,
        "point_source": args.pnp_point_source,
        "trajectory_K_candidates": traj_K_candidates,
        "num_total": len(centroids),
        "num_kept_after_filter": len(kept_stems),
        "rejection_breakdown": _rejection_breakdown(rejected),
    }

    # If we don't have known intrinsics and the user asked for joint
    # estimation, optimise (fx, R, t) on the trajectory. This produces
    # ONE K per episode (the camera is fixed within an episode).
    # MoGe-2 already gave us a per-episode K, so don't re-estimate over it.
    pnp = None
    if args.estimate_intrinsics and not K_known and K_moge is None:
        pnp = _solve_pnp_with_K_estimation(pts3d, pts2d, W, H, args)
        if pnp is not None and "K_estimated" in pnp:
            result["K"] = pnp["K_estimated"]
            K_mat = _K_to_mat(pnp["K_estimated"])
    if pnp is None and args.solve_tool_offset and R_kept is not None and len(R_kept):
        pnp = _solve_pnp_with_tool_offset(pts3d, R_kept, pts2d, K_mat, args)
    if pnp is None:
        pnp = _solve_pnp(pts3d, pts2d, K_mat, args)
    if pnp is None:
        result["status"] = "bad-pnp-fail"
        result["num_inliers"] = 0
        return result

    inlier_frac = len(pnp["inlier_idx"]) / max(1, len(kept_stems))
    is_bad = (
        len(pnp["inlier_idx"]) < args.min_inliers
        or pnp["rmse_px"] > args.max_rmse
        or inlier_frac < args.min_inlier_frac
    )
    result.update({
        "status": "bad" if is_bad else "ok",
        "num_inliers": len(pnp["inlier_idx"]),
        "inlier_frac": inlier_frac,
        "rmse_px": pnp["rmse_px"],
        "max_err_px": pnp["max_err_px"],
        "T_cam2base": pnp["T_cam2base"],
        "T_base2cam": pnp["T_base2cam"],
        "rvec": pnp["rvec"],
        "tvec": pnp["tvec"],
        "inlier_stems": [kept_stems[i] for i in pnp["inlier_idx"]],
    })
    if "tool_offset" in pnp:
        result["tool_offset"] = pnp["tool_offset"]
        result["tool_offset_norm_m"] = pnp["tool_offset_norm"]

    out_path = ep_dir_seg / args.pnp_json_name
    out_path.write_text(json.dumps(result, indent=2))

    if args.viz:
        viz_dir = ep_dir_seg / args.viz_dir_name
        viz_dir.mkdir(exist_ok=True)
        rvec = np.asarray(pnp["rvec"], dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(pnp["tvec"], dtype=np.float64).reshape(3, 1)
        inlier_set = set(int(i) for i in pnp["inlier_idx"])

        # If we solved a tool offset, project the TIP (= wrist + R_wrist@offset)
        # so the green/red circles in the viz match the actual gripper jaws.
        if "tool_offset" in pnp and R_kept is not None:
            offset = np.asarray(pnp["tool_offset"], dtype=np.float64)
            pts3d_for_viz = pts3d + (R_kept @ offset)
        else:
            pts3d_for_viz = pts3d
        proj_all = _project_pts(pts3d_for_viz, rvec, tvec, K_mat)
        first_stem = kept_stems[pnp["inlier_idx"][0]] if pnp["inlier_idx"] else kept_stems[0]
        first_img = cv2.imread(str(frames_dir / f"{first_stem}.jpg"))
        if first_img is not None:
            cv2.imwrite(
                str(viz_dir / "_traj_overview.jpg"),
                _viz_reproj(first_img, pts2d, pts3d_for_viz, inlier_set, None,
                            K_mat, rvec, tvec),
            )

        # per-frame reprojection check (only for kept frames)
        for i, stem in enumerate(kept_stems):
            img = cv2.imread(str(frames_dir / f"{stem}.jpg"))
            if img is None:
                continue
            cv2.imwrite(
                str(viz_dir / f"{stem}.jpg"),
                _viz_one_frame(img, pts2d[i], proj_all[i], i in inlier_set),
            )

    return result


def _rejection_breakdown(rejected):
    counts = {}
    for _, reason in rejected:
        key = reason.split("(")[0]
        counts[key] = counts.get(key, 0) + 1
    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--oxe_root", type=Path, required=True)
    p.add_argument("--mask_root", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Datasets to process. Default: every dir under mask_root "
                        "that has an entry in tools/oxe_registry.py.")
    p.add_argument("--K_json", type=Path, default=None,
                   help="Optional JSON: {dataset: {fx,fy,cx,cy}} intrinsics.")
    p.add_argument("--hfov_deg", type=float, default=60.0)

    # filtering
    p.add_argument("--use_observed_flag", action="store_true", default=True,
                   help="Honor centroids.json[*].observed (set by inference) as a "
                        "hard reject. Disable with --no_use_observed_flag.")
    p.add_argument("--no_use_observed_flag", dest="use_observed_flag",
                   action="store_false")
    p.add_argument("--min_conf", type=float, default=0.5,
                   help="Reject if mean sigmoid prob inside the gripper mask < this.")
    p.add_argument("--min_conf_max", type=float, default=0.5,
                   help="Reject if peak sigmoid prob inside the mask < this. Catches "
                        "frames where the model never strongly believed any pixel "
                        "was the gripper.")
    p.add_argument("--max_arm_overlap", type=float, default=1.0,
                   help="Reject if PRE-subtraction the gripper mask overlapped the "
                        "arm mask by more than this fraction. Off by default (=1.0) "
                        "since the real gate is the post-subtract mask quality "
                        "encoded in `observed`. Set to e.g. 0.6 if you want it.")
    p.add_argument("--min_area", type=float, default=0.0002,
                   help="Minimum mask area as fraction of image (~0.02%%). "
                        "Floored at --min_area_px regardless.")
    p.add_argument("--min_area_px", type=int, default=16,
                   help="Absolute pixel floor for the gripper mask.")
    p.add_argument(
        "--min_gripper_area_px",
        type=int,
        default=40,
        help="Stricter absolute pixel floor for post-subtraction gripper masks used by PnP.",
    )
    p.add_argument(
        "--min_post_subtract_area_ratio",
        type=float,
        default=0.08,
        help="Reject when arm subtraction leaves only this fraction of the original gripper mask.",
    )
    p.add_argument("--max_area", type=float, default=0.4)
    p.add_argument("--edge_margin", type=int, default=4,
                   help="Reject centroids within N px of any image border.")
    p.add_argument("--reject_fragmented", action="store_true", default=True,
                   help="Reject frames whose mask has >1 connected component.")
    p.add_argument(
        "--require_gripper_near_robot",
        action="store_true",
        help="Reject gripper centroids that are far from robot/body masks. "
             "Useful when gripper segmentation fires on a scene object.",
    )
    p.add_argument(
        "--require_gripper_inside_robot",
        action="store_true",
        help="Reject gripper centroids whose local neighborhood is not covered "
             "by the robot/body mask. Stricter than --require_gripper_near_robot.",
    )
    p.add_argument("--gripper_robot_disk_radius_px", type=int, default=5)
    p.add_argument("--min_gripper_robot_disk_frac", type=float, default=0.25)
    p.add_argument(
        "--robot_mask_roots",
        default="002,000",
        help="Comma-separated mask folders used for robot proximity. "
             "Default tries full robot 002, then arm 000.",
    )
    p.add_argument("--max_gripper_robot_dist_px", type=float, default=18.0)
    p.add_argument("--min_robot_mask_area_px", type=int, default=64)
    p.add_argument(
        "--skip_if_robot_mask_missing",
        action="store_true",
        help="If set, reject frames when no robot/arm mask exists for proximity gating.",
    )

    # PnP
    p.add_argument("--reproj_thresh", type=float, default=8.0)
    p.add_argument("--min_inliers", type=int, default=5,
                   help="EPNP needs >=4; we require a few more for stability.")
    p.add_argument("--max_rmse", type=float, default=20.0)
    p.add_argument("--min_inlier_frac", type=float, default=0.4)
    p.add_argument(
        "--pnp_point_source",
        default="eef_xyz",
        help=(
            "3D point paired with the 2D gripper centroid. Options: eef_xyz "
            "(DROID cartesian_position[:3]), finger_midpoint, inner_finger_midpoint, "
            "all_finger_midpoint, panda_link7, "
            "panda_link8, robotiq_base, left_outer_finger, right_outer_finger, "
            "urdf_link:<link>, urdf_midpoint:<link_a>,<link_b>."
        ),
    )
    p.add_argument("--urdf_path", type=Path, default=DEFAULT_URDF)
    p.add_argument("--mesh_dir", type=Path, default=None)
    p.add_argument(
        "--urdf_backend",
        choices=["simple", "yourdfpy", "auto"],
        default="simple",
        help="URDF backend used for FK point sources.",
    )

    p.add_argument("--moge_intrinsics", action="store_true",
                   help="When K is not provided, sample N equally-spaced frames "
                        "per episode and run MoGe-2 to estimate fx, fy. The "
                        "robust-median K is then used in PnP. Cached as "
                        "<episode_seg>/moge_K.json.")
    p.add_argument("--moge_num_samples", type=int, default=20,
                   help="Frames per episode to run MoGe-2 on.")
    p.add_argument("--moge_device", default=None,
                   help="cuda / cpu (default: auto-detect).")
    p.add_argument("--moge_repo_id", default="Ruicheng/moge-2-vitl-normal")
    p.add_argument("--moge_recompute", action="store_true",
                   help="Ignore cached <ep>/moge_K.json and re-run MoGe-2.")
    p.add_argument("--moge_verbose", action="store_true")
    p.add_argument("--trajectory_intrinsics", action="store_true",
                   help="Prefer intrinsics found inside trajectory.npz before "
                        "camera.json / MoGe / HFOV. Scans for K, flattened K, "
                        "or packed [fx,fy,cx,cy] arrays.")
    p.add_argument("--require_trajectory_intrinsics", action="store_true",
                   help="Skip episodes that do not contain usable trajectory.npz "
                        "intrinsics. Implies no fallback to camera.json/MoGe/HFOV.")
    p.add_argument("--print_intrinsics", action="store_true",
                   help="Print trajectory, camera.json, cached MoGe, and selected "
                        "intrinsics for each episode.")
    p.add_argument("--require_known_intrinsics", action="store_true",
                   help="Use only --K_json, requested trajectory.npz intrinsics, "
                        "or per-episode camera.json intrinsics. If none are "
                        "usable, skip the episode instead of falling back to "
                        "MoGe/HFOV/estimated intrinsics.")

    p.add_argument("--solve_tool_offset", action="store_true",
                   help="Jointly estimate (R, t, offset_tool) where offset_tool "
                        "is a constant 3-vector in the gripper/tool frame. Uses "
                        "trajectory.npz['eef_rot'] (+ eef_rot_format). Fixes the "
                        "common case where the proprioceptive EE is the wrist "
                        "but the seg-mask centroid is the fingertip.")
    p.add_argument("--tool_offset_bound", type=float, default=0.30,
                   help="Bound (in trajectory units, usually meters) for each "
                        "component of the solved tool offset.")

    p.add_argument("--estimate_intrinsics", action="store_true",
                   help="When K is not provided (no --K_json and no per-episode "
                        "camera.json), jointly estimate fx (=fy) alongside the "
                        "extrinsic. One K per episode. Requires scipy.")
    p.add_argument("--viz", action="store_true",
                   help="Save reprojection overlays per kept frame.")
    p.add_argument("--pnp_json_name", default="pnp.json",
                   help="Per-episode output filename. Use pnp_rlds.json to "
                        "compare against an existing pnp.json non-destructively.")
    p.add_argument("--viz_dir_name", default="pnp_viz",
                   help="Per-episode visualization directory.")
    args = p.parse_args()

    K_overrides = {}
    if args.K_json is not None and args.K_json.exists():
        K_overrides = json.loads(args.K_json.read_text())

    if args.datasets:
        datasets = args.datasets
    else:
        datasets = [
            d.name for d in args.mask_root.iterdir()
            if d.is_dir() and d.name in OXE_DATASETS
        ]
        datasets.sort()

    overall = {}
    for ds in datasets:
        if ds not in EE_XYZ_DIMS:
            print(f"[skip] {ds}: no EE state dims known")
            continue
        ds_oxe = args.oxe_root / ds
        ds_seg = args.mask_root / ds
        if not ds_oxe.exists() or not ds_seg.exists():
            print(f"[skip] {ds}: missing dir")
            continue

        episodes = sorted(
            p for p in ds_seg.iterdir()
            if p.is_dir() and p.name.startswith("episode_")
        )
        if not episodes:
            print(f"[skip] {ds}: no episodes")
            continue

        results = []
        for ep_seg in episodes:
            ep_oxe = ds_oxe / ep_seg.name
            K_override = _lookup_K_override(K_overrides, ds, ep_seg.name)
            r = process_episode(ds, ep_oxe, ep_seg, K_override, args)
            results.append(r)
            tag = r.get("status", "?")
            extra = ""
            if "rmse_px" in r:
                off_s = ""
                if "tool_offset" in r:
                    off_s = (f"  tool_off=[{r['tool_offset'][0]:+.3f},"
                             f"{r['tool_offset'][1]:+.3f},"
                             f"{r['tool_offset'][2]:+.3f}]m"
                             f"({r['tool_offset_norm_m']*100:.1f}cm)")
                extra = (f"  rmse={r['rmse_px']:.2f}px  "
                         f"inliers={r['num_inliers']}/{r['num_kept_after_filter']}"
                         f"  K={r.get('K', {}).get('source', '?')}"
                         f"  point={r.get('point_source', 'eef_xyz')}"
                         f"{off_s}")
            elif "rejection_breakdown" in r:
                extra = (f"  kept={r['num_kept_after_filter']}/{r['num_total']}"
                         f"  reasons={r['rejection_breakdown']}")
            print(f"  [{ds}/{ep_seg.name}] {tag}{extra}")

        bad = [r["episode"] for r in results if r.get("status", "").startswith("bad")]
        ok = [r["episode"] for r in results if r.get("status") == "ok"]
        summary = {
            "dataset": ds,
            "num_episodes": len(results),
            "num_ok": len(ok),
            "num_bad": len(bad),
            "bad_episodes": bad,
            "results": results,
        }
        (ds_seg / "pnp_summary.json").write_text(json.dumps(summary, indent=2))
        overall[ds] = {"num_ok": len(ok), "num_bad": len(bad),
                       "bad_episodes": bad}
        print(f"=== {ds}: {len(ok)} ok / {len(bad)} bad / {len(results)} total")

    (args.mask_root / "pnp_overall.json").write_text(
        json.dumps(overall, indent=2)
    )
    print(f"\nWrote {args.mask_root / 'pnp_overall.json'}")


if __name__ == "__main__":
    main()
