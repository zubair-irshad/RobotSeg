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
    """Parse a centroids.json value into (centroid, conf_mean, area_frac, n_components)."""
    if entry is None:
        return None, 0.0, 0.0, 0
    if isinstance(entry, dict):
        return (
            entry.get("centroid"),
            float(entry.get("conf_mean", 1.0)),
            float(entry.get("area_frac", 0.0)),
            int(entry.get("n_components", 1)),
        )
    # legacy: list [cx, cy]
    return entry, 1.0, 0.0, 1


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


def _default_K(W: int, H: int, hfov_deg: float) -> dict:
    fx = 0.5 * W / math.tan(0.5 * math.radians(hfov_deg))
    return {"fx": fx, "fy": fx, "cx": 0.5 * W, "cy": 0.5 * H,
            "width": W, "height": H}


def _K_to_mat(K: dict) -> np.ndarray:
    return np.array([[K["fx"], 0, K["cx"]],
                     [0, K["fy"], K["cy"]],
                     [0, 0, 1.0]], dtype=np.float64)


def _read_image_size(seq_frames_dir: Path) -> tuple[int, int] | None:
    for name in sorted(os.listdir(seq_frames_dir)):
        if name.lower().endswith((".jpg", ".png")):
            img = cv2.imread(str(seq_frames_dir / name))
            if img is not None:
                return img.shape[1], img.shape[0]  # W, H
    return None


def _filter_frames(centroids_json, ee_xyz_seq, W, H, args):
    """Return (pts3d Nx3, pts2d Nx2, kept_stems, rejected) post-filter."""
    img_area = float(W * H)
    min_area_eff = max(args.min_area, args.min_area_px / img_area)

    pts3d, pts2d, kept, rej = [], [], [], []
    for stem, entry in centroids_json.items():
        idx = int(stem)
        if idx >= len(ee_xyz_seq):
            rej.append((stem, "no-traj-state"))
            continue
        xyz = ee_xyz_seq[idx]
        if not np.all(np.isfinite(xyz)):
            rej.append((stem, "nan-state"))
            continue

        c, conf, area, ncomp = _entry_to_record(entry)
        if c is None:
            rej.append((stem, "no-centroid"))
            continue
        if conf < args.min_conf:
            rej.append((stem, f"low-conf({conf:.2f})"))
            continue
        if area < min_area_eff:
            rej.append((stem, f"area-small({area:.4f})"))
            continue
        if area > args.max_area:
            rej.append((stem, f"area-large({area:.4f})"))
            continue
        if ncomp > 1 and args.reject_fragmented:
            rej.append((stem, f"fragmented({ncomp})"))
            continue
        cx, cy = c
        if (cx < args.edge_margin or cy < args.edge_margin or
                cx > W - args.edge_margin or cy > H - args.edge_margin):
            rej.append((stem, "edge"))
            continue

        pts3d.append(xyz)
        pts2d.append([cx, cy])
        kept.append(stem)

    return (np.asarray(pts3d, dtype=np.float64),
            np.asarray(pts2d, dtype=np.float64),
            kept, rej)


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

    x0 = np.concatenate([[fx0], rvec0, tvec0])
    # Soft bounds keep fx in a sane range so we don't drift into degeneracy.
    fx_lo = 0.5 * W / math.tan(0.5 * math.radians(120.0))
    fx_hi = 0.5 * W / math.tan(0.5 * math.radians(15.0))
    bounds = (
        np.array([fx_lo,  -np.pi*2, -np.pi*2, -np.pi*2,  -100, -100, -0.01]),
        np.array([fx_hi,   np.pi*2,  np.pi*2,  np.pi*2,   100,  100,  100]),
    )
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

    centroids = json.loads(centroids_path.read_text())

    img_size = _read_image_size(frames_dir)
    if img_size is None:
        return {"status": "skip-no-image", "episode": ep_dir_seg.name}
    W, H = img_size

    # Resolution order for K: --K_json override > per-episode camera.json
    # (saved by download_oxe_subset.py from the RLDS spec) > HFOV guess.
    K = None
    if K_override is not None:
        K = dict(K_override)
        K.setdefault("width", W)
        K.setdefault("height", H)
        K["source"] = "K_json"
    if K is None:
        cam_path = ep_dir_oxe / "camera.json"
        if cam_path.exists():
            try:
                cam_json = json.loads(cam_path.read_text())
                K = _K_from_camera_json(cam_json, W, H)
            except Exception:
                K = None
    K_known = K is not None
    if K is None:
        K = _default_K(W, H, args.hfov_deg)
        K["source"] = f"hfov={args.hfov_deg}deg"
    K_mat = _K_to_mat(K)

    pts3d, pts2d, kept_stems, rejected = _filter_frames(
        centroids, ee, W, H, args
    )

    result = {
        "episode": ep_dir_seg.name,
        "dataset": ds_name,
        "image_size": [W, H],
        "K": K,
        "K_known": K_known,
        "num_total": len(centroids),
        "num_kept_after_filter": len(kept_stems),
        "rejection_breakdown": _rejection_breakdown(rejected),
    }

    # If we don't have known intrinsics and the user asked for joint
    # estimation, optimise (fx, R, t) on the trajectory. This produces
    # ONE K per episode (the camera is fixed within an episode).
    pnp = None
    if args.estimate_intrinsics and not K_known:
        pnp = _solve_pnp_with_K_estimation(pts3d, pts2d, W, H, args)
        if pnp is not None and "K_estimated" in pnp:
            result["K"] = pnp["K_estimated"]
            K_mat = _K_to_mat(pnp["K_estimated"])
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

    out_path = ep_dir_seg / "pnp.json"
    out_path.write_text(json.dumps(result, indent=2))

    if args.viz:
        viz_dir = ep_dir_seg / "pnp_viz"
        viz_dir.mkdir(exist_ok=True)
        rvec = np.asarray(pnp["rvec"], dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(pnp["tvec"], dtype=np.float64).reshape(3, 1)
        inlier_set = set(int(i) for i in pnp["inlier_idx"])

        # one combined-plot showing all kept points on the first inlier frame
        proj_all = _project_pts(pts3d, rvec, tvec, K_mat)
        first_stem = kept_stems[pnp["inlier_idx"][0]] if pnp["inlier_idx"] else kept_stems[0]
        first_img = cv2.imread(str(frames_dir / f"{first_stem}.jpg"))
        if first_img is not None:
            cv2.imwrite(
                str(viz_dir / "_traj_overview.jpg"),
                _viz_reproj(first_img, pts2d, pts3d, inlier_set, None,
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
    p.add_argument("--min_conf", type=float, default=0.5)
    p.add_argument("--min_area", type=float, default=0.0002,
                   help="Minimum mask area as fraction of image (~0.02%%). "
                        "Floored at --min_area_px regardless.")
    p.add_argument("--min_area_px", type=int, default=16,
                   help="Absolute pixel floor for the gripper mask.")
    p.add_argument("--max_area", type=float, default=0.4)
    p.add_argument("--edge_margin", type=int, default=4,
                   help="Reject centroids within N px of any image border.")
    p.add_argument("--reject_fragmented", action="store_true", default=True,
                   help="Reject frames whose mask has >1 connected component.")

    # PnP
    p.add_argument("--reproj_thresh", type=float, default=8.0)
    p.add_argument("--min_inliers", type=int, default=5,
                   help="EPNP needs >=4; we require a few more for stability.")
    p.add_argument("--max_rmse", type=float, default=20.0)
    p.add_argument("--min_inlier_frac", type=float, default=0.4)

    p.add_argument("--estimate_intrinsics", action="store_true",
                   help="When K is not provided (no --K_json and no per-episode "
                        "camera.json), jointly estimate fx (=fy) alongside the "
                        "extrinsic. One K per episode. Requires scipy.")
    p.add_argument("--viz", action="store_true",
                   help="Save reprojection overlays per kept frame.")
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

        K_override = K_overrides.get(ds)
        results = []
        for ep_seg in episodes:
            ep_oxe = ds_oxe / ep_seg.name
            r = process_episode(ds, ep_oxe, ep_seg, K_override, args)
            results.append(r)
            tag = r.get("status", "?")
            extra = ""
            if "rmse_px" in r:
                extra = (f"  rmse={r['rmse_px']:.2f}px  "
                         f"inliers={r['num_inliers']}/{r['num_kept_after_filter']}")
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
