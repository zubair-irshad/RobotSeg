"""Refine per-episode cam2base poses with Panda/body silhouette alignment.

This starts from an existing PnP JSON, renders a Panda-only URDF over sampled
frames, optimizes a 6-DoF perturbation of T_cam2base against body masks, and
writes a new PnP-compatible JSON. It is intentionally derivative-free: the
OpenCV rasterizer is not differentiable, but 6 parameters are small enough for
coordinate search plus optional Powell polishing.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from cam2base_json import invert_se3  # noqa: E402
from pnp_oxe import EE_XYZ_DIMS, _fk_point_sequence, _K_to_mat  # noqa: E402
from score_urdf_silhouette import (  # noqa: E402
    _aggregate,
    _available_stems,
    _load_gripper_array,
    _load_joint_array,
    _load_json,
    _load_target_mask,
    _make_score_viz,
    _mask_metrics,
    _parse_prefixes,
    _write_pruned_urdf,
)
from urdf_robot_masker import URDFRobotMasker  # noqa: E402
from viz_cam2base_urdf import DEFAULT_URDF  # noqa: E402


def _T_to_rvec_tvec(T_cam2base: np.ndarray) -> tuple[list[float], list[float], list[list[float]]]:
    T_base2cam = invert_se3(np.asarray(T_cam2base, dtype=np.float64))
    rvec, _ = cv2.Rodrigues(T_base2cam[:3, :3])
    return (
        rvec.reshape(3).astype(float).tolist(),
        T_base2cam[:3, 3].astype(float).tolist(),
        T_base2cam.astype(float).tolist(),
    )


def _centroid(entry: Any) -> np.ndarray | None:
    if entry is None:
        return None
    if isinstance(entry, dict):
        val = entry.get("centroid")
    else:
        val = entry
    if val is None:
        return None
    arr = np.asarray(val, dtype=np.float64).reshape(-1)
    return arr[:2] if arr.size >= 2 else None


def _reprojection_metrics(
    args: argparse.Namespace,
    ep_oxe: Path,
    ep_seg: Path,
    pnp: dict[str, Any],
    K: dict[str, Any],
    rvec: list[float],
    tvec: list[float],
) -> dict[str, Any]:
    stems = pnp.get("inlier_stems") or pnp.get("kept_stems") or []
    if not stems:
        return {"num_reprojection_stems": 0}
    centroids_path = ep_seg / "001" / "centroids.json"
    traj_path = ep_oxe / "trajectory.npz"
    if not centroids_path.exists() or not traj_path.exists():
        return {"num_reprojection_stems": 0}
    centroids = json.loads(centroids_path.read_text())
    with np.load(traj_path, allow_pickle=True) as traj:
        n = max([int(s) for s in stems if str(s).isdigit()] + [0]) + 1
        source = pnp.get("point_source", "eef_xyz")
        if source == "eef_xyz":
            if "eef_xyz" in traj.files:
                points = np.asarray(traj["eef_xyz"], dtype=np.float64)
            else:
                state = np.asarray(traj["state"], dtype=np.float64)
                a, b = EE_XYZ_DIMS[args.dataset]
                points = state[:, a:b]
        else:
            fk_args = argparse.Namespace(
                pnp_point_source=source,
                urdf_path=args.urdf_path,
                mesh_dir=args.mesh_dir,
                urdf_backend=args.urdf_backend,
            )
            points, status = _fk_point_sequence(traj, n, fk_args)
            if points is None:
                return {"num_reprojection_stems": 0, "reprojection_skip": status}

    K_mat = _K_to_mat(K)
    rvec_arr = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec_arr = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    errors = []
    used = []
    for stem in stems:
        stem = str(stem)
        if not stem.isdigit():
            continue
        idx = int(stem)
        if idx >= len(points):
            continue
        c = _centroid(centroids.get(stem))
        if c is None:
            continue
        uv, _ = cv2.projectPoints(
            np.asarray(points[idx], dtype=np.float64).reshape(1, 3),
            rvec_arr,
            tvec_arr,
            K_mat,
            np.zeros(5),
        )
        err = float(np.linalg.norm(uv.reshape(2) - c))
        if np.isfinite(err):
            errors.append(err)
            used.append(stem)
    if not errors:
        return {"num_reprojection_stems": 0}
    arr = np.asarray(errors, dtype=np.float64)
    return {
        "num_reprojection_stems": len(used),
        "reprojection_stems": used,
        "rmse_px": float(np.sqrt(np.mean(arr ** 2))),
        "max_err_px": float(arr.max()),
        "mean_err_px": float(arr.mean()),
        "median_err_px": float(np.median(arr)),
    }


def _apply_delta(T0: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Pose update in T_cam2base convention.

    Rotation is applied to camera orientation in base coordinates while
    translation shifts the camera center in base coordinates. This avoids
    rotating the camera center around the robot base during local refinement.
    """
    T = np.asarray(T0, dtype=np.float64).copy()
    R_delta, _ = cv2.Rodrigues(np.asarray(x[:3], dtype=np.float64).reshape(3, 1))
    T[:3, :3] = R_delta @ T[:3, :3]
    T[:3, 3] = T[:3, 3] + np.asarray(x[3:6], dtype=np.float64)
    return T


def _load_samples(args: argparse.Namespace, ep_oxe: Path, ep_seg: Path,
                  K: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    H = int(K.get("height", 0))
    W = int(K.get("width", 0))
    if H <= 0 or W <= 0:
        return [], {"bad-image-size": 1}

    traj = np.load(ep_oxe / "trajectory.npz")
    if getattr(args, "mujoco_xml_path", None) is not None and "google_robot_all_qpos" in traj.files:
        joints = np.asarray(traj["google_robot_all_qpos"], dtype=np.float64)
        if joints.ndim == 1:
            joints = joints.reshape(1, -1)
    else:
        joints = _load_joint_array(traj, args.joint_key)
    gripper = _load_gripper_array(traj, len(joints), args.gripper_key)
    stems = _available_stems(ep_seg, args.mask_dirs)
    if args.frame_stride > 1:
        stems = stems[::args.frame_stride]
    if args.max_frames > 0:
        stems = stems[:args.max_frames]

    samples: list[dict[str, Any]] = []
    skips: dict[str, int] = {}

    def skip(reason: str) -> None:
        skips[reason] = skips.get(reason, 0) + 1

    for stem in stems:
        try:
            idx = int(stem)
        except ValueError:
            skip("non-integer-stem")
            continue
        if idx >= len(joints):
            skip("idx-out-of-range")
            continue
        target = _load_target_mask(
            ep_seg,
            stem,
            args.mask_dirs,
            (H, W),
            subtract_dirs=args.subtract_mask_dirs,
        )
        if target is None:
            skip("missing-target")
            continue
        area_frac = float(target.sum()) / float(H * W)
        if area_frac < args.min_target_area:
            skip("target-area-small")
            continue
        if area_frac > args.max_target_area:
            skip("target-area-large")
            continue
        samples.append({
            "stem": stem,
            "idx": idx,
            "target": target,
            "joint": joints[idx],
            "gripper": float(gripper[idx]),
        })
    return samples, skips


def _score_pose(args: argparse.Namespace, masker,
                samples: list[dict[str, Any]], K: dict[str, Any],
                T_cam2base: np.ndarray) -> tuple[float, list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    losses = []
    H, W = int(K["height"]), int(K["width"])
    for sample in samples:
        render = masker.render(
            K,
            T_cam2base,
            sample["joint"],
            gripper_position=sample["gripper"],
            image_hw=(H, W),
        )
        rec = _mask_metrics(sample["target"], render, args.distance_clip)
        rec["stem"] = sample["stem"]
        rows.append(rec)

        if rec["render_area"] <= 0 or rec["target_area"] <= 0:
            losses.append(args.distance_clip * 4.0)
            continue
        t2r = min(args.distance_clip, float(rec["target_to_render_px"]))
        r2t = min(args.distance_clip, float(rec["render_to_target_px_trim75"]))
        coverage = float(rec["target_coverage"])
        precision = float(rec["render_precision"])
        area_ratio = abs(math.log((float(rec["render_area"]) + 1.0)
                                  / (float(rec["target_area"]) + 1.0)))
        loss = (
            args.w_target_to_render * t2r
            + args.w_render_to_target * r2t
            + args.w_coverage * (1.0 - coverage) * args.distance_clip
            + args.w_precision * (1.0 - precision) * args.distance_clip
            + args.w_area * area_ratio * args.distance_clip
        )
        losses.append(float(loss))
    if not losses:
        return float("inf"), rows
    return float(np.mean(losses)), rows


def _clip_params(args: argparse.Namespace, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).copy()
    rot_bound = math.radians(args.max_rot_deg)
    trans_bound = float(args.max_trans_m)
    x[:3] = np.clip(x[:3], -rot_bound, rot_bound)
    x[3:] = np.clip(x[3:], -trans_bound, trans_bound)
    return x


def _fmt_metric(value: Any, digits: int = 3, default: str = "-") -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return default


def _optimize_pose(args: argparse.Namespace, objective) -> tuple[np.ndarray, float, int]:
    x = np.zeros(6, dtype=np.float64)
    best = float(objective(x))
    evals = 1
    step = np.array(
        [math.radians(args.rot_step_deg)] * 3 + [args.trans_step_m] * 3,
        dtype=np.float64,
    )
    min_step = np.array(
        [math.radians(args.min_rot_step_deg)] * 3 + [args.min_trans_step_m] * 3,
        dtype=np.float64,
    )

    while evals < args.max_evals and np.any(step >= min_step):
        improved_any = False
        for _ in range(args.sweeps_per_scale):
            improved = False
            for dim in range(6):
                for sign in (1.0, -1.0):
                    cand = x.copy()
                    cand[dim] += sign * step[dim]
                    cand = _clip_params(args, cand)
                    val = float(objective(cand))
                    evals += 1
                    if val + args.min_loss_improvement < best:
                        x, best = cand, val
                        improved = True
                        improved_any = True
                    if evals >= args.max_evals:
                        return x, best, evals
            if not improved:
                break
        if not improved_any:
            step *= 0.5

    if args.polish and evals < args.max_evals:
        try:
            from scipy.optimize import minimize

            rot_bound = math.radians(args.max_rot_deg)
            trans_bound = float(args.max_trans_m)
            bounds = [(-rot_bound, rot_bound)] * 3 + [(-trans_bound, trans_bound)] * 3
            remaining = max(1, args.max_evals - evals)
            sol = minimize(
                lambda z: float(objective(_clip_params(args, z))),
                x,
                method="Powell",
                bounds=bounds,
                options={"maxfev": remaining, "maxiter": remaining, "disp": False},
            )
            evals += int(getattr(sol, "nfev", 0))
            if float(sol.fun) + args.min_loss_improvement < best:
                x = _clip_params(args, sol.x)
                best = float(sol.fun)
        except Exception as exc:
            if args.verbose:
                print(f"  [warn] scipy polish skipped: {exc}")
    return x, best, evals


def _write_viz(args: argparse.Namespace, ep_oxe: Path, ep_seg: Path,
               masker, samples: list[dict[str, Any]],
               K: dict[str, Any], T_cam2base: np.ndarray,
               rows: list[dict[str, float]]) -> None:
    if not args.viz_dir_name:
        return
    out_dir = ep_seg / args.viz_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    render_mask_dir = out_dir / "render_masks"
    target_mask_dir = out_dir / "target_masks"
    render_mask_dir.mkdir(exist_ok=True)
    target_mask_dir.mkdir(exist_ok=True)
    H, W = int(K["height"]), int(K["width"])
    row_by_stem = {str(row["stem"]): row for row in rows}
    for sample in samples:
        stem = sample["stem"]
        img = cv2.imread(str(ep_oxe / "frames" / f"{stem}.jpg"))
        if img is None:
            continue
        render = masker.render(
            K,
            T_cam2base,
            sample["joint"],
            gripper_position=sample["gripper"],
            image_hw=(H, W),
        )
        rec = row_by_stem.get(stem)
        if rec is None:
            rec = _mask_metrics(sample["target"], render, args.distance_clip)
        cv2.imwrite(str(out_dir / f"{stem}.jpg"), _make_score_viz(img, sample["target"], render, rec))
        cv2.imwrite(str(render_mask_dir / f"{stem}.png"), render.astype(np.uint8) * 255)
        cv2.imwrite(str(target_mask_dir / f"{stem}.png"), sample["target"].astype(np.uint8) * 255)


def _process_episode(args: argparse.Namespace, masker,
                     ep_name: str) -> dict[str, Any]:
    ep_oxe = args.oxe_root / args.dataset / ep_name
    ep_seg = args.mask_root / args.dataset / ep_name
    pnp_path = ep_seg / args.pnp_json_name
    pnp = _load_json(pnp_path)
    if pnp is None or "T_cam2base" not in pnp or "K" not in pnp:
        return {"episode": ep_name, "status": "skip-no-pnp"}
    if not (ep_oxe / "trajectory.npz").exists():
        return {"episode": ep_name, "status": "skip-no-trajectory"}

    K = copy.deepcopy(pnp["K"])
    samples, skips = _load_samples(args, ep_oxe, ep_seg, K)
    if not samples:
        return {
            "episode": ep_name,
            "status": "skip-no-samples",
            "skip_counts": skips,
        }

    T0 = np.asarray(pnp["T_cam2base"], dtype=np.float64)
    init_loss, init_rows = _score_pose(args, masker, samples, K, T0)
    init_rvec = pnp.get("rvec")
    init_tvec = pnp.get("tvec")
    init_reproj = {}
    if init_rvec is not None and init_tvec is not None:
        init_reproj = _reprojection_metrics(args, ep_oxe, ep_seg, pnp, K, init_rvec, init_tvec)

    cache: dict[tuple[float, ...], float] = {}

    def objective(x: np.ndarray) -> float:
        key = tuple(np.round(_clip_params(args, x), 8).tolist())
        if key not in cache:
            T = _apply_delta(T0, np.asarray(key, dtype=np.float64))
            cache[key] = _score_pose(args, masker, samples, K, T)[0]
        return cache[key]

    x_best, best_loss, evals = _optimize_pose(args, objective)
    T_best = _apply_delta(T0, x_best)
    final_loss, final_rows = _score_pose(args, masker, samples, K, T_best)

    improved = final_loss + args.min_loss_improvement < init_loss
    T_out = T_best if improved else T0
    rows_out = final_rows if improved else init_rows
    loss_out = final_loss if improved else init_loss

    rvec, tvec, T_base2cam = _T_to_rvec_tvec(T_out)
    reproj = _reprojection_metrics(args, ep_oxe, ep_seg, pnp, K, rvec, tvec)
    out = copy.deepcopy(pnp)
    out["T_cam2base"] = np.asarray(T_out, dtype=float).tolist()
    out["T_base2cam"] = T_base2cam
    out["rvec"] = rvec
    out["tvec"] = tvec
    out["pose_source"] = "silhouette_refined" if improved else "silhouette_refine_kept_initial"
    out["refined_from_pnp_json"] = args.pnp_json_name
    out["refined_from_pnp_rmse_px"] = pnp.get("rmse_px")
    if "rmse_px" in reproj:
        out["rmse_px"] = reproj["rmse_px"]
        out["max_err_px"] = reproj["max_err_px"]
        out["refined_reprojection"] = reproj
    out["silhouette_refinement"] = {
        "status": "improved" if improved else "kept-initial",
        "init_loss": init_loss,
        "final_loss": final_loss,
        "written_loss": loss_out,
        "delta_rotvec_rad": x_best[:3].astype(float).tolist(),
        "delta_translation_m": x_best[3:].astype(float).tolist(),
        "num_samples": len(samples),
        "num_evals": evals,
        "mask_dirs": args.mask_dirs,
        "subtract_mask_dirs": args.subtract_mask_dirs,
        "metrics": _aggregate(rows_out),
        "init_metrics": _aggregate(init_rows),
        "final_metrics": _aggregate(final_rows),
        "init_reprojection": init_reproj,
        "written_reprojection": reproj,
    }
    out_path = ep_seg / args.out_pnp_json_name
    out_path.write_text(json.dumps(out, indent=2))
    _write_viz(args, ep_oxe, ep_seg, masker, samples, K, T_out, rows_out)

    metrics = out["silhouette_refinement"]["metrics"]
    return {
        "episode": ep_name,
        "status": "ok",
        "improved": improved,
        "init_loss": init_loss,
        "final_loss": final_loss,
        "written_loss": loss_out,
        "num_samples": len(samples),
        "num_evals": evals,
        "iou_median": metrics.get("iou_median"),
        "coverage_median": metrics.get("target_coverage_median"),
        "t2r_median": metrics.get("target_to_render_px_median"),
        "out": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oxe_root", type=Path, default=Path("data/oxe_subset"))
    parser.add_argument("--mask_root", type=Path, default=Path("data/oxe_subset_seg"))
    parser.add_argument("--dataset", default="droid")
    parser.add_argument("--episodes", nargs="+", default=None)
    parser.add_argument("--pnp_json_name", default="pnp.json")
    parser.add_argument("--out_pnp_json_name", default="pnp_silhouette_refined.json")
    parser.add_argument("--urdf_path", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--mesh_dir", type=Path, default=None)
    parser.add_argument("--urdf_backend", choices=["simple", "yourdfpy", "auto"], default="yourdfpy")
    parser.add_argument(
        "--mujoco_xml_path",
        type=Path,
        default=None,
        help="Render silhouettes with this MuJoCo XML and trajectory['google_robot_all_qpos'] instead of URDF joints.",
    )
    parser.add_argument(
        "--mujoco_postprocess",
        choices=["none", "auge", "flip_x", "flip_y", "rot180"],
        default="none",
    )
    parser.add_argument("--mask_dirs", nargs="+", default=["000"])
    parser.add_argument("--subtract_mask_dirs", nargs="+", default=[])
    parser.add_argument("--render_link_prefixes", default="panda_link")
    parser.add_argument("--exclude_render_link_prefixes", default=None)
    parser.add_argument("--joint_key", default=None)
    parser.add_argument("--gripper_key", default=None)
    parser.add_argument("--frame_stride", type=int, default=8)
    parser.add_argument("--max_frames", type=int, default=32)
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--dilate_px", type=int, default=2)
    parser.add_argument("--distance_clip", type=float, default=30.0)
    parser.add_argument("--min_target_area", type=float, default=0.002)
    parser.add_argument("--max_target_area", type=float, default=0.60)
    parser.add_argument("--max_rot_deg", type=float, default=80.0)
    parser.add_argument("--max_trans_m", type=float, default=0.75)
    parser.add_argument("--rot_step_deg", type=float, default=8.0)
    parser.add_argument("--trans_step_m", type=float, default=0.04)
    parser.add_argument("--min_rot_step_deg", type=float, default=0.25)
    parser.add_argument("--min_trans_step_m", type=float, default=0.002)
    parser.add_argument("--sweeps_per_scale", type=int, default=4)
    parser.add_argument("--max_evals", type=int, default=240)
    parser.add_argument("--min_loss_improvement", type=float, default=1e-4)
    parser.add_argument("--polish", action="store_true", default=True)
    parser.add_argument("--no_polish", dest="polish", action="store_false")
    parser.add_argument("--w_target_to_render", type=float, default=1.0)
    parser.add_argument("--w_render_to_target", type=float, default=0.35)
    parser.add_argument("--w_coverage", type=float, default=0.45)
    parser.add_argument("--w_precision", type=float, default=0.25)
    parser.add_argument("--w_area", type=float, default=0.05)
    parser.add_argument("--viz_dir_name", default=None)
    parser.add_argument("--summary_json", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.mujoco_xml_path is not None:
        from mujoco_google_robot_renderer import MuJoCoGoogleRobotRenderer  # noqa: E402

        masker = MuJoCoGoogleRobotRenderer(
            args.mujoco_xml_path,
            verbose=True,
            postprocess=args.mujoco_postprocess,
        )
    else:
        include_prefixes = _parse_prefixes(args.render_link_prefixes)
        exclude_prefixes = _parse_prefixes(args.exclude_render_link_prefixes)
        urdf_path = args.urdf_path
        if include_prefixes or exclude_prefixes:
            urdf_path = _write_pruned_urdf(args.urdf_path, include_prefixes, exclude_prefixes)
            print(f"[urdf_refine] pruned URDF: {urdf_path}")

        masker = URDFRobotMasker(
            urdf_path,
            mesh_dir=args.mesh_dir,
            backend=args.urdf_backend,
            downsample=args.downsample,
            dilate_px=args.dilate_px,
            verbose=True,
        )

    ds_seg = args.mask_root / args.dataset
    episodes = args.episodes or sorted(
        p.name for p in ds_seg.iterdir()
        if p.is_dir() and p.name.startswith("episode_")
    )
    results = []
    try:
        for ep in episodes:
            result = _process_episode(args, masker, ep)
            results.append(result)
            if result.get("status") == "ok":
                marker = "improved" if result["improved"] else "kept"
                print(
                    f"[{args.dataset}/{ep}] {marker} "
                    f"loss {_fmt_metric(result.get('init_loss'))}->{_fmt_metric(result.get('written_loss'))} "
                    f"iou={_fmt_metric(result.get('iou_median'))} "
                    f"cov={_fmt_metric(result.get('coverage_median'))} "
                    f"t2r={_fmt_metric(result.get('t2r_median'), 2)}px "
                    f"evals={result.get('num_evals')} n={result.get('num_samples')}"
                )
            else:
                print(f"[{args.dataset}/{ep}] {result.get('status')}")
    finally:
        close = getattr(masker, "close", None)
        if callable(close):
            close()

    out_path = args.summary_json
    if out_path is None:
        out_path = ds_seg / f"silhouette_refine_summary_{Path(args.out_pnp_json_name).stem}.json"
    out_path.write_text(json.dumps({"dataset": args.dataset, "results": results}, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
