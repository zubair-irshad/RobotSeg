"""Score rendered URDF silhouettes against RobotSeg body/gripper masks.

This is a diagnostic for camera extrinsics: it renders the robot URDF using
the per-episode PnP K/T and trajectory joint positions, then compares the
rendered mask to segmentation masks such as 000=arm and 001=gripper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cam2base_json import episode_lookup_keys, find_T_cam2base  # noqa: E402
from urdf_robot_masker import URDFRobotMasker  # noqa: E402
from viz_cam2base_urdf import DEFAULT_URDF  # noqa: E402


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open() as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _load_joint_array(traj: np.lib.npyio.NpzFile, key: str | None) -> np.ndarray:
    candidates = []
    if key:
        candidates.append(key)
    candidates += ["joint_position", "joint_positions", "joints", "q", "arm_joints"]
    for name in candidates:
        if name in traj.files:
            arr = np.asarray(traj[name], dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] >= 7:
                return arr[:, :7]
    raise ValueError("trajectory.npz does not contain a usable 7-DoF joint array")


def _load_gripper_array(traj: np.lib.npyio.NpzFile, n: int, key: str | None) -> np.ndarray:
    candidates = []
    if key:
        candidates.append(key)
    candidates += ["gripper", "gripper_position", "finger_joint"]
    for name in candidates:
        if name in traj.files:
            arr = np.asarray(traj[name], dtype=np.float64).reshape(-1)
            if len(arr) >= n:
                return arr[:n]
    return np.zeros(n, dtype=np.float64)


def _mask_path(root: Path, stem: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg"):
        path = root / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def _is_primary_mask_image(path: Path) -> bool:
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        return False
    stem = path.stem
    if stem.startswith("_"):
        return False
    return not stem.endswith(("_overlay", "_prob", "_centroid"))


def _read_mask(path: Path, shape_hw: tuple[int, int]) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = img > 0
    if mask.shape[:2] != shape_hw:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (shape_hw[1], shape_hw[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return mask


def _load_target_mask(ep_seg: Path, stem: str, mask_dirs: list[str],
                      shape_hw: tuple[int, int],
                      subtract_dirs: list[str] | None = None) -> np.ndarray | None:
    out = np.zeros(shape_hw, dtype=bool)
    found = False
    for name in mask_dirs:
        path = _mask_path(ep_seg / name, stem)
        if path is None:
            continue
        mask = _read_mask(path, shape_hw)
        if mask is None:
            continue
        out |= mask
        found = True
    if found and subtract_dirs:
        subtract = np.zeros(shape_hw, dtype=bool)
        for name in subtract_dirs:
            path = _mask_path(ep_seg / name, stem)
            if path is None:
                continue
            mask = _read_mask(path, shape_hw)
            if mask is None:
                continue
            subtract |= mask
        out &= ~subtract
    return out if found else None


def _available_stems(ep_seg: Path, mask_dirs: list[str]) -> list[str]:
    stems = set()
    for name in mask_dirs:
        root = ep_seg / name
        if not root.exists():
            continue
        for p in root.iterdir():
            if _is_primary_mask_image(p):
                stems.add(p.stem)
    return sorted(stems)


def _mask_metrics(target: np.ndarray, render: np.ndarray,
                  distance_clip: float) -> dict[str, float]:
    inter = np.logical_and(target, render).sum()
    union = np.logical_or(target, render).sum()
    target_area = int(target.sum())
    render_area = int(render.sum())
    if target_area == 0 or render_area == 0:
        return {
            "iou": 0.0,
            "target_coverage": 0.0,
            "render_precision": 0.0,
            "target_to_render_px": float("inf"),
            "render_to_target_px_trim75": float("inf"),
            "target_area": float(target_area),
            "render_area": float(render_area),
        }

    inv_render = (~render).astype(np.uint8)
    inv_target = (~target).astype(np.uint8)
    dt_render = cv2.distanceTransform(inv_render, cv2.DIST_L2, 3)
    dt_target = cv2.distanceTransform(inv_target, cv2.DIST_L2, 3)
    t2r = np.minimum(dt_render[target], distance_clip).mean()
    # Visible segmentation is often a subset of the unoccluded URDF render.
    # Trim render->target distances so scene occluders do not dominate.
    r2t_vals = np.minimum(dt_target[render], distance_clip)
    keep = max(1, int(0.75 * len(r2t_vals)))
    r2t = np.partition(r2t_vals, keep - 1)[:keep].mean()
    return {
        "iou": float(inter / max(1, union)),
        "target_coverage": float(inter / max(1, target_area)),
        "render_precision": float(inter / max(1, render_area)),
        "target_to_render_px": float(t2r),
        "render_to_target_px_trim75": float(r2t),
        "target_area": float(target_area),
        "render_area": float(render_area),
    }


def _overlay_mask(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    color_bgr: tuple[int, int, int],
    alpha: float = 0.55,
) -> np.ndarray:
    out = image_bgr.copy().astype(np.float32)
    color = np.asarray(color_bgr, dtype=np.float32)
    sel = mask.astype(bool)
    out[sel] = (1.0 - alpha) * out[sel] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


def _make_score_viz(image_bgr: np.ndarray, target: np.ndarray,
                    render: np.ndarray, rec: dict[str, float]) -> np.ndarray:
    target_panel = _overlay_mask(image_bgr, target, (0, 220, 0), alpha=0.55)
    render_panel = _overlay_mask(image_bgr, render, (229, 132, 11), alpha=0.60)

    diff_panel = image_bgr.copy().astype(np.float32)
    both = target & render
    target_only = target & ~render
    render_only = render & ~target
    diff_panel[both] = 0.35 * diff_panel[both] + 0.65 * np.array([0, 220, 220], dtype=np.float32)
    diff_panel[target_only] = 0.30 * diff_panel[target_only] + 0.70 * np.array([0, 220, 0], dtype=np.float32)
    diff_panel[render_only] = 0.30 * diff_panel[render_only] + 0.70 * np.array([0, 0, 255], dtype=np.float32)
    diff_panel = np.clip(diff_panel, 0, 255).astype(np.uint8)

    panels = [target_panel, render_panel, diff_panel]
    labels = [
        "body mask",
        "panda render",
        "diff green=target red=render yellow=overlap",
    ]
    for panel, label in zip(panels, labels):
        cv2.putText(
            panel,
            label,
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            label,
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

    viz = np.hstack(panels)
    summary = (
        f"iou={rec['iou']:.3f} coverage={rec['target_coverage']:.3f} "
        f"t2r={rec['target_to_render_px']:.2f}px"
    )
    cv2.putText(
        viz,
        summary,
        (8, viz.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        viz,
        summary,
        (8, viz.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return viz


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {"n": float(len(rows))}
    if not rows:
        return out
    keys = [
        "iou",
        "target_coverage",
        "render_precision",
        "target_to_render_px",
        "render_to_target_px_trim75",
        "target_area",
        "render_area",
    ]
    for key in keys:
        vals = np.asarray([r[key] for r in rows if np.isfinite(r[key])], dtype=np.float64)
        if len(vals) == 0:
            continue
        out[f"{key}_mean"] = float(vals.mean())
        out[f"{key}_median"] = float(np.median(vals))
    out["empty_render_frames"] = float(
        sum(1 for r in rows if float(r.get("render_area", 0.0)) <= 0.0)
    )
    return out


def _parse_prefixes(text: str | None) -> tuple[str, ...] | None:
    if text is None:
        return None
    vals = tuple(x.strip() for x in text.split(",") if x.strip())
    return vals or None


def _pruned_urdf_path(
    urdf_path: Path,
    include_prefixes: tuple[str, ...] | None,
    exclude_prefixes: tuple[str, ...] | None,
) -> Path | None:
    if not include_prefixes and not exclude_prefixes:
        return None
    key = json.dumps(
        {"include": include_prefixes or (), "exclude": exclude_prefixes or ()},
        sort_keys=True,
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return urdf_path.parent / f".{urdf_path.stem}.score_pruned_{digest}.urdf"


def _link_allowed_for_prune(
    name: str,
    include_prefixes: tuple[str, ...] | None,
    exclude_prefixes: tuple[str, ...] | None,
) -> bool:
    if include_prefixes and not any(name.startswith(prefix) for prefix in include_prefixes):
        return False
    if exclude_prefixes and any(name.startswith(prefix) for prefix in exclude_prefixes):
        return False
    return True


def _write_pruned_urdf(
    urdf_path: Path,
    include_prefixes: tuple[str, ...] | None,
    exclude_prefixes: tuple[str, ...] | None,
) -> Path:
    out_path = _pruned_urdf_path(urdf_path, include_prefixes, exclude_prefixes)
    if out_path is None:
        return urdf_path
    if out_path.exists() and out_path.stat().st_mtime >= urdf_path.stat().st_mtime:
        return out_path

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    keep_links: set[str] = set()
    for link in root.findall("link"):
        name = link.attrib.get("name", "")
        if _link_allowed_for_prune(name, include_prefixes, exclude_prefixes):
            keep_links.add(name)
        else:
            root.remove(link)

    for joint in list(root.findall("joint")):
        parent = joint.find("parent")
        child = joint.find("child")
        parent_name = parent.attrib.get("link", "") if parent is not None else ""
        child_name = child.attrib.get("link", "") if child is not None else ""
        if parent_name not in keep_links or child_name not in keep_links:
            root.remove(joint)

    for tag in ("transmission", "gazebo"):
        for elem in list(root.findall(tag)):
            root.remove(elem)

    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


def _load_extrinsics_override(args: argparse.Namespace, ep_name: str) -> tuple[np.ndarray | None, str | None]:
    if args.extrinsics_data is None:
        return None, None
    keys = episode_lookup_keys(args.oxe_root, args.dataset, ep_name)
    return find_T_cam2base(
        args.extrinsics_data,
        keys,
        preferred_fields=args.extrinsics_preferred_fields,
        sixd_mode=args.extrinsics_sixd_mode,
    )


def process_episode(args: argparse.Namespace, masker: URDFRobotMasker,
                    ep_name: str) -> dict[str, Any]:
    ep_oxe = args.oxe_root / args.dataset / ep_name
    ep_seg = args.mask_root / args.dataset / ep_name
    pnp = _load_json(ep_seg / args.pnp_json_name)
    if pnp is None or "T_cam2base" not in pnp:
        return {"episode": ep_name, "status": "skip-no-pnp"}
    traj_path = ep_oxe / "trajectory.npz"
    if not traj_path.exists():
        return {"episode": ep_name, "status": "skip-no-trajectory"}

    traj = np.load(traj_path)
    joints = _load_joint_array(traj, args.joint_key)
    gripper = _load_gripper_array(traj, len(joints), args.gripper_key)
    K = pnp["K"]
    T_cam2base = np.asarray(pnp["T_cam2base"], dtype=np.float64)
    T_source = f"{args.pnp_json_name}:T_cam2base"
    T_override, T_override_source = _load_extrinsics_override(args, ep_name)
    if T_override is not None:
        T_cam2base = np.asarray(T_override, dtype=np.float64)
        T_source = f"{args.extrinsics_json}:{T_override_source}"
    elif args.extrinsics_data is not None:
        return {"episode": ep_name, "status": "skip-no-extrinsics"}
    H = int(K.get("height", pnp.get("image_size", [0, 0])[1]))
    W = int(K.get("width", pnp.get("image_size", [0, 0])[0]))
    if H <= 0 or W <= 0:
        return {"episode": ep_name, "status": "skip-bad-image-size"}

    mask_dirs = args.mask_dirs
    if args.auto_robot_mask and (ep_seg / "002").exists():
        mask_dirs = ["002"]
    stems = _available_stems(ep_seg, mask_dirs)
    num_candidate_stems = len(stems)
    if args.frame_stride > 1:
        stems = stems[::args.frame_stride]
    if args.max_frames > 0:
        stems = stems[:args.max_frames]

    rows = []
    skip_counts: dict[str, int] = {}
    viz_dir = ep_seg / args.viz_dir_name if args.viz_dir_name else None
    if viz_dir is not None:
        viz_dir.mkdir(parents=True, exist_ok=True)

    def skip(reason: str) -> None:
        skip_counts[reason] = skip_counts.get(reason, 0) + 1

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
            ep_seg, stem, mask_dirs, (H, W),
            subtract_dirs=args.subtract_mask_dirs,
        )
        if target is None:
            skip("missing-target-mask")
            continue
        area_frac = float(target.sum()) / float(H * W)
        if area_frac < args.min_target_area:
            skip("target-area-small")
            continue
        if area_frac > args.max_target_area:
            skip("target-area-large")
            continue
        render = masker.render(
            K,
            T_cam2base,
            joints[idx],
            gripper_position=float(gripper[idx]),
            image_hw=(H, W),
            include_link_prefixes=args.render_include_prefixes,
            exclude_link_prefixes=args.render_exclude_prefixes,
        )
        rec = _mask_metrics(target, render, args.distance_clip)
        rec["stem"] = stem
        rec["area_frac"] = area_frac
        rows.append(rec)
        if viz_dir is not None:
            image = cv2.imread(str(ep_oxe / "frames" / f"{stem}.jpg"))
            if image is not None:
                cv2.imwrite(str(viz_dir / f"{stem}.jpg"), _make_score_viz(image, target, render, rec))

    agg = _aggregate(rows)
    empty_render_frames = int(agg.get("empty_render_frames", 0))
    if not rows:
        status = "bad-no-scored-frames"
    elif empty_render_frames == len(rows):
        status = "bad-empty-render"
    else:
        status = "ok"
    return {
        "episode": ep_name,
        "status": status,
        "pnp_status": pnp.get("status"),
        "K_source": K.get("source", "unknown") if isinstance(K, dict) else "unknown",
        "pnp_rmse_px": pnp.get("rmse_px"),
        "pnp_inliers": pnp.get("num_inliers"),
        "T_source": T_source,
        "mask_dirs": mask_dirs,
        "subtract_mask_dirs": args.subtract_mask_dirs,
        "num_candidate_stems": num_candidate_stems,
        "num_stems_after_sampling": len(stems),
        "skip_counts": skip_counts,
        "render_include_prefixes": list(args.render_include_prefixes or ()),
        "render_exclude_prefixes": list(args.render_exclude_prefixes or ()),
        "metrics": agg,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oxe_root", type=Path, default=Path("data/oxe_subset"))
    parser.add_argument("--mask_root", type=Path, default=Path("data/oxe_subset_seg"))
    parser.add_argument("--dataset", default="droid")
    parser.add_argument("--episodes", nargs="+", default=None)
    parser.add_argument("--pnp_json_name", default="pnp.json")
    parser.add_argument("--urdf_path", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--mesh_dir", type=Path, default=None)
    parser.add_argument(
        "--urdf_backend",
        choices=["simple", "yourdfpy", "auto"],
        default="simple",
        help="URDF renderer backend. Use yourdfpy to match the visual-scene path.",
    )
    parser.add_argument("--mask_dirs", nargs="+", default=["000", "001"],
                        help="Segmentation folders to union. Defaults to arm+gripper.")
    parser.add_argument("--subtract_mask_dirs", nargs="+", default=[],
                        help="Segmentation folders to subtract from the target mask.")
    parser.add_argument("--auto_robot_mask", action="store_true",
                        help="Use folder 002 if present, otherwise --mask_dirs.")
    parser.add_argument(
        "--render_link_prefixes",
        default=None,
        help="Comma-separated URDF link prefixes to render, e.g. panda_link for arm/body only.",
    )
    parser.add_argument(
        "--exclude_render_link_prefixes",
        default=None,
        help="Comma-separated URDF link prefixes to suppress from the render.",
    )
    parser.add_argument(
        "--prune_urdf_for_filtered_render",
        action="store_true",
        default=True,
        help="For --render_link_prefixes/--exclude_render_link_prefixes, write a pruned "
             "URDF and render it through the normal full-scene backend. This is more "
             "robust than per-link mesh extraction in yourdfpy.",
    )
    parser.add_argument(
        "--no_prune_urdf_for_filtered_render",
        dest="prune_urdf_for_filtered_render",
        action="store_false",
    )
    parser.add_argument("--extrinsics_json", type=Path, default=None)
    parser.add_argument("--extrinsics_sixd_mode", default="rpy_cam2base")
    parser.add_argument("--extrinsics_preferred_fields", nargs="*", default=None)
    parser.add_argument("--joint_key", default=None)
    parser.add_argument("--gripper_key", default=None)
    parser.add_argument("--frame_stride", type=int, default=4)
    parser.add_argument("--max_frames", type=int, default=80)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--dilate_px", type=int, default=2)
    parser.add_argument("--distance_clip", type=float, default=30.0)
    parser.add_argument("--min_target_area", type=float, default=0.002)
    parser.add_argument("--max_target_area", type=float, default=0.60)
    parser.add_argument("--out_json", type=Path, default=None)
    parser.add_argument(
        "--viz_dir_name",
        default=None,
        help="Optional per-episode directory name for body-mask/render/diff visualizations.",
    )
    args = parser.parse_args()
    args.render_include_prefixes = _parse_prefixes(args.render_link_prefixes)
    args.render_exclude_prefixes = _parse_prefixes(args.exclude_render_link_prefixes)
    args.extrinsics_data = _load_json(args.extrinsics_json) if args.extrinsics_json else None

    ds_seg = args.mask_root / args.dataset
    episodes = args.episodes or sorted(
        p.name for p in ds_seg.iterdir()
        if p.is_dir() and p.name.startswith("episode_")
    )

    urdf_path = args.urdf_path
    render_include_prefixes = args.render_include_prefixes
    render_exclude_prefixes = args.render_exclude_prefixes
    if args.prune_urdf_for_filtered_render and (render_include_prefixes or render_exclude_prefixes):
        urdf_path = _write_pruned_urdf(
            args.urdf_path,
            render_include_prefixes,
            render_exclude_prefixes,
        )
        print(f"[urdf_render] pruned URDF for scoring: {urdf_path}")
        render_include_prefixes = None
        render_exclude_prefixes = None
    args.render_include_prefixes = render_include_prefixes
    args.render_exclude_prefixes = render_exclude_prefixes

    masker = URDFRobotMasker(
        urdf_path,
        mesh_dir=args.mesh_dir,
        backend=args.urdf_backend,
        downsample=args.downsample,
        dilate_px=args.dilate_px,
        verbose=True,
    )

    results = []
    for ep in episodes:
        result = process_episode(args, masker, ep)
        results.append(result)
        metrics = result.get("metrics", {})
        if result.get("status") == "ok":
            print(
                f"[{args.dataset}/{ep}] ok "
                f"T={result.get('T_source')} "
                f"pnp_rmse={result.get('pnp_rmse_px')} "
                f"iou_med={metrics.get('iou_median', 0.0):.3f} "
                f"coverage_med={metrics.get('target_coverage_median', 0.0):.3f} "
                f"t2r_med={metrics.get('target_to_render_px_median', 0.0):.2f}px "
                f"render_area_med={metrics.get('render_area_median', 0.0):.0f} "
                f"n={int(metrics.get('n', 0))}"
            )
        else:
            extra = ""
            if result.get("status") == "bad-no-scored-frames":
                extra = (
                    f" candidates={result.get('num_candidate_stems', 0)} "
                    f"sampled={result.get('num_stems_after_sampling', 0)} "
                    f"skips={result.get('skip_counts', {})}"
                )
            elif result.get("status") == "bad-empty-render":
                metrics = result.get("metrics", {})
                extra = (
                    f" n={int(metrics.get('n', 0))} "
                    f"render_area_med={metrics.get('render_area_median', 0.0):.0f}"
                )
            print(f"[{args.dataset}/{ep}] {result.get('status')}{extra}")

    out_path = args.out_json
    if out_path is None:
        stem = Path(args.pnp_json_name).stem
        out_path = ds_seg / f"urdf_silhouette_scores_{stem}.json"
    out_path.write_text(json.dumps({"dataset": args.dataset, "results": results}, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
