"""Visualize exactly which frames were used by PnP.

For each kept frame in pnp.json, writes separate raw/mask/URDF/reprojection
images and a side-by-side audit panel. This is intended to answer:
  - which segmentation frames survived filtering?
  - which of those were RANSAC inliers?
  - where is the observed gripper centroid vs projected 3D TCP?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from cam2base_json import (  # noqa: E402
    SIXD_MODES,
    episode_lookup_keys,
    find_T_cam2base,
    load_json,
)
from pnp_oxe import EE_XYZ_DIMS, _fk_point_sequence, _K_to_mat  # noqa: E402
from urdf_robot_masker import URDFRobotMasker  # noqa: E402
from viz_cam2base_urdf import DEFAULT_URDF  # noqa: E402


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


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
        p = ep_seg / root / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _mask_overlay(image: np.ndarray, body: np.ndarray, grip: np.ndarray) -> np.ndarray:
    out = image.copy().astype(np.float32)
    if np.any(body):
        out[body] = 0.45 * out[body] + 0.55 * np.array([0, 220, 0], dtype=np.float32)
    if np.any(grip):
        out[grip] = 0.35 * out[grip] + 0.65 * np.array([0, 0, 255], dtype=np.float32)
    return out.astype(np.uint8)


def _centroid_from_entry(entry) -> list[float] | None:
    if isinstance(entry, dict):
        return entry.get("centroid")
    if isinstance(entry, list):
        return entry
    return None


def _load_points(dataset: str, traj, n: int, args, point_source: str) -> np.ndarray:
    if point_source == "eef_xyz":
        if "eef_xyz" in traj.files:
            return np.asarray(traj["eef_xyz"], dtype=np.float64)
        state = np.asarray(traj["state"], dtype=np.float64)
        a, b = EE_XYZ_DIMS[dataset]
        return state[:, a:b]
    joints = _load_joints(traj)
    if joints is None:
        raise RuntimeError(f"could not load joint positions for {point_source}")
    fk_args = argparse.Namespace(
        pnp_point_source=point_source,
        urdf_path=args.urdf_path,
        mesh_dir=args.mesh_dir,
        urdf_backend=args.urdf_backend,
    )
    pts, status = _fk_point_sequence(traj, len(joints), fk_args)
    if pts is None:
        raise RuntimeError(f"could not load PnP point source {point_source}: {status}")
    return pts


def _load_joints(traj) -> np.ndarray | None:
    for key in ("joint_position", "joint_positions", "joints", "q", "arm_joints"):
        if key in traj.files:
            arr = np.asarray(traj[key], dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.ndim == 2 and arr.shape[1] >= 7:
                return arr[:, :7]
    return None


def _load_gripper(traj, n: int) -> np.ndarray:
    for key in ("gripper", "gripper_position", "finger_joint"):
        if key in traj.files:
            arr = np.asarray(traj[key], dtype=np.float64).reshape(-1)
            if len(arr) >= n:
                return arr[:n]
    return np.zeros(n, dtype=np.float64)


def _load_mujoco_qpos(traj) -> np.ndarray | None:
    if "google_robot_all_qpos" in traj.files:
        arr = np.asarray(traj["google_robot_all_qpos"], dtype=np.float64)
    elif "all_qpos" in traj.files:
        arr = np.asarray(traj["all_qpos"], dtype=np.float64)
    else:
        return None
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr if arr.ndim == 2 else None


def _serial_from_pnp(pnp: dict) -> str | None:
    source = pnp.get("K", {}).get("source") if isinstance(pnp.get("K"), dict) else None
    if not isinstance(source, str):
        return None
    marker = "serial="
    if marker not in source:
        return None
    return source.split(marker, 1)[1].split()[0]


def _episode_id_from_pnp(pnp: dict) -> str | None:
    source = pnp.get("K", {}).get("source") if isinstance(pnp.get("K"), dict) else None
    if not isinstance(source, str):
        return None
    marker = "episode_id="
    if marker not in source:
        return None
    return source.split(marker, 1)[1].split()[0]


def _load_reference_T(dataset: str, ep: str, pnp: dict, args) -> tuple[np.ndarray | None, str | None]:
    if args.extrinsics_json is None:
        return None, None
    data = load_json(args.extrinsics_json)
    keys = episode_lookup_keys(args.oxe_root, dataset, ep)
    raw_id = _episode_id_from_pnp(pnp)
    if raw_id and raw_id not in keys:
        keys += [raw_id, f"{dataset}/{raw_id}"]
    preferred = []
    serial = args.camera_serial or _serial_from_pnp(pnp)
    if serial:
        preferred.append(serial)
    T_ref, source = find_T_cam2base(
        data,
        keys,
        preferred_fields=preferred,
        sixd_mode=args.extrinsics_sixd_mode,
    )
    return T_ref, source


def _project(point3d: np.ndarray, K: dict, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    uv, _ = cv2.projectPoints(
        np.asarray(point3d, dtype=np.float64).reshape(1, 3),
        rvec,
        tvec,
        _K_to_mat(K),
        np.zeros(5),
    )
    return uv.reshape(2)


def _draw_header(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _resize_to_h(img: np.ndarray, h: int) -> np.ndarray:
    scale = h / float(img.shape[0])
    return cv2.resize(img, (max(1, int(round(img.shape[1] * scale))), h))


def _write_contact_sheet(paths: list[Path], out_path: Path, tile_w: int = 320, cols: int = 2):
    tiles = []
    for p in paths[:24]:
        img = cv2.imread(str(p))
        if img is None:
            continue
        scale = tile_w / float(img.shape[1])
        tiles.append(cv2.resize(img, (tile_w, max(1, int(round(img.shape[0] * scale))))))
    if not tiles:
        return
    tile_h = max(t.shape[0] for t in tiles)
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i:i + cols]
        padded = []
        for tile in row:
            if tile.shape[0] < tile_h:
                pad = np.zeros((tile_h - tile.shape[0], tile.shape[1], 3), dtype=np.uint8)
                tile = np.vstack([tile, pad])
            padded.append(tile)
        while len(padded) < cols:
            padded.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
        rows.append(np.hstack(padded))
    cv2.imwrite(str(out_path), np.vstack(rows))


def process_episode(dataset: str, ep: str, args, masker: URDFRobotMasker | None) -> dict:
    ep_oxe = args.oxe_root / dataset / ep
    ep_seg = args.mask_root / dataset / ep
    pnp_path = ep_seg / args.pnp_json_name
    if not pnp_path.exists():
        return {"episode": ep, "status": "skip-missing-pnp"}
    pnp = _read_json(pnp_path)
    if args.skip_bad_pnp and pnp.get("status") != "ok":
        return {"episode": ep, "status": f"skip-pnp-{pnp.get('status')}"}
    kept = pnp.get("kept_stems") or pnp.get("inlier_stems") or []
    inliers = set(pnp.get("inlier_stems") or [])
    if not kept:
        return {"episode": ep, "status": "skip-no-kept-stems"}

    centroids = _read_json(ep_seg / "001" / "centroids.json")
    K = pnp["K"]
    rvec = np.asarray(pnp["rvec"], dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(pnp["tvec"], dtype=np.float64).reshape(3, 1)
    T_cam2base = np.asarray(pnp["T_cam2base"], dtype=np.float64)
    T_ref_cam2base, T_ref_source = _load_reference_T(dataset, ep, pnp, args)

    with np.load(ep_oxe / "trajectory.npz", allow_pickle=True) as traj:
        points = _load_points(dataset, traj, len(centroids), args, pnp.get("point_source", "eef_xyz"))
        joints = _load_joints(traj)
        mujoco_qpos = _load_mujoco_qpos(traj) if args.mujoco_xml_path is not None else None
        render_state = mujoco_qpos if mujoco_qpos is not None else joints
        gripper = _load_gripper(traj, len(joints)) if joints is not None else None

        out_root = ep_seg / args.out_dir_name
        have_urdf = masker is not None and render_state is not None
        dir_names = ["raw", "masks", "reprojection", "panel"]
        if have_urdf:
            dir_names += ["urdf_pnp", "urdf_multiview"]
        dirs = {name: out_root / name for name in dir_names}
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        panel_paths = []
        for ordinal, stem in enumerate(kept[: args.max_frames if args.max_frames > 0 else None]):
            img_path = ep_oxe / "frames" / f"{stem}.jpg"
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            h, w = image.shape[:2]
            shape_hw = (h, w)
            body = _read_mask(_mask_path(ep_seg, "000", stem) or _mask_path(ep_seg, "002", stem), shape_hw)
            grip = _read_mask(_mask_path(ep_seg, "001", stem), shape_hw)
            mask_img = _mask_overlay(image, body, grip)

            entry = centroids.get(stem)
            c = _centroid_from_entry(entry)
            idx = int(stem)
            proj = _project(points[idx], K, rvec, tvec) if idx < len(points) else None
            is_inlier = stem in inliers
            status = "INLIER" if is_inlier else "RANSAC_OUTLIER"
            color = (0, 220, 0) if is_inlier else (0, 0, 255)
            reproj = image.copy()
            err = None
            if c is not None:
                cc = (int(round(c[0])), int(round(c[1])))
                cv2.drawMarker(reproj, cc, (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
                cv2.drawMarker(mask_img, cc, (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
            if proj is not None:
                pp = (int(round(proj[0])), int(round(proj[1])))
                cv2.circle(reproj, pp, 7, color, 2)
                if c is not None:
                    cc = (int(round(c[0])), int(round(c[1])))
                    cv2.line(reproj, cc, pp, color, 2)
                    err = float(np.linalg.norm(np.asarray(c, dtype=np.float64) - proj))

            urdf_img = None
            if have_urdf and idx < len(render_state):
                urdf_img, _ = masker.overlay(
                    image,
                    K,
                    T_cam2base,
                    render_state[idx],
                    gripper_position=float(gripper[idx]) if gripper is not None and idx < len(gripper) else 0.0,
                    color_bgr=(255, 170, 40),
                    outline_bgr=(255, 225, 120),
                    outline_px=1,
                    alpha=0.60,
                )
            urdf_ref_img = None
            if (
                have_urdf
                and T_ref_cam2base is not None
                and idx < len(render_state)
            ):
                urdf_ref_img, _ = masker.overlay(
                    image,
                    K,
                    T_ref_cam2base,
                    render_state[idx],
                    gripper_position=float(gripper[idx]) if gripper is not None and idx < len(gripper) else 0.0,
                    color_bgr=(0, 145, 255),
                    outline_bgr=(0, 210, 255),
                    outline_px=1,
                    alpha=0.62,
                )

            cv2.imwrite(str(dirs["raw"] / f"{stem}.jpg"), _draw_header(image, f"raw {stem}"))
            cv2.imwrite(str(dirs["masks"] / f"{stem}.jpg"), _draw_header(mask_img, "body=green gripper=red"))
            label = f"{status} err={err:.1f}px" if err is not None else status
            cv2.imwrite(str(dirs["reprojection"] / f"{stem}.jpg"), _draw_header(reproj, label))

            panels = [
                _draw_header(image, f"raw {stem}"),
                _draw_header(mask_img, "seg: body green / gripper red"),
            ]
            if have_urdf and urdf_img is not None:
                cv2.imwrite(str(dirs["urdf_pnp"] / f"{stem}.jpg"), _draw_header(urdf_img, "URDF from PnP"))
                panels.append(_draw_header(urdf_img, "URDF: PnP estimate"))
            if have_urdf:
                ref_label = "URDF from multiview GT" if T_ref_cam2base is not None else "multiview GT missing"
                if urdf_ref_img is None:
                    urdf_ref_img = image.copy()
                cv2.imwrite(str(dirs["urdf_multiview"] / f"{stem}.jpg"), _draw_header(urdf_ref_img, ref_label))
                panels.append(_draw_header(urdf_ref_img, ref_label))
            panels.append(_draw_header(reproj, label))
            ph = max(p.shape[0] for p in panels)
            panels = [_resize_to_h(p, ph) for p in panels]
            panel = np.hstack(panels)
            out_panel = dirs["panel"] / f"{stem}.jpg"
            cv2.imwrite(str(out_panel), panel)
            panel_paths.append(out_panel)

    _write_contact_sheet(panel_paths, ep_seg / args.out_dir_name / "_contact_sheet.jpg")
    return {
        "episode": ep,
        "status": "ok",
        "pnp_status": pnp.get("status"),
        "point_source": pnp.get("point_source"),
        "num_kept": len(kept),
        "num_inliers": len(inliers),
        "rendered": len(panel_paths),
        "out_dir": str(ep_seg / args.out_dir_name),
        "multiview_source": T_ref_source,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oxe_root", type=Path, required=True)
    parser.add_argument("--mask_root", type=Path, required=True)
    parser.add_argument("--dataset", default="droid")
    parser.add_argument("--episodes", nargs="+", default=None)
    parser.add_argument("--pnp_json_name", default="pnp.json")
    parser.add_argument("--out_dir_name", default="pnp_audit")
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--skip_bad_pnp", action="store_true")
    parser.add_argument("--no_urdf", action="store_true")
    parser.add_argument("--mujoco_xml_path", type=Path, default=None,
                        help="Render robot overlays with this MuJoCo XML instead of URDF. "
                             "For Fractal, use AugE robot_xml/google_robot/scene.xml.")
    parser.add_argument(
        "--mujoco_postprocess",
        choices=["none", "auge", "flip_x", "flip_y", "rot180"],
        default="none",
        help="Optional postprocess for MuJoCo masks. 'auge' matches AugE get_overlay_img.",
    )
    parser.add_argument("--urdf_path", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--mesh_dir", type=Path, default=None)
    parser.add_argument("--urdf_backend", choices=["simple", "yourdfpy", "auto"], default="simple")
    parser.add_argument(
        "--arm_joint_names",
        nargs="+",
        default=None,
        help="Optional URDF joint names corresponding to trajectory['joint_position'] columns.",
    )
    parser.add_argument("--gripper_joint_name", default="finger_joint")
    parser.add_argument(
        "--gripper_joint_names",
        nargs="+",
        default=None,
        help="Optional multiple gripper joints to set to the same open/closed value.",
    )
    parser.add_argument("--gripper_open_rad", type=float, default=0.0)
    parser.add_argument("--gripper_closed_rad", type=float, default=0.7)
    parser.add_argument(
        "--extrinsics_json",
        type=Path,
        default=None,
        help="Optional multiview/GT cam2base JSON to render as a separate orange URDF panel.",
    )
    parser.add_argument("--camera_serial", default=None)
    parser.add_argument(
        "--extrinsics_sixd_mode",
        choices=SIXD_MODES,
        default="rpy_cam2base",
        help="How to interpret 6-vector entries in --extrinsics_json. "
             "cam2base_json.py also inverts explicit base2cam records automatically.",
    )
    args = parser.parse_args()

    masker = None
    if args.mujoco_xml_path is not None and not args.no_urdf:
        from mujoco_google_robot_renderer import MuJoCoGoogleRobotRenderer  # noqa: E402
        masker = MuJoCoGoogleRobotRenderer(
            args.mujoco_xml_path,
            verbose=True,
            postprocess=args.mujoco_postprocess,
        )
    elif not args.no_urdf:
        masker = URDFRobotMasker(
            args.urdf_path,
            mesh_dir=args.mesh_dir,
            backend=args.urdf_backend,
            arm_joint_names=tuple(args.arm_joint_names) if args.arm_joint_names else None,
            gripper_joint_name=args.gripper_joint_name,
            gripper_joint_names=tuple(args.gripper_joint_names) if args.gripper_joint_names else None,
            gripper_open_rad=args.gripper_open_rad,
            gripper_closed_rad=args.gripper_closed_rad,
            downsample=2,
            dilate_px=2,
            verbose=True,
        )

    ds_seg = args.mask_root / args.dataset
    episodes = args.episodes or sorted(
        p.name for p in ds_seg.iterdir()
        if p.is_dir() and p.name.startswith("episode_")
    )
    results = [process_episode(args.dataset, ep, args, masker) for ep in episodes]
    for r in results:
        print(
            f"[{args.dataset}/{r['episode']}] {r['status']} "
            f"pnp={r.get('pnp_status')} kept={r.get('num_kept')} "
            f"inliers={r.get('num_inliers')} rendered={r.get('rendered')} "
            f"multiview={r.get('multiview_source')} "
            f"out={r.get('out_dir', '')}"
        )
    (ds_seg / "pnp_audit_summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
