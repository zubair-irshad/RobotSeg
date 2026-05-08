"""Extract matching DROID raw-MP4 frames for a local RLDS/OXE subset.

The local OXE subset stores low-res RLDS frames and trajectory.npz. This tool
maps each local episode_XXXX to the raw DROID episode id, downloads only the
matching raw camera MP4 when requested, and extracts the same sampled trajectory
rows as JPEGs.

Output layout mirrors the OXE subset:
  <out_root>/droid/episode_0000/frames/<same-stem>.jpg
  <out_root>/droid/episode_0000/trajectory.npz  (symlink/copy)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
try:
    from natsort import natsorted
except ImportError:
    natsorted = sorted

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from droid_hf_intrinsics import (  # noqa: E402
    FILES,
    _build_path_to_id,
    _ensure_file,
    _flatten_strings,
    _load_dataset_map_metadata,
    _load_episode_id_overrides,
    _load_json,
    _resolve_episode_id,
    _serial_for_camera,
)


def _select_stems(stems: list[str], frame_stride: int, max_frames: int) -> list[str]:
    if frame_stride > 1:
        stems = stems[::frame_stride]
    if max_frames > 0 and len(stems) > max_frames:
        idx = np.linspace(0, len(stems) - 1, max_frames, dtype=np.int64)
        stems = [stems[int(i)] for i in idx]
    return stems


def _path_candidates(raw_value: Any) -> list[str]:
    vals = _flatten_strings(raw_value)
    out: list[str] = []
    prefixes = (
        "gs://xembodiment_data/",
        "gs://gresearch/robotics/",
        "r2d2/r2d2-data-full/",
        "r2d2-data-full/",
    )
    for val in vals:
        val = str(val).strip().rstrip("/")
        if not val:
            continue
        queue = [val]
        for prefix in prefixes:
            if val.startswith(prefix):
                queue.append(val[len(prefix):])
        for q in queue:
            q = q.strip("/").rstrip("/")
            if not q:
                continue
            out.append(q)
            if not q.endswith("recordings/MP4"):
                out.append(f"{q}/recordings/MP4")
            if "recordings/MP4" in q:
                out.append(q.split("/recordings/MP4", 1)[0] + "/recordings/MP4")
    uniq = []
    seen = set()
    for val in out:
        if val and val not in seen:
            seen.add(val)
            uniq.append(val)
    return uniq


def _episode_id_path_candidates(episode_id: str) -> list[str]:
    """Derive droid_raw/1.0.1 paths from raw IDs like LAB+uuid+timestamp."""
    parts = episode_id.split("+")
    if len(parts) < 3:
        return []
    lab = parts[0]
    stamp = parts[-1]
    try:
        when = dt.datetime.strptime(stamp, "%Y-%m-%d-%Hh-%Mm-%Ss")
    except ValueError:
        return []
    date_s = when.strftime("%Y-%m-%d")
    # Raw sessions are named e.g. Tue_Dec_19_22:57:46_2023. Use wildcards
    # for weekday/month/day formatting differences and success/failure split.
    time_s = when.strftime("%H:%M:%S")
    year_s = when.strftime("%Y")
    return [
        f"1.0.1/{lab}/*/{date_s}/*_{time_s}_{year_s}/recordings/MP4",
        f"1.0.1/{lab}/success/{date_s}/*_{time_s}_{year_s}/recordings/MP4",
        f"1.0.1/{lab}/failure/{date_s}/*_{time_s}_{year_s}/recordings/MP4",
    ]


def _gsutil_ls(pattern: str) -> list[str]:
    proc = subprocess.run(
        ["gsutil", "ls", pattern],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _is_stereo_mp4(path_like: str | Path) -> bool:
    return "stereo" in Path(str(path_like)).name.lower()


def _rank_mp4(path_like: str | Path, serial: str | None, prefer_stereo: bool) -> tuple[int, int, str]:
    path_s = str(path_like)
    name = Path(path_s).name
    stem = Path(name).stem
    is_stereo = _is_stereo_mp4(name)
    exact_serial = bool(serial) and stem == str(serial)
    stereo_rank = 0 if (is_stereo == prefer_stereo) else 1
    exact_rank = 0 if exact_serial else 1
    return stereo_rank, exact_rank, path_s


def _choose_mp4(hits: list[str], serial: str | None, prefer_stereo: bool) -> str | None:
    if not hits:
        return None
    if serial:
        serial_hits = [h for h in hits if serial in Path(h).name or serial in h]
        if serial_hits:
            return sorted(serial_hits, key=lambda h: _rank_mp4(h, serial, prefer_stereo))[0]
    return sorted(hits, key=lambda h: _rank_mp4(h, serial, prefer_stereo))[0]


def _find_gcs_mp4(args, episode_id: str, episode_id_to_path: dict[str, Any],
                  serial: str | None) -> str | None:
    raw_value = episode_id_to_path.get(episode_id)
    rels = []
    if raw_value is not None:
        rels += _path_candidates(raw_value)
    rels += _episode_id_path_candidates(episode_id)
    for rel in rels:
        bases = []
        if rel.startswith("gs://"):
            bases.append(rel)
        bases.append(f"{args.gcs_root.rstrip('/')}/{rel.lstrip('/')}")
        for base in bases:
            hits = _gsutil_ls(f"{base.rstrip('/')}/*.mp4")
            if not hits:
                hits = _gsutil_ls(f"{base.rstrip('/')}/*/*.mp4")
            if not hits:
                hits = _gsutil_ls(f"{base.rstrip('/')}/*/*/*.mp4")
            if hits:
                selected = _choose_mp4(hits, serial, args.prefer_stereo_mp4)
                if selected:
                    return selected
    return None


def _local_mp4_path(raw_root: Path, episode_id: str, serial: str | None,
                    prefer_stereo: bool = False) -> Path | None:
    ep_dir = raw_root / episode_id
    if not ep_dir.exists():
        return None
    hits = sorted(ep_dir.rglob("*.mp4"))
    if serial:
        serial_hits = [p for p in hits if serial in p.name or serial in str(p)]
        if serial_hits:
            return sorted(serial_hits, key=lambda p: _rank_mp4(p, serial, prefer_stereo))[0]
    return sorted(hits, key=lambda p: _rank_mp4(p, serial, prefer_stereo))[0] if hits else None


def _download_mp4(args, episode_id: str, gcs_uri: str, serial: str | None) -> Path:
    name = Path(gcs_uri).name
    if serial and serial not in name:
        name = f"{serial}_{name}"
    dst = args.raw_root / episode_id / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    subprocess.run(["gsutil", "-m", "cp", gcs_uri, str(dst)], check=True)
    return dst


def _extract_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    return frame if ok else None


def _resolve_stereo_crop(mode: str, camera: str) -> str:
    if mode != "auto":
        return mode
    if camera.endswith("_left"):
        return "left"
    if camera.endswith("_right"):
        return "right"
    return "none"


def _maybe_crop_stereo_frame(
    frame: np.ndarray,
    crop_mode: str,
    camera: str,
    mp4_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Crop side-by-side stereo MP4 frames to the requested mono view.

    DROID raw MP4s are inconsistent in this subset: some are normal 1280x720
    mono frames, while others are 2560x720 side-by-side stereo. The local OXE
    camera key exterior_image_1_left expects the left image, so feeding the
    stitched frame into RobotSeg corrupts both segmentation and PnP.
    """
    h, w = frame.shape[:2]
    resolved = _resolve_stereo_crop(crop_mode, camera)
    meta = {
        "input_width": int(w),
        "input_height": int(h),
        "stereo_crop": crop_mode,
        "resolved_stereo_crop": resolved,
        "source_mp4_is_stereo": _is_stereo_mp4(mp4_path),
        "cropped": False,
        "output_width": int(w),
        "output_height": int(h),
    }
    is_side_by_side = w >= 2 * h and (w / max(1, h)) > 2.5
    if resolved == "none" or not is_side_by_side:
        return frame, meta
    mid = w // 2
    if resolved == "left":
        cropped = frame[:, :mid]
    elif resolved == "right":
        cropped = frame[:, mid:]
    else:
        raise ValueError(f"unsupported stereo crop mode: {crop_mode}")
    meta.update({
        "cropped": True,
        "output_width": int(cropped.shape[1]),
        "output_height": int(cropped.shape[0]),
    })
    return cropped, meta


def _safe_link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:
        shutil.copy2(src, dst)


def process_episode(args, ep: str, maps: dict[str, Any]) -> dict[str, Any]:
    ep_oxe = args.oxe_root / args.dataset / ep
    frames_dir = ep_oxe / "frames"
    traj_path = ep_oxe / "trajectory.npz"
    if not frames_dir.exists() or not traj_path.exists():
        return {"episode": ep, "status": "skip-missing-local-episode"}

    episode_id, id_source = _resolve_episode_id(
        ep,
        ep_oxe,
        args.dataset,
        maps["episode_id_overrides"],
        maps["dataset_map_metadata"],
        maps["path_to_id"],
        maps["intrinsics"],
    )
    if episode_id is None:
        return {"episode": ep, "status": "skip-no-raw-episode-id", "id_source": id_source}

    intr_entry = maps["intrinsics"].get(episode_id, {})
    serial = _serial_for_camera(
        maps["camera_serials"].get(episode_id),
        args.camera,
        intr_entry if isinstance(intr_entry, dict) else {},
        args.camera_index,
    )

    mp4 = _local_mp4_path(args.raw_root, episode_id, serial, args.prefer_stereo_mp4)
    gcs_uri = None
    if mp4 is None:
        gcs_uri = _find_gcs_mp4(args, episode_id, maps["episode_id_to_path"], serial)
        if gcs_uri is None:
            return {
                "episode": ep,
                "status": "skip-no-raw-mp4",
                "episode_id": episode_id,
                "serial": serial,
            }
        if not args.download:
            return {
                "episode": ep,
                "status": "skip-needs-download",
                "episode_id": episode_id,
                "serial": serial,
                "gcs_uri": gcs_uri,
            }
        mp4 = _download_mp4(args, episode_id, gcs_uri, serial)

    stems = [
        Path(f).stem
        for f in natsorted(os.listdir(frames_dir))
        if Path(f).suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    stems = _select_stems(stems, args.frame_stride, args.max_frames_per_seq)
    with np.load(traj_path, allow_pickle=True) as traj:
        if "frame_indices_in_episode" in traj.files:
            raw_indices = np.asarray(traj["frame_indices_in_episode"], dtype=np.int64)
        else:
            raw_indices = np.arange(len(stems), dtype=np.int64)

    out_ep = args.out_root / args.dataset / ep
    out_frames = out_ep / "frames"
    out_frames.mkdir(parents=True, exist_ok=True)
    _safe_link_or_copy(traj_path, out_ep / "trajectory.npz")
    for name in ("schema.txt", "episode_metadata.json", "camera.json"):
        src = ep_oxe / name
        if src.exists():
            _safe_link_or_copy(src, out_ep / name)

    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        return {"episode": ep, "status": "skip-mp4-open-failed", "mp4": str(mp4)}

    saved = 0
    missing = []
    crop_meta: dict[str, Any] | None = None
    for stem in stems:
        row_idx = int(stem)
        raw_idx = int(raw_indices[row_idx]) if row_idx < len(raw_indices) else row_idx
        frame = _extract_frame(cap, raw_idx)
        if frame is None:
            missing.append({"stem": stem, "raw_idx": raw_idx})
            continue
        frame, this_crop_meta = _maybe_crop_stereo_frame(frame, args.stereo_crop, args.camera, mp4)
        if crop_meta is None:
            crop_meta = this_crop_meta
        cv2.imwrite(str(out_frames / f"{stem}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        saved += 1
    cap.release()

    meta = {
        "episode": ep,
        "status": "ok" if saved else "bad-no-frames-saved",
        "episode_id": episode_id,
        "episode_id_source": id_source,
        "camera": args.camera,
        "serial": serial,
        "mp4": str(mp4),
        "gcs_uri": gcs_uri,
        "num_requested": len(stems),
        "num_saved": saved,
        "missing": missing[:20],
    }
    if crop_meta is not None:
        meta.update(crop_meta)
    (out_ep / "raw_mp4_extract.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--oxe_root", type=Path, default=Path("data/oxe_subset"))
    p.add_argument("--dataset", default="droid")
    p.add_argument("--out_root", type=Path, default=Path("data/oxe_subset_droid_raw_mp4"))
    p.add_argument("--raw_root", type=Path, default=Path("data/droid_raw_mp4"))
    p.add_argument("--cache_dir", type=Path, default=Path("data/droid_hf_calib"))
    p.add_argument("--gcs_root", default="gs://gresearch/robotics/droid_raw")
    p.add_argument("--camera", default="exterior_image_1_left",
                   choices=["exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"])
    p.add_argument("--camera_index", type=int, default=0)
    p.add_argument("--episodes", nargs="+", default=None)
    p.add_argument("--episode_id_json", type=Path, default=None)
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--max_frames_per_seq", type=int, default=32)
    p.add_argument("--stereo_crop", choices=["auto", "left", "right", "none"], default="auto",
                   help="Crop side-by-side stereo raw MP4 frames before saving. "
                        "auto uses left/right from the camera name and leaves mono frames untouched.")
    p.add_argument("--prefer_stereo_mp4", action="store_true",
                   help="Prefer *stereo*.mp4 over the mono serial MP4 when both are present. "
                        "Default is to prefer mono files such as 20521388.mp4.")
    p.add_argument("--download", action="store_true")
    args = p.parse_args()

    paths = {name: _ensure_file(args.cache_dir, name) for name in FILES}
    episode_id_to_path = _load_json(paths["episode_id_to_path.json"])
    maps = {
        "intrinsics": _load_json(paths["intrinsics.json"]),
        "episode_id_to_path": episode_id_to_path,
        "path_to_id": _build_path_to_id(episode_id_to_path),
        "camera_serials": _load_json(paths["camera_serials.json"]),
        "dataset_map_metadata": _load_dataset_map_metadata(args.oxe_root, args.dataset),
    }
    episode_id_json = args.episode_id_json
    ds_root = args.oxe_root / args.dataset
    if episode_id_json is None and (ds_root / "episode_id_map.json").exists():
        episode_id_json = ds_root / "episode_id_map.json"
    maps["episode_id_overrides"] = _load_episode_id_overrides(episode_id_json, args.dataset)

    episodes = args.episodes or sorted(
        p.name for p in ds_root.iterdir()
        if p.is_dir() and p.name.startswith("episode_")
    )
    results = [process_episode(args, ep, maps) for ep in episodes]
    out_summary = args.out_root / args.dataset / "raw_mp4_extract_summary.json"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps({"dataset": args.dataset, "results": results}, indent=2))
    for r in results:
        print(
            f"[{args.dataset}/{r['episode']}] {r['status']} "
            f"saved={r.get('num_saved', 0)}/{r.get('num_requested', 0)} "
            f"serial={r.get('serial')} "
            f"size={r.get('input_width')}x{r.get('input_height')}->{r.get('output_width')}x{r.get('output_height')} "
            f"crop={r.get('resolved_stereo_crop')} cropped={r.get('cropped')} "
            f"mp4={r.get('mp4') or r.get('gcs_uri', '')}"
        )
    print(f"Wrote {out_summary}")


if __name__ == "__main__":
    main()
