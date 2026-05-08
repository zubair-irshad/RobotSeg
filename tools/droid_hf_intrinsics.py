"""Build per-episode DROID K overrides from KarlP/droid intrinsics.

KarlP/droid stores raw ZED intrinsics by raw DROID episode ID and camera
serial. This script maps local RLDS episodes to those IDs, selects the
requested camera serial, scales raw intrinsics to the local RLDS image size,
and writes a K_json file consumable by tools/pnp_oxe.py.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

import cv2


HF_REPO = "KarlP/droid"
HF_BASE = f"https://huggingface.co/{HF_REPO}/resolve/main"
FILES = ("intrinsics.json", "episode_id_to_path.json", "camera_serials.json")


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url} -> {dst}")
    with urllib.request.urlopen(url) as r, dst.open("wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def _ensure_file(cache_dir: Path, name: str) -> Path:
    dst = cache_dir / name
    if dst.exists():
        return dst
    try:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(repo_id=HF_REPO, filename=name,
                                    local_dir=str(cache_dir),
                                    repo_type="model"))
    except Exception:
        _download(f"{HF_BASE}/{name}", dst)
        return dst


def _load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def _path_variants(path: str) -> list[str]:
    vals = []
    queue = [path, path.strip(), path.strip().rstrip("/")]
    prefixes = (
        "gs://xembodiment_data/",
        "gs://gresearch/robotics/",
        "r2d2/r2d2-data-full/",
        "r2d2-data-full/",
    )
    for val in list(queue):
        for prefix in prefixes:
            if val.startswith(prefix):
                queue.append(val[len(prefix):])
    for val in queue:
        if not val:
            continue
        vals.append(val)
        vals.append(val.lstrip("/"))
        if val.endswith("/trajectory.h5"):
            vals.append(val[: -len("/trajectory.h5")])
        if val.endswith("/recordings/MP4"):
            vals.append(val[: -len("/recordings/MP4")])
        for marker in ("/r2d2/r2d2-data-full/", "/r2d2-data-full/"):
            if marker in val:
                vals.append(val.split(marker, 1)[1])
    out = []
    seen = set()
    for val in vals:
        if val and val not in seen:
            seen.add(val)
            out.append(val)
    return out


def _build_path_to_id(episode_id_to_path: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for episode_id, value in episode_id_to_path.items():
        for raw in _flatten_strings(value):
            for variant in _path_variants(raw):
                out.setdefault(variant, episode_id)
    return out


def _read_image_size(ep_dir: Path) -> tuple[int, int]:
    for p in sorted((ep_dir / "frames").glob("*")):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        img = cv2.imread(str(p))
        if img is not None:
            h, w = img.shape[:2]
            return w, h
    raise FileNotFoundError(f"no readable frame under {ep_dir / 'frames'}")


def _metadata_candidates(meta: dict) -> list[str]:
    vals: list[str] = []
    for key in ("episode_id", "id", "recording_folderpath", "file_path"):
        val = meta.get(key)
        if isinstance(val, str) and val:
            vals.append(val)
    rec = meta.get("recording_folderpath")
    fp = meta.get("file_path")
    if isinstance(rec, str) and isinstance(fp, str):
        vals += [f"{rec}--{fp}", f"{rec}/{fp}", str(Path(rec) / fp)]
    vals += [str(v) for v in meta.values() if isinstance(v, str)]
    out = []
    seen = set()
    for val in vals:
        for cand in (val, val.rstrip("/"), val.lstrip("/")):
            if cand and cand not in seen:
                seen.add(cand)
                out.append(cand)
    return out


def _episode_id_from_metadata(meta: dict, path_to_id: dict[str, str]) -> str | None:
    for val in _metadata_candidates(meta):
        m = re.search(r"metadata_([^/\\]+)\.json", val)
        if m:
            return m.group(1)
        for cand in _path_variants(val):
            if cand in path_to_id:
                return path_to_id[cand]
        for path, episode_id in path_to_id.items():
            if path.endswith(val) or val.endswith(path):
                return episode_id
    return None


def _load_episode_id_overrides(path: Path | None, dataset: str) -> dict[str, str]:
    """Load local episode name -> raw DROID episode ID overrides.

    Accepted forms:
      {"episode_0000": "AUTOLab+..."}
      {"droid/episode_0000": "AUTOLab+..."}
      {"droid": {"episode_0000": "AUTOLab+..."}}
      {"droid": {"episode_0000": {"episode_id": "AUTOLab+..."}}}
    """
    if path is None:
        return {}
    data = _load_json(path)
    if isinstance(data, dict) and isinstance(data.get(dataset), dict):
        data = data[dataset]
    if not isinstance(data, dict):
        raise ValueError(f"episode id map must be a JSON object: {path}")

    out: dict[str, str] = {}
    for key, val in data.items():
        if isinstance(val, str):
            out[str(key)] = val
            continue
        if isinstance(val, dict):
            for field in ("episode_id", "droid_episode_id", "raw_episode_id", "id", "key"):
                raw_id = val.get(field)
                if isinstance(raw_id, str) and raw_id:
                    out[str(key)] = raw_id
                    break
    return out


def _resolve_episode_id(ep: str, ep_dir: Path, dataset: str,
                        overrides: dict[str, str],
                        dataset_map_metadata: dict[str, dict],
                        path_to_id: dict[str, str],
                        intrinsics: dict[str, Any]) -> tuple[str | None, str]:
    for key in (f"{dataset}/{ep}", ep, str(ep_dir)):
        if key in overrides:
            return overrides[key], "episode_id_json"

    # If the local folder is already named by the raw DROID key, use it.
    if ep in intrinsics:
        return ep, "episode_dir_name"

    for name in ("episode_id.txt", "droid_episode_id.txt", "raw_episode_id.txt"):
        path = ep_dir / name
        if path.exists():
            raw_id = path.read_text().strip().splitlines()[0]
            if raw_id:
                return raw_id, name

    meta = _load_episode_metadata(ep_dir)
    if meta is not None:
        episode_id = _episode_id_from_metadata(meta, path_to_id)
        if episode_id is not None:
            return episode_id, "episode_metadata.json"

    meta = dataset_map_metadata.get(ep)
    if meta is not None:
        episode_id = _episode_id_from_metadata(meta, path_to_id)
        if episode_id is not None:
            return episode_id, "dataset_map.json"

    return None, "no episode id mapping"


def _camera_aliases(camera: str) -> tuple[str, ...]:
    if camera == "exterior_image_1_left":
        return (
            "exterior_image_1_left", "exterior_image_1", "exterior_1",
            "exterior1", "external_camera_1", "external_1", "ext_camera_1",
            "ext1", "camera_1", "cam1", "zed1",
        )
    if camera == "exterior_image_2_left":
        return (
            "exterior_image_2_left", "exterior_image_2", "exterior_2",
            "exterior2", "external_camera_2", "external_2", "ext_camera_2",
            "ext2", "camera_2", "cam2", "zed2",
        )
    if camera == "wrist_image_left":
        return ("wrist_image_left", "wrist_image", "wrist", "wrist_camera", "camera_wrist")
    return (camera,)


def _flatten_strings(x: Any) -> list[str]:
    if isinstance(x, (str, int)):
        return [str(x)]
    if isinstance(x, list):
        out = []
        for v in x:
            out += _flatten_strings(v)
        return out
    if isinstance(x, dict):
        out = []
        for v in x.values():
            out += _flatten_strings(v)
        return out
    return []


def _load_dataset_map_metadata(oxe_root: Path, dataset: str) -> dict[str, dict]:
    path = oxe_root / "dataset_map.json"
    if not path.exists():
        return {}
    try:
        data = _load_json(path)
    except json.JSONDecodeError:
        return {}
    ds = data.get(dataset) if isinstance(data, dict) else None
    episodes = ds.get("episodes", []) if isinstance(ds, dict) else []
    out: dict[str, dict] = {}
    for rec in episodes:
        if not isinstance(rec, dict):
            continue
        ep = rec.get("episode_id")
        meta = rec.get("episode_metadata")
        if isinstance(ep, str) and isinstance(meta, dict):
            out[ep] = meta
    return out


def _serial_for_camera(serial_meta: Any, camera: str,
                       intrinsics_entry: dict, camera_index: int) -> str | None:
    aliases = _camera_aliases(camera)
    if isinstance(serial_meta, dict):
        # Common case: camera name -> serial.
        for key, val in serial_meta.items():
            key_l = str(key).lower()
            if any(a in key_l for a in aliases):
                vals = _flatten_strings(val)
                for v in vals:
                    if v in intrinsics_entry:
                        return v
        # Alternate case: serial -> camera name.
        for key, val in serial_meta.items():
            vals = " ".join(_flatten_strings(val)).lower()
            if str(key) in intrinsics_entry and any(a in vals for a in aliases):
                return str(key)

    serials = sorted(str(s) for s in intrinsics_entry.keys())
    if len(serials) == 1:
        return serials[0]
    if 0 <= camera_index < len(serials):
        return serials[camera_index]
    return None


def _choose_source_resolution(fx: float, cx: float, fy: float, cy: float,
                              target_w: int, target_h: int,
                              raw_w: int | None, raw_h: int | None) -> tuple[int, int, str]:
    if raw_w and raw_h:
        return raw_w, raw_h, "cli"
    if cx <= target_w and cy <= target_h and fx <= 3 * target_w and fy <= 3 * target_h:
        return target_w, target_h, "already_target"
    candidates = [(1920, 1080), (1280, 720), (960, 540), (640, 360), (320, 180)]
    best = min(
        candidates,
        key=lambda wh: abs(cx - 0.5 * wh[0]) / wh[0] + abs(cy - 0.5 * wh[1]) / wh[1],
    )
    return best[0], best[1], "auto_known_16x9"


def _extract_camera_matrix(record: Any) -> tuple[list[float], int | None, int | None]:
    if isinstance(record, dict):
        mat = (
            record.get("cameraMatrix")
            or record.get("camera_matrix")
            or record.get("K")
            or record.get("intrinsics")
        )
        raw_w = record.get("width")
        raw_h = record.get("height")
    else:
        mat = record
        raw_w = None
        raw_h = None

    if not isinstance(mat, list):
        raise ValueError(f"cameraMatrix must be a list, got {type(mat).__name__}")
    if len(mat) == 4:
        vals = [float(x) for x in mat]
    elif len(mat) == 9:
        # 3x3 row-major K fallback.
        vals = [float(mat[0]), float(mat[2]), float(mat[4]), float(mat[5])]
    elif len(mat) == 3 and all(isinstance(row, list) and len(row) == 3 for row in mat):
        vals = [float(mat[0][0]), float(mat[0][2]), float(mat[1][1]), float(mat[1][2])]
    else:
        raise ValueError(f"unsupported cameraMatrix shape: {mat}")
    return vals, int(raw_w) if raw_w else None, int(raw_h) if raw_h else None


def _scale_intrinsics(record: Any, target_w: int, target_h: int,
                      raw_w: int | None, raw_h: int | None) -> dict:
    # KarlP/droid stores [fx, cx, fy, cy] under cameraMatrix.
    raw, rec_w, rec_h = _extract_camera_matrix(record)
    fx, cx, fy, cy = raw
    raw_w = raw_w or rec_w
    raw_h = raw_h or rec_h
    src_w, src_h, mode = _choose_source_resolution(fx, cx, fy, cy, target_w,
                                                   target_h, raw_w, raw_h)
    sx = target_w / float(src_w)
    sy = target_h / float(src_h)
    return {
        "fx": fx * sx,
        "fy": fy * sy,
        "cx": cx * sx,
        "cy": cy * sy,
        "width": target_w,
        "height": target_h,
        "raw_width": src_w,
        "raw_height": src_h,
        "scale_x": sx,
        "scale_y": sy,
        "scale_mode": mode,
    }


def _load_episode_metadata(ep_dir: Path) -> dict | None:
    path = ep_dir / "episode_metadata.json"
    if path.exists():
        return _load_json(path)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oxe_root", type=Path, default=Path("data/oxe_subset"))
    parser.add_argument("--dataset", default="droid")
    parser.add_argument("--cache_dir", type=Path, default=Path("data/droid_hf_calib"))
    parser.add_argument("--out_json", type=Path,
                        default=Path("data/droid_hf_intrinsics_scaled.json"))
    parser.add_argument("--camera", default="exterior_image_1_left",
                        choices=["exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"])
    parser.add_argument("--camera_index", type=int, default=0,
                        help="Fallback serial index if camera_serials.json cannot identify the camera.")
    parser.add_argument("--raw_width", type=int, default=None)
    parser.add_argument("--raw_height", type=int, default=None)
    parser.add_argument("--episodes", nargs="+", default=None)
    parser.add_argument("--episode_id_json", type=Path, default=None,
                        help="Optional JSON mapping local episode names to raw DROID episode IDs.")
    args = parser.parse_args()

    paths = {name: _ensure_file(args.cache_dir, name) for name in FILES}
    intrinsics = _load_json(paths["intrinsics.json"])
    episode_id_to_path = _load_json(paths["episode_id_to_path.json"])
    path_to_id = _build_path_to_id(episode_id_to_path)
    camera_serials = _load_json(paths["camera_serials.json"])

    ds_root = args.oxe_root / args.dataset
    episode_id_json = args.episode_id_json
    if episode_id_json is None and (ds_root / "episode_id_map.json").exists():
        episode_id_json = ds_root / "episode_id_map.json"
    episode_id_overrides = _load_episode_id_overrides(episode_id_json,
                                                      args.dataset)
    if episode_id_json is not None:
        print(f"Using episode id map: {episode_id_json}")
    dataset_map_metadata = _load_dataset_map_metadata(args.oxe_root, args.dataset)

    episodes = args.episodes or sorted(
        p.name for p in ds_root.iterdir()
        if p.is_dir() and not p.name.startswith(".") and not p.name.startswith("_")
    )

    out = {args.dataset: {}}
    resolved_episode_id_map: dict[str, str] = {}
    missing = []
    for ep in episodes:
        ep_dir = ds_root / ep
        episode_id, id_source = _resolve_episode_id(ep, ep_dir, args.dataset,
                                                    episode_id_overrides,
                                                    dataset_map_metadata,
                                                    path_to_id, intrinsics)
        if episode_id is None:
            missing.append((ep, id_source))
            continue
        intr_entry = intrinsics.get(episode_id)
        if not isinstance(intr_entry, dict):
            missing.append((ep, f"no intrinsics for episode_id={episode_id}"))
            continue
        serial = _serial_for_camera(camera_serials.get(episode_id), args.camera,
                                    intr_entry, args.camera_index)
        if serial is None or serial not in intr_entry:
            missing.append((ep, f"no serial for {args.camera}; episode_id={episode_id}"))
            continue
        W, H = _read_image_size(ep_dir)
        try:
            K = _scale_intrinsics(intr_entry[serial], W, H, args.raw_width, args.raw_height)
        except ValueError as exc:
            missing.append((ep, f"bad intrinsics for episode_id={episode_id} serial={serial}: {exc}"))
            continue
        K["source"] = (
            f"KarlP/droid intrinsics.json episode_id={episode_id} "
            f"camera={args.camera} serial={serial}"
        )
        out[args.dataset][ep] = K
        resolved_episode_id_map[ep] = episode_id
        print(
            f"[{ep}] id={episode_id} ({id_source}) serial={serial} "
            f"raw={K['raw_width']}x{K['raw_height']} -> {W}x{H} "
            f"fx={K['fx']:.2f} fy={K['fy']:.2f} cx={K['cx']:.2f} cy={K['cy']:.2f}"
        )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out_json} with {len(out[args.dataset])} episode K entries")
    if resolved_episode_id_map:
        map_path = ds_root / "episode_id_map.json"
        map_path.write_text(json.dumps({args.dataset: resolved_episode_id_map}, indent=2))
        print(f"Wrote resolved episode id map: {map_path}")
    if missing:
        print("\nMissing:")
        for ep, reason in missing:
            print(f"  {ep}: {reason}")
        print(
            "\nIf your local folders are named episode_0000, provide "
            "--episode_id_json mapping those names to raw DROID keys such as "
            "AUTOLab+5d05c5aa+2023-07-07-10h-29m-59s."
        )


if __name__ == "__main__":
    main()
