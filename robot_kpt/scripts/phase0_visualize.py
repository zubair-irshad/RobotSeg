"""Phase 0: visualize mask-derived keypoints on VRS test set.

Expects test-set layout (matches VRS test/ folder):
    <root>/image/<video>/<frame>.jpg
    <root>/mask_gt/<video>/000/<frame>.png   # arm     (binary, >0 = mask)
    <root>/mask_gt/<video>/001/<frame>.png   # gripper

Per-embodiment overlay grid + rejection-reason histogram.

Usage:
    python phase0_visualize.py --root ~/Downloads/test \
        --out robot_kpt/out/phase0 \
        --per-emb 20 --seed 0
"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from robot_kpt.data.kpt_from_mask import KptThresholds, extract_keypoints

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None  # type: ignore

EMB_RE = re.compile(r"^[a-z]+_video_\d+___([^_]+_[A-Za-z0-9]+)___")
ARM_ID = "000"
GRIPPER_ID = "001"


def parse_embodiment(video_name: str) -> str:
    m = EMB_RE.match(video_name)
    if m:
        return m.group(1)
    # Fallback: middle "___" segment.
    parts = video_name.split("___")
    return parts[1] if len(parts) >= 2 else "unknown"


def _parse_simple_yaml(text: str) -> dict:
    """Minimal YAML reader for our 2-level config (top-level keys + scalar fields).
    Used only when PyYAML is not installed."""
    out: dict = {}
    cur: dict | None = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" "):
            key = line.rstrip(":").strip()
            cur = {}
            out[key] = cur
        else:
            k, _, v = line.strip().partition(":")
            v = v.strip()
            try:
                val: float | int | str = int(v)
            except ValueError:
                try:
                    val = float(v)
                except ValueError:
                    val = v
            assert cur is not None
            cur[k.strip()] = val
    return out


def load_thresholds(path: Path) -> dict[str, KptThresholds]:
    if not path.exists():
        return {"default": KptThresholds()}
    text = path.read_text()
    raw = yaml.safe_load(text) if yaml is not None else _parse_simple_yaml(text)
    raw = raw or {}
    out: dict[str, KptThresholds] = {}
    for k, v in raw.items():
        out[k] = KptThresholds(**v)
    if "default" not in out:
        out["default"] = KptThresholds()
    return out


def list_videos(root: Path) -> list[str]:
    img_root = root / "image"
    return sorted(d.name for d in img_root.iterdir() if d.is_dir())


def list_frames(root: Path, video: str) -> list[str]:
    """Frame stems for which BOTH arm and gripper masks exist."""
    img_dir = root / "image" / video
    arm_dir = root / "mask_gt" / video / ARM_ID
    gri_dir = root / "mask_gt" / video / GRIPPER_ID
    if not (arm_dir.is_dir() and gri_dir.is_dir()):
        return []
    img_stems = {Path(p).stem for p in glob(str(img_dir / "*.jpg"))}
    arm_stems = {Path(p).stem for p in glob(str(arm_dir / "*.png"))}
    gri_stems = {Path(p).stem for p in glob(str(gri_dir / "*.png"))}
    return sorted(img_stems & arm_stems & gri_stems)


def read_mask(p: Path) -> np.ndarray | None:
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return m > 0


def overlay(img_bgr: np.ndarray, arm: np.ndarray, gri: np.ndarray,
            results: list) -> np.ndarray:
    out = img_bgr.copy()
    tint = np.zeros_like(out)
    tint[arm] = (200, 60, 0)        # arm: blue
    tint[gri] = (0, 40, 200)        # gripper: red
    out = cv2.addWeighted(out, 0.65, tint, 0.35, 0.0)

    for res in results:
        if res.junction_xy is not None:
            cv2.circle(out, (int(res.junction_xy[0]), int(res.junction_xy[1])),
                       5, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        if res.center_xy is not None:
            cv2.circle(out, (int(res.center_xy[0]), int(res.center_xy[1])),
                       5, (0, 255, 255), -1, lineType=cv2.LINE_AA)
        if (res.junction_xy is not None) and (res.center_xy is not None):
            cv2.line(out,
                     (int(res.junction_xy[0]), int(res.junction_xy[1])),
                     (int(res.center_xy[0]), int(res.center_xy[1])),
                     (255, 255, 255), 1, lineType=cv2.LINE_AA)

    n_ok = sum(1 for r in results if r.reason == "ok")
    if n_ok == len(results) and n_ok > 0:
        label = f"ok x{n_ok}" if n_ok > 1 else "ok"
        color = (0, 255, 0)
    else:
        reasons = [r.reason for r in results if r.reason != "ok"]
        label = reasons[0] if reasons else "no_instances"
        if n_ok > 0:
            label = f"{n_ok}ok+{label}"
        color = (0, 0, 255)
    cv2.putText(out, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(out, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def make_grid(tiles: list[np.ndarray], cols: int, tile_hw: tuple[int, int],
              row_label: str | None = None) -> np.ndarray:
    th, tw = tile_hw
    rows = (len(tiles) + cols - 1) // cols
    grid = np.full((rows * th, cols * tw, 3), 30, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        t = cv2.resize(t, (tw, th))
        grid[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
    if row_label:
        cv2.putText(grid, row_label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(grid, row_label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True,
                    help="Test data root (containing image/, mask_gt/, mask_gt_info/)")
    ap.add_argument("--out", type=Path, default=Path("robot_kpt/out/phase0"))
    ap.add_argument("--per-emb", type=int, default=20, help="frames per embodiment")
    ap.add_argument("--cols", type=int, default=10)
    ap.add_argument("--tile", type=int, default=180, help="tile width px")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thresholds", type=Path,
                    default=Path("robot_kpt/configs/thresholds.yaml"))
    ap.add_argument("--max-embodiments", type=int, default=0,
                    help="0 = all; otherwise cap (useful for quick iteration)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    thr_map = load_thresholds(args.thresholds)

    videos = list_videos(args.root)
    if not videos:
        print(f"No videos under {args.root / 'image'}", file=sys.stderr)
        sys.exit(1)

    by_emb: dict[str, list[str]] = defaultdict(list)
    for v in videos:
        by_emb[parse_embodiment(v)].append(v)

    embodiments = sorted(by_emb.keys())
    if args.max_embodiments > 0:
        embodiments = embodiments[: args.max_embodiments]

    print(f"Found {len(videos)} videos across {len(embodiments)} embodiments: {embodiments}")

    overall_reasons: Counter = Counter()
    per_emb_reasons: dict[str, Counter] = {}

    for emb in embodiments:
        thr = thr_map.get(emb, thr_map["default"])
        emb_videos = by_emb[emb]
        rng.shuffle(emb_videos)

        # Build a pool of (video, frame_stem) for this embodiment.
        pool: list[tuple[str, str]] = []
        for v in emb_videos:
            for f in list_frames(args.root, v):
                pool.append((v, f))
            if len(pool) > args.per_emb * 10:
                break
        if not pool:
            print(f"[{emb}] no frames found, skipping")
            continue
        rng.shuffle(pool)

        tiles: list[np.ndarray] = []
        reasons: Counter = Counter()
        idx = 0
        while len(tiles) < args.per_emb and idx < len(pool):
            v, f = pool[idx]; idx += 1
            img = cv2.imread(str(args.root / "image" / v / f"{f}.jpg"))
            arm = read_mask(args.root / "mask_gt" / v / ARM_ID / f"{f}.png")
            gri = read_mask(args.root / "mask_gt" / v / GRIPPER_ID / f"{f}.png")
            if img is None or arm is None or gri is None:
                reasons["missing_file"] += 1
                continue
            if arm.shape != gri.shape or arm.shape != img.shape[:2]:
                reasons["shape_mismatch"] += 1
                continue
            results = extract_keypoints(arm, gri, thr)
            for r in results:
                reasons[r.reason] += 1
            tiles.append(overlay(img, arm, gri, results))

        if not tiles:
            print(f"[{emb}] all candidate frames were unreadable")
            continue

        per_emb_reasons[emb] = reasons
        overall_reasons.update(reasons)

        # Aspect-preserving tile size from first tile.
        h0, w0 = tiles[0].shape[:2]
        tw = args.tile
        th = max(1, int(round(args.tile * h0 / w0)))
        grid = make_grid(tiles, args.cols, (th, tw), row_label=emb)
        out_path = args.out / f"phase0_{emb}.png"
        cv2.imwrite(str(out_path), grid)
        ok = reasons.get("ok", 0)
        print(f"[{emb}] saved {out_path}  ok={ok}/{sum(reasons.values())}  "
              f"reasons={dict(reasons)}  thr={thr}")

    # ---- summary ----
    print("\n=== rejection histogram (all embodiments) ===")
    total = sum(overall_reasons.values()) or 1
    for r, c in overall_reasons.most_common():
        print(f"  {r:28s} {c:6d}  ({100.0 * c / total:5.1f}%)")

    print("\n=== ok-rate per embodiment ===")
    for emb in sorted(per_emb_reasons):
        rc = per_emb_reasons[emb]
        n = sum(rc.values())
        ok = rc.get("ok", 0)
        print(f"  {emb:20s} ok={ok:3d}/{n:3d}  ({100.0 * ok / max(n, 1):5.1f}%)")


if __name__ == "__main__":
    main()
