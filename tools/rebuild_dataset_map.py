"""Rebuild dataset_map.json from an already-downloaded oxe_subset directory.

Use when a prior run of download_oxe_subset.py clobbered the map. Walks
<out_dir>/<dataset>/<episode>/{frames,trajectory.npz,schema.txt} and re-emits
a dataset_map.json with embodiment + per-episode stats.

Usage:
    python tools/rebuild_dataset_map.py --out_dir ~/RobotSeg/data/oxe_subset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from oxe_registry import OXE_DATASETS


def summarize_episode(ep_dir: Path) -> dict | None:
    frames_dir = ep_dir / "frames"
    if not frames_dir.is_dir():
        return None
    num_frames = sum(1 for _ in frames_dir.glob("*.jpg"))
    if num_frames == 0:
        return None
    meta = {"num_frames_saved": num_frames, "episode_id": ep_dir.name}
    traj_path = ep_dir / "trajectory.npz"
    if traj_path.exists():
        with np.load(traj_path) as z:
            meta["state_dim"] = int(z["state"].shape[1]) if "state" in z else 0
            meta["action_dim"] = int(z["action"].shape[1]) if "action" in z else 0
            if "frame_indices_in_episode" in z:
                meta["num_steps_total"] = int(z["frame_indices_in_episode"].max()) + 1
    schema_path = ep_dir / "schema.txt"
    if schema_path.exists():
        for line in schema_path.read_text().splitlines():
            if line.startswith("rgb_key_used:"):
                meta["rgb_key_used"] = line.split(":", 1)[1].strip()
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=Path, required=True)
    args = p.parse_args()

    dataset_map = {}
    for ds_dir in sorted(args.out_dir.iterdir()):
        if not ds_dir.is_dir() or ds_dir.name not in OXE_DATASETS:
            continue
        cfg = OXE_DATASETS[ds_dir.name]
        episodes = []
        for ep_dir in sorted(ds_dir.iterdir()):
            if not ep_dir.is_dir():
                continue
            m = summarize_episode(ep_dir)
            if m is not None:
                episodes.append(m)
        if not episodes:
            continue
        dataset_map[ds_dir.name] = {
            "embodiment": cfg["embodiment"],
            "version": cfg["version"],
            "state_schema": cfg["state_schema"],
            "num_episodes": len(episodes),
            "episodes": episodes,
        }

    out_path = args.out_dir / "dataset_map.json"
    out_path.write_text(json.dumps(dataset_map, indent=2))
    print(f"Wrote {out_path} with {len(dataset_map)} datasets.")


if __name__ == "__main__":
    main()
