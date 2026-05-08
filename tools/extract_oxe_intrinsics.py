"""Re-stream OXE RLDS metadata to populate per-episode camera.json without
re-downloading frames. Run this if you downloaded datasets before
download_oxe_subset.py learned to extract intrinsics.

Streams `train[:N]` of each requested dataset, scans episode_metadata and
the first step's observation/step for anything that looks like camera
intrinsics, and writes <oxe_root>/<dataset>/episode_<i>/camera.json.

Usage:
    python tools/extract_oxe_intrinsics.py \
        --oxe_root ~/RobotSeg/data/oxe_subset \
        --num_episodes 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from oxe_registry import OXE_DATASETS, DEFAULT_DATASETS_FOR_PNP  # noqa: E402
from download_oxe_subset import (  # noqa: E402
    GCS_ROOT, _jsonable, _lazy_import_tfds, extract_intrinsics,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--oxe_root", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--num_episodes", type=int, default=5)
    args = p.parse_args()

    tfds = _lazy_import_tfds()
    names = args.datasets or DEFAULT_DATASETS_FOR_PNP

    for name in names:
        if name not in OXE_DATASETS:
            print(f"[skip] unknown dataset: {name}")
            continue
        cfg = OXE_DATASETS[name]
        ds_root = args.oxe_root / name
        if not ds_root.exists():
            print(f"[skip] {name}: {ds_root} missing")
            continue

        builder_dir = f"{GCS_ROOT}/{name}/{cfg['version']}"
        try:
            builder = tfds.builder_from_directory(builder_dir=builder_dir)
        except Exception as e:
            print(f"[skip] {name}: cannot open builder: {e}")
            continue
        ds = builder.as_dataset(split=f"train[:{args.num_episodes}]")

        n_written = 0
        for i, episode in enumerate(ds):
            ep_dir = ds_root / f"episode_{i:04d}"
            if not ep_dir.exists():
                continue
            try:
                ep_meta = _jsonable(episode["episode_metadata"])
                (ep_dir / "episode_metadata.json").write_text(
                    json.dumps(ep_meta, indent=2)
                )
            except Exception:
                pass
            try:
                first_step = next(iter(episode["steps"].as_numpy_iterator()))
            except StopIteration:
                continue
            cam_meta = extract_intrinsics(episode, first_step)
            if cam_meta is None:
                continue
            (ep_dir / "camera.json").write_text(json.dumps(cam_meta, indent=2))
            n_written += 1
        print(f"  [{name}] wrote camera.json for {n_written} episodes")


if __name__ == "__main__":
    main()
