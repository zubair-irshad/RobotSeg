"""Stream a small subset of Open X-Embodiment (OXE) datasets and save
RGB frames + end-effector / gripper trajectories in a RobotSeg-compatible
layout, for stress-testing the RobotSeg segmentation pipeline.

Inspiration: https://github.com/GuanhuaJi/oxe-auge#step-by-step

Key property: we do NOT download full datasets. `tfds.builder_from_directory`
points TFDS at the OXE GCS bucket and `split='train[:N]'` only pulls the
first N episodes' shards.

Output layout (matches VRS/RoboEngine conventions so the RobotSeg inference
scripts can drop in later):

    <out>/
      dataset_map.json                  # embodiment + schema per dataset
      <dataset>/
        <episode_id>/
          frames/
            00000.jpg
            00001.jpg
            ...
          trajectory.npz                # raw per-step arrays
          schema.txt                    # state/action dim meaning
        ...

Usage:
    python tools/download_oxe_subset.py \
        --out_dir ~/RobotSeg/data/oxe_subset \
        --num_episodes 5 \
        --frame_stride 4

    # only a few datasets:
    python tools/download_oxe_subset.py --datasets bridge taco_play

    # list registry and exit:
    python tools/download_oxe_subset.py --list

Requires:
    pip install tensorflow tensorflow-datasets pillow tqdm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from oxe_registry import OXE_DATASETS, DEFAULT_DATASETS_FOR_PNP

GCS_ROOT = "gs://gresearch/robotics"


def _lazy_import_tfds():
    try:
        import tensorflow as tf  # noqa: F401
        import tensorflow_datasets as tfds
        return tfds
    except ImportError as e:
        print(
            "ERROR: tensorflow + tensorflow-datasets are required.\n"
            "  pip install tensorflow tensorflow-datasets",
            file=sys.stderr,
        )
        raise e


def _to_numpy(x):
    """TFDS yields tf.Tensor or nested dicts. Convert leaves to numpy."""
    import tensorflow as tf
    if isinstance(x, dict):
        return {k: _to_numpy(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_numpy(v) for v in x]
    if isinstance(x, tf.Tensor):
        return x.numpy()
    return x


def _nested_get(obs, key):
    """Look up a possibly-nested key like 'clip_function_input/base_pose_tool_reached'.

    Some OXE shards expose the key as flat with the slash kept as a literal,
    others expose it as a nested dict. Try flat first, then nested traversal.
    Returns None if not found."""
    if key in obs:
        return obs[key]
    cur = obs
    for part in key.split("/"):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _pick_rgb_key(obs: dict, candidates: list[str]) -> str | None:
    for k in candidates:
        if k in obs:
            return k
    # fall back: any key whose value looks like an HxWx3 uint8 image
    for k, v in obs.items():
        arr = np.asarray(v)
        if arr.ndim == 3 and arr.shape[-1] == 3 and arr.dtype == np.uint8:
            return k
    return None


def process_episode(episode, cfg: dict, ep_dir: Path, frame_stride: int) -> dict:
    """Write frames/ and trajectory.npz for a single episode. Returns metadata."""
    ep_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = ep_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    steps = list(episode["steps"].as_numpy_iterator())
    if not steps:
        return {"num_frames_saved": 0, "num_steps_total": 0}

    first_obs = steps[0]["observation"]
    rgb_key = _pick_rgb_key(first_obs, cfg["rgb_keys"])
    if rgb_key is None:
        raise RuntimeError(
            f"No RGB key found. Tried {cfg['rgb_keys']}. "
            f"Available obs keys: {list(first_obs.keys())}"
        )

    state_key = cfg.get("state_key")
    action_key = cfg.get("action_key")

    states, actions, kept_idx = [], [], []

    for i, step in enumerate(steps):
        if i % frame_stride != 0:
            continue
        img = step["observation"][rgb_key]
        Image.fromarray(img).save(frames_dir / f"{len(kept_idx):05d}.jpg", quality=92)
        kept_idx.append(i)

        if state_key is not None:
            v = _nested_get(step["observation"], state_key)
            if v is not None:
                states.append(np.asarray(v).ravel())
        if action_key is not None and action_key in step:
            a = step[action_key]
            # action may be a dict (Google robot) or array
            if isinstance(a, dict):
                states_flat = np.concatenate(
                    [np.asarray(v).ravel() for v in a.values()]
                )
                actions.append(states_flat)
            else:
                actions.append(np.asarray(a).ravel())

    traj = {"frame_indices_in_episode": np.asarray(kept_idx, dtype=np.int32)}
    if states:
        traj["state"] = np.stack(states, axis=0)
    if actions:
        traj["action"] = np.stack(actions, axis=0)
    np.savez(ep_dir / "trajectory.npz", **traj)

    (ep_dir / "schema.txt").write_text(
        f"embodiment: {cfg['embodiment']}\n"
        f"rgb_key_used: {rgb_key}\n"
        f"state_key: {state_key}\n"
        f"action_key: {action_key}\n"
        f"frame_stride: {frame_stride}\n"
        f"native_fps: {cfg.get('fps_note', 'unknown')}\n"
        f"state_schema: {cfg['state_schema']}\n"
    )

    return {
        "num_frames_saved": len(kept_idx),
        "num_steps_total": len(steps),
        "rgb_key_used": rgb_key,
        "state_dim": int(states[0].shape[0]) if states else 0,
        "action_dim": int(actions[0].shape[0]) if actions else 0,
    }


def process_dataset(name: str, cfg: dict, out_dir: Path,
                    num_episodes: int, frame_stride: int) -> dict:
    tfds = _lazy_import_tfds()

    builder_dir = f"{GCS_ROOT}/{name}/{cfg['version']}"
    print(f"\n=== {name}  ({cfg['embodiment']})  <- {builder_dir}")

    try:
        builder = tfds.builder_from_directory(builder_dir=builder_dir)
    except Exception as e:
        print(f"  [skip] cannot open builder: {e}")
        return {"error": str(e)}

    ds = builder.as_dataset(split=f"train[:{num_episodes}]")

    ds_out = out_dir / name
    ds_out.mkdir(parents=True, exist_ok=True)

    episodes_meta = []
    for i, episode in enumerate(tqdm(ds, total=num_episodes, desc=name)):
        ep_dir = ds_out / f"episode_{i:04d}"
        try:
            meta = process_episode(episode, cfg, ep_dir, frame_stride)
            meta["episode_id"] = f"episode_{i:04d}"
            episodes_meta.append(meta)
        except Exception as e:
            print(f"  [warn] episode {i} failed: {e}")

    return {
        "embodiment": cfg["embodiment"],
        "version": cfg["version"],
        "state_schema": cfg["state_schema"],
        "num_episodes": len(episodes_meta),
        "episodes": episodes_meta,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=Path,
                   default=Path.home() / "RobotSeg" / "data" / "oxe_subset")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="OXE dataset names. Default: a PnP-friendly subset.")
    p.add_argument("--all", action="store_true",
                   help="Use every dataset in the registry.")
    p.add_argument("--num_episodes", type=int, default=5)
    p.add_argument("--frame_stride", type=int, default=4,
                   help="Keep 1 of every N raw frames.")
    p.add_argument("--list", action="store_true",
                   help="Print registry and exit.")
    args = p.parse_args()

    if args.list:
        for name, cfg in OXE_DATASETS.items():
            print(f"{name:70s} -> {cfg['embodiment']}")
        return

    if args.all:
        names = list(OXE_DATASETS.keys())
    elif args.datasets:
        names = args.datasets
    else:
        names = DEFAULT_DATASETS_FOR_PNP

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Merge into any existing dataset_map.json from prior runs so we don't
    # clobber entries from earlier invocations.
    map_path = args.out_dir / "dataset_map.json"
    if map_path.exists():
        try:
            dataset_map = json.loads(map_path.read_text())
        except json.JSONDecodeError:
            dataset_map = {}
    else:
        dataset_map = {}

    for name in names:
        if name not in OXE_DATASETS:
            print(f"[skip] unknown dataset: {name}")
            continue
        dataset_map[name] = process_dataset(
            name, OXE_DATASETS[name], args.out_dir,
            num_episodes=args.num_episodes, frame_stride=args.frame_stride,
        )
        # Persist incrementally so a crash mid-run still yields useful output
        map_path.write_text(json.dumps(dataset_map, indent=2))

    print(f"\nDone. Wrote {args.out_dir / 'dataset_map.json'}")


if __name__ == "__main__":
    main()
