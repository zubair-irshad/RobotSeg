"""Sanity-check the EE_XYZ_DIMS claims in tools/pnp_oxe.py against actual
RLDS feature specs.

For each registered dataset:
  1. Open the TFDS builder from GCS.
  2. Print the steps.observation feature spec — shape & dtype of every field.
  3. Pull one episode, print:
       - the first state vector (full)
       - the slice we claim is EE xyz, with values
       - per-dim min/max/mean over the whole episode
       - a delta = state[-1] - state[0] to make sure xyz is moving in
         metric units (~10 cm scale, not joint radians).

If a "claimed EE xyz" channel doesn't move on the order of cm/m through
the episode, or has values outside [-2, 2] meters, the slice is wrong.

Usage:
  python tools/inspect_oxe_state.py
  python tools/inspect_oxe_state.py --datasets bridge cmu_stretch droid
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from oxe_registry import OXE_DATASETS  # noqa: E402

# Mirrors EE_XYZ_DIMS in pnp_oxe.py
CLAIMED_EE_DIMS = {
    "taco_play": (0, 3),
    "fanuc_manipulation_v2": (0, 3),
    "berkeley_autolab_ur5": (7, 10),
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": (0, 3),
    "bridge": (0, 3),
    "cmu_stretch": (0, 3),
    "fractal20220817_data": (0, 3),
    "droid": (0, 3),
}

GCS_ROOT = "gs://gresearch/robotics"


def _spec_to_str(spec, indent=2):
    out = []
    if hasattr(spec, "items"):
        for k, v in spec.items():
            out.append(" " * indent + f"{k}:")
            out.append(_spec_to_str(v, indent + 2))
    else:
        out.append(" " * indent + f"{spec}")
    return "\n".join(out)


def inspect_dataset(name: str, cfg: dict):
    print(f"\n========== {name}  ({cfg['embodiment']})  v{cfg['version']} ==========")
    import tensorflow_datasets as tfds  # noqa
    builder_dir = f"{GCS_ROOT}/{name}/{cfg['version']}"
    try:
        builder = tfds.builder_from_directory(builder_dir=builder_dir)
    except Exception as e:
        print(f"  [skip] cannot open builder: {e}")
        return

    obs_spec = builder.info.features["steps"].feature["observation"]
    print("  observation feature spec:")
    print(_spec_to_str(obs_spec, indent=4))

    state_key = cfg.get("state_key")
    if state_key is None:
        print("  (no state_key in registry)")
        return

    ds = builder.as_dataset(split="train[:1]")
    ep = next(iter(ds))
    steps = list(ep["steps"].as_numpy_iterator())
    if not steps:
        print("  [skip] empty episode")
        return

    if state_key not in steps[0]["observation"]:
        print(f"  [warn] state_key '{state_key}' not in observation. "
              f"Available: {list(steps[0]['observation'].keys())}")
        return

    states = np.stack(
        [np.asarray(s["observation"][state_key]).ravel() for s in steps], axis=0
    )
    print(f"  state shape over episode: {states.shape}, dtype={states.dtype}")
    print(f"  first  state[0]: {np.array2string(states[0], precision=4, suppress_small=True)}")
    print(f"  last   state[T]: {np.array2string(states[-1], precision=4, suppress_small=True)}")
    print(f"  per-dim min:    {np.array2string(states.min(0), precision=4, suppress_small=True)}")
    print(f"  per-dim max:    {np.array2string(states.max(0), precision=4, suppress_small=True)}")
    print(f"  per-dim range:  {np.array2string(states.max(0)-states.min(0), precision=4, suppress_small=True)}")

    if name in CLAIMED_EE_DIMS:
        a, b = CLAIMED_EE_DIMS[name]
        if states.shape[1] >= b:
            xyz = states[:, a:b]
            print(f"  claimed EE xyz dims [{a}:{b}]:")
            print(f"     first :   {xyz[0]}")
            print(f"     last  :   {xyz[-1]}")
            print(f"     range :   {xyz.max(0) - xyz.min(0)}  (expect cm-to-m)")
            looks_metric = bool(np.all(np.abs(xyz).max() < 5.0)
                                and np.all((xyz.max(0) - xyz.min(0)) < 5.0))
            looks_moving = bool(np.any((xyz.max(0) - xyz.min(0)) > 1e-3))
            verdict = "✅ plausible" if (looks_metric and looks_moving) else "⚠ suspicious"
            print(f"     verdict: {verdict}")
        else:
            print(f"  [warn] state has {states.shape[1]} dims, claimed slice is [{a}:{b}]")
    else:
        print("  (no claimed EE-xyz slice; fix CLAIMED_EE_DIMS once you decide)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=None)
    args = p.parse_args()

    targets = args.datasets or list(OXE_DATASETS.keys())
    for name in targets:
        if name not in OXE_DATASETS:
            print(f"[skip] unknown {name}")
            continue
        try:
            inspect_dataset(name, OXE_DATASETS[name])
        except Exception as e:
            print(f"[fail] {name}: {e}")


if __name__ == "__main__":
    main()
