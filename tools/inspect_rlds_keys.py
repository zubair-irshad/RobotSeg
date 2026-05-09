"""Print RLDS step keys and numeric observation/action vector candidates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from download_oxe_subset import GCS_ROOT, _lazy_import_tfds  # noqa: E402
from oxe_registry import OXE_DATASETS  # noqa: E402


def _walk(prefix: str, obj: Any, out: list[tuple[str, Any]]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _walk(f"{prefix}/{k}" if prefix else str(k), v, out)
    else:
        out.append((prefix, obj))


def _shape_dtype(v: Any) -> tuple[str, str]:
    arr = np.asarray(v)
    return str(tuple(arr.shape)), str(arr.dtype)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="fractal20220817_data")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--step", type=int, default=0)
    args = p.parse_args()

    cfg = OXE_DATASETS[args.dataset]
    tfds = _lazy_import_tfds()
    builder = tfds.builder_from_directory(
        builder_dir=f"{GCS_ROOT}/{args.dataset}/{cfg['version']}"
    )
    ds = builder.as_dataset(split=f"train[{args.episode}:{args.episode + 1}]")
    episode = next(iter(ds))
    step = list(episode["steps"].as_numpy_iterator())[args.step]

    leaves: list[tuple[str, Any]] = []
    _walk("", step, leaves)
    print(f"{args.dataset} episode={args.episode} step={args.step}")
    for path, value in leaves:
        shape, dtype = _shape_dtype(value)
        print(f"{path:70s} shape={shape:18s} dtype={dtype}")

    print("\nNumeric vector candidates:")
    for path, value in leaves:
        arr = np.asarray(value)
        if arr.dtype.kind in "fiu" and 1 <= arr.ndim <= 2 and arr.size <= 64:
            print(f"{path:70s} shape={arr.shape} value={arr.reshape(-1)[:12]}")


if __name__ == "__main__":
    main()
