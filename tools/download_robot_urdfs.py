"""Clone robot description assets used by the URDF overlay tools.

The DROID Franka asset is intentionally kept as a local checkout under
``data/urdfs`` instead of vendoring the meshes into this repository.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


DEFAULT_REPO = "https://github.com/rerun-io/python-example-droid-dataset.git"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=Path("data/urdfs"))
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--name", default="python-example-droid-dataset")
    parser.add_argument("--depth", type=int, default=1)
    args = parser.parse_args()

    dst = args.out_dir / args.name
    if (dst / "franka_description" / "panda.urdf").exists():
        print(f"[ok] {dst} already contains franka_description/panda.urdf")
        return
    if dst.exists() and any(dst.iterdir()):
        raise SystemExit(f"{dst} exists but does not look like the expected clone")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", str(args.depth), args.repo, str(dst)]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[ok] cloned {args.repo} -> {dst}")


if __name__ == "__main__":
    main()
