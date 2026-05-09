#!/usr/bin/env bash
# Convert MuJoCo Menagerie Google Robot MJCF to a URDF package.
#
# This gives a URDF asset tree for future rendering work. Fractal/OXE does not
# currently provide Google Robot joint positions in the local trajectory.npz, so
# this URDF cannot be used for articulated per-frame projection unless joint
# states are also extracted or reconstructed.

set -euo pipefail

ROOT="${ROOT:-$HOME/RobotSeg/data/urdfs/google_robot}"
MENAGERIE_DIR="${MENAGERIE_DIR:-$ROOT/mujoco_menagerie}"
OUT_PKG="${OUT_PKG:-$ROOT/google_robot_description}"

python -m pip install mjcf-urdf-simple-converter

if [[ ! -d "$MENAGERIE_DIR/.git" ]]; then
  mkdir -p "$ROOT"
  git clone https://github.com/google-deepmind/mujoco_menagerie.git "$MENAGERIE_DIR"
fi

mkdir -p "$OUT_PKG/urdf" "$OUT_PKG/meshes"

python - "$MENAGERIE_DIR" "$OUT_PKG" <<'PY'
from pathlib import Path
import shutil
import sys

from mjcf_urdf_simple_converter import convert

menagerie_root = Path(sys.argv[1])
out_pkg = Path(sys.argv[2])
src_xml = menagerie_root / "google_robot" / "robot.xml"
out_urdf = out_pkg / "urdf" / "google_robot.urdf"
out_mesh_dir = out_pkg / "meshes"

for ext in ("*.obj", "*.stl", "*.dae", "*.ply"):
    for mesh in (menagerie_root / "google_robot").rglob(ext):
        shutil.copy2(mesh, out_mesh_dir / mesh.name)

convert(
    str(src_xml),
    str(out_urdf),
    asset_file_prefix="package://google_robot_description/meshes/",
)
print(f"Wrote: {out_urdf}")
PY

echo "URDF package: $OUT_PKG"
