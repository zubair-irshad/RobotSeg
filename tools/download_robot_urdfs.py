"""Fetch URDFs for the embodiments in tools/oxe_registry.py.

Shallow-clones community URDF repos into <out_dir>/<embodiment>/ and
records the canonical URDF path + suggested mesh_dir in `urdf_map.json`,
which tools/viz_cam2base_urdf.py can consume.

Usage:
    python tools/download_robot_urdfs.py
    python tools/download_robot_urdfs.py --out_dir ~/RobotSeg/data/urdfs
    python tools/download_robot_urdfs.py --only Franka UR5

Notes:
- Google Everyday Robot has no public URDF — skipped.
- MobileALOHA = 2 × ViperX 300; we point to the Interbotix description.
- Some repos are large (xarm, sawyer ship many CAD meshes). Shallow clones
  keep this manageable but they're still ~50–200 MB each.
- Each repo's URDF may need extra surgery (xacro -> urdf, ros_pkg path
  rewriting). yourdfpy handles plain URDFs; for xacro you'll need to run
  `xacro panda.urdf.xacro > panda.urdf` once.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

# Embodiment -> { repo, urdf (relative to clone), mesh_dir (relative), notes }
URDF_SOURCES = {
    "Franka": {
        "repo": "https://github.com/rerun-io/franka_description.git",
        "urdf": "panda.urdf",
        "mesh_dir": "meshes",
        "notes": "rerun-io fork ships a flat panda.urdf with the 2f-85 hand.",
    },
    "UR5": {
        "repo": "https://github.com/ros-industrial/universal_robot.git",
        "urdf": "ur_description/urdf/ur5.urdf.xacro",
        "mesh_dir": "ur_description/meshes",
        "notes": "Run `xacro ur5.urdf.xacro -o ur5.urdf` to flatten.",
    },
    "WidowX": {
        "repo": "https://github.com/Interbotix/interbotix_ros_manipulators.git",
        "urdf": "interbotix_ros_xsarms/interbotix_xsarm_descriptions/urdf/wx250s.urdf.xacro",
        "mesh_dir": "interbotix_ros_xsarms/interbotix_xsarm_descriptions/meshes",
        "notes": "Bridge uses WidowX 250s. Xacro flatten required.",
    },
    "xArm": {
        "repo": "https://github.com/xArm-Developer/xarm_ros.git",
        "urdf": "xarm_description/urdf/xarm7.urdf.xacro",
        "mesh_dir": "xarm_description/meshes",
        "notes": "UCSD pick-and-place uses xArm7. Xacro flatten required.",
    },
    "Hello Stretch": {
        "repo": "https://github.com/hello-robot/stretch_urdf.git",
        "urdf": "stretch_urdf/RE2V0/stretch_description_RE2V0.urdf",
        "mesh_dir": "stretch_urdf/RE2V0/meshes",
        "notes": "Stretch RE2 generation; check generation matches your data.",
    },
    "Sawyer": {
        "repo": "https://github.com/RethinkRobotics/sawyer_robot.git",
        "urdf": "sawyer_description/urdf/sawyer.urdf",
        "mesh_dir": "sawyer_description/meshes",
        "notes": "RoboTurk = Sawyer. URDF ships pre-flattened.",
    },
    "Kuka iiwa": {
        "repo": "https://github.com/IFL-CAMP/iiwa_stack.git",
        "urdf": "iiwa_description/urdf/iiwa14.urdf.xacro",
        "mesh_dir": "iiwa_description/meshes",
        "notes": "Kuka uses iiwa14. Xacro flatten required.",
    },
    "Fanuc Mate": {
        "repo": "https://github.com/ros-industrial/fanuc.git",
        "urdf": "fanuc_lrmate200ic_support/urdf/lrmate200ic.xacro",
        "mesh_dir": "fanuc_lrmate200ic_support/meshes",
        "notes": "fanuc_manipulation_v2 = LR Mate 200iC family.",
    },
    "MobileALOHA": {
        "repo": "https://github.com/Interbotix/interbotix_ros_manipulators.git",
        "urdf": "interbotix_ros_xsarms/interbotix_xsarm_descriptions/urdf/vx300s.urdf.xacro",
        "mesh_dir": "interbotix_ros_xsarms/interbotix_xsarm_descriptions/meshes",
        "notes": "Two ViperX 300s arms; URDF here is one arm only.",
    },
    "Google Everyday Robot": {
        "repo": None,
        "notes": "No public URDF available.",
    },
}


def _git_clone(repo: str, dst: Path, depth: int = 1) -> bool:
    if dst.exists() and any(dst.iterdir()):
        print(f"  [keep] {dst} (already populated)")
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", str(depth), repo, str(dst)]
    print("  $", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [fail] {repo}\n{r.stderr}")
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        return False
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=Path,
                   default=Path.home() / "RobotSeg" / "data" / "urdfs")
    p.add_argument("--only", nargs="+", default=None,
                   help="Embodiment names to fetch (default: all known).")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets = args.only or list(URDF_SOURCES.keys())

    map_path = args.out_dir / "urdf_map.json"
    urdf_map = {}
    if map_path.exists():
        urdf_map = json.loads(map_path.read_text())

    for embodiment in targets:
        if embodiment not in URDF_SOURCES:
            print(f"[skip] unknown embodiment: {embodiment}")
            continue
        spec = URDF_SOURCES[embodiment]
        if spec.get("repo") is None:
            print(f"[skip] {embodiment}: {spec.get('notes', 'no repo')}")
            continue

        dst = args.out_dir / embodiment.replace(" ", "_")
        print(f"=== {embodiment}")
        ok = _git_clone(spec["repo"], dst)
        urdf_path = dst / spec["urdf"] if ok else None
        mesh_dir = dst / spec["mesh_dir"] if ok else None

        urdf_map[embodiment] = {
            "clone_dir": str(dst),
            "urdf": str(urdf_path) if urdf_path else None,
            "mesh_dir": str(mesh_dir) if mesh_dir else None,
            "urdf_exists": urdf_path.exists() if urdf_path else False,
            "needs_xacro": str(spec["urdf"]).endswith(".xacro"),
            "notes": spec.get("notes", ""),
        }

    map_path.write_text(json.dumps(urdf_map, indent=2))
    print(f"\nWrote {map_path}")

    # Final summary
    needs_xacro = [k for k, v in urdf_map.items() if v.get("needs_xacro")]
    if needs_xacro:
        print("\nThese repos ship xacro (run once before use):")
        for k in needs_xacro:
            v = urdf_map[k]
            stem = Path(v["urdf"]).stem.replace(".urdf", "")
            out = Path(v["urdf"]).with_suffix("").with_suffix(".urdf")
            print(f"  xacro {v['urdf']} -o {out}")


if __name__ == "__main__":
    main()
