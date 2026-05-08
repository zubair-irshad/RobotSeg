"""Inspect which gripper centroid frames pass PnP quality/proximity filters."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from pnp_oxe import _filter_frames, _read_image_size  # noqa: E402
from pnp_oxe import _entry_to_record  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oxe_root", type=Path, default=Path("data/oxe_subset"))
    parser.add_argument("--mask_root", type=Path, default=Path("data/oxe_subset_seg"))
    parser.add_argument("--dataset", default="droid")
    parser.add_argument("--episode", required=True)
    parser.add_argument("--point_key", default="eef_xyz")
    parser.add_argument("--use_observed_flag", action="store_true", default=True)
    parser.add_argument("--no_use_observed_flag", dest="use_observed_flag", action="store_false")
    parser.add_argument("--use_mask_stage_accept", action="store_true", default=True)
    parser.add_argument("--no_use_mask_stage_accept", dest="use_mask_stage_accept", action="store_false")
    parser.add_argument("--min_conf", type=float, default=0.5)
    parser.add_argument("--min_conf_max", type=float, default=0.5)
    parser.add_argument("--min_conf_p10", type=float, default=0.0)
    parser.add_argument("--min_conf_p50", type=float, default=0.0)
    parser.add_argument("--min_stability_05_07", type=float, default=0.0)
    parser.add_argument("--min_stability_05_09", type=float, default=0.0)
    parser.add_argument("--max_arm_overlap", type=float, default=1.0)
    parser.add_argument("--min_area", type=float, default=0.0002)
    parser.add_argument("--min_area_px", type=int, default=16)
    parser.add_argument("--min_gripper_area_px", type=int, default=40)
    parser.add_argument("--min_post_subtract_area_ratio", type=float, default=0.08)
    parser.add_argument("--max_area", type=float, default=0.4)
    parser.add_argument("--edge_margin", type=int, default=4)
    parser.add_argument("--reject_fragmented", action="store_true", default=True)
    parser.add_argument("--require_gripper_near_robot", action="store_true")
    parser.add_argument("--require_gripper_inside_robot", action="store_true")
    parser.add_argument("--gripper_robot_disk_radius_px", type=int, default=5)
    parser.add_argument("--min_gripper_robot_disk_frac", type=float, default=0.25)
    parser.add_argument("--robot_mask_roots", default="002,000")
    parser.add_argument("--max_gripper_robot_dist_px", type=float, default=18.0)
    parser.add_argument("--max_gripper_robot_dist_frac", type=float, default=0.035)
    parser.add_argument("--min_robot_mask_area_px", type=int, default=64)
    parser.add_argument("--skip_if_robot_mask_missing", action="store_true")
    parser.add_argument("--show_rejected", type=int, default=40)
    args = parser.parse_args()

    ep_oxe = args.oxe_root / args.dataset / args.episode
    ep_seg = args.mask_root / args.dataset / args.episode
    frames_dir = ep_oxe / "frames"
    W, H = _read_image_size(frames_dir)
    centroids = json.loads((ep_seg / "001" / "centroids.json").read_text())
    with np.load(ep_oxe / "trajectory.npz", allow_pickle=True) as traj:
        if args.point_key in traj.files:
            pts = np.asarray(traj[args.point_key], dtype=np.float64)
        elif "eef_xyz" in traj.files:
            pts = np.asarray(traj["eef_xyz"], dtype=np.float64)
        else:
            pts = np.asarray(traj["state"], dtype=np.float64)[:, :3]

    _, _, kept, rejected, _ = _filter_frames(
        centroids,
        pts,
        W,
        H,
        args,
        ee_R_seq=None,
        ep_dir_seg=ep_seg,
    )
    counts = {}
    for _, reason in rejected:
        key = reason.split("(")[0]
        counts[key] = counts.get(key, 0) + 1
    print(f"[{args.dataset}/{args.episode}] kept={len(kept)} rejected={len(rejected)} total={len(centroids)}")
    print("rejection breakdown:")
    for key, val in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {key:24s} {val}")
    if args.show_rejected > 0:
        print("\nfirst rejected:")
        for stem, reason in rejected[:args.show_rejected]:
            rec = _entry_to_record(centroids.get(stem))
            post = rec["area_frac"]
            pre = rec.get("pre_subtract_area_frac", post)
            ratio = post / pre if pre > 0 else 0.0
            print(
                f"  {stem}: {reason} "
                f"centroid={rec['centroid']} observed={rec['observed']} "
                f"accepted={rec['accepted_for_pnp']} mask_reason={rec['reject_reason']} "
                f"area={post:.5f} pre={pre:.5f} ratio={ratio:.3f} "
                f"conf_p10={rec['conf_p10']:.2f} conf_p50={rec['conf_p50']:.2f} "
                f"conf_max={rec['conf_max']:.2f} "
                f"stab07={rec['stability_05_07']:.2f} comps={rec['n_components']} "
                f"arm_overlap={rec['arm_overlap_frac']:.2f}"
            )


if __name__ == "__main__":
    main()
