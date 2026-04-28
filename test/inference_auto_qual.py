"""Qualitative-only RobotSeg inference on arbitrary image-folder sequences
(e.g. the OXE subset produced by tools/download_oxe_subset.py).

Supports running multiple categories (e.g. arm + gripper) in a single pass.
Per-category binary masks are saved under <save_root>/<seq>/<instance_id>/.
When more than one category is requested, a combined overlay is produced
under <save_root>/<seq>/combined/, with each category drawn in its own color
and the gripper centroid (treated as the end-effector) marked.

Layout produced:
    <save_root>/<seq_name>/<instance_id>/
      00000.png              # binary mask
      00000_overlay.jpg      # per-category overlay (if --save_overlay)
      centroids.json
    <save_root>/<seq_name>/combined/
      00000.jpg              # arm + gripper overlay + EE centroid marker
"""

import os
import json
import argparse
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from natsort import natsorted
from tqdm import tqdm


CATEGORY2ID = {"arm": "000", "gripper": "001", "robot": "002"}
# BGR colors per category
CATEGORY_COLOR = {
    "arm":     np.array([0, 255, 0],   dtype=np.uint8),  # green
    "gripper": np.array([0, 0, 255],   dtype=np.uint8),  # red
    "robot":   np.array([255, 0, 0],   dtype=np.uint8),  # blue
}


def _save_png(path, arr):
    cv2.imwrite(path, arr)


def _save_overlay(path, image_bgr, mask_u8, color, alpha=0.5):
    overlay = image_bgr.copy()
    sel = mask_u8 > 127
    overlay[sel] = ((1 - alpha) * image_bgr[sel] + alpha * color).astype(np.uint8)
    cv2.imwrite(path, overlay)


def _save_centroid_viz(path, image_bgr, centroid, color=(0, 0, 255)):
    viz = image_bgr.copy()
    if centroid is not None:
        cx, cy = int(round(centroid[0])), int(round(centroid[1]))
        cv2.drawMarker(viz, (cx, cy), color,
                       markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2)
        cv2.circle(viz, (cx, cy), 6, (0, 255, 255), 2)
    cv2.imwrite(path, viz)


def _save_combined(path, image_bgr, masks_by_cat, ee_centroid, alpha=0.5):
    """Overlay multiple category masks with their colors, and mark EE centroid."""
    viz = image_bgr.copy().astype(np.float32)
    for cat, mask in masks_by_cat.items():
        if mask is None:
            continue
        sel = mask > 127
        if not np.any(sel):
            continue
        color = CATEGORY_COLOR[cat].astype(np.float32)
        viz[sel] = (1 - alpha) * viz[sel] + alpha * color
    viz = viz.astype(np.uint8)

    if ee_centroid is not None:
        cx, cy = int(round(ee_centroid[0])), int(round(ee_centroid[1]))
        cv2.drawMarker(viz, (cx, cy), (255, 255, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=26, thickness=3)
        cv2.circle(viz, (cx, cy), 7, (0, 255, 255), 2)
        cv2.putText(viz, "EE", (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(path, viz)


def _entry_centroid(entry):
    """Extract a centroid [cx,cy] from a centroids.json entry that may be
    legacy list form or new dict form."""
    if entry is None:
        return None
    if isinstance(entry, dict):
        return entry.get("centroid")
    return entry


def _mask_stats(mask_u8, prob_u8=None):
    """Per-frame mask stats: centroid + confidence + area + #components.

    Confidence is mean/max sigmoid probability inside the binary mask.
    Without prob_u8, confidence is reported as 1.0 (legacy)."""
    sel = mask_u8 > 127
    h, w = mask_u8.shape
    n = int(sel.sum())
    if n == 0:
        return {
            "centroid": None,
            "area_frac": 0.0,
            "conf_mean": 0.0,
            "conf_max": 0.0,
            "n_components": 0,
        }
    ys, xs = np.where(sel)
    centroid = [float(xs.mean()), float(ys.mean())]
    n_components = int(cv2.connectedComponents(mask_u8 // 255)[0] - 1)
    if prob_u8 is not None:
        p = prob_u8[sel].astype(np.float32) / 255.0
        conf_mean = float(p.mean())
        conf_max = float(p.max())
    else:
        conf_mean = 1.0
        conf_max = 1.0
    return {
        "centroid": centroid,
        "area_frac": n / float(h * w),
        "conf_mean": conf_mean,
        "conf_max": conf_max,
        "n_components": n_components,
    }


def _run_category(predictor, torch, seq_path, category, start_idx=0):
    """Run inference for a single category over the whole sequence.

    Returns (frame_indices_sorted, masks_u8 [N,H,W], probs_u8 [N,H,W])
    where probs_u8 is sigmoid(logits) quantized to uint8 (per-pixel
    confidence)."""
    state = predictor.init_state(
        video_path=seq_path,
        async_loading_frames=False,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
    )
    _, object_ids, masks_logits = predictor.add_new_robot(
        inference_state=state,
        frame_idx=start_idx,
        obj_id=0,
        robot=category,
    )

    gpu_logits = [masks_logits[0][0]]  # logits for object 0
    frame_indices = [start_idx]

    for out_idx, out_obj_ids, out_logits in predictor.propagate_in_video(
        inference_state=state, robot=category
    ):
        if out_idx == start_idx:
            continue
        gpu_logits.append(out_logits[0][0])
        frame_indices.append(out_idx)

    logits = torch.stack(gpu_logits, 0)
    probs = torch.sigmoid(logits)
    masks_u8 = ((probs > 0.5).cpu().numpy() * 255).astype(np.uint8)
    probs_u8 = (probs.float().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

    order = np.argsort(frame_indices)
    frame_indices = [frame_indices[i] for i in order]
    masks_u8 = masks_u8[order]
    probs_u8 = probs_u8[order]
    return frame_indices, masks_u8, probs_u8


def process_sequences(args, gpu_id, seq_list):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"

    from robotseg.build_robotseg import build_robotseg_video_predictor

    model_cfg = f"../robotseg/configs/{args.yaml}"
    checkpoint = f"../checkpoints/{args.ckpt}.pt"
    predictor = build_robotseg_video_predictor(model_cfg, checkpoint)

    save_pool = ThreadPoolExecutor(max_workers=8)

    for seq_name in tqdm(seq_list, desc=f"GPU {gpu_id}"):
        seq_path = os.path.join(args.image_root, seq_name)
        frame_files = natsorted([
            f for f in os.listdir(seq_path) if f.lower().endswith((".jpg", ".png"))
        ])
        if not frame_files:
            continue
        frame_stems = [os.path.splitext(f)[0] for f in frame_files]

        # Decide which categories still need running
        cats_to_run = []
        per_cat_outdirs = {}
        for cat in args.categories:
            instance_id = CATEGORY2ID[cat]
            out_dir = os.path.join(args.save_root, seq_name, instance_id)
            per_cat_outdirs[cat] = out_dir
            if os.path.isdir(out_dir) and os.listdir(out_dir) and not args.overwrite:
                continue
            cats_to_run.append(cat)
            os.makedirs(out_dir, exist_ok=True)

        # If everything already exists and combined is not requested, skip.
        do_combined = (
            (args.save_overlay or args.save_centroid_viz)
            and len(args.categories) > 1
        )
        combined_dir = os.path.join(args.save_root, seq_name, "combined")
        if do_combined:
            os.makedirs(combined_dir, exist_ok=True)
        if not cats_to_run and not (do_combined and args.overwrite):
            continue

        # Load images once if we'll need them for any visualization.
        need_images = args.save_overlay or args.save_centroid_viz or do_combined

        # Run each category that needs running.
        masks_by_cat = {}     # cat -> dict[stem] = mask_u8
        probs_by_cat = {}     # cat -> dict[stem] = prob_u8
        centroids_by_cat = {} # cat -> dict[stem] = stats dict

        # Also pull existing masks for cats already on disk so combined can use them.
        for cat in args.categories:
            if cat in cats_to_run:
                continue
            out_dir = per_cat_outdirs[cat]
            d = {}
            for stem in frame_stems:
                p = os.path.join(out_dir, f"{stem}.png")
                if os.path.exists(p):
                    d[stem] = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if d:
                masks_by_cat[cat] = d
                cj = os.path.join(out_dir, "centroids.json")
                if os.path.exists(cj):
                    with open(cj) as f:
                        centroids_by_cat[cat] = json.load(f)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for cat in cats_to_run:
                instance_id = CATEGORY2ID[cat]
                out_dir = per_cat_outdirs[cat]
                color = CATEGORY_COLOR[cat]
                frame_indices, stacked, probs = _run_category(predictor, torch, seq_path, cat)

                stats_by_stem = {}
                cat_masks = {}
                cat_probs = {}
                for k, idx in enumerate(frame_indices):
                    stem = frame_stems[idx]
                    mask = stacked[k]
                    prob = probs[k]
                    cat_masks[stem] = mask
                    cat_probs[stem] = prob
                    save_pool.submit(_save_png, os.path.join(out_dir, f"{stem}.png"), mask)
                    if args.save_prob:
                        save_pool.submit(
                            _save_png, os.path.join(out_dir, f"{stem}_prob.png"), prob
                        )
                    stats_by_stem[stem] = _mask_stats(mask, prob)
                    centroid = stats_by_stem[stem]["centroid"]

                    if (args.save_overlay or args.save_centroid_viz) and need_images:
                        img_path = os.path.join(seq_path, frame_files[idx])
                        img = cv2.imread(img_path)
                        if img is not None:
                            if args.save_overlay:
                                save_pool.submit(
                                    _save_overlay,
                                    os.path.join(out_dir, f"{stem}_overlay.jpg"),
                                    img, mask, color,
                                )
                            if args.save_centroid_viz:
                                save_pool.submit(
                                    _save_centroid_viz,
                                    os.path.join(out_dir, f"{stem}_centroid.jpg"),
                                    img, centroid,
                                )

                masks_by_cat[cat] = cat_masks
                probs_by_cat[cat] = cat_probs
                centroids_by_cat[cat] = stats_by_stem
                with open(os.path.join(out_dir, "centroids.json"), "w") as f:
                    json.dump(stats_by_stem, f, indent=2)

        # Optional: subtract arm mask out of gripper mask, since the arm
        # prediction is reliable and the gripper one sometimes leaks onto
        # the arm shoulder/body. Recompute gripper centroid + stats from
        # the disjoint mask and overwrite 001/<stem>.png + centroids.json.
        if (args.subtract_arm_from_gripper
                and "arm" in masks_by_cat and "gripper" in masks_by_cat):
            grip_dir = per_cat_outdirs["gripper"]
            arm_masks = masks_by_cat["arm"]
            grip_masks = masks_by_cat["gripper"]
            grip_probs = probs_by_cat.get("gripper", {})
            new_stats = {}
            for stem, gmask in list(grip_masks.items()):
                amask = arm_masks.get(stem)
                if amask is None:
                    new_stats[stem] = _mask_stats(gmask, grip_probs.get(stem))
                    continue
                disjoint = np.where(amask > 127, 0, gmask).astype(np.uint8)
                grip_masks[stem] = disjoint
                save_pool.submit(
                    _save_png, os.path.join(grip_dir, f"{stem}.png"), disjoint
                )
                if args.save_prob and stem in grip_probs:
                    new_prob = np.where(amask > 127, 0, grip_probs[stem]).astype(np.uint8)
                    grip_probs[stem] = new_prob
                    save_pool.submit(
                        _save_png,
                        os.path.join(grip_dir, f"{stem}_prob.png"), new_prob,
                    )
                new_stats[stem] = _mask_stats(disjoint, grip_probs.get(stem))
            centroids_by_cat["gripper"] = new_stats
            with open(os.path.join(grip_dir, "centroids.json"), "w") as f:
                json.dump(new_stats, f, indent=2)

        # Combined overlay (one image per frame, all categories together,
        # with EE = gripper centroid annotated when available).
        if do_combined:
            ee_cat = "gripper" if "gripper" in masks_by_cat else None
            for idx, stem in enumerate(frame_stems):
                per_cat_mask = {
                    cat: masks_by_cat.get(cat, {}).get(stem)
                    for cat in args.categories
                }
                if all(m is None for m in per_cat_mask.values()):
                    continue
                img_path = os.path.join(seq_path, frame_files[idx])
                img = cv2.imread(img_path)
                if img is None:
                    continue
                ee = None
                if ee_cat is not None:
                    ee = _entry_centroid(centroids_by_cat.get(ee_cat, {}).get(stem))
                save_pool.submit(
                    _save_combined,
                    os.path.join(combined_dir, f"{stem}.jpg"),
                    img, per_cat_mask, ee,
                )

    save_pool.shutdown(wait=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_root", required=True,
                   help="Dir containing per-sequence frame subdirs.")
    p.add_argument("--save_root", required=True)
    p.add_argument("--category", default=None, choices=list(CATEGORY2ID),
                   help="Single category (legacy). Use --categories to pass several.")
    p.add_argument("--categories", default=None,
                   help="Comma-separated categories, e.g. arm,gripper")
    p.add_argument("--ckpt", default="robotseg")
    p.add_argument("--yaml", default="robotseg-infer.yaml")
    p.add_argument("--save_overlay", action="store_true")
    p.add_argument("--save_centroid_viz", action="store_true",
                   help="Save per-frame image with mask centroid marker drawn.")
    p.add_argument("--save_prob", action="store_true",
                   help="Save per-frame sigmoid probability map as <stem>_prob.png.")
    p.add_argument("--subtract_arm_from_gripper", action="store_true", default=True,
                   help="Set gripper_mask = gripper_mask AND NOT arm_mask before "
                        "computing centroids. Arm prediction is more reliable, so "
                        "subtracting it out cleans up gripper-mask bleed-over.")
    p.add_argument("--no_subtract_arm_from_gripper", dest="subtract_arm_from_gripper",
                   action="store_false")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.categories:
        args.categories = [c.strip().lower() for c in args.categories.split(",") if c.strip()]
    elif args.category:
        args.categories = [args.category.lower()]
    else:
        raise SystemExit("Pass --categories arm,gripper (or legacy --category).")
    for c in args.categories:
        if c not in CATEGORY2ID:
            raise SystemExit(f"Unknown category {c}; valid: {list(CATEGORY2ID)}")

    seqs = sorted(
        d for d in os.listdir(args.image_root)
        if os.path.isdir(os.path.join(args.image_root, d)) and not d.startswith(".")
    )
    if not seqs:
        raise SystemExit(f"No sequence subdirs under {args.image_root}")

    # If sequences are nested one level deeper (our OXE layout puts
    # frames under <episode>/frames/*.jpg), rewrite image_root to point
    # at <episode>/frames by creating a flattened view.
    expanded = []
    new_root_map = {}
    for s in seqs:
        maybe_frames = os.path.join(args.image_root, s, "frames")
        if os.path.isdir(maybe_frames):
            expanded.append(s)
            new_root_map[s] = maybe_frames
    if expanded and len(expanded) == len(seqs):
        link_root = os.path.join(args.save_root, "_frames_view")
        os.makedirs(link_root, exist_ok=True)
        for s, real in new_root_map.items():
            link = os.path.join(link_root, s)
            if not os.path.exists(link):
                os.symlink(os.path.abspath(real), link)
        args.image_root = link_root

    mp.set_start_method("spawn")
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpu_ids = [int(x) for x in cvd.split(",") if x.strip()]

    gpu_sequences = [[] for _ in gpu_ids]
    for i, s in enumerate(seqs):
        gpu_sequences[i % len(gpu_ids)].append(s)

    procs = []
    for i, gid in enumerate(gpu_ids):
        if not gpu_sequences[i]:
            continue
        proc = mp.Process(target=process_sequences, args=(args, gid, gpu_sequences[i]))
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join()


if __name__ == "__main__":
    main()
