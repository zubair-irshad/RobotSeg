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

try:
    from utils import guided_refine_mask
except ImportError:
    guided_refine_mask = None

_guided_filter_warning_shown = False


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


def _apply_guided_refine(mask, image_bgr):
    """Apply author guided refinement when OpenCV exposes ximgproc.guidedFilter."""
    global _guided_filter_warning_shown
    if guided_refine_mask is None:
        raise RuntimeError(
            "--guided_filter requested, but test/utils.py guided_refine_mask "
            "could not be imported"
        )
    try:
        return guided_refine_mask(mask, image_bgr)
    except AttributeError as e:
        if "guidedFilter" not in str(e):
            raise
        if not _guided_filter_warning_shown:
            print(
                "[warn] cv2.ximgproc.guidedFilter is unavailable; "
                "saving raw masks for --guided_filter frames. Install an "
                "OpenCV contrib build to reproduce author guided refinement.",
                flush=True,
            )
            _guided_filter_warning_shown = True
        return mask


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
            "conf_p10": 0.0,
            "conf_p50": 0.0,
            "conf_p90": 0.0,
            "stability_05_07": 0.0,
            "stability_05_09": 0.0,
            "n_components": 0,
        }
    ys, xs = np.where(sel)
    centroid = [float(xs.mean()), float(ys.mean())]
    n_components = int(cv2.connectedComponents(mask_u8 // 255)[0] - 1)
    if prob_u8 is not None:
        prob = prob_u8.astype(np.float32) / 255.0
        p = prob[sel]
        conf_mean = float(p.mean())
        conf_max = float(p.max())
        conf_p10, conf_p50, conf_p90 = [
            float(x) for x in np.percentile(p, [10, 50, 90])
        ]
        area05 = max(1, int((prob > 0.5).sum()))
        stability_05_07 = float((prob > 0.7).sum() / area05)
        stability_05_09 = float((prob > 0.9).sum() / area05)
    else:
        conf_mean = 1.0
        conf_max = 1.0
        conf_p10 = 1.0
        conf_p50 = 1.0
        conf_p90 = 1.0
        stability_05_07 = 1.0
        stability_05_09 = 1.0
    return {
        "centroid": centroid,
        "area_frac": n / float(h * w),
        "conf_mean": conf_mean,
        "conf_max": conf_max,
        "conf_p10": conf_p10,
        "conf_p50": conf_p50,
        "conf_p90": conf_p90,
        "stability_05_07": stability_05_07,
        "stability_05_09": stability_05_09,
        "n_components": n_components,
    }


def _select_frame_files(frame_files, args):
    """Return a deterministic subset while preserving original frame names."""
    if args.frame_stride > 1:
        frame_files = frame_files[::args.frame_stride]
    if args.max_frames_per_seq > 0 and len(frame_files) > args.max_frames_per_seq:
        idx = np.linspace(
            0, len(frame_files) - 1, args.max_frames_per_seq, dtype=np.int64
        )
        frame_files = [frame_files[int(i)] for i in idx]
    return frame_files


def _maybe_make_subset_view(orig_seq_path, frame_files, all_frame_files, args):
    """Create a frame-folder view when inference should only see a subset."""
    if len(frame_files) == len(all_frame_files):
        return orig_seq_path, False
    view_dir = os.path.join(
        args.save_root,
        "_subset_views",
        os.path.basename(orig_seq_path)
        + f"__stride{args.frame_stride}__max{args.max_frames_per_seq}",
    )
    os.makedirs(view_dir, exist_ok=True)
    for f in frame_files:
        src = os.path.abspath(os.path.join(orig_seq_path, f))
        dst = os.path.join(view_dir, f)
        if os.path.lexists(dst):
            continue
        try:
            os.symlink(src, dst)
        except OSError:
            import shutil
            shutil.copy2(src, dst)
    return view_dir, True


def _maybe_make_lowres_view(orig_seq_path, frame_files, args):
    """If --infer_max_side is set and frames exceed it, write a resized
    copy of the frames to a side dir and return (low_res_path, scale).
    Otherwise return (orig_seq_path, 1.0)."""
    if args.infer_max_side <= 0:
        return orig_seq_path, 1.0
    sample = cv2.imread(os.path.join(orig_seq_path, frame_files[0]))
    if sample is None:
        return orig_seq_path, 1.0
    H, W = sample.shape[:2]
    longest = max(H, W)
    if longest <= args.infer_max_side:
        return orig_seq_path, 1.0
    scale = args.infer_max_side / float(longest)
    new_w = int(round(W * scale))
    new_h = int(round(H * scale))
    lowres_dir = orig_seq_path.rstrip("/") + f"__lowres{args.infer_max_side}"
    os.makedirs(lowres_dir, exist_ok=True)
    for f in frame_files:
        dst = os.path.join(lowres_dir, f)
        if os.path.exists(dst):
            continue
        img = cv2.imread(os.path.join(orig_seq_path, f))
        if img is None:
            continue
        cv2.imwrite(dst, cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA))
    return lowres_dir, scale


def _run_category(predictor, torch, seq_path, category, args, start_idx=0):
    """Run inference for a single category over the whole sequence.

    Returns (frame_indices_sorted, masks_u8 [N,H,W], probs_u8 [N,H,W])
    where probs_u8 is sigmoid(logits) quantized to uint8 (per-pixel
    confidence). Cleans up the predictor state + clears the CUDA cache
    on exit so long episode runs don't leak GPU memory across episodes."""
    import gc

    state = None
    gpu_logits = None
    masks_logits = None
    try:
        state = predictor.init_state(
            video_path=seq_path,
            async_loading_frames=False,
            offload_video_to_cpu=args.offload_to_cpu,
            offload_state_to_cpu=args.offload_to_cpu,
        )
        _, object_ids, masks_logits = predictor.add_new_robot(
            inference_state=state,
            frame_idx=start_idx,
            obj_id=0,
            robot=category,
        )

        # Move per-frame logits to CPU as we go so they don't pile up on GPU
        # for long trajectories.
        cpu_logits = [masks_logits[0][0].detach().cpu()]
        frame_indices = [start_idx]

        for out_idx, out_obj_ids, out_logits in predictor.propagate_in_video(
            inference_state=state, robot=category
        ):
            if out_idx == start_idx:
                continue
            cpu_logits.append(out_logits[0][0].detach().cpu())
            frame_indices.append(out_idx)

        logits = torch.stack(cpu_logits, 0)
        probs = torch.sigmoid(logits)
        masks_u8 = ((probs > 0.5).numpy() * 255).astype(np.uint8)
        probs_u8 = (probs.float().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        order = np.argsort(frame_indices)
        frame_indices = [frame_indices[i] for i in order]
        masks_u8 = masks_u8[order]
        probs_u8 = probs_u8[order]
        return frame_indices, masks_u8, probs_u8
    finally:
        # Hard cleanup so the next episode starts with a fresh allocator.
        try:
            del state, masks_logits, gpu_logits
        except NameError:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


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
        orig_seq_path = os.path.join(args.image_root, seq_name)
        all_frame_files = natsorted([
            f for f in os.listdir(orig_seq_path) if f.lower().endswith((".jpg", ".png"))
        ])
        frame_files = _select_frame_files(all_frame_files, args)
        if not frame_files:
            continue
        frame_stems = [os.path.splitext(f)[0] for f in frame_files]

        # Optional half/quarter-res inference: write a downscaled view of the
        # frames into a temp dir, run the predictor on those, then upsample
        # masks back to the original size. Trades a bit of segmentation
        # quality for ~scale^2 GPU-memory savings.
        input_seq_path, made_subset_view = _maybe_make_subset_view(
            orig_seq_path, frame_files, all_frame_files, args
        )
        seq_path, downscale = _maybe_make_lowres_view(
            input_seq_path, frame_files, args
        )
        # Read the original frame size once for upsampling masks back later.
        first_full = cv2.imread(os.path.join(orig_seq_path, frame_files[0]))
        orig_h, orig_w = first_full.shape[:2] if first_full is not None else (None, None)

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
                frame_indices, stacked, probs = _run_category(
                    predictor, torch, seq_path, cat, args
                )
                # If we ran on a downscaled view, upsample masks/probs back
                # to the original resolution before saving anything.
                if downscale != 1.0 and orig_h is not None:
                    stacked_up = np.empty((len(stacked), orig_h, orig_w), dtype=np.uint8)
                    probs_up = np.empty_like(stacked_up)
                    for k in range(len(stacked)):
                        stacked_up[k] = cv2.resize(
                            stacked[k], (orig_w, orig_h),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        probs_up[k] = cv2.resize(
                            probs[k], (orig_w, orig_h),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    stacked = stacked_up
                    probs = probs_up

                stats_by_stem = {}
                cat_masks = {}
                cat_probs = {}
                for k, idx in enumerate(frame_indices):
                    stem = frame_stems[idx]
                    mask = stacked[k]
                    prob = probs[k]
                    img = None
                    if args.guided_filter and idx != 0:
                        img_path = os.path.join(orig_seq_path, frame_files[idx])
                        img = cv2.imread(img_path)
                        if img is not None:
                            mask = _apply_guided_refine(mask, img)
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
                        if img is None:
                            img_path = os.path.join(orig_seq_path, frame_files[idx])
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
                    s = _mask_stats(gmask, grip_probs.get(stem))
                    s["arm_overlap_frac"] = 0.0
                    s["pre_subtract_area_frac"] = s["area_frac"]
                    s["observed"] = bool(s["centroid"] is not None
                                         and s["area_frac"] > 0
                                         and s["conf_max"] >= 0.7)
                    new_stats[stem] = s
                    continue
                # How much of the original gripper mask was actually arm?
                # arm_overlap_frac ≈ 1 means the gripper category was
                # almost entirely bleed-over onto the arm — strong signal
                # that the gripper itself was not visible.
                gpix_total = int((gmask > 127).sum())
                gpix_overlap = int(((gmask > 127) & (amask > 127)).sum())
                arm_overlap = (gpix_overlap / gpix_total) if gpix_total > 0 else 1.0

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
                # All stats here are computed on the POST-subtraction mask
                # (the actual gripper) — that's what PnP will use.
                s = _mask_stats(disjoint, grip_probs.get(stem))
                s["arm_overlap_frac"] = float(arm_overlap)  # informational only
                s["pre_subtract_area_frac"] = float(
                    gpix_total / float(gmask.shape[0] * gmask.shape[1])
                )
                # "Gripper observed" verdict, judged purely on what survived
                # the subtraction: non-empty, single component, peak confidence
                # high enough, and area at least a small floor.
                disjoint_area = int((disjoint > 127).sum())
                s["observed"] = bool(
                    s["centroid"] is not None
                    and disjoint_area >= 16  # absolute pixel floor
                    and s["conf_max"] >= 0.7
                    and s["n_components"] == 1
                )
                new_stats[stem] = s
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
                img_path = os.path.join(orig_seq_path, frame_files[idx])
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

        # Clean up the temporary low-res view (if any) for this episode.
        if (downscale != 1.0 and seq_path != input_seq_path) or made_subset_view:
            try:
                import shutil
                if downscale != 1.0 and seq_path != input_seq_path:
                    shutil.rmtree(seq_path, ignore_errors=True)
                if made_subset_view:
                    shutil.rmtree(input_seq_path, ignore_errors=True)
            except Exception:
                pass

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
    p.add_argument("--guided_filter", action="store_true",
                   help="Apply the author-provided guided_refine_mask before saving masks.")
    p.add_argument("--offload_to_cpu", action="store_true", default=True,
                   help="Pass offload_video_to_cpu=True / offload_state_to_cpu=True "
                        "to predictor.init_state. Big memory win for long episodes; "
                        "small speed hit.")
    p.add_argument("--no_offload_to_cpu", dest="offload_to_cpu", action="store_false")
    p.add_argument("--infer_max_side", type=int, default=0,
                   help="If >0, downscale frames so the longest side <= N before "
                        "running the predictor. Masks are upsampled back to the "
                        "original resolution. Try 256 or 384 if you OOM.")
    p.add_argument("--frame_stride", type=int, default=1,
                   help="Keep one of every N frames per sequence before inference.")
    p.add_argument("--max_frames_per_seq", type=int, default=0,
                   help="If >0, uniformly sample at most this many frames per sequence.")
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
        if proc.exitcode != 0:
            raise SystemExit(f"Worker process {proc.pid} failed with exit code {proc.exitcode}")


if __name__ == "__main__":
    main()
