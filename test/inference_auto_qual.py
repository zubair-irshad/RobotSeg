"""Qualitative-only RobotSeg inference on arbitrary image-folder sequences
(e.g. the OXE subset produced by tools/download_oxe_subset.py).

Derived from inference_auto_semi_vrs.py, with:
  - no GT mask dependency (no mask_gt_info/*.npy required)
  - only `input == "auto"` mode (no prompts needed)
  - configurable --image_root (a dir of per-sequence subdirs of .jpg frames)
  - saves binary masks + colored overlays for visual inspection
  - saves per-frame mask centroid to centroids.json (for PnP downstream)

Layout expected:
    <image_root>/
      <seq_name_1>/ 00000.jpg 00001.jpg ...
      <seq_name_2>/ ...

Layout produced:
    <save_root>/<seq_name>/<instance_id>/
      00000.png              # binary mask
      00000_overlay.jpg      # overlay (if --save_overlay)
      centroids.json         # {frame_name: [cx, cy] or null}
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
OVERLAY_COLOR = np.array([0, 255, 0], dtype=np.uint8)  # green


def _save_png(path, arr):
    cv2.imwrite(path, arr)


def _save_overlay(path, image_bgr, mask_u8, alpha=0.5):
    overlay = image_bgr.copy()
    sel = mask_u8 > 127
    overlay[sel] = ((1 - alpha) * image_bgr[sel] + alpha * OVERLAY_COLOR).astype(np.uint8)
    cv2.imwrite(path, overlay)


def _save_centroid_viz(path, image_bgr, centroid):
    viz = image_bgr.copy()
    if centroid is not None:
        cx, cy = int(round(centroid[0])), int(round(centroid[1]))
        cv2.drawMarker(viz, (cx, cy), (0, 0, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2)
        cv2.circle(viz, (cx, cy), 6, (0, 255, 255), 2)
    cv2.imwrite(path, viz)


def _mask_centroid(mask_u8):
    ys, xs = np.where(mask_u8 > 127)
    if xs.size == 0:
        return None
    return [float(xs.mean()), float(ys.mean())]


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

    instance_id = CATEGORY2ID[args.category.lower()]
    save_pool = ThreadPoolExecutor(max_workers=8)

    for seq_name in tqdm(seq_list, desc=f"GPU {gpu_id}"):
        seq_path = os.path.join(args.image_root, seq_name)
        frame_files = natsorted([
            f for f in os.listdir(seq_path) if f.lower().endswith((".jpg", ".png"))
        ])
        if not frame_files:
            continue
        frame_stems = [os.path.splitext(f)[0] for f in frame_files]

        out_dir = os.path.join(args.save_root, seq_name, instance_id)
        if os.path.isdir(out_dir) and os.listdir(out_dir) and not args.overwrite:
            continue
        os.makedirs(out_dir, exist_ok=True)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(
                video_path=seq_path,
                async_loading_frames=False,
                offload_video_to_cpu=False,
                offload_state_to_cpu=False,
            )

            # auto mode: prompt-free, starts from frame 0
            start_idx = 0
            _, object_ids, masks = predictor.add_new_robot(
                inference_state=state,
                frame_idx=start_idx,
                obj_id=0,
                robot=args.category,
            )
            first = {oid: (masks[i] > 0.0) for i, oid in enumerate(object_ids)}

            gpu_masks = [first[0][0]]
            frame_indices = [start_idx]

            for out_idx, out_obj_ids, out_logits in predictor.propagate_in_video(
                inference_state=state, robot=args.category
            ):
                if out_idx == start_idx:
                    continue
                gpu_masks.append((out_logits[0] > 0.0)[0])
                frame_indices.append(out_idx)

        stacked = (torch.stack(gpu_masks, 0).cpu().numpy() * 255).astype(np.uint8)
        order = np.argsort(frame_indices)
        frame_indices = [frame_indices[i] for i in order]
        stacked = stacked[order]

        centroids = {}
        for k, idx in enumerate(frame_indices):
            stem = frame_stems[idx]
            mask = stacked[k]
            save_pool.submit(_save_png, os.path.join(out_dir, f"{stem}.png"), mask)
            centroids[stem] = _mask_centroid(mask)

            if args.save_overlay or args.save_centroid_viz:
                img_path = os.path.join(seq_path, frame_files[idx])
                img = cv2.imread(img_path)
                if img is not None:
                    if args.save_overlay:
                        save_pool.submit(
                            _save_overlay,
                            os.path.join(out_dir, f"{stem}_overlay.jpg"),
                            img, mask,
                        )
                    if args.save_centroid_viz:
                        save_pool.submit(
                            _save_centroid_viz,
                            os.path.join(out_dir, f"{stem}_centroid.jpg"),
                            img, centroids[stem],
                        )

        with open(os.path.join(out_dir, "centroids.json"), "w") as f:
            json.dump(centroids, f, indent=2)

    save_pool.shutdown(wait=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_root", required=True,
                   help="Dir containing per-sequence frame subdirs.")
    p.add_argument("--save_root", required=True)
    p.add_argument("--category", required=True, choices=list(CATEGORY2ID))
    p.add_argument("--ckpt", default="robotseg")
    p.add_argument("--yaml", default="robotseg-infer.yaml")
    p.add_argument("--save_overlay", action="store_true")
    p.add_argument("--save_centroid_viz", action="store_true",
                   help="Save per-frame image with mask centroid marker drawn.")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    seqs = sorted(
        d for d in os.listdir(args.image_root)
        if os.path.isdir(os.path.join(args.image_root, d)) and not d.startswith(".")
    )
    if not seqs:
        raise SystemExit(f"No sequence subdirs under {args.image_root}")

    # If sequences are nested one level deeper (our OXE layout puts
    # frames under <episode>/frames/*.jpg), rewrite image_root to point
    # at <episode>/frames by creating a flattened view.
    # Simplest: detect and auto-expand.
    expanded = []
    new_root_map = {}
    for s in seqs:
        maybe_frames = os.path.join(args.image_root, s, "frames")
        if os.path.isdir(maybe_frames):
            expanded.append(s)
            new_root_map[s] = maybe_frames
    if expanded and len(expanded) == len(seqs):
        # create a symlink root so predictor.init_state sees flat layout
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
