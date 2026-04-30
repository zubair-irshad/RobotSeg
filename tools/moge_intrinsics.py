"""Per-episode intrinsics estimation with MoGe-2.

The OXE camera is fixed within an episode, so we sample N equally-spaced
frames, run MoGe-2 (Microsoft) on each, convert its normalized intrinsics
to pixel intrinsics, and aggregate via per-component robust median. The
resulting K is then handed to PnP for the extrinsic solve.

MoGe-2 returns `intrinsics` as a 3x3 matrix in normalized [0,1] image
coordinates, so:
    fx_px = K_norm[0,0] * W
    fy_px = K_norm[1,1] * H
    cx_px = K_norm[0,2] * W   (always ~0.5 * W)
    cy_px = K_norm[1,2] * H   (always ~0.5 * H)

(See moge/model/v2.py: focal is relative to half the image diagonal and the
returned intrinsics use principal point = (0.5, 0.5) in normalized coords.)

CLI:
    python tools/moge_intrinsics.py \
        --frames_dir /path/to/episode_xxxx/frames \
        --num_samples 20

The function `estimate_K_for_frames(frames_dir, ...)` is also imported by
tools/pnp_oxe.py via `--moge_intrinsics`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np


_MODEL = None  # cached singleton


def _get_model(device: str, repo_id: str):
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    import torch  # local import keeps pnp_oxe importable without torch
    from moge.model.v2 import MoGeModel
    model = MoGeModel.from_pretrained(repo_id).to(device).eval()
    if device.startswith("cuda"):
        try:
            model = model.to(memory_format=torch.channels_last)
        except Exception:
            pass
    _MODEL = model
    return _MODEL


def _list_frames(frames_dir: Path) -> list[Path]:
    return sorted(
        p for p in frames_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )


def _equally_spaced(n_total: int, n_pick: int) -> list[int]:
    if n_total <= n_pick:
        return list(range(n_total))
    return np.linspace(0, n_total - 1, n_pick, dtype=int).tolist()


def _infer_K_pixel(model, img_rgb_uint8: np.ndarray, device: str) -> np.ndarray:
    """Run MoGe-2 on a single (H, W, 3) uint8 RGB image and return a 3x3 K
    in pixel units."""
    import torch
    H, W = img_rgb_uint8.shape[:2]
    img = torch.from_numpy(img_rgb_uint8).to(device).float().permute(2, 0, 1) / 255.0
    with torch.no_grad():
        out = model.infer(img)
    K_norm = out["intrinsics"].detach().cpu().numpy().astype(np.float64)
    K_px = np.eye(3)
    K_px[0, 0] = K_norm[0, 0] * W
    K_px[1, 1] = K_norm[1, 1] * H
    K_px[0, 2] = K_norm[0, 2] * W
    K_px[1, 2] = K_norm[1, 2] * H
    return K_px


def estimate_K_for_frames(
    frames_dir: Path,
    num_samples: int = 20,
    device: Optional[str] = None,
    repo_id: str = "Ruicheng/moge-2-vitl-normal",
    verbose: bool = False,
) -> Optional[dict]:
    """Sample N frames, run MoGe-2, and return a robust-median K dict.

    Returns None if no frames found or all inferences fail.
    """
    import cv2

    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    frames = _list_frames(frames_dir)
    if not frames:
        return None
    pick_idx = _equally_spaced(len(frames), num_samples)
    picked = [frames[i] for i in pick_idx]

    model = _get_model(device, repo_id)

    fxs, fys, cxs, cys, Ws, Hs = [], [], [], [], [], []
    for fp in picked:
        bgr = cv2.imread(str(fp))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        try:
            K = _infer_K_pixel(model, rgb, device)
        except Exception as e:
            if verbose:
                print(f"  [moge] {fp.name}: infer failed ({e})")
            continue
        fxs.append(K[0, 0])
        fys.append(K[1, 1])
        cxs.append(K[0, 2])
        cys.append(K[1, 2])
        Ws.append(W)
        Hs.append(H)
        if verbose:
            print(f"  [moge] {fp.name}: fx={K[0,0]:.1f} fy={K[1,1]:.1f}")

    if not fxs:
        return None

    fxs = np.asarray(fxs)
    fys = np.asarray(fys)
    fx_med = float(np.median(fxs))
    fy_med = float(np.median(fys))
    # Principal point: MoGe always returns (0.5W, 0.5H); take median anyway.
    cx_med = float(np.median(cxs))
    cy_med = float(np.median(cys))
    W_ref = int(np.median(Ws))
    H_ref = int(np.median(Hs))

    fx_mad = float(np.median(np.abs(fxs - fx_med)))
    fy_mad = float(np.median(np.abs(fys - fy_med)))

    return {
        "fx": fx_med,
        "fy": fy_med,
        "cx": cx_med,
        "cy": cy_med,
        "width": W_ref,
        "height": H_ref,
        "source": f"moge-2:{repo_id}",
        "n_frames": len(fxs),
        "fx_mad": fx_mad,
        "fy_mad": fy_mad,
        "fx_per_frame": fxs.tolist(),
        "fy_per_frame": fys.tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames_dir", type=Path, required=True)
    p.add_argument("--num_samples", type=int, default=20)
    p.add_argument("--device", default=None)
    p.add_argument("--repo_id", default="Ruicheng/moge-2-vitl-normal")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    K = estimate_K_for_frames(
        args.frames_dir, args.num_samples, args.device, args.repo_id, args.verbose
    )
    if K is None:
        print("MoGe estimation failed (no frames or all inferences errored)")
        return
    print(json.dumps(K, indent=2))
    if args.out is not None:
        args.out.write_text(json.dumps(K, indent=2))


if __name__ == "__main__":
    main()
