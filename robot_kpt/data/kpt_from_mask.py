r"""Extract (junction, gripper_center) keypoints from arm + gripper binary masks.

Junction   = centroid of (gripper ∩ dilate(arm, k_j))     — where the gripper meets the arm
Center     = centroid of (gripper \ dilate(junction_region, k_c)) — far end of the gripper
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class KptThresholds:
    min_gripper_area: int = 200            # px
    largest_cc_frac: float = 0.70          # fraction of gripper in single CC
    junction_dilate: int = 3               # px (arm dilation for junction)
    center_exclude_dilate: int = 5         # px (junction-region dilation to exclude)
    min_separation: float = 8.0            # px between junction and center
    edge_margin: int = 4                   # px; keypoints inside this border = invisible


@dataclass
class KptResult:
    junction_xy: tuple[float, float] | None
    center_xy: tuple[float, float] | None
    vis_j: bool
    vis_c: bool
    reason: str                            # "ok" or rejection cause
    # debug fields (useful for overlays)
    junction_region: np.ndarray | None = None
    center_region: np.ndarray | None = None


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))


def _largest_cc(mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (largest_cc_mask, frac_of_total). Empty mask -> (mask, 0.0)."""
    total = int(mask.sum())
    if total == 0:
        return mask, 0.0
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    k = 1 + int(np.argmax(areas))
    largest = lbl == k
    return largest, float(areas.max()) / float(total)


def _dilate(mask: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * k + 1, 2 * k + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def _on_edge(xy: tuple[float, float], shape_hw: tuple[int, int], margin: int) -> bool:
    x, y = xy
    h, w = shape_hw
    return x < margin or y < margin or x >= w - margin or y >= h - margin


def extract_keypoints(
    arm_mask: np.ndarray,
    gripper_mask: np.ndarray,
    thr: KptThresholds = KptThresholds(),
) -> KptResult:
    """Both masks: HxW bool/uint8."""
    arm = arm_mask.astype(bool)
    gri = gripper_mask.astype(bool)
    H, W = gri.shape

    if gri.sum() < thr.min_gripper_area:
        return KptResult(None, None, False, False, "gripper_too_small")

    gri_cc, frac = _largest_cc(gri)
    if frac < thr.largest_cc_frac:
        return KptResult(None, None, False, False, f"gripper_fragmented({frac:.2f})")
    gri = gri_cc

    if arm.sum() == 0:
        return KptResult(None, None, False, False, "no_arm")

    arm_dil = _dilate(arm, thr.junction_dilate)
    junc_region = gri & arm_dil
    if not junc_region.any():
        return KptResult(None, None, False, False, "no_junction_overlap")

    junction = _centroid(junc_region)

    junc_excl = _dilate(junc_region, thr.center_exclude_dilate)
    center_region = gri & ~junc_excl
    if not center_region.any():
        return KptResult(None, None, False, False, "no_center_region")

    # Prefer the largest CC of the center region (avoid finger-tip splits pulling the centroid).
    center_cc, _ = _largest_cc(center_region)
    if center_cc.any():
        center_region = center_cc
    center = _centroid(center_region)

    if junction is None or center is None:
        return KptResult(None, None, False, False, "empty_centroid")

    dx, dy = junction[0] - center[0], junction[1] - center[1]
    if (dx * dx + dy * dy) ** 0.5 < thr.min_separation:
        return KptResult(None, None, False, False, "kpts_too_close")

    vis_j = not _on_edge(junction, (H, W), thr.edge_margin)
    vis_c = not _on_edge(center, (H, W), thr.edge_margin)
    if not (vis_j and vis_c):
        return KptResult(junction, center, vis_j, vis_c, "near_edge")

    return KptResult(junction, center, True, True, "ok",
                     junction_region=junc_region, center_region=center_region)
