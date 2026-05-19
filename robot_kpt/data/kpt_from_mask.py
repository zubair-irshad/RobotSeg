r"""Extract (junction, gripper_center) keypoints from arm + gripper binary masks.

Junction = centroid of (gripper ∩ dilate(arm, k_j))      — arm/gripper interface
Center   = centroid of (gripper \ dilate(junction_region, k_c)) — far end of the gripper

Returns a list of KptResult — one per gripper instance.
  - Single-arm robots: list of length 1.
  - Bimanual (e.g. mobile-aloha): list of length N (= number of gripper clusters).
Two-finger grippers (parallel-jaw) are handled by *unioning* all CCs above a per-CC
area floor before computing the junction — the centroid then lands between the fingers
on the wrist, which is the desired anatomical keypoint.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class KptThresholds:
    min_gripper_area: int = 200          # total px across all CCs of a gripper instance
    min_cc_area: int = 30                # per-CC floor; smaller CCs are dropped as noise
    max_cc: int = 6                      # safety cap: reject if more surviving CCs than this
    junction_dilate: int = 3             # arm dilation for junction region
    center_exclude_dilate: int = 5       # junction-region dilation when carving the center
    min_separation: float = 8.0          # px between junction and center
    edge_margin: int = 4                 # px; kpts inside this border = invisible
    bimanual: bool = False               # if True, cluster gripper CCs into instances
    cluster_eps: float = 60.0            # px; CC centroids within this distance = same instance


@dataclass
class KptResult:
    junction_xy: tuple[float, float] | None
    center_xy: tuple[float, float] | None
    vis_j: bool
    vis_c: bool
    reason: str
    junction_region: np.ndarray | None = None
    center_region: np.ndarray | None = None


# --------------------------------------------------------------------------- utils

def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))


def _dilate(mask: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * k + 1, 2 * k + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def _on_edge(xy: tuple[float, float], shape_hw: tuple[int, int], margin: int) -> bool:
    x, y = xy
    h, w = shape_hw
    return x < margin or y < margin or x >= w - margin or y >= h - margin


def _surviving_ccs(mask: np.ndarray, min_area: int) -> tuple[list[np.ndarray], np.ndarray]:
    """Return (list_of_cc_masks, centroids_Nx2_xy) for CCs with area >= min_area."""
    n, lbl, stats, cents = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    out_masks: list[np.ndarray] = []
    out_cents: list[tuple[float, float]] = []
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] >= min_area:
            out_masks.append(lbl == k)
            out_cents.append((float(cents[k, 0]), float(cents[k, 1])))
    return out_masks, np.asarray(out_cents, dtype=np.float32).reshape(-1, 2)


def _cluster_by_distance(centroids: np.ndarray, eps: float) -> list[list[int]]:
    """Greedy union-find clustering. Two CCs join a cluster if any pair is within eps."""
    n = len(centroids)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(centroids[i] - centroids[j]) <= eps:
                union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


# --------------------------------------------------------------------------- core

def _kpt_for_instance(
    inst_gripper: np.ndarray,
    arm_dil: np.ndarray,
    thr: KptThresholds,
    shape_hw: tuple[int, int],
) -> KptResult:
    if int(inst_gripper.sum()) < thr.min_gripper_area:
        return KptResult(None, None, False, False, "gripper_too_small")

    junc_region = inst_gripper & arm_dil
    if not junc_region.any():
        return KptResult(None, None, False, False, "no_junction_overlap")
    junction = _centroid(junc_region)

    junc_excl = _dilate(junc_region, thr.center_exclude_dilate)
    center_region = inst_gripper & ~junc_excl
    if not center_region.any():
        return KptResult(None, None, False, False, "no_center_region")

    # Largest CC of remainder, so the centroid doesn't sit between two finger tips.
    cc_masks, _ = _surviving_ccs(center_region, min_area=1)
    if cc_masks:
        center_region = max(cc_masks, key=lambda m: int(m.sum()))
    center = _centroid(center_region)
    if junction is None or center is None:
        return KptResult(None, None, False, False, "empty_centroid")

    dx, dy = junction[0] - center[0], junction[1] - center[1]
    if (dx * dx + dy * dy) ** 0.5 < thr.min_separation:
        return KptResult(None, None, False, False, "kpts_too_close")

    vis_j = not _on_edge(junction, shape_hw, thr.edge_margin)
    vis_c = not _on_edge(center, shape_hw, thr.edge_margin)
    if not (vis_j and vis_c):
        return KptResult(junction, center, vis_j, vis_c, "near_edge")

    return KptResult(junction, center, True, True, "ok",
                     junction_region=junc_region, center_region=center_region)


def extract_keypoints(
    arm_mask: np.ndarray,
    gripper_mask: np.ndarray,
    thr: KptThresholds = KptThresholds(),
) -> list[KptResult]:
    """Extract one keypoint pair per gripper instance.

    Behavior:
      * All gripper CCs with area >= thr.min_cc_area are kept (this handles
        parallel-jaw two-finger splits naturally — the union is taken).
      * If thr.bimanual: surviving CCs are clustered by centroid distance and
        one KptResult is emitted per cluster.
      * Otherwise: a single KptResult is emitted using the union of all CCs.
    """
    arm = arm_mask.astype(bool)
    gri = gripper_mask.astype(bool)
    H, W = gri.shape

    if arm.sum() == 0:
        return [KptResult(None, None, False, False, "no_arm")]

    cc_masks, cc_cents = _surviving_ccs(gri, thr.min_cc_area)
    if not cc_masks:
        return [KptResult(None, None, False, False, "gripper_too_small")]
    if len(cc_masks) > thr.max_cc:
        return [KptResult(None, None, False, False, f"too_many_ccs({len(cc_masks)})")]

    arm_dil = _dilate(arm, thr.junction_dilate)

    if thr.bimanual and len(cc_masks) > 1:
        clusters = _cluster_by_distance(cc_cents, thr.cluster_eps)
    else:
        clusters = [list(range(len(cc_masks)))]

    results: list[KptResult] = []
    for cluster in clusters:
        inst = np.zeros_like(gri)
        for i in cluster:
            inst |= cc_masks[i]
        results.append(_kpt_for_instance(inst, arm_dil, thr, (H, W)))
    return results
