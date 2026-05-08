"""Compare per-episode PnP cam2base extrinsics against another JSON source."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cam2base_json import (  # noqa: E402
    SIXD_MODES,
    compare_T,
    episode_lookup_keys,
    find_T_cam2base,
    find_T_cam2base_candidates,
    load_json,
)


def _load_pnp(path: Path) -> dict[str, Any] | None:
    try:
        data = load_json(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _serial_from_pnp(pnp: dict[str, Any]) -> str | None:
    K = pnp.get("K", {})
    source = K.get("source") if isinstance(K, dict) else None
    if isinstance(source, str):
        m = re.search(r"serial=([0-9]+)", source)
        if m:
            return m.group(1)
    return None


def _episode_id_from_pnp(pnp: dict[str, Any]) -> str | None:
    K = pnp.get("K", {})
    source = K.get("source") if isinstance(K, dict) else None
    if isinstance(source, str):
        m = re.search(r"episode_id=([^ ]+)", source)
        if m:
            return m.group(1)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oxe_root", type=Path, default=Path("data/oxe_subset"))
    parser.add_argument("--mask_root", type=Path, default=Path("data/oxe_subset_seg"))
    parser.add_argument("--dataset", default="droid")
    parser.add_argument("--episodes", nargs="+", default=None)
    parser.add_argument("--pnp_json_name", default="pnp.json")
    parser.add_argument("--reference_json", type=Path,
                        default=Path("pnp_cam2base_multiview.json"))
    parser.add_argument("--camera_serial", default=None,
                        help="Optional camera serial key in the reference JSON. "
                             "Defaults to serial parsed from PnP K source when available.")
    parser.add_argument("--sixd_mode", default="auto",
                        choices=["auto", *SIXD_MODES],
                        help="How to interpret 6-vector reference poses. "
                             "auto picks the convention closest to the PnP pose.")
    parser.add_argument("--out_json", type=Path, default=None)
    args = parser.parse_args()

    ref = load_json(args.reference_json)
    ds_seg = args.mask_root / args.dataset
    episodes = args.episodes or sorted(
        p.name for p in ds_seg.iterdir()
        if p.is_dir() and p.name.startswith("episode_")
    )

    rows = []
    print(
        "episode      status        serial   source                                             "
        "trans_cm  rot_deg  dx_cm    dy_cm    dz_cm    pnp_rmse K"
    )
    print(
        "------------ ------------- -------- -------------------------------------------------- "
        "--------- -------- -------- -------- -------- -------- ------------------"
    )
    for ep in episodes:
        pnp_path = ds_seg / ep / args.pnp_json_name
        pnp = _load_pnp(pnp_path)
        if pnp is None or "T_cam2base" not in pnp:
            print(f"{ep:<12} {'missing-pnp':<13}")
            rows.append({"episode": ep, "status": "missing-pnp"})
            continue

        keys = episode_lookup_keys(args.oxe_root, args.dataset, ep)
        raw_id = _episode_id_from_pnp(pnp)
        if raw_id and raw_id not in keys:
            keys += [raw_id, f"{args.dataset}/{raw_id}"]
        preferred = []
        serial = args.camera_serial or _serial_from_pnp(pnp)
        if serial:
            preferred.append(serial)
        if args.sixd_mode == "auto":
            candidates = find_T_cam2base_candidates(ref, keys, preferred_fields=preferred)
            scored = []
            T_pnp = np.asarray(pnp["T_cam2base"], dtype=np.float64)
            for T_candidate, source_candidate in candidates:
                metrics_candidate = compare_T(T_pnp, T_candidate)
                score = (
                    metrics_candidate["rotation_delta_deg"]
                    + 100.0 * metrics_candidate["translation_delta_m"]
                )
                scored.append((score, T_candidate, source_candidate, metrics_candidate))
            if scored:
                _, T_ref, source, precomputed_metrics = min(scored, key=lambda x: x[0])
            else:
                T_ref, source, precomputed_metrics = None, None, None
        else:
            T_ref, source = find_T_cam2base(
                ref, keys, preferred_fields=preferred, sixd_mode=args.sixd_mode
            )
            precomputed_metrics = None
        if T_ref is None:
            print(f"{ep:<12} {'missing-ref':<13} keys={keys}")
            rows.append({"episode": ep, "status": "missing-reference", "lookup_keys": keys})
            continue

        T_pnp = np.asarray(pnp["T_cam2base"], dtype=np.float64)
        metrics = precomputed_metrics or compare_T(T_pnp, T_ref)
        K = pnp.get("K", {}) if isinstance(pnp.get("K"), dict) else {}
        rec = {
            "episode": ep,
            "status": "ok",
            "reference_source": source,
            "lookup_keys": keys,
            "camera_serial": serial,
            "pnp_status": pnp.get("status"),
            "pnp_rmse_px": pnp.get("rmse_px"),
            "K_source": K.get("source"),
            **metrics,
            "T_cam2base_pnp": T_pnp.tolist(),
            "T_cam2base_reference": T_ref.tolist(),
        }
        rows.append(rec)
        print(
            f"{ep:<12} {'ok':<13} {str(serial or '-'):<8} {str(source)[:50]:<50} "
            f"{100 * metrics['translation_delta_m']:9.2f} "
            f"{metrics['rotation_delta_deg']:8.2f} "
            f"{100 * metrics['dx_m']:8.2f} "
            f"{100 * metrics['dy_m']:8.2f} "
            f"{100 * metrics['dz_m']:8.2f} "
            f"{float(pnp.get('rmse_px', float('nan'))):8.2f} "
            f"{str(K.get('source', '-'))[:18]}"
        )

    out_path = args.out_json
    if out_path is None:
        stem = Path(args.pnp_json_name).stem
        out_path = ds_seg / f"cam2base_compare_{stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"dataset": args.dataset, "results": rows}, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
