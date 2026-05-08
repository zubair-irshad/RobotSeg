"""Print per-episode PnP reprojection errors from RobotSeg PnP JSON outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open() as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _fmt_float(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _fmt_pct(value: Any) -> str:
    try:
        return f"{100.0 * float(value):5.1f}%"
    except (TypeError, ValueError):
        return "    -"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_root", type=Path, default=Path("data/oxe_subset_seg"))
    parser.add_argument("--dataset", default="droid")
    parser.add_argument("--pnp_json_name", default="pnp.json")
    parser.add_argument("--episodes", nargs="+", default=None)
    parser.add_argument("--include_moge", action="store_true")
    args = parser.parse_args()

    ds_root = args.mask_root / args.dataset
    episodes = args.episodes or sorted(
        p.name for p in ds_root.iterdir()
        if p.is_dir() and p.name.startswith("episode_")
    )

    headers = [
        "episode", "status", "K", "rmse", "max", "inliers", "frac",
        "fx", "fy", "cx", "cy",
    ]
    if args.include_moge:
        headers += ["moge_fx", "fx_delta"]
    widths = [12, 13, 18, 7, 7, 12, 7, 8, 8, 8, 8]
    if args.include_moge:
        widths += [8, 9]

    print(" ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print(" ".join("-" * w for w in widths))

    for ep in episodes:
        path = ds_root / ep / args.pnp_json_name
        data = _load_json(path)
        if data is None:
            row = [ep, "missing", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
            if args.include_moge:
                row += ["-", "-"]
            print(" ".join(str(v).ljust(w) for v, w in zip(row, widths)))
            continue

        K = data.get("K", {}) if isinstance(data.get("K"), dict) else {}
        inliers = data.get("num_inliers")
        kept = data.get("num_kept_after_filter")
        inlier_text = "-"
        if inliers is not None and kept is not None:
            inlier_text = f"{int(inliers)}/{int(kept)}"

        row = [
            ep,
            str(data.get("status", "-")),
            str(K.get("source", data.get("K_selected_from", "-")))[:18],
            _fmt_float(data.get("rmse_px")),
            _fmt_float(data.get("max_err_px")),
            inlier_text,
            _fmt_pct(data.get("inlier_frac")),
            _fmt_float(K.get("fx")),
            _fmt_float(K.get("fy")),
            _fmt_float(K.get("cx")),
            _fmt_float(K.get("cy")),
        ]

        if args.include_moge:
            moge = _load_json(ds_root / ep / "moge_K.json") or {}
            moge_fx = moge.get("fx")
            delta = "-"
            try:
                delta = f"{100.0 * (float(K.get('fx')) / float(moge_fx) - 1.0):+.1f}%"
            except (TypeError, ValueError, ZeroDivisionError):
                pass
            row += [_fmt_float(moge_fx), delta]

        print(" ".join(str(v).ljust(w) for v, w in zip(row, widths)))


if __name__ == "__main__":
    main()
