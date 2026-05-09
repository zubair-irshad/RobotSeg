#!/usr/bin/env python3
"""Print silhouette refinement IoU summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _fmt_delta(init: Any, final: Any) -> str:
    try:
        return f"{float(final) - float(init):+.3f}"
    except (TypeError, ValueError):
        return "-"


def _summary_path(mask_root: Path, dataset: str, pnp_json_name: str) -> Path:
    stem = Path(pnp_json_name).stem
    return mask_root / dataset / f"silhouette_refine_summary_{stem}.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pnp_json_names", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", default=None)
    args = parser.parse_args()

    labels = args.labels or args.pnp_json_names
    if len(labels) != len(args.pnp_json_names):
        raise SystemExit("--labels must have the same length as --pnp_json_names")

    headers = ["label", "episode", "init", "final", "delta", "written", "cov", "t2r", "evals", "n", "source"]
    widths = [14, 12, 7, 7, 7, 7, 7, 7, 6, 5, 12]
    print(" ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print(" ".join("-" * w for w in widths))

    for label, name in zip(labels, args.pnp_json_names):
        path = _summary_path(args.mask_root, args.dataset, name)
        data = _load(path)
        if data is None:
            print(f"{label:<14} {'missing':<12} path={path}")
            continue
        for row in data.get("results", []):
            if not isinstance(row, dict):
                continue
            if row.get("status") != "ok":
                out = [
                    label,
                    str(row.get("episode", "-")),
                    str(row.get("status", "-")),
                    "-", "-", "-", "-", "-", "-", "-", "-",
                ]
                print(" ".join(str(v).ljust(w) for v, w in zip(out, widths)))
                continue
            init_iou = row.get("init_iou_median")
            final_iou = row.get("final_iou_median")
            source = str(row.get("init_pose_source", "-"))
            if row.get("viewpoint_index") is not None:
                source += f":v{int(row['viewpoint_index']):02d}"
            out = [
                label,
                str(row.get("episode", "-")),
                _fmt(init_iou),
                _fmt(final_iou),
                _fmt_delta(init_iou, final_iou),
                _fmt(row.get("iou_median")),
                _fmt(row.get("coverage_median")),
                _fmt(row.get("t2r_median"), 2),
                str(row.get("num_evals", "-")),
                str(row.get("num_samples", "-")),
                source[:12],
            ]
            print(" ".join(str(v).ljust(w) for v, w in zip(out, widths)))
        print(f"Wrote/read {path}")


if __name__ == "__main__":
    main()
