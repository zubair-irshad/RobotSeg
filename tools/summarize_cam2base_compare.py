"""Summarize one or two cam2base comparison JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_rows(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text())
    rows = {}
    for row in data.get("results", []):
        if isinstance(row, dict) and row.get("episode"):
            rows[str(row["episode"])] = row
    return rows


def _vals(rows: dict[str, dict[str, Any]], key: str, scale: float = 1.0) -> np.ndarray:
    vals = []
    for row in rows.values():
        if row.get("status") != "ok":
            continue
        val = row.get(key)
        if val is None:
            continue
        vals.append(float(val) * scale)
    return np.asarray(vals, dtype=np.float64)


def _summary(rows: dict[str, dict[str, Any]]) -> dict[str, float | int]:
    out: dict[str, float | int] = {"ok": sum(1 for r in rows.values() if r.get("status") == "ok")}
    metrics = {
        "center_cm": ("translation_delta_m", 100.0),
        "rot_deg": ("rotation_delta_deg", 1.0),
        "rel_cm": ("relative_translation_m", 100.0),
    }
    for name, (key, scale) in metrics.items():
        vals = _vals(rows, key, scale)
        if len(vals):
            out[f"{name}_median"] = float(np.median(vals))
            out[f"{name}_mean"] = float(vals.mean())
            out[f"{name}_p90"] = float(np.percentile(vals, 90))
    return out


def _fmt(v: Any, digits: int = 2) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _print_summary(label: str, rows: dict[str, dict[str, Any]]) -> None:
    s = _summary(rows)
    print(
        f"{label:<10} ok={s.get('ok', 0):3d} "
        f"center_med={_fmt(s.get('center_cm_median'))}cm "
        f"center_mean={_fmt(s.get('center_cm_mean'))}cm "
        f"rot_med={_fmt(s.get('rot_deg_median'))}deg "
        f"rel_med={_fmt(s.get('rel_cm_median'))}cm"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, default=None)
    args = p.parse_args()

    before = _load_rows(args.before)
    _print_summary("before", before)

    if args.after is None:
        return
    after = _load_rows(args.after)
    _print_summary("after", after)

    common = sorted(
        ep for ep in before.keys() & after.keys()
        if before[ep].get("status") == "ok" and after[ep].get("status") == "ok"
    )
    print("\nepisode      center_before center_after  delta_cm  rot_before rot_after delta_deg")
    print("------------ ------------- ------------ --------- ---------- --------- ---------")
    improved = 0
    worsened = 0
    for ep in common:
        b = before[ep]
        a = after[ep]
        b_cm = 100.0 * float(b["translation_delta_m"])
        a_cm = 100.0 * float(a["translation_delta_m"])
        b_rot = float(b["rotation_delta_deg"])
        a_rot = float(a["rotation_delta_deg"])
        delta_cm = a_cm - b_cm
        delta_rot = a_rot - b_rot
        if delta_cm < 0:
            improved += 1
        elif delta_cm > 0:
            worsened += 1
        print(
            f"{ep:<12} {b_cm:13.2f} {a_cm:12.2f} {delta_cm:9.2f} "
            f"{b_rot:10.2f} {a_rot:9.2f} {delta_rot:9.2f}"
        )
    print(f"\nCommon ok episodes: {len(common)}  center improved: {improved}  worsened: {worsened}")


if __name__ == "__main__":
    main()
