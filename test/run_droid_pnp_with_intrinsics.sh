#!/usr/bin/env bash
# Build scaled DROID camera intrinsics and run PnP without HFOV fallback.
#
# Usage from repo root:
#   bash test/run_droid_pnp_with_intrinsics.sh
#
# Usage from test/:
#   bash run_droid_pnp_with_intrinsics.sh

set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OXE_ROOT="${OXE_ROOT:-$HOME/RobotSeg/data/oxe_subset}"
MASK_ROOT="${MASK_ROOT:-$HOME/RobotSeg/data/oxe_subset_seg_compare/current_pipeline}"
DATASET="${DATASET:-droid}"
CAMERA="${CAMERA:-exterior_image_1_left}"
K_JSON="${K_JSON:-$HOME/RobotSeg/data/droid_hf_intrinsics_scaled.json}"
CACHE_DIR="${CACHE_DIR:-$HOME/RobotSeg/data/droid_hf_calib}"

echo "==> build DROID intrinsics: camera=$CAMERA"
"$PYTHON" "$REPO_ROOT/tools/droid_hf_intrinsics.py" \
  --oxe_root "$OXE_ROOT" \
  --dataset "$DATASET" \
  --camera "$CAMERA" \
  --cache_dir "$CACHE_DIR" \
  --out_json "$K_JSON"

echo
echo "==> pnp with DROID intrinsics only"
"$PYTHON" "$REPO_ROOT/tools/pnp_oxe.py" \
  --oxe_root "$OXE_ROOT" \
  --mask_root "$MASK_ROOT" \
  --datasets "$DATASET" \
  --K_json "$K_JSON" \
  --require_known_intrinsics \
  --print_intrinsics \
  --viz
