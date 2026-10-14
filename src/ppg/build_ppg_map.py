#!/usr/bin/env python3
"""Build the default 36-parameter PPG map from the Beads TIFF."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile

from ppg.beads_initialization import fit_beads_initialization


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "datasets/training_sets/Zeiss/63x1.4_642.tif"
DEFAULT_CATALOG = ROOT / "configs/calibration/ppg/bead_catalog_zeiss_63x1.4_642.npz"
DEFAULT_OUTPUT = ROOT / "configs/calibration/ppg/ppg_map_initial.npz"
DEFAULT_HOLDOUT_IDS = np.asarray([44, 54, 11, 43, 53])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-tiff", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--bead-catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--grid-size", type=int, default=40)
    parser.add_argument("--boundary-anchor-strength", type=float, default=1.0)
    parser.add_argument(
        "--holdout-ids", type=int, nargs="+", default=DEFAULT_HOLDOUT_IDS.tolist()
    )
    args = parser.parse_args()

    stack = tifffile.imread(args.source_tiff)
    with np.load(args.bead_catalog, allow_pickle=False) as catalog:
        result = fit_beads_initialization(
            stack,
            bead_ids=catalog["bead_ids"],
            centers_yx=catalog["centers_yx"],
            positions_xy_normalized=catalog["positions_xy_normalized"],
            field_shape_yx=catalog["field_shape_yx"],
            holdout_ids=np.asarray(args.holdout_ids),
            grid_size=args.grid_size,
            boundary_anchor_strength=args.boundary_anchor_strength,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result.arrays)
    manifest = {
        **result.summary,
        "artifact": str(args.output.resolve()),
        "source_tiff": str(args.source_tiff.resolve()),
        "source_tiff_sha256": sha256(args.source_tiff),
        "bead_catalog": str(args.bead_catalog.resolve()),
        "bead_catalog_sha256": sha256(args.bead_catalog),
        "status": "approved_default",
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
