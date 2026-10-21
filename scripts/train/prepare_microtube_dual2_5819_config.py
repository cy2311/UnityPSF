from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare v0.4 config for the 5819-style resumed run.")
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--raw-tiff", type=Path, required=True)
    parser.add_argument("--zmap-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def prepare_config(
    *,
    base_config: Path,
    raw_tiff: Path,
    zmap_root: Path,
    output: Path,
) -> None:
    raw_tiff = raw_tiff.resolve(strict=True)
    zmap_root = zmap_root.resolve(strict=True)
    config = yaml.safe_load(base_config.read_text(encoding="utf-8"))

    config["phase_retrieval"]["input"]["tiff_path"] = str(raw_tiff)
    train = config["train"]
    if isinstance(train.get("real_tiff_wake"), dict):
        train["real_tiff_wake"]["tiff_path"] = str(raw_tiff)

    online = train["online_generation"]
    lut_config = online.pop("lut_simulation", {})
    vector_config = config.get("simulation", {}).get("psf", {}).get("vector", {})
    online["simulation_backend"] = "cuda_gemm"
    online["vector_psf_size"] = int(
        online.get("vector_psf_size", lut_config.get("psf_size", vector_config.get("psf_size", 51)))
    )

    domains = online["dual_domain_coeff_maps"]
    expected_names = {"left", "right"}
    actual_names = {str(domain.get("name")) for domain in domains}
    if actual_names != expected_names or len(domains) != 2:
        raise ValueError(f"Expected exactly left/right Z-map domains, got {sorted(actual_names)}")

    for domain in domains:
        side = str(domain["name"])
        map_path = zmap_root / side / "export_nat_zmap" / "preferred_full_roi_zernike_maps_nm.npz"
        if not map_path.is_file():
            raise FileNotFoundError(map_path)
        domain["coeff_maps_npz"] = str(map_path)

    metadata = config.setdefault("metadata", {})
    metadata["training_mode"] = "unity_v04_microtube_dual2_5819_cuda_gemm_reproduction"
    metadata["initial_zmap_tag"] = "unity_v04_5815_zmap_rebuild"
    metadata["simulation_backend"] = "cuda_gemm"
    metadata["source_reference"] = str(base_config.resolve())

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    prepare_config(
        base_config=args.base_config,
        raw_tiff=args.raw_tiff,
        zmap_root=args.zmap_root,
        output=args.output,
    )
    print(f"resolved_config={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
