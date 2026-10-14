from __future__ import annotations

from pathlib import Path

from unity_psf.data.origami import build_origami_manifest, split_origami_acquisitions


def test_origami_manifest_hashes_files_without_copying_data(tmp_path: Path) -> None:
    source = tmp_path / "archive"
    source.mkdir()
    (source / "cellA_spool01_part001.ome.tif").write_bytes(b"first")
    (source / "cellA_spool01_part002.ome.tif").write_bytes(b"second")
    (source / "cellB_spool02.ome.tiff").write_bytes(b"third")
    logical = tmp_path / "origami_2d"
    logical.symlink_to(source, target_is_directory=True)

    manifest = build_origami_manifest(logical)

    assert manifest.logical_root == str(logical)
    assert manifest.resolved_root == str(source.resolve())
    assert len(manifest.files) == 3
    assert all(len(item.sha256) == 64 for item in manifest.files)
    assert {item.acquisition_group for item in manifest.files} == {
        "cellA_spool01",
        "cellB_spool02",
    }
    assert all(not Path(item.relative_path).is_absolute() for item in manifest.files)


def test_origami_split_keeps_acquisition_groups_isolated(tmp_path: Path) -> None:
    for group in ("a", "b", "c", "d", "e"):
        (tmp_path / f"{group}_spool01.ome.tif").write_bytes(group.encode("ascii"))
    manifest = build_origami_manifest(tmp_path)

    split = split_origami_acquisitions(manifest, seed=7)

    train = set(split["train"])
    validation = set(split["validation"])
    test = set(split["test"])
    assert train and validation and test
    assert not (train & validation or train & test or validation & test)
    assert train | validation | test == {item.acquisition_group for item in manifest.files}
