from pathlib import Path

import pytest
import yaml


def test_native_density_param_generation_preserves_core_contract(tmp_path):
    import tools.run_native_smlm_density_sweep as sweep

    template = {
        "CalibrInOut": {
            "cam_path": "/tmp/camera.pkl",
            "csplinePSF_path": "/tmp/csplinePSF.pkl",
            "chnlT_path": "/tmp/chnlT.pkl",
        },
        "ExpInOut": {
            "experiment_out": "output/unused",
            "checkpoint_init": None,
        },
        "HyperParameter": {
            "epochs": 100,
            "batch_size": 8,
        },
        "DataFilter": {
            "max_number_targets": 250,
        },
        "Simulation": {
            "emitter_extent": [[-0.5, 63.5], [-0.5, 63.5], [0.0, 121.0]],
            "density": 1.0,
            "em_avg": None,
        },
    }

    payload = sweep.build_density_param(
        template,
        density=0.2,
        output_root=tmp_path / "native_density",
        epochs=500,
        batch_size=24,
        n_max=576,
    )

    assert payload["Simulation"]["density"] == pytest.approx(0.2)
    assert payload["Simulation"]["em_avg"] is None
    assert payload["Simulation"]["emitter_extent"][:2] == [[0, 64], [0, 64]]
    assert payload["Simulation"]["emitter_extent"][2] == [0.0, 121.0]
    assert payload["HyperParameter"]["epochs"] == 500
    assert payload["HyperParameter"]["batch_size"] == 24
    assert payload["DataFilter"]["max_number_targets"] == 576
    assert "density_0p2" in payload["ExpInOut"]["experiment_out"]


def test_native_density_summary_reads_best_and_last_metrics(tmp_path):
    import json

    import tools.run_native_smlm_density_sweep as sweep

    sweep_root = tmp_path / "native_density"
    density_dir = sweep_root / "runs" / "density_2"
    density_dir.mkdir(parents=True)

    run_info = density_dir / "run_info.json"
    experiment_path = density_dir / "20260420_120000_density_2"
    experiment_path.mkdir(parents=True)
    run_info.write_text(
        json.dumps(
            {
                "experiment_id": experiment_path.name,
                "experiment_path": str(experiment_path),
            }
        ),
        encoding="utf-8",
    )

    (experiment_path / "best_jaccard.json").write_text(
        json.dumps({"epoch": 321, "jaccard": 0.456}),
        encoding="utf-8",
    )

    log_path = sweep_root / "logs" / "density_2.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text(
        "rec: 0.111, prec: 0.222, jacc: 0.333, rmse_r: 4.444, rmse_z: 5.555\n",
        encoding="utf-8",
    )

    rows = sweep.summarize_density_runs(
        output_root=sweep_root,
        densities=[2.0],
        log_dir=sweep_root / "logs",
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["density"] == 2.0
    assert row["best_epoch"] == 321
    assert row["best_jaccard"] == pytest.approx(0.456)
    assert row["last_recall"] == pytest.approx(0.111)
    assert row["last_precision"] == pytest.approx(0.222)
    assert row["last_jaccard"] == pytest.approx(0.333)
    assert row["last_rmse_lat"] == pytest.approx(4.444)
    assert row["last_rmse_ax"] == pytest.approx(5.555)
    assert Path(row["experiment_path"]) == experiment_path


def test_native_density_slurm_array_uses_four_gpus_when_available():
    script = Path("scripts/slurm/slurm_smlm_native_density_array.sh").read_text(encoding="utf-8")

    assert "#SBATCH --array=0-5%4" in script


def test_native_density_slurm_array_includes_density_0p5():
    script = Path("scripts/slurm/slurm_smlm_native_density_array.sh").read_text(encoding="utf-8")

    assert "#SBATCH --array=0-5%4" in script
    assert "DENSITIES=(0.1 0.2 0.5 1 2 5)" in script


def test_native_density_slurm_array_uses_300_epoch_limit():
    script = Path("scripts/slurm/slurm_smlm_native_density_array.sh").read_text(encoding="utf-8")

    assert "--best_epoch_limit 300" in script
    assert "--epochs_override 300" in script


def test_native_density_template_defaults_to_300_epochs():
    payload = yaml.safe_load(Path("config/native_smlm_density_template.yaml").read_text(encoding="utf-8"))

    assert payload["HyperParameter"]["epochs"] == 300


def test_resume_best_state_reads_existing_best_jaccard(tmp_path):
    import json

    import smlm_v2a.training.train_engine as train_engine

    experiment_path = tmp_path / "resume_case"
    experiment_path.mkdir(parents=True)
    (experiment_path / "best_jaccard.json").write_text(
        json.dumps({"epoch": 42, "jaccard": 0.777}),
        encoding="utf-8",
    )

    best_jaccard, best_epoch = train_engine._load_existing_best_state(experiment_path)

    assert best_epoch == 42
    assert best_jaccard == pytest.approx(0.777)
