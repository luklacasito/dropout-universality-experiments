"""Content-addressed provenance must freeze dirty and cluster-side inputs."""

from __future__ import annotations

import json
import tarfile

import pytest

import dropout_mft.provenance as provenance_module
from dropout_mft.provenance import (
    freeze_provenance,
    freeze_source_snapshot,
    load_frozen_provenance,
    load_source_snapshot,
    provenance_sha256,
    sha256_file,
)


def test_frozen_provenance_records_source_environment_and_slurm(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "train.py").write_text("print('frozen')\n")
    scheduler_log = source / "slurm-123.out"
    scheduler_log.write_text("queued\n")
    slurm = tmp_path / "job.sbatch"
    slurm.write_text("#!/bin/bash\n#SBATCH --gpus=h100-80:1\n")
    run_dir = tmp_path / "run"

    record = freeze_provenance(
        run_dir,
        source_root=source,
        slurm_scripts=[slurm],
    )

    assert len(record["source_tree_sha256"]) == 64
    assert record["git"]["dirty"] is None
    assert record["slurm_scripts"][0]["sha256"] == sha256_file(slurm)
    assert len(provenance_sha256(record)) == 64
    environment = json.loads((run_dir / record["environment_lock"]["path"]).read_text())
    assert environment["python_version"]
    assert environment["packages"]
    with tarfile.open(run_dir / record["source_archive"]["path"]) as archive:
        assert archive.getnames() == ["train.py"]
        assert archive.extractfile("train.py").read() == b"print('frozen')\n"
    scheduler_log.write_text("running\n")
    assert load_frozen_provenance(run_dir, source_root=source) == record


def test_frozen_provenance_blocks_source_or_script_relabeling(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    code = source / "train.py"
    code.write_text("version = 1\n")
    slurm = tmp_path / "job.sbatch"
    slurm.write_text("#!/bin/bash\n")
    run_dir = tmp_path / "run"
    freeze_provenance(run_dir, source_root=source, slurm_scripts=[slurm])

    slurm.write_text("#!/bin/bash\n#SBATCH -t 1:00:00\n")
    with pytest.raises(ValueError, match="Slurm scripts differ"):
        freeze_provenance(run_dir, source_root=source, slurm_scripts=[slurm])

    code.write_text("version = 2\n")
    with pytest.raises(ValueError, match="source tree differs"):
        load_frozen_provenance(run_dir, source_root=source)


def test_missing_slurm_script_does_not_poison_run_directory(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "train.py").write_text("version = 1\n")
    run_dir = tmp_path / "run"
    slurm = tmp_path / "job.sbatch"

    with pytest.raises(FileNotFoundError, match="Slurm script does not exist"):
        freeze_provenance(run_dir, source_root=source, slurm_scripts=[slurm])
    assert not (run_dir / "provenance").exists()

    slurm.write_text("#!/bin/bash\n")
    record = freeze_provenance(run_dir, source_root=source, slurm_scripts=[slurm])
    assert record["slurm_scripts"][0]["sha256"] == sha256_file(slurm)


def test_portable_source_snapshot_then_cluster_environment_lock(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "train.py").write_text("version = 'portable'\n")
    slurm = tmp_path / "job.sbatch"
    slurm.write_text("#!/bin/bash\n#SBATCH --gpus=h100-80:1\n")
    snapshot_dir = tmp_path / "portable-snapshot"

    snapshot = freeze_source_snapshot(
        snapshot_dir,
        source_root=source,
        slurm_scripts=[slurm],
    )
    assert not (snapshot_dir / "environment.lock.json").exists()
    assert load_source_snapshot(snapshot_dir, source_root=source) == snapshot

    cluster_lock = {
        "schema_version": 1,
        "python_version": "3.11-bridges",
        "packages": [{"name": "torch", "version": "cluster-build"}],
    }
    monkeypatch.setattr(
        provenance_module,
        "_environment_lock_payload",
        lambda: cluster_lock,
    )
    run_dir = tmp_path / "cluster-run"
    record = freeze_provenance(
        run_dir,
        source_root=source,
        source_snapshot=snapshot_dir,
    )

    assert record["source_tree_sha256"] == snapshot["source_tree_sha256"]
    assert record["source_snapshot"]["sha256"] == provenance_sha256(snapshot)
    assert record["source_archive"]["sha256"] == snapshot["source_archive"]["sha256"]
    assert (
        json.loads((run_dir / record["environment_lock"]["path"]).read_text())
        == cluster_lock
    )
    assert load_frozen_provenance(run_dir, source_root=source) == record


def test_cluster_plan_rejects_checkout_different_from_portable_snapshot(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    code = source / "train.py"
    code.write_text("version = 1\n")
    snapshot_dir = tmp_path / "portable-snapshot"
    freeze_source_snapshot(snapshot_dir, source_root=source)

    code.write_text("version = 2\n")
    with pytest.raises(ValueError, match="differs from the portable source snapshot"):
        freeze_provenance(
            tmp_path / "cluster-run",
            source_root=source,
            source_snapshot=snapshot_dir,
        )
