"""Selection-stage guards for the manifest-driven scale-transfer CLI."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dropout_mft.experiments.scale_transfer.protocol import (
    mup_transfer_specs,
    mup_tune_extension_specs,
    mup_tune_specs,
    read_manifest,
    write_manifest,
)
from dropout_mft.results import load_npz_result, save_npz_result

SCRIPT = (
    Path(__file__).resolve().parents[2] / "experiments" / "scale_transfer" / "run.py"
)
SPEC = importlib.util.spec_from_file_location("run_scale_transfer", SCRIPT)
assert SPEC and SPEC.loader
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


FROZEN_PROVENANCE = {
    "schema_version": 1,
    "git": {"commit": "test", "dirty": True},
    "source_tree_sha256": "a" * 64,
    "source_archive": {"path": "source.tar", "sha256": "b" * 64},
    "environment_lock": {"path": "environment.json", "sha256": "c" * 64},
    "slurm_scripts": [],
}


@pytest.fixture(autouse=True)
def _fixed_frozen_provenance(monkeypatch):
    monkeypatch.setattr(CLI, "_load_run_provenance", lambda _run_dir: FROZEN_PROVENANCE)


def _write_proxy_trial(run_dir: Path, spec, *, validation_loss: float) -> None:
    path = run_dir / "trials" / spec.phase / f"{spec.trial_id}.npz"
    save_npz_result(
        path,
        {
            "schema_version": CLI.SCHEMA_VERSION,
            "trial": {
                "trial_id": spec.trial_id,
                "config_hash": spec.config_hash,
                "phase": spec.phase,
                "status": "complete",
                "seed": spec.seed,
            },
            "factors": asdict(spec),
            "randomization": CLI.seed_streams(spec),
            "provenance": {
                "source_provenance_sha256": CLI.provenance_sha256(FROZEN_PROVENANCE)
            },
            "data": {"split_hash": "fixed-split"},
            "curves": {"validation_loss": np.asarray([validation_loss])},
        },
    )


def _lock_plan(run_dir: Path, phase: str, specs, split_hash: str = "fixed-split"):
    manifest = run_dir / "manifests" / f"{phase}.jsonl"
    bundle = SimpleNamespace(
        split_hash=split_hash,
        split_protocol="test-split-v1",
        test_subset_hash="test-subset-hash",
        test_subset_protocol="test-subset-v1",
        test_subset_seed=0,
    )
    CLI._record_split_manifest(run_dir, specs[0], bundle)
    CLI._write_locked_manifest(
        run_dir,
        phase,
        manifest,
        specs,
        {CLI._split_key(spec): split_hash for spec in specs},
    )


def _write_coord_artifact(path: Path, *, strict: bool = True, passes: bool = True):
    save_npz_result(
        path,
        {
            "schema_version": CLI.SCHEMA_VERSION,
            "artifact_type": "mup_coordinate_check",
            "strict_requested": strict,
            "status": "passed" if strict and passes else "failed",
            "steps": 2,
            "passes_protocol": passes,
            "passes_threshold_0p1": passes,
            "passes_nontrivial_update": passes,
            "max_abs_hidden_slope_after_update": 0.05 if passes else 0.2,
            "max_abs_output_slope_after_update": 0.04 if passes else 0.2,
            "learning_rate": CLI.MIN_COORD_LEARNING_RATE,
            "widths": np.asarray(CLI.EXPECTED_COORD_WIDTHS),
            "seeds": np.asarray(CLI.EXPECTED_COORD_SEEDS),
            "source_provenance_sha256": CLI.provenance_sha256(FROZEN_PROVENANCE),
        },
    )


def test_proxy_selection_refuses_an_incomplete_manifest(tmp_path):
    specs = mup_tune_specs()
    write_manifest(
        tmp_path / "manifests" / "lr_proxy.jsonl",
        specs,
        provenance=FROZEN_PROVENANCE,
    )
    _lock_plan(tmp_path, "lr_proxy", specs)
    _write_proxy_trial(tmp_path, specs[0], validation_loss=1.0)

    with pytest.raises(SystemExit, match="complete locked cohort"):
        CLI.command_select(SimpleNamespace(run_dir=tmp_path))


def test_complete_proxy_selection_uses_validation_and_writes_transfer_manifest(
    tmp_path,
):
    specs = mup_tune_specs()
    write_manifest(
        tmp_path / "manifests" / "lr_proxy.jsonl",
        specs,
        provenance=FROZEN_PROVENANCE,
    )
    _lock_plan(tmp_path, "lr_proxy", specs)
    for spec in specs:
        # Make 1e-4 the unique optimum for both parameterizations.
        loss = 0.5 if spec.learning_rate == 1e-4 else 1.0
        _write_proxy_trial(tmp_path, spec, validation_loss=loss)

    CLI.command_select(SimpleNamespace(run_dir=tmp_path))

    selection = json.loads((tmp_path / "selection.json").read_text())
    assert selection["selected_learning_rates"] == {"mup": 1e-4, "sp": 1e-4}
    assert selection["test_metrics_used"] is False
    assert selection["completed_manifest_trials"] == 36
    assert len(selection["candidate_summary"]) == 12
    transfer = read_manifest(tmp_path / "manifests" / "width_transfer.jsonl")
    assert len(transfer) == 400
    assert {spec.depth for spec in transfer} == {6}
    assert {spec.width for spec in transfer} == {256, 512, 1024, 2048}


def _unstable_base_loss(spec) -> float:
    if spec.learning_rate == 1e-4:
        return {0: 0.0, 1: 0.0, 2: 10.0}[spec.seed]
    if spec.learning_rate == 3e-4:
        return 1.0
    return 20.0


def test_unstable_proxy_selection_plans_then_uses_fresh_seed_extension(tmp_path):
    with pytest.raises(SystemExit, match="Complete lr_proxy and run select"):
        CLI._require_proxy_extension_trigger(tmp_path, FROZEN_PROVENANCE)

    base_specs = mup_tune_specs()
    write_manifest(
        tmp_path / "manifests" / "lr_proxy.jsonl",
        base_specs,
        provenance=FROZEN_PROVENANCE,
    )
    _lock_plan(tmp_path, "lr_proxy", base_specs)
    for spec in base_specs:
        _write_proxy_trial(
            tmp_path,
            spec,
            validation_loss=_unstable_base_loss(spec),
        )

    with pytest.raises(SystemExit, match="lr_proxy_extension"):
        CLI.command_select(SimpleNamespace(run_dir=tmp_path))
    CLI._require_proxy_extension_trigger(tmp_path, FROZEN_PROVENANCE)
    assert not (tmp_path / "selection.json").exists()
    assert not (tmp_path / "manifests" / "width_transfer.jsonl").exists()
    extension_manifest = tmp_path / "manifests" / "lr_proxy_extension.jsonl"
    extension_specs = read_manifest(extension_manifest)
    assert extension_specs == sorted(
        mup_tune_extension_specs(), key=lambda spec: spec.trial_id
    )
    assert all(not spec.evaluate_test for spec in extension_specs)

    _lock_plan(tmp_path, "lr_proxy_extension", extension_specs)
    for spec in extension_specs:
        loss = 10.0 if spec.learning_rate == 1e-4 else 20.0
        if spec.learning_rate == 3e-4:
            loss = 1.0
        _write_proxy_trial(tmp_path, spec, validation_loss=loss)

    CLI.command_select(SimpleNamespace(run_dir=tmp_path))
    selection = json.loads((tmp_path / "selection.json").read_text())
    assert selection["selected_learning_rates"] == {"mup": 3e-4, "sp": 3e-4}
    assert selection["completed_manifest_trials"] == 72
    assert selection["conditional_extension_used"] is True
    assert selection["leave_one_seed_out_stability"]["stable"] is True


def test_runtime_filters_select_one_calibration_trial_before_sharding():
    specs = mup_transfer_specs({"sp": 1e-4, "mup": 3e-4})
    args = SimpleNamespace(
        widths=[2048],
        parameterizations=["mup"],
        profiles=["uniform"],
        seeds=[100],
    )
    selected = CLI._filter_runtime_specs(specs, args)
    assert len(selected) == 1
    assert selected[0].width == 2048
    assert selected[0].parameterization == "mup"
    assert selected[0].profile_id == "uniform"
    assert selected[0].seed == 100

    with pytest.raises(SystemExit, match="matched no trials"):
        CLI._filter_runtime_specs(
            specs,
            SimpleNamespace(
                widths=[999],
                parameterizations=None,
                profiles=None,
                seeds=None,
            ),
        )


def test_canonical_plan_excludes_conditional_and_deleted_phases():
    assert "field_sensitivity" not in CLI.PHASE_BUILDERS
    assert "lr_proxy_extension" not in CLI.PRIMARY_PHASE_BUILDERS
    assert "lr_proxy_extension" in CLI.CONDITIONAL_PHASE_BUILDERS


def test_two_stage_provenance_cli_routes_snapshot_into_cluster_plan():
    parser = CLI.build_parser()
    snapshot = parser.parse_args(
        [
            "snapshot-source",
            "--output",
            "/ocean/source-snapshot",
            "--slurm-script",
            "run.sbatch",
        ]
    )
    assert snapshot.function is CLI.command_snapshot_source
    assert snapshot.slurm_script == ["run.sbatch"]

    plan = parser.parse_args(
        [
            "plan",
            "--run-dir",
            "/ocean/run",
            "--source-snapshot",
            "/ocean/source-snapshot",
        ]
    )
    assert plan.function is CLI.command_plan
    assert plan.source_snapshot == "/ocean/source-snapshot"


def test_real_training_requires_a_passing_strict_coordinate_artifact(tmp_path):
    path = tmp_path / "coord_check.npz"
    with pytest.raises(SystemExit, match="coordinate-check artifact"):
        CLI._require_coordinate_check(path)

    _write_coord_artifact(path, strict=False, passes=True)
    with pytest.raises(SystemExit, match="strict protocol"):
        CLI._require_coordinate_check(path)

    _write_coord_artifact(path, strict=True, passes=True)
    payload = CLI._require_coordinate_check(path)
    assert payload["passes_protocol"] is True


def test_strict_coordinate_command_is_multiseed_nontrivial_and_provenance_bound(
    tmp_path, monkeypatch
):
    args = SimpleNamespace(
        run_dir=None,
        widths=list(CLI.EXPECTED_COORD_WIDTHS),
        seeds=list(CLI.EXPECTED_COORD_SEEDS),
        steps=CLI.MIN_COORD_STEPS,
        learning_rate=CLI.MIN_COORD_LEARNING_RATE,
        device="cpu",
        output=tmp_path / "coord_check.npz",
        strict=True,
    )
    with pytest.raises(SystemExit, match="requires --run-dir"):
        CLI.command_coord_check(args)

    monkeypatch.setattr(
        CLI,
        "mup_coordinate_check",
        lambda **_kwargs: {
            "widths": np.asarray(CLI.EXPECTED_COORD_WIDTHS),
            "seeds": np.asarray(CLI.EXPECTED_COORD_SEEDS),
            "learning_rate": CLI.MIN_COORD_LEARNING_RATE,
            "device": "cpu",
            "device_name": "test-cpu",
            "passes_threshold_0p1": True,
            "passes_nontrivial_update": True,
            "max_abs_hidden_slope_after_update": 0.05,
            "max_abs_output_slope_after_update": 0.04,
        },
    )
    args.run_dir = tmp_path
    CLI.command_coord_check(args)
    artifact = load_npz_result(args.output)
    assert artifact["source_provenance_sha256"] == CLI.provenance_sha256(
        FROZEN_PROVENANCE
    )
    CLI._require_coordinate_check(args.output, source_provenance=FROZEN_PROVENANCE)


def test_run_writes_locked_manifest_with_realized_split_hash(tmp_path, monkeypatch):
    spec = mup_tune_specs()[0]
    manifest = tmp_path / "manifests" / "lr_proxy.jsonl"
    write_manifest(manifest, [spec], provenance=FROZEN_PROVENANCE)
    _write_coord_artifact(tmp_path / "coord_check.npz")
    bundle = SimpleNamespace(
        split_hash="realized-split",
        split_protocol="test-split-v1",
        test_subset_hash="test-subset-hash",
        test_subset_protocol="test-subset-v1",
        test_subset_seed=0,
    )
    executed = []
    monkeypatch.setattr(CLI, "_bundle_for_spec", lambda _spec, _args: bundle)
    monkeypatch.setattr(
        CLI,
        "run_trial",
        lambda trial_spec, *_args, **_kwargs: executed.append(trial_spec.trial_id),
    )
    args = SimpleNamespace(
        run_dir=tmp_path,
        manifest=manifest,
        phase="lr_proxy",
        num_shards=1,
        shard_index=0,
        coord_check_artifact=None,
        device="cpu",
        force=False,
    )

    CLI.command_run(args)

    locked = json.loads((tmp_path / "manifests" / "lr_proxy.locked.jsonl").read_text())
    split_manifest = json.loads(
        (tmp_path / "manifests" / "data_splits.json").read_text()
    )
    assert locked["trial_id"] == spec.trial_id
    assert locked["split_hash"] == "realized-split"
    assert locked["source_manifest_sha256"]
    assert split_manifest["splits"][locked["data_split_key"]]["split_hash"] == (
        "realized-split"
    )
    assert executed == [spec.trial_id]


def test_target_oracle_gate_requires_complete_locked_zero_shot_cohort(
    tmp_path, monkeypatch
):
    with pytest.raises(SystemExit, match="zero-shot width_transfer"):
        CLI._require_complete_width_transfer(tmp_path)

    selected = {"sp": 1e-4, "mup": 1e-4}
    (tmp_path / "selection.json").write_text(
        json.dumps(
            {
                "schema_version": CLI.SCHEMA_VERSION,
                "selected_learning_rates": selected,
            }
        )
    )
    specs = mup_transfer_specs(selected)
    manifest = tmp_path / "manifests" / "width_transfer.jsonl"
    write_manifest(manifest, specs, provenance=FROZEN_PROVENANCE)
    _lock_plan(tmp_path, "width_transfer", specs)
    trials = [
        {
            "schema_version": CLI.SCHEMA_VERSION,
            "trial": {
                "trial_id": spec.trial_id,
                "config_hash": spec.config_hash,
                "status": "complete",
            },
            "factors": asdict(spec),
            "randomization": CLI.seed_streams(spec),
            "provenance": {
                "source_provenance_sha256": CLI.provenance_sha256(FROZEN_PROVENANCE)
            },
            "data": {"split_hash": "fixed-split"},
            "test": {"evaluated": True, "selected_epoch": spec.epochs - 1},
        }
        for spec in specs
    ]
    monkeypatch.setattr(CLI, "_load_phase_trials", lambda *_args: trials)
    CLI._require_complete_width_transfer(tmp_path)

    monkeypatch.setattr(CLI, "_load_phase_trials", lambda *_args: trials[:-1])
    with pytest.raises(SystemExit, match="complete locked cohort"):
        CLI._require_complete_width_transfer(tmp_path)
