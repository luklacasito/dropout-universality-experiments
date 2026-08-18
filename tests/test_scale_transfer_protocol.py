"""Protocol-level checks for profiles, manifests, hashes, and trial output."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from dropout_mft.scale_transfer import (
    PROFILE_IDS,
    SCHEMA_VERSION,
    TrialSpec,
    canonical_json,
    mup_transfer_specs,
    mup_tune_extension_specs,
    mup_tune_specs,
    oracle_specs,
    profile_layers,
    profile_confirmation_specs,
    profile_pilot_specs,
    read_manifest,
    run_trial,
    schedule_metadata,
    seed_streams,
    vit_confirmation_specs,
    write_manifest,
)
from dropout_mft.training import synthetic_bundle


FROZEN_PROVENANCE = {
    "schema_version": 1,
    "git": {"commit": "test", "dirty": True},
    "source_tree_sha256": "a" * 64,
    "source_archive": {"path": "source.tar", "sha256": "b" * 64},
    "environment_lock": {"path": "environment.json", "sha256": "c" * 64},
    "slurm_scripts": [],
}


def _spec(**overrides):
    values = {
        "phase": "unit",
        "model_kind": "mlp",
        "parameterization": "sp",
        "dataset": "cifar10",
        "activation": "relu",
        "profile_id": "quadratic_early",
        "mean_dropout": 0.1,
        "max_dropout": 0.2,
        "depth": 8,
        "width": 16,
        "learning_rate": 1e-3,
        "seed": 7,
        "epochs": 1,
        "batch_size": 5,
        "weight_decay": 0.0,
    }
    values.update(overrides)
    return TrialSpec(**values)


def test_trial_hash_and_canonical_json_are_deterministic():
    first = _spec()
    second = _spec()
    changed = replace(first, seed=8)
    assert first.trial_id == second.trial_id
    assert first.trial_id != changed.trial_id
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})
    assert len(first.trial_id) == 20


def test_manifest_bytes_are_order_independent_and_round_trip(tmp_path):
    specs = [_spec(seed=3), _spec(seed=1), _spec(seed=2)]
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    write_manifest(first_path, specs, provenance=FROZEN_PROVENANCE)
    write_manifest(second_path, list(reversed(specs)), provenance=FROZEN_PROVENANCE)
    assert first_path.read_bytes() == second_path.read_bytes()
    assert read_manifest(first_path) == sorted(specs, key=lambda spec: spec.trial_id)
    assert all(
        json.loads(line)["schema_version"] == SCHEMA_VERSION
        for line in first_path.read_text().splitlines()
    )


def test_manifest_rejects_content_tampering(tmp_path):
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, [_spec()], provenance=FROZEN_PROVENANCE)
    row = json.loads(path.read_text())
    row["seed"] += 1
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="hash"):
        read_manifest(path)


def test_manifest_rejects_schedule_tampering(tmp_path):
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, [_spec()], provenance=FROZEN_PROVENANCE)
    row = json.loads(path.read_text())
    row["dropout_probabilities"][0] += 0.01
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="schedule"):
        read_manifest(path)


def test_manifest_rejects_provenance_and_randomization_tampering(tmp_path):
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, [_spec()], provenance=FROZEN_PROVENANCE)
    row = json.loads(path.read_text())
    row["provenance"]["git"]["dirty"] = False
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="provenance"):
        read_manifest(path)

    write_manifest(path, [_spec()], provenance=FROZEN_PROVENANCE)
    row = json.loads(path.read_text())
    row["randomization"]["dropout_seed"] += 1
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="randomization"):
        read_manifest(path)


def test_all_profiles_have_exact_probability_budget_cap_and_reversal():
    profiles = {name: profile_layers(_spec(profile_id=name)) for name in PROFILE_IDS}
    assert profiles["none"] == [0.0] * 8
    for name, values in profiles.items():
        if name == "none":
            continue
        assert np.mean(values) == pytest.approx(0.1, abs=1e-12)
        assert max(values) <= 0.2 + 1e-12

    for family in ("linear", "quadratic", "quartic", "step"):
        assert profiles[f"{family}_early"] == pytest.approx(
            profiles[f"{family}_late"][::-1]
        )


def test_canonical_allocation_uses_30_run_pilot_and_all_width_oracles():
    pilot = profile_pilot_specs()
    assert len(pilot) == 30
    assert {spec.profile_id for spec in pilot} == set(PROFILE_IDS)
    assert {spec.mean_dropout for spec in pilot if spec.profile_id != "none"} == {0.10}
    assert {spec.mean_dropout for spec in pilot if spec.profile_id == "none"} == {0.0}

    oracles = oracle_specs()
    assert len(oracles) == 144
    assert {spec.width for spec in oracles} == {256, 512, 1024, 2048}

    extension = mup_tune_extension_specs()
    assert len(extension) == 36
    assert {spec.seed for spec in extension} == {3, 4, 5}
    assert all(not spec.evaluate_test for spec in extension)

    canonical_total = sum(
        (
            len(pilot),
            len(profile_confirmation_specs()),
            len(mup_tune_specs()),
            len(mup_transfer_specs({"sp": 1e-4, "mup": 1e-4})),
            len(oracles),
            len(vit_confirmation_specs()),
        )
    )
    assert canonical_total == 740


def test_seed_streams_are_separate_and_share_dropout_crns_within_reversals():
    early_streams = seed_streams(_spec(profile_id="quadratic_early"))
    late_streams = seed_streams(_spec(profile_id="quadratic_late"))
    other_streams = seed_streams(_spec(profile_id="step_early"))

    assert (
        len(
            {
                early_streams["initialization_seed"],
                early_streams["minibatch_seed"],
                early_streams["dropout_seed"],
            }
        )
        == 3
    )
    assert early_streams["initialization_seed"] == late_streams["initialization_seed"]
    assert early_streams["minibatch_seed"] == late_streams["minibatch_seed"]
    assert early_streams["dropout_seed"] == late_streams["dropout_seed"]
    assert early_streams["dropout_crn_group"] == late_streams["dropout_crn_group"]
    assert other_streams["dropout_seed"] != early_streams["dropout_seed"]


def test_residual_vit_bridge_has_both_reversals_six_seeds_and_original_recipe():
    specs = vit_confirmation_specs()

    assert len(specs) == 30
    assert {spec.phase for spec in specs} == {"vit_confirmation"}
    assert {spec.profile_id for spec in specs} == {
        "uniform",
        "quadratic_early",
        "quadratic_late",
        "step_early",
        "step_late",
    }
    assert {spec.seed for spec in specs} == set(range(200, 206))
    assert {spec.depth for spec in specs} == {10}
    assert {spec.gradient_clip_norm for spec in specs} == {1.0}

    by_profile = {
        profile: profile_layers(
            next(spec for spec in specs if spec.profile_id == profile)
        )
        for profile in {spec.profile_id for spec in specs}
    }
    assert by_profile["quadratic_early"] == pytest.approx(
        by_profile["quadratic_late"][::-1]
    )
    assert by_profile["step_early"] == pytest.approx(by_profile["step_late"][::-1])


def test_residual_vit_bridge_passes_clipping_through_trial_result(tmp_path):
    spec = replace(
        vit_confirmation_specs()[0],
        depth=1,
        width=16,
        epochs=1,
        batch_size=2,
        train_size=4,
        validation_size=2,
        test_size=2,
    )
    flat_bundle = synthetic_bundle(
        input_dim=3 * 32 * 32,
        classes=100,
        train_size=4,
        validation_size=2,
        test_size=2,
        seed=808,
    )

    def as_images(dataset):
        features, targets = dataset.tensors
        return torch.utils.data.TensorDataset(features.reshape(-1, 3, 32, 32), targets)

    bundle = replace(
        flat_bundle,
        train=as_images(flat_bundle.train),
        validation=as_images(flat_bundle.validation),
        test=as_images(flat_bundle.test),
        dataset="cifar100",
    )

    result = run_trial(spec, bundle, tmp_path / "vit.npz", device="cpu")

    assert result["factors"]["gradient_clip_norm"] == 1.0
    assert result["training"]["training_config"]["gradient_clip_norm"] == 1.0


def test_schedule_metadata_has_exact_shapes_and_consistent_damage():
    spec = _spec(depth=6)
    metadata = schedule_metadata(spec)
    assert metadata["dropout_probabilities"].shape == (6,)
    assert metadata["reference_fields"].shape == (6,)
    assert metadata["propagated_local_fields"].shape == (6,)
    assert metadata["variance_trajectory"].shape == (7,)
    assert metadata["mean_dropout_probability"] == pytest.approx(0.1)
    assert metadata["field_damage_total"] == pytest.approx(
        metadata["field_damage_mean"] * spec.depth
    )
    assert metadata["xi_proxy"] == pytest.approx(1 / metadata["field_damage_mean"])


def test_sp_trial_smoke_writes_protocol_result_and_reuses_complete_cache(tmp_path):
    spec = _spec(
        profile_id="uniform",
        depth=1,
        width=4,
        seed=11,
        epochs=1,
        batch_size=5,
    )
    bundle = synthetic_bundle(
        input_dim=3 * 32 * 32,
        classes=10,
        train_size=10,
        validation_size=10,
        test_size=10,
        seed=123,
    )
    output = tmp_path / "trial.npz"
    result = run_trial(spec, bundle, output, device="cpu")
    cached = run_trial(spec, bundle, output, device="cpu")

    assert output.exists()
    assert result["trial"]["trial_id"] == spec.trial_id
    assert cached["trial"]["trial_id"] == spec.trial_id
    assert result["trial"]["status"] == "complete"
    assert result["test"]["selected_epoch"] == 0
    assert "test_loss" not in result["curves"]
    assert "validation_loss" in result["curves"]
    assert result["compute"]["optimizer_steps"] == 2
    assert result["data"]["split_hash"] == bundle.split_hash
    assert result["randomization"] == seed_streams(spec)


def test_trial_is_invariant_to_prior_rng_consumption_and_can_seal_test(tmp_path):
    spec = _spec(
        profile_id="quadratic_early",
        depth=1,
        width=4,
        seed=19,
        epochs=1,
        batch_size=5,
        evaluate_test=False,
    )
    bundle = synthetic_bundle(
        input_dim=3 * 32 * 32,
        classes=10,
        train_size=10,
        validation_size=10,
        test_size=10,
        seed=456,
    )
    torch.manual_seed(1)
    torch.randn(37)
    first = run_trial(spec, bundle, tmp_path / "first.npz", device="cpu")
    torch.manual_seed(999)
    torch.randn(101)
    second = run_trial(spec, bundle, tmp_path / "second.npz", device="cpu")

    for key in first["curves"]:
        np.testing.assert_array_equal(first["curves"][key], second["curves"][key])
    assert first["test"] == {
        "evaluated": False,
        "selected_epoch": 0,
        "loss": None,
        "accuracy": None,
    }
    assert second["test"] == first["test"]


def test_cached_trial_rejects_a_different_data_split(tmp_path):
    spec = _spec(profile_id="uniform", depth=1, width=4, epochs=1, batch_size=5)
    first_bundle = synthetic_bundle(
        input_dim=3 * 32 * 32,
        classes=10,
        train_size=10,
        validation_size=10,
        test_size=10,
        seed=1,
    )
    second_bundle = synthetic_bundle(
        input_dim=3 * 32 * 32,
        classes=10,
        train_size=10,
        validation_size=10,
        test_size=10,
        seed=2,
    )
    output = tmp_path / "trial.npz"
    run_trial(spec, first_bundle, output, device="cpu")
    with pytest.raises(ValueError, match="mismatched"):
        run_trial(spec, second_bundle, output, device="cpu")
