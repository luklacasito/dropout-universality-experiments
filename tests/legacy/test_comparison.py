"""Exact-recipe, provenance, and analysis tests for the legacy comparison."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from dropout_mft.experiments.legacy.analysis import (
    exact_paired_signflip_pvalue,
    paired_comparison_rows,
    profile_summary_rows,
    reference_reproduction_rows,
    validate_complete_cohort,
)
from dropout_mft.experiments.legacy.protocol import (
    LEGACY_EXTENSION_PROFILE_IDS,
    LEGACY_PROFILE_IDS,
    LEGACY_SEEDS,
    LEGACY_TEST_INDEX_HASH,
    LEGACY_TRAIN_INDEX_HASH,
    LegacyDatasetBundle,
    LegacyTrialSpec,
    legacy_cifar10_indices,
    legacy_profile_layers,
    legacy_trial_specs,
    read_legacy_manifest,
    run_legacy_trial,
    write_legacy_manifest,
)

FROZEN_PROVENANCE = {
    "schema_version": 1,
    "git": {"commit": "legacy-test", "dirty": True},
    "source_tree_sha256": "a" * 64,
    "source_archive": {"path": "source.tar", "sha256": "b" * 64},
    "environment_lock": {"path": "environment.json", "sha256": "c" * 64},
    "slurm_scripts": [],
}


def _hash_indices(values) -> str:
    import hashlib

    return hashlib.sha256(
        np.asarray(values, dtype="<i8").tobytes(order="C")
    ).hexdigest()


def test_exact_legacy_cohort_and_recipe_constants():
    specs = legacy_trial_specs()
    assert len(specs) == 125
    assert {spec.profile_id for spec in specs} == set(LEGACY_PROFILE_IDS)
    assert {spec.seed for spec in specs} == set(LEGACY_SEEDS)
    for profile_id in LEGACY_PROFILE_IDS:
        assert sum(spec.profile_id == profile_id for spec in specs) == 25
    assert {
        (
            spec.depth,
            spec.width,
            spec.epochs,
            spec.batch_size,
            spec.learning_rate,
            spec.lr_min,
            spec.weight_decay,
            spec.sigma_w_sq,
            spec.sigma_b_sq,
            spec.train_size,
            spec.test_size,
            spec.data_seed,
        )
        for spec in specs
    } == {(6, 256, 75, 100, 1e-4, 1e-7, 1e-7, 1.98, 0.02, 5000, 5000, 0)}


def test_profiles_keep_exact_discrete_budget_and_legacy_reference_shapes():
    profiles = {
        profile_id: legacy_profile_layers(LegacyTrialSpec(profile_id, 42))
        for profile_id in LEGACY_PROFILE_IDS
    }
    assert profiles["uniform"] == pytest.approx([0.1] * 6)
    assert profiles["linear_early"] == pytest.approx([0.2, 0.16, 0.12, 0.08, 0.04, 0.0])
    assert profiles["step_early"] == pytest.approx([0.2, 0.2, 0.2, 0, 0, 0])
    for values in profiles.values():
        assert np.mean(values) == pytest.approx(0.1, abs=1e-12)
        assert max(values) <= 0.2 + 1e-12
    assert np.all(np.diff(profiles["quadratic_early"]) <= 0)
    assert np.all(np.diff(profiles["quartic_early"]) <= 0)


def test_big_step_extension_is_exact_historical_profile_but_not_cap_matched():
    specs = legacy_trial_specs(LEGACY_EXTENSION_PROFILE_IDS)
    assert len(specs) == 25
    assert {spec.profile_id for spec in specs} == {"big_step"}
    assert {spec.max_dropout for spec in specs} == {0.30}
    assert legacy_profile_layers(specs[0]) == pytest.approx([0.3, 0.3, 0, 0, 0, 0])


def test_legacy_train_and_test_draws_match_frozen_original_hashes():
    train, test = legacy_cifar10_indices()
    assert _hash_indices(train) == LEGACY_TRAIN_INDEX_HASH
    assert _hash_indices(test) == LEGACY_TEST_INDEX_HASH
    np.testing.assert_array_equal(train[:5], [11841, 19602, 45519, 25747, 42642])
    np.testing.assert_array_equal(test[:5], [4223, 1725, 4360, 9043, 8132])


def test_manifest_is_deterministic_provenance_bound_and_tamper_evident(tmp_path):
    specs = legacy_trial_specs()
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_legacy_manifest(first, specs, provenance=FROZEN_PROVENANCE)
    write_legacy_manifest(second, list(reversed(specs)), provenance=FROZEN_PROVENANCE)
    assert first.read_bytes() == second.read_bytes()
    assert read_legacy_manifest(first, provenance=FROZEN_PROVENANCE) == sorted(
        specs, key=lambda item: item.trial_id
    )

    rows = first.read_text().splitlines()
    row = json.loads(rows[0])
    row["dropout_probabilities"][0] += 0.01
    rows[0] = json.dumps(row)
    first.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="hash validation"):
        read_legacy_manifest(first, provenance=FROZEN_PROVENANCE)


def test_machine_readable_preflight_covers_matching_and_no_pooling():
    script = Path(__file__).resolve().parents[2] / "experiments" / "legacy" / "run.py"
    module_spec = importlib.util.spec_from_file_location("legacy_runner", script)
    assert module_spec and module_spec.loader
    runner = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(runner)

    payload = runner.preflight_payload(legacy_trial_specs(), FROZEN_PROVENANCE)

    assert payload["status"] == "passed"
    assert payload["trial_count"] == 125
    assert payload["all_profiles_share_seed_cohort"] is True
    assert payload["invariant_training_and_data_factors"]["batch_size"] == 100
    assert all(row["budget_matches_0p10"] for row in payload["profile_budget_audit"])
    assert all(row["cap_at_most_0p20"] for row in payload["profile_budget_audit"])
    assert (
        payload["comparison_scope"]["pool_absolute_endpoints_with_transfer_protocol"]
        is False
    )


def test_cpu_smoke_and_content_checked_resume(tmp_path):
    spec = replace(
        LegacyTrialSpec("uniform", 42),
        depth=1,
        width=4,
        epochs=2,
        batch_size=5,
        train_size=10,
        test_size=10,
    )
    generator = torch.Generator().manual_seed(4)
    train = torch.utils.data.TensorDataset(
        torch.randn(10, 3, 32, 32, generator=generator),
        torch.arange(10, dtype=torch.long),
    )
    test = torch.utils.data.TensorDataset(
        torch.randn(10, 3, 32, 32, generator=generator),
        torch.arange(10, dtype=torch.long),
    )
    bundle = LegacyDatasetBundle(
        train=train,
        test=test,
        split_hash="synthetic-legacy",
        train_index_hash="train",
        test_index_hash="test",
        protocol="synthetic-legacy-v1",
    )
    output = tmp_path / "trial.npz"
    result = run_legacy_trial(
        spec,
        bundle,
        output,
        provenance=FROZEN_PROVENANCE,
        device="cpu",
    )
    cached = run_legacy_trial(
        spec,
        bundle,
        output,
        provenance=FROZEN_PROVENANCE,
        device="cpu",
    )
    assert result["trial"]["status"] == "complete"
    assert cached["trial"]["trial_id"] == spec.trial_id
    assert result["curves"]["test_loss"].shape == (2,)
    assert result["test"]["selected_epoch"] == 1
    assert result["protocol"]["global_rng_stream"] is True
    assert result["compute"]["optimizer_steps"] == 4


def _fake_complete_trials() -> list[dict]:
    base_loss = {
        "uniform": 2.0,
        "linear_early": 1.8,
        "step_early": 1.7,
        "quadratic_early": 1.6,
        "quartic_early": 1.5,
    }
    trials = []
    for spec in legacy_trial_specs():
        offset = (spec.seed - 54) * 0.001
        loss = np.linspace(3.0, base_loss[spec.profile_id] + offset, 75)
        accuracy = np.linspace(10.0, 40.0 - base_loss[spec.profile_id] + offset, 75)
        trials.append(
            {
                "schema_version": 1,
                "artifact_type": "legacy_apples_to_apples_trial",
                "trial": {
                    "trial_id": spec.trial_id,
                    "config_hash": spec.config_hash,
                    "phase": spec.phase,
                    "status": "complete",
                    "seed": spec.seed,
                },
                "factors": asdict(spec),
                "curves": {"test_loss": loss, "test_accuracy": accuracy},
                "test": {
                    "selected_epoch": 74,
                    "loss": float(loss[-1]),
                    "accuracy_percent": float(accuracy[-1]),
                },
            }
        )
    return trials


def test_analysis_compares_new_profiles_to_uniform_best_legacy_and_saved():
    indexed = validate_complete_cohort(_fake_complete_trials())
    summaries = profile_summary_rows(indexed)
    paired, best_legacy = paired_comparison_rows(indexed)
    assert summaries[0]["profile_id"] == "quartic_early"
    assert best_legacy == "step_early"
    assert any(
        row["candidate_profile"] == "quadratic_early"
        and row["baseline_profile"] == "uniform"
        for row in paired
    )
    prespecified = [row for row in paired if row["contrast_status"] == "prespecified"]
    assert len(prespecified) == 4
    assert all(
        row["loss_holm_p_value_prespecified_family"] is not None for row in prespecified
    )
    assert any(
        row["candidate_profile"] == "quartic_early"
        and row["baseline_profile"] == "step_early"
        for row in paired
    )

    results = {}
    for saved_id, final in (
        ("constant", 2.0),
        ("reverse_linear", 1.8),
        ("reverse_step", 1.7),
    ):
        results[saved_id] = {
            "test_loss": np.tile(np.linspace(3.0, final, 75), (25, 1)),
            "test_acc": np.tile(np.linspace(10.0, 40.0 - final, 75), (25, 1)),
        }
    saved = {
        "config": {
            "N_SIMULATIONS": 25,
            "EPOCHS": 75,
            "DEPTH": 6,
            "WIDTH": 256,
            "H_BAR": 0.1,
            "H_MAX": 0.2,
            "SIGMA_W_SQ": 1.98,
            "SIGMA_B_SQ": 0.02,
            "LEARNING_RATE": 1e-4,
            "LR_MIN": 1e-7,
        },
        "results": results,
    }
    reproduction = reference_reproduction_rows(indexed, saved)
    assert {row["rerun_profile"] for row in reproduction} == {
        "uniform",
        "linear_early",
        "step_early",
    }


def test_meet_in_middle_sign_flip_matches_brute_force():
    import itertools

    values = np.asarray([-0.5, -0.2, 0.1, 0.4, 0.8])
    observed = abs(values.mean())
    statistics = [
        abs(float(np.mean(values * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    brute = np.mean(np.asarray(statistics) >= observed - 1e-15)
    assert exact_paired_signflip_pvalue(values) == pytest.approx(brute)
