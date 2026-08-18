"""Protocol guards for the Transformer linear/no-dropout sidecar."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from dropout_mft.experiments.benchmark.sidecar import (
    SIDECAR_COHORT_ID,
    SIDECAR_DATASETS,
    select_sidecar_records,
    sidecar_confirm_specs,
    sidecar_lr_search_specs,
    sidecar_trial_count,
)
from dropout_mft.experiments.benchmark.protocol import (
    BENCHMARK_SCHEMA_VERSION,
    CONFIRM_SEEDS,
    SIDECAR_PROFILE_IDS,
    TUNED_CONTROL_PROFILE_ID,
    benchmark_profile_layers,
)


def test_sidecar_schema_and_cohort_prevent_wandb_id_reuse():
    spec = sidecar_lr_search_specs("fi2010")[0]
    assert BENCHMARK_SCHEMA_VERSION == 3
    assert spec.cohort_id == SIDECAR_COHORT_ID
    assert replace(spec, cohort_id="different-cohort").trial_id != spec.trial_id


def test_linear_profiles_are_exact_budget_reversals_at_depth12():
    specs = sidecar_lr_search_specs("fi2010")
    early = next(spec for spec in specs if spec.profile_id == "linear_early")
    late = next(spec for spec in specs if spec.profile_id == "linear_late")
    early_layers = benchmark_profile_layers(early)
    late_layers = benchmark_profile_layers(late)

    assert early_layers == pytest.approx(list(reversed(late_layers)))
    assert np.mean(early_layers) == pytest.approx(0.10, abs=1e-12)
    assert early_layers[0] == pytest.approx(0.20)
    assert early_layers[-1] == pytest.approx(0.0)
    assert max(early_layers) <= early.max_dropout


def test_tuned_no_dropout_is_screened_at_all_transformer_learning_rates():
    specs = sidecar_lr_search_specs("openml_jannis")
    assert len(specs) == 15
    assert {spec.profile_id for spec in specs} == set(SIDECAR_PROFILE_IDS)
    assert all(spec.model_kind == "transformer" for spec in specs)
    assert all(not spec.evaluate_test for spec in specs)

    controls = [spec for spec in specs if spec.profile_id == TUNED_CONTROL_PROFILE_ID]
    assert len(controls) == 5
    assert len({spec.learning_rate for spec in controls}) == 5
    assert all(spec.mean_dropout == 0.0 for spec in controls)
    assert all(benchmark_profile_layers(spec) == [0.0] * 12 for spec in controls)


def _selection_records(dataset: str) -> list[dict]:
    losses = {
        "linear_early": ((1e-4, 0.80), (3e-4, 0.70)),
        "linear_late": ((1e-4, 0.65), (3e-4, 0.75)),
        TUNED_CONTROL_PROFILE_ID: ((1e-4, 0.90), (3e-4, 0.85)),
    }
    records = []
    for profile_id, candidates in losses.items():
        for learning_rate, validation_loss in candidates:
            records.append(
                {
                    "cell": f"{dataset}/transformer",
                    "dataset": dataset,
                    "model_kind": "transformer",
                    "profile_id": profile_id,
                    "learning_rate": learning_rate,
                    "mean_dropout": (
                        0.0 if profile_id == TUNED_CONTROL_PROFILE_ID else 0.10
                    ),
                    "seed": 0,
                    "validation_loss": validation_loss,
                    "validation_accuracy": 1.0 - validation_loss / 2.0,
                }
            )
    return records


def test_selection_picks_direction_and_independently_tunes_no_dropout():
    records = [
        record for dataset in SIDECAR_DATASETS for record in _selection_records(dataset)
    ]
    selection = select_sidecar_records(records)
    assert set(selection) == {
        "fi2010/transformer",
        "openml_jannis/transformer",
    }
    for choice in selection.values():
        assert choice["selected_linear"]["profile_id"] == "linear_late"
        assert choice["selected_linear"]["learning_rate"] == 1e-4
        assert choice[TUNED_CONTROL_PROFILE_ID]["learning_rate"] == 3e-4
        assert choice[TUNED_CONTROL_PROFILE_ID]["mean_dropout"] == 0.0


def test_confirmation_has_selected_linear_and_tuned_control_on_fresh_seeds():
    selected = select_sidecar_records(_selection_records("fi2010"))[
        "fi2010/transformer"
    ]
    specs = sidecar_confirm_specs("fi2010", selected)
    assert len(specs) == 10
    assert {spec.profile_id for spec in specs} == {
        "linear_late",
        TUNED_CONTROL_PROFILE_ID,
    }
    assert {spec.seed for spec in specs} == set(CONFIRM_SEEDS)
    assert all(spec.evaluate_test for spec in specs)
    assert all(spec.cohort_id == SIDECAR_COHORT_ID for spec in specs)


def test_sidecar_has_exactly_50_trials():
    assert sidecar_trial_count() == {
        "lr_search": 30,
        "confirm": 20,
        "total": 50,
    }
