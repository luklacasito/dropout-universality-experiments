"""Protocol guards for the zero-weight-decay dropout sweep."""

from __future__ import annotations

import numpy as np
import pytest

from dropout_mft.benchmark_suite import (
    BUDGET_SEARCH_SEEDS,
    CONFIRM_SEEDS,
    MEAN_DROPOUT_GRID,
    TUNED_CONTROL_PROFILE_ID,
    benchmark_profile_layers,
)
from dropout_mft.benchmark_zero_decay import (
    ZERO_DECAY_COHORT_ID,
    ZERO_DECAY_DATASETS,
    ZERO_DECAY_DROPOUT_PROFILE_IDS,
    ZERO_DECAY_MODEL_KINDS,
    ZERO_DECAY_PROFILE_IDS,
    zero_decay_budget_search_specs,
    zero_decay_confirm_specs,
    zero_decay_lr_search_specs,
    zero_decay_trial_count,
)


def _lr_selection() -> dict[str, dict]:
    return {
        profile_id: {
            "learning_rate": 1e-4,
            "mean_dropout": 0.0
            if profile_id == TUNED_CONTROL_PROFILE_ID
            else 0.10,
        }
        for profile_id in ZERO_DECAY_PROFILE_IDS
    }


def _budget_selection() -> dict[str, dict]:
    return {
        profile_id: {"learning_rate": 1e-4, "mean_dropout": 0.10}
        for profile_id in ZERO_DECAY_DROPOUT_PROFILE_IDS
    }


def test_complete_cohort_has_480_trials_and_exactly_zero_weight_decay():
    assert zero_decay_trial_count() == {
        "lr_search": 120,
        "budget_search": 240,
        "confirm": 120,
        "total": 480,
    }
    specs = []
    for dataset in ZERO_DECAY_DATASETS:
        for model_kind in ZERO_DECAY_MODEL_KINDS:
            specs.extend(zero_decay_lr_search_specs(dataset, model_kind))
            specs.extend(
                zero_decay_budget_search_specs(
                    dataset, model_kind, {profile: 1e-4 for profile in ZERO_DECAY_PROFILE_IDS}
                )
            )
            specs.extend(
                zero_decay_confirm_specs(
                    dataset,
                    model_kind,
                    _budget_selection(),
                    _lr_selection(),
                )
            )
    assert len(specs) == 480
    assert all(spec.weight_decay == 0.0 for spec in specs)
    assert all(spec.depth == 12 for spec in specs)
    assert all(spec.cohort_id == ZERO_DECAY_COHORT_ID for spec in specs)


def test_shared_protocol_supports_new_modalities_without_a_forked_engine():
    datasets = ("amazon_reviews", "speech_commands")
    assert zero_decay_trial_count(datasets) == {
        "lr_search": 120,
        "budget_search": 240,
        "confirm": 120,
        "total": 480,
    }
    for dataset in datasets:
        for model_kind in ZERO_DECAY_MODEL_KINDS:
            specs = zero_decay_lr_search_specs(
                dataset,
                model_kind,
                cohort_id="new-modalities-test",
            )
            assert len(specs) == 30
            assert all(spec.cohort_id == "new-modalities-test" for spec in specs)
            assert all(spec.depth == 12 and spec.epochs == 50 for spec in specs)
            assert all(spec.weight_decay == 0.0 for spec in specs)


def test_lr_search_tunes_all_six_arms_independently():
    specs = zero_decay_lr_search_specs("openml_jannis", "transformer")
    assert len(specs) == 30
    assert {spec.profile_id for spec in specs} == set(ZERO_DECAY_PROFILE_IDS)
    assert all(spec.epochs == 100 and not spec.evaluate_test for spec in specs)
    for profile_id in ZERO_DECAY_PROFILE_IDS:
        arm = [spec for spec in specs if spec.profile_id == profile_id]
        assert len(arm) == 5
        assert len({spec.learning_rate for spec in arm}) == 5


def test_budget_search_sweeps_four_rates_on_three_validation_seeds():
    specs = zero_decay_budget_search_specs(
        "fi2010",
        "mlp",
        {profile_id: 1e-4 for profile_id in ZERO_DECAY_PROFILE_IDS},
    )
    assert len(specs) == 60
    assert {spec.mean_dropout for spec in specs} == set(MEAN_DROPOUT_GRID)
    assert {spec.seed for spec in specs} == set(BUDGET_SEARCH_SEEDS)
    assert TUNED_CONTROL_PROFILE_ID not in {spec.profile_id for spec in specs}
    assert all(spec.epochs == 50 and not spec.evaluate_test for spec in specs)
    for spec in specs:
        assert np.mean(benchmark_profile_layers(spec)) == pytest.approx(
            spec.mean_dropout, abs=1e-12
        )


def test_confirmation_uses_fresh_seeds_and_independently_tuned_no_dropout():
    lr_selection = _lr_selection()
    lr_selection[TUNED_CONTROL_PROFILE_ID]["learning_rate"] = 3e-4
    specs = zero_decay_confirm_specs(
        "openml_jannis",
        "transformer",
        _budget_selection(),
        lr_selection,
    )
    assert len(specs) == 30
    assert {spec.seed for spec in specs} == set(CONFIRM_SEEDS)
    assert all(spec.evaluate_test for spec in specs)
    controls = [
        spec for spec in specs if spec.profile_id == TUNED_CONTROL_PROFILE_ID
    ]
    assert len(controls) == len(CONFIRM_SEEDS)
    assert all(spec.mean_dropout == 0.0 for spec in controls)
    assert all(spec.learning_rate == 3e-4 for spec in controls)
    assert not set(BUDGET_SEARCH_SEEDS) & set(CONFIRM_SEEDS)


def test_high_budget_linear_and_big_step_profiles_remain_valid_probabilities():
    specs = zero_decay_budget_search_specs(
        "fi2010",
        "transformer",
        {profile_id: 1e-4 for profile_id in ZERO_DECAY_PROFILE_IDS},
    )
    high = [spec for spec in specs if spec.mean_dropout == 0.20]
    for spec in high:
        probabilities = benchmark_profile_layers(spec)
        assert max(probabilities) < 1.0
        assert np.mean(probabilities) == pytest.approx(0.20, abs=1e-12)
