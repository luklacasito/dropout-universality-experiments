"""Protocol guards for the Tiny ImageNet zero-decay sweep."""

from __future__ import annotations

import numpy as np
import pytest

from dropout_mft.experiments.benchmark.protocol import (
    BUDGET_SEARCH_SEEDS,
    CONFIRM_SEEDS,
    MEAN_DROPOUT_GRID,
    TUNED_CONTROL_PROFILE_ID,
    benchmark_profile_layers,
)
from dropout_mft.experiments.benchmark.zero_decay import (
    VISION_ZERO_DECAY_COHORT_ID,
    VISION_ZERO_DECAY_MODEL_KINDS,
    ZERO_DECAY_DROPOUT_PROFILE_IDS,
    ZERO_DECAY_PROFILE_IDS,
    vision_zero_decay_budget_search_specs,
    vision_zero_decay_confirm_specs,
    vision_zero_decay_lr_search_specs,
    vision_zero_decay_trial_count,
)


def _lr_selection():
    return {
        profile: {
            "learning_rate": 1e-4,
            "mean_dropout": 0.0 if profile == TUNED_CONTROL_PROFILE_ID else 0.10,
        }
        for profile in ZERO_DECAY_PROFILE_IDS
    }


def _budget_selection():
    return {
        profile: {"learning_rate": 1e-4, "mean_dropout": 0.10}
        for profile in ZERO_DECAY_DROPOUT_PROFILE_IDS
    }


def test_vision_zero_decay_cohort_has_240_trials():
    assert vision_zero_decay_trial_count() == {
        "lr_search": 60,
        "budget_search": 120,
        "confirm": 60,
        "total": 240,
    }


@pytest.mark.parametrize("model_kind", VISION_ZERO_DECAY_MODEL_KINDS)
def test_all_vision_stages_are_depth12_75epoch_and_exactly_zero_decay(model_kind):
    specs = vision_zero_decay_lr_search_specs(model_kind)
    specs += vision_zero_decay_budget_search_specs(
        model_kind, {profile: 1e-4 for profile in ZERO_DECAY_PROFILE_IDS}
    )
    specs += vision_zero_decay_confirm_specs(
        model_kind, _budget_selection(), _lr_selection()
    )
    assert all(spec.dataset == "tiny_imagenet" for spec in specs)
    assert all(spec.model_kind == model_kind for spec in specs)
    assert all(spec.depth == 12 and spec.epochs == 75 for spec in specs)
    assert all(spec.weight_decay == 0.0 for spec in specs)
    assert all(spec.cohort_id == VISION_ZERO_DECAY_COHORT_ID for spec in specs)


def test_vision_budget_search_uses_four_rates_and_three_validation_seeds():
    specs = vision_zero_decay_budget_search_specs(
        "transformer", {profile: 1e-4 for profile in ZERO_DECAY_PROFILE_IDS}
    )
    assert len(specs) == 60
    assert {spec.mean_dropout for spec in specs} == set(MEAN_DROPOUT_GRID)
    assert {spec.seed for spec in specs} == set(BUDGET_SEARCH_SEEDS)
    assert all(not spec.evaluate_test for spec in specs)
    for spec in specs:
        assert np.mean(benchmark_profile_layers(spec)) == pytest.approx(
            spec.mean_dropout, abs=1e-12
        )


def test_vision_confirmation_has_six_arms_and_fresh_test_seeds():
    specs = vision_zero_decay_confirm_specs("mlp", _budget_selection(), _lr_selection())
    assert len(specs) == 30
    assert {spec.profile_id for spec in specs} == set(ZERO_DECAY_PROFILE_IDS)
    assert {spec.seed for spec in specs} == set(CONFIRM_SEEDS)
    assert all(spec.stage == "confirm" and spec.evaluate_test for spec in specs)
    assert not set(BUDGET_SEARCH_SEEDS) & set(CONFIRM_SEEDS)
