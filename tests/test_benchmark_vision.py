from __future__ import annotations

import pytest

from dropout_mft.benchmark_suite import (
    CONFIRM_SEEDS,
    LR_GRIDS,
    benchmark_profile_layers,
)
from dropout_mft.benchmark_vision import (
    VISION_COHORT_ID,
    VISION_PROFILE_IDS,
    vision_confirm_specs,
    vision_lr_search_specs,
    vision_trial_count,
)


@pytest.mark.parametrize("model_kind", ("mlp", "transformer"))
def test_vision_lr_search_is_complete_and_validation_only(model_kind):
    specs = vision_lr_search_specs(model_kind)
    assert len(specs) == 30
    assert {spec.profile_id for spec in specs} == set(VISION_PROFILE_IDS)
    assert {spec.learning_rate for spec in specs} == set(LR_GRIDS[model_kind])
    assert {spec.seed for spec in specs} == {0}
    assert all(spec.dataset == "tiny_imagenet" for spec in specs)
    assert all(spec.model_kind == model_kind for spec in specs)
    assert all(spec.depth == 12 and spec.epochs == 75 for spec in specs)
    assert all(not spec.evaluate_test for spec in specs)
    assert all(spec.cohort_id == VISION_COHORT_ID for spec in specs)


def test_vision_no_dropout_is_tuned_independently_and_exactly_zero():
    specs = vision_lr_search_specs("transformer")
    controls = [spec for spec in specs if spec.profile_id == "none_tuned"]
    assert len(controls) == 5
    assert {spec.learning_rate for spec in controls} == set(LR_GRIDS["transformer"])
    assert all(spec.mean_dropout == 0.0 for spec in controls)
    assert all(sum(benchmark_profile_layers(spec)) == 0.0 for spec in controls)


def test_vision_confirm_uses_all_profiles_and_fresh_paired_seeds():
    selected = {
        profile_id: {
            "profile_id": profile_id,
            "learning_rate": 1e-4,
            "mean_dropout": 0.0 if profile_id == "none_tuned" else 0.10,
        }
        for profile_id in VISION_PROFILE_IDS
    }
    specs = vision_confirm_specs("transformer", selected)
    assert len(specs) == 30
    assert {spec.profile_id for spec in specs} == set(VISION_PROFILE_IDS)
    assert {spec.seed for spec in specs} == set(CONFIRM_SEEDS)
    assert all(spec.evaluate_test for spec in specs)


def test_vision_confirm_rejects_missing_or_wrong_control_selection():
    selected = {
        profile_id: {
            "learning_rate": 1e-4,
            "mean_dropout": 0.0 if profile_id == "none_tuned" else 0.10,
        }
        for profile_id in VISION_PROFILE_IDS
    }
    selected.pop("uniform")
    with pytest.raises(ValueError, match="missing profiles"):
        vision_confirm_specs("mlp", selected)

    selected["uniform"] = {"learning_rate": 1e-4, "mean_dropout": 0.10}
    selected["none_tuned"]["mean_dropout"] = 0.10
    with pytest.raises(ValueError, match="Invalid selected dropout"):
        vision_confirm_specs("mlp", selected)


def test_vision_trial_budget_is_exact():
    assert vision_trial_count() == {"lr_search": 60, "confirm": 60, "total": 120}
