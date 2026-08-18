"""Protocol guards for the prespecified sample-size cohorts."""

from __future__ import annotations

import pytest

from dropout_mft.benchmark_data_regimes import (
    DATA_REGIMES,
    DATA_REGIME_IDS,
    DATA_REGIME_MODEL_KINDS,
    data_regime,
    data_regime_budget_search_specs,
    data_regime_confirm_specs,
    data_regime_lr_search_specs,
    data_regime_trial_count,
)
from dropout_mft.benchmark_suite import (
    BUDGET_SEARCH_SEEDS,
    CONFIRM_SEEDS,
    LR_SEARCH_SEEDS,
    TUNED_CONTROL_PROFILE_ID,
)
from dropout_mft.benchmark_zero_decay import (
    ZERO_DECAY_DROPOUT_PROFILE_IDS,
    ZERO_DECAY_PROFILE_IDS,
)


def _lr_selection() -> dict[str, dict]:
    return {
        profile_id: {
            "learning_rate": 1e-4,
            "mean_dropout": (0.0 if profile_id == TUNED_CONTROL_PROFILE_ID else 0.10),
        }
        for profile_id in ZERO_DECAY_PROFILE_IDS
    }


def _budget_selection() -> dict[str, dict]:
    return {
        profile_id: {"learning_rate": 1e-4, "mean_dropout": 0.10}
        for profile_id in ZERO_DECAY_DROPOUT_PROFILE_IDS
    }


@pytest.mark.parametrize(
    ("regime_id", "dataset", "train_size", "validation_size", "test_size", "epochs"),
    (
        ("amazon_n2000", "amazon_reviews", 2_000, 5_000, 10_000, 100),
        ("amazon_n5000", "amazon_reviews", 5_000, 5_000, 10_000, 100),
        ("amazon_n20000", "amazon_reviews", 20_000, 5_000, 10_000, 100),
        ("tiny_n80000", "tiny_imagenet", 80_000, 10_000, 10_000, 75),
    ),
)
def test_regime_metadata_is_exact(
    regime_id, dataset, train_size, validation_size, test_size, epochs
):
    regime = data_regime(regime_id)
    assert regime.dataset == dataset
    assert (regime.train_size, regime.validation_size, regime.test_size) == (
        train_size,
        validation_size,
        test_size,
    )
    assert regime.epochs == epochs
    assert regime.cohort_id == f"data-regime-scaling-v1/{regime_id}"


def test_regime_catalog_is_closed_and_unknown_ids_fail():
    assert DATA_REGIME_IDS == (
        "amazon_n2000",
        "amazon_n5000",
        "amazon_n20000",
        "tiny_n80000",
    )
    assert set(DATA_REGIMES) == set(DATA_REGIME_IDS)
    with pytest.raises(ValueError, match="Unknown data regime"):
        data_regime("amazon_n2001")


@pytest.mark.parametrize("regime_id", DATA_REGIME_IDS)
def test_every_regime_is_a_separate_240_trial_zero_decay_cohort(regime_id):
    assert data_regime_trial_count(regime_id) == {
        "lr_search": 60,
        "budget_search": 120,
        "confirm": 60,
        "total": 240,
    }
    regime = data_regime(regime_id)
    all_specs = []
    for model_kind in DATA_REGIME_MODEL_KINDS:
        all_specs.extend(data_regime_lr_search_specs(regime_id, model_kind))
        all_specs.extend(
            data_regime_budget_search_specs(
                regime_id,
                model_kind,
                {profile_id: 1e-4 for profile_id in ZERO_DECAY_PROFILE_IDS},
            )
        )
        all_specs.extend(
            data_regime_confirm_specs(
                regime_id,
                model_kind,
                _budget_selection(),
                _lr_selection(),
            )
        )
    assert len(all_specs) == 240
    assert len({spec.trial_id for spec in all_specs}) == 240
    assert all(spec.depth == 12 and spec.weight_decay == 0.0 for spec in all_specs)
    assert all(spec.cohort_id == regime.cohort_id for spec in all_specs)
    assert all(spec.epochs == regime.epochs for spec in all_specs)
    assert all(
        (spec.train_size, spec.validation_size, spec.test_size)
        == (regime.train_size, regime.validation_size, regime.test_size)
        for spec in all_specs
    )


def test_confirmation_alone_evaluates_test_on_fresh_seeds():
    lr = data_regime_lr_search_specs("amazon_n2000", "transformer")
    budget = data_regime_budget_search_specs(
        "amazon_n2000",
        "transformer",
        {profile_id: 1e-4 for profile_id in ZERO_DECAY_PROFILE_IDS},
    )
    confirm = data_regime_confirm_specs(
        "amazon_n2000",
        "transformer",
        _budget_selection(),
        _lr_selection(),
    )
    assert {spec.seed for spec in lr} == set(LR_SEARCH_SEEDS)
    assert {spec.seed for spec in budget} == set(BUDGET_SEARCH_SEEDS)
    assert {spec.seed for spec in confirm} == set(CONFIRM_SEEDS)
    assert not any(spec.evaluate_test for spec in lr + budget)
    assert all(spec.evaluate_test for spec in confirm)
    assert not (set(LR_SEARCH_SEEDS) | set(BUDGET_SEARCH_SEEDS)) & set(CONFIRM_SEEDS)


def test_no_dropout_is_independently_lr_tuned_then_confirmed():
    lr = data_regime_lr_search_specs("tiny_n80000", "mlp")
    controls = [spec for spec in lr if spec.profile_id == TUNED_CONTROL_PROFILE_ID]
    assert len(controls) == 5
    assert len({spec.learning_rate for spec in controls}) == 5

    lr_selection = _lr_selection()
    lr_selection[TUNED_CONTROL_PROFILE_ID]["learning_rate"] = 3e-4
    confirm = data_regime_confirm_specs(
        "tiny_n80000",
        "mlp",
        _budget_selection(),
        lr_selection,
    )
    controls = [spec for spec in confirm if spec.profile_id == TUNED_CONTROL_PROFILE_ID]
    assert len(controls) == len(CONFIRM_SEEDS)
    assert all(spec.mean_dropout == 0.0 for spec in controls)
    assert all(spec.learning_rate == 3e-4 for spec in controls)


def test_cohort_and_size_make_trials_unique_across_amazon_regimes():
    trial_ids = {
        regime_id: {
            spec.trial_id for spec in data_regime_lr_search_specs(regime_id, "mlp")
        }
        for regime_id in ("amazon_n2000", "amazon_n5000", "amazon_n20000")
    }
    assert not trial_ids["amazon_n2000"] & trial_ids["amazon_n5000"]
    assert not trial_ids["amazon_n2000"] & trial_ids["amazon_n20000"]
    assert not trial_ids["amazon_n5000"] & trial_ids["amazon_n20000"]
