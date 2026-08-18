"""Protocol guards for the 100-epoch Jannis Transformer follow-up."""

from __future__ import annotations

import numpy as np
import pytest

from dropout_mft.benchmark_jannis_100 import (
    JANNIS_100_COHORT_ID,
    JANNIS_100_PROFILES,
    JANNIS_100_SEEDS,
    jannis_100epoch_specs,
    jannis_100epoch_trial_count,
)
from dropout_mft.benchmark_suite import benchmark_profile_layers


def test_followup_is_sixty_paired_100epoch_confirmation_trials():
    specs = jannis_100epoch_specs()
    assert jannis_100epoch_trial_count() == 60
    assert len(specs) == 60
    assert {spec.dataset for spec in specs} == {"openml_jannis"}
    assert {spec.model_kind for spec in specs} == {"transformer"}
    assert {spec.profile_id for spec in specs} == set(JANNIS_100_PROFILES)
    assert {spec.seed for spec in specs} == set(JANNIS_100_SEEDS)
    assert {spec.epochs for spec in specs} == {100}
    assert {spec.depth for spec in specs} == {12}
    assert all(spec.stage == "confirm" and spec.evaluate_test for spec in specs)
    assert all(spec.cohort_id == JANNIS_100_COHORT_ID for spec in specs)
    assert all(spec.learning_rate == 1e-4 for spec in specs)


def test_followup_profiles_have_the_declared_exact_dropout_budget():
    for spec in jannis_100epoch_specs():
        probabilities = benchmark_profile_layers(spec)
        expected = 0.0 if spec.profile_id == "none_tuned" else 0.10
        assert np.mean(probabilities) == pytest.approx(expected, abs=1e-12)
        if spec.profile_id == "none_tuned":
            assert probabilities == [0.0] * 12
