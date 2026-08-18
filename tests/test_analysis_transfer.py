"""Tests for preregistered selection, pairing, and censoring rules."""

from __future__ import annotations

import numpy as np
import pytest

from dropout_mft.analysis import (
    align_by_seed,
    learning_rate_grid_distance,
    mup_transfer_claim_gate,
    profile_transfer_claim_gate,
    restricted_time_to_threshold,
    select_proxy_learning_rates,
)


def test_align_by_seed_is_explicit_and_sorted():
    candidate = [
        {"seed": 9, "value": 0.7},
        {"seed": 2, "value": 0.3},
        {"seed": 5, "value": 0.4},
    ]
    baseline = [
        {"seed": 5, "value": 0.8},
        {"seed": 9, "value": 0.9},
        {"seed": 2, "value": 0.6},
    ]
    aligned_candidate, aligned_baseline = align_by_seed(candidate, baseline)
    np.testing.assert_array_equal(aligned_candidate, [0.3, 0.4, 0.7])
    np.testing.assert_array_equal(aligned_baseline, [0.6, 0.8, 0.9])


def test_align_by_seed_rejects_missing_or_duplicate_seeds():
    with pytest.raises(ValueError, match="identical seed sets"):
        align_by_seed(
            [{"seed": 1, "value": 0.1}],
            [{"seed": 2, "value": 0.1}],
        )
    with pytest.raises(ValueError, match="Duplicate seed"):
        align_by_seed(
            [{"seed": 1, "value": 0.1}, {"seed": 1, "value": 0.2}],
            [{"seed": 1, "value": 0.3}],
        )


def _selection_record(parameterization, learning_rate, validation, test_loss):
    return {
        "factors": {
            "parameterization": parameterization,
            "learning_rate": learning_rate,
        },
        "curves": {"validation_loss": validation},
        # Deliberately contradictory held-out evidence: selection must ignore it.
        "test": {"loss": test_loss},
    }


def test_proxy_lr_selection_uses_final_validation_only():
    records = [
        _selection_record("sp", 1e-4, [0.1, 0.2], 0.01),
        _selection_record("sp", 3e-4, [0.9, 0.1], 9.0),
        _selection_record("mup", 1e-3, [0.4, 0.3], 0.01),
        _selection_record("mup", 3e-3, [0.5, 0.2], 9.0),
    ]
    assert select_proxy_learning_rates(records) == {"mup": 3e-3, "sp": 3e-4}


def test_restricted_time_censors_non_reachers_at_horizon():
    summary = restricted_time_to_threshold(
        [[0.0, 0.5, 0.9], [0.0, 0.1, 0.2]],
        0.8,
        horizon=3,
        direction="max",
    )
    assert summary == {
        "restricted_mean": 2.5,
        "reach_rate": 0.5,
        "horizon": 3.0,
    }


def test_crossing_after_restriction_horizon_is_censored():
    summary = restricted_time_to_threshold(
        [[0.0, 0.1, 0.2, 0.9]],
        0.8,
        horizon=2,
        direction="max",
    )
    assert summary["restricted_mean"] == 2.0
    assert summary["reach_rate"] == 0.0


def test_learning_rate_grid_distance_and_claim_gates_are_conservative():
    grid = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
    assert learning_rate_grid_distance(1e-4, 3e-4, grid) == 1
    assert learning_rate_grid_distance(1e-4, 1e-3, grid) == 2

    partial = mup_transfer_claim_gate(
        [
            {"width": 512, "lr_grid_distance": 0, "relative_regret": 0.005},
            {"width": 2048, "lr_grid_distance": 1, "relative_regret": 0.009},
        ]
    )
    assert partial["evaluable"] is False
    assert partial["successful"] is False

    complete = mup_transfer_claim_gate(
        [
            {"width": width, "lr_grid_distance": distance, "relative_regret": regret}
            for width, distance, regret in (
                (256, 0, 0.005),
                (512, 1, 0.009),
                (1024, 0, 0.010),
                (2048, 2, 0.020),
            )
        ]
    )
    assert complete["passing_widths"] == 3
    assert complete["successful"] is True


def test_profile_transfer_gate_pools_by_seed_and_reports_width_interaction():
    seeds = list(range(10))
    seed_offsets = np.linspace(-0.001, 0.001, len(seeds))
    records = [
        {
            "width": width,
            "seeds": seeds,
            "paired_differences": (-0.030 + 0.002 * index + seed_offsets).tolist(),
        }
        for index, width in enumerate((256, 512, 1024, 2048))
    ]
    gate = profile_transfer_claim_gate(records, harm_margin=0.01)

    assert gate["successful"] is True
    assert gate["pooled_paired_effect"] == pytest.approx(-0.027)
    assert gate["pooled_permutation_p_value"] == pytest.approx(2 / 1024)
    assert gate["width_interaction_per_doubling"] == pytest.approx(0.002)
    assert gate["widths_noninferior"] == 4
    assert gate["widths_observed_beyond_harm_margin"] == 0
    assert len(gate["width_effects"]) == 4


def test_profile_transfer_gate_fails_when_one_width_exceeds_harm_margin():
    seeds = list(range(10))
    records = [
        {
            "width": width,
            "seeds": seeds,
            "paired_differences": [-0.04] * len(seeds),
        }
        for width in (256, 512, 1024, 2048)
    ]
    records[-1]["paired_differences"] = [0.02] * len(seeds)

    gate = profile_transfer_claim_gate(records, harm_margin=0.01)

    assert gate["pooled_benefit"] is True
    assert gate["all_widths_noninferior"] is False
    assert gate["widths_observed_beyond_harm_margin"] == 1
    assert gate["successful"] is False


def test_profile_transfer_gate_requires_identical_ordered_seed_cohorts():
    records = [
        {"width": 256, "seeds": [1, 2], "paired_differences": [-0.1, -0.2]},
        {"width": 512, "seeds": [2, 1], "paired_differences": [-0.2, -0.1]},
    ]
    with pytest.raises(ValueError, match="identical ordered seed cohort"):
        profile_transfer_claim_gate(records, expected_widths=2)
