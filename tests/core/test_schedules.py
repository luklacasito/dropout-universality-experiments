"""Sanity checks for the dropout schedules."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dropout_mft.schedules import (
    effective_xi,
    field_damage,
    power_profile_layers,
    schedule_layers,
)

# Schedules that spend exactly the budget h_bar on average.
BUDGET_PRESERVING = [
    "constant",
    "linear",
    "reverse_linear",
    "quadratic",
    "reverse_quadratic",
    "step",
    "reverse_step",
    "big_step",
]

# Every schedule schedule_layers() knows how to build.
IMPLEMENTED = ["none", "double", "triple", *BUDGET_PRESERVING]


@pytest.mark.parametrize("schedule", BUDGET_PRESERVING)
@pytest.mark.parametrize("depth", [4, 12, 24])
def test_budget_preserving_schedules_keep_mean(schedule, depth):
    h_bar = 0.1
    layers = schedule_layers(schedule, depth, h_bar)
    assert len(layers) == depth
    assert np.mean(layers) == pytest.approx(h_bar)
    assert all(h >= 0 for h in layers)


def test_none_is_all_zero():
    assert schedule_layers("none", 8, 0.1) == [0.0] * 8


@pytest.mark.parametrize("schedule,factor", [("double", 2.0), ("triple", 3.0)])
def test_multiplier_schedules(schedule, factor):
    layers = schedule_layers(schedule, 8, 0.1)
    assert np.mean(layers) == pytest.approx(factor * 0.1)


def test_unknown_schedule_raises():
    with pytest.raises(ValueError):
        schedule_layers("does_not_exist", 8, 0.1)


@pytest.mark.parametrize("name", IMPLEMENTED)
def test_implemented_schedules_build(name):
    layers = schedule_layers(name, 12, 0.1)
    assert len(layers) == 12


def test_effective_xi_infinite_without_dropout():
    assert math.isinf(effective_xi(schedule_layers("none", 8, 0.1)))


def test_effective_xi_finite_with_dropout():
    xi = effective_xi(schedule_layers("constant", 8, 0.1))
    assert math.isfinite(xi)
    assert xi > 0


@pytest.mark.parametrize("depth", [1, 2, 7, 24])
@pytest.mark.parametrize("power", [0.0, 0.5, 1.0, 2.0, 4.0])
@pytest.mark.parametrize("h_max", [None, 0.1, 0.2])
def test_power_profiles_preserve_exact_budget(depth, power, h_max):
    layers = power_profile_layers(depth, 0.1, power, orientation="early", h_max=h_max)
    assert len(layers) == depth
    assert np.mean(layers) == pytest.approx(0.1, abs=1e-12)
    assert all(h >= 0 for h in layers)
    if h_max is not None:
        assert max(layers) <= h_max + 1e-12


@pytest.mark.parametrize("power", [0.0, 1.0, 2.0, 8.0])
@pytest.mark.parametrize("h_max", [None, 0.2])
def test_early_power_profile_is_exact_reversal_of_late(power, h_max):
    late = power_profile_layers(11, 0.1, power, orientation="late", h_max=h_max)
    early = power_profile_layers(11, 0.1, power, orientation="early", h_max=h_max)
    assert early == pytest.approx(late[::-1])


def test_zero_power_profile_is_constant():
    assert power_profile_layers(
        7, 0.1, 0.0, orientation="early", h_max=0.2
    ) == pytest.approx([0.1] * 7)


def test_quadratic_schedule_aliases_are_budget_matched_reversals():
    late = schedule_layers("quadratic", 12, 0.1, h_max=0.2)
    early = schedule_layers("reverse_quadratic", 12, 0.1, h_max=0.2)
    assert np.mean(late) == pytest.approx(0.1)
    assert max(late) <= 0.2 + 1e-12
    assert early == pytest.approx(late[::-1])


def test_power_profiles_are_monotone_in_requested_orientation():
    late = power_profile_layers(12, 0.1, 2.0, orientation="late", h_max=0.2)
    early = power_profile_layers(12, 0.1, 2.0, orientation="early", h_max=0.2)
    assert np.all(np.diff(late) >= 0)
    assert np.all(np.diff(early) <= 0)


def test_legacy_linear_profiles_are_unchanged():
    assert schedule_layers("linear", 4, 0.1) == pytest.approx(
        [0.0, 0.2 / 3, 0.4 / 3, 0.2]
    )
    assert schedule_layers("reverse_linear", 4, 0.1) == pytest.approx(
        [0.2, 0.4 / 3, 0.2 / 3, 0.0]
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"depth": 0, "h_bar": 0.1, "power": 2.0},
        {"depth": 8, "h_bar": -0.1, "power": 2.0},
        {"depth": 8, "h_bar": 0.1, "power": -1.0},
        {"depth": 8, "h_bar": 0.1, "power": 2.0, "orientation": "middle"},
        {"depth": 8, "h_bar": 0.2, "power": 2.0, "h_max": 0.1},
    ],
)
def test_power_profile_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        power_profile_layers(**kwargs)


def test_field_damage_keeps_zero_layers_in_depth_average():
    h_layers = [0.2, 0.2, 0.0, 0.0]
    expected = 0.5 * 0.2 ** (1 / 3)
    assert field_damage(h_layers, "kinked") == pytest.approx(expected)
    assert effective_xi(h_layers, "kinked") == pytest.approx(1 / expected)


def test_field_damage_uses_smooth_class_exponent():
    h_layers = [0.2, 0.0]
    assert field_damage(h_layers, "smooth") == pytest.approx(0.5 * np.sqrt(0.2))


def test_sparse_schedule_has_longer_xi_at_fixed_budget():
    constant = [0.1] * 8
    sparse = [0.2] * 4 + [0.0] * 4
    assert np.mean(sparse) == pytest.approx(np.mean(constant))
    assert effective_xi(sparse, "kinked") > effective_xi(constant, "kinked")


def test_effective_xi_supports_legacy_notebook_prefactor():
    h_layers = [0.1] * 6
    prefactor = 0.75
    assert effective_xi(h_layers, "kinked", prefactor=prefactor) == pytest.approx(
        effective_xi(h_layers, "kinked") / prefactor
    )


def test_legacy_v_profiles_are_available_from_canonical_builder():
    assert schedule_layers("v_shape", 5, 0.1) == pytest.approx(
        [0.2, 0.1, 0.0, 0.1, 0.2]
    )
    assert schedule_layers("inverse_v", 5, 0.1) == pytest.approx(
        [0.0, 0.1, 0.2, 0.1, 0.0]
    )
    assert schedule_layers("v_shape", 1, 0.1) == [0.1]
    assert schedule_layers("inverse_v", 1, 0.1) == [0.1]


def test_legacy_budget_control_profiles_match_notebook_values():
    assert schedule_layers("double", 6, 0.1, 0.2) == pytest.approx([0.2] * 6)
    assert schedule_layers("triple", 6, 0.1, 0.2) == pytest.approx([0.3] * 6)
    assert schedule_layers("big_step", 6, 0.1, 0.2) == pytest.approx(
        [0.3, 0.3, 0.0, 0.0, 0.0, 0.0]
    )
    assert schedule_layers("reverse_step", 6, 0.1, 0.2) == pytest.approx(
        [0.2, 0.2, 0.2, 0.0, 0.0, 0.0]
    )
    assert schedule_layers("step", 6, 0.1, 0.2) == pytest.approx(
        [0.0, 0.0, 0.0, 0.2, 0.2, 0.2]
    )


def test_step_saturates_cap_before_using_one_partial_layer():
    early = schedule_layers("reverse_step", 6, 0.05, 0.2)
    late = schedule_layers("step", 6, 0.05, 0.2)
    assert early == pytest.approx([0.2, 0.1, 0.0, 0.0, 0.0, 0.0])
    assert late == pytest.approx(early[::-1])
    assert np.mean(early) == pytest.approx(0.05)
    assert max(early) == pytest.approx(0.2)


@pytest.mark.parametrize("values", [[-0.1, 0.2], [0.1, float("nan")]])
def test_field_damage_rejects_invalid_fields(values):
    with pytest.raises(ValueError, match="nonnegative"):
        field_damage(values, "kinked")
