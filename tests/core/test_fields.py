"""Focused checks for exact dropout fields and field-matched profiles."""

from __future__ import annotations

import numpy as np
import pytest

from dropout_mft.fields import (
    activation_second_moment,
    dropout_field,
    field_matched_power_profile,
    propagated_field_profile,
    reference_field_profile,
)


@pytest.mark.parametrize("variance", [0.0, 0.25, 1.0, 3.5])
def test_relu_second_moment_is_exact(variance):
    assert activation_second_moment(variance, "relu") == pytest.approx(variance / 2)


@pytest.mark.parametrize("activation", ["relu", "gelu"])
@pytest.mark.parametrize("probability", [0.0, 0.01, 0.1, 0.35])
def test_zero_bias_field_equals_dropout_probability(activation, probability):
    field = dropout_field(
        probability,
        variance=1.7,
        activation=activation,
        sigma_w_sq=2.0,
        sigma_b_sq=0.0,
    )
    assert field == pytest.approx(probability, abs=1e-14)


def test_relu_field_with_bias_matches_closed_form():
    probability = 0.1
    variance = 1.0
    sigma_w_sq = 1.98
    sigma_b_sq = 0.02
    moment = variance / 2
    covariance = sigma_w_sq * moment + sigma_b_sq
    next_variance = sigma_w_sq * moment / (1 - probability) + sigma_b_sq
    expected = 1 - covariance / next_variance
    assert dropout_field(
        probability,
        variance=variance,
        activation="relu",
        sigma_w_sq=sigma_w_sq,
        sigma_b_sq=sigma_b_sq,
    ) == pytest.approx(expected)


def test_reference_and_propagated_profiles_are_exact_in_zero_bias_relu_case():
    probabilities = [0.2, 0.1, 0.0, 0.05]
    reference = reference_field_profile(
        probabilities,
        activation="relu",
        sigma_w_sq=2.0,
        sigma_b_sq=0.0,
    )
    propagated, variances = propagated_field_profile(
        probabilities,
        activation="relu",
        sigma_w_sq=2.0,
        sigma_b_sq=0.0,
    )
    assert reference == pytest.approx(probabilities)
    assert propagated == pytest.approx(probabilities)
    assert len(variances) == len(probabilities) + 1
    expected_variances = [1.0]
    for probability in probabilities:
        expected_variances.append(expected_variances[-1] / (1 - probability))
    assert variances == pytest.approx(expected_variances)


def test_field_matched_quadratic_profile_hits_exact_field_budget_and_cap():
    early = field_matched_power_profile(
        depth=8,
        target_mean_field=0.1,
        power=2.0,
        orientation="early",
        p_max=0.2,
        activation="relu",
        sigma_w_sq=2.0,
        sigma_b_sq=0.0,
    )
    late = field_matched_power_profile(
        depth=8,
        target_mean_field=0.1,
        power=2.0,
        orientation="late",
        p_max=0.2,
        activation="relu",
        sigma_w_sq=2.0,
        sigma_b_sq=0.0,
    )
    fields = reference_field_profile(
        early,
        activation="relu",
        sigma_w_sq=2.0,
        sigma_b_sq=0.0,
    )
    assert np.mean(fields) == pytest.approx(0.1, abs=1e-12)
    assert max(early) <= 0.2 + 1e-12
    assert early == pytest.approx(late[::-1])


def test_gelu_second_moment_is_positive_and_deterministic():
    first = activation_second_moment(1.0, "gelu", quadrature_order=80)
    second = activation_second_moment(1.0, "gelu", quadrature_order=80)
    assert first > 0
    assert first == pytest.approx(second, rel=0, abs=0)


@pytest.mark.parametrize("probability", [-0.1, 1.0])
def test_dropout_field_rejects_invalid_probability(probability):
    with pytest.raises(ValueError):
        dropout_field(probability, variance=1.0, activation="relu")
