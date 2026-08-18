"""Exact one-step dropout fields and field-matched profile construction."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral

import numpy as np


def activation_second_moment(
    variance: float,
    activation: str,
    *,
    quadrature_order: int = 80,
) -> float:
    """Return E[phi(sqrt(q) Z)^2] for a standard-normal ``Z``."""

    if not math.isfinite(variance) or variance < 0:
        raise ValueError("variance must be finite and nonnegative")
    if activation == "relu":
        return 0.5 * variance
    if activation != "gelu":
        raise ValueError(f"Unsupported activation: {activation!r}")
    if not isinstance(quadrature_order, Integral) or quadrature_order < 8:
        raise ValueError("quadrature_order must be an integer of at least 8")

    nodes, weights = np.polynomial.hermite.hermgauss(int(quadrature_order))
    values = np.sqrt(variance) * np.sqrt(2.0) * nodes
    # Exact GELU: x Phi(x) = x/2 * (1 + erf(x/sqrt(2))).
    from scipy.special import erf

    activated = 0.5 * values * (1.0 + erf(values / np.sqrt(2.0)))
    return float(np.dot(weights, activated**2) / np.sqrt(np.pi))


def dropout_field(
    dropout_probability: float,
    *,
    variance: float,
    activation: str,
    sigma_w_sq: float = 1.98,
    sigma_b_sq: float = 0.02,
) -> float:
    """Compute the exact one-layer field ``1 - Fbar_rho(1)``.

    The calculation holds the incoming preactivation variance fixed.  It does
    not assume that a stationary variance exists under dropout, which matters
    for the finite-depth near-critical ReLU experiments.
    """

    if not 0 <= dropout_probability < 1:
        raise ValueError("dropout_probability must lie in [0, 1)")
    if sigma_w_sq <= 0 or sigma_b_sq < 0:
        raise ValueError("sigma_w_sq must be positive and sigma_b_sq nonnegative")
    keep_probability = 1.0 - dropout_probability
    moment = activation_second_moment(variance, activation)
    covariance = sigma_w_sq * moment + sigma_b_sq
    next_variance = sigma_w_sq * moment / keep_probability + sigma_b_sq
    if next_variance <= 0:
        return 0.0
    return 1.0 - covariance / next_variance


def reference_field_profile(
    dropout_layers: Sequence[float],
    *,
    activation: str,
    reference_variance: float = 1.0,
    sigma_w_sq: float = 1.98,
    sigma_b_sq: float = 0.02,
) -> list[float]:
    """Map raw probabilities to exact fields at a common reference variance."""

    return [
        dropout_field(
            value,
            variance=reference_variance,
            activation=activation,
            sigma_w_sq=sigma_w_sq,
            sigma_b_sq=sigma_b_sq,
        )
        for value in dropout_layers
    ]


def propagated_field_profile(
    dropout_layers: Sequence[float],
    *,
    activation: str,
    initial_variance: float = 1.0,
    sigma_w_sq: float = 1.98,
    sigma_b_sq: float = 0.02,
) -> tuple[list[float], list[float]]:
    """Return exact local fields and the finite-depth variance trajectory."""

    variance = float(initial_variance)
    variances = [variance]
    fields: list[float] = []
    for probability in dropout_layers:
        fields.append(
            dropout_field(
                probability,
                variance=variance,
                activation=activation,
                sigma_w_sq=sigma_w_sq,
                sigma_b_sq=sigma_b_sq,
            )
        )
        keep_probability = 1.0 - probability
        moment = activation_second_moment(variance, activation)
        variance = sigma_w_sq * moment / keep_probability + sigma_b_sq
        variances.append(variance)
    return fields, variances


def field_matched_power_profile(
    *,
    depth: int,
    target_mean_field: float,
    power: float,
    orientation: str,
    p_max: float,
    activation: str,
    reference_variance: float = 1.0,
    sigma_w_sq: float = 1.98,
    sigma_b_sq: float = 0.02,
    tolerance: float = 1e-12,
) -> list[float]:
    """Construct a capped power profile with an exact reference-field budget."""

    from .schedules import power_profile_layers

    if target_mean_field < 0 or not math.isfinite(target_mean_field):
        raise ValueError("target_mean_field must be finite and nonnegative")
    if not 0 < p_max < 1:
        raise ValueError("p_max must lie in (0, 1)")
    if target_mean_field == 0:
        return [0.0] * depth

    max_field = dropout_field(
        p_max,
        variance=reference_variance,
        activation=activation,
        sigma_w_sq=sigma_w_sq,
        sigma_b_sq=sigma_b_sq,
    )
    if target_mean_field > max_field + tolerance:
        raise ValueError("Requested mean field is infeasible under p_max")

    lower, upper = 0.0, p_max
    result = [0.0] * depth
    for _ in range(100):
        mean_probability = 0.5 * (lower + upper)
        result = power_profile_layers(
            depth,
            mean_probability,
            power,
            orientation=orientation,
            h_max=p_max,
        )
        fields = reference_field_profile(
            result,
            activation=activation,
            reference_variance=reference_variance,
            sigma_w_sq=sigma_w_sq,
            sigma_b_sq=sigma_b_sq,
        )
        error = float(np.mean(fields)) - target_mean_field
        if abs(error) <= tolerance:
            break
        if error < 0:
            lower = mean_probability
        else:
            upper = mean_probability
    return result
