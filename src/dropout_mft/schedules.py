"""Dropout schedules used in the experiments."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral

import numpy as np


DISPLAY_NAMES = {
    "none": "No dropout",
    "constant": "Constant",
    "linear": "Linear (late)",
    "reverse_linear": "Linear (early)",
    "quadratic": "Quadratic (late)",
    "reverse_quadratic": "Quadratic (early)",
    "step": "Step (late)",
    "reverse_step": "Step (early)",
    "big_step": "Big step (1/3)",
    "double": "Double (2h)",
    "triple": "Triple (3h)",
    "v_shape": "V-shape",
    "inverse_v": "Inverse V",
}

MARKERS = {
    "none": "s",
    "constant": "o",
    "reverse_step": "D",
    "big_step": "v",
    "step": "^",
    "linear": ">",
    "reverse_linear": "<",
    "quadratic": "p",
    "reverse_quadratic": "H",
    "double": "P",
    "triple": "X",
    "v_shape": "*",
    "inverse_v": "h",
}

FIELD_EXPONENTS = {
    "smooth": 1 / 2,
    "kinked": 1 / 3,
}


def _rescale_profile(
    weights: np.ndarray,
    h_bar: float,
    h_max: float | None,
) -> np.ndarray:
    """Rescale positive profile weights to an exact mean, with an optional cap."""

    if not math.isfinite(h_bar) or h_bar < 0:
        raise ValueError("h_bar must be a finite non-negative number")

    if h_max is None:
        if h_bar == 0:
            return np.zeros_like(weights)
        return weights * (h_bar / float(np.mean(weights)))

    if not math.isfinite(h_max) or h_max < 0:
        raise ValueError("h_max must be a finite non-negative number")
    if h_bar > h_max:
        raise ValueError("h_max must be at least h_bar for an exact-budget profile")
    if h_bar == 0:
        return np.zeros_like(weights)

    # Active-set water filling solves
    #   h_l = min(scale * weight_l, h_max)
    # while preserving sum_l h_l = depth * h_bar.  Power-profile weights are
    # strictly positive, so h_max >= h_bar is sufficient for feasibility.
    values = np.zeros_like(weights)
    free = np.ones(weights.size, dtype=bool)
    remaining = h_bar * weights.size

    while np.any(free):
        free_indices = np.flatnonzero(free)
        scale = remaining / float(np.sum(weights[free]))
        proposals = scale * weights[free]
        saturated = proposals > h_max

        if not np.any(saturated):
            values[free] = proposals
            break

        saturated_indices = free_indices[saturated]
        values[saturated_indices] = h_max
        free[saturated_indices] = False
        remaining -= h_max * saturated_indices.size

    return values


def power_profile_layers(
    depth: int,
    h_bar: float,
    power: float,
    *,
    orientation: str = "late",
    h_max: float | None = None,
) -> list[float]:
    """Build an exact-budget monotone power profile.

    Layer locations are sampled at cell centers.  This keeps every raw weight
    positive at finite depth, so capped profiles remain feasible whenever
    ``h_max >= h_bar``.  ``orientation='early'`` is the exact layer reversal of
    ``orientation='late'``.
    """

    if not isinstance(depth, Integral) or isinstance(depth, bool) or depth < 1:
        raise ValueError("depth must be a positive integer")
    if not math.isfinite(power) or power < 0:
        raise ValueError("power must be a finite non-negative number")
    if orientation not in {"early", "late"}:
        raise ValueError("orientation must be 'early' or 'late'")

    depth = int(depth)
    locations = (np.arange(depth, dtype=float) + 0.5) / depth
    weights = locations**power
    if orientation == "early":
        weights = weights[::-1]

    return _rescale_profile(weights, h_bar, h_max).tolist()


def _saturated_step_layers(
    depth: int,
    h_bar: float,
    h_max: float | None,
    *,
    orientation: str,
) -> list[float]:
    """Return a bang--bang step with at most one partially filled layer."""

    if not isinstance(depth, Integral) or isinstance(depth, bool) or depth < 1:
        raise ValueError("depth must be a positive integer")
    if not math.isfinite(h_bar) or h_bar < 0:
        raise ValueError("h_bar must be a finite non-negative number")
    cap = 2.0 * h_bar if h_max is None else h_max
    if not math.isfinite(cap) or cap < 0:
        raise ValueError("h_max must be a finite non-negative number")
    if h_bar > cap:
        raise ValueError("h_max must be at least h_bar for an exact-budget step")
    if orientation not in {"early", "late"}:
        raise ValueError("orientation must be 'early' or 'late'")
    if h_bar == 0:
        return [0.0] * int(depth)

    total = int(depth) * h_bar
    n_full = min(int(depth), int(math.floor(total / cap + 1e-12)))
    remainder = total - n_full * cap
    tolerance = 1e-12 * max(1.0, total, cap)
    if remainder < tolerance:
        remainder = 0.0
    active = [cap] * n_full
    if remainder > 0:
        active.append(remainder)
    values = active + [0.0] * (int(depth) - len(active))
    if orientation == "late":
        values.reverse()
    return values


def schedule_layers(
    schedule: str, depth: int, h_bar: float, h_max: float | None = None
) -> list[float]:
    """Build the layer-wise dropout field h_l with mean h_bar."""

    if schedule == "none":
        return [0.0] * depth
    if schedule == "constant":
        return [h_bar] * depth
    if schedule == "double":
        return [2.0 * h_bar] * depth
    if schedule == "triple":
        return [3.0 * h_bar] * depth
    if schedule == "linear":
        if depth == 1:
            return [h_bar]
        return [2.0 * h_bar * i / (depth - 1) for i in range(depth)]
    if schedule == "reverse_linear":
        if depth == 1:
            return [h_bar]
        return [2.0 * h_bar * (depth - 1 - i) / (depth - 1) for i in range(depth)]
    if schedule == "quadratic":
        return power_profile_layers(depth, h_bar, 2.0, orientation="late", h_max=h_max)
    if schedule == "reverse_quadratic":
        return power_profile_layers(depth, h_bar, 2.0, orientation="early", h_max=h_max)
    if schedule == "v_shape":
        if depth == 1:
            return [h_bar]
        return [2.0 * h_bar * abs(2.0 * i / (depth - 1) - 1.0) for i in range(depth)]
    if schedule == "inverse_v":
        if depth == 1:
            return [h_bar]
        return [
            2.0 * h_bar * (1.0 - abs(2.0 * i / (depth - 1) - 1.0)) for i in range(depth)
        ]
    if schedule == "step":
        return _saturated_step_layers(depth, h_bar, h_max, orientation="late")
    if schedule == "reverse_step":
        return _saturated_step_layers(depth, h_bar, h_max, orientation="early")
    if schedule == "big_step":
        n_drop = max(1, int(math.ceil(depth / 3)))
        h_adj = h_bar * depth / n_drop
        return [h_adj] * n_drop + [0.0] * (depth - n_drop)
    raise ValueError(f"Unknown schedule: {schedule}")


def field_damage(h_layers: Sequence[float], activation_class: str = "kinked") -> float:
    """Return ``L^-1 sum_l h_l^a`` for the activation's field exponent.

    The leading exponent is ``a=1/2`` for smooth activations and ``a=1/3``
    for the kinked class.  Zero-field layers remain in the denominator, which
    is essential when comparing sparse and dense schedules at fixed depth.
    """

    values = [float(value) for value in h_layers]
    if not values:
        return 0.0
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("field values must be finite and nonnegative")

    # Retain the legacy fallback: unrecognized labels use the kinked exponent.
    power = FIELD_EXPONENTS.get(activation_class, FIELD_EXPONENTS["kinked"])
    return float(sum(h**power for h in values) / len(values))


def effective_xi(
    h_layers: Sequence[float],
    activation_class: str = "kinked",
    *,
    prefactor: float = 1.0,
) -> float:
    """Mean-field correlation length with an optional class-specific prefactor."""

    if not math.isfinite(prefactor) or prefactor <= 0:
        raise ValueError("prefactor must be finite and positive")
    damage = field_damage(h_layers, activation_class)
    if damage <= 0:
        return float("inf")
    return 1.0 / (prefactor * damage)
