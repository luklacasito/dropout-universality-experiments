"""Paired analysis primitives; study modules define contrasts and claim gates.

Historical sign-flip implementations retain their original sample limits and
roundoff tolerances so published comparisons are unchanged.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np


def align_by_seed(
    candidate: Sequence[dict], baseline: Sequence[dict]
) -> tuple[np.ndarray, np.ndarray]:
    """Align scalar records by explicit seed and reject ambiguous pairing."""

    def index(records: Sequence[dict]) -> dict[int, float]:
        output: dict[int, float] = {}
        for record in records:
            seed = int(record["seed"])
            if seed in output:
                raise ValueError(f"Duplicate seed: {seed}")
            output[seed] = float(record["value"])
        return output

    candidate_by_seed = index(candidate)
    baseline_by_seed = index(baseline)
    if set(candidate_by_seed) != set(baseline_by_seed):
        raise ValueError("Candidate and baseline must contain identical seed sets")
    seeds = sorted(candidate_by_seed)
    return (
        np.asarray([candidate_by_seed[seed] for seed in seeds]),
        np.asarray([baseline_by_seed[seed] for seed in seeds]),
    )


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("values must be a nonempty one-dimensional sequence")
    if not np.all(np.isfinite(values)):
        raise ValueError("values must be finite")
    if not 0 < confidence < 1 or n_resamples <= 0:
        raise ValueError("Invalid bootstrap configuration")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_resamples, len(values)), replace=True).mean(
        axis=1
    )
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(np.mean(values)),
        "lower": float(np.quantile(samples, alpha)),
        "upper": float(np.quantile(samples, 1.0 - alpha)),
    }


def exact_paired_permutation_pvalue(values: Sequence[float]) -> float:
    """Two-sided exact sign-flip test for paired differences."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("values must be a finite one-dimensional sequence")
    values = values[values != 0]
    if len(values) == 0:
        return 1.0
    if len(values) > 20:
        raise ValueError("Exact sign enumeration is limited to 20 nonzero pairs")
    observed = abs(float(values.mean()))
    extreme = 0
    total = 2 ** len(values)
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistic = abs(float(np.mean(values * np.asarray(signs))))
        extreme += statistic >= observed - 1e-15
    return extreme / total


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def exact_paired_signflip_pvalue(values: Sequence[float]) -> float:
    """Exact two-sided paired sign-flip test via meet-in-the-middle sums.

    Twenty-five pairs imply 33,554,432 sign assignments. Enumerating their
    Cartesian product in Python would be unnecessarily slow; splitting the
    values into two halves gives two arrays of at most 8,192 signed sums and
    counts the exact tail with binary searches.
    """

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("values must be a finite one-dimensional sequence")
    values = values[values != 0]
    if len(values) == 0:
        return 1.0
    if len(values) > 40:
        raise ValueError("Exact meet-in-the-middle sign flips are limited to 40 pairs")

    def signed_sums(part: np.ndarray) -> np.ndarray:
        sums = np.asarray([0.0])
        for value in part:
            sums = np.concatenate((sums + value, sums - value))
        return sums

    midpoint = len(values) // 2
    left = signed_sums(values[:midpoint])
    right = np.sort(signed_sums(values[midpoint:]))
    observed = abs(float(values.sum()))
    if observed == 0:
        return 1.0
    threshold = max(0.0, observed - 1e-14 * max(1.0, observed))
    extreme = 0
    for value in left:
        extreme += int(np.searchsorted(right, -threshold - value, side="right"))
        extreme += int(
            len(right) - np.searchsorted(right, threshold - value, side="left")
        )
    return float(extreme / (2 ** len(values)))


def paired_percentile_interval(
    values: Sequence[float],
    *,
    resamples: int = 10_000,
    seed: int = 20260812,
) -> tuple[float, float]:
    """Return the deterministic 95% bootstrap interval used by all cohorts."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("values must be a nonempty finite one-dimensional sequence")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
