"""Preregistered statistics for dropout profile and muTransfer trials."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

from dropout_mft.statistics import align_by_seed as align_by_seed
from dropout_mft.statistics import (
    bootstrap_mean_ci,
    exact_paired_permutation_pvalue,
    holm_adjust,
)


def select_proxy_learning_rates(records: Sequence[dict]) -> dict[str, float]:
    """Select each parameterization by final validation loss only."""

    grouped: dict[tuple[str, float], list[float]] = {}
    for record in records:
        parameterization = record["factors"]["parameterization"]
        learning_rate = float(record["factors"]["learning_rate"])
        validation = np.asarray(record["curves"]["validation_loss"], dtype=float)
        grouped.setdefault((parameterization, learning_rate), []).append(
            float(validation[-1])
        )
    selected: dict[str, float] = {}
    for parameterization in sorted({key[0] for key in grouped}):
        candidates = [
            (float(np.mean(losses)), learning_rate)
            for (kind, learning_rate), losses in grouped.items()
            if kind == parameterization
        ]
        selected[parameterization] = min(candidates)[1]
    return selected


def transfer_regret(transferred_loss: float, oracle_loss: float) -> dict[str, float]:
    if oracle_loss <= 0:
        raise ValueError("oracle_loss must be positive")
    return {
        "absolute": float(transferred_loss - oracle_loss),
        "relative": float((transferred_loss - oracle_loss) / oracle_loss),
    }


def learning_rate_grid_distance(
    transferred_learning_rate: float,
    oracle_learning_rate: float,
    grid: Sequence[float],
) -> int:
    """Return the number of prespecified LR-grid intervals between two values."""

    values = np.asarray(grid, dtype=float)
    if values.ndim != 1 or len(values) < 2 or np.any(values <= 0):
        raise ValueError("grid must contain at least two positive learning rates")
    values = np.unique(values)
    if len(values) < 2:
        raise ValueError("grid must contain at least two distinct learning rates")

    def locate(value: float) -> int:
        matches = np.flatnonzero(np.isclose(values, value, rtol=1e-12, atol=0.0))
        if len(matches) != 1:
            raise ValueError(f"Learning rate {value:g} is not a unique grid point")
        return int(matches[0])

    return abs(locate(transferred_learning_rate) - locate(oracle_learning_rate))


def mup_transfer_claim_gate(
    width_records: Sequence[dict],
    *,
    expected_widths: int = 4,
    required_passes: int = 3,
    max_grid_intervals: int = 1,
    max_relative_regret: float = 0.01,
) -> dict:
    """Evaluate the preregistered muTransfer claim without filling missing widths."""

    widths = [int(record["width"]) for record in width_records]
    if len(widths) != len(set(widths)):
        raise ValueError("width_records must contain one row per width")
    passes = [
        int(record["lr_grid_distance"]) <= max_grid_intervals
        and float(record["relative_regret"]) <= max_relative_regret
        for record in width_records
    ]
    evaluable = len(width_records) == expected_widths
    return {
        "evaluable": evaluable,
        "passing_widths": int(sum(passes)),
        "required_passing_widths": required_passes,
        "expected_widths": expected_widths,
        "successful": bool(evaluable and sum(passes) >= required_passes),
    }


def profile_transfer_claim_gate(
    width_records: Sequence[dict],
    *,
    expected_widths: int = 4,
    alpha: float = 0.05,
    harm_margin: float = 0.01,
) -> dict:
    """Gate one prespecified paired profile contrast across widths.

    Each record must contain a width, the identical ordered seed cohort, and
    one paired loss difference per seed. Negative differences favor the first
    profile in the contrast. The pooled test treats seed as the resampling and
    sign-flip unit: each seed is averaged over widths before inference. The
    width interaction is the mean within-seed change in the contrast per
    doubling of width.

    Success requires a nonzero pooled benefit and noninferiority at every
    width. Noninferiority uses the upper endpoint of the two-sided 95% paired
    bootstrap interval, which is conservative relative to a one-sided 95%
    interval. The interaction is reported, not used as a gate.
    """

    if expected_widths < 1:
        raise ValueError("expected_widths must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    if not math.isfinite(harm_margin) or harm_margin < 0:
        raise ValueError("harm_margin must be finite and nonnegative")

    widths = [int(record["width"]) for record in width_records]
    if len(widths) != len(set(widths)):
        raise ValueError("width_records must contain one row per width")
    ordered = sorted(width_records, key=lambda record: int(record["width"]))
    ordered_widths = np.asarray([int(record["width"]) for record in ordered])
    if np.any(ordered_widths <= 0):
        raise ValueError("widths must be positive")

    seed_cohort: tuple[int, ...] | None = None
    columns: list[np.ndarray] = []
    for record in ordered:
        seeds = tuple(int(seed) for seed in record["seeds"])
        differences = np.asarray(record["paired_differences"], dtype=float)
        if len(seeds) == 0 or len(seeds) != len(set(seeds)):
            raise ValueError("each width must contain a nonempty unique seed cohort")
        if differences.ndim != 1 or len(differences) != len(seeds):
            raise ValueError("paired_differences must align one-to-one with seeds")
        if not np.all(np.isfinite(differences)):
            raise ValueError("paired_differences must be finite")
        if seed_cohort is None:
            seed_cohort = seeds
        elif seeds != seed_cohort:
            raise ValueError("every width must use the identical ordered seed cohort")
        columns.append(differences)

    evaluable = len(ordered) == expected_widths
    if not ordered:
        return {
            "evaluable": False,
            "expected_widths": expected_widths,
            "harm_margin": harm_margin,
            "successful": False,
        }

    paired_matrix = np.column_stack(columns)
    width_summaries = []
    width_p_values = []
    for width, differences in zip(ordered_widths, columns, strict=True):
        interval = bootstrap_mean_ci(differences, seed=int(width))
        p_value = exact_paired_permutation_pvalue(differences)
        width_p_values.append(p_value)
        width_summaries.append(
            {
                "width": int(width),
                "mean": interval["mean"],
                "ci95_lower": interval["lower"],
                "ci95_upper": interval["upper"],
                "permutation_p_value": p_value,
                "noninferior": bool(interval["upper"] <= harm_margin),
                "observed_beyond_harm_margin": bool(interval["mean"] > harm_margin),
            }
        )
    adjusted = holm_adjust(width_p_values)
    for summary, adjusted_p_value in zip(width_summaries, adjusted, strict=True):
        summary["holm_p_value_across_widths"] = float(adjusted_p_value)

    pooled_seed_effects = paired_matrix.mean(axis=1)
    pooled_interval = bootstrap_mean_ci(pooled_seed_effects, seed=0)
    pooled_p_value = exact_paired_permutation_pvalue(pooled_seed_effects)

    log_width = np.log2(ordered_widths.astype(float))
    centered_log_width = log_width - log_width.mean()
    denominator = float(np.dot(centered_log_width, centered_log_width))
    if denominator == 0.0:
        seed_slopes = np.zeros(len(pooled_seed_effects), dtype=float)
    else:
        seed_slopes = paired_matrix @ centered_log_width / denominator
    interaction_interval = bootstrap_mean_ci(seed_slopes, seed=1)
    interaction_p_value = exact_paired_permutation_pvalue(seed_slopes)

    pooled_benefit = bool(pooled_interval["mean"] < 0 and pooled_p_value < alpha)
    all_widths_noninferior = bool(
        evaluable and all(summary["noninferior"] for summary in width_summaries)
    )
    return {
        "evaluable": evaluable,
        "expected_widths": expected_widths,
        "n_seeds": int(paired_matrix.shape[0]),
        "alpha": alpha,
        "harm_margin": harm_margin,
        "pooled_unit": "seed_mean_across_widths",
        "noninferiority_rule": "paired_bootstrap_ci95_upper_le_harm_margin",
        "width_interaction_unit": "test_loss_per_log2_width",
        "pooled_paired_effect": pooled_interval["mean"],
        "pooled_ci95_lower": pooled_interval["lower"],
        "pooled_ci95_upper": pooled_interval["upper"],
        "pooled_permutation_p_value": pooled_p_value,
        "pooled_benefit": pooled_benefit,
        "all_widths_noninferior": all_widths_noninferior,
        "widths_noninferior": int(
            sum(summary["noninferior"] for summary in width_summaries)
        ),
        "widths_observed_beyond_harm_margin": int(
            sum(summary["observed_beyond_harm_margin"] for summary in width_summaries)
        ),
        "width_interaction_per_doubling": interaction_interval["mean"],
        "width_interaction_ci95_lower": interaction_interval["lower"],
        "width_interaction_ci95_upper": interaction_interval["upper"],
        "width_interaction_permutation_p_value": interaction_p_value,
        "width_effects": width_summaries,
        "successful": bool(evaluable and pooled_benefit and all_widths_noninferior),
    }


def first_crossing(
    curve: Sequence[float],
    threshold: float,
    *,
    direction: str = "max",
) -> int | None:
    values = np.asarray(curve, dtype=float)
    if direction == "max":
        matches = np.flatnonzero(values >= threshold)
    elif direction == "min":
        matches = np.flatnonzero(values <= threshold)
    else:
        raise ValueError("direction must be 'max' or 'min'")
    return int(matches[0]) if len(matches) else None


def restricted_time_to_threshold(
    curves: Iterable[Sequence[float]],
    threshold: float,
    *,
    horizon: int,
    direction: str = "max",
) -> dict[str, float]:
    """Restricted mean time with non-reachers censored at ``horizon``."""

    observed: list[int] = []
    reached = 0
    for curve in curves:
        crossing = first_crossing(curve, threshold, direction=direction)
        if crossing is None or crossing > horizon:
            observed.append(horizon)
        else:
            observed.append(crossing)
            reached += 1
    if not observed:
        raise ValueError("At least one curve is required")
    return {
        "restricted_mean": float(np.mean(observed)),
        "reach_rate": reached / len(observed),
        "horizon": float(horizon),
    }


def profile_rank_stability(
    proxy_scores: dict[str, float], target_scores: dict[str, float]
) -> float:
    if set(proxy_scores) != set(target_scores) or len(proxy_scores) < 2:
        raise ValueError("Rank comparisons require the same two or more profiles")
    labels = sorted(proxy_scores)
    from scipy.stats import rankdata

    proxy_order = rankdata([proxy_scores[label] for label in labels], method="average")
    target_order = rankdata(
        [target_scores[label] for label in labels], method="average"
    )
    correlation = np.corrcoef(proxy_order, target_order)[0, 1]
    return float(correlation) if math.isfinite(correlation) else 0.0
