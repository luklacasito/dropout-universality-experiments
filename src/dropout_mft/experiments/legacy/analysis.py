"""Analysis helpers for the isolated exact-paper comparison cohort."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from dropout_mft.experiments.scale_transfer.analysis import bootstrap_mean_ci
from dropout_mft.experiments.legacy.protocol import (
    LEGACY_PROFILE_IDS,
    LEGACY_SCHEMA_VERSION,
    LEGACY_SEEDS,
    LegacyTrialSpec,
    legacy_trial_specs,
)


REFERENCE_PROFILE_MAP = {
    "uniform": "constant",
    "linear_early": "reverse_linear",
    "step_early": "reverse_step",
}
NEW_PROFILE_IDS = ("quadratic_early", "quartic_early")
PRESPECIFIED_LEGACY_BASELINE = "step_early"


class LegacyResultsError(RuntimeError):
    """Raised when results cannot support an apples-to-apples comparison."""


def validate_complete_cohort(trials: Sequence[dict]) -> dict[tuple[str, int], dict]:
    """Validate all 125 exact specs and return an explicit paired index."""

    expected = {(spec.profile_id, spec.seed): spec for spec in legacy_trial_specs()}
    indexed: dict[tuple[str, int], dict] = {}
    for trial in trials:
        try:
            spec = LegacyTrialSpec(**trial["factors"])
            key = (spec.profile_id, spec.seed)
            curves = trial["curves"]
            test_loss = np.asarray(curves["test_loss"], dtype=float)
            test_accuracy = np.asarray(curves["test_accuracy"], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise LegacyResultsError("Malformed legacy trial") from exc
        if key in indexed:
            raise LegacyResultsError(f"Duplicate legacy profile/seed pair: {key}")
        if key not in expected or spec != expected[key]:
            raise LegacyResultsError(f"Unexpected legacy trial factors: {key}")
        valid = (
            trial.get("schema_version") == LEGACY_SCHEMA_VERSION
            and trial.get("artifact_type") == "legacy_apples_to_apples_trial"
            and trial.get("trial", {}).get("status") == "complete"
            and trial.get("trial", {}).get("trial_id") == spec.trial_id
            and trial.get("trial", {}).get("config_hash") == spec.config_hash
            and test_loss.shape == (spec.epochs,)
            and test_accuracy.shape == (spec.epochs,)
            and np.all(np.isfinite(test_loss))
            and np.all(np.isfinite(test_accuracy))
            and np.all((0 <= test_accuracy) & (test_accuracy <= 100))
            and trial.get("test", {}).get("selected_epoch") == spec.epochs - 1
            and np.isclose(trial.get("test", {}).get("loss"), test_loss[-1])
            and np.isclose(
                trial.get("test", {}).get("accuracy_percent"), test_accuracy[-1]
            )
        )
        if not valid:
            raise LegacyResultsError(f"Invalid completed legacy trial: {key}")
        indexed[key] = trial
    missing = set(expected) - set(indexed)
    unexpected = set(indexed) - set(expected)
    if missing or unexpected or len(indexed) != 125:
        raise LegacyResultsError(
            f"Exact legacy cohort is incomplete: missing={len(missing)}, "
            f"unexpected={len(unexpected)}, files={len(indexed)}, expected=125"
        )
    return indexed


def _final_arrays(indexed, profile_id: str) -> tuple[np.ndarray, np.ndarray]:
    loss = np.asarray(
        [indexed[(profile_id, seed)]["test"]["loss"] for seed in LEGACY_SEEDS],
        dtype=float,
    )
    accuracy = np.asarray(
        [
            indexed[(profile_id, seed)]["test"]["accuracy_percent"]
            for seed in LEGACY_SEEDS
        ],
        dtype=float,
    )
    return loss, accuracy


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


def _holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def profile_summary_rows(indexed: dict[tuple[str, int], dict]) -> list[dict]:
    rows = []
    for profile_id in LEGACY_PROFILE_IDS:
        loss, accuracy = _final_arrays(indexed, profile_id)
        rows.append(
            {
                "profile_id": profile_id,
                "n_paired_seeds": len(loss),
                "mean_final_test_loss": float(loss.mean()),
                "sem_final_test_loss": float(loss.std(ddof=1) / math.sqrt(len(loss))),
                "mean_final_test_accuracy_percent": float(accuracy.mean()),
                "sem_final_test_accuracy_pp": float(
                    accuracy.std(ddof=1) / math.sqrt(len(accuracy))
                ),
            }
        )
    return sorted(rows, key=lambda row: row["mean_final_test_loss"])


def _paired_row(indexed, candidate_id: str, baseline_id: str, *, ci_seed: int) -> dict:
    candidate_loss, candidate_accuracy = _final_arrays(indexed, candidate_id)
    baseline_loss, baseline_accuracy = _final_arrays(indexed, baseline_id)
    loss_difference = candidate_loss - baseline_loss
    accuracy_difference = candidate_accuracy - baseline_accuracy
    loss_ci = bootstrap_mean_ci(loss_difference, seed=ci_seed)
    accuracy_ci = bootstrap_mean_ci(accuracy_difference, seed=ci_seed + 10_000)

    crossing_epochs = []
    reached = []
    for seed, threshold in zip(LEGACY_SEEDS, baseline_accuracy, strict=True):
        curve = np.asarray(indexed[(candidate_id, seed)]["curves"]["test_accuracy"])
        matches = np.flatnonzero(curve >= threshold)
        reached.append(bool(len(matches)))
        crossing_epochs.append(int(matches[0] + 1) if len(matches) else len(curve))
    return {
        "candidate_profile": candidate_id,
        "baseline_profile": baseline_id,
        "n_paired_seeds": len(LEGACY_SEEDS),
        "mean_final_test_loss_difference": loss_ci["mean"],
        "loss_difference_ci95_lower": loss_ci["lower"],
        "loss_difference_ci95_upper": loss_ci["upper"],
        "loss_exact_paired_signflip_p_value": exact_paired_signflip_pvalue(
            loss_difference
        ),
        "loss_holm_p_value_prespecified_family": None,
        "relative_loss_reduction_of_means": float(
            (baseline_loss.mean() - candidate_loss.mean()) / baseline_loss.mean()
        ),
        "mean_final_accuracy_difference_pp": accuracy_ci["mean"],
        "accuracy_difference_ci95_lower_pp": accuracy_ci["lower"],
        "accuracy_difference_ci95_upper_pp": accuracy_ci["upper"],
        "accuracy_threshold_reach_rate": float(np.mean(reached)),
        "restricted_mean_epochs_to_baseline_final_accuracy": float(
            np.mean(crossing_epochs)
        ),
        "compute_endpoint_is_diagnostic": True,
    }


def paired_comparison_rows(
    indexed: dict[tuple[str, int], dict]
) -> tuple[list[dict], str]:
    """Run descriptive references and the four prespecified new-profile contrasts."""

    best_legacy = PRESPECIFIED_LEGACY_BASELINE
    rows = []
    for index, candidate_id in enumerate(
        profile for profile in LEGACY_PROFILE_IDS if profile != "uniform"
    ):
        row = _paired_row(indexed, candidate_id, "uniform", ci_seed=100 + index)
        row["contrast_status"] = (
            "prespecified"
            if candidate_id in NEW_PROFILE_IDS
            else "descriptive_legacy_reference"
        )
        rows.append(row)
    for index, candidate_id in enumerate(NEW_PROFILE_IDS):
        row = _paired_row(indexed, candidate_id, best_legacy, ci_seed=200 + index)
        row["contrast_status"] = "prespecified"
        rows.append(row)
    inferential = [row for row in rows if row["contrast_status"] == "prespecified"]
    adjusted = _holm_adjust(
        [row["loss_exact_paired_signflip_p_value"] for row in inferential]
    )
    for row, p_value in zip(inferential, adjusted, strict=True):
        row["loss_holm_p_value_prespecified_family"] = float(p_value)
    return rows, best_legacy


def validate_saved_original(saved: dict) -> None:
    expected_config = {
        "N_SIMULATIONS": 25,
        "EPOCHS": 75,
        "DEPTH": 6,
        "WIDTH": 256,
        "H_BAR": 0.1,
        "H_MAX": 0.2,
        "SIGMA_W_SQ": 1.98,
        "SIGMA_B_SQ": 0.02,
        "LEARNING_RATE": 1e-4,
        "LR_MIN": 1e-7,
    }
    try:
        config = saved["config"]
        results = saved["results"]
    except (KeyError, TypeError) as exc:
        raise LegacyResultsError("Saved original result has no config/results") from exc
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise LegacyResultsError("Saved original config is not the exact MLP reference")
    for saved_id in REFERENCE_PROFILE_MAP.values():
        for metric in ("test_loss", "test_acc"):
            values = np.asarray(results.get(saved_id, {}).get(metric), dtype=float)
            if values.shape != (25, 75) or not np.all(np.isfinite(values)):
                raise LegacyResultsError(
                    f"Saved original {saved_id}/{metric} is absent or malformed"
                )


def reference_reproduction_rows(
    indexed: dict[tuple[str, int], dict], saved: dict
) -> list[dict]:
    """Compare rerun references with the corresponding saved original arrays."""

    validate_saved_original(saved)
    rows = []
    for rerun_id, saved_id in REFERENCE_PROFILE_MAP.items():
        rerun_loss = np.stack(
            [indexed[(rerun_id, seed)]["curves"]["test_loss"] for seed in LEGACY_SEEDS]
        )
        rerun_accuracy = np.stack(
            [
                indexed[(rerun_id, seed)]["curves"]["test_accuracy"]
                for seed in LEGACY_SEEDS
            ]
        )
        saved_loss = np.asarray(saved["results"][saved_id]["test_loss"], dtype=float)
        saved_accuracy = np.asarray(saved["results"][saved_id]["test_acc"], dtype=float)
        final_loss_difference = rerun_loss[:, -1] - saved_loss[:, -1]
        final_accuracy_difference = rerun_accuracy[:, -1] - saved_accuracy[:, -1]
        loss_ci = bootstrap_mean_ci(final_loss_difference, seed=300)
        accuracy_ci = bootstrap_mean_ci(final_accuracy_difference, seed=301)
        rows.append(
            {
                "rerun_profile": rerun_id,
                "saved_original_profile": saved_id,
                "n_paired_seeds": 25,
                "saved_mean_final_test_loss": float(saved_loss[:, -1].mean()),
                "rerun_mean_final_test_loss": float(rerun_loss[:, -1].mean()),
                "rerun_minus_saved_loss": loss_ci["mean"],
                "rerun_minus_saved_loss_ci95_lower": loss_ci["lower"],
                "rerun_minus_saved_loss_ci95_upper": loss_ci["upper"],
                "saved_mean_final_test_accuracy_percent": float(
                    saved_accuracy[:, -1].mean()
                ),
                "rerun_mean_final_test_accuracy_percent": float(
                    rerun_accuracy[:, -1].mean()
                ),
                "rerun_minus_saved_accuracy_pp": accuracy_ci["mean"],
                "rerun_minus_saved_accuracy_ci95_lower_pp": accuracy_ci["lower"],
                "rerun_minus_saved_accuracy_ci95_upper_pp": accuracy_ci["upper"],
                "test_loss_curve_rmse": float(
                    np.sqrt(np.mean((rerun_loss - saved_loss) ** 2))
                ),
                "test_accuracy_curve_rmse_pp": float(
                    np.sqrt(np.mean((rerun_accuracy - saved_accuracy) ** 2))
                ),
                "bitwise_identical_arrays": bool(
                    np.array_equal(rerun_loss, saved_loss)
                    and np.array_equal(rerun_accuracy, saved_accuracy)
                ),
            }
        )
    return rows
