#!/usr/bin/env python3
"""Build scale-transfer figures and summaries from completed trials only.

The script never fabricates placeholder rows or interpolates missing experiment
phases. Each requested artifact validates its required phase and grid before
writing output:

* profiles: profile_pilot
* early-late: profile_confirmation
* vit: vit_confirmation
* lr: lr_proxy and target_oracle
* summary: profile_confirmation, lr_proxy, target_oracle, and width_transfer

The zero-shot table uses validation loss to select learning rates and calculate
transfer regret. Its test loss and accuracy columns are the single, locked test
evaluations from each transferred profile. The prespecified compute endpoint is
the restricted mean updates, examples, and estimated dense FLOPs needed to
reach each seed's paired uniform profile final validation accuracy. A parallel
time-to-terminal-validation-loss calculation is retained as a separate
diagnostic. Non-reachers are censored at the full training horizon and exposed
through the corresponding reach rate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dropout_mft.experiments.scale_transfer.analysis import (
    bootstrap_mean_ci,
    exact_paired_permutation_pvalue,
    holm_adjust,
    learning_rate_grid_distance,
    mup_transfer_claim_gate,
    profile_rank_stability,
    profile_transfer_claim_gate,
)
from dropout_mft.paths import project_root
from dropout_mft.plotting import save_figure
from dropout_mft.results import load_npz_result
from dropout_mft.style import COLORS, apply_paper_style

REPO_ROOT = project_root()
SCHEMA_VERSION = 2
DEFAULT_RUN_DIR = REPO_ROOT / "results" / "scale_transfer"
DEFAULT_FIGURE_DIR = REPO_ROOT / "figures" / "paper" / "experiments" / "scale_transfer"

PROFILE_GAMMA = {
    "uniform": 0.0,
    "linear_early": 1.0,
    "linear_late": 1.0,
    "quadratic_early": 2.0,
    "quadratic_late": 2.0,
    "quartic_early": 4.0,
    "quartic_late": 4.0,
}
ORIENTATION_MARKERS = {"early": "o", "late": "s", "common": "D"}
PARAMETERIZATION_NAMES = {"sp": "SP", "mup": r"$\mu$P"}
EXPECTED_TRANSFER_WIDTHS = (256, 512, 1024, 2048)
TRANSFER_PROFILE_IDS = (
    "uniform",
    "quadratic_early",
    "quadratic_late",
    "step_early",
    "step_late",
)
LATEX_PROFILE_IDS = ("uniform", "quadratic_early", "step_early")
PRESPECIFIED_EARLY_PROFILE_IDS = ("quadratic_early", "step_early")
EXPECTED_LR_GRID = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3)
EXPECTED_TUNING_SEEDS = frozenset(range(3))
EXPECTED_TUNING_EXTENSION_SEEDS = frozenset(range(3, 6))
EXPECTED_TRANSFER_SEEDS = frozenset(range(100, 110))
EXPECTED_PROFILE_BUDGETS = (0.10,)
EXPECTED_PROFILE_PILOT_IDS = (*PROFILE_GAMMA, "step_early", "step_late")
EXPECTED_LR_WIDTHS = (128, 256, 512, 1024, 2048)
FINAL_WINDOW_FRACTION = 0.10
PROFILE_HARM_MARGIN = 0.01


class ResultsError(RuntimeError):
    """Raised when completed results cannot support a requested artifact."""


def _as_finite_array(value, *, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ResultsError(f"{label} must be nonempty and finite")
    return array


def _validate_completed_trial(trial: dict, *, source: str) -> None:
    if trial.get("schema_version") != SCHEMA_VERSION:
        raise ResultsError(
            f"{source}: unsupported schema_version={trial.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    trial_meta = trial.get("trial", {})
    if trial_meta.get("status") != "complete":
        raise ResultsError(f"{source}: trial is not complete")
    required_meta = ("trial_id", "config_hash", "phase", "seed")
    required_factors = (
        "model_kind",
        "parameterization",
        "width",
        "depth",
        "profile_id",
        "learning_rate",
        "activation",
        "mean_dropout",
        "budget_space",
    )
    for key in required_meta:
        if key not in trial_meta:
            raise ResultsError(f"{source}: missing trial.{key}")
    for key in required_factors:
        if key not in trial.get("factors", {}):
            raise ResultsError(f"{source}: missing factors.{key}")
    for key in ("validation_loss", "validation_accuracy"):
        _as_finite_array(
            trial.get("curves", {}).get(key), label=f"{source}: curves.{key}"
        )
    test = trial.get("test", {})
    evaluated = test.get("evaluated")
    if not isinstance(evaluated, bool):
        raise ResultsError(f"{source}: test.evaluated must be a boolean")
    if evaluated:
        for key in ("loss", "accuracy"):
            values = _as_finite_array([test.get(key)], label=f"{source}: test.{key}")
            if key == "accuracy" and not 0.0 <= float(values[0]) <= 1.0:
                raise ResultsError(f"{source}: test.accuracy must lie in [0, 1]")
    elif test.get("loss") is not None or test.get("accuracy") is not None:
        raise ResultsError(
            f"{source}: unevaluated tests must store loss=None and accuracy=None"
        )
    flops = float(trial.get("compute", {}).get("estimated_training_flops", math.nan))
    if not math.isfinite(flops) or flops <= 0:
        raise ResultsError(
            f"{source}: compute.estimated_training_flops must be positive"
        )
    optimizer_steps = trial.get("compute", {}).get("optimizer_steps")
    if (
        isinstance(optimizer_steps, bool)
        or not isinstance(optimizer_steps, (int, np.integer))
        or optimizer_steps <= 0
    ):
        raise ResultsError(
            f"{source}: compute.optimizer_steps must be a positive integer"
        )
    examples_seen = trial.get("compute", {}).get("examples_seen")
    if (
        isinstance(examples_seen, bool)
        or not isinstance(examples_seen, (int, np.integer))
        or examples_seen <= 0
    ):
        raise ResultsError(
            f"{source}: compute.examples_seen must be a positive integer"
        )


def load_completed_trials(run_dir: str | Path, *, source: str = "auto") -> list[dict]:
    """Load one immutable snapshot: aggregate data or individual trial files."""

    run_dir = Path(run_dir)
    aggregate_path = run_dir / "aggregate.npz"
    if source == "auto":
        source = "aggregate" if aggregate_path.exists() else "trials"
    if source == "aggregate":
        if not aggregate_path.exists():
            raise ResultsError(f"Completed aggregate is absent: {aggregate_path}")
        payload = load_npz_result(aggregate_path)
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ResultsError(
                f"{aggregate_path}: unsupported aggregate schema "
                f"{payload.get('schema_version')!r}"
            )
        raw_trials = payload.get("trials")
        if not isinstance(raw_trials, list):
            raise ResultsError(f"{aggregate_path}: missing trials list")
        labelled = [
            (trial, f"{aggregate_path}:trials[{index}]")
            for index, trial in enumerate(raw_trials)
        ]
    elif source == "trials":
        paths = sorted((run_dir / "trials").glob("*/*.npz"))
        if not paths:
            raise ResultsError(f"No trial files found under {run_dir / 'trials'}")
        labelled = [(load_npz_result(path), str(path)) for path in paths]
    else:
        raise ValueError("source must be 'auto', 'aggregate', or 'trials'")

    completed: list[dict] = []
    seen_ids: set[str] = set()
    for trial, label in labelled:
        if trial.get("trial", {}).get("status") != "complete":
            continue
        _validate_completed_trial(trial, source=label)
        trial_id = str(trial["trial"]["trial_id"])
        if trial_id in seen_ids:
            raise ResultsError(f"Duplicate completed trial_id: {trial_id}")
        seen_ids.add(trial_id)
        completed.append(trial)
    if not completed:
        raise ResultsError(
            f"No completed schema-v{SCHEMA_VERSION} trials found in {run_dir}"
        )
    return completed


def require_phase(trials: Sequence[dict], phase: str) -> list[dict]:
    selected = [trial for trial in trials if trial["trial"]["phase"] == phase]
    if not selected:
        raise ResultsError(
            f"Required completed phase '{phase}' is absent. "
            "Run that phase and rebuild aggregate.npz before generating this artifact."
        )
    return selected


def _seed(trial: dict) -> int:
    return int(trial["trial"]["seed"])


def _factor(trial: dict, key: str):
    return trial["factors"][key]


def _final_validation_loss(trial: dict) -> float:
    return float(
        _as_finite_array(trial["curves"]["validation_loss"], label="validation_loss")[
            -1
        ]
    )


def _final_validation_accuracy(trial: dict) -> float:
    value = float(
        _as_finite_array(
            trial["curves"]["validation_accuracy"], label="validation_accuracy"
        )[-1]
    )
    if not 0.0 <= value <= 1.0:
        raise ResultsError("validation_accuracy must lie in [0, 1]")
    return value


def _validation_loss_normalized_auc(trial: dict) -> float:
    """Trapezoidal validation-loss AUC divided by the observed epoch span."""

    values = _as_finite_array(
        trial["curves"]["validation_loss"], label="validation_loss"
    )
    if len(values) == 1:
        return float(values[0])
    return float(np.trapz(values) / (len(values) - 1))


def _validation_loss_final_window_mean(
    trial: dict, *, fraction: float = FINAL_WINDOW_FRACTION
) -> float:
    """Mean validation loss over the final, preregistered 10% epoch window."""

    if not 0 < fraction <= 1:
        raise ValueError("fraction must lie in (0, 1]")
    values = _as_finite_array(
        trial["curves"]["validation_loss"], label="validation_loss"
    )
    window = max(1, int(math.ceil(fraction * len(values))))
    return float(np.mean(values[-window:]))


def _locked_test_metric(trial: dict, key: str, *, label: str) -> float:
    test = trial.get("test", {})
    if test.get("evaluated") is not True:
        raise ResultsError(f"{label}: locked test evaluation is absent")
    value = float(test.get(key, math.nan))
    if not math.isfinite(value):
        raise ResultsError(f"{label}: test.{key} must be finite")
    if key == "accuracy" and not 0.0 <= value <= 1.0:
        raise ResultsError(f"{label}: test.accuracy must lie in [0, 1]")
    return value


def _sem(values: Sequence[float]) -> float:
    values = np.asarray(values, dtype=float)
    return (
        float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    )


def _orientation(profile_id: str) -> str:
    if profile_id.endswith("_early"):
        return "early"
    if profile_id.endswith("_late"):
        return "late"
    return "common"


def _family(profile_id: str) -> str:
    if profile_id.endswith("_early"):
        return profile_id[: -len("_early")]
    if profile_id.endswith("_late"):
        return profile_id[: -len("_late")]
    return profile_id


def _group(trials: Iterable[dict], keys: Sequence[str]) -> dict[tuple, list[dict]]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for trial in trials:
        grouped[
            tuple(
                trial["trial"][key.split(".", 1)[1]]
                if key.startswith("trial.")
                else trial["factors"][key]
                for key in keys
            )
        ].append(trial)
    return dict(grouped)


def _assert_unique_seeds(records: Sequence[dict], *, label: str) -> set[int]:
    seeds = [_seed(record) for record in records]
    if len(seeds) != len(set(seeds)):
        raise ResultsError(f"{label}: duplicate seeds")
    return set(seeds)


def _assert_balanced_lr_grid(
    records: Sequence[dict],
    *,
    label: str,
    expected_seeds: frozenset[int] = EXPECTED_TUNING_SEEDS,
) -> tuple[float, ...]:
    by_lr = _group(records, ("learning_rate",))
    actual_lrs = {float(key[0]) for key in by_lr}
    if actual_lrs != set(EXPECTED_LR_GRID):
        raise ResultsError(
            f"{label}: completed LR grid {sorted(actual_lrs)} does not match the "
            f"preregistered grid {list(EXPECTED_LR_GRID)}"
        )
    seed_sets = {
        float(key[0]): _assert_unique_seeds(group, label=f"{label}, lr={key[0]:g}")
        for key, group in by_lr.items()
    }
    if any(seeds != expected_seeds for seeds in seed_sets.values()):
        raise ResultsError(
            f"{label}: every learning rate requires completed seeds "
            f"{sorted(expected_seeds)}"
        )
    return EXPECTED_LR_GRID


def _select_validation_lr(
    records: Sequence[dict],
    *,
    label: str,
    expected_seeds: frozenset[int] = EXPECTED_TUNING_SEEDS,
) -> tuple[float, float]:
    _assert_balanced_lr_grid(records, label=label, expected_seeds=expected_seeds)
    candidates = []
    for (learning_rate,), group in _group(records, ("learning_rate",)).items():
        losses = [_final_validation_loss(trial) for trial in group]
        candidates.append((float(np.mean(losses)), float(learning_rate)))
    mean_loss, learning_rate = min(candidates)
    return learning_rate, mean_loss


def make_profile_loss_figure(trials: Sequence[dict], output_dir: str | Path) -> Path:
    records = require_phase(trials, "profile_pilot")
    records = [
        trial
        for trial in records
        if _factor(trial, "model_kind") == "mlp"
        and _factor(trial, "parameterization") == "sp"
        and _factor(trial, "activation") == "relu"
    ]
    if not records:
        raise ResultsError("profile_pilot has no completed ReLU/SP MLP records")

    grouped = _group(records, ("mean_dropout", "profile_id"))
    expected_groups = {
        (budget, profile_id)
        for budget in EXPECTED_PROFILE_BUDGETS
        for profile_id in EXPECTED_PROFILE_PILOT_IDS
    } | {(0.0, "none")}
    actual_groups = {(float(budget), str(profile_id)) for budget, profile_id in grouped}
    missing_groups = sorted(expected_groups - actual_groups)
    unexpected_groups = sorted(actual_groups - expected_groups)
    if missing_groups or unexpected_groups:
        raise ResultsError(
            "profile_pilot budget/profile grid mismatch; missing="
            + ", ".join(
                f"p={budget:g}/{profile_id}" for budget, profile_id in missing_groups
            )
            + "; unexpected="
            + ", ".join(
                f"p={budget:g}/{profile_id}" for budget, profile_id in unexpected_groups
            )
        )
    for (budget, profile_id), group in grouped.items():
        if (
            _assert_unique_seeds(
                group, label=f"profile_pilot, budget={budget:g}, profile={profile_id}"
            )
            != EXPECTED_TUNING_SEEDS
        ):
            raise ResultsError(
                f"profile_pilot, budget={budget:g}, profile={profile_id}: requires "
                f"completed seeds {sorted(EXPECTED_TUNING_SEEDS)}"
            )
    no_dropout_records = grouped[(0.0, "none")]
    no_dropout_values = [_final_validation_loss(trial) for trial in no_dropout_records]
    no_dropout_mean = float(np.mean(no_dropout_values))
    no_dropout_sem = _sem(no_dropout_values)
    records = [trial for trial in records if _factor(trial, "profile_id") != "none"]
    grouped = _group(records, ("mean_dropout", "profile_id"))
    budgets = sorted({float(_factor(trial, "mean_dropout")) for trial in records})
    palette = [COLORS["muted_blue"], COLORS["rose"], COLORS["teal"]]
    colors = {
        budget: palette[index % len(palette)] for index, budget in enumerate(budgets)
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    axes[0].axhline(
        no_dropout_mean,
        color=COLORS["baseline"],
        linestyle=":",
        linewidth=1.4,
        label="No dropout",
    )

    # Uniform is one observed baseline shared by the two orientation curves.
    for budget in budgets:
        budget_groups = {
            profile_id: group
            for (group_budget, profile_id), group in grouped.items()
            if math.isclose(float(group_budget), budget)
        }
        for orientation, linestyle in (("early", "-"), ("late", "--")):
            points = []
            for profile_id, gamma in PROFILE_GAMMA.items():
                profile_orientation = _orientation(profile_id)
                if profile_orientation not in (orientation, "common"):
                    continue
                group = budget_groups.get(profile_id)
                if not group:
                    raise ResultsError(
                        f"profile_pilot is incomplete: budget={budget:g}, profile={profile_id}"
                    )
                values = [_final_validation_loss(trial) for trial in group]
                points.append((gamma, float(np.mean(values)), _sem(values)))
            points.sort()
            axes[0].errorbar(
                [point[0] for point in points],
                [point[1] for point in points],
                yerr=[point[2] for point in points],
                marker=ORIENTATION_MARKERS[orientation],
                linestyle=linestyle,
                capsize=3,
                color=colors[budget],
                label=rf"$\bar p={budget:g}$, {orientation}",
            )

    # The damage panel also includes saturated steps, which have no finite gamma.
    axes[1].errorbar(
        [0.0],
        [no_dropout_mean],
        yerr=[no_dropout_sem],
        marker="D",
        linestyle="none",
        capsize=3,
        color=COLORS["baseline"],
        label="No dropout",
    )
    for budget in budgets:
        for orientation in ("early", "late", "common"):
            x_values, y_values, y_errors = [], [], []
            for (group_budget, profile_id), group in grouped.items():
                if not math.isclose(float(group_budget), budget):
                    continue
                if _orientation(profile_id) != orientation:
                    continue
                damages = {
                    round(float(trial["schedule"]["field_damage_mean"]), 14)
                    for trial in group
                }
                if len(damages) != 1:
                    raise ResultsError(
                        f"field_damage_mean varies across seeds for budget={budget:g}, "
                        f"profile={profile_id}"
                    )
                values = [_final_validation_loss(trial) for trial in group]
                x_values.append(next(iter(damages)))
                y_values.append(float(np.mean(values)))
                y_errors.append(_sem(values))
            if x_values:
                order = np.argsort(x_values)
                axes[1].errorbar(
                    np.asarray(x_values)[order],
                    np.asarray(y_values)[order],
                    yerr=np.asarray(y_errors)[order],
                    marker=ORIENTATION_MARKERS[orientation],
                    linestyle="none",
                    capsize=3,
                    color=colors[budget],
                    alpha=0.9,
                )

    axes[0].set_title(r"(a) Power-profile interpolation", loc="left")
    axes[0].set_xlabel(r"Power exponent $\gamma$")
    axes[0].set_ylabel("Final validation cross-entropy")
    axes[0].set_xticks([0, 1, 2, 4])
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].set_title(r"(b) Loss versus predicted field damage", loc="left")
    axes[1].set_xlabel(r"$D_a=L^{-1}\sum_\ell h_\ell^a$")
    axes[1].set_ylabel("Final validation cross-entropy")
    axes[1].text(
        0.02,
        0.98,
        "circles: early   squares: late   diamonds: uniform",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color=COLORS["neutral"],
    )
    fig.tight_layout()
    path = Path(output_dir) / "profile_loss_vs_gamma_damage.png"
    save_figure(fig, path)
    plt.close(fig)
    return path.with_suffix(".pdf")


def make_early_late_figure(trials: Sequence[dict], output_dir: str | Path) -> Path:
    records = require_phase(trials, "profile_confirmation")
    by_activation_profile = _group(records, ("activation", "profile_id"))
    expected_pairs = {
        (activation, f"{family}_{orientation}")
        for activation in ("relu", "gelu")
        for family in ("quadratic", "step")
        for orientation in ("early", "late")
    }
    missing_pairs = sorted(expected_pairs - set(by_activation_profile))
    if missing_pairs:
        raise ResultsError(
            "profile_confirmation is missing completed groups: "
            + ", ".join(
                f"{activation}/{profile}" for activation, profile in missing_pairs
            )
        )
    pairs: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    activations = sorted({str(_factor(trial, "activation")) for trial in records})
    families = sorted(
        {
            _family(str(_factor(trial, "profile_id")))
            for trial in records
            if _orientation(str(_factor(trial, "profile_id"))) in ("early", "late")
        }
    )
    for activation in activations:
        for family in families:
            early_id, late_id = f"{family}_early", f"{family}_late"
            early = by_activation_profile.get((activation, early_id))
            late = by_activation_profile.get((activation, late_id))
            if early is None and late is None:
                continue
            if not early or not late:
                raise ResultsError(
                    f"profile_confirmation is incomplete: {activation} {family} "
                    "requires both early and late records"
                )
            early_by_seed = {
                _seed(trial): _locked_test_metric(
                    trial, "loss", label=f"{activation} {early_id}"
                )
                for trial in early
            }
            late_by_seed = {
                _seed(trial): _locked_test_metric(
                    trial, "loss", label=f"{activation} {late_id}"
                )
                for trial in late
            }
            if len(early_by_seed) != len(early) or len(late_by_seed) != len(late):
                raise ResultsError(
                    f"profile_confirmation has duplicate seeds: {activation} {family}"
                )
            if set(early_by_seed) != set(late_by_seed):
                raise ResultsError(
                    f"profile_confirmation seed mismatch: {activation} {family}"
                )
            if set(early_by_seed) != EXPECTED_TRANSFER_SEEDS:
                raise ResultsError(
                    f"profile_confirmation, {activation} {family}: requires completed "
                    f"seeds {sorted(EXPECTED_TRANSFER_SEEDS)}"
                )
            seeds = sorted(early_by_seed)
            pairs.append(
                (
                    activation,
                    family,
                    np.asarray([early_by_seed[seed] for seed in seeds]),
                    np.asarray([late_by_seed[seed] for seed in seeds]),
                )
            )
    if not pairs:
        raise ResultsError(
            "profile_confirmation contains no completed early/late pairs"
        )

    activation_colors = {"relu": COLORS["rose"], "gelu": COLORS["smooth"]}
    family_markers = {"quadratic": "o", "step": "s", "linear": "^", "quartic": "v"}
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
    all_values = []
    for activation, family, early, late in pairs:
        color = activation_colors.get(activation, COLORS["muted_blue"])
        marker = family_markers.get(family, "D")
        axes[0].scatter(
            late,
            early,
            color=color,
            marker=marker,
            alpha=0.78,
            label=(
                f"{activation.upper()} {family} "
                f"({'primary' if family == 'quadratic' else 'confirmatory'})"
            ),
        )
        all_values.extend(early.tolist())
        all_values.extend(late.tolist())
    lo, hi = min(all_values), max(all_values)
    padding = max(0.01, 0.06 * (hi - lo))
    axes[0].plot(
        [lo - padding, hi + padding],
        [lo - padding, hi + padding],
        "--",
        color="#777777",
    )
    axes[0].set_xlim(lo - padding, hi + padding)
    axes[0].set_ylim(lo - padding, hi + padding)
    axes[0].set_xlabel("Late-profile test loss")
    axes[0].set_ylabel("Early-profile test loss")
    axes[0].set_title("(a) Paired seeds", loc="left")
    axes[0].legend(fontsize=8)

    labels, means, lower_errors, upper_errors, colors = [], [], [], [], []
    for activation, family, early, late in pairs:
        interval = bootstrap_mean_ci(early - late, seed=0)
        labels.append(f"{activation.upper()}\n{family}")
        means.append(interval["mean"])
        lower_errors.append(interval["mean"] - interval["lower"])
        upper_errors.append(interval["upper"] - interval["mean"])
        colors.append(activation_colors.get(activation, COLORS["muted_blue"]))
    positions = np.arange(len(labels))
    axes[1].errorbar(
        positions,
        means,
        yerr=np.vstack([lower_errors, upper_errors]),
        fmt="none",
        ecolor=COLORS["baseline"],
        capsize=4,
        linewidth=1.5,
    )
    axes[1].scatter(positions, means, c=colors, s=48, zorder=3)
    axes[1].axhline(0.0, linestyle="--", color="#777777", linewidth=1.2)
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylabel(r"Paired loss difference (early $-$ late)")
    axes[1].set_title("(b) Prespecified exact reversals", loc="left")
    fig.tight_layout()
    path = Path(output_dir) / "early_vs_late_profiles.png"
    save_figure(fig, path)
    plt.close(fig)
    return path.with_suffix(".pdf")


def make_vit_confirmation_figure(
    trials: Sequence[dict], output_dir: str | Path
) -> Path:
    """Plot the residual/ViT bridge relative to the locked uniform baseline."""

    records = require_phase(trials, "vit_confirmation")
    expected_profiles = (
        "uniform",
        "quadratic_early",
        "quadratic_late",
        "step_early",
        "step_late",
    )
    expected_seeds = set(range(200, 206))
    groups = {
        str(profile_id): group
        for (profile_id,), group in _group(records, ("profile_id",)).items()
    }
    if set(groups) != set(expected_profiles):
        raise ResultsError(
            "vit_confirmation profile grid mismatch: "
            f"expected={list(expected_profiles)}, observed={sorted(groups)}"
        )
    by_profile_seed = {}
    for profile_id, group in groups.items():
        seeds = _assert_unique_seeds(group, label=f"ViT {profile_id}")
        if seeds != expected_seeds:
            raise ResultsError(
                f"ViT {profile_id} requires completed seeds {sorted(expected_seeds)}"
            )
        by_profile_seed[profile_id] = {
            _seed(trial): (
                _locked_test_metric(trial, "loss", label=f"ViT {profile_id}"),
                _locked_test_metric(trial, "accuracy", label=f"ViT {profile_id}"),
            )
            for trial in group
        }

    uniform = by_profile_seed["uniform"]
    labels = [profile.replace("_", " ") for profile in expected_profiles]
    loss_means, loss_low, loss_high = [], [], []
    accuracy_means, accuracy_low, accuracy_high = [], [], []
    for profile_id in expected_profiles:
        loss_difference = [
            by_profile_seed[profile_id][seed][0] - uniform[seed][0]
            for seed in sorted(expected_seeds)
        ]
        accuracy_difference = [
            100.0 * (by_profile_seed[profile_id][seed][1] - uniform[seed][1])
            for seed in sorted(expected_seeds)
        ]
        loss_interval = bootstrap_mean_ci(loss_difference, seed=0)
        accuracy_interval = bootstrap_mean_ci(accuracy_difference, seed=0)
        loss_means.append(loss_interval["mean"])
        loss_low.append(loss_interval["mean"] - loss_interval["lower"])
        loss_high.append(loss_interval["upper"] - loss_interval["mean"])
        accuracy_means.append(accuracy_interval["mean"])
        accuracy_low.append(accuracy_interval["mean"] - accuracy_interval["lower"])
        accuracy_high.append(accuracy_interval["upper"] - accuracy_interval["mean"])

    colors = [
        COLORS["neutral"],
        COLORS["teal"],
        COLORS["rose"],
        COLORS["muted_blue"],
        COLORS["dark_gold"],
    ]
    x = np.arange(len(expected_profiles))
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    axes[0].bar(x, loss_means, color=colors, alpha=0.9)
    axes[0].errorbar(
        x,
        loss_means,
        yerr=np.asarray([loss_low, loss_high]),
        fmt="none",
        color="black",
        capsize=3,
    )
    axes[0].axhline(0.0, color="#777777", linewidth=0.9)
    axes[0].set_ylabel(r"Paired test-loss change vs. uniform")
    axes[0].set_title("(a) Residual/ViT bridge loss", loc="left")
    axes[1].bar(x, accuracy_means, color=colors, alpha=0.9)
    axes[1].errorbar(
        x,
        accuracy_means,
        yerr=np.asarray([accuracy_low, accuracy_high]),
        fmt="none",
        color="black",
        capsize=3,
    )
    axes[1].axhline(0.0, color="#777777", linewidth=0.9)
    axes[1].set_ylabel("Paired test-accuracy change (percentage points)")
    axes[1].set_title("(b) Residual/ViT bridge accuracy", loc="left")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=20, ha="right")
    fig.tight_layout()
    path = Path(output_dir) / "vit_profile_confirmation.png"
    save_figure(fig, path)
    plt.close(fig)
    return path.with_suffix(".pdf")


def _proxy_lr_records(
    trials: Sequence[dict],
) -> tuple[list[dict], frozenset[int]]:
    """Combine the base proxy grid with its conditional stability extension."""

    proxy = [
        trial
        for trial in require_phase(trials, "lr_proxy")
        if _factor(trial, "profile_id") == "uniform"
    ]
    extension = [
        trial
        for trial in trials
        if trial["trial"]["phase"] == "lr_proxy_extension"
        and _factor(trial, "profile_id") == "uniform"
    ]
    expected_seeds = EXPECTED_TUNING_SEEDS
    if extension:
        expected_seeds = EXPECTED_TUNING_SEEDS | EXPECTED_TUNING_EXTENSION_SEEDS
    return [*proxy, *extension], expected_seeds


def _lr_curve_records(trials: Sequence[dict]) -> dict[tuple[str, int], list[dict]]:
    proxy, proxy_seeds = _proxy_lr_records(trials)
    oracle = require_phase(trials, "target_oracle")
    records = [
        trial
        for trial in [*proxy, *oracle]
        if _factor(trial, "profile_id") == "uniform"
    ]
    grouped = _group(records, ("parameterization", "width"))
    for (parameterization, width), group in grouped.items():
        expected_seeds = proxy_seeds if int(width) == 128 else EXPECTED_TUNING_SEEDS
        _assert_balanced_lr_grid(
            group,
            label=f"{parameterization}, width={int(width)}",
            expected_seeds=expected_seeds,
        )
    if {key[0] for key in grouped} != {"sp", "mup"}:
        raise ResultsError("LR curves require completed SP and muP grids")
    expected_groups = {
        (parameterization, width)
        for parameterization in ("sp", "mup")
        for width in EXPECTED_LR_WIDTHS
    }
    missing_groups = expected_groups - {
        (str(parameterization), int(width)) for parameterization, width in grouped
    }
    if missing_groups:
        raise ResultsError(
            "LR curves are missing completed parameterization/width grids: "
            + ", ".join(
                f"{parameterization}/N={width}"
                for parameterization, width in sorted(missing_groups)
            )
        )
    return grouped


def make_lr_curve_figure(trials: Sequence[dict], output_dir: str | Path) -> Path:
    grouped = _lr_curve_records(trials)
    widths = sorted({int(key[1]) for key in grouped})
    width_colors = dict(
        zip(widths, plt.cm.viridis(np.linspace(0.12, 0.88, len(widths))))
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharey=True)
    for axis, parameterization in zip(axes, ("sp", "mup")):
        parameterization_groups = {
            int(width): group
            for (kind, width), group in grouped.items()
            if kind == parameterization
        }
        if len(parameterization_groups) < 2:
            raise ResultsError(
                f"{parameterization}: LR stability plot requires proxy and target widths"
            )
        for width, records in sorted(parameterization_groups.items()):
            points = []
            for (learning_rate,), lr_records in _group(
                records, ("learning_rate",)
            ).items():
                values = [_final_validation_loss(trial) for trial in lr_records]
                points.append(
                    (float(learning_rate), float(np.mean(values)), _sem(values))
                )
            points.sort()
            axis.errorbar(
                [point[0] for point in points],
                [point[1] for point in points],
                yerr=[point[2] for point in points],
                marker="o",
                capsize=3,
                color=width_colors[width],
                label=rf"$N={width}$",
            )
            expected_seeds = (
                EXPECTED_TUNING_SEEDS | EXPECTED_TUNING_EXTENSION_SEEDS
                if any(
                    trial["trial"]["phase"] == "lr_proxy_extension" for trial in records
                )
                else EXPECTED_TUNING_SEEDS
            )
            selected_lr, selected_loss = _select_validation_lr(
                records,
                label=f"{parameterization}, width={width}",
                expected_seeds=expected_seeds,
            )
            axis.scatter(
                [selected_lr],
                [selected_loss],
                marker="*",
                s=95,
                color=width_colors[width],
                edgecolor="black",
                linewidth=0.5,
                zorder=4,
            )
        axis.set_xscale("log")
        axis.set_xlabel("Base learning rate")
        axis.set_title(
            f"({'a' if parameterization == 'sp' else 'b'}) "
            f"{PARAMETERIZATION_NAMES[parameterization]}",
            loc="left",
        )
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Final validation cross-entropy")
    fig.tight_layout()
    path = Path(output_dir) / "sp_vs_mup_lr_curves.png"
    save_figure(fig, path)
    plt.close(fig)
    return path.with_suffix(".pdf")


def _crossing_resources(
    trial: dict,
    threshold: float,
    *,
    curve_key: str = "validation_loss",
    direction: str = "min",
) -> tuple[float, float, float, bool]:
    """Return censored updates/examples/estimated FLOPs to a curve threshold."""

    if curve_key not in {"validation_loss", "validation_accuracy"}:
        raise ValueError("curve_key must be validation_loss or validation_accuracy")
    if direction not in {"min", "max"}:
        raise ValueError("direction must be min or max")
    curve = _as_finite_array(trial["curves"][curve_key], label=curve_key)
    step_curve = _as_finite_array(
        trial["curves"].get("optimizer_steps"), label="curves.optimizer_steps"
    )
    if curve.ndim != 1 or step_curve.ndim != 1 or len(step_curve) != len(curve):
        raise ResultsError(
            f"{curve_key} and curves.optimizer_steps must be aligned vectors"
        )
    if np.any(step_curve <= 0) or np.any(np.diff(step_curve) <= 0):
        raise ResultsError(
            "curves.optimizer_steps must be positive and strictly increasing"
        )
    total_steps = int(trial["compute"]["optimizer_steps"])
    total_examples = int(trial["compute"]["examples_seen"])
    total_flops = float(trial["compute"]["estimated_training_flops"])
    if not math.isclose(float(step_curve[-1]), total_steps):
        raise ResultsError(
            "final curves.optimizer_steps must equal compute.optimizer_steps"
        )
    if total_examples <= 0:
        raise ResultsError("compute.examples_seen must be positive")
    if not math.isfinite(total_flops) or total_flops <= 0:
        raise ResultsError("compute.estimated_training_flops must be positive")
    if curve_key == "validation_accuracy" and np.any((curve < 0.0) | (curve > 1.0)):
        raise ResultsError("validation_accuracy must lie in [0, 1]")
    matches = np.flatnonzero(
        curve <= threshold if direction == "min" else curve >= threshold
    )
    if not len(matches):
        return float(total_steps), float(total_examples), total_flops, False
    crossing_steps = float(step_curve[int(matches[0])])
    resource_fraction = crossing_steps / total_steps
    crossing_examples = total_examples * crossing_steps / total_steps
    crossing_flops = total_flops * resource_fraction
    return crossing_steps, float(crossing_examples), float(crossing_flops), True


def zero_shot_rows(trials: Sequence[dict]) -> list[dict]:
    proxy, proxy_seeds = _proxy_lr_records(trials)
    oracle = [
        trial
        for trial in require_phase(trials, "target_oracle")
        if _factor(trial, "profile_id") == "uniform"
    ]
    transfer = require_phase(trials, "width_transfer")

    proxy_groups = _group(proxy, ("parameterization", "width"))
    selected_proxy: dict[str, tuple[int, float, float]] = {}
    for parameterization in ("sp", "mup"):
        candidates = [
            (int(width), group)
            for (kind, width), group in proxy_groups.items()
            if kind == parameterization
        ]
        if len(candidates) != 1 or candidates[0][0] != 128:
            raise ResultsError(
                f"lr_proxy must contain exactly the uniform N=128 grid for "
                f"{parameterization}"
            )
        width, group = candidates[0]
        lr, loss = _select_validation_lr(
            group,
            label=f"{parameterization} proxy width={width}",
            expected_seeds=proxy_seeds,
        )
        selected_proxy[parameterization] = (width, lr, loss)

    oracle_groups = {
        (str(parameterization), int(width)): group
        for (parameterization, width), group in _group(
            oracle, ("parameterization", "width")
        ).items()
    }
    expected_oracles = {
        (parameterization, width)
        for parameterization in ("sp", "mup")
        for width in EXPECTED_TRANSFER_WIDTHS
    }
    missing_oracles = sorted(expected_oracles - set(oracle_groups))
    unexpected_oracles = sorted(set(oracle_groups) - expected_oracles)
    if missing_oracles or unexpected_oracles:
        raise ResultsError(
            "target_oracle grid mismatch; missing="
            + ", ".join(f"{kind}/N={width}" for kind, width in missing_oracles)
            + "; unexpected="
            + ", ".join(f"{kind}/N={width}" for kind, width in unexpected_oracles)
        )
    oracle_diagnostics: dict[tuple[str, int], dict] = {}
    for key, oracle_records in oracle_groups.items():
        parameterization, width = key
        target_lr, oracle_validation_loss = _select_validation_lr(
            oracle_records, label=f"{parameterization} target oracle width={width}"
        )
        proxy_lr = selected_proxy[parameterization][1]
        proxy_lr_oracle_records = [
            trial
            for trial in oracle_records
            if math.isclose(float(_factor(trial, "learning_rate")), proxy_lr)
        ]
        if not proxy_lr_oracle_records:
            raise ResultsError(
                f"{parameterization}, width={width}: target_oracle does not contain "
                f"the proxy-selected LR {proxy_lr:g}"
            )
        proxy_lr_target_loss = float(
            np.mean(
                [_final_validation_loss(trial) for trial in proxy_lr_oracle_records]
            )
        )
        best_oracle_records = [
            trial
            for trial in oracle_records
            if math.isclose(float(_factor(trial, "learning_rate")), target_lr)
        ]
        oracle_diagnostics[key] = {
            "target_lr": target_lr,
            "proxy_lr_target_loss": proxy_lr_target_loss,
            "validation_loss": oracle_validation_loss,
            "n_seeds": len(best_oracle_records),
        }

    transfer_groups = {
        (str(parameterization), int(width), str(profile_id)): group
        for (parameterization, width, profile_id), group in _group(
            transfer, ("parameterization", "width", "profile_id")
        ).items()
    }
    expected_transfer = {
        (parameterization, width, profile_id)
        for parameterization in ("sp", "mup")
        for width in EXPECTED_TRANSFER_WIDTHS
        for profile_id in TRANSFER_PROFILE_IDS
    }
    missing_transfer = sorted(expected_transfer - set(transfer_groups))
    unexpected_transfer = sorted(set(transfer_groups) - expected_transfer)
    if missing_transfer or unexpected_transfer:
        preview = ", ".join(
            f"{kind}/N={width}/{profile}"
            for kind, width, profile in missing_transfer[:8]
        )
        suffix = " ..." if len(missing_transfer) > 8 else ""
        raise ResultsError(
            f"width_transfer grid mismatch: missing {len(missing_transfer)} "
            f"groups ({preview}{suffix}); unexpected {len(unexpected_transfer)}"
        )

    rows = []
    profile_order = {
        profile: index for index, profile in enumerate(TRANSFER_PROFILE_IDS)
    }
    ordered_groups = sorted(
        transfer_groups.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            profile_order.get(item[0][2], len(profile_order)),
            item[0][2],
        ),
    )
    for (parameterization, width, profile_id), transferred_records in ordered_groups:
        if parameterization not in selected_proxy:
            raise ResultsError(f"No proxy selection for {parameterization}")
        proxy_width, proxy_lr, proxy_validation_loss = selected_proxy[parameterization]
        uniform_records = transfer_groups.get((parameterization, width, "uniform"))
        if not uniform_records:
            raise ResultsError(
                f"No uniform width-transfer baseline for {parameterization}, N={width}"
            )
        uniform_seed_set = _assert_unique_seeds(
            uniform_records,
            label=f"{parameterization}, width={width}, uniform width_transfer",
        )
        candidate_seed_set = _assert_unique_seeds(
            transferred_records,
            label=f"{parameterization}, width={width}, {profile_id} width_transfer",
        )
        if uniform_seed_set != EXPECTED_TRANSFER_SEEDS:
            raise ResultsError(
                f"{parameterization}, width={width}: every width-transfer profile "
                f"requires completed seeds {sorted(EXPECTED_TRANSFER_SEEDS)}"
            )
        if candidate_seed_set != uniform_seed_set:
            raise ResultsError(
                f"{parameterization}, width={width}, {profile_id}: completed seeds do "
                "not match the uniform baseline"
            )
        transferred_lrs = {
            float(_factor(trial, "learning_rate")) for trial in transferred_records
        }
        if transferred_lrs != {proxy_lr}:
            raise ResultsError(
                f"{parameterization}, width={width}, {profile_id}: width_transfer "
                f"LR(s) {sorted(transferred_lrs)} do not match proxy-selected LR "
                f"{proxy_lr:g}"
            )

        uniform_by_seed = {_seed(trial): trial for trial in uniform_records}
        transferred_by_seed = {_seed(trial): trial for trial in transferred_records}
        ordered_seeds = sorted(uniform_seed_set)
        uniform_terminal_values = [
            _final_validation_loss(uniform_by_seed[seed]) for seed in ordered_seeds
        ]
        transferred_terminal_values = [
            _final_validation_loss(transferred_by_seed[seed]) for seed in ordered_seeds
        ]
        uniform_final_accuracy_values = [
            _final_validation_accuracy(uniform_by_seed[seed]) for seed in ordered_seeds
        ]
        transferred_final_accuracy_values = [
            _final_validation_accuracy(transferred_by_seed[seed])
            for seed in ordered_seeds
        ]
        uniform_final_window_values = [
            _validation_loss_final_window_mean(uniform_by_seed[seed])
            for seed in ordered_seeds
        ]
        transferred_final_window_values = [
            _validation_loss_final_window_mean(transferred_by_seed[seed])
            for seed in ordered_seeds
        ]
        uniform_normalized_auc_values = [
            _validation_loss_normalized_auc(uniform_by_seed[seed])
            for seed in ordered_seeds
        ]
        transferred_normalized_auc_values = [
            _validation_loss_normalized_auc(transferred_by_seed[seed])
            for seed in ordered_seeds
        ]
        uniform_validation_loss = float(np.mean(uniform_terminal_values))
        transferred_validation_loss = float(np.mean(transferred_terminal_values))
        uniform_final_window_loss = float(np.mean(uniform_final_window_values))
        transferred_final_window_loss = float(np.mean(transferred_final_window_values))
        uniform_normalized_auc = float(np.mean(uniform_normalized_auc_values))
        transferred_normalized_auc = float(np.mean(transferred_normalized_auc_values))
        uniform_validation_accuracy = float(np.mean(uniform_final_accuracy_values))
        transferred_validation_accuracy = float(
            np.mean(transferred_final_accuracy_values)
        )
        uniform_full_update_values = tuple(
            float(uniform_by_seed[seed]["compute"]["optimizer_steps"])
            for seed in ordered_seeds
        )
        uniform_full_example_values = tuple(
            float(uniform_by_seed[seed]["compute"]["examples_seen"])
            for seed in ordered_seeds
        )
        uniform_full_flop_values = tuple(
            float(uniform_by_seed[seed]["compute"]["estimated_training_flops"])
            for seed in ordered_seeds
        )
        if profile_id == "uniform":
            # The fixed uniform reference pays its complete training cost.
            accuracy_update_values = uniform_full_update_values
            accuracy_example_values = uniform_full_example_values
            accuracy_flop_values = uniform_full_flop_values
            accuracy_reached = tuple(True for _ in ordered_seeds)
            loss_update_values = uniform_full_update_values
            loss_example_values = uniform_full_example_values
            loss_flop_values = uniform_full_flop_values
            loss_reached = tuple(True for _ in ordered_seeds)
        else:
            (
                accuracy_update_values,
                accuracy_example_values,
                accuracy_flop_values,
                accuracy_reached,
            ) = zip(
                *[
                    _crossing_resources(
                        transferred_by_seed[seed],
                        _final_validation_accuracy(uniform_by_seed[seed]),
                        curve_key="validation_accuracy",
                        direction="max",
                    )
                    for seed in ordered_seeds
                ]
            )
            (
                loss_update_values,
                loss_example_values,
                loss_flop_values,
                loss_reached,
            ) = zip(
                *[
                    _crossing_resources(
                        transferred_by_seed[seed],
                        _final_validation_loss(uniform_by_seed[seed]),
                        curve_key="validation_loss",
                        direction="min",
                    )
                    for seed in ordered_seeds
                ]
            )
        accuracy_reach_rate = float(np.mean(accuracy_reached))
        loss_reach_rate = float(np.mean(loss_reached))
        mean_uniform_full_flops = float(np.mean(uniform_full_flop_values))
        accuracy_flop_fraction = float(
            np.mean(accuracy_flop_values) / mean_uniform_full_flops
        )
        test_losses = [
            _locked_test_metric(
                trial,
                "loss",
                label=(
                    f"{parameterization}, width={width}, {profile_id} width_transfer"
                ),
            )
            for trial in transferred_records
        ]
        test_accuracies = [
            100.0
            * _locked_test_metric(
                trial,
                "accuracy",
                label=(
                    f"{parameterization}, width={width}, {profile_id} width_transfer"
                ),
            )
            for trial in transferred_records
        ]
        uniform_test_by_seed = {
            _seed(trial): _locked_test_metric(
                trial,
                "loss",
                label=f"{parameterization}, width={width}, uniform width_transfer",
            )
            for trial in uniform_records
        }
        candidate_test_by_seed = {
            _seed(trial): _locked_test_metric(
                trial,
                "loss",
                label=(
                    f"{parameterization}, width={width}, {profile_id} width_transfer"
                ),
            )
            for trial in transferred_records
        }
        paired_test_differences = np.asarray(
            [
                candidate_test_by_seed[seed] - uniform_test_by_seed[seed]
                for seed in sorted(uniform_seed_set)
            ]
        )
        paired_interval = bootstrap_mean_ci(paired_test_differences, seed=0)
        paired_p_value = exact_paired_permutation_pvalue(paired_test_differences)
        terminal_validation_differences = np.asarray(
            transferred_terminal_values
        ) - np.asarray(uniform_terminal_values)
        final_window_validation_differences = np.asarray(
            transferred_final_window_values
        ) - np.asarray(uniform_final_window_values)
        normalized_auc_differences = np.asarray(
            transferred_normalized_auc_values
        ) - np.asarray(uniform_normalized_auc_values)

        # The target oracle is uniform-only. Therefore LR drift and regret are
        # intentionally populated only on uniform rows at the oracle widths.
        diagnostic = (
            oracle_diagnostics.get((parameterization, width))
            if profile_id == "uniform"
            else None
        )
        if diagnostic is None:
            target_lr = None
            proxy_lr_target_loss = None
            oracle_validation_loss = None
            lr_drift_ratio = None
            lr_drift_log10 = None
            absolute_regret = None
            relative_regret = None
            n_oracle_seeds = None
        else:
            target_lr = float(diagnostic["target_lr"])
            proxy_lr_target_loss = float(diagnostic["proxy_lr_target_loss"])
            oracle_validation_loss = float(diagnostic["validation_loss"])
            lr_drift_ratio = target_lr / proxy_lr
            lr_drift_log10 = math.log10(lr_drift_ratio)
            absolute_regret = proxy_lr_target_loss - oracle_validation_loss
            relative_regret = absolute_regret / oracle_validation_loss
            n_oracle_seeds = int(diagnostic["n_seeds"])

        rows.append(
            {
                "parameterization": parameterization,
                "proxy_width": proxy_width,
                "target_width": width,
                "profile_id": profile_id,
                "proxy_learning_rate": proxy_lr,
                "target_oracle_learning_rate": target_lr,
                "lr_drift_ratio": lr_drift_ratio,
                "lr_drift_log10": lr_drift_log10,
                "proxy_validation_loss": proxy_validation_loss,
                "validation_loss_terminal_mean": transferred_validation_loss,
                "validation_loss_terminal_sem": _sem(transferred_terminal_values),
                "uniform_validation_loss_terminal_mean": uniform_validation_loss,
                "validation_loss_terminal_delta_vs_uniform": float(
                    np.mean(terminal_validation_differences)
                ),
                "validation_accuracy_terminal_mean": transferred_validation_accuracy,
                "validation_accuracy_terminal_sem": _sem(
                    transferred_final_accuracy_values
                ),
                "uniform_validation_accuracy_terminal_mean": (
                    uniform_validation_accuracy
                ),
                "validation_accuracy_terminal_delta_vs_uniform": float(
                    np.mean(
                        np.asarray(transferred_final_accuracy_values)
                        - np.asarray(uniform_final_accuracy_values)
                    )
                ),
                "validation_loss_final_window_fraction": FINAL_WINDOW_FRACTION,
                "validation_loss_final_window_mean": transferred_final_window_loss,
                "validation_loss_final_window_sem": _sem(
                    transferred_final_window_values
                ),
                "uniform_validation_loss_final_window_mean": (
                    uniform_final_window_loss
                ),
                "validation_loss_final_window_delta_vs_uniform": float(
                    np.mean(final_window_validation_differences)
                ),
                "validation_loss_normalized_auc_mean": transferred_normalized_auc,
                "validation_loss_normalized_auc_sem": _sem(
                    transferred_normalized_auc_values
                ),
                "uniform_validation_loss_normalized_auc_mean": (uniform_normalized_auc),
                "validation_loss_normalized_auc_delta_vs_uniform": float(
                    np.mean(normalized_auc_differences)
                ),
                "proxy_lr_target_validation_loss": proxy_lr_target_loss,
                "target_oracle_validation_loss": oracle_validation_loss,
                "validation_regret_absolute": absolute_regret,
                "validation_regret_relative": relative_regret,
                "test_loss_mean": float(np.mean(test_losses)),
                "test_loss_sem": _sem(test_losses),
                "test_accuracy_pct_mean": float(np.mean(test_accuracies)),
                "test_accuracy_pct_sem": _sem(test_accuracies),
                "paired_test_loss_delta_vs_uniform": paired_interval["mean"],
                "paired_test_loss_ci95_lower": paired_interval["lower"],
                "paired_test_loss_ci95_upper": paired_interval["upper"],
                "paired_permutation_p_value": paired_p_value,
                "holm_p_value_across_widths": None,
                "profile_rank_stability_vs_width256": None,
                "paired_uniform_final_validation_accuracy_mean": (
                    uniform_validation_accuracy
                ),
                "restricted_mean_updates_to_paired_uniform_final_accuracy": float(
                    np.mean(accuracy_update_values)
                ),
                "updates_to_paired_uniform_final_accuracy_sem": _sem(
                    accuracy_update_values
                ),
                "restricted_mean_examples_to_paired_uniform_final_accuracy": float(
                    np.mean(accuracy_example_values)
                ),
                "examples_to_paired_uniform_final_accuracy_sem": _sem(
                    accuracy_example_values
                ),
                "restricted_mean_estimated_flops_to_paired_uniform_final_accuracy": float(
                    np.mean(accuracy_flop_values)
                ),
                "estimated_flops_to_paired_uniform_final_accuracy_sem": _sem(
                    accuracy_flop_values
                ),
                "paired_uniform_final_accuracy_reach_rate": accuracy_reach_rate,
                "estimated_flops_fraction_of_uniform_full_horizon_at_fixed_accuracy": (
                    accuracy_flop_fraction
                ),
                "estimated_flops_savings_fraction_vs_uniform_full_horizon_at_fixed_accuracy": (
                    1.0 - accuracy_flop_fraction
                ),
                # Separate diagnostic: time to the paired uniform terminal loss.
                "paired_uniform_terminal_validation_loss_mean": (
                    uniform_validation_loss
                ),
                "restricted_mean_updates_to_paired_uniform_terminal_loss": float(
                    np.mean(loss_update_values)
                ),
                "updates_to_paired_uniform_terminal_loss_sem": _sem(loss_update_values),
                "restricted_mean_examples_to_paired_uniform_terminal_loss": float(
                    np.mean(loss_example_values)
                ),
                "examples_to_paired_uniform_terminal_loss_sem": _sem(
                    loss_example_values
                ),
                "restricted_mean_estimated_flops_to_paired_uniform_terminal_loss": float(
                    np.mean(loss_flop_values)
                ),
                "estimated_flops_to_paired_uniform_terminal_loss_sem": _sem(
                    loss_flop_values
                ),
                "paired_uniform_terminal_loss_reach_rate": loss_reach_rate,
                "n_transfer_seeds": len(transferred_records),
                "n_oracle_seeds": n_oracle_seeds,
            }
        )
    for parameterization in ("sp", "mup"):
        for profile_id in TRANSFER_PROFILE_IDS:
            indices = [
                index
                for index, row in enumerate(rows)
                if row["parameterization"] == parameterization
                and row["profile_id"] == profile_id
            ]
            if len(indices) != len(EXPECTED_TRANSFER_WIDTHS):
                raise ResultsError(
                    f"Cannot Holm-correct incomplete widths for "
                    f"{parameterization}/{profile_id}"
                )
            adjusted = holm_adjust(
                [rows[index]["paired_permutation_p_value"] for index in indices]
            )
            for index, p_value in zip(indices, adjusted, strict=True):
                rows[index]["holm_p_value_across_widths"] = float(p_value)
    for parameterization in ("sp", "mup"):
        proxy_scores = {
            row["profile_id"]: row["test_loss_mean"]
            for row in rows
            if row["parameterization"] == parameterization
            and row["target_width"] == 256
        }
        for width in EXPECTED_TRANSFER_WIDTHS:
            target_scores = {
                row["profile_id"]: row["test_loss_mean"]
                for row in rows
                if row["parameterization"] == parameterization
                and row["target_width"] == width
            }
            stability = profile_rank_stability(proxy_scores, target_scores)
            for row in rows:
                if (
                    row["parameterization"] == parameterization
                    and row["target_width"] == width
                ):
                    row["profile_rank_stability_vs_width256"] = stability
    return rows


def _paired_width_records(
    trials: Sequence[dict],
    *,
    parameterization: str,
    candidate_profile: str,
    reference_profile: str,
) -> list[dict]:
    """Build seed-blocked test-loss contrasts at every transfer width."""

    transfer = [
        trial
        for trial in require_phase(trials, "width_transfer")
        if _factor(trial, "parameterization") == parameterization
        and _factor(trial, "profile_id") in (candidate_profile, reference_profile)
    ]
    groups = {
        (int(width), str(profile_id)): group
        for (width, profile_id), group in _group(
            transfer, ("width", "profile_id")
        ).items()
    }
    records = []
    for width in EXPECTED_TRANSFER_WIDTHS:
        candidate = groups.get((width, candidate_profile))
        reference = groups.get((width, reference_profile))
        if not candidate or not reference:
            raise ResultsError(
                f"Missing paired contrast at {parameterization}, N={width}: "
                f"{candidate_profile} minus {reference_profile}"
            )
        candidate_seeds = _assert_unique_seeds(
            candidate,
            label=f"{parameterization}, N={width}, {candidate_profile}",
        )
        reference_seeds = _assert_unique_seeds(
            reference,
            label=f"{parameterization}, N={width}, {reference_profile}",
        )
        if candidate_seeds != EXPECTED_TRANSFER_SEEDS:
            raise ResultsError(
                f"{parameterization}, N={width}, {candidate_profile}: requires "
                f"completed seeds {sorted(EXPECTED_TRANSFER_SEEDS)}"
            )
        if reference_seeds != candidate_seeds:
            raise ResultsError(
                f"{parameterization}, N={width}: contrast seed cohorts differ"
            )
        candidate_by_seed = {_seed(trial): trial for trial in candidate}
        reference_by_seed = {_seed(trial): trial for trial in reference}
        seeds = sorted(candidate_seeds)
        differences = [
            _locked_test_metric(
                candidate_by_seed[seed],
                "loss",
                label=f"{parameterization}, N={width}, {candidate_profile}",
            )
            - _locked_test_metric(
                reference_by_seed[seed],
                "loss",
                label=f"{parameterization}, N={width}, {reference_profile}",
            )
            for seed in seeds
        ]
        records.append(
            {
                "width": width,
                "seeds": seeds,
                "paired_differences": differences,
            }
        )
    return records


def _select_best_prespecified_early(
    trials: Sequence[dict],
) -> tuple[str, dict[str, float]]:
    """Select once on the separate-run ReLU-confirmation validation nAUC."""

    confirmation = [
        trial
        for trial in require_phase(trials, "profile_confirmation")
        if _factor(trial, "activation") == "relu"
        and _factor(trial, "parameterization") == "sp"
        and int(_factor(trial, "width")) == 256
        and _factor(trial, "profile_id") in PRESPECIFIED_EARLY_PROFILE_IDS
    ]
    groups = {
        str(profile_id): group
        for (profile_id,), group in _group(confirmation, ("profile_id",)).items()
    }
    if set(groups) != set(PRESPECIFIED_EARLY_PROFILE_IDS):
        raise ResultsError(
            "Secondary profile selection requires both prespecified early profiles "
            "in the separate-run ReLU N=256 confirmation cohort"
        )
    scores = {}
    for profile_id in PRESPECIFIED_EARLY_PROFILE_IDS:
        records = groups[profile_id]
        seeds = _assert_unique_seeds(
            records, label=f"ReLU confirmation selection, {profile_id}"
        )
        if seeds != EXPECTED_TRANSFER_SEEDS:
            raise ResultsError(
                f"ReLU confirmation selection, {profile_id}: requires completed "
                f"seeds {sorted(EXPECTED_TRANSFER_SEEDS)}"
            )
        scores[profile_id] = float(
            np.mean([_validation_loss_normalized_auc(trial) for trial in records])
        )
    if any(not math.isfinite(score) for score in scores.values()):
        raise ResultsError(
            "Cannot select an early profile: validation nAUC is incomplete"
        )
    profile_order = {
        profile: index for index, profile in enumerate(PRESPECIFIED_EARLY_PROFILE_IDS)
    }
    selected = min(
        scores,
        key=lambda profile: (scores[profile], profile_order[profile]),
    )
    return selected, scores


def profile_contrast_gates(trials: Sequence[dict]) -> dict:
    """Evaluate primary, confirmatory, and practical-secondary contrasts."""

    output = {
        "harm_margin_absolute_test_loss": PROFILE_HARM_MARGIN,
        "primary": {
            "label": "quadratic early minus quadratic late",
            "candidate_profile": "quadratic_early",
            "reference_profile": "quadratic_late",
            "by_parameterization": {},
        },
        "confirmatory": {
            "label": "step early minus step late",
            "candidate_profile": "step_early",
            "reference_profile": "step_late",
            "by_parameterization": {},
        },
        "secondary": {
            "label": "validation-selected prespecified early minus uniform",
            "selection_cohort": "ReLU SP N=256 profile_confirmation",
            "selection_endpoint": "mean validation-loss normalized AUC",
            "selection_dependence_note": (
                "Separate training runs, but the confirmation and transfer cohorts "
                "share the fixed validation split and seed identifiers."
            ),
            "candidate_set": list(PRESPECIFIED_EARLY_PROFILE_IDS),
            "reference_profile": "uniform",
            "by_parameterization": {},
        },
    }
    selected, selection_scores = _select_best_prespecified_early(trials)
    output["secondary"].update(
        {
            "selected_profile": selected,
            "selection_scores": selection_scores,
        }
    )
    for parameterization in ("sp", "mup"):
        for role, candidate, reference in (
            ("primary", "quadratic_early", "quadratic_late"),
            ("confirmatory", "step_early", "step_late"),
        ):
            records = _paired_width_records(
                trials,
                parameterization=parameterization,
                candidate_profile=candidate,
                reference_profile=reference,
            )
            output[role]["by_parameterization"][parameterization] = (
                profile_transfer_claim_gate(records, harm_margin=PROFILE_HARM_MARGIN)
            )

        records = _paired_width_records(
            trials,
            parameterization=parameterization,
            candidate_profile=selected,
            reference_profile="uniform",
        )
        secondary_gate = profile_transfer_claim_gate(
            records, harm_margin=PROFILE_HARM_MARGIN
        )
        secondary_gate.update(
            {
                "selected_profile": selected,
            }
        )
        output["secondary"]["by_parameterization"][parameterization] = secondary_gate
    return output


def _contrast_rows(profile_gates: dict) -> list[dict]:
    """Flatten width-specific gate diagnostics for a machine-readable CSV."""

    rows = []
    for role in ("primary", "confirmatory", "secondary"):
        section = profile_gates[role]
        for parameterization, gate in section["by_parameterization"].items():
            candidate = (
                section.get("candidate_profile")
                or section.get("selected_profile")
                or gate["selected_profile"]
            )
            reference = section["reference_profile"]
            for width_effect in gate["width_effects"]:
                rows.append(
                    {
                        "role": role,
                        "parameterization": parameterization,
                        "candidate_profile": candidate,
                        "reference_profile": reference,
                        "harm_margin": gate["harm_margin"],
                        **width_effect,
                        "pooled_paired_effect": gate["pooled_paired_effect"],
                        "pooled_ci95_lower": gate["pooled_ci95_lower"],
                        "pooled_ci95_upper": gate["pooled_ci95_upper"],
                        "pooled_permutation_p_value": gate[
                            "pooled_permutation_p_value"
                        ],
                        "width_interaction_per_doubling": gate[
                            "width_interaction_per_doubling"
                        ],
                        "width_interaction_ci95_lower": gate[
                            "width_interaction_ci95_lower"
                        ],
                        "width_interaction_ci95_upper": gate[
                            "width_interaction_ci95_upper"
                        ],
                    }
                )
    return rows


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    os.replace(temporary, path)


def write_zero_shot_summary(
    trials: Sequence[dict], output_dir: str | Path
) -> tuple[Path, Path, Path, Path]:
    rows = zero_shot_rows(trials)
    profile_gates = profile_contrast_gates(trials)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "zero_shot_transfer_summary.csv"
    contrast_csv_path = output_dir / "profile_transfer_contrasts.csv"
    tex_path = output_dir / "zero_shot_transfer_summary.tex"
    gates_path = output_dir / "claim_gates.json"

    columns = list(rows[0])
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_csv, csv_path)

    contrast_rows = _contrast_rows(profile_gates)
    temporary_contrast_csv = contrast_csv_path.with_suffix(".csv.tmp")
    with temporary_contrast_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(contrast_rows[0]))
        writer.writeheader()
        writer.writerows(contrast_rows)
    os.replace(temporary_contrast_csv, contrast_csv_path)

    oracle_rows = [
        row for row in rows if row["target_oracle_learning_rate"] is not None
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Zero-shot learning-rate transfer for the uniform profile at "
        r"the preregistered oracle widths. "
        r"Learning-rate drift and regret use the same target-oracle validation "
        r"seed cohort; test metrics are "
        r"single locked evaluations.}",
        r"\label{tab:zero_shot_scale_transfer}",
        r"\small",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Param. & $N$ & $\eta_{\rm proxy}$ & $\eta_{\rm oracle}$ & "
        r"$\log_{10}$ drift & Val. regret & Test loss & Test acc. \\",
        r"\midrule",
    ]
    for row in oracle_rows:
        parameterization = "SP" if row["parameterization"] == "sp" else r"$\mu$P"
        lines.append(
            f"{parameterization} & {row['target_width']} & "
            f"{row['proxy_learning_rate']:.1e} & "
            f"{row['target_oracle_learning_rate']:.1e} & "
            f"{row['lr_drift_log10']:+.2f} & "
            f"{100.0 * row['validation_regret_relative']:+.2f}\\% & "
            f"{row['test_loss_mean']:.3f} $\\pm$ {row['test_loss_sem']:.3f} & "
            f"{row['test_accuracy_pct_mean']:.2f} $\\pm$ "
            f"{row['test_accuracy_pct_sem']:.2f}\\% \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Prespecified profile-transfer contrasts. Quadratic exact "
            r"reversal is primary, step exact reversal is confirmatory, and the "
            r"practical secondary compares the early profile selected by validation "
            r"nAUC on the separate-run ReLU $N=256$ confirmation cohort with uniform "
            r"dropout. That cohort shares the fixed split and seed identifiers with "
            r"the transfer study and is not statistically independent. Effects are "
            r"seed-blocked and pooled "
            r"across widths; the interaction is the change per width doubling. "
            r"Noninferiority (NI) uses an absolute test-loss harm margin of 0.01.}",
            r"\label{tab:profile_transfer_contrasts}",
            r"\scriptsize",
            r"\begin{tabular}{lllrrrrr}",
            r"\toprule",
            r"Param. & Role & Contrast & Pooled $\Delta$ test [95\% CI] & "
            r"$p_{\rm pool}$ & $\Delta$/doubling [95\% CI] & NI widths & Gate \\",
            r"\midrule",
        ]
    )
    role_names = {
        "primary": "Primary",
        "confirmatory": "Confirm.",
        "secondary": "Secondary",
    }
    for role in ("primary", "confirmatory", "secondary"):
        section = profile_gates[role]
        for parameterization in ("sp", "mup"):
            gate = section["by_parameterization"][parameterization]
            candidate = section.get("candidate_profile") or gate["selected_profile"]
            reference = section["reference_profile"]
            parameterization_label = "SP" if parameterization == "sp" else r"$\mu$P"
            contrast_label = (
                candidate.replace("_", " ") + " $-$ " + reference.replace("_", " ")
            )
            lines.append(
                f"{parameterization_label} & {role_names[role]} & "
                f"{contrast_label} & {gate['pooled_paired_effect']:+.3f} "
                f"[{gate['pooled_ci95_lower']:+.3f}, "
                f"{gate['pooled_ci95_upper']:+.3f}] & "
                f"{gate['pooled_permutation_p_value']:.3g} & "
                f"{gate['width_interaction_per_doubling']:+.3f} "
                f"[{gate['width_interaction_ci95_lower']:+.3f}, "
                f"{gate['width_interaction_ci95_upper']:+.3f}] & "
                f"{gate['widths_noninferior']}/{gate['expected_widths']} & "
                f"{'pass' if gate['successful'] else 'fail'} \\\\"
            )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Prespecified compute-to-fixed-accuracy endpoint. Each "
            r"candidate seed must match its paired same-width, same-parameterization "
            r"uniform run's final validation accuracy. Non-reachers are censored at "
            r"the full horizon; the fixed uniform reference is charged its complete "
            r"training cost. FLOPs use the stated dense-training estimate rather "
            r"than measured hardware operations.}",
            r"\label{tab:profile_validation_compute}",
            r"\scriptsize",
            r"\begin{tabular}{lllrrrrrr}",
            r"\toprule",
            r"Param. & $N$ & Profile & Uniform target acc. & Updates & Examples & "
            r"Est. TFLOPs & Reach & Est. saving \\",
            r"\midrule",
        ]
    )
    for row in rows:
        if row["profile_id"] not in LATEX_PROFILE_IDS:
            continue
        parameterization = "SP" if row["parameterization"] == "sp" else r"$\mu$P"
        profile_name = row["profile_id"].replace("_", " ")
        reach_pct = 100.0 * row["paired_uniform_final_accuracy_reach_rate"]
        savings_pct = (
            100.0
            * row[
                "estimated_flops_savings_fraction_vs_uniform_full_horizon_at_fixed_accuracy"
            ]
        )
        lines.append(
            f"{parameterization} & {row['target_width']} & {profile_name} & "
            f"{100.0 * row['paired_uniform_final_validation_accuracy_mean']:.2f}\\% & "
            f"{row['restricted_mean_updates_to_paired_uniform_final_accuracy']:.0f} & "
            f"{row['restricted_mean_examples_to_paired_uniform_final_accuracy']:.0f} & "
            f"{row['restricted_mean_estimated_flops_to_paired_uniform_final_accuracy'] / 1e12:.3f} & "
            f"{reach_pct:.0f}\\% & {savings_pct:.1f}\\% \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    lines.extend(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Separate validation-loss diagnostic. Validation endpoints "
            r"are terminal loss, the final-10\%-epoch mean, and epoch-normalized "
            r"AUC. Resource columns use the paired uniform terminal validation-loss "
            r"threshold with the same full-horizon censoring. This diagnostic is "
            r"not the prespecified fixed-accuracy compute endpoint.}",
            r"\label{tab:profile_validation_loss_diagnostic}",
            r"\scriptsize",
            r"\begin{tabular}{lllrrrrrrrr}",
            r"\toprule",
            r"Param. & $N$ & Profile & Val. terminal & Val. final-window & "
            r"Val. nAUC & Test loss & Updates & Examples & Est. TFLOPs & Reach \\",
            r"\midrule",
        ]
    )
    for row in rows:
        if row["profile_id"] not in LATEX_PROFILE_IDS:
            continue
        parameterization = "SP" if row["parameterization"] == "sp" else r"$\mu$P"
        profile_name = row["profile_id"].replace("_", " ")
        loss_reach_pct = 100.0 * row["paired_uniform_terminal_loss_reach_rate"]
        lines.append(
            f"{parameterization} & {row['target_width']} & {profile_name} & "
            f"{row['validation_loss_terminal_mean']:.3f} & "
            f"{row['validation_loss_final_window_mean']:.3f} & "
            f"{row['validation_loss_normalized_auc_mean']:.3f} & "
            f"{row['test_loss_mean']:.3f} & "
            f"{row['restricted_mean_updates_to_paired_uniform_terminal_loss']:.0f} & "
            f"{row['restricted_mean_examples_to_paired_uniform_terminal_loss']:.0f} & "
            f"{row['restricted_mean_estimated_flops_to_paired_uniform_terminal_loss'] / 1e12:.3f} & "
            f"{loss_reach_pct:.0f}\\% \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    _atomic_write_text(tex_path, "\n".join(lines))

    mup_gate_records = [
        {
            "width": row["target_width"],
            "lr_grid_distance": learning_rate_grid_distance(
                row["proxy_learning_rate"],
                row["target_oracle_learning_rate"],
                EXPECTED_LR_GRID,
            ),
            "relative_regret": row["validation_regret_relative"],
        }
        for row in rows
        if row["parameterization"] == "mup"
        and row["profile_id"] == "uniform"
        and row["target_oracle_learning_rate"] is not None
    ]
    gates = {
        "schema_version": SCHEMA_VERSION,
        "profile_transfer": profile_gates,
        "mup_learning_rate_transfer": mup_transfer_claim_gate(mup_gate_records),
        "mup_gate_note": (
            "The preregistered target oracle covers all four held-out widths; "
            "the gate requires at least three widths within one LR-grid interval "
            "and 1% relative validation regret."
        ),
    }
    _atomic_write_text(gates_path, json.dumps(gates, indent=2, sort_keys=True) + "\n")
    return csv_path, contrast_csv_path, tex_path, gates_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--source",
        choices=("auto", "aggregate", "trials"),
        default="auto",
        help="load one aggregate snapshot or individual completed trial files",
    )
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument(
        "--summary-dir",
        type=Path,
        help="defaults to RUN_DIR/analysis",
    )
    parser.add_argument(
        "--only",
        choices=(
            "all",
            "profiles",
            "early-late",
            "vit",
            "lr",
            "summary",
        ),
        default="all",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    apply_paper_style()
    trials = load_completed_trials(args.run_dir, source=args.source)
    summary_dir = args.summary_dir or args.run_dir / "analysis"
    requested = (
        (
            "profiles",
            "early-late",
            "vit",
            "lr",
            "summary",
        )
        if args.only == "all"
        else (args.only,)
    )
    written: list[Path] = []
    if "profiles" in requested:
        written.append(make_profile_loss_figure(trials, args.figure_dir))
    if "early-late" in requested:
        written.append(make_early_late_figure(trials, args.figure_dir))
    if "vit" in requested:
        written.append(make_vit_confirmation_figure(trials, args.figure_dir))
    if "lr" in requested:
        written.append(make_lr_curve_figure(trials, args.figure_dir))
    if "summary" in requested:
        written.extend(write_zero_shot_summary(trials, summary_dir))
    for path in written:
        print(path)


if __name__ == "__main__":
    try:
        main()
    except ResultsError as exc:
        raise SystemExit(f"scale-transfer analysis failed: {exc}") from exc
