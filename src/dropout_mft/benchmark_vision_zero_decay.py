"""Zero-weight-decay dropout-budget sweep for Tiny ImageNet.

The cohort mirrors the completed FI-2010/Jannis protocol: depth 12, six arms,
an independently tuned learning rate per arm, a four-point dropout-rate sweep
on validation-only seeds, and fresh-seed test confirmation.  Test metrics are
never read during either tuning stage.
"""

from __future__ import annotations

from .benchmark_suite import (
    BUDGET_SEARCH_SEEDS,
    CONFIRM_SEEDS,
    LR_GRIDS,
    LR_SEARCH_MEAN_DROPOUT,
    LR_SEARCH_SEEDS,
    MEAN_DROPOUT_GRID,
    TUNED_CONTROL_PROFILE_ID,
    BenchmarkTrialSpec,
    ModelKind,
    _spec_defaults,
)
from .benchmark_zero_decay import (
    ZERO_DECAY_DROPOUT_PROFILE_IDS,
    ZERO_DECAY_PROFILE_IDS,
    ZERO_DECAY_WEIGHT_DECAY,
    _max_dropout,
    _mean_dropout,
)


VISION_ZERO_DECAY_COHORT_ID = "tiny-imagenet-zero-weight-decay-depth12-sweep-v1"
VISION_ZERO_DECAY_DATASET = "tiny_imagenet"
VISION_ZERO_DECAY_MODEL_KINDS: tuple[ModelKind, ...] = ("mlp", "transformer")
VISION_ZERO_DECAY_DEPTH = 12
VISION_ZERO_DECAY_EPOCHS = 75


def _defaults(model_kind: ModelKind) -> dict:
    if model_kind not in VISION_ZERO_DECAY_MODEL_KINDS:
        raise ValueError(f"Unknown vision model kind: {model_kind!r}")
    defaults = _spec_defaults(VISION_ZERO_DECAY_DATASET, model_kind)
    defaults["epochs"] = VISION_ZERO_DECAY_EPOCHS
    return defaults


def vision_zero_decay_lr_search_specs(
    model_kind: ModelKind,
    *,
    depth: int = VISION_ZERO_DECAY_DEPTH,
) -> list[BenchmarkTrialSpec]:
    defaults = _defaults(model_kind)
    return [
        BenchmarkTrialSpec(
            stage="lr_search",
            profile_id=profile_id,
            mean_dropout=_mean_dropout(profile_id, LR_SEARCH_MEAN_DROPOUT),
            max_dropout=_max_dropout(profile_id, LR_SEARCH_MEAN_DROPOUT),
            learning_rate=learning_rate,
            seed=seed,
            depth=depth,
            weight_decay=ZERO_DECAY_WEIGHT_DECAY,
            evaluate_test=False,
            cohort_id=VISION_ZERO_DECAY_COHORT_ID,
            **defaults,
        )
        for profile_id in ZERO_DECAY_PROFILE_IDS
        for learning_rate in LR_GRIDS[model_kind]
        for seed in LR_SEARCH_SEEDS
    ]


def vision_zero_decay_budget_search_specs(
    model_kind: ModelKind,
    selected_learning_rates: dict[str, float],
    *,
    depth: int = VISION_ZERO_DECAY_DEPTH,
) -> list[BenchmarkTrialSpec]:
    defaults = _defaults(model_kind)
    missing = set(ZERO_DECAY_DROPOUT_PROFILE_IDS) - set(selected_learning_rates)
    if missing:
        raise ValueError(f"Missing selected learning rates: {sorted(missing)!r}")
    return [
        BenchmarkTrialSpec(
            stage="budget_search",
            profile_id=profile_id,
            mean_dropout=mean_dropout,
            max_dropout=_max_dropout(profile_id, mean_dropout),
            learning_rate=float(selected_learning_rates[profile_id]),
            seed=seed,
            depth=depth,
            weight_decay=ZERO_DECAY_WEIGHT_DECAY,
            evaluate_test=False,
            cohort_id=VISION_ZERO_DECAY_COHORT_ID,
            **defaults,
        )
        for profile_id in ZERO_DECAY_DROPOUT_PROFILE_IDS
        for mean_dropout in MEAN_DROPOUT_GRID
        for seed in BUDGET_SEARCH_SEEDS
    ]


def vision_zero_decay_confirm_specs(
    model_kind: ModelKind,
    budget_selection: dict[str, dict],
    lr_selection: dict[str, dict],
    *,
    depth: int = VISION_ZERO_DECAY_DEPTH,
) -> list[BenchmarkTrialSpec]:
    defaults = _defaults(model_kind)
    missing = set(ZERO_DECAY_DROPOUT_PROFILE_IDS) - set(budget_selection)
    if missing:
        raise ValueError(f"Missing budget selections: {sorted(missing)!r}")
    if TUNED_CONTROL_PROFILE_ID not in lr_selection:
        raise ValueError("Missing independently tuned no-dropout selection")

    selected = {
        profile_id: budget_selection[profile_id]
        for profile_id in ZERO_DECAY_DROPOUT_PROFILE_IDS
    }
    selected[TUNED_CONTROL_PROFILE_ID] = lr_selection[TUNED_CONTROL_PROFILE_ID]
    specs: list[BenchmarkTrialSpec] = []
    for profile_id in ZERO_DECAY_PROFILE_IDS:
        choice = selected[profile_id]
        mean_dropout = _mean_dropout(
            profile_id, float(choice.get("mean_dropout", 0.0))
        )
        for seed in CONFIRM_SEEDS:
            specs.append(
                BenchmarkTrialSpec(
                    stage="confirm",
                    profile_id=profile_id,
                    mean_dropout=mean_dropout,
                    max_dropout=_max_dropout(profile_id, mean_dropout),
                    learning_rate=float(choice["learning_rate"]),
                    seed=seed,
                    depth=depth,
                    weight_decay=ZERO_DECAY_WEIGHT_DECAY,
                    evaluate_test=True,
                    cohort_id=VISION_ZERO_DECAY_COHORT_ID,
                    **defaults,
                )
            )
    return specs


def vision_zero_decay_trial_count() -> dict[str, int]:
    counts = {"lr_search": 0, "budget_search": 0, "confirm": 0}
    dummy_lrs = {profile_id: 1e-4 for profile_id in ZERO_DECAY_PROFILE_IDS}
    dummy_budgets = {
        profile_id: {"learning_rate": 1e-4, "mean_dropout": 0.10}
        for profile_id in ZERO_DECAY_DROPOUT_PROFILE_IDS
    }
    dummy_lr_selection = {
        profile_id: {
            "learning_rate": 1e-4,
            "mean_dropout": _mean_dropout(profile_id, 0.10),
        }
        for profile_id in ZERO_DECAY_PROFILE_IDS
    }
    for model_kind in VISION_ZERO_DECAY_MODEL_KINDS:
        counts["lr_search"] += len(
            vision_zero_decay_lr_search_specs(model_kind)
        )
        counts["budget_search"] += len(
            vision_zero_decay_budget_search_specs(model_kind, dummy_lrs)
        )
        counts["confirm"] += len(
            vision_zero_decay_confirm_specs(
                model_kind, dummy_budgets, dummy_lr_selection
            )
        )
    counts["total"] = sum(counts.values())
    return counts
