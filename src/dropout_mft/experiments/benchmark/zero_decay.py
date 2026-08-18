"""Shared zero-weight-decay dropout-budget sweep for benchmark datasets.

The default cohort reruns FI-2010 and Jannis, while the same generators accept
the other prepared benchmark datasets.  Every cohort uses depth-12 MLP and
Transformer comparisons with weight
decay fixed to exactly zero.  Learning rate is tuned independently for every
schedule (including no dropout), dropout magnitude is selected using
validation-only tuning seeds, and the test set is evaluated once on fresh
confirmation seeds.
"""

from __future__ import annotations

from dropout_mft.experiments.benchmark.protocol import (
    BENCHMARK_PROFILE_IDS,
    BUDGET_SEARCH_SEEDS,
    CAP_EXEMPT_PROFILES,
    CONFIRM_SEEDS,
    LINEAR_PROFILE_IDS,
    LR_GRIDS,
    LR_SEARCH_MEAN_DROPOUT,
    LR_SEARCH_SEEDS,
    MEAN_DROPOUT_GRID,
    TUNED_CONTROL_PROFILE_ID,
    BenchmarkTrialSpec,
    ModelKind,
    _spec_defaults,
)
from dropout_mft.training import BenchmarkDatasetName


ZERO_DECAY_COHORT_ID = "zero-weight-decay-depth12-dropout-sweep-v1"
ZERO_DECAY_DATASETS: tuple[BenchmarkDatasetName, ...] = (
    "fi2010",
    "openml_jannis",
)
ZERO_DECAY_SUPPORTED_DATASETS: tuple[BenchmarkDatasetName, ...] = (
    "fi2010",
    "tiny_imagenet",
    "amazon_reviews",
    "speech_commands",
    "openml_jannis",
)
ZERO_DECAY_MODEL_KINDS: tuple[ModelKind, ...] = ("mlp", "transformer")
ZERO_DECAY_DROPOUT_PROFILE_IDS = (*BENCHMARK_PROFILE_IDS, *LINEAR_PROFILE_IDS)
ZERO_DECAY_PROFILE_IDS = (
    *ZERO_DECAY_DROPOUT_PROFILE_IDS,
    TUNED_CONTROL_PROFILE_ID,
)
ZERO_DECAY_DEPTH = 12
ZERO_DECAY_WEIGHT_DECAY = 0.0
ZERO_DECAY_EPOCHS = {
    "fi2010": 50,
    "tiny_imagenet": 75,
    "amazon_reviews": 50,
    "speech_commands": 50,
    "openml_jannis": 100,
}


def _check_cell(dataset: BenchmarkDatasetName, model_kind: ModelKind) -> None:
    if dataset not in ZERO_DECAY_SUPPORTED_DATASETS:
        raise ValueError(
            f"Dataset {dataset!r} is not supported by the zero-decay protocol"
        )
    if model_kind not in ZERO_DECAY_MODEL_KINDS:
        raise ValueError(f"Model kind {model_kind!r} is not in the zero-decay cohort")


def _defaults(
    dataset: BenchmarkDatasetName,
    model_kind: ModelKind,
    *,
    epochs: int | None = None,
    train_size: int | None = None,
    validation_size: int | None = None,
    test_size: int | None = None,
) -> dict:
    _check_cell(dataset, model_kind)
    defaults = _spec_defaults(dataset, model_kind)
    defaults["epochs"] = ZERO_DECAY_EPOCHS[dataset] if epochs is None else epochs
    for key, value in (
        ("train_size", train_size),
        ("validation_size", validation_size),
        ("test_size", test_size),
    ):
        if value is not None:
            if value <= 0:
                raise ValueError(f"{key} must be positive")
            defaults[key] = value
    return defaults


def _mean_dropout(profile_id: str, mean_dropout: float) -> float:
    return 0.0 if profile_id == TUNED_CONTROL_PROFILE_ID else mean_dropout


def _max_dropout(profile_id: str, mean_dropout: float) -> float:
    """Return a declared cap large enough for the exact discrete profile.

    Linear profiles range from zero to twice their mean.  Big-step concentrates
    the depth-12 budget in four layers and therefore peaks at three times its
    mean.  The saturated step retains the original 0.20 cap.
    """

    if profile_id in CAP_EXEMPT_PROFILES:
        return max(0.30, 3.0 * mean_dropout)
    if profile_id in LINEAR_PROFILE_IDS:
        return max(0.20, 2.0 * mean_dropout)
    return 0.20


def zero_decay_lr_search_specs(
    dataset: BenchmarkDatasetName,
    model_kind: ModelKind,
    *,
    depth: int = ZERO_DECAY_DEPTH,
    cohort_id: str = ZERO_DECAY_COHORT_ID,
    epochs: int | None = None,
    train_size: int | None = None,
    validation_size: int | None = None,
    test_size: int | None = None,
) -> list[BenchmarkTrialSpec]:
    """Tune each schedule's learning rate at mean dropout 0.10."""

    defaults = _defaults(
        dataset,
        model_kind,
        epochs=epochs,
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
    )
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
            cohort_id=cohort_id,
            **defaults,
        )
        for profile_id in ZERO_DECAY_PROFILE_IDS
        for learning_rate in LR_GRIDS[model_kind]
        for seed in LR_SEARCH_SEEDS
    ]


def zero_decay_budget_search_specs(
    dataset: BenchmarkDatasetName,
    model_kind: ModelKind,
    selected_learning_rates: dict[str, float],
    *,
    depth: int = ZERO_DECAY_DEPTH,
    cohort_id: str = ZERO_DECAY_COHORT_ID,
    epochs: int | None = None,
    train_size: int | None = None,
    validation_size: int | None = None,
    test_size: int | None = None,
) -> list[BenchmarkTrialSpec]:
    """Sweep four nonzero dropout budgets using three validation seeds."""

    defaults = _defaults(
        dataset,
        model_kind,
        epochs=epochs,
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
    )
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
            cohort_id=cohort_id,
            **defaults,
        )
        for profile_id in ZERO_DECAY_DROPOUT_PROFILE_IDS
        for mean_dropout in MEAN_DROPOUT_GRID
        for seed in BUDGET_SEARCH_SEEDS
    ]


def zero_decay_confirm_specs(
    dataset: BenchmarkDatasetName,
    model_kind: ModelKind,
    budget_selection: dict[str, dict],
    lr_selection: dict[str, dict],
    *,
    depth: int = ZERO_DECAY_DEPTH,
    cohort_id: str = ZERO_DECAY_COHORT_ID,
    epochs: int | None = None,
    train_size: int | None = None,
    validation_size: int | None = None,
    test_size: int | None = None,
) -> list[BenchmarkTrialSpec]:
    """Confirm selected dropout arms and tuned no-dropout on fresh seeds."""

    defaults = _defaults(
        dataset,
        model_kind,
        epochs=epochs,
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
    )
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
        mean_dropout = _mean_dropout(profile_id, float(choice.get("mean_dropout", 0.0)))
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
                    cohort_id=cohort_id,
                    **defaults,
                )
            )
    return specs


def zero_decay_trial_count(
    datasets: tuple[BenchmarkDatasetName, ...] = ZERO_DECAY_DATASETS,
) -> dict[str, int]:
    """Return exact counts for any datasets using the shared protocol."""

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
    for dataset in datasets:
        for model_kind in ZERO_DECAY_MODEL_KINDS:
            counts["lr_search"] += len(zero_decay_lr_search_specs(dataset, model_kind))
            counts["budget_search"] += len(
                zero_decay_budget_search_specs(dataset, model_kind, dummy_lrs)
            )
            counts["confirm"] += len(
                zero_decay_confirm_specs(
                    dataset,
                    model_kind,
                    dummy_budgets,
                    dummy_lr_selection,
                )
            )
    counts["total"] = sum(counts.values())
    return counts
