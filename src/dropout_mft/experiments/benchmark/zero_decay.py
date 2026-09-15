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

import argparse

from dropout_mft.experiments.benchmark.protocol import (
    BENCHMARK_PROFILE_IDS,
    BUDGET_SEARCH_SEEDS,
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
from dropout_mft.experiments.benchmark.workflow import (
    plan_selected_stage,
    save_plan,
    stage_selections,
)
from dropout_mft.schedules import comparison_profile_cap
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
    return comparison_profile_cap(profile_id, mean_dropout)


def zero_decay_specs(
    stage: str,
    dataset: BenchmarkDatasetName,
    model_kind: ModelKind,
    *,
    selected_learning_rates: dict[str, float] | None = None,
    budget_selection: dict[str, dict] | None = None,
    lr_selection: dict[str, dict] | None = None,
    depth: int = ZERO_DECAY_DEPTH,
    cohort_id: str = ZERO_DECAY_COHORT_ID,
    epochs: int | None = None,
    train_size: int | None = None,
    validation_size: int | None = None,
    test_size: int | None = None,
) -> list[BenchmarkTrialSpec]:
    """One zero-decay recipe; stages differ only in candidates and seed sets."""
    defaults = _defaults(
        dataset,
        model_kind,
        epochs=epochs,
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
    )
    if stage == "lr_search":
        profiles, seeds = ZERO_DECAY_PROFILE_IDS, LR_SEARCH_SEEDS
        choices = {
            profile: [(lr, LR_SEARCH_MEAN_DROPOUT) for lr in LR_GRIDS[model_kind]]
            for profile in profiles
        }
    elif stage == "budget_search":
        profiles, seeds = ZERO_DECAY_DROPOUT_PROFILE_IDS, BUDGET_SEARCH_SEEDS
        selected_learning_rates = selected_learning_rates or {}
        missing = set(profiles) - set(selected_learning_rates)
        if missing:
            raise ValueError(f"Missing selected learning rates: {sorted(missing)!r}")
        choices = {
            profile: [
                (float(selected_learning_rates[profile]), mean)
                for mean in MEAN_DROPOUT_GRID
            ]
            for profile in profiles
        }
    elif stage == "confirm":
        profiles, seeds = ZERO_DECAY_PROFILE_IDS, CONFIRM_SEEDS
        budget_selection, lr_selection = budget_selection or {}, lr_selection or {}
        missing = set(ZERO_DECAY_DROPOUT_PROFILE_IDS) - set(budget_selection)
        if missing:
            raise ValueError(f"Missing budget selections: {sorted(missing)!r}")
        if TUNED_CONTROL_PROFILE_ID not in lr_selection:
            raise ValueError("Missing independently tuned no-dropout selection")
        selected = {p: budget_selection[p] for p in ZERO_DECAY_DROPOUT_PROFILE_IDS}
        selected[TUNED_CONTROL_PROFILE_ID] = lr_selection[TUNED_CONTROL_PROFILE_ID]
        choices = {
            profile: [
                (
                    float(choice["learning_rate"]),
                    _mean_dropout(profile, float(choice.get("mean_dropout", 0.0))),
                )
            ]
            for profile, choice in selected.items()
        }
    else:
        raise ValueError(f"Unknown zero-decay stage: {stage!r}")
    return [
        BenchmarkTrialSpec(
            **defaults,
            stage=stage,
            profile_id=profile,
            mean_dropout=_mean_dropout(profile, budget),
            max_dropout=_max_dropout(profile, budget),
            learning_rate=lr,
            seed=seed,
            depth=depth,
            weight_decay=ZERO_DECAY_WEIGHT_DECAY,
            evaluate_test=stage == "confirm",
            cohort_id=cohort_id,
        )
        for profile in profiles
        for lr, budget in choices[profile]
        for seed in seeds
    ]


def zero_decay_lr_search_specs(
    dataset: BenchmarkDatasetName, model_kind: ModelKind, **options
) -> list[BenchmarkTrialSpec]:
    return zero_decay_specs("lr_search", dataset, model_kind, **options)


def zero_decay_budget_search_specs(
    dataset: BenchmarkDatasetName,
    model_kind: ModelKind,
    selected_learning_rates: dict[str, float],
    **options,
) -> list[BenchmarkTrialSpec]:
    return zero_decay_specs(
        "budget_search",
        dataset,
        model_kind,
        selected_learning_rates=selected_learning_rates,
        **options,
    )


def zero_decay_confirm_specs(
    dataset: BenchmarkDatasetName,
    model_kind: ModelKind,
    budget_selection: dict[str, dict],
    lr_selection: dict[str, dict],
    **options,
) -> list[BenchmarkTrialSpec]:
    return zero_decay_specs(
        "confirm",
        dataset,
        model_kind,
        budget_selection=budget_selection,
        lr_selection=lr_selection,
        **options,
    )


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


def command_plan(args: argparse.Namespace) -> None:
    datasets = tuple(args.datasets or ZERO_DECAY_DATASETS)
    options = dict(depth=args.depth, cohort_id=args.cohort_id)
    specs = plan_selected_stage(
        args.stage,
        ((d, m) for d in datasets for m in ZERO_DECAY_MODEL_KINDS),
        stage_selections(args.run_dir, args.stage),
        lr_search=lambda d, m: zero_decay_lr_search_specs(d, m, **options),
        budget_search=lambda d, m, rates: zero_decay_budget_search_specs(
            d, m, rates, **options
        ),
        confirm=lambda d, m, budgets, rates: zero_decay_confirm_specs(
            d, m, budgets, rates, **options
        ),
    )
    save_plan(args.run_dir, args.stage, specs)


def command_cost(args: argparse.Namespace) -> None:
    datasets = tuple(args.datasets or ZERO_DECAY_DATASETS)
    print(f"datasets={','.join(datasets)} models=mlp,transformer depth=12")
    print("weight_decay=0 dropout_grid=0.05,0.10,0.15,0.20")
    for stage, count in zero_decay_trial_count(datasets).items():
        print(f"{stage:14s} {count:5d} trials")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--run-dir", required=True)
    plan.add_argument(
        "--stage", choices=("lr_search", "budget_search", "confirm"), required=True
    )
    plan.add_argument("--depth", type=int, default=ZERO_DECAY_DEPTH)
    plan.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        choices=ZERO_DECAY_SUPPORTED_DATASETS,
        help="dataset to include; repeat for a multi-dataset cohort",
    )
    plan.add_argument("--cohort-id", default=ZERO_DECAY_COHORT_ID)
    plan.set_defaults(func=command_plan)
    cost = commands.add_parser("cost")
    cost.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        choices=ZERO_DECAY_SUPPORTED_DATASETS,
    )
    cost.set_defaults(func=command_cost)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


# Tiny ImageNet preset of the same zero-decay study.

VISION_ZERO_DECAY_COHORT_ID = "tiny-imagenet-zero-weight-decay-depth12-sweep-v1"
VISION_ZERO_DECAY_DATASET = "tiny_imagenet"
VISION_ZERO_DECAY_MODEL_KINDS: tuple[ModelKind, ...] = ("mlp", "transformer")
VISION_ZERO_DECAY_DEPTH = 12
VISION_ZERO_DECAY_EPOCHS = 75


def vision_zero_decay_lr_search_specs(
    model_kind: ModelKind, *, depth: int = VISION_ZERO_DECAY_DEPTH
) -> list[BenchmarkTrialSpec]:
    return zero_decay_lr_search_specs(
        VISION_ZERO_DECAY_DATASET,
        model_kind,
        depth=depth,
        cohort_id=VISION_ZERO_DECAY_COHORT_ID,
        epochs=VISION_ZERO_DECAY_EPOCHS,
    )


def vision_zero_decay_budget_search_specs(
    model_kind: ModelKind,
    selected_learning_rates: dict[str, float],
    *,
    depth: int = VISION_ZERO_DECAY_DEPTH,
) -> list[BenchmarkTrialSpec]:
    return zero_decay_budget_search_specs(
        VISION_ZERO_DECAY_DATASET,
        model_kind,
        selected_learning_rates,
        depth=depth,
        cohort_id=VISION_ZERO_DECAY_COHORT_ID,
        epochs=VISION_ZERO_DECAY_EPOCHS,
    )


def vision_zero_decay_confirm_specs(
    model_kind: ModelKind,
    budget_selection: dict[str, dict],
    lr_selection: dict[str, dict],
    *,
    depth: int = VISION_ZERO_DECAY_DEPTH,
) -> list[BenchmarkTrialSpec]:
    return zero_decay_confirm_specs(
        VISION_ZERO_DECAY_DATASET,
        model_kind,
        budget_selection,
        lr_selection,
        depth=depth,
        cohort_id=VISION_ZERO_DECAY_COHORT_ID,
        epochs=VISION_ZERO_DECAY_EPOCHS,
    )


def vision_zero_decay_trial_count() -> dict[str, int]:
    return zero_decay_trial_count((VISION_ZERO_DECAY_DATASET,))


def vision_command_plan(args: argparse.Namespace) -> None:
    command_plan(
        argparse.Namespace(
            **(
                vars(args)
                | {
                    "datasets": [VISION_ZERO_DECAY_DATASET],
                    "cohort_id": VISION_ZERO_DECAY_COHORT_ID,
                }
            )
        )
    )


def vision_command_cost(_args: argparse.Namespace) -> None:
    print("dataset=tiny_imagenet models=mlp,transformer depth=12 epochs=75")
    print("weight_decay=0 dropout_grid=0.05,0.10,0.15,0.20")
    for stage, count in vision_zero_decay_trial_count().items():
        print(f"{stage:14s} {count:5d} trials")


def vision_build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--run-dir", required=True)
    plan.add_argument(
        "--stage", choices=("lr_search", "budget_search", "confirm"), required=True
    )
    plan.add_argument("--depth", type=int, default=VISION_ZERO_DECAY_DEPTH)
    plan.set_defaults(func=vision_command_plan)
    cost = commands.add_parser("cost")
    cost.set_defaults(func=vision_command_cost)
    return parser


def vision_main(argv: list[str] | None = None) -> None:
    args = vision_build_parser().parse_args(argv)
    args.func(args)
