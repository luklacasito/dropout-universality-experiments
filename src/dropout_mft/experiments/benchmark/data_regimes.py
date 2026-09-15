"""Prespecified sample-size regimes for the zero-decay dropout benchmark.

Each regime is an independent immutable cohort.  Keeping the three Amazon
sample sizes in separate run directories prevents the existing
``dataset/model_kind`` selection cells from mixing arms across sample sizes.
Amazon's cohort prefix is also the explicit signal used by the data loader for
the nested-training/shared-heldout split protocol.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace

from dropout_mft.experiments.benchmark.protocol import BenchmarkTrialSpec, ModelKind
from dropout_mft.experiments.benchmark.workflow import (
    plan_selected_stage,
    save_plan,
    stage_selections,
)
from dropout_mft.experiments.benchmark.zero_decay import (
    ZERO_DECAY_DEPTH,
    ZERO_DECAY_DROPOUT_PROFILE_IDS,
    ZERO_DECAY_MODEL_KINDS,
    ZERO_DECAY_PROFILE_IDS,
    zero_decay_budget_search_specs,
    zero_decay_confirm_specs,
    zero_decay_lr_search_specs,
)
from dropout_mft.training import BenchmarkDatasetName

DATA_REGIME_COHORT_PREFIX = "data-regime-scaling-v1"
DATA_REGIME_DEPTH = ZERO_DECAY_DEPTH
DATA_REGIME_MODEL_KINDS: tuple[ModelKind, ...] = ZERO_DECAY_MODEL_KINDS


@dataclass(frozen=True)
class BenchmarkDataRegime:
    """One sample-size intervention with fixed heldout sizes and epoch budget."""

    regime_id: str
    dataset: BenchmarkDatasetName
    train_size: int
    validation_size: int
    test_size: int
    epochs: int

    def __post_init__(self) -> None:
        if not self.regime_id.strip():
            raise ValueError("regime_id must not be empty")
        if min(self.train_size, self.validation_size, self.test_size) <= 0:
            raise ValueError("all split sizes must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")

    @property
    def cohort_id(self) -> str:
        return f"{DATA_REGIME_COHORT_PREFIX}/{self.regime_id}"


DATA_REGIMES: dict[str, BenchmarkDataRegime] = {
    "amazon_n2000": BenchmarkDataRegime(
        regime_id="amazon_n2000",
        dataset="amazon_reviews",
        train_size=2_000,
        validation_size=5_000,
        test_size=10_000,
        epochs=100,
    ),
    "amazon_n5000": BenchmarkDataRegime(
        regime_id="amazon_n5000",
        dataset="amazon_reviews",
        train_size=5_000,
        validation_size=5_000,
        test_size=10_000,
        epochs=100,
    ),
    "amazon_n20000": BenchmarkDataRegime(
        regime_id="amazon_n20000",
        dataset="amazon_reviews",
        train_size=20_000,
        validation_size=5_000,
        test_size=10_000,
        epochs=100,
    ),
    "tiny_n80000": BenchmarkDataRegime(
        regime_id="tiny_n80000",
        dataset="tiny_imagenet",
        train_size=80_000,
        validation_size=10_000,
        test_size=10_000,
        epochs=75,
    ),
}
DATA_REGIME_IDS = tuple(DATA_REGIMES)


def data_regime(regime_id: str) -> BenchmarkDataRegime:
    try:
        return DATA_REGIMES[regime_id]
    except KeyError as exc:
        raise ValueError(
            f"Unknown data regime {regime_id!r}; choose from {DATA_REGIME_IDS!r}"
        ) from exc


def _apply_regime(
    specs: list[BenchmarkTrialSpec], regime: BenchmarkDataRegime
) -> list[BenchmarkTrialSpec]:
    """Apply only the sample-size intervention to shared zero-decay specs."""

    return [
        replace(
            spec,
            train_size=regime.train_size,
            validation_size=regime.validation_size,
            test_size=regime.test_size,
            epochs=regime.epochs,
            depth=DATA_REGIME_DEPTH,
            cohort_id=regime.cohort_id,
        )
        for spec in specs
    ]


def data_regime_lr_search_specs(
    regime_id: str, model_kind: ModelKind
) -> list[BenchmarkTrialSpec]:
    regime = data_regime(regime_id)
    return _apply_regime(
        zero_decay_lr_search_specs(
            regime.dataset,
            model_kind,
            depth=DATA_REGIME_DEPTH,
            cohort_id=regime.cohort_id,
            epochs=regime.epochs,
        ),
        regime,
    )


def data_regime_budget_search_specs(
    regime_id: str,
    model_kind: ModelKind,
    selected_learning_rates: dict[str, float],
) -> list[BenchmarkTrialSpec]:
    regime = data_regime(regime_id)
    return _apply_regime(
        zero_decay_budget_search_specs(
            regime.dataset,
            model_kind,
            selected_learning_rates,
            depth=DATA_REGIME_DEPTH,
            cohort_id=regime.cohort_id,
            epochs=regime.epochs,
        ),
        regime,
    )


def data_regime_confirm_specs(
    regime_id: str,
    model_kind: ModelKind,
    budget_selection: dict[str, dict],
    lr_selection: dict[str, dict],
) -> list[BenchmarkTrialSpec]:
    regime = data_regime(regime_id)
    return _apply_regime(
        zero_decay_confirm_specs(
            regime.dataset,
            model_kind,
            budget_selection,
            lr_selection,
            depth=DATA_REGIME_DEPTH,
            cohort_id=regime.cohort_id,
            epochs=regime.epochs,
        ),
        regime,
    )


def data_regime_trial_count(regime_id: str) -> dict[str, int]:
    """Return exact validation/test protocol costs for one independent run."""

    data_regime(regime_id)
    dummy_lrs = {profile_id: 1e-4 for profile_id in ZERO_DECAY_PROFILE_IDS}
    dummy_budgets = {
        profile_id: {"learning_rate": 1e-4, "mean_dropout": 0.10}
        for profile_id in ZERO_DECAY_DROPOUT_PROFILE_IDS
    }
    dummy_lr_selection = {
        profile_id: {
            "learning_rate": 1e-4,
            "mean_dropout": 0.0 if profile_id == "none_tuned" else 0.10,
        }
        for profile_id in ZERO_DECAY_PROFILE_IDS
    }
    counts = {"lr_search": 0, "budget_search": 0, "confirm": 0}
    for model_kind in DATA_REGIME_MODEL_KINDS:
        counts["lr_search"] += len(data_regime_lr_search_specs(regime_id, model_kind))
        counts["budget_search"] += len(
            data_regime_budget_search_specs(regime_id, model_kind, dummy_lrs)
        )
        counts["confirm"] += len(
            data_regime_confirm_specs(
                regime_id,
                model_kind,
                dummy_budgets,
                dummy_lr_selection,
            )
        )
    counts["total"] = sum(counts.values())
    return counts


def command_plan(args: argparse.Namespace) -> None:
    regime = data_regime(args.regime)
    specs = plan_selected_stage(
        args.stage,
        ((regime.dataset, m) for m in DATA_REGIME_MODEL_KINDS),
        stage_selections(args.run_dir, args.stage),
        lr_search=lambda d, m: data_regime_lr_search_specs(args.regime, m),
        budget_search=lambda d, m, rates: data_regime_budget_search_specs(
            args.regime, m, rates
        ),
        confirm=lambda d, m, budgets, rates: data_regime_confirm_specs(
            args.regime, m, budgets, rates
        ),
    )
    save_plan(args.run_dir, args.stage, specs, description=f"regime={args.regime} ")


def command_cost(args: argparse.Namespace) -> None:
    regime = data_regime(args.regime)
    print(
        f"regime={regime.regime_id} dataset={regime.dataset} "
        f"train={regime.train_size} validation={regime.validation_size} "
        f"test={regime.test_size}"
    )
    print(
        f"models=mlp,transformer depth={DATA_REGIME_DEPTH} "
        f"epochs={regime.epochs} weight_decay=0"
    )
    print("dropout_grid=0.05,0.10,0.15,0.20")
    for stage, count in data_regime_trial_count(args.regime).items():
        print(f"{stage:14s} {count:5d} trials")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--run-dir", required=True)
    plan.add_argument("--regime", choices=DATA_REGIME_IDS, required=True)
    plan.add_argument(
        "--stage", choices=("lr_search", "budget_search", "confirm"), required=True
    )
    plan.set_defaults(func=command_plan)
    cost = commands.add_parser("cost")
    cost.add_argument("--regime", choices=DATA_REGIME_IDS, required=True)
    cost.set_defaults(func=command_cost)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
