"""Depth-12 Tiny ImageNet comparison across every benchmark schedule.

The vision cohort keeps the validation-only learning-rate screen and the
fresh-seed test confirmation separate:

* 60 LR-search trials: two architectures x six profiles x five learning rates;
* 60 confirmation trials: two architectures x six profiles x five fresh seeds.

The no-dropout arm is tuned independently.  This prevents a learning rate
chosen under dropout from accidentally weakening the control.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dropout_mft.experiments.benchmark.protocol import (
    BENCHMARK_PROFILE_IDS,
    CAP_EXEMPT_PROFILES,
    CONFIRM_SEEDS,
    LINEAR_PROFILE_IDS,
    LR_GRIDS,
    LR_SEARCH_MEAN_DROPOUT,
    LR_SEARCH_SEEDS,
    TUNED_CONTROL_PROFILE_ID,
    BenchmarkTrialSpec,
    ModelKind,
    _spec_defaults,
)
from dropout_mft.experiments.benchmark.workflow import (
    load_selection,
    save_plan,
)

VISION_COHORT_ID = "tiny-imagenet-depth12-all-schedules-v1"
VISION_DATASET = "tiny_imagenet"
VISION_DEPTH = 12
VISION_MODEL_KINDS: tuple[ModelKind, ...] = ("mlp", "transformer")
VISION_PROFILE_IDS = (
    *BENCHMARK_PROFILE_IDS,
    *LINEAR_PROFILE_IDS,
    TUNED_CONTROL_PROFILE_ID,
)


def _mean_dropout(profile_id: str) -> float:
    return 0.0 if profile_id == TUNED_CONTROL_PROFILE_ID else LR_SEARCH_MEAN_DROPOUT


def _max_dropout(profile_id: str) -> float:
    return 0.30 if profile_id in CAP_EXEMPT_PROFILES else 0.20


def vision_lr_search_specs(
    model_kind: ModelKind,
    *,
    depth: int = VISION_DEPTH,
) -> list[BenchmarkTrialSpec]:
    """Return the 30 validation-only screens for one vision architecture."""

    if model_kind not in VISION_MODEL_KINDS:
        raise ValueError(f"Unknown vision model kind: {model_kind!r}")
    defaults = _spec_defaults(VISION_DATASET, model_kind)
    return [
        BenchmarkTrialSpec(
            stage="lr_search",
            profile_id=profile_id,
            mean_dropout=_mean_dropout(profile_id),
            max_dropout=_max_dropout(profile_id),
            learning_rate=learning_rate,
            seed=seed,
            depth=depth,
            evaluate_test=False,
            cohort_id=VISION_COHORT_ID,
            **defaults,
        )
        for profile_id in VISION_PROFILE_IDS
        for learning_rate in LR_GRIDS[model_kind]
        for seed in LR_SEARCH_SEEDS
    ]


def vision_confirm_specs(
    model_kind: ModelKind,
    selected: dict[str, dict],
    *,
    depth: int = VISION_DEPTH,
) -> list[BenchmarkTrialSpec]:
    """Return the 30 fresh-seed test confirmations for one architecture."""

    if model_kind not in VISION_MODEL_KINDS:
        raise ValueError(f"Unknown vision model kind: {model_kind!r}")
    missing = set(VISION_PROFILE_IDS) - set(selected)
    if missing:
        raise ValueError(f"Vision selection is missing profiles: {sorted(missing)!r}")

    defaults = _spec_defaults(VISION_DATASET, model_kind)
    specs: list[BenchmarkTrialSpec] = []
    for profile_id in VISION_PROFILE_IDS:
        choice = selected[profile_id]
        mean_dropout = float(choice["mean_dropout"])
        if mean_dropout != _mean_dropout(profile_id):
            raise ValueError(f"Invalid selected dropout budget for {profile_id!r}")
        for seed in CONFIRM_SEEDS:
            specs.append(
                BenchmarkTrialSpec(
                    stage="confirm",
                    profile_id=profile_id,
                    mean_dropout=mean_dropout,
                    max_dropout=_max_dropout(profile_id),
                    learning_rate=float(choice["learning_rate"]),
                    seed=seed,
                    depth=depth,
                    evaluate_test=True,
                    cohort_id=VISION_COHORT_ID,
                    **defaults,
                )
            )
    return specs


def vision_trial_count() -> dict[str, int]:
    """Return the exact fixed trial budget for the complete cohort."""

    lr_search = sum(
        len(vision_lr_search_specs(model_kind)) for model_kind in VISION_MODEL_KINDS
    )
    dummy = {
        profile_id: {
            "profile_id": profile_id,
            "learning_rate": 1e-4,
            "mean_dropout": _mean_dropout(profile_id),
        }
        for profile_id in VISION_PROFILE_IDS
    }
    confirm = sum(
        len(vision_confirm_specs(model_kind, dummy))
        for model_kind in VISION_MODEL_KINDS
    )
    return {"lr_search": lr_search, "confirm": confirm, "total": lr_search + confirm}


def command_plan(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    if args.stage == "lr_search":
        specs = [
            spec
            for model_kind in VISION_MODEL_KINDS
            for spec in vision_lr_search_specs(model_kind, depth=args.depth)
        ]
    else:
        selection = load_selection(run_dir, "lr_search")
        specs = []
        for model_kind in VISION_MODEL_KINDS:
            cell = f"tiny_imagenet/{model_kind}"
            if cell not in selection:
                raise SystemExit(f"Missing vision selection for {cell}")
            specs.extend(
                vision_confirm_specs(
                    model_kind,
                    selection[cell],
                    depth=args.depth,
                )
            )

    save_plan(run_dir, args.stage, specs)


def command_cost(_args: argparse.Namespace) -> None:
    for stage, count in vision_trial_count().items():
        print(f"{stage:14s} {count:5d} trials")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan")
    plan.add_argument("--run-dir", required=True)
    plan.add_argument("--stage", choices=("lr_search", "confirm"), required=True)
    plan.add_argument("--depth", type=int, default=12)
    plan.set_defaults(func=command_plan)

    cost = commands.add_parser("cost")
    cost.set_defaults(func=command_cost)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
