#!/usr/bin/env python
"""Plan an immutable zero-weight-decay depth-12 dropout sweep."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dropout_mft.experiments.benchmark.zero_decay import (  # noqa: E402
    ZERO_DECAY_COHORT_ID,
    ZERO_DECAY_DATASETS,
    ZERO_DECAY_DEPTH,
    ZERO_DECAY_MODEL_KINDS,
    ZERO_DECAY_SUPPORTED_DATASETS,
    zero_decay_budget_search_specs,
    zero_decay_confirm_specs,
    zero_decay_lr_search_specs,
    zero_decay_trial_count,
)
from dropout_mft.experiments.benchmark.workflow import (  # noqa: E402
    load_selection as _load_selection,
    write_immutable_manifest,
)
from dropout_mft.provenance import provenance_sha256  # noqa: E402


def command_plan(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    datasets = tuple(args.datasets or ZERO_DECAY_DATASETS)
    specs = []
    if args.stage == "lr_search":
        for dataset in datasets:
            for model_kind in ZERO_DECAY_MODEL_KINDS:
                specs.extend(
                    zero_decay_lr_search_specs(
                        dataset,
                        model_kind,
                        depth=args.depth,
                        cohort_id=args.cohort_id,
                    )
                )
    elif args.stage == "budget_search":
        lr_selection = _load_selection(run_dir, "lr_search")
        for dataset in datasets:
            for model_kind in ZERO_DECAY_MODEL_KINDS:
                cell = f"{dataset}/{model_kind}"
                if cell not in lr_selection:
                    raise SystemExit(f"Missing LR selection for {cell}")
                selected_lrs = {
                    profile_id: float(choice["learning_rate"])
                    for profile_id, choice in lr_selection[cell].items()
                }
                specs.extend(
                    zero_decay_budget_search_specs(
                        dataset,
                        model_kind,
                        selected_lrs,
                        depth=args.depth,
                        cohort_id=args.cohort_id,
                    )
                )
    else:
        lr_selection = _load_selection(run_dir, "lr_search")
        budget_selection = _load_selection(run_dir, "budget_search")
        for dataset in datasets:
            for model_kind in ZERO_DECAY_MODEL_KINDS:
                cell = f"{dataset}/{model_kind}"
                if cell not in lr_selection or cell not in budget_selection:
                    raise SystemExit(f"Missing confirmation selection for {cell}")
                specs.extend(
                    zero_decay_confirm_specs(
                        dataset,
                        model_kind,
                        budget_selection[cell],
                        lr_selection[cell],
                        depth=args.depth,
                        cohort_id=args.cohort_id,
                    )
                )

    path, provenance = write_immutable_manifest(run_dir, args.stage, specs)
    print(
        f"stage={args.stage} trials={len(specs)} "
        f"cells={len({spec.cell for spec in specs})}"
    )
    print(f"manifest={path}")
    print(f"provenance_sha256={provenance_sha256(provenance)}")


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


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
