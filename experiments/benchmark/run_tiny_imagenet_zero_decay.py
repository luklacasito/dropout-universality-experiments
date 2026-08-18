#!/usr/bin/env python
"""Plan the immutable Tiny ImageNet zero-weight-decay dropout sweep."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dropout_mft.experiments.benchmark.vision_zero_decay import (  # noqa: E402
    VISION_ZERO_DECAY_DEPTH,
    VISION_ZERO_DECAY_MODEL_KINDS,
    vision_zero_decay_budget_search_specs,
    vision_zero_decay_confirm_specs,
    vision_zero_decay_lr_search_specs,
    vision_zero_decay_trial_count,
)
from dropout_mft.experiments.benchmark.workflow import (  # noqa: E402
    load_selection as _selection,
    write_immutable_manifest,
)
from dropout_mft.provenance import provenance_sha256  # noqa: E402


def command_plan(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    specs = []
    if args.stage == "lr_search":
        for model_kind in VISION_ZERO_DECAY_MODEL_KINDS:
            specs.extend(
                vision_zero_decay_lr_search_specs(model_kind, depth=args.depth)
            )
    elif args.stage == "budget_search":
        lr_selection = _selection(run_dir, "lr_search")
        for model_kind in VISION_ZERO_DECAY_MODEL_KINDS:
            cell = f"tiny_imagenet/{model_kind}"
            if cell not in lr_selection:
                raise SystemExit(f"Missing LR selection for {cell}")
            selected_lrs = {
                profile_id: float(choice["learning_rate"])
                for profile_id, choice in lr_selection[cell].items()
            }
            specs.extend(
                vision_zero_decay_budget_search_specs(
                    model_kind, selected_lrs, depth=args.depth
                )
            )
    else:
        lr_selection = _selection(run_dir, "lr_search")
        budget_selection = _selection(run_dir, "budget_search")
        for model_kind in VISION_ZERO_DECAY_MODEL_KINDS:
            cell = f"tiny_imagenet/{model_kind}"
            if cell not in lr_selection or cell not in budget_selection:
                raise SystemExit(f"Missing confirmation selection for {cell}")
            specs.extend(
                vision_zero_decay_confirm_specs(
                    model_kind,
                    budget_selection[cell],
                    lr_selection[cell],
                    depth=args.depth,
                )
            )

    path, provenance = write_immutable_manifest(run_dir, args.stage, specs)
    print(
        f"stage={args.stage} trials={len(specs)} "
        f"cells={len({spec.cell for spec in specs})}"
    )
    print(f"manifest={path}")
    print(f"provenance_sha256={provenance_sha256(provenance)}")


def command_cost(_args: argparse.Namespace) -> None:
    print("dataset=tiny_imagenet models=mlp,transformer depth=12 epochs=75")
    print("weight_decay=0 dropout_grid=0.05,0.10,0.15,0.20")
    for stage, count in vision_zero_decay_trial_count().items():
        print(f"{stage:14s} {count:5d} trials")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--run-dir", required=True)
    plan.add_argument(
        "--stage", choices=("lr_search", "budget_search", "confirm"), required=True
    )
    plan.add_argument("--depth", type=int, default=VISION_ZERO_DECAY_DEPTH)
    plan.set_defaults(func=command_plan)
    cost = commands.add_parser("cost")
    cost.set_defaults(func=command_cost)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
