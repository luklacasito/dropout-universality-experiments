#!/usr/bin/env python
"""Plan the immutable depth-12 Tiny ImageNet all-schedule cohort."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dropout_mft.experiments.benchmark.vision import (  # noqa: E402
    VISION_MODEL_KINDS,
    vision_confirm_specs,
    vision_lr_search_specs,
    vision_trial_count,
)
from dropout_mft.experiments.benchmark.workflow import (  # noqa: E402
    load_selection,
    write_immutable_manifest,
)
from dropout_mft.provenance import provenance_sha256  # noqa: E402


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

    path, provenance = write_immutable_manifest(run_dir, args.stage, specs)
    print(
        f"stage={args.stage} trials={len(specs)} "
        f"cells={len({spec.cell for spec in specs})}"
    )
    print(f"manifest={path}")
    print(f"provenance_sha256={provenance_sha256(provenance)}")


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


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
