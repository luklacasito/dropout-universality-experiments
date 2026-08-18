#!/usr/bin/env python
"""Plan one immutable sample-size/zero-decay dropout cohort per run directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dropout_mft.experiments.benchmark.data_regimes import (  # noqa: E402
    DATA_REGIME_DEPTH,
    DATA_REGIME_IDS,
    DATA_REGIME_MODEL_KINDS,
    data_regime,
    data_regime_budget_search_specs,
    data_regime_confirm_specs,
    data_regime_lr_search_specs,
    data_regime_trial_count,
)
from dropout_mft.experiments.benchmark.workflow import (  # noqa: E402
    load_selection as _load_selection,
    write_immutable_manifest,
)
from dropout_mft.provenance import provenance_sha256  # noqa: E402


def command_plan(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    regime = data_regime(args.regime)
    specs = []
    if args.stage == "lr_search":
        for model_kind in DATA_REGIME_MODEL_KINDS:
            specs.extend(data_regime_lr_search_specs(args.regime, model_kind))
    elif args.stage == "budget_search":
        lr_selection = _load_selection(run_dir, "lr_search")
        for model_kind in DATA_REGIME_MODEL_KINDS:
            cell = f"{regime.dataset}/{model_kind}"
            if cell not in lr_selection:
                raise SystemExit(f"Missing LR selection for {cell}")
            selected_lrs = {
                profile_id: float(choice["learning_rate"])
                for profile_id, choice in lr_selection[cell].items()
            }
            specs.extend(
                data_regime_budget_search_specs(args.regime, model_kind, selected_lrs)
            )
    else:
        lr_selection = _load_selection(run_dir, "lr_search")
        budget_selection = _load_selection(run_dir, "budget_search")
        for model_kind in DATA_REGIME_MODEL_KINDS:
            cell = f"{regime.dataset}/{model_kind}"
            if cell not in lr_selection or cell not in budget_selection:
                raise SystemExit(f"Missing confirmation selection for {cell}")
            specs.extend(
                data_regime_confirm_specs(
                    args.regime,
                    model_kind,
                    budget_selection[cell],
                    lr_selection[cell],
                )
            )

    path, provenance = write_immutable_manifest(run_dir, args.stage, specs)
    print(
        f"regime={args.regime} stage={args.stage} trials={len(specs)} "
        f"cells={len({spec.cell for spec in specs})}"
    )
    print(f"manifest={path}")
    print(f"provenance_sha256={provenance_sha256(provenance)}")


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


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
