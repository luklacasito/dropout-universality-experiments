"""Transformer-only linear-profile and tuned-no-dropout sidecar.

The primary depth-12 pilot is immutable once submitted.  This module defines a
separate 50-trial extension:

* 30 validation-only LR-search trials: two datasets x three profiles x five LRs;
* 20 fresh-seed confirmation trials: the better linear direction and the
  independently tuned no-dropout arm, five paired seeds per dataset.

The existing primary confirmation supplies the paired uniform and LR-matched
no-dropout baselines.  This sidecar never rewrites its manifests or results.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from dropout_mft.experiments.benchmark.protocol import (
    CONFIRM_SEEDS,
    CONTROL_PROFILE_ID,
    LINEAR_PROFILE_IDS,
    LR_GRIDS,
    LR_SEARCH_MEAN_DROPOUT,
    LR_SEARCH_SEEDS,
    SIDECAR_PROFILE_IDS,
    TUNED_CONTROL_PROFILE_ID,
    BenchmarkTrialSpec,
    _spec_defaults,
)
from dropout_mft.experiments.benchmark.workflow import (
    collect_records,
    index_records,
    load_selection,
    paired_comparison,
    save_plan,
    selection_path,
    trial_record,
    write_json_atomic,
)
from dropout_mft.results import load_npz_result
from dropout_mft.training import BenchmarkDatasetName

SIDECAR_COHORT_ID = "transformer-linear-sidecar-v1"
SIDECAR_DATASETS: tuple[BenchmarkDatasetName, ...] = (
    "fi2010",
    "openml_jannis",
)
SIDECAR_DEPTH = 12


def sidecar_lr_search_specs(
    dataset: BenchmarkDatasetName,
    *,
    depth: int = SIDECAR_DEPTH,
) -> list[BenchmarkTrialSpec]:
    """Return the 15 validation-only screening trials for one dataset."""

    if dataset not in SIDECAR_DATASETS:
        raise ValueError(f"Dataset {dataset!r} is not in the Transformer sidecar")
    defaults = _spec_defaults(dataset, "transformer")
    return [
        BenchmarkTrialSpec(
            stage="lr_search",
            profile_id=profile_id,
            mean_dropout=(
                0.0
                if profile_id == TUNED_CONTROL_PROFILE_ID
                else LR_SEARCH_MEAN_DROPOUT
            ),
            max_dropout=0.20,
            learning_rate=learning_rate,
            seed=seed,
            depth=depth,
            evaluate_test=False,
            cohort_id=SIDECAR_COHORT_ID,
            **defaults,
        )
        for profile_id in SIDECAR_PROFILE_IDS
        for learning_rate in LR_GRIDS["transformer"]
        for seed in LR_SEARCH_SEEDS
    ]


def select_sidecar_records(records: Iterable[dict]) -> dict[str, dict]:
    """Select each profile's LR, then the better linear direction per cell."""

    grouped: dict[str, dict[str, list[dict]]] = {}
    for record in records:
        if record["model_kind"] != "transformer":
            raise ValueError("The linear sidecar only accepts Transformer records")
        grouped.setdefault(record["cell"], {}).setdefault(
            record["profile_id"], []
        ).append(record)

    selection: dict[str, dict] = {}
    for cell, profiles in sorted(grouped.items()):
        missing = set(SIDECAR_PROFILE_IDS) - set(profiles)
        if missing:
            raise ValueError(f"Cell {cell!r} is missing profiles: {sorted(missing)!r}")

        best_by_profile: dict[str, dict] = {}
        for profile_id in SIDECAR_PROFILE_IDS:
            candidates = profiles[profile_id]
            winner = min(candidates, key=lambda row: float(row["validation_loss"]))
            best_by_profile[profile_id] = {
                "profile_id": profile_id,
                "learning_rate": float(winner["learning_rate"]),
                "mean_dropout": float(winner["mean_dropout"]),
                "validation_loss": float(winner["validation_loss"]),
                "validation_accuracy": float(winner["validation_accuracy"]),
                "seed": int(winner["seed"]),
                "criterion": "minimum_validation_loss_v1",
            }

        linear = min(
            (best_by_profile[profile_id] for profile_id in LINEAR_PROFILE_IDS),
            key=lambda row: row["validation_loss"],
        )
        selection[cell] = {
            "selected_linear": linear,
            "linear_candidates": {
                profile_id: best_by_profile[profile_id]
                for profile_id in LINEAR_PROFILE_IDS
            },
            TUNED_CONTROL_PROFILE_ID: best_by_profile[TUNED_CONTROL_PROFILE_ID],
        }
    return selection


def sidecar_confirm_specs(
    dataset: BenchmarkDatasetName,
    selected: dict,
    *,
    depth: int = SIDECAR_DEPTH,
) -> list[BenchmarkTrialSpec]:
    """Return ten fresh-seed trials for one selected Transformer cell."""

    if dataset not in SIDECAR_DATASETS:
        raise ValueError(f"Dataset {dataset!r} is not in the Transformer sidecar")
    defaults = _spec_defaults(dataset, "transformer")
    choices = (selected["selected_linear"], selected[TUNED_CONTROL_PROFILE_ID])
    specs: list[BenchmarkTrialSpec] = []
    for choice in choices:
        profile_id = str(choice["profile_id"])
        expected_dropout = profile_id != TUNED_CONTROL_PROFILE_ID
        if expected_dropout != (float(choice["mean_dropout"]) > 0):
            raise ValueError(f"Invalid selected dropout budget for {profile_id!r}")
        for seed in CONFIRM_SEEDS:
            specs.append(
                BenchmarkTrialSpec(
                    stage="confirm",
                    profile_id=profile_id,
                    mean_dropout=float(choice["mean_dropout"]),
                    max_dropout=0.20,
                    learning_rate=float(choice["learning_rate"]),
                    seed=seed,
                    depth=depth,
                    evaluate_test=True,
                    cohort_id=SIDECAR_COHORT_ID,
                    **defaults,
                )
            )
    return specs


def sidecar_trial_count() -> dict[str, int]:
    """Exact fixed-budget trial count for the two-dataset sidecar."""

    lr_search = sum(
        len(sidecar_lr_search_specs(dataset)) for dataset in SIDECAR_DATASETS
    )
    dummy_selection = {
        "selected_linear": {
            "profile_id": "linear_early",
            "learning_rate": 1e-4,
            "mean_dropout": 0.10,
        },
        TUNED_CONTROL_PROFILE_ID: {
            "profile_id": TUNED_CONTROL_PROFILE_ID,
            "learning_rate": 1e-4,
            "mean_dropout": 0.0,
        },
    }
    confirm = sum(
        len(sidecar_confirm_specs(dataset, dummy_selection))
        for dataset in SIDECAR_DATASETS
    )
    return {"lr_search": lr_search, "confirm": confirm, "total": lr_search + confirm}


def _collect_manifest(run_dir: Path, stage: str) -> list[dict]:
    return collect_records(run_dir, stage, require_complete=True)


def _record(spec: BenchmarkTrialSpec, result: dict, path: Path) -> dict:
    return trial_record(spec, result, path)


def command_plan(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    if args.stage == "lr_search":
        specs = [
            spec
            for dataset in SIDECAR_DATASETS
            for spec in sidecar_lr_search_specs(dataset, depth=args.depth)
        ]
    else:
        selection = load_selection(run_dir, "lr_search")
        specs = []
        for dataset in SIDECAR_DATASETS:
            cell = f"{dataset}/transformer"
            if cell not in selection:
                raise SystemExit(f"Missing sidecar selection for {cell}")
            specs.extend(
                sidecar_confirm_specs(dataset, selection[cell], depth=args.depth)
            )

    save_plan(run_dir, args.stage, specs)


def command_select(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    records = _collect_manifest(run_dir, "lr_search")
    selection = select_sidecar_records(records)
    path = selection_path(run_dir, "lr_search")
    write_json_atomic(path, selection)
    for cell, choices in sorted(selection.items()):
        linear = choices["selected_linear"]
        none = choices[TUNED_CONTROL_PROFILE_ID]
        print(
            f"{cell:34s} linear={linear['profile_id']:12s} "
            f"lr={linear['learning_rate']:.2e} val_loss={linear['validation_loss']:.4f}"
        )
        print(
            f"{cell:34s} none_tuned   lr={none['learning_rate']:.2e} "
            f"val_loss={none['validation_loss']:.4f}"
        )
    print(f"selection={path}")


def _baseline_records(run_dir: Path) -> list[dict]:
    records: list[dict] = []
    pattern = "trials/*/transformer/confirm/*.npz"
    for path in sorted(run_dir.glob(pattern)):
        result = load_npz_result(path)
        factors = result.get("factors", {})
        if factors.get("profile_id") not in {"uniform", CONTROL_PROFILE_ID}:
            continue
        if int(factors.get("depth", -1)) != 12:
            continue
        spec = BenchmarkTrialSpec(**factors)
        records.append(_record(spec, result, path))
    return records


def _index(records: list[dict]) -> dict:
    return index_records(records, group="dataset")


def command_aggregate(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    sidecar = _collect_manifest(run_dir, "confirm")
    baseline = _baseline_records(Path(args.baseline_run_dir))
    by_dataset = _index([*baseline, *sidecar])
    expected_seeds = set(CONFIRM_SEEDS)
    output: dict[str, dict] = {}

    for dataset in SIDECAR_DATASETS:
        profiles = by_dataset.get(dataset, {})
        linear_profiles = [
            profile
            for profile in profiles
            if profile in {"linear_early", "linear_late"}
        ]
        if len(linear_profiles) != 1:
            raise ValueError(
                f"Expected one selected linear profile for {dataset}, got {linear_profiles}"
            )
        linear_profile = linear_profiles[0]
        required = {
            "uniform",
            CONTROL_PROFILE_ID,
            TUNED_CONTROL_PROFILE_ID,
            linear_profile,
        }
        missing = required - set(profiles)
        if missing:
            raise ValueError(f"Missing {dataset} profiles: {sorted(missing)!r}")
        for profile in required:
            seeds = set(profiles[profile])
            if seeds != expected_seeds:
                raise ValueError(
                    f"{dataset}/{profile} seeds {sorted(seeds)} != {sorted(expected_seeds)}"
                )
            if not all(profiles[profile][seed]["test_evaluated"] for seed in seeds):
                raise ValueError(f"{dataset}/{profile} lacks locked test metrics")

        split_hashes = {
            profiles[profile][seed]["split_hash"]
            for profile in required
            for seed in expected_seeds
        }
        if len(split_hashes) != 1:
            raise ValueError(f"{dataset} sidecar/baseline split hashes disagree")

        arms: dict[str, dict] = {}
        for profile in sorted(required):
            losses = np.asarray(
                [profiles[profile][seed]["test_loss"] for seed in CONFIRM_SEEDS],
                dtype=float,
            )
            accuracies = np.asarray(
                [profiles[profile][seed]["test_accuracy"] for seed in CONFIRM_SEEDS],
                dtype=float,
            )
            arms[profile] = {
                "mean_test_loss": float(losses.mean()),
                "mean_test_accuracy": float(accuracies.mean()),
                "test_losses": losses.tolist(),
                "test_accuracies": accuracies.tolist(),
            }

        comparisons: dict[str, dict] = {}
        pairs = (
            (linear_profile, "uniform"),
            (TUNED_CONTROL_PROFILE_ID, "uniform"),
            (CONTROL_PROFILE_ID, "uniform"),
            (linear_profile, TUNED_CONTROL_PROFILE_ID),
        )
        for candidate, reference in pairs:
            comparison = paired_comparison(arms[candidate], arms[reference])
            comparison.pop("paired_seed_win_rate_loss")
            comparison["test_loss_delta_95ci"] = list(
                comparison["test_loss_delta_95ci"]
            )
            comparison["test_accuracy_delta_pp_95ci"] = list(
                comparison["test_accuracy_delta_pp_95ci"]
            )
            comparisons[f"{candidate}_vs_{reference}"] = comparison

        output[dataset] = {
            "linear_profile": linear_profile,
            "split_hash": split_hashes.pop(),
            "seeds": list(CONFIRM_SEEDS),
            "arms": arms,
            "comparisons": comparisons,
        }

        print(f"\n{dataset}: selected {linear_profile}")
        for profile, arm in sorted(arms.items()):
            print(
                f"  {profile:12s} test_loss={arm['mean_test_loss']:.4f} "
                f"test_acc={100.0 * arm['mean_test_accuracy']:.2f}%"
            )
        for label, comparison in comparisons.items():
            print(
                f"  {label:29s} dloss={comparison['mean_test_loss_delta']:+.4f} "
                f"dacc={comparison['mean_test_accuracy_delta_pp']:+.2f}pp"
            )

    path = run_dir / "summary" / "linear_sidecar_summary.json"
    write_json_atomic(path, output)
    print(f"\nsummary={path}")


def command_cost(_args: argparse.Namespace) -> None:
    for stage, count in sidecar_trial_count().items():
        print(f"{stage:12s} {count:4d} trials")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan")
    plan.add_argument("--run-dir", required=True)
    plan.add_argument("--stage", required=True, choices=("lr_search", "confirm"))
    plan.add_argument("--depth", type=int, default=12)
    plan.set_defaults(func=command_plan)

    select = commands.add_parser("select")
    select.add_argument("--run-dir", required=True)
    select.set_defaults(func=command_select)

    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--run-dir", required=True)
    aggregate.add_argument("--baseline-run-dir", required=True)
    aggregate.set_defaults(func=command_aggregate)

    cost = commands.add_parser("cost")
    cost.set_defaults(func=command_cost)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
