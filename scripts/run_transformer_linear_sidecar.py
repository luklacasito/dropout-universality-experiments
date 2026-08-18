#!/usr/bin/env python
"""Plan, select, and aggregate the depth-12 Transformer linear sidecar."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dropout_mft.benchmark_sidecar import (  # noqa: E402
    SIDECAR_DATASETS,
    TUNED_CONTROL_PROFILE_ID,
    select_sidecar_records,
    sidecar_confirm_specs,
    sidecar_lr_search_specs,
    sidecar_trial_count,
)
from dropout_mft.benchmark_suite import (  # noqa: E402
    CONFIRM_SEEDS,
    CONTROL_PROFILE_ID,
    BenchmarkTrialSpec,
    read_benchmark_manifest,
    trial_output_path,
    write_benchmark_manifest,
)
from dropout_mft.provenance import provenance_sha256  # noqa: E402
from dropout_mft.results import load_npz_result  # noqa: E402
from dropout_mft.scale_transfer import _provenance  # noqa: E402


BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260812


def manifest_path(run_dir: Path, stage: str) -> Path:
    return run_dir / "manifests" / f"{stage}.jsonl"


def selection_path(run_dir: Path) -> Path:
    return run_dir / "selections" / "lr_search.json"


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _collect_manifest(run_dir: Path, stage: str) -> list[dict]:
    path = manifest_path(run_dir, stage)
    if not path.exists():
        raise SystemExit(f"Missing manifest {path}")
    records: list[dict] = []
    missing: list[str] = []
    for spec in read_benchmark_manifest(path):
        output = trial_output_path(run_dir, spec)
        if not output.exists():
            missing.append(spec.trial_id)
            continue
        result = load_npz_result(output)
        if result["trial"]["trial_id"] != spec.trial_id:
            raise ValueError(f"Trial/result mismatch: {output}")
        records.append(_record(spec, result, output))
    if missing:
        raise SystemExit(
            f"Stage {stage} is incomplete: {len(missing)} missing trials; "
            f"first={missing[0]}"
        )
    return records


def _record(spec: BenchmarkTrialSpec, result: dict, path: Path) -> dict:
    return {
        "path": str(path),
        "trial_id": spec.trial_id,
        "cell": spec.cell,
        "dataset": spec.dataset,
        "model_kind": spec.model_kind,
        "profile_id": spec.profile_id,
        "learning_rate": spec.learning_rate,
        "mean_dropout": spec.mean_dropout,
        "seed": spec.seed,
        "validation_loss": float(result["selection"]["validation_loss"]),
        "validation_accuracy": float(result["selection"]["validation_accuracy"]),
        "test_evaluated": bool(result["test"]["evaluated"]),
        "test_loss": result["test"]["loss"],
        "test_accuracy": result["test"]["accuracy"],
        "split_hash": result["data"]["split_hash"],
        "split_protocol": result["data"]["split_protocol"],
    }


def command_plan(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    if args.stage == "lr_search":
        specs = [
            spec
            for dataset in SIDECAR_DATASETS
            for spec in sidecar_lr_search_specs(dataset, depth=args.depth)
        ]
    else:
        path = selection_path(run_dir)
        if not path.exists():
            raise SystemExit(f"Missing selection {path}")
        selection = json.loads(path.read_text())
        specs = []
        for dataset in SIDECAR_DATASETS:
            cell = f"{dataset}/transformer"
            if cell not in selection:
                raise SystemExit(f"Missing sidecar selection for {cell}")
            specs.extend(
                sidecar_confirm_specs(dataset, selection[cell], depth=args.depth)
            )

    provenance = _provenance("planning", None)
    path = manifest_path(run_dir, args.stage)
    write_benchmark_manifest(path, specs, provenance=provenance)
    print(
        f"stage={args.stage} trials={len(specs)} cells={len({s.cell for s in specs})}"
    )
    print(f"manifest={path}")
    print(f"provenance_sha256={provenance_sha256(provenance)}")


def command_select(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    records = _collect_manifest(run_dir, "lr_search")
    selection = select_sidecar_records(records)
    path = selection_path(run_dir)
    _write_json_atomic(path, selection)
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


def _paired_ci(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.choice(
        values, size=(BOOTSTRAP_RESAMPLES, len(values)), replace=True
    ).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _index(records: list[dict]) -> dict[str, dict[str, dict[int, dict]]]:
    indexed: dict[str, dict[str, dict[int, dict]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for record in records:
        dataset = record["dataset"]
        profile = record["profile_id"]
        seed = int(record["seed"])
        if seed in indexed[dataset][profile]:
            raise ValueError(f"Duplicate {dataset}/{profile}/seed={seed}")
        indexed[dataset][profile][seed] = record
    return indexed


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
            loss_delta = np.asarray(arms[candidate]["test_losses"]) - np.asarray(
                arms[reference]["test_losses"]
            )
            accuracy_delta = 100.0 * (
                np.asarray(arms[candidate]["test_accuracies"])
                - np.asarray(arms[reference]["test_accuracies"])
            )
            comparisons[f"{candidate}_vs_{reference}"] = {
                "mean_test_loss_delta": float(loss_delta.mean()),
                "test_loss_delta_95ci": list(_paired_ci(loss_delta)),
                "mean_test_accuracy_delta_pp": float(accuracy_delta.mean()),
                "test_accuracy_delta_pp_95ci": list(_paired_ci(accuracy_delta)),
            }

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
    _write_json_atomic(path, output)
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


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
