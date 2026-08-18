#!/usr/bin/env python
"""Plan and aggregate the 100-epoch, ten-seed Jannis Transformer follow-up."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dropout_mft.benchmark_jannis_100 import (  # noqa: E402
    JANNIS_100_PROFILES,
    JANNIS_100_SEEDS,
    jannis_100epoch_specs,
)
from dropout_mft.benchmark_suite import (  # noqa: E402
    read_benchmark_manifest,
    trial_checkpoint_path,
    trial_output_path,
    write_benchmark_manifest,
)
from dropout_mft.provenance import provenance_sha256, sha256_file  # noqa: E402
from dropout_mft.results import load_npz_result  # noqa: E402
from dropout_mft.scale_transfer import _provenance  # noqa: E402


BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260812


def _manifest_path(run_dir: Path) -> Path:
    return run_dir / "manifests" / "confirm.jsonl"


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def command_plan(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    path = _manifest_path(run_dir)
    if path.exists():
        raise SystemExit(f"Refusing to overwrite immutable manifest: {path}")
    specs = jannis_100epoch_specs()
    provenance = _provenance("planning", None)
    write_benchmark_manifest(path, specs, provenance=provenance)
    print(f"stage=confirm trials={len(specs)} cells=1")
    print(f"manifest={path}")
    print(f"provenance_sha256={provenance_sha256(provenance)}")


def _paired_ci(values: np.ndarray) -> list[float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = rng.choice(
        values, size=(BOOTSTRAP_RESAMPLES, len(values)), replace=True
    ).mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def command_aggregate(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    specs = read_benchmark_manifest(_manifest_path(run_dir))
    by_profile: dict[str, dict[int, dict]] = {}
    for spec in specs:
        result_path = trial_output_path(run_dir, spec)
        checkpoint_path = trial_checkpoint_path(run_dir, spec)
        if not result_path.is_file() or not checkpoint_path.is_file():
            raise SystemExit(
                f"Incomplete trial {spec.trial_id}: result/checkpoint missing"
            )
        result = load_npz_result(result_path)
        checkpoint = result.get("checkpoint", {})
        if checkpoint.get("sha256") != sha256_file(checkpoint_path):
            raise ValueError(f"Checkpoint hash mismatch: {checkpoint_path}")
        by_profile.setdefault(spec.profile_id, {})[spec.seed] = result

    expected_seeds = set(JANNIS_100_SEEDS)
    for profile_id in JANNIS_100_PROFILES:
        if set(by_profile.get(profile_id, {})) != expected_seeds:
            raise ValueError(f"Incomplete seed set for {profile_id}")

    arms: dict[str, dict] = {}
    for profile_id in JANNIS_100_PROFILES:
        rows = by_profile[profile_id]
        losses = np.asarray(
            [rows[seed]["test"]["loss"] for seed in JANNIS_100_SEEDS], dtype=float
        )
        accuracies = np.asarray(
            [rows[seed]["test"]["accuracy"] for seed in JANNIS_100_SEEDS],
            dtype=float,
        )
        selected_epochs = [
            int(rows[seed]["selection"]["selected_epoch"])
            for seed in JANNIS_100_SEEDS
        ]
        arms[profile_id] = {
            "test_loss_mean": float(losses.mean()),
            "test_loss_sd": float(losses.std(ddof=1)),
            "test_accuracy_mean": float(accuracies.mean()),
            "test_accuracy_sd": float(accuracies.std(ddof=1)),
            "test_losses": losses.tolist(),
            "test_accuracies": accuracies.tolist(),
            "selected_epochs": selected_epochs,
            "selected_epoch_mean": float(np.mean(selected_epochs)),
        }

    comparisons: dict[str, dict] = {}
    baseline = arms["uniform"]
    for profile_id in JANNIS_100_PROFILES:
        if profile_id == "uniform":
            continue
        loss_delta = np.asarray(arms[profile_id]["test_losses"]) - np.asarray(
            baseline["test_losses"]
        )
        accuracy_delta = 100.0 * (
            np.asarray(arms[profile_id]["test_accuracies"])
            - np.asarray(baseline["test_accuracies"])
        )
        comparisons[f"{profile_id}_vs_uniform"] = {
            "mean_test_loss_delta": float(loss_delta.mean()),
            "test_loss_delta_95ci": _paired_ci(loss_delta),
            "mean_test_accuracy_delta_pp": float(accuracy_delta.mean()),
            "test_accuracy_delta_pp_95ci": _paired_ci(accuracy_delta),
            "paired_seed_win_rate_loss": float((loss_delta < 0).mean()),
        }

    summary = {
        "protocol": "jannis_transformer_depth12_100epoch_10seed_v1",
        "seeds": list(JANNIS_100_SEEDS),
        "arms": arms,
        "comparisons": comparisons,
    }
    path = run_dir / "summary" / "confirm_summary.json"
    _write_json_atomic(path, summary)

    print("\nprofile       test loss      test acc   selected epoch")
    for profile_id in JANNIS_100_PROFILES:
        arm = arms[profile_id]
        print(
            f"{profile_id:12s} {arm['test_loss_mean']:.4f}±{arm['test_loss_sd']:.4f} "
            f"{100 * arm['test_accuracy_mean']:.2f}%±"
            f"{100 * arm['test_accuracy_sd']:.2f}  "
            f"{arm['selected_epoch_mean']:.1f}"
        )
    print("\nPaired deltas versus uniform (negative loss / positive accuracy wins):")
    for label, comparison in comparisons.items():
        print(
            f"{label:29s} dloss={comparison['mean_test_loss_delta']:+.4f} "
            f"dacc={comparison['mean_test_accuracy_delta_pp']:+.2f}pp"
        )
    print(f"summary={path}")


def command_cost(_args: argparse.Namespace) -> None:
    print("dataset=openml_jannis model=transformer depth=12 epochs=100")
    print(f"profiles={len(JANNIS_100_PROFILES)} seeds={len(JANNIS_100_SEEDS)}")
    print(f"total={len(jannis_100epoch_specs())} trials")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--run-dir", required=True)
    plan.set_defaults(func=command_plan)
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--run-dir", required=True)
    aggregate.set_defaults(func=command_aggregate)
    cost = commands.add_parser("cost")
    cost.set_defaults(func=command_cost)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
