"""Pre-registered 100-epoch, ten-seed Jannis Transformer confirmation.

The original depth-12 cohort trains for 50 epochs.  Extending a cosine schedule
to 100 epochs changes every learning-rate trajectory, so this cohort reruns all
ten seeds under the longer horizon instead of pooling incompatible protocols.
The six arms use the validation-selected learning rate from the preceding
screens and differ only in their fixed layerwise dropout profile.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dropout_mft.experiments.benchmark.protocol import (
    BenchmarkTrialSpec,
    _spec_defaults,
)
from dropout_mft.experiments.benchmark.workflow import (
    load_completed_trials,
    paired_comparison,
    save_plan,
    write_json_atomic,
)

JANNIS_100_COHORT_ID = "jannis-transformer-depth12-100epoch-10seed-v1"
JANNIS_100_DEPTH = 12
JANNIS_100_EPOCHS = 100
JANNIS_100_SEEDS = tuple(range(100, 110))
JANNIS_100_PROFILES = (
    "uniform",
    "step_early",
    "big_step",
    "linear_early",
    "linear_late",
    "none_tuned",
)

# All six previous validation-only screens selected the same initial LR.
JANNIS_100_SELECTED_LR = {profile: 1e-4 for profile in JANNIS_100_PROFILES}


def jannis_100epoch_specs() -> list[BenchmarkTrialSpec]:
    """Return 60 paired confirmation trials with retained best checkpoints."""

    defaults = _spec_defaults("openml_jannis", "transformer")
    defaults["epochs"] = JANNIS_100_EPOCHS
    specs: list[BenchmarkTrialSpec] = []
    for profile_id in JANNIS_100_PROFILES:
        for seed in JANNIS_100_SEEDS:
            specs.append(
                BenchmarkTrialSpec(
                    stage="confirm",
                    profile_id=profile_id,
                    mean_dropout=0.0 if profile_id == "none_tuned" else 0.10,
                    max_dropout=0.30 if profile_id == "big_step" else 0.20,
                    learning_rate=JANNIS_100_SELECTED_LR[profile_id],
                    seed=seed,
                    depth=JANNIS_100_DEPTH,
                    evaluate_test=True,
                    cohort_id=JANNIS_100_COHORT_ID,
                    **defaults,
                )
            )
    return specs


def jannis_100epoch_trial_count() -> int:
    return len(jannis_100epoch_specs())


def command_plan(args: argparse.Namespace) -> None:
    save_plan(args.run_dir, "confirm", jannis_100epoch_specs())


def command_aggregate(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    by_profile: dict[str, dict[int, dict]] = {}
    for spec, result, _path in load_completed_trials(
        run_dir, "confirm", require_complete=True, require_checkpoint=True
    ):
        arm = by_profile.setdefault(spec.profile_id, {})
        if spec.seed in arm:
            raise ValueError(f"Duplicate {spec.profile_id}/seed={spec.seed}")
        arm[spec.seed] = result

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
            int(rows[seed]["selection"]["selected_epoch"]) for seed in JANNIS_100_SEEDS
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
        comparisons[f"{profile_id}_vs_uniform"] = paired_comparison(
            arms[profile_id], baseline
        )

    summary = {
        "protocol": "jannis_transformer_depth12_100epoch_10seed_v1",
        "seeds": list(JANNIS_100_SEEDS),
        "arms": arms,
        "comparisons": comparisons,
    }
    path = run_dir / "summary" / "confirm_summary.json"
    write_json_atomic(path, summary)

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


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
