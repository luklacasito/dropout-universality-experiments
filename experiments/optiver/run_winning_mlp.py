#!/usr/bin/env python
"""Plan, run, and summarize shallow/deep schedules on winning Optiver features."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dropout_mft.experiments.optiver.winning_mlp import (  # noqa: E402
    WinningMLPTrial,
    DEEP_PROFILE_IDS,
    EPOCH_PROBE_PROTOCOL,
    SHALLOW_PROFILE_IDS,
    confirmation_trials,
    epoch_probe_trials,
    load_feature_cache,
    run_trial,
    save_result,
    select_dropout_budgets,
    selection_trials,
    unique_profiles,
)


def command_plan(args) -> None:
    trials = selection_trials(seeds=tuple(range(args.seeds)))
    path = Path(args.manifest)
    if path.exists():
        raise SystemExit(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(asdict(trial), sort_keys=True) + "\n" for trial in trials)
    )
    print(f"wrote {path} trials={len(trials)}")
    print("selection uses validation only; no final-fold test is inspected")
    print("exact_shallow: 4 unique shapes; 30 trials at three seeds")
    print("matched_depth12: 6 profiles; 63 trials at three seeds")


def command_plan_epoch_probe(args) -> None:
    trials = epoch_probe_trials()
    path = Path(args.manifest)
    if path.exists():
        raise SystemExit(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(asdict(trial), sort_keys=True) + "\n" for trial in trials)
    )
    print(f"wrote {path} trials={len(trials)}")
    print("paired epoch budgets: 30=50, 100=50; seeds=100..104")
    print(f"evaluation_protocol={EPOCH_PROBE_PROTOCOL}")


def command_run(args) -> None:
    rows = [
        json.loads(line)
        for line in Path(args.manifest).read_text().splitlines()
        if line.strip()
    ]
    if not 0 <= args.index < len(rows):
        raise SystemExit(f"index {args.index} outside manifest with {len(rows)} rows")
    trial = WinningMLPTrial(**rows[args.index])
    output = Path(args.output_dir) / f"{trial.trial_id}.npz"
    if trial.stage == "epoch_probe" and not args.checkpoint_dir:
        raise SystemExit("epoch_probe requires --checkpoint-dir")
    checkpoint = (
        Path(args.checkpoint_dir) / f"{trial.trial_id}.pt"
        if trial.stage == "epoch_probe"
        else None
    )
    marker = (
        Path(args.wandb_dir) / "tracked" / f"{trial.trial_id}.json"
        if args.wandb_mode != "disabled"
        else None
    )
    files_exist = output.exists() and (checkpoint is None or checkpoint.exists())
    if files_exist and (marker is None or marker.exists()):
        print(f"exists {output}")
        return
    if files_exist:
        with np.load(output, allow_pickle=True) as payload:
            result = payload["payload"].item()
    else:
        data = load_feature_cache(args.cache)
        if trial.stage == "selection":
            train_indices = data["selection_train_indices"]
            validation_indices = data["selection_validation_indices"]
            test_indices = None
        elif trial.stage == "confirmation":
            train_indices = data["confirmation_train_indices"]
            validation_indices = data["test_indices"]
            test_indices = None
        else:
            train_indices = data["selection_train_indices"]
            validation_indices = data["selection_validation_indices"]
            test_indices = data["test_indices"]
        result = run_trial(
            trial,
            numeric=data["numeric"],
            stock_ids=data["stock_ids"],
            targets=data["targets"],
            train_indices=train_indices,
            validation_indices=validation_indices,
            test_indices=test_indices,
            checkpoint_path=checkpoint,
            device=args.device,
        )
        save_result(output, result)
    if marker is not None:
        if not args.wandb_project:
            raise SystemExit("W&B tracking requires --wandb-project or WANDB_PROJECT")
        _log_wandb(args, trial, result, output, checkpoint, marker)
    score = (
        f"val={result['best_validation_rmspe']:.6f} test={result['test_rmspe']:.6f}"
        if trial.stage == "epoch_probe"
        else f"best_rmspe={result['best_validation_rmspe']:.6f}"
    )
    print(
        f"{trial.architecture} {trial.profile_id} p={trial.mean_dropout:.3f} "
        f"seed={trial.seed} epochs={trial.epochs} {score}"
    )
    print(output)


def _log_wandb(
    args, trial, result, output: Path, checkpoint: Path | None, marker: Path
) -> None:
    import wandb

    wandb_dir = Path(args.wandb_dir)
    wandb_dir.mkdir(parents=True, exist_ok=True)
    probe = trial.stage == "epoch_probe"
    stage_group = f"epoch_probe_{trial.epochs}" if probe else trial.stage
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        id=trial.trial_id,
        name=(
            f"optiver-{trial.architecture}-{trial.profile_id}-"
            f"{'e' + str(trial.epochs) + '-' if probe else ''}"
            f"s{trial.seed}-{trial.trial_id[:8]}"
        ),
        group=(f"{args.wandb_run_group}/{stage_group}/optiver/{trial.architecture}"),
        job_type=trial.stage,
        tags=[
            "optiver",
            "winning-features",
            trial.stage,
            *(["corrected-final-test", f"epochs-{trial.epochs}"] if probe else []),
            trial.architecture,
            trial.profile_id,
        ],
        config={
            **trial.__dict__,
            "trial_id": trial.trial_id,
            "dropout_layers": np.asarray(result["dropout_layers"]).tolist(),
            "hidden_width": result["hidden_width"],
            "parameter_count": result["parameter_count"],
            "preprocessing_protocol": result["preprocessing_protocol"],
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
        dir=str(wandb_dir),
        mode=args.wandb_mode,
        resume="allow" if args.wandb_mode == "online" else None,
    )
    try:
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("validation/*", step_metric="epoch")
        train = np.asarray(result["history"]["train_rmspe"])
        validation = np.asarray(result["history"]["validation_rmspe"])
        for epoch, (train_rmspe, validation_rmspe) in enumerate(
            zip(train, validation, strict=True)
        ):
            run.log(
                {
                    "epoch": epoch,
                    "train/rmspe": float(train_rmspe),
                    "validation/rmspe": float(validation_rmspe),
                }
            )
        if probe:
            run.summary.update(
                {
                    "validation/best_rmspe": result["best_validation_rmspe"],
                    "validation/best_epoch": result["best_epoch"],
                    "test/fixed_checkpoint_rmspe": result["test_rmspe"],
                    "compute/parameter_count": result["parameter_count"],
                }
            )
        else:
            metric_namespace = "test" if trial.stage == "confirmation" else "selection"
            run.summary.update(
                {
                    f"{metric_namespace}/best_rmspe": result["best_validation_rmspe"],
                    f"{metric_namespace}/best_epoch": result["best_epoch"],
                    "compute/parameter_count": result["parameter_count"],
                }
            )
        artifact = wandb.Artifact(
            f"optiver-winning-mlp-{trial.trial_id}",
            type="optiver-trial",
            metadata={"trial_id": trial.trial_id, "stage": trial.stage},
        )
        artifact.add_file(str(output), name=output.name)
        if checkpoint is not None:
            artifact.add_file(str(checkpoint), name=checkpoint.name)
        run.log_artifact(artifact, aliases=[trial.stage])
        offline_path = Path(run.dir).parent if run.dir else None
        run.finish(exit_code=0)
    except BaseException:
        run.finish(exit_code=1)
        raise
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "status": "complete",
                "trial_id": trial.trial_id,
                "project": args.wandb_project,
                "entity": args.wandb_entity,
                "mode": args.wandb_mode,
                "offline_sync_path": (
                    str(offline_path)
                    if args.wandb_mode == "offline" and offline_path
                    else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    os.replace(temporary, marker)


def _load_results(directory: Path) -> list[dict]:
    results = []
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=True) as payload:
            results.append(payload["payload"].item())
    return results


def command_plan_confirmation(args) -> None:
    selection = select_dropout_budgets(_load_results(Path(args.selection_dir)))
    path = Path(args.manifest)
    if path.exists():
        raise SystemExit(f"Refusing to overwrite {path}")
    trials = confirmation_trials(selection)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(asdict(trial), sort_keys=True) + "\n" for trial in trials)
    )
    Path(args.selection_json).write_text(
        json.dumps(selection, indent=2, sort_keys=True)
    )
    print(f"wrote {path} trials={len(trials)}")
    print(f"wrote {args.selection_json}")
    print("final fold remains untouched until these 50 fresh-seed trials")


def command_profiles(_args) -> None:
    shallow_names = (
        "none",
        "uniform",
        "step_early",
        "big_step",
        "linear_early",
        "linear_late",
    )
    for depth, mean in ((2, 0.10), (12, 0.10)):
        print(f"depth={depth} mean_dropout={mean}")
        for layers, aliases in unique_profiles(
            shallow_names, depth=depth, mean_dropout=mean
        ).items():
            print(f"  {aliases}: {layers}")


def command_status(args) -> None:
    run_dir = Path(args.run_dir)
    for stage, expected in (
        ("selection", 93),
        ("confirmation", 50),
        ("epoch_probe", 100),
    ):
        manifest = run_dir / "manifests" / f"{stage}.jsonl"
        output = run_dir / stage
        planned = (
            len([line for line in manifest.read_text().splitlines() if line.strip()])
            if manifest.is_file()
            else 0
        )
        finished = len(list(output.glob("*.npz"))) if output.is_dir() else 0
        suffix = f"/{planned}" if planned else f"/{expected} (not planned)"
        print(f"{stage:12s} {finished:3d}{suffix}")


def command_aggregate(args) -> None:
    run_dir = Path(args.run_dir)
    results = _load_results(run_dir / "confirmation")
    if len(results) != 50:
        raise SystemExit(f"Expected 50 confirmation results, found {len(results)}")
    trial_ids = [result["trial_id"] for result in results]
    if len(set(trial_ids)) != len(trial_ids):
        raise SystemExit("Confirmation results contain duplicate trial IDs")

    grouped: dict[tuple[str, str], list[dict]] = {}
    for result in results:
        trial = WinningMLPTrial(**result["trial"])
        if trial.stage != "confirmation":
            raise SystemExit(f"Non-confirmation result found: {trial.trial_id}")
        grouped.setdefault((trial.architecture, trial.profile_id), []).append(result)

    rows = []
    for (architecture, profile_id), values in sorted(grouped.items()):
        if len(values) != 5:
            raise SystemExit(
                f"Expected five seeds for {architecture}/{profile_id}, "
                f"found {len(values)}"
            )
        scores = np.asarray(
            [value["best_validation_rmspe"] for value in values], dtype=np.float64
        )
        budgets = {float(value["trial"]["mean_dropout"]) for value in values}
        if len(budgets) != 1:
            raise SystemExit(f"Mixed budgets for {architecture}/{profile_id}")
        rows.append(
            {
                "architecture": architecture,
                "profile_id": profile_id,
                "mean_dropout": budgets.pop(),
                "test_rmspe_mean": float(scores.mean()),
                "test_rmspe_std": float(scores.std(ddof=1)),
                "seed_count": len(scores),
            }
        )

    expected_groups = len(SHALLOW_PROFILE_IDS) + len(DEEP_PROFILE_IDS)
    if len(rows) != expected_groups:
        raise SystemExit(
            f"Expected {expected_groups} profile groups, found {len(rows)}"
        )
    winners = {}
    for architecture in ("exact_shallow", "matched_depth12"):
        candidates = [row for row in rows if row["architecture"] == architecture]
        winners[architecture] = min(candidates, key=lambda row: row["test_rmspe_mean"])

    output_dir = run_dir / "aggregate"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "confirmation-summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_dir / "confirmation-summary.json"
    json_path.write_text(
        json.dumps({"rows": rows, "winners": winners}, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    for architecture, winner in winners.items():
        print(
            f"{architecture}: {winner['profile_id']} "
            f"p={winner['mean_dropout']:.3f} "
            f"test_rmspe={winner['test_rmspe_mean']:.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    profiles = subparsers.add_parser("profiles")
    profiles.set_defaults(func=command_profiles)
    status = subparsers.add_parser("status")
    status.add_argument("--run-dir", required=True)
    status.set_defaults(func=command_status)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--run-dir", required=True)
    aggregate.set_defaults(func=command_aggregate)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--manifest", required=True)
    plan.add_argument("--seeds", type=int, default=3)
    plan.set_defaults(func=command_plan)
    epoch_probe = subparsers.add_parser("plan-epoch-probe")
    epoch_probe.add_argument("--manifest", required=True)
    epoch_probe.set_defaults(func=command_plan_epoch_probe)
    confirm = subparsers.add_parser("plan-confirmation")
    confirm.add_argument("--selection-dir", required=True)
    confirm.add_argument("--manifest", required=True)
    confirm.add_argument("--selection-json", required=True)
    confirm.set_defaults(func=command_plan_confirmation)
    run = subparsers.add_parser("run")
    run.add_argument("--manifest", required=True)
    run.add_argument("--index", type=int, required=True)
    run.add_argument("--cache", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--checkpoint-dir")
    run.add_argument("--device", default="auto")
    run.add_argument(
        "--wandb-mode",
        choices=("disabled", "offline", "online"),
        default=os.environ.get("WANDB_MODE", "disabled"),
    )
    run.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT"))
    run.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    run.add_argument(
        "--wandb-run-group",
        default=os.environ.get("WANDB_RUN_GROUP", "optiver-winning-mlp"),
    )
    run.add_argument("--wandb-dir", default=os.environ.get("WANDB_DIR", "wandb"))
    run.set_defaults(func=command_run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
