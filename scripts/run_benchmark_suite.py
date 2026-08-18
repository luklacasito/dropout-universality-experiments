#!/usr/bin/env python
"""Driver for the multi-modality dropout-profile confirmation.

Staged flow, one stage at a time, because each stage's manifest depends on the
previous stage's validation-only selection:

    plan   --stage lr_search       # write the immutable manifest
    run    --stage lr_search       # execute a shard (Slurm array task)
    select --stage lr_search       # pick a learning rate per profile

    plan   --stage budget_search
    run    --stage budget_search
    select --stage budget_search   # pick a mean dropout per profile

    plan   --stage confirm
    run    --stage confirm         # the only stage that touches the test set
    aggregate                      # paired per-seed deltas with bootstrap CIs

Every stage is resumable: ``run`` skips trials whose stored result already
matches the manifest row, so a requeued array task costs nothing.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dropout_mft.benchmark_suite import (  # noqa: E402
    BENCHMARK_PROFILE_IDS,
    CONTROL_PROFILE_ID,
    STAGES,
    BenchmarkTrialSpec,
    budget_search_specs,
    bundle_for,
    confirm_specs,
    lr_search_specs,
    read_benchmark_manifest,
    run_benchmark_trial,
    shard,
    trial_output_path,
    trial_checkpoint_path,
    write_benchmark_manifest,
)
from dropout_mft.benchmarks import (  # noqa: E402
    BENCHMARK_NAMES,
    BENCHMARK_SPECS,
    load_benchmark_bundle,
)
from dropout_mft.provenance import provenance_sha256  # noqa: E402
from dropout_mft.results import load_npz_result  # noqa: E402
from dropout_mft.scale_transfer import _provenance  # noqa: E402
from dropout_mft.wandb_tracking import (  # noqa: E402
    WandbOptions,
    benchmark_wandb_run,
    finish_benchmark_wandb_run,
    log_benchmark_wandb_result,
    tracking_is_complete,
)

MODEL_KINDS = ("mlp", "transformer")
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260812


def manifest_path(run_dir: Path, stage: str) -> Path:
    return run_dir / "manifests" / f"{stage}.jsonl"


def selection_path(run_dir: Path, stage: str) -> Path:
    return run_dir / "selections" / f"{stage}.json"


def _cells(datasets: list[str] | None, model_kinds: list[str] | None):
    chosen_datasets = datasets or list(BENCHMARK_NAMES)
    chosen_kinds = model_kinds or list(MODEL_KINDS)
    for dataset in chosen_datasets:
        if dataset not in BENCHMARK_SPECS:
            raise SystemExit(f"Unknown dataset: {dataset!r}")
        for model_kind in chosen_kinds:
            if model_kind not in MODEL_KINDS:
                raise SystemExit(f"Unknown model kind: {model_kind!r}")
            yield dataset, model_kind


def _load_selection(run_dir: Path, stage: str) -> dict:
    path = selection_path(run_dir, stage)
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Run the previous stage and its `select` step first."
        )
    return json.loads(path.read_text())


def command_plan(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    specs: list[BenchmarkTrialSpec] = []
    if args.stage == "lr_search":
        for dataset, model_kind in _cells(args.datasets, args.model_kinds):
            specs.extend(lr_search_specs(dataset, model_kind, depth=args.depth))
    elif args.stage == "budget_search":
        if args.selection_stage not in (None, "lr_search"):
            raise SystemExit("budget_search must select from lr_search")
        selection = _load_selection(run_dir, "lr_search")
        for dataset, model_kind in _cells(args.datasets, args.model_kinds):
            cell = f"{dataset}/{model_kind}"
            if cell not in selection:
                raise SystemExit(f"No learning-rate selection for cell {cell}")
            learning_rates = {
                profile: float(choice["learning_rate"])
                for profile, choice in selection[cell].items()
            }
            specs.extend(
                budget_search_specs(
                    dataset, model_kind, learning_rates, depth=args.depth
                )
            )
    else:
        source_stage = args.selection_stage or "budget_search"
        selection = _load_selection(run_dir, source_stage)
        for dataset, model_kind in _cells(args.datasets, args.model_kinds):
            cell = f"{dataset}/{model_kind}"
            if cell not in selection:
                raise SystemExit(f"No budget selection for cell {cell}")
            specs.extend(
                confirm_specs(dataset, model_kind, selection[cell], depth=args.depth)
            )

    provenance = _provenance("planning", None)
    path = manifest_path(run_dir, args.stage)
    write_benchmark_manifest(path, specs, provenance=provenance)
    cells = sorted({spec.cell for spec in specs})
    print(f"stage={args.stage} trials={len(specs)} cells={len(cells)}")
    print(f"manifest={path}")
    print(f"provenance_sha256={provenance_sha256(provenance)}")


def command_run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    wandb_options: WandbOptions | None = None
    if args.wandb_mode != "disabled":
        if not args.wandb_project:
            raise SystemExit(
                "W&B tracking is enabled but no project was provided. Set "
                "WANDB_PROJECT or pass --wandb-project."
            )
        wandb_options = WandbOptions(
            project=args.wandb_project,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            run_group=args.wandb_run_group or run_dir.name,
            directory=(Path(args.wandb_dir) if args.wandb_dir else run_dir / "wandb"),
        )
    path = Path(args.manifest) if args.manifest else manifest_path(run_dir, args.stage)
    if not path.exists():
        raise SystemExit(f"Missing manifest {path}; run `plan --stage {args.stage}`")
    specs = read_benchmark_manifest(path)
    if args.datasets:
        specs = [spec for spec in specs if spec.dataset in args.datasets]
    if args.model_kinds:
        specs = [spec for spec in specs if spec.model_kind in args.model_kinds]
    if args.profiles:
        specs = [spec for spec in specs if spec.profile_id in args.profiles]
    assigned = shard(specs, args.shard_index, args.num_shards)
    if not assigned:
        print(f"shard {args.shard_index}/{args.num_shards} has no work")
        return

    # One bundle load per (dataset, view) rather than per trial: the raw arrays
    # dominate memory and are identical across every trial in the group.
    bundles: dict[tuple[str, str], object] = {}
    completed = 0
    for index, spec in enumerate(assigned, start=1):
        key = (spec.dataset, spec.data_view)
        if key not in bundles:
            bundles.clear()
            bundles[key] = bundle_for(spec, root=args.data_root)
        output_path = trial_output_path(run_dir, spec)
        checkpoint_path = (
            trial_checkpoint_path(run_dir, spec) if args.save_best_checkpoint else None
        )
        wandb_status = "disabled"
        if wandb_options is None or tracking_is_complete(wandb_options, spec.trial_id):
            result = run_benchmark_trial(
                spec,
                bundles[key],
                output_path,
                device=args.device,
                force=args.force,
                source_provenance=None,
                checkpoint_path=checkpoint_path,
            )
            if wandb_options is not None:
                wandb_status = "already-tracked"
        else:
            # Starting W&B before training captures GPU/CPU system metrics.  If
            # the NPZ already exists (for example from a pre-W&B run), this
            # same path backfills its saved curves and artifact exactly once.
            with benchmark_wandb_run(wandb_options, spec, bundles[key]) as (
                wandb_run,
                wandb_module,
            ):
                result = run_benchmark_trial(
                    spec,
                    bundles[key],
                    output_path,
                    device=args.device,
                    force=args.force,
                    source_provenance=None,
                    checkpoint_path=checkpoint_path,
                )
                log_benchmark_wandb_result(wandb_run, wandb_module, result, output_path)
                finish_benchmark_wandb_run(wandb_run, wandb_options, spec.trial_id)
            wandb_status = f"tracked-{wandb_options.mode}"
        completed += 1
        print(
            f"[{index}/{len(assigned)}] {spec.cell} {spec.profile_id} "
            f"lr={spec.learning_rate:.2e} p={spec.mean_dropout:.2f} "
            f"seed={spec.seed} "
            f"val_loss={result['selection']['validation_loss']:.4f} "
            f"wandb={wandb_status} "
            f"({result['trial']['duration_seconds']:.0f}s)",
            flush=True,
        )
    print(f"shard {args.shard_index}/{args.num_shards} completed {completed} trials")


def _collect(run_dir: Path, stage: str) -> list[dict]:
    path = manifest_path(run_dir, stage)
    if not path.exists():
        raise SystemExit(f"Missing manifest {path}")
    records: list[dict] = []
    missing = 0
    for spec in read_benchmark_manifest(path):
        output_path = trial_output_path(run_dir, spec)
        if not output_path.exists():
            missing += 1
            continue
        result = load_npz_result(output_path)
        records.append(
            {
                "cell": spec.cell,
                "dataset": spec.dataset,
                "model_kind": spec.model_kind,
                "profile_id": spec.profile_id,
                "learning_rate": spec.learning_rate,
                "mean_dropout": spec.mean_dropout,
                "seed": spec.seed,
                "validation_loss": float(result["selection"]["validation_loss"]),
                "validation_accuracy": float(
                    result["selection"]["validation_accuracy"]
                ),
                "test_loss": result["test"]["loss"],
                "test_accuracy": result["test"]["accuracy"],
                "final_epoch_test_loss": result["test"]
                .get("fixed_final_epoch", {})
                .get("loss"),
                "final_epoch_test_accuracy": result["test"]
                .get("fixed_final_epoch", {})
                .get("accuracy"),
            }
        )
    if missing:
        print(f"warning: {missing} trials of stage {stage} are not finished yet")
    if not records:
        raise SystemExit(f"No completed trials found for stage {stage}")
    return records


def command_select(args: argparse.Namespace) -> None:
    """Choose hyperparameters from validation loss only, per profile per cell."""

    run_dir = Path(args.run_dir)
    records = _collect(run_dir, args.stage)
    grouped: dict[str, dict[str, list[dict]]] = {}
    for record in records:
        grouped.setdefault(record["cell"], {}).setdefault(
            record["profile_id"], []
        ).append(record)

    selection: dict[str, dict] = {}
    for cell, profiles in sorted(grouped.items()):
        selection[cell] = {}
        for profile_id, rows in sorted(profiles.items()):
            if profile_id == CONTROL_PROFILE_ID:
                continue
            # Average across seeds so a single lucky draw cannot win the grid.
            keys: dict[tuple[float, float], list[float]] = {}
            for row in rows:
                keys.setdefault((row["learning_rate"], row["mean_dropout"]), []).append(
                    row["validation_loss"]
                )
            scored = [
                (float(np.mean(losses)), learning_rate, mean_dropout, len(losses))
                for (learning_rate, mean_dropout), losses in keys.items()
            ]
            score, learning_rate, mean_dropout, seeds = min(scored)
            selection[cell][profile_id] = {
                "learning_rate": learning_rate,
                "mean_dropout": mean_dropout,
                "validation_loss": score,
                "seeds": seeds,
                "criterion": "mean_validation_loss_over_seeds_v1",
            }
            print(
                f"{cell:34s} {profile_id:12s} lr={learning_rate:.2e} "
                f"p={mean_dropout:.2f} val_loss={score:.4f} (n={seeds})"
            )

    path = selection_path(run_dir, args.stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    print(f"selection={path}")


def _paired_bootstrap(
    deltas: np.ndarray, *, resamples: int, seed: int
) -> tuple[float, float]:
    """Percentile CI over a resample of the paired per-seed differences."""

    rng = np.random.default_rng(seed)
    draws = rng.choice(deltas, size=(resamples, len(deltas)), replace=True)
    means = draws.mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def command_aggregate(args: argparse.Namespace) -> None:
    """Report paired per-seed loss and accuracy deltas against uniform."""

    run_dir = Path(args.run_dir)
    records = _collect(run_dir, "confirm")
    by_cell: dict[str, dict[str, dict[int, dict]]] = {}
    for record in records:
        by_cell.setdefault(record["cell"], {}).setdefault(record["profile_id"], {})[
            record["seed"]
        ] = record

    summary: dict[str, dict] = {}
    print(
        f"\n{'cell':34s} {'profile':12s} {'test loss':>10s} {'accuracy':>10s} "
        f"{'final loss':>10s} {'final acc':>10s} {'loss delta':>11s} "
        f"{'95% CI':>20s} {'win':>5s}"
    )
    print("-" * 134)
    for cell, profiles in sorted(by_cell.items()):
        if "uniform" not in profiles:
            print(f"{cell}: no uniform arm, skipping")
            continue
        baseline = profiles["uniform"]
        baseline_losses = np.array(
            [baseline[seed]["test_loss"] for seed in sorted(baseline)], dtype=float
        )
        baseline_accuracies = np.array(
            [baseline[seed]["test_accuracy"] for seed in sorted(baseline)],
            dtype=float,
        )
        baseline_final_losses = np.array(
            [baseline[seed]["final_epoch_test_loss"] for seed in sorted(baseline)],
            dtype=float,
        )
        baseline_final_accuracies = np.array(
            [baseline[seed]["final_epoch_test_accuracy"] for seed in sorted(baseline)],
            dtype=float,
        )
        summary[cell] = {
            "uniform": {
                "test_loss_mean": float(baseline_losses.mean()),
                "test_loss_sd": float(baseline_losses.std(ddof=1)),
                "test_accuracy_mean": float(baseline_accuracies.mean()),
                "test_accuracy_sd": float(baseline_accuracies.std(ddof=1)),
                "fixed_final_test_loss_mean": float(baseline_final_losses.mean()),
                "fixed_final_test_accuracy_mean": float(
                    baseline_final_accuracies.mean()
                ),
                "seeds": sorted(baseline),
            }
        }
        print(
            f"{cell:34s} {'uniform':12s} {baseline_losses.mean():10.4f} "
            f"{baseline_accuracies.mean():10.4f} "
            f"{baseline_final_losses.mean():10.4f} "
            f"{baseline_final_accuracies.mean():10.4f} "
            f"{'':>11s} {'':>20s} {'':>5s}"
        )
        for profile_id, arm in sorted(profiles.items()):
            if profile_id == "uniform":
                continue
            shared = sorted(set(arm) & set(baseline))
            if len(shared) < 2:
                print(f"{cell} {profile_id}: fewer than 2 paired seeds, skipping")
                continue
            losses = np.array([arm[seed]["test_loss"] for seed in shared], dtype=float)
            accuracies = np.array(
                [arm[seed]["test_accuracy"] for seed in shared], dtype=float
            )
            final_losses = np.array(
                [arm[seed]["final_epoch_test_loss"] for seed in shared], dtype=float
            )
            final_accuracies = np.array(
                [arm[seed]["final_epoch_test_accuracy"] for seed in shared],
                dtype=float,
            )
            paired_baseline = np.array(
                [baseline[seed]["test_loss"] for seed in shared], dtype=float
            )
            paired_baseline_accuracy = np.array(
                [baseline[seed]["test_accuracy"] for seed in shared], dtype=float
            )
            paired_baseline_final_loss = np.array(
                [baseline[seed]["final_epoch_test_loss"] for seed in shared],
                dtype=float,
            )
            paired_baseline_final_accuracy = np.array(
                [baseline[seed]["final_epoch_test_accuracy"] for seed in shared],
                dtype=float,
            )
            deltas = losses - paired_baseline
            accuracy_deltas = accuracies - paired_baseline_accuracy
            final_loss_deltas = final_losses - paired_baseline_final_loss
            final_accuracy_deltas = final_accuracies - paired_baseline_final_accuracy
            low, high = _paired_bootstrap(
                deltas, resamples=args.resamples, seed=BOOTSTRAP_SEED
            )
            relative = float(deltas.mean() / paired_baseline.mean() * 100)
            summary[cell][profile_id] = {
                "test_loss_mean": float(losses.mean()),
                "test_loss_sd": float(losses.std(ddof=1)),
                "test_accuracy_mean": float(accuracies.mean()),
                "test_accuracy_sd": float(accuracies.std(ddof=1)),
                "paired_delta_mean": float(deltas.mean()),
                "paired_delta_ci95": [low, high],
                "relative_change_percent": relative,
                "seed_win_rate": float((deltas < 0).mean()),
                "paired_accuracy_delta_mean": float(accuracy_deltas.mean()),
                "fixed_final_test_loss_mean": float(final_losses.mean()),
                "fixed_final_test_accuracy_mean": float(final_accuracies.mean()),
                "fixed_final_paired_loss_delta_mean": float(final_loss_deltas.mean()),
                "fixed_final_paired_accuracy_delta_mean": float(
                    final_accuracy_deltas.mean()
                ),
                "paired_seeds": shared,
            }
            marker = "yes" if high < 0 else ("no" if low > 0 else "-")
            print(
                f"{cell:34s} {profile_id:12s} {losses.mean():10.4f} "
                f"{accuracies.mean():10.4f} {final_losses.mean():10.4f} "
                f"{final_accuracies.mean():10.4f} {deltas.mean():+11.4f} "
                f"[{low:+.4f}, {high:+.4f}] {marker:>5s}"
            )

    path = run_dir / "summary" / "confirm_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        "\nNegative delta favours the profile over uniform. 'win' marks a 95% "
        "paired-bootstrap CI that excludes zero."
    )
    print(f"summary={path}")


def command_status(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    for stage in STAGES:
        path = manifest_path(run_dir, stage)
        if not path.exists():
            print(f"{stage:14s} not planned")
            continue
        specs = read_benchmark_manifest(path)
        done = sum(1 for spec in specs if trial_output_path(run_dir, spec).exists())
        selected = "selected" if selection_path(run_dir, stage).exists() else ""
        print(f"{stage:14s} {done:5d}/{len(specs):5d} trials  {selected}")


def command_cost(args: argparse.Namespace) -> None:
    """Print the trial count per stage without touching the filesystem."""

    totals = {stage: 0 for stage in STAGES}
    dummy_lr = {profile: 1e-4 for profile in BENCHMARK_PROFILE_IDS}
    dummy_selection = {
        profile: {"learning_rate": 1e-4, "mean_dropout": 0.10}
        for profile in BENCHMARK_PROFILE_IDS
    }
    for dataset, model_kind in _cells(args.datasets, args.model_kinds):
        totals["lr_search"] += len(
            lr_search_specs(dataset, model_kind, depth=args.depth)
        )
        if not args.skip_budget_search:
            totals["budget_search"] += len(
                budget_search_specs(dataset, model_kind, dummy_lr, depth=args.depth)
            )
        totals["confirm"] += len(
            confirm_specs(dataset, model_kind, dummy_selection, depth=args.depth)
        )
    print(f"depth={args.depth}")
    for stage in STAGES:
        print(f"{stage:14s} {totals[stage]:5d} trials")
    print(f"{'total':14s} {sum(totals.values()):5d} trials")


def command_verify_data(args: argparse.Namespace) -> None:
    """Load every prepared cache through the real loader and report what it holds.

    Run this after preparation and before submitting any array: it catches a
    wrong shape, a collapsed class balance, or a leaked split for the price of a
    few seconds, rather than after a stage of GPU time.
    """

    failures: list[str] = []
    for name in args.datasets or list(BENCHMARK_NAMES):
        spec = BENCHMARK_SPECS[name]
        print(f"\n=== {name} ===")
        try:
            # Verify one representation at a time.  A full Tiny ImageNet view
            # is ~4.9 GB, so retaining both would exceed the 10 GB Bridges-2
            # host-memory limit before Torch/Python overhead is counted.
            flat = load_benchmark_bundle(name, root=args.data_root, view="mlp")
        except (FileNotFoundError, ValueError, KeyError) as exc:
            print(f"FAIL {exc}")
            failures.append(name)
            continue
        flat_shapes = {
            split: tuple(getattr(flat, split).tensors[0].shape)
            for split in ("train", "validation", "test")
        }
        print(f"  {'mlp':9s} {flat_shapes}")
        flat_labels = {
            split: getattr(flat, split).tensors[1].numpy().copy()
            for split in ("train", "validation", "test")
        }
        flat_split_hash = flat.split_hash
        flat_protocol = flat.split_protocol
        flat_width_ok = flat.train.tensors[0].shape[1:].numel() == spec.mlp_input_dim
        flat_finite = torch_all_finite(flat.train.tensors[0])
        labels = flat_labels["train"]
        balance = np.bincount(labels, minlength=spec.classes) / len(labels)
        del flat
        gc.collect()

        try:
            sequence = load_benchmark_bundle(name, root=args.data_root, view="sequence")
        except (FileNotFoundError, ValueError, KeyError) as exc:
            print(f"FAIL {exc}")
            failures.append(name)
            continue
        sequence_shapes = {
            split: tuple(getattr(sequence, split).tensors[0].shape)
            for split in ("train", "validation", "test")
        }
        print(f"  {'sequence':9s} {sequence_shapes}")
        checks: list[tuple[str, bool]] = [
            (
                "views share labels",
                all(
                    np.array_equal(
                        flat_labels[split],
                        getattr(sequence, split).tensors[1].numpy(),
                    )
                    for split in ("train", "validation", "test")
                ),
            ),
            ("split hashes agree", flat_split_hash == sequence.split_hash),
            ("all classes present in train", bool((balance > 0).all())),
            ("MLP width matches spec", flat_width_ok),
            ("train features finite", flat_finite),
        ]
        for label, ok in checks:
            print(f"  {'PASS' if ok else 'FAIL'} {label}")
            if not ok:
                failures.append(f"{name}: {label}")
        print(f"  train class balance: {np.round(balance, 4).tolist()}")
        print(f"  split_hash={flat_split_hash[:16]} protocol={flat_protocol}")

    if failures:
        raise SystemExit(f"\nFAILURES: {failures}")
    print("\nAll prepared caches verified.")


def torch_all_finite(tensor) -> bool:
    import torch

    return bool(torch.isfinite(tensor.float()).all())


def command_smoke(args: argparse.Namespace) -> None:
    """One tiny CPU trial per architecture, verifying shapes and budgets end to end."""

    from dropout_mft.benchmark_suite import (
        benchmark_profile_layers,
        build_benchmark_model,
    )

    failures = 0
    for dataset in BENCHMARK_NAMES:
        for model_kind in MODEL_KINDS:
            for profile_id in (*BENCHMARK_PROFILE_IDS, CONTROL_PROFILE_ID):
                spec = BenchmarkTrialSpec(
                    stage="lr_search",
                    dataset=dataset,
                    model_kind=model_kind,
                    profile_id=profile_id,
                    mean_dropout=0.0 if profile_id == CONTROL_PROFILE_ID else 0.10,
                    max_dropout=0.30 if profile_id == "big_step" else 0.20,
                    learning_rate=1e-4,
                    seed=0,
                )
                probabilities = benchmark_profile_layers(spec)
                data = BENCHMARK_SPECS[dataset]
                model = build_benchmark_model(spec)
                import torch

                if model_kind == "mlp":
                    inputs = torch.randn(2, data.mlp_input_dim)
                elif data.image_channels is not None:
                    inputs = torch.randn(
                        2, data.image_channels, data.image_size, data.image_size
                    )
                elif data.vocab_size is not None:
                    inputs = torch.randint(
                        0, data.vocab_size, (2, data.sequence_length)
                    )
                else:
                    inputs = torch.randn(
                        2, data.sequence_length, data.input_features or 1
                    )
                logits = model(inputs)
                ok = logits.shape == (2, data.classes)
                budget_ok = (
                    abs(float(np.mean(probabilities)) - spec.mean_dropout) < 1e-12
                )
                if not (ok and budget_ok):
                    failures += 1
                print(
                    f"{'PASS' if ok and budget_ok else 'FAIL'} "
                    f"{dataset:16s} {model_kind:12s} {profile_id:11s} "
                    f"p={np.round(probabilities, 3).tolist()} "
                    f"logits={tuple(logits.shape)}"
                )
    if failures:
        raise SystemExit(f"{failures} smoke checks failed")
    print("\nAll shape and budget checks passed.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def add_filters(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--dataset", dest="datasets", action="append")
        sub.add_argument(
            "--model-kind", dest="model_kinds", action="append", choices=MODEL_KINDS
        )

    plan = commands.add_parser("plan", help="write an immutable stage manifest")
    plan.add_argument("--run-dir", required=True)
    plan.add_argument("--stage", required=True, choices=STAGES)
    plan.add_argument("--depth", type=int, default=6)
    plan.add_argument(
        "--selection-stage",
        choices=("lr_search", "budget_search"),
        help="selection feeding confirm; defaults to budget_search",
    )
    add_filters(plan)
    plan.set_defaults(func=command_plan)

    run = commands.add_parser("run", help="execute one manifest shard")
    run.add_argument("--run-dir", required=True)
    run.add_argument("--stage", required=True, choices=STAGES)
    run.add_argument("--manifest")
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--num-shards", type=int, default=1)
    run.add_argument("--device", default="auto")
    run.add_argument("--data-root", default="data")
    run.add_argument("--force", action="store_true")
    run.add_argument(
        "--save-best-checkpoint",
        action="store_true",
        help="atomically retain the selected best-validation model weights",
    )
    run.add_argument(
        "--wandb-mode",
        choices=("disabled", "offline", "online"),
        default=os.environ.get("WANDB_MODE", "disabled"),
        help="track every trial; Bridges-2 should normally use offline",
    )
    run.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT"))
    run.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    run.add_argument("--wandb-run-group", default=os.environ.get("WANDB_RUN_GROUP"))
    run.add_argument("--wandb-dir", default=os.environ.get("WANDB_DIR"))
    run.add_argument("--profile", dest="profiles", action="append")
    add_filters(run)
    run.set_defaults(func=command_run)

    select = commands.add_parser("select", help="pick hyperparameters on validation")
    select.add_argument("--run-dir", required=True)
    select.add_argument(
        "--stage", required=True, choices=("lr_search", "budget_search")
    )
    select.set_defaults(func=command_select)

    aggregate = commands.add_parser(
        "aggregate", help="paired deltas with bootstrap CIs"
    )
    aggregate.add_argument("--run-dir", required=True)
    aggregate.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    aggregate.set_defaults(func=command_aggregate)

    status = commands.add_parser("status", help="per-stage completion counts")
    status.add_argument("--run-dir", required=True)
    status.set_defaults(func=command_status)

    cost = commands.add_parser("cost", help="trial counts per stage")
    cost.add_argument("--depth", type=int, default=6)
    cost.add_argument(
        "--skip-budget-search",
        action="store_true",
        help="count a fixed-budget LR-search-to-confirm pilot",
    )
    add_filters(cost)
    cost.set_defaults(func=command_cost)

    verify = commands.add_parser(
        "verify-data", help="load every prepared cache and check it"
    )
    verify.add_argument("--data-root", default="data")
    add_filters(verify)
    verify.set_defaults(func=command_verify_data)

    smoke = commands.add_parser("smoke", help="CPU shape and budget checks")
    smoke.set_defaults(func=command_smoke)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
