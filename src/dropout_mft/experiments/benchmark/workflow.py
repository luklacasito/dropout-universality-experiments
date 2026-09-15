"""Shared filesystem and statistics helpers for benchmark experiment drivers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from dropout_mft.experiments.benchmark.protocol import (
    BenchmarkTrialSpec,
    read_benchmark_manifest,
    trial_checkpoint_path,
    trial_output_path,
    write_benchmark_manifest,
)
from dropout_mft.provenance import provenance_sha256, sha256_file
from dropout_mft.provenance import runtime_provenance as _provenance
from dropout_mft.results import load_npz_result
from dropout_mft.statistics import paired_percentile_interval


def manifest_path(run_dir: str | Path, stage: str) -> Path:
    return Path(run_dir) / "manifests" / f"{stage}.jsonl"


def selection_path(run_dir: str | Path, stage: str) -> Path:
    return Path(run_dir) / "selections" / f"{stage}.json"


def load_selection(run_dir: str | Path, stage: str) -> dict:
    path = selection_path(run_dir, stage)
    if not path.is_file():
        raise SystemExit(f"Missing selection: {path}")
    return json.loads(path.read_text())


def write_immutable_manifest(
    run_dir: str | Path,
    stage: str,
    specs: Sequence[BenchmarkTrialSpec],
) -> tuple[Path, dict]:
    """Write a provenance-bound plan once and refuse accidental replacement."""

    path = manifest_path(run_dir, stage)
    if path.exists():
        raise SystemExit(f"Refusing to overwrite immutable manifest: {path}")
    provenance = _provenance("planning", None)
    write_benchmark_manifest(path, list(specs), provenance=provenance)
    return path, provenance


def write_json_atomic(path: str | Path, value: object) -> Path:
    """Replace a JSON file only after serialization and writing succeed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def plan_selected_stage(stage, cells, selections, *, lr_search, budget_search, confirm):
    """Expand a study's cells through the common three-stage selection flow.

    Callbacks contain the study-specific intervention. This function owns the
    prerequisite checks, ordering, and extraction of selected learning rates.
    """
    if stage not in ("lr_search", "budget_search", "confirm"):
        raise ValueError(f"Unknown stage: {stage!r}")
    specs = []
    for dataset, model_kind in cells:
        if stage == "lr_search":
            specs.extend(lr_search(dataset, model_kind))
            continue
        cell = f"{dataset}/{model_kind}"
        required = (
            ("lr_search",)
            if stage == "budget_search"
            else ("lr_search", "budget_search")
        )
        for source in required:
            if cell not in selections.get(source, {}):
                raise SystemExit(f"Missing {source} selection for {cell}")
        rates = selections["lr_search"][cell]
        if stage == "budget_search":
            rates = {
                profile: float(choice["learning_rate"])
                for profile, choice in rates.items()
            }
            specs.extend(budget_search(dataset, model_kind, rates))
        else:
            specs.extend(
                confirm(dataset, model_kind, selections["budget_search"][cell], rates)
            )
    return specs


def stage_selections(run_dir: str | Path, stage: str) -> dict[str, dict]:
    """Read only the prior selections required by this stage."""
    stages = {
        "lr_search": (),
        "budget_search": ("lr_search",),
        "confirm": ("lr_search", "budget_search"),
    }
    return {name: load_selection(run_dir, name) for name in stages[stage]}


def save_plan(
    run_dir: str | Path,
    stage: str,
    specs: Sequence[BenchmarkTrialSpec],
    *,
    description: str = "",
) -> Path:
    """Persist and describe the same immutable plan for every study."""
    path, provenance = write_immutable_manifest(run_dir, stage, specs)
    print(
        f"{description}stage={stage} trials={len(specs)} cells={len({s.cell for s in specs})}"
    )
    print(f"manifest={path}")
    print(f"provenance_sha256={provenance_sha256(provenance)}")
    return path


def load_completed_trials(
    run_dir, stage, *, require_complete=False, require_checkpoint=False
):
    """Load manifest-ordered results, with explicit completeness requirements."""
    path = manifest_path(run_dir, stage)
    if not path.is_file():
        raise SystemExit(f"Missing manifest {path}")
    completed, missing = [], []
    for spec in read_benchmark_manifest(path):
        output = trial_output_path(run_dir, spec)
        checkpoint = trial_checkpoint_path(run_dir, spec)
        if not output.is_file() or (require_checkpoint and not checkpoint.is_file()):
            missing.append(spec.trial_id)
            continue
        result = load_npz_result(output)
        if result.get("trial", {}).get("trial_id") != spec.trial_id:
            raise ValueError(f"Trial/result mismatch: {output}")
        if require_checkpoint and result.get("checkpoint", {}).get(
            "sha256"
        ) != sha256_file(checkpoint):
            raise ValueError(f"Checkpoint hash mismatch: {checkpoint}")
        completed.append((spec, result, output))
    if missing and require_complete:
        raise SystemExit(
            f"Stage {stage} is incomplete: {len(missing)} missing trials; first={missing[0]}"
        )
    if missing:
        print(f"warning: {len(missing)} trials of stage {stage} are not finished yet")
    if not completed:
        raise SystemExit(f"No completed trials found for stage {stage}")
    return completed


def trial_record(spec: BenchmarkTrialSpec, result: dict, path: Path) -> dict:
    """Flatten common metrics once for selection and paired analysis."""
    test = result["test"]
    data = result.get("data", {})
    final = test.get("fixed_final_epoch", {})
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
        "test_evaluated": bool(test.get("evaluated", False)),
        "test_loss": test["loss"],
        "test_accuracy": test["accuracy"],
        "final_epoch_test_loss": final.get("loss"),
        "final_epoch_test_accuracy": final.get("accuracy"),
        "split_hash": data.get("split_hash"),
        "split_protocol": data.get("split_protocol"),
    }


def collect_records(
    run_dir: str | Path, stage: str, *, require_complete: bool = False
) -> list[dict]:
    return [
        trial_record(*row)
        for row in load_completed_trials(
            run_dir, stage, require_complete=require_complete
        )
    ]


def index_records(
    records: Sequence[dict], *, group: str = "cell"
) -> dict[str, dict[str, dict[int, dict]]]:
    """Index paired observations while rejecting duplicate seeds."""
    indexed = {}
    for record in records:
        key, profile, seed = record[group], record["profile_id"], int(record["seed"])
        arm = indexed.setdefault(key, {}).setdefault(profile, {})
        if seed in arm:
            raise ValueError(f"Duplicate {key}/{profile}/seed={seed}")
        arm[seed] = record
    return indexed


def select_candidates(
    records: Sequence[dict], *, excluded_profiles: Sequence[str] = ()
) -> dict[str, dict]:
    """Select by mean validation loss, retaining deterministic grid tie breaks."""
    grouped = {}
    for row in records:
        if row["profile_id"] not in excluded_profiles:
            grouped.setdefault(row["cell"], {}).setdefault(
                row["profile_id"], []
            ).append(row)
    selection = {}
    for cell, profiles in sorted(grouped.items()):
        selection[cell] = {}
        for profile, rows in sorted(profiles.items()):
            losses_by_candidate = {}
            for row in rows:
                candidate = (row["learning_rate"], row["mean_dropout"])
                losses_by_candidate.setdefault(candidate, []).append(
                    row["validation_loss"]
                )
            score, lr, budget, count = min(
                (float(np.mean(losses)), lr, budget, len(losses))
                for (lr, budget), losses in losses_by_candidate.items()
            )
            selection[cell][profile] = {
                "learning_rate": lr,
                "mean_dropout": budget,
                "validation_loss": score,
                "seeds": count,
                "criterion": "mean_validation_loss_over_seeds_v1",
            }
    return selection


def paired_comparison(candidate, baseline, *, resamples=10_000, seed=20260812):
    """Compare loss and accuracy arrays already aligned by their seed IDs."""
    arrays = [
        np.asarray(arm[metric])
        for arm in (candidate, baseline)
        for metric in ("test_losses", "test_accuracies")
    ]
    if any(a.ndim != 1 or a.shape != arrays[0].shape for a in arrays):
        raise ValueError(
            "Paired loss and accuracy arrays must be one-dimensional and aligned"
        )
    candidate_losses, candidate_accuracies, baseline_losses, baseline_accuracies = (
        arrays
    )
    losses = candidate_losses - baseline_losses
    accuracies = 100.0 * (candidate_accuracies - baseline_accuracies)
    return {
        "mean_test_loss_delta": float(losses.mean()),
        "test_loss_delta_95ci": paired_percentile_interval(
            losses, resamples=resamples, seed=seed
        ),
        "mean_test_accuracy_delta_pp": float(accuracies.mean()),
        "test_accuracy_delta_pp_95ci": paired_percentile_interval(
            accuracies, resamples=resamples, seed=seed
        ),
        "paired_seed_win_rate_loss": float((losses < 0).mean()),
    }
