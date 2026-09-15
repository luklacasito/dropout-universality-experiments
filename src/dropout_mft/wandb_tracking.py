"""Weights & Biases tracking for the multi-modality benchmark suite.

The benchmark result files remain the authoritative, content-addressed record.
W&B is an indexed mirror for monitoring and comparison: every trial gets one
deterministic run ID, its epoch curves, compact scalar summaries, and the exact
saved result as an artifact.
"""

from __future__ import annotations

import json
import math
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Literal

import numpy as np

if TYPE_CHECKING:
    from dropout_mft.experiments.benchmark.protocol import BenchmarkTrialSpec

    from .training import DatasetBundle


WandbMode = Literal["online", "offline"]


@dataclass(frozen=True)
class WandbOptions:
    """Explicit W&B destination and local spool settings."""

    project: str
    directory: Path
    run_group: str
    entity: str | None = None
    mode: WandbMode = "offline"

    def __post_init__(self) -> None:
        if not self.project.strip():
            raise ValueError("W&B project must not be empty")
        if not self.run_group.strip():
            raise ValueError("W&B run group must not be empty")
        if self.mode not in {"online", "offline"}:
            raise ValueError(f"Unsupported W&B mode: {self.mode!r}")


def tracking_marker_path(options: WandbOptions, trial_id: str) -> Path:
    return options.directory / "tracked" / f"{trial_id}.json"


def tracking_is_complete(options: WandbOptions, trial_id: str) -> bool:
    """Return whether this exact destination already received the trial."""

    path = tracking_marker_path(options, trial_id)
    if not path.exists():
        return False
    try:
        marker = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        marker.get("trial_id") == trial_id
        and marker.get("project") == options.project
        and marker.get("entity") == options.entity
        and marker.get("mode") == options.mode
        and marker.get("status") == "complete"
    )


def benchmark_wandb_config(
    spec: BenchmarkTrialSpec, bundle: DatasetBundle
) -> dict[str, Any]:
    """Build the searchable W&B config from the immutable trial inputs."""

    # Imported lazily to keep importing this module independent of torch.
    from dropout_mft.experiments.benchmark.protocol import (
        _schedule_record,
        benchmark_profile_layers,
    )

    schedule = _schedule_record(spec)
    return {
        **asdict(spec),
        "trial_id": spec.trial_id,
        "config_hash": spec.config_hash,
        "cell": spec.cell,
        "data_view": spec.data_view,
        "data_split_hash": bundle.split_hash,
        "data_split_protocol": bundle.split_protocol,
        "dropout_probabilities": benchmark_profile_layers(spec),
        "criticality_status": schedule["criticality_status"],
        "field_mapping": schedule["field_mapping"],
        "field_interpretation": schedule["interpretation"],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }


_CURVE_KEYS = {
    "train_loss": "train/loss",
    "train_accuracy": "train/accuracy",
    "validation_loss": "validation/loss",
    "validation_accuracy": "validation/accuracy",
    "first_optimizer_group_lr": "optimizer/first_group_lr",
    "global_learning_rate": "optimizer/global_learning_rate",
    "lr_multiplier": "optimizer/lr_multiplier",
    "optimizer_steps": "optimizer/steps",
}


def benchmark_wandb_history(result: dict) -> list[dict[str, float | int]]:
    """Convert stored epoch arrays into W&B-native scalar history rows."""

    curves = result["curves"]
    epochs = np.asarray(curves["epoch"])
    rows: list[dict[str, float | int]] = []
    for index, epoch in enumerate(epochs):
        row: dict[str, float | int] = {"epoch": int(epoch)}
        for source, target in _CURVE_KEYS.items():
            if source not in curves:
                continue
            values = np.asarray(curves[source])
            if values.ndim == 1 and index < len(values):
                value = float(values[index])
                if math.isfinite(value):
                    row[target] = value
        if "optimizer_group_lrs" in curves:
            group_lrs = np.asarray(curves["optimizer_group_lrs"])
            if group_lrs.ndim == 2 and index < len(group_lrs):
                for group_index, value in enumerate(group_lrs[index]):
                    value = float(value)
                    if math.isfinite(value):
                        row[f"optimizer/group_{group_index}_lr"] = value
        rows.append(row)
    return rows


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def benchmark_wandb_summary(result: dict) -> dict[str, Any]:
    """Flatten the decision-relevant result scalars into a run summary."""

    selection = result["selection"]
    test = result["test"]
    compute = result["compute"]
    schedule = result["schedule"]
    provenance = result["provenance"]
    curves = result.get("curves", {})
    final_validation_accuracy = selection.get("final_validation_accuracy")
    if final_validation_accuracy is None and "validation_accuracy" in curves:
        values = np.asarray(curves["validation_accuracy"])
        if values.size:
            final_validation_accuracy = float(values[-1])
    best_test = test.get("best_validation_checkpoint", {})
    fixed_test = test.get("fixed_final_epoch", {})
    summary = {
        "trial/status": result["trial"]["status"],
        "trial/duration_seconds": result["trial"]["duration_seconds"],
        "selection/criterion": selection["criterion"],
        "selection/selected_epoch": selection["selected_epoch"],
        "selection/validation_loss": selection["validation_loss"],
        "selection/validation_accuracy": selection["validation_accuracy"],
        "selection/final_validation_loss": selection["final_validation_loss"],
        "selection/final_validation_accuracy": final_validation_accuracy,
        "test/evaluated": test["evaluated"],
        "test/loss": test["loss"],
        "test/accuracy": test["accuracy"],
        "test/best_validation_epoch": best_test.get("epoch", test.get("epoch")),
        "test/best_validation_loss": best_test.get("loss", test.get("loss")),
        "test/best_validation_accuracy": best_test.get(
            "accuracy", test.get("accuracy")
        ),
        "test/final_epoch": fixed_test.get("epoch"),
        "test/final_epoch_loss": fixed_test.get("loss"),
        "test/final_epoch_accuracy": fixed_test.get("accuracy"),
        "compute/optimizer_steps": compute["optimizer_steps"],
        "compute/parameter_count": compute["parameter_count"],
        "compute/estimated_training_flops": compute["estimated_training_flops"],
        "compute/examples_seen": compute["examples_seen"],
        "schedule/mean_dropout_probability": schedule["mean_dropout_probability"],
        "schedule/max_dropout_probability": schedule["max_dropout_probability"],
        "schedule/mean_reference_field": schedule["mean_reference_field"],
        "schedule/field_damage_mean": schedule["field_damage_mean"],
        "schedule/field_damage_total": schedule["field_damage_total"],
        "schedule/xi_proxy": schedule["xi_proxy"],
        "data/split_hash": result["data"]["split_hash"],
        "provenance/source_sha256": provenance.get("source_provenance_sha256"),
    }
    return {key: _finite_or_none(value) for key, value in summary.items()}


def _run_name(spec: BenchmarkTrialSpec) -> str:
    return (
        f"{spec.dataset}-{spec.model_kind}-{spec.profile_id}-"
        f"s{spec.seed}-{spec.trial_id[:8]}"
    )


def _tags(spec: BenchmarkTrialSpec, criticality_status: str) -> list[str]:
    return [
        "benchmark-suite",
        spec.stage,
        spec.dataset,
        spec.model_kind,
        spec.profile_id,
        f"depth-{spec.depth}",
        criticality_status,
    ]


@contextmanager
def benchmark_wandb_run(
    options: WandbOptions,
    spec: BenchmarkTrialSpec,
    bundle: DatasetBundle,
) -> Iterator[tuple[Any, Any]]:
    """Open one deterministic W&B run and finish it on every exit path."""

    import wandb

    options.directory.mkdir(parents=True, exist_ok=True)
    config = benchmark_wandb_config(spec, bundle)
    init_kwargs: dict[str, Any] = {
        "project": options.project,
        "entity": options.entity,
        "id": spec.trial_id,
        "name": _run_name(spec),
        "group": f"{options.run_group}/{spec.stage}/{spec.cell}",
        "job_type": spec.stage,
        "tags": _tags(spec, config["criticality_status"]),
        "config": config,
        "dir": str(options.directory),
        "mode": options.mode,
    }
    if options.mode == "online":
        init_kwargs["resume"] = "allow"
    run = wandb.init(**init_kwargs)
    try:
        yield run, wandb
    except BaseException as exc:
        try:
            run.summary["trial/status"] = "failed"
            run.summary["trial/error_type"] = type(exc).__name__
            run.finish(exit_code=1)
        except Exception:
            pass
        raise


def log_benchmark_wandb_result(
    run: Any,
    wandb_module: Any,
    result: dict,
    output_path: str | Path,
) -> None:
    """Log curves, scalar summary, and the exact result artifact."""

    run.define_metric("epoch")
    for namespace in ("train/*", "validation/*", "optimizer/*"):
        run.define_metric(namespace, step_metric="epoch")
    for row in benchmark_wandb_history(result):
        run.log(row)
    run.summary.update(benchmark_wandb_summary(result))

    trial = result["trial"]
    artifact = wandb_module.Artifact(
        name=f"benchmark-trial-{trial['trial_id']}",
        type="benchmark-trial",
        metadata={
            "trial_id": trial["trial_id"],
            "config_hash": trial["config_hash"],
            "stage": trial["stage"],
            "cell": trial["cell"],
            "schema_version": result["schema_version"],
        },
    )
    output_path = Path(output_path)
    artifact.add_file(str(output_path), name=output_path.name)
    checkpoint = result.get("checkpoint", {})
    if checkpoint.get("saved"):
        checkpoint_path = Path(checkpoint["path"])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Saved checkpoint is missing before W&B logging: {checkpoint_path}"
            )
        artifact.add_file(str(checkpoint_path), name=checkpoint_path.name)
    run.log_artifact(artifact, aliases=[result["trial"]["stage"]])


def finish_benchmark_wandb_run(
    run: Any,
    options: WandbOptions,
    trial_id: str,
) -> Path:
    """Finish a successful run and atomically record its local tracking state."""

    run_dir = getattr(run, "dir", None)
    run_url = getattr(run, "url", None)
    run.finish(exit_code=0)
    marker = tracking_marker_path(options, trial_id)
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "complete",
        "trial_id": trial_id,
        "wandb_run_id": getattr(run, "id", trial_id),
        "project": options.project,
        "entity": options.entity,
        "mode": options.mode,
        "run_url": run_url,
        "offline_sync_path": (
            str(Path(run_dir).parent) if options.mode == "offline" and run_dir else None
        ),
    }
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, marker)
    return marker
