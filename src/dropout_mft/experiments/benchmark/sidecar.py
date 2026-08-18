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

from collections.abc import Iterable

from dropout_mft.experiments.benchmark.protocol import (
    CONFIRM_SEEDS,
    LINEAR_PROFILE_IDS,
    LR_GRIDS,
    LR_SEARCH_MEAN_DROPOUT,
    LR_SEARCH_SEEDS,
    SIDECAR_PROFILE_IDS,
    TUNED_CONTROL_PROFILE_ID,
    BenchmarkTrialSpec,
    _spec_defaults,
)
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
