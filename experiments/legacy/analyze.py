#!/usr/bin/env python3
"""Analyze the complete exact-paper MLP comparison without pooling protocols."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import dropout_mft.experiments.legacy.protocol as legacy_module
from dropout_mft.experiments.legacy.analysis import (
    REFERENCE_PROFILE_MAP,
    paired_comparison_rows,
    profile_summary_rows,
    reference_reproduction_rows,
    validate_complete_cohort,
)
from dropout_mft.experiments.legacy.protocol import (
    LEGACY_PHASE,
    LEGACY_PROFILE_IDS,
    LEGACY_TEST_INDEX_HASH,
    LEGACY_TRAIN_INDEX_HASH,
    legacy_trial_specs,
    read_legacy_manifest,
)
from dropout_mft.plotting import save_figure
from dropout_mft.provenance import (
    load_frozen_provenance,
    provenance_sha256,
    sha256_file,
)
from dropout_mft.results import load_npz_result
from dropout_mft.style import apply_paper_style

SAVED_ORIGINAL_SHA256 = (
    "0c347bca4756e65f7409388daf43519a02d58807f3b44dae0af62ba6f48496a7"
)
PROFILE_LABELS = {
    "uniform": "Uniform",
    "linear_early": "Linear early",
    "step_early": "Step early",
    "quadratic_early": "Quadratic early",
    "quartic_early": "Quartic early",
}
PROFILE_COLORS = {
    "uniform": "#C9A961",
    "linear_early": "#D4AF6A",
    "step_early": "#3A5F3A",
    "quadratic_early": "#4477AA",
    "quartic_early": "#AA3377",
}


def assert_checkout_import() -> None:
    expected = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "dropout_mft"
        / "experiments"
        / "legacy"
        / "protocol.py"
    )
    actual = Path(legacy_module.__file__).resolve()
    if actual != expected:
        raise SystemExit(
            "dropout_mft was imported from a different checkout: "
            f"actual={actual}, expected={expected}. Prepend PROJECT_DIR/src to PYTHONPATH."
        )


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_exact_trials(run_dir: Path) -> tuple[list[dict], dict]:
    provenance = load_frozen_provenance(
        run_dir,
        validate_source=True,
        # Cluster and laptop package locks may differ; source/result hashes are
        # still verified for offline analysis.
        validate_environment=False,
    )
    manifest = run_dir / "manifests" / f"{LEGACY_PHASE}.jsonl"
    specs = read_legacy_manifest(manifest, provenance=provenance)
    if specs != sorted(legacy_trial_specs(), key=lambda item: item.trial_id):
        raise RuntimeError("Legacy analysis requires the exact planned cohort")
    provenance_hash = provenance_sha256(provenance)
    preflight = json.loads((run_dir / "preflight.json").read_text())
    if not (
        preflight.get("status") == "passed"
        and preflight.get("trial_count") == 125
        and preflight.get("source_provenance_sha256") == provenance_hash
        and preflight.get("comparison_scope", {}).get(
            "pool_absolute_endpoints_with_transfer_protocol"
        )
        is False
    ):
        raise RuntimeError("Legacy preflight is absent, changed, or did not pass")
    data_record = json.loads((run_dir / "manifests" / "legacy_data.json").read_text())
    if not (
        data_record.get("train_index_hash") == LEGACY_TRAIN_INDEX_HASH
        and data_record.get("test_index_hash") == LEGACY_TEST_INDEX_HASH
        and data_record.get("train_size") == 5000
        and data_record.get("validation_size") == 0
        and data_record.get("test_size") == 5000
        and isinstance(data_record.get("split_hash"), str)
    ):
        raise RuntimeError("Legacy runtime data record does not match the exact recipe")
    locked_path = run_dir / "manifests" / f"{LEGACY_PHASE}.locked.jsonl"
    locked_rows = [json.loads(line) for line in locked_path.read_text().splitlines()]
    source_manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if not (
        len(locked_rows) == 125
        and {row.get("trial_id") for row in locked_rows}
        == {spec.trial_id for spec in specs}
        and all(
            row.get("source_manifest_sha256") == source_manifest_hash
            and row.get("source_provenance_sha256") == provenance_hash
            and row.get("split_hash") == data_record["split_hash"]
            for row in locked_rows
        )
    ):
        raise RuntimeError("Legacy locked manifest does not bind the plan and data")
    expected_paths = {
        run_dir / "trials" / LEGACY_PHASE / f"{spec.trial_id}.npz" for spec in specs
    }
    actual_paths = set((run_dir / "trials" / LEGACY_PHASE).glob("*.npz"))
    missing = expected_paths - actual_paths
    unexpected = actual_paths - expected_paths
    if missing or unexpected:
        raise RuntimeError(
            f"Legacy trial files differ from the manifest: missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )
    trials = [load_npz_result(path) for path in sorted(expected_paths)]
    if any(
        trial.get("provenance", {}).get("source_provenance_sha256") != provenance_hash
        or trial.get("data", {}).get("split_hash") != data_record["split_hash"]
        for trial in trials
    ):
        raise RuntimeError("A legacy trial is not bound to the run provenance")
    return trials, provenance


def _curves(indexed, profile_id: str, metric: str) -> np.ndarray:
    return np.stack(
        [indexed[(profile_id, seed)]["curves"][metric] for seed in range(42, 67)]
    )


def make_learning_figure(indexed, output_dir: Path) -> None:
    apply_paper_style()
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.25))
    epochs = np.arange(1, 76)
    for profile_id in LEGACY_PROFILE_IDS:
        for axis, metric in zip(axes, ("test_loss", "test_accuracy"), strict=True):
            values = _curves(indexed, profile_id, metric)
            mean = values.mean(axis=0)
            sem = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
            color = PROFILE_COLORS[profile_id]
            axis.plot(epochs, mean, color=color, label=PROFILE_LABELS[profile_id])
            axis.fill_between(epochs, mean - sem, mean + sem, color=color, alpha=0.16)
    axes[0].set_title("(a) Test cross-entropy", loc="left")
    axes[0].set_ylabel("Test loss")
    axes[1].set_title("(b) Test accuracy", loc="left")
    axes[1].set_ylabel("Accuracy (%)")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_xlim(1, 75)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.04),
    )
    figure.tight_layout(rect=(0, 0.13, 1, 1))
    save_figure(figure, output_dir / "legacy_test_learning_dynamics.png")
    plt.close(figure)


def make_reproduction_figure(indexed, saved: dict, output_dir: Path) -> None:
    apply_paper_style()
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    for rerun_id, saved_id in REFERENCE_PROFILE_MAP.items():
        rerun_loss = _curves(indexed, rerun_id, "test_loss")[:, -1]
        rerun_accuracy = _curves(indexed, rerun_id, "test_accuracy")[:, -1]
        saved_loss = np.asarray(saved["results"][saved_id]["test_loss"])[:, -1]
        saved_accuracy = np.asarray(saved["results"][saved_id]["test_acc"])[:, -1]
        color = PROFILE_COLORS[rerun_id]
        label = PROFILE_LABELS[rerun_id]
        axes[0].scatter(saved_loss, rerun_loss, color=color, alpha=0.72, label=label)
        axes[1].scatter(
            saved_accuracy, rerun_accuracy, color=color, alpha=0.72, label=label
        )
    for axis in axes:
        low, high = axis.get_xlim()
        lower, upper = axis.get_ylim()
        minimum, maximum = min(low, lower), max(high, upper)
        axis.plot([minimum, maximum], [minimum, maximum], "--", color="#777777", lw=1)
        axis.set_xlim(minimum, maximum)
        axis.set_ylim(minimum, maximum)
    axes[0].set_title("(a) Final test loss", loc="left")
    axes[0].set_xlabel("Saved original")
    axes[0].set_ylabel("Exact rerun")
    axes[1].set_title("(b) Final test accuracy", loc="left")
    axes[1].set_xlabel("Saved original (%)")
    axes[1].set_ylabel("Exact rerun (%)")
    axes[0].legend(loc="best")
    figure.tight_layout()
    save_figure(figure, output_dir / "legacy_reference_reproduction.png")
    plt.close(figure)


def command_analyze(args) -> None:
    assert_checkout_import()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "analysis"
    saved_path = (
        Path(args.saved_original)
        if args.saved_original
        else run_dir / "reference" / "dropout_experiment_results.npz"
    )
    if sha256_file(saved_path) != SAVED_ORIGINAL_SHA256:
        raise SystemExit(f"Saved original has the wrong SHA-256: {saved_path}")
    trials, provenance = _load_exact_trials(run_dir)
    indexed = validate_complete_cohort(trials)
    saved = load_npz_result(saved_path)
    summary_rows = profile_summary_rows(indexed)
    paired_rows, best_legacy = paired_comparison_rows(indexed)
    reproduction_rows = reference_reproduction_rows(indexed, saved)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "legacy_profile_summary.csv", summary_rows)
    _write_csv(output_dir / "legacy_paired_comparisons.csv", paired_rows)
    _write_csv(output_dir / "legacy_reference_reproduction.csv", reproduction_rows)
    make_learning_figure(indexed, output_dir)
    make_reproduction_figure(indexed, saved, output_dir)

    profile_by_id = {row["profile_id"]: row for row in summary_rows}
    observed_best_legacy_reference = min(
        ("linear_early", "step_early"),
        key=lambda profile: profile_by_id[profile]["mean_final_test_loss"],
    )
    payload = {
        "schema_version": 1,
        "artifact_type": "legacy_apples_to_apples_analysis",
        "status": "complete",
        "source_provenance_sha256": provenance_sha256(provenance),
        "saved_original_path": str(saved_path),
        "saved_original_sha256": sha256_file(saved_path),
        "trial_count": len(trials),
        "primary_endpoint": "final_epoch_test_cross_entropy",
        "best_profile_by_mean_final_test_loss": summary_rows[0]["profile_id"],
        "prespecified_best_legacy_early_baseline_from_saved_artifact": best_legacy,
        "observed_best_rerun_legacy_early_profile_descriptive": (
            observed_best_legacy_reference
        ),
        "new_profile_endpoints": {
            profile: profile_by_id[profile]
            for profile in ("quadratic_early", "quartic_early")
        },
        "paired_comparisons": paired_rows,
        "reference_reproduction": reproduction_rows,
        "interpretation_guardrail": (
            "This 5000-train/no-validation legacy cohort is not pooled with the "
            "separate 710-run 4000/1000 transfer protocol. Only within-protocol "
            "paired contrasts are inferentially interpreted."
        ),
        "derived_artifact_sha256": {
            name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
            for name in (
                "legacy_profile_summary.csv",
                "legacy_paired_comparisons.csv",
                "legacy_reference_reproduction.csv",
            )
        },
    }
    (output_dir / "legacy_analysis.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--saved-original")
    return parser


if __name__ == "__main__":
    command_analyze(build_parser().parse_args())
