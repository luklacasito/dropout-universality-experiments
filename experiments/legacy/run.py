#!/usr/bin/env python3
"""Plan, audit, run, and monitor the isolated legacy MLP comparison."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import dropout_mft.experiments.legacy.protocol as legacy_module
from dropout_mft.experiments.legacy.protocol import (
    ALL_LEGACY_PROFILE_IDS,
    LEGACY_DATA_PROTOCOL,
    LEGACY_PHASE,
    LEGACY_PROFILE_IDS,
    LEGACY_SCHEMA_VERSION,
    LEGACY_SEEDS,
    LEGACY_TEST_INDEX_HASH,
    LEGACY_TRAIN_INDEX_HASH,
    LegacyTrialSpec,
    canonical_json,
    legacy_profile_layers,
    legacy_trial_specs,
    load_legacy_cifar10,
    read_legacy_manifest,
    run_legacy_trial,
    write_legacy_manifest,
)
from dropout_mft.provenance import (
    freeze_provenance,
    freeze_source_snapshot,
    load_frozen_provenance,
    provenance_sha256,
    sha256_file,
)
from dropout_mft.results import load_npz_result

SAVED_ORIGINAL_SHA256 = (
    "0c347bca4756e65f7409388daf43519a02d58807f3b44dae0af62ba6f48496a7"
)
SAVED_ORIGINAL_NAME = "dropout_experiment_results.npz"


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


def manifest_path(run_dir: Path) -> Path:
    return run_dir / "manifests" / f"{LEGACY_PHASE}.jsonl"


def locked_manifest_path(run_dir: Path) -> Path:
    return run_dir / "manifests" / f"{LEGACY_PHASE}.locked.jsonl"


def _atomic_immutable_json(path: Path, payload: object) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != content:
            raise SystemExit(f"Refusing to change frozen audit artifact: {path}")
        return
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content)
    os.replace(temporary, path)


def preflight_payload(specs: list[LegacyTrialSpec], provenance: dict) -> dict:
    """Return a machine-readable audit of invariants and discrete budgets."""

    rows = []
    profile_ids = tuple(sorted({spec.profile_id for spec in specs}))
    for profile_id in profile_ids:
        spec = next(spec for spec in specs if spec.profile_id == profile_id)
        probabilities = np.asarray(legacy_profile_layers(spec), dtype=float)
        rows.append(
            {
                "profile_id": profile_id,
                "dropout_probabilities": probabilities.tolist(),
                "discrete_mean": float(probabilities.mean()),
                "observed_max": float(probabilities.max()),
                "budget_matches_0p10": bool(
                    np.isclose(probabilities.mean(), 0.10, rtol=0.0, atol=1e-12)
                ),
                "cap_at_most_0p20": bool(probabilities.max() <= 0.20 + 1e-12),
            }
        )
    factors = [asdict(spec) for spec in specs]
    invariant_keys = tuple(
        key for key in factors[0] if key not in {"profile_id", "seed"}
    )
    invariants = {
        key: factors[0][key]
        for key in invariant_keys
        if all(row[key] == factors[0][key] for row in factors)
    }
    return {
        "schema_version": LEGACY_SCHEMA_VERSION,
        "artifact_type": "legacy_apples_to_apples_preflight",
        "status": "passed",
        "phase": LEGACY_PHASE,
        "trial_count": len(specs),
        "profile_count": len(profile_ids),
        "profiles": list(profile_ids),
        "paired_seed_count": len(LEGACY_SEEDS),
        "paired_seeds": list(LEGACY_SEEDS),
        "all_profiles_share_seed_cohort": all(
            {spec.seed for spec in specs if spec.profile_id == profile_id}
            == set(LEGACY_SEEDS)
            for profile_id in profile_ids
        ),
        "invariant_training_and_data_factors": invariants,
        "data_contract": {
            "protocol": LEGACY_DATA_PROTOCOL,
            "train_index_hash": LEGACY_TRAIN_INDEX_HASH,
            "test_index_hash": LEGACY_TEST_INDEX_HASH,
            "validation_size": 0,
            "test_evaluated_every_epoch": True,
            "comparison_endpoint": "final_epoch_test_cross_entropy",
        },
        "profile_budget_audit": rows,
        "reference_profile_mapping": {
            "uniform": "saved_original:constant",
            "linear_early": "saved_original:reverse_linear",
            "step_early": "saved_original:reverse_step",
        },
        "saved_original_artifact": {
            "path": f"reference/{SAVED_ORIGINAL_NAME}",
            "sha256": SAVED_ORIGINAL_SHA256,
        },
        "comparison_scope": {
            "purpose": (
                "run a source-isolated, paired exact-recipe cohort; the base "
                "cohort compares quadratic/quartic with rerun uniform and "
                "prespecified step early, while the big_step extension is a "
                "historical benchmark with a 0.30 peak rather than the 0.20 cap"
            ),
            "existing_transfer_protocol": "separate_710_run_internally_paired_study",
            "pool_absolute_endpoints_with_transfer_protocol": False,
            "reason_not_pooled": (
                "the transfer protocol uses a 4000/1000 stratified train/validation "
                "split, batch size 75, named RNG streams, and final-only test access"
            ),
        },
        "source_provenance_sha256": provenance_sha256(provenance),
    }


def command_snapshot(args) -> None:
    record = freeze_source_snapshot(
        args.output,
        slurm_scripts=args.slurm_script or (),
    )
    reference_source = Path(args.saved_original)
    if sha256_file(reference_source) != SAVED_ORIGINAL_SHA256:
        raise SystemExit(
            f"Saved original artifact has the wrong SHA-256: {reference_source}"
        )
    reference_dir = Path(args.output) / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    reference_destination = reference_dir / SAVED_ORIGINAL_NAME
    if reference_destination.exists():
        if sha256_file(reference_destination) != SAVED_ORIGINAL_SHA256:
            raise SystemExit(
                f"Snapshot reference artifact differs: {reference_destination}"
            )
    else:
        shutil.copyfile(reference_source, reference_destination)
    _atomic_immutable_json(
        Path(args.output) / "legacy-reference.json",
        {
            "artifact_type": "saved_original_mlp_reference",
            "path": f"reference/{SAVED_ORIGINAL_NAME}",
            "sha256": SAVED_ORIGINAL_SHA256,
        },
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "source_tree_sha256": record["source_tree_sha256"],
                "source_snapshot_sha256": provenance_sha256(record),
                "trial_count": len(
                    legacy_trial_specs(args.profiles or LEGACY_PROFILE_IDS)
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


def command_plan(args) -> None:
    run_dir = Path(args.run_dir)
    try:
        provenance = freeze_provenance(
            run_dir,
            slurm_scripts=args.slurm_script or (),
            source_snapshot=args.source_snapshot,
        )
        reference_source = (
            Path(args.saved_original)
            if args.saved_original
            else Path(args.source_snapshot) / "reference" / SAVED_ORIGINAL_NAME
            if args.source_snapshot
            else Path("results") / "mlp" / SAVED_ORIGINAL_NAME
        )
        if sha256_file(reference_source) != SAVED_ORIGINAL_SHA256:
            raise ValueError(
                f"Saved original artifact has the wrong SHA-256: {reference_source}"
            )
        reference_destination = run_dir / "reference" / SAVED_ORIGINAL_NAME
        reference_destination.parent.mkdir(parents=True, exist_ok=True)
        if reference_destination.exists():
            if sha256_file(reference_destination) != SAVED_ORIGINAL_SHA256:
                raise ValueError(
                    f"Frozen saved-original artifact differs: {reference_destination}"
                )
        else:
            shutil.copyfile(reference_source, reference_destination)
        specs = legacy_trial_specs(tuple(args.profiles or LEGACY_PROFILE_IDS))
        path = write_legacy_manifest(
            manifest_path(run_dir), specs, provenance=provenance
        )
        audit = preflight_payload(specs, provenance)
        _atomic_immutable_json(run_dir / "preflight.json", audit)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Could not freeze the legacy plan: {exc}") from exc
    print(f"{LEGACY_PHASE}: {len(specs)} trials -> {path}")
    print(f"preflight: passed -> {run_dir / 'preflight.json'}")


def _load_provenance(run_dir: Path) -> dict:
    try:
        return load_frozen_provenance(run_dir, validate_source=True)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Invalid frozen legacy provenance: {exc}") from exc


def command_audit(args) -> None:
    run_dir = Path(args.run_dir)
    provenance = _load_provenance(run_dir)
    try:
        specs = read_legacy_manifest(manifest_path(run_dir), provenance=provenance)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    manifest_profiles = tuple(sorted({spec.profile_id for spec in specs}))
    expected = sorted(
        legacy_trial_specs(manifest_profiles), key=lambda item: item.trial_id
    )
    if specs != expected:
        raise SystemExit("Legacy manifest is not an exact prespecified legacy cohort")
    expected_audit = preflight_payload(specs, provenance)
    audit_path = run_dir / "preflight.json"
    try:
        actual_audit = json.loads(audit_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"Preflight artifact is absent or invalid: {audit_path}"
        ) from exc
    if actual_audit != expected_audit:
        raise SystemExit(
            f"Preflight artifact does not match the frozen plan: {audit_path}"
        )
    reference = run_dir / "reference" / SAVED_ORIGINAL_NAME
    if sha256_file(reference) != SAVED_ORIGINAL_SHA256:
        raise SystemExit(f"Saved-original reference is absent or changed: {reference}")
    print(json.dumps(actual_audit, indent=2, sort_keys=True))


def _record_and_lock_data(run_dir: Path, manifest: Path, specs, bundle, provenance):
    record = {
        "schema_version": LEGACY_SCHEMA_VERSION,
        "artifact_type": "legacy_apples_to_apples_data",
        "dataset": "cifar10",
        "protocol": bundle.protocol,
        "split_hash": bundle.split_hash,
        "train_index_hash": bundle.train_index_hash,
        "test_index_hash": bundle.test_index_hash,
        "train_size": len(bundle.train),
        "validation_size": 0,
        "test_size": len(bundle.test),
    }
    record_path = run_dir / "manifests" / "legacy_data.json"
    lock_path = run_dir / "manifests" / "legacy_runtime.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _atomic_immutable_json(record_path, record)
        source_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
        provenance_hash = provenance_sha256(provenance)
        rows = []
        source_rows = {
            row["trial_id"]: row
            for row in (json.loads(line) for line in manifest.read_text().splitlines())
        }
        for spec in specs:
            rows.append(
                {
                    **source_rows[spec.trial_id],
                    "source_manifest_sha256": source_hash,
                    "split_hash": bundle.split_hash,
                    "source_provenance_sha256": provenance_hash,
                }
            )
        content = "".join(
            canonical_json(row) + "\n"
            for row in sorted(rows, key=lambda row: row["trial_id"])
        )
        destination = locked_manifest_path(run_dir)
        if destination.exists() and destination.read_text() != content:
            raise SystemExit(f"Locked legacy manifest differs: {destination}")
        if not destination.exists():
            temporary = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")
            temporary.write_text(content)
            os.replace(temporary, destination)
        fcntl.flock(lock, fcntl.LOCK_UN)
    return record


def _filter_specs(specs: list[LegacyTrialSpec], args) -> list[LegacyTrialSpec]:
    profiles = set(args.profiles or ())
    seeds = set(args.seeds or ())
    selected = [
        spec
        for spec in specs
        if (not profiles or spec.profile_id in profiles)
        and (not seeds or spec.seed in seeds)
    ]
    if not selected:
        raise SystemExit("Runtime profile/seed filters matched no manifest rows")
    return selected


def command_run(args) -> None:
    run_dir = Path(args.run_dir)
    provenance = _load_provenance(run_dir)
    manifest = manifest_path(run_dir)
    try:
        all_specs = read_legacy_manifest(manifest, provenance=provenance)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    manifest_profiles = tuple(sorted({spec.profile_id for spec in all_specs}))
    expected_specs = sorted(
        legacy_trial_specs(manifest_profiles), key=lambda item: item.trial_id
    )
    if all_specs != expected_specs:
        raise SystemExit("Runtime manifest is not an exact prespecified legacy cohort")
    if args.require_h100:
        if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
            raise SystemExit("--require-h100 requested but the runtime is not an H100")
    bundle = load_legacy_cifar10(args.data_root, download=args.download)
    _record_and_lock_data(run_dir, manifest, all_specs, bundle, provenance)
    selected = _filter_specs(all_specs, args)
    specs = [
        spec
        for index, spec in enumerate(selected)
        if index % args.num_shards == args.shard_index
    ]
    output_dir = run_dir / "trials" / LEGACY_PHASE
    for index, spec in enumerate(specs, start=1):
        output = output_dir / f"{spec.trial_id}.npz"
        print(
            f"[{index}/{len(specs)}] {spec.trial_id} {spec.profile_id} seed={spec.seed}"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        trial_lock = output.with_suffix(".lock")
        with trial_lock.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            run_legacy_trial(
                spec,
                bundle,
                output,
                provenance=provenance,
                device=args.device,
                force=args.force,
            )
            fcntl.flock(lock, fcntl.LOCK_UN)


def _valid_result(path: Path, spec: LegacyTrialSpec, provenance_hash: str) -> bool:
    try:
        result = load_npz_result(path)
    except (OSError, KeyError, ValueError):
        return False
    return (
        result.get("schema_version") == LEGACY_SCHEMA_VERSION
        and result.get("artifact_type") == "legacy_apples_to_apples_trial"
        and result.get("trial", {}).get("status") == "complete"
        and result.get("trial", {}).get("trial_id") == spec.trial_id
        and result.get("trial", {}).get("config_hash") == spec.config_hash
        and result.get("provenance", {}).get("source_provenance_sha256")
        == provenance_hash
        and result.get("data", {}).get("train_index_hash") == LEGACY_TRAIN_INDEX_HASH
        and result.get("data", {}).get("test_index_hash") == LEGACY_TEST_INDEX_HASH
    )


def command_status(args) -> None:
    run_dir = Path(args.run_dir)
    provenance = _load_provenance(run_dir)
    specs = read_legacy_manifest(manifest_path(run_dir), provenance=provenance)
    manifest_profiles = tuple(sorted({spec.profile_id for spec in specs}))
    provenance_hash = provenance_sha256(provenance)
    output_dir = run_dir / "trials" / LEGACY_PHASE
    complete = [
        spec
        for spec in specs
        if _valid_result(output_dir / f"{spec.trial_id}.npz", spec, provenance_hash)
    ]
    by_profile = {
        profile_id: sum(spec.profile_id == profile_id for spec in complete)
        for profile_id in manifest_profiles
    }
    payload = {
        "planned": len(specs),
        "complete": len(complete),
        "remaining": len(specs) - len(complete),
        "complete_by_profile": by_profile,
        "cohort_complete": len(complete) == len(specs),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser("snapshot-source")
    snapshot.add_argument("--output", required=True)
    snapshot.add_argument("--slurm-script", action="append", default=[])
    snapshot.add_argument(
        "--profile", dest="profiles", action="append", choices=ALL_LEGACY_PROFILE_IDS
    )
    snapshot.add_argument(
        "--saved-original",
        default="results/mlp/dropout_experiment_results.npz",
    )
    snapshot.set_defaults(func=command_snapshot)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--run-dir", required=True)
    plan.add_argument("--source-snapshot")
    plan.add_argument("--slurm-script", action="append", default=[])
    plan.add_argument("--saved-original")
    plan.add_argument(
        "--profile", dest="profiles", action="append", choices=ALL_LEGACY_PROFILE_IDS
    )
    plan.set_defaults(func=command_plan)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--run-dir", required=True)
    audit.set_defaults(func=command_audit)

    run = subparsers.add_parser("run")
    run.add_argument("--run-dir", required=True)
    run.add_argument("--data-root", required=True)
    run.add_argument("--device", default="auto")
    run.add_argument("--download", action="store_true")
    run.add_argument("--require-h100", action="store_true")
    run.add_argument("--num-shards", type=int, default=1)
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument(
        "--profile", dest="profiles", action="append", choices=ALL_LEGACY_PROFILE_IDS
    )
    run.add_argument("--seed", dest="seeds", action="append", type=int)
    run.add_argument("--force", action="store_true")
    run.set_defaults(func=command_run)

    status = subparsers.add_parser("status")
    status.add_argument("--run-dir", required=True)
    status.set_defaults(func=command_status)
    return parser


def main() -> None:
    assert_checkout_import()
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run" and (
        args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards
    ):
        parser.error("run requires 0 <= shard-index < num-shards")
    args.func(args)


if __name__ == "__main__":
    main()
