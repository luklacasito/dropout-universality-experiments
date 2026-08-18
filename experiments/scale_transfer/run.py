#!/usr/bin/env python3
"""Plan, run, select, and aggregate the scale-transfer experiments."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from dropout_mft.experiments.scale_transfer.analysis import select_proxy_learning_rates
from dropout_mft.diagnostics import mup_coordinate_check
from dropout_mft.models import MLPConfig, build_mlp, make_optimizer
from dropout_mft.results import load_npz_result, save_npz_result
from dropout_mft.experiments.scale_transfer.protocol import (
    SCHEMA_VERSION,
    TrialSpec,
    canonical_json,
    mup_transfer_specs,
    mup_tune_extension_specs,
    mup_tune_specs,
    oracle_specs,
    profile_confirmation_specs,
    profile_pilot_specs,
    read_manifest,
    read_manifest_provenance,
    run_trial,
    seed_streams,
    vit_confirmation_specs,
    write_manifest,
)
from dropout_mft.provenance import (
    freeze_provenance,
    freeze_source_snapshot,
    load_frozen_provenance,
    provenance_sha256,
)
from dropout_mft.training import (
    TrainingConfig,
    load_cifar_bundle,
    seed_everything,
    synthetic_bundle,
    train_model,
)


PRIMARY_PHASE_BUILDERS = {
    "profile_pilot": profile_pilot_specs,
    "profile_confirmation": profile_confirmation_specs,
    "lr_proxy": mup_tune_specs,
    "target_oracle": oracle_specs,
    "vit_confirmation": vit_confirmation_specs,
}
CONDITIONAL_PHASE_BUILDERS = {"lr_proxy_extension": mup_tune_extension_specs}
PHASE_BUILDERS = {**PRIMARY_PHASE_BUILDERS, **CONDITIONAL_PHASE_BUILDERS}

EXPECTED_COORD_WIDTHS = (64, 128, 256, 512, 1024, 2048)
EXPECTED_COORD_SEEDS = (0, 1, 2)
MIN_COORD_STEPS = 2
MIN_COORD_LEARNING_RATE = 3e-3


def _manifest_path(run_dir: Path, phase: str) -> Path:
    return run_dir / "manifests" / f"{phase}.jsonl"


def _locked_manifest_path(run_dir: Path, phase: str) -> Path:
    return run_dir / "manifests" / f"{phase}.locked.jsonl"


def _split_key(spec: TrialSpec) -> str:
    return (
        f"{spec.dataset}:{spec.train_size}:{spec.validation_size}:"
        f"{spec.test_size}:{spec.split_seed}"
    )


def _coord_check_path(run_dir: Path, args) -> Path:
    explicit = getattr(args, "coord_check_artifact", None)
    return Path(explicit) if explicit else run_dir / "coord_check.npz"


def _require_coordinate_check(
    path: str | Path, *, source_provenance: dict | None = None
) -> dict:
    path = Path(path)
    if not path.exists():
        raise SystemExit(
            f"Training requires a passing strict coordinate-check artifact: {path}. "
            "Run `coord-check --strict` first."
        )
    try:
        payload = load_npz_result(path)
    except (OSError, KeyError, ValueError) as exc:
        raise SystemExit(f"Invalid coordinate-check artifact {path}: {exc}") from exc
    widths = tuple(int(value) for value in np.asarray(payload.get("widths", [])))
    seeds = tuple(int(value) for value in np.asarray(payload.get("seeds", [])))
    try:
        max_hidden_slope = float(payload["max_abs_hidden_slope_after_update"])
        max_output_slope = float(payload["max_abs_output_slope_after_update"])
        learning_rate = float(payload["learning_rate"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid coordinate-check artifact {path}: {exc}") from exc
    expected_source_hash = (
        provenance_sha256(source_provenance)
        if source_provenance is not None
        else payload.get("source_provenance_sha256")
    )
    valid = (
        payload.get("schema_version") == SCHEMA_VERSION
        and payload.get("artifact_type") == "mup_coordinate_check"
        and payload.get("strict_requested") is True
        and payload.get("status") == "passed"
        and payload.get("passes_threshold_0p1") is True
        and payload.get("passes_nontrivial_update") is True
        and payload.get("passes_protocol") is True
        and np.isfinite(max_hidden_slope)
        and np.isfinite(max_output_slope)
        and max_hidden_slope <= 0.1
        and max_output_slope <= 0.1
        and widths == EXPECTED_COORD_WIDTHS
        and seeds == EXPECTED_COORD_SEEDS
        and int(payload.get("steps", 0)) >= MIN_COORD_STEPS
        and learning_rate >= MIN_COORD_LEARNING_RATE
        and payload.get("source_provenance_sha256") == expected_source_hash
    )
    if not valid:
        raise SystemExit(
            f"Coordinate-check artifact does not satisfy the strict protocol: {path}. "
            f"Required widths={list(EXPECTED_COORD_WIDTHS)}, steps>={MIN_COORD_STEPS}, "
            f"seeds={list(EXPECTED_COORD_SEEDS)}, learning_rate>="
            f"{MIN_COORD_LEARNING_RATE:g}, nontrivial updates, and max hidden/output "
            "slopes <= 0.1."
        )
    return payload


def command_plan(args) -> None:
    run_dir = Path(args.run_dir)
    try:
        provenance = freeze_provenance(
            run_dir,
            slurm_scripts=getattr(args, "slurm_script", ()) or (),
            source_snapshot=getattr(args, "source_snapshot", None),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Could not freeze experiment provenance: {exc}") from exc
    phases = list(PRIMARY_PHASE_BUILDERS) if args.phase == "all" else [args.phase]
    for phase in phases:
        specs = PHASE_BUILDERS[phase]()
        path = _manifest_path(run_dir, phase)
        write_manifest(path, specs, provenance=provenance)
        print(f"{phase}: {len(specs)} trials -> {path}")


def command_snapshot_source(args) -> None:
    try:
        record = freeze_source_snapshot(
            args.output,
            slurm_scripts=args.slurm_script,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Could not freeze portable source snapshot: {exc}") from exc
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "source_tree_sha256": record["source_tree_sha256"],
                "source_snapshot_sha256": provenance_sha256(record),
                "git_dirty": record["git"]["dirty"],
                "slurm_scripts": record["slurm_scripts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def _load_run_provenance(run_dir: Path) -> dict:
    try:
        return load_frozen_provenance(run_dir, validate_source=True)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Invalid frozen experiment provenance: {exc}") from exc


def _require_manifest_provenance(manifest: Path, provenance: dict) -> None:
    try:
        manifest_provenance = read_manifest_provenance(manifest)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Invalid manifest provenance in {manifest}: {exc}") from exc
    if manifest_provenance != provenance:
        raise SystemExit(
            f"Manifest {manifest} is not bound to this run's frozen provenance"
        )


def _bundle_for_spec(spec: TrialSpec, args):
    return load_cifar_bundle(
        spec.dataset,
        root=args.data_root,
        train_size=spec.train_size,
        validation_size=spec.validation_size,
        test_size=spec.test_size,
        split_seed=spec.split_seed,
        download=args.download,
    )


def _record_split_manifest(run_dir: Path, spec: TrialSpec, bundle) -> dict:
    path = run_dir / "manifests" / "data_splits.json"
    key = _split_key(spec)
    record = {
        "dataset": spec.dataset,
        "train_size": spec.train_size,
        "validation_size": spec.validation_size,
        "test_size": spec.test_size,
        "split_seed": spec.split_seed,
        "split_hash": bundle.split_hash,
        "split_protocol": bundle.split_protocol,
        "test_subset_hash": bundle.test_subset_hash,
        "test_subset_protocol": bundle.test_subset_protocol,
        "test_subset_seed": bundle.test_subset_seed,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        payload = {"schema_version": SCHEMA_VERSION, "splits": {}}
        if path.exists():
            payload = json.loads(path.read_text())
            if payload.get("schema_version") != SCHEMA_VERSION:
                raise SystemExit(f"Unsupported split manifest: {path}")
        existing = payload["splits"].get(key)
        if existing is not None and existing != record:
            raise SystemExit(f"Data split hash changed for {key}")
        payload["splits"][key] = record
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
        fcntl.flock(lock, fcntl.LOCK_UN)
    return record


def _read_split_manifest(run_dir: Path) -> dict[str, dict]:
    path = run_dir / "manifests" / "data_splits.json"
    if not path.exists():
        raise SystemExit(f"Data-split manifest is absent: {path}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid data-split manifest {path}: {exc}") from exc
    if payload.get("schema_version") != SCHEMA_VERSION or not isinstance(
        payload.get("splits"), dict
    ):
        raise SystemExit(f"Unsupported data-split manifest: {path}")
    return payload["splits"]


def _write_locked_manifest(
    run_dir: Path,
    phase: str,
    source_manifest: Path,
    specs: list[TrialSpec],
    split_hashes: dict[str, str],
) -> Path:
    """Bind an immutable pre-data plan to the exact realized data split."""

    provenance = _load_run_provenance(run_dir)
    _require_manifest_provenance(source_manifest, provenance)
    source_bytes = source_manifest.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    source_rows = {
        row["trial_id"]: row
        for row in (
            json.loads(line) for line in source_bytes.decode().splitlines() if line
        )
    }
    if set(source_rows) != {spec.trial_id for spec in specs}:
        raise SystemExit("Source manifest rows do not match the validated trial plan")

    rows = []
    for spec in sorted(specs, key=lambda item: item.trial_id):
        split_key = _split_key(spec)
        split_hash = split_hashes.get(split_key)
        if not split_hash:
            raise SystemExit(f"No recorded split hash for {split_key}")
        rows.append(
            {
                **source_rows[spec.trial_id],
                "data_split_key": split_key,
                "split_hash": split_hash,
                "source_manifest_sha256": source_hash,
            }
        )
    content = "".join(canonical_json(row) + "\n" for row in rows)
    path = _locked_manifest_path(run_dir, phase)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != content:
            raise SystemExit(
                f"Locked manifest differs from the current plan or data split: {path}"
            )
        return path
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content)
    os.replace(temporary, path)
    return path


def _read_locked_manifest(
    run_dir: Path, phase: str, source_manifest: Path, specs: list[TrialSpec]
) -> dict[str, dict]:
    path = _locked_manifest_path(run_dir, phase)
    if not path.exists():
        raise SystemExit(f"Locked runtime manifest is absent: {path}")
    source_hash = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    provenance = _load_run_provenance(run_dir)
    _require_manifest_provenance(source_manifest, provenance)
    expected_provenance_hash = provenance_sha256(provenance)
    rows: dict[str, dict] = {}
    try:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            trial_id = str(row["trial_id"])
            if trial_id in rows:
                raise SystemExit(f"Duplicate trial in locked manifest: {trial_id}")
            rows[trial_id] = row
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid locked manifest {path}: {exc}") from exc

    expected = {spec.trial_id: spec for spec in specs}
    if set(rows) != set(expected):
        raise SystemExit(
            f"Locked manifest does not match the complete {phase} trial plan"
        )
    split_records = _read_split_manifest(run_dir)
    for trial_id, spec in expected.items():
        row = rows[trial_id]
        split_key = _split_key(spec)
        split_record = split_records.get(split_key)
        if (
            row.get("schema_version") != SCHEMA_VERSION
            or row.get("config_hash") != spec.config_hash
            or row.get("source_manifest_sha256") != source_hash
            or row.get("provenance_sha256") != expected_provenance_hash
            or row.get("data_split_key") != split_key
            or not isinstance(row.get("split_hash"), str)
            or not row["split_hash"]
            or split_record is None
            or split_record.get("split_hash") != row["split_hash"]
        ):
            raise SystemExit(
                f"Locked manifest/data-split binding is invalid for trial {trial_id}"
            )
    return rows


def _validate_completed_phase(
    run_dir: Path,
    phase: str,
    expected_specs: list[TrialSpec],
    *,
    require_test: bool,
) -> list[dict]:
    manifest_path = _manifest_path(run_dir, phase)
    if not manifest_path.exists():
        raise SystemExit(f"Prespecified {phase} manifest is absent: {manifest_path}")
    planned_specs = read_manifest(manifest_path)
    if planned_specs != sorted(expected_specs, key=lambda spec: spec.trial_id):
        raise SystemExit(f"{phase} manifest is not the prespecified complete cohort")
    locked = _read_locked_manifest(run_dir, phase, manifest_path, planned_specs)
    trials = _load_phase_trials(run_dir, phase)
    by_id: dict[str, dict] = {}
    for trial in trials:
        trial_id = trial.get("trial", {}).get("trial_id")
        if trial_id in by_id:
            raise SystemExit(f"Duplicate completed trial for {phase}: {trial_id}")
        by_id[trial_id] = trial
    expected_by_id = {spec.trial_id: spec for spec in planned_specs}
    missing = set(expected_by_id) - set(by_id)
    unexpected = set(by_id) - set(expected_by_id)
    if missing or unexpected or len(trials) != len(planned_specs):
        raise SystemExit(
            f"{phase} requires the complete locked cohort: missing={len(missing)}, "
            f"unexpected={len(unexpected)}, files={len(trials)}, "
            f"expected={len(planned_specs)}"
        )
    for trial_id, spec in expected_by_id.items():
        trial = by_id[trial_id]
        try:
            result_spec = TrialSpec(**trial["factors"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"Invalid {phase} trial factors: {trial_id}") from exc
        if (
            trial.get("schema_version") != SCHEMA_VERSION
            or trial.get("trial", {}).get("status") != "complete"
            or trial.get("trial", {}).get("config_hash") != spec.config_hash
            or result_spec != spec
            or trial.get("randomization") != seed_streams(spec)
            or trial.get("provenance", {}).get("source_provenance_sha256")
            != locked[trial_id]["provenance_sha256"]
            or trial.get("data", {}).get("split_hash") != locked[trial_id]["split_hash"]
            or (require_test and trial.get("test", {}).get("evaluated") is not True)
            or (
                require_test
                and trial.get("test", {}).get("selected_epoch") != spec.epochs - 1
            )
        ):
            raise SystemExit(
                f"{phase} trial is incomplete or not bound to its locked split: {trial_id}"
            )
    return [by_id[spec.trial_id] for spec in planned_specs]


def _require_complete_width_transfer(run_dir: Path) -> None:
    selection_path = run_dir / "selection.json"
    if not selection_path.exists():
        raise SystemExit(
            "target_oracle is blocked until proxy selection and zero-shot "
            "width_transfer are complete"
        )
    try:
        selection = json.loads(selection_path.read_text())
        selected_lrs = selection["selected_learning_rates"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid proxy selection {selection_path}: {exc}") from exc
    if selection.get("schema_version") != SCHEMA_VERSION:
        raise SystemExit(f"Unsupported proxy selection: {selection_path}")
    try:
        expected = mup_transfer_specs(
            {name: float(selected_lrs[name]) for name in ("sp", "mup")}
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid selected learning rates: {selection_path}") from exc
    _validate_completed_phase(run_dir, "width_transfer", expected, require_test=True)


def _require_proxy_extension_trigger(run_dir: Path, provenance: dict) -> None:
    path = run_dir / "selection_stability.json"
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(
            "lr_proxy_extension is conditional on a failed base leave-one-seed-out "
            "gate. Complete lr_proxy and run select first."
        ) from exc
    valid = (
        payload.get("schema_version") == SCHEMA_VERSION
        and payload.get("status") == "conditional_extension_required"
        and payload.get("source_provenance_sha256") == provenance_sha256(provenance)
        and payload.get("base", {}).get("stable") is False
    )
    if not valid:
        raise SystemExit(
            "lr_proxy_extension has not been triggered by this run's failed base "
            "leave-one-seed-out gate"
        )


def command_run(args) -> None:
    run_dir = Path(args.run_dir)
    provenance = _load_run_provenance(run_dir)
    _require_coordinate_check(
        _coord_check_path(run_dir, args), source_provenance=provenance
    )
    manifest = (
        Path(args.manifest) if args.manifest else _manifest_path(run_dir, args.phase)
    )
    all_specs = read_manifest(manifest)
    _require_manifest_provenance(manifest, provenance)
    if not all_specs:
        raise SystemExit(f"Manifest contains no trials: {manifest}")
    phases = {spec.phase for spec in all_specs}
    if len(phases) != 1:
        raise SystemExit("A runtime manifest must contain exactly one phase")
    phase = next(iter(phases))
    if phase == "target_oracle":
        _require_complete_width_transfer(run_dir)
    elif phase == "lr_proxy_extension":
        _require_proxy_extension_trigger(run_dir, provenance)

    bundles = {}
    split_hashes = {}
    for spec in all_specs:
        key = (
            spec.dataset,
            spec.train_size,
            spec.validation_size,
            spec.test_size,
            spec.split_seed,
        )
        if key not in bundles:
            bundles[key] = _bundle_for_spec(spec, args)
            record = _record_split_manifest(run_dir, spec, bundles[key])
            split_hashes[_split_key(spec)] = record["split_hash"]
    _write_locked_manifest(run_dir, phase, manifest, all_specs, split_hashes)
    locked = _read_locked_manifest(run_dir, phase, manifest, all_specs)

    filtered_specs = _filter_runtime_specs(all_specs, args)
    specs = [
        spec
        for index, spec in enumerate(filtered_specs)
        if index % args.num_shards == args.shard_index
    ]
    for index, spec in enumerate(specs, start=1):
        key = (
            spec.dataset,
            spec.train_size,
            spec.validation_size,
            spec.test_size,
            spec.split_seed,
        )
        if bundles[key].split_hash != locked[spec.trial_id]["split_hash"]:
            raise SystemExit(
                f"Loaded data no longer matches the locked split for {spec.trial_id}"
            )
        output = run_dir / "trials" / spec.phase / f"{spec.trial_id}.npz"
        print(
            f"[{index}/{len(specs)}] {spec.trial_id} {spec.parameterization} "
            f"N={spec.width} {spec.profile_id} lr={spec.learning_rate:g}"
        )
        run_trial(
            spec,
            bundles[key],
            output,
            device=args.device,
            force=args.force,
            source_provenance=provenance,
        )


def _filter_runtime_specs(specs: list[TrialSpec], args) -> list[TrialSpec]:
    """Apply explicit execution filters without changing the locked cohort."""

    filters = {
        "width": set(getattr(args, "widths", None) or ()),
        "parameterization": set(getattr(args, "parameterizations", None) or ()),
        "profile_id": set(getattr(args, "profiles", None) or ()),
        "seed": set(getattr(args, "seeds", None) or ()),
    }
    selected = [
        spec
        for spec in specs
        if all(
            not values or getattr(spec, field) in values
            for field, values in filters.items()
        )
    ]
    if not selected:
        active = {key: sorted(values) for key, values in filters.items() if values}
        raise SystemExit(f"Runtime filters matched no trials in the manifest: {active}")
    return selected


def _load_phase_trials(run_dir: Path, phase: str) -> list[dict]:
    return [
        load_npz_result(path)
        for path in sorted((run_dir / "trials" / phase).glob("*.npz"))
    ]


def _proxy_leave_one_seed_out_stability(
    trials: list[dict], selected: dict[str, float]
) -> dict:
    """Require every leave-one-seed-out proxy choice to equal the full choice."""

    diagnostics: dict[str, dict] = {}
    stable = True
    for parameterization in ("sp", "mup"):
        cohort = [
            trial
            for trial in trials
            if trial["factors"]["parameterization"] == parameterization
        ]
        seeds = sorted({int(trial["factors"]["seed"]) for trial in cohort})
        choices = {}
        for held_out_seed in seeds:
            remaining = [
                trial
                for trial in cohort
                if int(trial["factors"]["seed"]) != held_out_seed
            ]
            choices[str(held_out_seed)] = select_proxy_learning_rates(remaining)[
                parameterization
            ]
        parameterization_stable = all(
            choice == selected[parameterization] for choice in choices.values()
        )
        stable = stable and parameterization_stable
        diagnostics[parameterization] = {
            "full_cohort_choice": selected[parameterization],
            "leave_one_seed_out_choices": choices,
            "stable": parameterization_stable,
        }
    return {
        "rule": "all_leave_one_seed_out_choices_equal_full_cohort_choice",
        "stable": stable,
        "by_parameterization": diagnostics,
    }


def command_select(args) -> None:
    run_dir = Path(args.run_dir)
    provenance = _load_run_provenance(run_dir)
    expected_specs = mup_tune_specs()
    trials = _validate_completed_phase(
        run_dir, "lr_proxy", expected_specs, require_test=False
    )
    split_hashes = {trial.get("data", {}).get("split_hash") for trial in trials}
    if len(split_hashes) != 1 or None in split_hashes:
        raise SystemExit("lr_proxy trials do not share one valid data split hash")
    for trial in trials:
        spec = TrialSpec(**trial["factors"])
        if (
            trial["trial"].get("trial_id") != spec.trial_id
            or trial["trial"].get("config_hash") != spec.config_hash
        ):
            raise SystemExit("lr_proxy trial metadata does not match its factors")
    base_selected = select_proxy_learning_rates(trials)
    base_stability = _proxy_leave_one_seed_out_stability(trials, base_selected)
    stability_path = run_dir / "selection_stability.json"
    selection_trials = trials
    extension_used = False
    combined_stability = None
    selected = base_selected
    if not base_stability["stable"]:
        extension_used = True
        extension_specs = mup_tune_extension_specs()
        extension_manifest = _manifest_path(run_dir, "lr_proxy_extension")
        if not extension_manifest.exists():
            write_manifest(
                extension_manifest,
                extension_specs,
                provenance=provenance,
            )
        completed_extension_files = list(
            (run_dir / "trials" / "lr_proxy_extension").glob("*.npz")
        )
        extension_locked = _locked_manifest_path(run_dir, "lr_proxy_extension").exists()
        if not extension_locked or len(completed_extension_files) != len(
            extension_specs
        ):
            stability_payload = {
                "schema_version": SCHEMA_VERSION,
                "status": "conditional_extension_required",
                "source_provenance_sha256": provenance_sha256(provenance),
                "base": base_stability,
                "extension_manifest": str(extension_manifest),
                "extension_trials_required": len(extension_specs),
                "extension_trials_completed": len(completed_extension_files),
            }
            temporary = stability_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(stability_payload, indent=2, sort_keys=True) + "\n"
            )
            temporary.replace(stability_path)
            raise SystemExit(
                "Base proxy LR selection is unstable under leave-one-seed-out. "
                "The prespecified validation-only lr_proxy_extension (fresh seeds "
                "3, 4, 5 across the full LR grid) must complete before selection. "
                f"Run its manifest, then rerun select. Diagnostics: {stability_path}"
            )
        extension_trials = _validate_completed_phase(
            run_dir,
            "lr_proxy_extension",
            extension_specs,
            require_test=False,
        )
        selection_trials = [*trials, *extension_trials]
        selected = select_proxy_learning_rates(selection_trials)
        combined_stability = _proxy_leave_one_seed_out_stability(
            selection_trials, selected
        )

    final_stability = combined_stability or base_stability
    stability_payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if final_stability["stable"] else "failed",
        "source_provenance_sha256": provenance_sha256(provenance),
        "base": base_stability,
        "extension_used": extension_used,
        "combined": combined_stability,
        "final": final_stability,
    }
    temporary = stability_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(stability_payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(stability_path)
    if not final_stability["stable"]:
        raise SystemExit(
            "Proxy LR selection remains unstable after the prespecified fresh-seed "
            "extension. Do not launch width_transfer; revise and preregister the "
            f"selection design. Diagnostics: {stability_path}"
        )
    candidate_summary = []
    for parameterization in ("sp", "mup"):
        for learning_rate in sorted(
            {
                float(trial["factors"]["learning_rate"])
                for trial in selection_trials
                if trial["factors"]["parameterization"] == parameterization
            }
        ):
            losses = [
                float(np.asarray(trial["curves"]["validation_loss"])[-1])
                for trial in selection_trials
                if trial["factors"]["parameterization"] == parameterization
                and float(trial["factors"]["learning_rate"]) == learning_rate
            ]
            candidate_summary.append(
                {
                    "parameterization": parameterization,
                    "learning_rate": learning_rate,
                    "count": len(losses),
                    "mean_final_validation_cross_entropy": float(np.mean(losses)),
                }
            )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "selection_metric": "mean_final_validation_cross_entropy",
        "selected_learning_rates": selected,
        "candidate_summary": candidate_summary,
        "split_hash": next(iter(split_hashes)),
        "completed_manifest_trials": len(selection_trials),
        "base_manifest_trials": len(trials),
        "conditional_extension_used": extension_used,
        "leave_one_seed_out_stability": final_stability,
        "source_provenance_sha256": provenance_sha256(provenance),
        "test_metrics_used": False,
    }
    path = run_dir / "selection.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    specs = mup_transfer_specs(selected)
    write_manifest(
        _manifest_path(run_dir, "width_transfer"), specs, provenance=provenance
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"width_transfer: {len(specs)} trials planned")


def command_aggregate(args) -> None:
    run_dir = Path(args.run_dir)
    paths = sorted((run_dir / "trials").glob("*/*.npz"))
    trials = [load_npz_result(path) for path in paths]
    if any(trial.get("schema_version") != SCHEMA_VERSION for trial in trials):
        raise SystemExit("Mixed or unsupported trial schemas")
    seen_ids = set()
    locked_by_phase: dict[str, dict[str, dict]] = {}
    for trial in trials:
        if trial.get("trial", {}).get("status") != "complete":
            raise SystemExit("Aggregation accepts completed trials only")
        try:
            spec = TrialSpec(**trial["factors"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"Invalid trial factors: {exc}") from exc
        trial_id = trial["trial"].get("trial_id")
        if trial_id in seen_ids:
            raise SystemExit(f"Duplicate trial_id: {trial_id}")
        seen_ids.add(trial_id)
        if (
            trial_id != spec.trial_id
            or trial["trial"].get("config_hash") != spec.config_hash
            or trial.get("randomization") != seed_streams(spec)
        ):
            raise SystemExit(f"Trial hash mismatch: {trial_id}")
        phase = spec.phase
        if phase not in locked_by_phase:
            manifest_path = _manifest_path(run_dir, phase)
            planned_specs = read_manifest(manifest_path)
            locked_by_phase[phase] = _read_locked_manifest(
                run_dir, phase, manifest_path, planned_specs
            )
        locked = locked_by_phase[phase].get(trial_id)
        if (
            locked is None
            or trial.get("data", {}).get("split_hash") != locked.get("split_hash")
            or trial.get("provenance", {}).get("source_provenance_sha256")
            != locked.get("provenance_sha256")
        ):
            raise SystemExit(f"Trial is not bound to its locked data split: {trial_id}")
    trials.sort(
        key=lambda trial: (
            trial["trial"]["phase"],
            trial["factors"]["parameterization"],
            int(trial["factors"]["width"]),
            trial["factors"]["profile_id"],
            float(trial["factors"]["learning_rate"]),
            int(trial["trial"]["seed"]),
        )
    )
    save_npz_result(
        run_dir / "aggregate.npz", {"schema_version": SCHEMA_VERSION, "trials": trials}
    )

    columns = (
        "trial_id",
        "phase",
        "seed",
        "parameterization",
        "width",
        "depth",
        "profile_id",
        "learning_rate",
        "validation_loss",
        "test_loss",
        "test_accuracy",
        "test_evaluated",
        "optimizer_steps",
        "estimated_training_flops",
        "examples_seen",
    )
    with (run_dir / "trial_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for trial in trials:
            factors = trial["factors"]
            writer.writerow(
                {
                    "trial_id": trial["trial"]["trial_id"],
                    "phase": trial["trial"]["phase"],
                    "seed": trial["trial"]["seed"],
                    "parameterization": factors["parameterization"],
                    "width": factors["width"],
                    "depth": factors["depth"],
                    "profile_id": factors["profile_id"],
                    "learning_rate": factors["learning_rate"],
                    "validation_loss": trial["curves"]["validation_loss"][-1],
                    "test_loss": trial["test"]["loss"],
                    "test_accuracy": trial["test"]["accuracy"],
                    "test_evaluated": trial["test"]["evaluated"],
                    "optimizer_steps": trial["compute"]["optimizer_steps"],
                    "estimated_training_flops": trial["compute"][
                        "estimated_training_flops"
                    ],
                    "examples_seen": trial["compute"]["examples_seen"],
                }
            )
    print(f"Aggregated {len(trials)} trials in {run_dir}")


def command_coord_check(args) -> None:
    widths = tuple(args.widths)
    seeds = tuple(args.seeds)
    source_provenance = None
    if args.run_dir:
        source_provenance = _load_run_provenance(Path(args.run_dir))
    elif args.strict:
        raise SystemExit(
            "A strict coordinate check requires --run-dir so its artifact is "
            "bound to the frozen source provenance."
        )
    result = mup_coordinate_check(
        widths=widths,
        steps=args.steps,
        seeds=seeds,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    passes_protocol = bool(
        result["passes_threshold_0p1"]
        and result["passes_nontrivial_update"]
        and widths == EXPECTED_COORD_WIDTHS
        and seeds == EXPECTED_COORD_SEEDS
        and args.steps >= MIN_COORD_STEPS
        and args.learning_rate >= MIN_COORD_LEARNING_RATE
    )
    status = "passed" if args.strict and passes_protocol else "failed"
    output = Path(args.output)
    save_npz_result(
        output,
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "mup_coordinate_check",
            "strict_requested": bool(args.strict),
            "status": status,
            "steps": int(args.steps),
            "passes_protocol": passes_protocol,
            "source_provenance_sha256": (
                provenance_sha256(source_provenance)
                if source_provenance is not None
                else None
            ),
            **result,
        },
    )
    print(
        f"max |hidden slope| after updates: {result['max_abs_hidden_slope_after_update']:.4f}"
    )
    print(
        f"max |output slope| after updates: {result['max_abs_output_slope_after_update']:.4f}"
    )
    print(f"nontrivial updates: {result['passes_nontrivial_update']}")
    print(f"passes +/-0.1: {result['passes_threshold_0p1']}")
    if args.strict and not passes_protocol:
        raise SystemExit(1)


def command_smoke(args) -> None:
    bundle = synthetic_bundle(input_dim=16, classes=3, seed=7)
    summaries = {}
    for parameterization in ("sp", "mup"):
        seed_everything(7)
        config = MLPConfig(
            input_dim=16,
            width=8,
            output_dim=3,
            depth=2,
            sigma_b_sq=0.0,
        )
        model = build_mlp(
            config,
            [0.1, 0.0],
            parameterization=parameterization,
            base_width=4,
            delta_width=8,
        )
        optimizer = make_optimizer(model, learning_rate=1e-3, weight_decay=0.0)
        result = train_model(
            model,
            optimizer,
            bundle,
            TrainingConfig(
                epochs=1,
                batch_size=16,
                learning_rate=1e-3,
                seed=7,
                device="cpu",
                weight_decay=0.0,
            ),
        )
        summaries[parameterization] = {
            "test_loss": result["final_test_loss"],
            "test_accuracy": result["final_test_accuracy"],
        }
    print(json.dumps(summaries, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    snapshot = commands.add_parser(
        "snapshot-source",
        help="freeze a portable source/Slurm snapshot before cluster transfer",
    )
    snapshot.add_argument("--output", required=True)
    snapshot.add_argument(
        "--slurm-script",
        action="append",
        default=[],
        help="Slurm script to freeze and hash (repeat for multiple scripts)",
    )
    snapshot.set_defaults(function=command_snapshot_source)

    plan = commands.add_parser("plan", help="write deterministic manifests")
    plan.add_argument("--run-dir", required=True)
    plan.add_argument("--phase", choices=[*PHASE_BUILDERS, "all"], default="all")
    plan.add_argument(
        "--slurm-script",
        action="append",
        default=[],
        help="Slurm script to freeze and hash (repeat for multiple scripts)",
    )
    plan.add_argument(
        "--source-snapshot",
        help="portable snapshot created by snapshot-source; captures env on this host",
    )
    plan.set_defaults(function=command_plan)

    run = commands.add_parser("run", help="execute a manifest shard")
    run.add_argument("--run-dir", required=True)
    run.add_argument("--phase", default="profile_pilot")
    run.add_argument("--manifest")
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--num-shards", type=int, default=1)
    run.add_argument("--device", default="auto")
    run.add_argument("--data-root", default="data")
    run.add_argument("--download", action="store_true")
    run.add_argument("--force", action="store_true")
    run.add_argument("--width", dest="widths", action="append", type=int)
    run.add_argument(
        "--parameterization",
        dest="parameterizations",
        action="append",
        choices=("sp", "mup"),
    )
    run.add_argument("--profile", dest="profiles", action="append")
    run.add_argument("--seed", dest="seeds", action="append", type=int)
    run.add_argument(
        "--coord-check-artifact",
        help="passing strict artifact (default: RUN_DIR/coord_check.npz)",
    )
    run.set_defaults(function=command_run)

    select = commands.add_parser(
        "select", help="select proxy LRs using validation only"
    )
    select.add_argument("--run-dir", required=True)
    select.set_defaults(function=command_select)

    aggregate = commands.add_parser("aggregate", help="aggregate completed trials")
    aggregate.add_argument("--run-dir", required=True)
    aggregate.set_defaults(function=command_aggregate)

    coord = commands.add_parser("coord-check", help="run muP coordinate checks")
    coord.add_argument(
        "--widths", type=int, nargs="+", default=list(EXPECTED_COORD_WIDTHS)
    )
    coord.add_argument(
        "--seeds", type=int, nargs="+", default=list(EXPECTED_COORD_SEEDS)
    )
    coord.add_argument("--steps", type=int, default=2)
    coord.add_argument("--learning-rate", type=float, default=3e-3)
    coord.add_argument("--device", default="cpu")
    coord.add_argument(
        "--run-dir",
        help="run directory whose frozen source provenance binds a strict artifact",
    )
    coord.add_argument("--output", default="results/scale_transfer/coord_check.npz")
    coord.add_argument("--strict", action="store_true")
    coord.set_defaults(function=command_coord_check)

    smoke = commands.add_parser("smoke", help="one-epoch SP/muP CPU smoke test")
    smoke.set_defaults(function=command_smoke)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "num_shards", 1) <= 0:
        raise SystemExit("num_shards must be positive")
    if not 0 <= getattr(args, "shard_index", 0) < getattr(args, "num_shards", 1):
        raise SystemExit("shard_index must lie in [0, num_shards)")
    args.function(args)


if __name__ == "__main__":
    main()
