"""Shared filesystem and statistics helpers for benchmark experiment drivers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from dropout_mft.experiments.benchmark.protocol import (
    BenchmarkTrialSpec,
    write_benchmark_manifest,
)
from dropout_mft.experiments.scale_transfer.protocol import _provenance


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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return path


def paired_percentile_interval(
    values: Sequence[float],
    *,
    resamples: int = 10_000,
    seed: int = 20260812,
) -> tuple[float, float]:
    """Return the deterministic 95% bootstrap interval used by all cohorts."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("values must be a nonempty finite one-dimensional sequence")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
