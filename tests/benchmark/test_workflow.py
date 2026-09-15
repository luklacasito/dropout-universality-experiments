"""Shared benchmark workflow behavior must stay deterministic and immutable."""

from __future__ import annotations

import numpy as np
import pytest

from dropout_mft.experiments.benchmark import workflow
from dropout_mft.experiments.benchmark.workflow import (
    load_selection,
    paired_percentile_interval,
    selection_path,
    write_immutable_manifest,
    write_json_atomic,
)


def test_shared_selection_paths_and_atomic_json_round_trip(tmp_path):
    expected = {"fi2010/mlp": {"uniform": {"learning_rate": 3e-4}}}
    path = selection_path(tmp_path, "lr_search")
    assert write_json_atomic(path, expected) == path
    assert load_selection(tmp_path, "lr_search") == expected


def test_failed_json_replace_preserves_selection_and_removes_temporary_file(
    tmp_path, monkeypatch
):
    path = selection_path(tmp_path, "lr_search")
    original = {"selected": "original"}
    write_json_atomic(path, original)

    def fail_replace(*_args):
        raise OSError("simulated filesystem failure")

    monkeypatch.setattr(workflow.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated filesystem failure"):
        write_json_atomic(path, {"selected": "replacement"})

    assert load_selection(tmp_path, "lr_search") == original
    assert list(path.parent.iterdir()) == [path]


def test_shared_manifest_writer_refuses_replacement(tmp_path):
    path, provenance = write_immutable_manifest(tmp_path, "confirm", [])
    assert path.read_text() == ""
    assert provenance["runtime"]["device"] == "planning"
    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        write_immutable_manifest(tmp_path, "confirm", [])


def test_shared_paired_interval_matches_the_previous_calculation():
    values = np.asarray([-0.2, -0.1, 0.05, -0.3, -0.15])
    actual = paired_percentile_interval(values, resamples=2_000, seed=17)
    rng = np.random.default_rng(17)
    expected = np.percentile(
        rng.choice(values, size=(2_000, len(values)), replace=True).mean(axis=1),
        [2.5, 97.5],
    )
    assert actual == pytest.approx(tuple(expected))
