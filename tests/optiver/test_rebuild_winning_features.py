from __future__ import annotations

import runpy
from pathlib import Path


def test_build_paths_are_resolved_before_chdir(tmp_path, monkeypatch):
    project = Path(__file__).resolve().parents[2]
    namespace = runpy.run_path(
        project / "experiments/optiver/rebuild_winning_features.py"
    )
    canonical_build_paths = namespace["_canonical_build_paths"]
    monkeypatch.chdir(tmp_path)

    notebook, raw, output = canonical_build_paths(
        Path("notebook.ipynb"),
        Path("raw"),
        Path("derived"),
    )

    assert notebook == tmp_path / "notebook.ipynb"
    assert raw == tmp_path / "raw"
    assert output == tmp_path / "derived"
    assert all(path.is_absolute() for path in (notebook, raw, output))


def test_mahalanobis_covariance_is_rewritten_for_modern_sklearn():
    project = Path(__file__).resolve().parents[2]
    namespace = runpy.run_path(
        project / "experiments/optiver/rebuild_winning_features.py"
    )
    cell_source = namespace["_cell_source"]
    notebook = {
        "cells": [{"source": []} for _ in range(10)]
        + [
            {
                "source": [
                    "Neighbors(metric='mahalanobis', "
                    "metric_params={'V':np.cov(pivot.values.T)})\n"
                ]
            }
        ]
    }

    transformed = cell_source(notebook, 10)

    assert "metric_params={'V':" not in transformed
    assert "metric_params={'VI':np.linalg.inv(np.cov(pivot.values.T))}" in transformed
