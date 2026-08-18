"""Checks for non-vacuous, multi-seed muP coordinate diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from dropout_mft.diagnostics import mup_coordinate_check


def test_coordinate_check_records_multiple_seeds_hidden_and_output_coordinates():
    pytest.importorskip("mup")
    result = mup_coordinate_check(
        widths=(8, 16, 32),
        depth=2,
        input_dim=8,
        output_dim=3,
        batch_size=8,
        steps=2,
        learning_rate=3e-3,
        seeds=(3, 7),
    )

    assert result["coordinate_l1"].shape == (2, 3, 3, 3)
    assert result["log_width_slopes"].shape == (2, 3, 3)
    np.testing.assert_array_equal(result["seeds"], [3, 7])
    assert result["learning_rate"] == pytest.approx(3e-3)
    assert np.all(result["min_output_l1_after_update_by_seed"] > 0)
    assert np.all(result["max_relative_hidden_change_by_seed"] > 0)
    assert isinstance(result["passes_nontrivial_update"], bool)
    assert isinstance(result["passes_threshold_0p1"], bool)


@pytest.mark.parametrize("seeds", [(), (1, 1)])
def test_coordinate_check_rejects_empty_or_duplicate_seed_sets(seeds):
    pytest.importorskip("mup")
    with pytest.raises(ValueError, match="unique integers"):
        mup_coordinate_check(widths=(8, 16), depth=1, steps=1, seeds=seeds)
