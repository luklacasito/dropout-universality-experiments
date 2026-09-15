"""Check the evidence rules behind the manuscript's best-observed comparisons."""

import copy
import gzip
import json
from pathlib import Path

import pytest

from experiments.paper.make_appendix import original_rows, summarize_group

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def rows():
    with gzip.open(ROOT / "results/benchmarks/confirmation.json.gz", "rt") as stream:
        return json.load(stream)["runs"]


def cell(rows, suffix):
    group = next(row["group"] for row in rows if suffix in row["group"])
    return [row for row in rows if row["group"] == group]


def test_missing_arms_are_visible_but_cannot_win(rows):
    summary = summarize_group(cell(rows, "amazon_n20000/confirm/amazon_reviews/mlp"))
    assert summary["arms"]["linear_early"]["n"] == 4
    assert summary["arms"]["linear_late"]["n"] == 4
    assert not summary["complete"]
    assert summary["winner"] not in ("linear_early", "linear_late")


def test_losses_to_uniform_remain_in_the_results(rows):
    summary = summarize_group(
        cell(rows, "amazon_n2000/confirm/amazon_reviews/transformer")
    )
    assert summary["paired"]["reduction_percent"] < 0


def test_missing_uniform_is_not_replaced_by_an_unrelated_cohort(rows):
    summary = summarize_group(cell(rows, "linear-sidecar-20260811-v1/confirm/fi2010"))
    assert "uniform" not in summary["arms"]
    assert summary["paired"] is None


def test_duplicate_seeds_and_changed_splits_are_rejected(rows):
    selected = cell(rows, "zero-decay-depth12-sweep-20260811-v1/confirm/fi2010/mlp")
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_group(selected + [selected[0]])
    altered = copy.deepcopy(selected)
    altered[0]["config"]["data_split_hash"] = "different"
    with pytest.raises(ValueError, match="Do not pool"):
        summarize_group(altered)


def test_original_schedule_names_follow_the_paper_convention():
    row = original_rows()[0]
    assert row["winner"] == "Step early"
    assert row["reduction_percent"] == pytest.approx(17.864486937117)
